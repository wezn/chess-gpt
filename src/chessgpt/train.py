"""Train a GPT on a tokenized dataset.

    uv run python -m chessgpt.train configs/experiments/tiny-uci-move.yaml
    uv run python -m chessgpt.train --resume runs/tiny-uci-move/20261023-120000

Each run writes runs/<experiment>/<timestamp>/:
    config.yaml     fully resolved config (experiment + model size + git commit)
    metrics.jsonl   one JSON line per log/eval step
    last.pt         latest checkpoint incl. optimizer state, used by --resume (overwritten)
    best.pt         checkpoint with the lowest validation loss so far (overwritten)
    snapshots/step-NNNNNN.pt  permanent model-only snapshots, for evaluating progress

All three are written every `train.checkpoint_every` of training (e.g. 0.02 = every
2%), so snapshots line up at the same number of games seen whatever the tokenizer.
"""

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from chessgpt.batches import IGNORE, EpochPlan, TokenizedSplit, make_batch, pack_rows
from chessgpt.data.common import REPO_ROOT, git_commit
from chessgpt.data.pretokenize import TOKENIZED_DIR
from chessgpt.model import GPT, GPTConfig, attention_mask
from chessgpt.tokenizers import get_tokenizer

RUNS_DIR = REPO_ROOT / "runs"


def resolve_config(path: str) -> dict:
    """Load an experiment config and merge in its model size config (configs/model/<model>.yaml)."""
    cfg = yaml.safe_load(open(path))
    model = yaml.safe_load(open(REPO_ROOT / "configs" / "model" / f"{cfg['model']}.yaml"))
    cfg["train"].setdefault("lr", model.pop("lr"))
    cfg["model"] = {"name": cfg["model"], **{k: v for k, v in model.items() if k != "lr"}}
    return cfg


def lr_at(step: int, total: int, t: dict) -> float:
    """Linear warmup, then cosine decay to lr * min_lr_ratio."""
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    progress = (step - t["warmup_steps"]) / max(1, total - t["warmup_steps"])
    return t["lr"] * (t["min_lr_ratio"] + (1 - t["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress)))


def to_device(batch, device: str) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(getattr(batch, k)).to(device, non_blocking=True) for k in ("x", "y", "pos", "doc")}


def loss_sum(model: torch.nn.Module, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed cross-entropy over real (non-padding) targets, and the number of those targets."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(b["x"], b["pos"], attention_mask(b["doc"]))
    loss = F.cross_entropy(logits.flatten(0, 1).float(), b["y"].flatten(), ignore_index=IGNORE, reduction="sum")
    return loss, (b["y"] != IGNORE).sum()


class Validator:
    """Validation loss on a fixed set of games, per token and per move (comparable across tokenizers)."""

    def __init__(self, split: TokenizedSplit, tokenizer_name: str, block_size: int, num_games: int, rows: int):
        order, bounds = pack_rows(split.pair_lengths(block_size), np.arange(min(num_games, len(split))), block_size)
        games = [order[bounds[r] : bounds[r + 1]] for r in range(len(bounds) - 1)]
        self.batches = [make_batch(split, games[i : i + rows], block_size) for i in range(0, len(games), rows)]
        # Moves actually scored: decode each (possibly truncated) game's tokens.
        tokenizer = get_tokenizer(tokenizer_name)
        self.moves = 0
        for g in order:
            s = int(split.starts[g])
            e = min(int(split.ends[g]), s + block_size + 1)
            self.moves += len(tokenizer.decode(split.data[s:e].tolist()))

    @torch.no_grad()
    def __call__(self, model: torch.nn.Module, device: str) -> dict:
        model.eval()
        total, count = torch.zeros((), device=device), torch.zeros((), device=device)
        for batch in self.batches:
            loss, n = loss_sum(model, to_device(batch, device))
            total += loss
            count += n
        model.train()
        total = total.item()
        return {"val_loss_token": total / count.item(), "val_loss_move": total / self.moves}


def prefetch(fn, items, depth: int = 4):
    """Run fn over items in a background thread, yielding results in order."""
    q: queue.Queue = queue.Queue(depth)

    def worker():
        for item in items:
            q.put(fn(item))
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (result := q.get()) is not None:
        yield result


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never leaves a corrupt checkpoint


def train(cfg: dict, run_dir: Path, resume: bool = False, max_steps: int | None = None) -> None:
    t, device = cfg["train"], "cuda"
    torch.manual_seed(t["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    data_dir = TOKENIZED_DIR / cfg["data"]
    train_split, val_split = TokenizedSplit(data_dir, "train"), TokenizedSplit(data_dir, "val")
    block = cfg["block_size"]
    rows = t["batch_tokens"] // block

    plans = [EpochPlan(train_split, block, rows, t["seed"], e) for e in range(t["epochs"])]
    total_steps = sum(p.num_batches for p in plans)
    if max_steps:
        total_steps = min(total_steps, max_steps)
    every = t["checkpoint_every"]
    checkpoint_steps = {round(k * every * total_steps) for k in range(1, math.ceil(1 / every) + 1)} | {total_steps}
    checkpoint_steps.discard(0)
    (run_dir / "snapshots").mkdir(exist_ok=True)

    model = GPT(GPTConfig(vocab_size=train_split.meta["vocab_size"], block_size=block, **{k: v for k, v in cfg["model"].items() if k != "name"})).to(device)
    optimizer = model.configure_optimizer(t["lr"], t["weight_decay"], tuple(t["betas"]))
    step, best_val = 0, float("inf")
    if resume:
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt["best_val"]
    compiled = torch.compile(model) if t.get("compile", True) else model
    validator = Validator(val_split, train_split.meta["tokenizer"], block, t["eval_games"], rows)

    n_params = model.num_params()
    L, d = cfg["model"]["n_layer"], cfg["model"]["n_embd"]
    flops_per_token = 6 * n_params + 12 * L * d * block  # forward + backward, incl. attention (PaLM appendix B)
    print(
        f"{cfg['name']}: {n_params / 1e6:.1f}M params, {total_steps:,} steps of {rows} rows x {block} tokens, "
        f"{len(train_split):,} train games x {t['epochs']} epochs" + (f", resuming at step {step:,}" if resume else ""),
        file=sys.stderr,
    )

    def batch_for(s: int):
        for p in plans:
            if s < p.num_batches:
                return make_batch(train_split, p.batch_games(s), block)
            s -= p.num_batches
        raise IndexError(s)

    metrics = open(run_dir / "metrics.jsonl", "a")

    def log(record: dict) -> None:
        metrics.write(json.dumps(record) + "\n")
        metrics.flush()

    def evaluate_and_checkpoint(s: int) -> None:
        nonlocal best_val
        result = validator(compiled, device)
        log({"type": "eval", "step": s, **result})
        print(f"step {s:,}  val loss/move {result['val_loss_move']:.4f}  val loss/token {result['val_loss_token']:.4f}", file=sys.stderr)
        best_val = min(best_val, result["val_loss_move"])
        snapshot = {"model": model.state_dict(), "step": s, "progress": s / total_steps, "val_loss_move": result["val_loss_move"], "config": cfg}
        save_checkpoint(run_dir / "snapshots" / f"step-{s:06d}.pt", snapshot)
        save_checkpoint(run_dir / "last.pt", {**snapshot, "optimizer": optimizer.state_dict(), "best_val": best_val})
        if best_val == result["val_loss_move"]:
            save_checkpoint(run_dir / "best.pt", snapshot)

    train_start = window_start = time.monotonic()
    window_tokens = window_targets = 0
    for s, batch in zip(range(step, total_steps), prefetch(batch_for, range(step, total_steps))):
        lr = lr_at(s, total_steps, t)
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss, n = loss_sum(compiled, to_device(batch, device))
        (loss / n).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        window_tokens += batch.x.size
        window_targets += int((batch.y != IGNORE).sum())
        if (s + 1) % t["log_every"] == 0:
            dt = time.monotonic() - window_start
            record = {
                "type": "train", "step": s + 1, "loss": (loss / n).item(), "lr": lr,
                "tokens_per_s": window_tokens / dt,
                "padding": 1 - window_targets / window_tokens,
                "tflops": flops_per_token * window_tokens / dt / 1e12,
                "elapsed_s": round(time.monotonic() - train_start),
            }
            log(record)
            print(
                f"step {s + 1:,}/{total_steps:,}  loss {record['loss']:.4f}  lr {lr:.2e}  "
                f"{record['tokens_per_s'] / 1e3:,.0f}k tok/s  {record['tflops']:.1f} TFLOPS  pad {record['padding']:.0%}",
                file=sys.stderr,
            )
            window_start, window_tokens, window_targets = time.monotonic(), 0, 0
        if s + 1 in checkpoint_steps:
            evaluate_and_checkpoint(s + 1)
    metrics.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="configs/experiments/<name>.yaml")
    parser.add_argument("--resume", type=Path, help="run directory to resume from its last.pt")
    parser.add_argument("--max-steps", type=int, help="stop early (benchmarks); the LR schedule is shortened to match")
    args = parser.parse_args()

    if args.resume:
        run_dir = args.resume
        cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    else:
        cfg = resolve_config(args.config)
        cfg["git_commit"] = git_commit()
        run_dir = RUNS_DIR / cfg["name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True)
        yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
    print(f"run dir: {run_dir}", file=sys.stderr)
    train(cfg, run_dir, resume=bool(args.resume), max_steps=args.max_steps)


if __name__ == "__main__":
    main()
