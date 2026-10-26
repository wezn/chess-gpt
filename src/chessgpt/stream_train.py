"""Train several models round-robin on a dataset that is still being pulled.

    uv run python -m chessgpt.stream_train configs/experiments/stream-large.yaml
    uv run python -m chessgpt.stream_train --resume 20261025-093000     # the group's timestamp

One process holds every model (one per tokenizer). In each round every model trains up to
its next snapshot step, is validated and snapshotted, then the next model takes the GPU.
Snapshot steps follow the models' original runs (`snapshot_period` = steps per 2% there),
so snapshots line up step for step with the single-epoch experiments.

Data arrives as chunks (see `chessgpt.data.pretokenize --follow`). Chunks are grouped into
blocks of `block_chunks`; each block is shuffled with a seed shared by all tokenizers, so the
models see the same games in the same order. Every game is seen once. When the next block is
not tokenized yet the trainer waits.

Schedule: warmup, then a constant learning rate for as long as data flows. A `STOP` file in
runs/<name>/ (all models) or in a model's run directory starts that model's cooldown: the
learning rate decays linearly over `cooldown_snapshots` snapshot periods, a last snapshot is
written, and the model is done. Running out of data does the same. The process exits when
every model is done.

A member with `warm_start: <run dir>` continues that finished run: weights, optimizer state,
step counter and snapshot numbering are taken from its last.pt, its metrics.jsonl and
eval.jsonl are carried over so the dashboard shows one continuous history, and the stream
is entered at `start_block` (the first games it has not already trained on).

Each model gets an ordinary run directory, runs/<name>-<tokenizer>/<timestamp>/, with the
usual config.yaml, metrics.jsonl, last.pt, best.pt and snapshots/, so chess-monitor,
chessgpt.eval.watch and chess-play work unchanged. runs/<name>/status.json says which
model is training now.
"""

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from chessgpt.batches import IGNORE, TokenizedSplit, make_batch, pack_rows
from chessgpt.data.common import REPO_ROOT, git_commit
from chessgpt.data.pretokenize import TOKENIZED_DIR
from chessgpt.model import GPT, GPTConfig
from chessgpt.train import Validator, loss_sum, save_checkpoint, to_device

RUNS_DIR = REPO_ROOT / "runs"


class Block:
    """`block_chunks` consecutive tokenized chunks loaded into memory, with the TokenizedSplit interface make_batch needs."""

    def __init__(self, tok_dir: Path, chunks: list[int]):
        parts, starts, offset = [], [], 0
        for c in chunks:
            data = np.fromfile(tok_dir / f"chunk-{c:04d}.bin", dtype=np.uint16)
            idx = np.load(tok_dir / f"chunk-{c:04d}_idx.npy")
            parts.append(data)
            starts.append(idx + offset)
            offset += len(data)
        self.data = np.concatenate(parts)
        self.starts = np.concatenate(starts)
        self.ends = np.append(self.starts[1:], len(self.data))

    def __len__(self) -> int:
        return len(self.starts)

    def pair_lengths(self, block_size: int) -> np.ndarray:
        return np.minimum(self.ends - self.starts - 1, block_size)


class StreamData:
    """Batches from a growing sequence of tokenized chunks, block by block, each block shuffled by (seed, block).

    Position = (block, batch). `next_batch()` blocks while the next block is not tokenized yet
    and returns None once the pull is finished and everything has been consumed.
    """

    def __init__(self, dataset: str, tokenizer: str, block_size: int, rows: int, seed: int, block_chunks: int, position: tuple[int, int] = (0, 0)):
        self.source = REPO_ROOT / "data" / "processed" / dataset
        self.tok_dir = TOKENIZED_DIR / f"{dataset}__{tokenizer}"
        self.block_size, self.rows, self.seed, self.block_chunks = block_size, rows, seed, block_chunks
        self.block_index, self.batch_index = position
        self.waiting = False
        self.games_in_block = 0
        self._q: queue.Queue = queue.Queue(4)
        threading.Thread(target=self._producer, daemon=True).start()

    def _ready_chunks(self) -> list[int]:
        return sorted(int(p.stem.split("-")[1].split("_")[0]) for p in self.tok_dir.glob("chunk-*_idx.npy"))

    def _pull_done(self) -> bool:
        st = self.source / "status.json"
        return st.exists() and json.loads(st.read_text()).get("done", False)

    def _chunks_for_block(self, b: int) -> list[int] | None:
        """Chunk ids of block b, waiting for them to be tokenized; None when the stream has ended before block b."""
        want = list(range(b * self.block_chunks + 1, (b + 1) * self.block_chunks + 1))
        while True:
            ready = set(self._ready_chunks())
            if all(c in ready for c in want):
                self.waiting = False
                return want
            if self._pull_done() and self._all_tokenized():
                partial = [c for c in want if c in ready]
                self.waiting = False
                return partial or None
            self.waiting = True
            time.sleep(10)

    def _all_tokenized(self) -> bool:
        pulled = sorted(int(p.stem.split("-")[1]) for p in self.source.glob("chunk-*.parquet"))
        return set(pulled) <= set(self._ready_chunks())

    def _producer(self) -> None:
        b, start_batch = self.block_index, self.batch_index
        while True:
            chunks = self._chunks_for_block(b)
            if chunks is None:
                self._q.put(None)
                return
            block = Block(self.tok_dir, chunks)
            order = np.random.default_rng([self.seed, b]).permutation(len(block))
            packed, bounds = pack_rows(block.pair_lengths(self.block_size), order, self.block_size)
            num_batches = (len(bounds) - 1) // self.rows
            for i in range(start_batch, num_batches):
                r0 = i * self.rows
                games = [packed[bounds[r] : bounds[r + 1]] for r in range(r0, r0 + self.rows)]
                self._q.put((b, i, len(block), make_batch(block, games, self.block_size), sum(len(g) for g in games)))
            b, start_batch = b + 1, 0

    @property
    def starved(self) -> bool:
        """True when the trainer would block: nothing prefetched and the producer is waiting for chunks."""
        return self.waiting and self._q.empty()

    def next_batch(self):
        """(block, batch_index, games_in_block, Batch, games_in_batch) or None at the end of the stream."""
        item = self._q.get()
        if item is not None:
            self.block_index, self.batch_index = item[0], item[1] + 1
            self.games_in_block = item[2]
        return item


@dataclass
class Member:
    """One model of the group: its run directory, data stream and training state."""

    tokenizer: str
    cfg: dict
    run_dir: Path
    period: float  # steps per snapshot
    spec: dict = field(default_factory=dict)  # the member's entry in the experiment config
    model: torch.nn.Module = None
    compiled: torch.nn.Module = None
    optimizer: torch.optim.Optimizer = None
    validator: Validator = None
    data: StreamData = None
    step: int = 0
    snapshots: int = 0  # snapshots written so far
    games_seen: int = 0
    best_val: float = float("inf")
    cooldown_start: int | None = None
    cooldown_end: int | None = None
    done: bool = False
    metrics: object = None
    flops_per_token: float = 0.0
    tag: str = ""
    games_per_s: float = 0.0  # over the last log window, while this model holds the GPU

    def next_snapshot_step(self) -> int:
        return round((self.snapshots + 1) * self.period)

    start_step: int = 0  # step at which this process started training the model (warmup counts from here)

    def lr_at(self, step: int, t: dict) -> float:
        if step - self.start_step < t["warmup_steps"]:
            return t["lr"] * (step - self.start_step + 1) / t["warmup_steps"]
        if self.cooldown_start is not None:
            frac = min(1.0, (step - self.cooldown_start) / max(1, self.cooldown_end - self.cooldown_start))
            return t["lr"] * (1 - frac * (1 - t["min_lr_ratio"]))
        return t["lr"]

    def log(self, record: dict) -> None:
        self.metrics.write(json.dumps(record) + "\n")
        self.metrics.flush()


def member_config(cfg: dict, m: dict) -> dict:
    """A single-model config (as a normal run would have), so ModelPlayer, watch and monitor work unchanged."""
    return {
        "name": f"{cfg['name']}-{m['tokenizer']}",
        "data": f"{cfg['data']}__{m['tokenizer']}",
        "model": cfg["model"],
        "block_size": m["block_size"],
        "train": {**cfg["train"], "snapshot_period": m["snapshot_period"]},
        "streaming": True,
        "git_commit": cfg.get("git_commit"),
    }


def write_group_status(group_dir: Path, members: list[Member], now: Member | None, round_no: int) -> None:
    st = {
        "now": now.tokenizer if now else None,
        "round": round_no,
        "updated": time.time(),
        "runs": {m.tokenizer: {"run_dir": str(m.run_dir), "step": m.step, "games_seen": m.games_seen, "snapshots": m.snapshots, "done": m.done,
                               "cooldown": m.cooldown_start is not None, "waiting_for_data": m.data.starved if m.data else False, "games_per_s": m.games_per_s} for m in members},
    }
    tmp = group_dir / "status.json.tmp"
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(group_dir / "status.json")


def train_group(cfg: dict, group_dir: Path, members: list[Member], resume: bool) -> None:
    t, device = cfg["train"], "cuda"
    torch.manual_seed(t["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    rows_of = lambda m: t["batch_tokens"] // m.cfg["block_size"]

    for m in members:
        tok_dir = TOKENIZED_DIR / m.cfg["data"]
        while not (tok_dir / "val_idx.npy").exists():  # the follower writes it with the first chunk
            print(f"{m.tokenizer}: waiting for {tok_dir}/val_idx.npy", file=sys.stderr)
            time.sleep(15)
        meta = json.loads((tok_dir / "meta.json").read_text())
        block = m.cfg["block_size"]
        m.model = GPT(GPTConfig(vocab_size=meta["vocab_size"], block_size=block, **{k: v for k, v in m.cfg["model"].items() if k != "name"})).to(device)
        m.optimizer = m.model.configure_optimizer(t["lr"], t["weight_decay"], tuple(t["betas"]))
        position = (m.spec.get("start_block", 0), 0)
        if m.spec.get("warm_start") and not (m.run_dir / "last.pt").exists():  # first start, or a resume before any snapshot was saved
            src = REPO_ROOT / m.spec["warm_start"]
            ckpt = torch.load(src / "last.pt", map_location=device, weights_only=False)
            m.model.load_state_dict(ckpt["model"])
            m.optimizer.load_state_dict(ckpt["optimizer"])
            m.step, m.best_val = ckpt["step"], ckpt["best_val"]
            m.snapshots = round(m.step / m.period)
            m.games_seen = int(m.spec.get("games_already_seen", 0))
            for name in ("metrics.jsonl", "eval.jsonl"):
                if (src / name).exists() and not (m.run_dir / name).exists():
                    (m.run_dir / name).write_bytes((src / name).read_bytes())
            print(f"{m.tokenizer}: warm start from {src} at step {m.step:,} (snapshot {m.snapshots}), entering the stream at block {position[0]}", file=sys.stderr)
        if (m.run_dir / "last.pt").exists():
            ckpt = torch.load(m.run_dir / "last.pt", map_location=device, weights_only=False)
            m.model.load_state_dict(ckpt["model"])
            m.optimizer.load_state_dict(ckpt["optimizer"])
            m.step, m.best_val, m.snapshots, m.games_seen = ckpt["step"], ckpt["best_val"], ckpt["snapshots"], ckpt["games_seen"]
            m.cooldown_start, m.cooldown_end, m.done = ckpt.get("cooldown_start"), ckpt.get("cooldown_end"), ckpt.get("done", False)
            position = tuple(ckpt["stream_position"])
        m.start_step = m.step
        m.compiled = torch.compile(m.model) if t.get("compile", True) else m.model
        m.validator = Validator(TokenizedSplit(tok_dir, "val"), meta["tokenizer"], block, t["eval_games"], rows_of(m))
        m.data = StreamData(cfg["data"], m.tokenizer, block, rows_of(m), t["seed"], t["block_chunks"], position)
        n_params = m.model.num_params()
        m.flops_per_token = 6 * n_params + 12 * m.cfg["model"]["n_layer"] * m.cfg["model"]["n_embd"] * block
        m.metrics = open(m.run_dir / "metrics.jsonl", "a")
        m.log({"type": "start", "step": m.step, "total_steps": None, "streaming": True, "params": n_params, "snapshot_period": m.period})
        m.tag = f"{m.tokenizer:<10}"
        print(f"{m.tag} {n_params / 1e6:.1f}M params, snapshot every {m.period:.2f} steps, {rows_of(m)} rows x {block} tokens" + (f", resuming at step {m.step:,}" if m.step else ""), file=sys.stderr)

    def snapshot(m: Member) -> None:
        result = m.validator(m.compiled, device)
        m.log({"type": "eval", "step": m.step, "games_seen": m.games_seen, **result})
        m.best_val = min(m.best_val, result["val_loss_move"])
        snap = {"model": m.model.state_dict(), "step": m.step, "progress": None, "games_seen": m.games_seen, "val_loss_move": result["val_loss_move"], "config": m.cfg}
        save_checkpoint(m.run_dir / "snapshots" / f"step-{m.step:06d}.pt", snap)
        save_checkpoint(m.run_dir / "last.pt", {**snap, "optimizer": m.optimizer.state_dict(), "best_val": m.best_val, "snapshots": m.snapshots + 1,
                                                 "stream_position": [m.data.block_index, m.data.batch_index], "cooldown_start": m.cooldown_start, "cooldown_end": m.cooldown_end, "done": m.done})
        if m.best_val == result["val_loss_move"]:
            save_checkpoint(m.run_dir / "best.pt", snap)
        m.snapshots += 1
        print(f"{m.tag} step {m.step:,}  snapshot {m.snapshots}  val loss/move {result['val_loss_move']:.4f}  games seen {m.games_seen:,}", file=sys.stderr)

    def stop_requested(m: Member) -> bool:
        return (group_dir / "STOP").exists() or (m.run_dir / "STOP").exists()

    round_no = 0
    while not all(m.done for m in members):
        round_no += 1
        for m in members:
            if m.done:
                continue
            target = m.next_snapshot_step()
            write_group_status(group_dir, members, m, round_no)
            window_start, window_tokens, window_targets, window_games, exhausted = time.monotonic(), 0, 0, 0, False
            while m.step < target:
                item = m.data.next_batch()
                if item is None:
                    exhausted = True
                    break
                block_no, _, games_in_block, batch, games = item
                lr = m.lr_at(m.step, t)
                for g in m.optimizer.param_groups:
                    g["lr"] = lr
                loss, n = loss_sum(m.compiled, to_device(batch, device))
                (loss / n).backward()
                torch.nn.utils.clip_grad_norm_(m.model.parameters(), t["grad_clip"])
                m.optimizer.step()
                m.optimizer.zero_grad(set_to_none=True)
                m.step += 1
                m.games_seen += games
                window_tokens += batch.x.size
                window_targets += int((batch.y != IGNORE).sum())
                window_games += games
                if m.step % t["log_every"] == 0:
                    dt = time.monotonic() - window_start
                    rec = {"type": "train", "step": m.step, "loss": (loss / n).item(), "lr": lr, "tokens_per_s": window_tokens / dt,
                           "padding": 1 - window_targets / window_tokens, "tflops": m.flops_per_token * window_tokens / dt / 1e12,
                           "games_seen": m.games_seen, "games_per_s": window_games / dt, "block": block_no, "elapsed_s": round(time.monotonic() - group_start)}
                    m.log(rec)
                    m.games_per_s = rec["games_per_s"]
                    write_group_status(group_dir, members, m, round_no)
                    print(f"{m.tag} step {m.step:,}  loss {rec['loss']:.4f}  lr {lr:.2e}  {rec['tokens_per_s'] / 1e3:,.0f}k tok/s  block {block_no}  games {m.games_seen:,}", file=sys.stderr)
                    window_start, window_tokens, window_targets, window_games = time.monotonic(), 0, 0, 0
            if m.step > (round(m.snapshots * m.period) if m.snapshots else 0):
                snapshot(m)
            if exhausted:
                print(f"{m.tag} stream exhausted; done", file=sys.stderr)
                m.done = True
            elif m.cooldown_end is not None and m.step >= m.cooldown_end:
                print(f"{m.tag} cooldown complete; done", file=sys.stderr)
                m.done = True
            elif m.cooldown_start is None and stop_requested(m):
                m.cooldown_start = m.step
                m.cooldown_end = round((m.snapshots + t["cooldown_snapshots"]) * m.period)
                print(f"{m.tag} STOP seen: cooling down from step {m.step:,} to {m.cooldown_end:,}", file=sys.stderr)
            if m.done:
                save_checkpoint(m.run_dir / "last.pt", {**torch.load(m.run_dir / "last.pt", map_location="cpu", weights_only=False), "done": True})
        write_group_status(group_dir, members, None, round_no)
    for m in members:
        m.metrics.close()
    print("all models done", file=sys.stderr)


group_start = time.monotonic()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="configs/experiments/stream-*.yaml")
    parser.add_argument("--resume", help="timestamp of an existing group, e.g. 20261025-093000")
    args = parser.parse_args()

    if args.resume:
        stamp = args.resume
        groups = sorted(RUNS_DIR.glob(f"*/{stamp}/group.yaml"))
        if not groups:
            sys.exit(f"no group {stamp} found under runs/")
        cfg = yaml.safe_load(open(groups[0]))
    else:
        cfg = yaml.safe_load(open(args.config))
        size = yaml.safe_load(open(REPO_ROOT / "configs" / "model" / f"{cfg['model']}.yaml"))
        cfg["train"].setdefault("lr", size.pop("lr"))
        cfg["model"] = {"name": cfg["model"], **{k: v for k, v in size.items() if k != "lr"}}
        cfg["git_commit"] = git_commit()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    group_dir = RUNS_DIR / cfg["name"]
    group_dir.mkdir(parents=True, exist_ok=True)
    members = []
    for spec in cfg["members"]:
        mcfg = member_config(cfg, spec)
        run_dir = RUNS_DIR / mcfg["name"] / stamp
        (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        if not (run_dir / "config.yaml").exists():
            yaml.safe_dump(mcfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
            yaml.safe_dump(cfg, open(run_dir / "group.yaml", "w"), sort_keys=False)
        members.append(Member(spec["tokenizer"], mcfg, run_dir, spec["snapshot_period"], spec))
    print("runs: " + ", ".join(str(m.run_dir) for m in members), file=sys.stderr)
    train_group(cfg, group_dir, members, resume=bool(args.resume))


if __name__ == "__main__":
    main()
