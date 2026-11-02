"""Export final checkpoints as self-contained model directories for the Hugging Face Hub.

    uv run python scripts/export_hf.py runs/stream-large-uci-move/<stamp>/last.pt uci-move-out/

Writes config.json (architecture + tokenizer, no local data needed) and model.safetensors
(weights only, no optimizer state). Load with `chessgpt.eval.player.ModelPlayer(<dir>)`.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from chessgpt.data.pretokenize import TOKENIZED_DIR


def export(checkpoint: Path, out: Path) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    meta = json.loads((TOKENIZED_DIR / cfg["data"] / "meta.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    weights = {k.removeprefix("_orig_mod."): v.contiguous() for k, v in ckpt["model"].items()}
    # tied embeddings share storage; safetensors wants each tensor stored once
    seen: dict[int, str] = {}
    for k in list(weights):
        ptr = weights[k].data_ptr()
        if ptr in seen:
            weights[k] = weights[k].clone()
        seen[ptr] = k
    save_file(weights, out / "model.safetensors")
    config = {
        "name": cfg["name"],
        "tokenizer": meta["tokenizer"],
        "vocab_size": meta["vocab_size"],
        "block_size": cfg["block_size"],
        "model": cfg["model"],
        "step": ckpt["step"],
        "games_seen": ckpt["games_seen"],
        "val_loss_move": ckpt["val_loss_move"],
        "git_commit": cfg.get("git_commit"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


if __name__ == "__main__":
    print(json.dumps(export(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
