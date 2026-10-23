"""Tokenize a processed dataset once, so training can read token ids straight from disk.

    uv run python -m chessgpt.data.pretokenize lichess-2026-09-rapid uci-move

Writes data/tokenized/<dataset>__<tokenizer>/:
    train.bin, val.bin          every game as <bos> ... <eos>, concatenated, uint16 token ids
    train_idx.npy, val_idx.npy  start offset of each game in the matching .bin (int64)
    meta.json                   tokenizer vocab and token/move counts

The validation split is chosen by hashing game IDs, so it holds the same games
for every tokenizer and experiments stay comparable.
"""

import argparse
import json
import shutil
import sys
import time
import zlib
import multiprocessing
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from chessgpt.data.common import PROCESSED_DIR, REPO_ROOT, git_commit
from chessgpt.tokenizers import Tokenizer, get_tokenizer

TOKENIZED_DIR = REPO_ROOT / "data" / "tokenized"
SPLITS = ("train", "val")

_tokenizer: Tokenizer


def is_val(game_id: str, val_per_mille: int) -> bool:
    return zlib.crc32(game_id.encode()) % 1000 < val_per_mille


def _init_worker(name: str) -> None:
    global _tokenizer
    _tokenizer = get_tokenizer(name)


def _encode_batch(args: tuple[list[str], list[str], int]) -> dict:
    """Encode a batch of games; returns per-split token arrays, game lengths and move counts."""
    game_ids, moves, val_per_mille = args
    out = {s: ([], [], 0) for s in SPLITS}
    for game_id, game in zip(game_ids, moves):
        sans = game.split()
        split = "val" if is_val(game_id, val_per_mille) else "train"
        tokens, lengths, plies = out[split]
        ids = _tokenizer.encode(sans)
        tokens.extend(ids)
        lengths.append(len(ids))
        out[split] = (tokens, lengths, plies + len(sans))
    return {s: (np.array(t, dtype=np.uint16), l, p) for s, (t, l, p) in out.items()}


def pretokenize(dataset: str, tokenizer_name: str, val_per_mille: int = 10, workers: int = 16) -> Path:
    tokenizer = get_tokenizer(tokenizer_name)
    assert tokenizer.vocab_size < 2**16, "token ids are stored as uint16"
    source = PROCESSED_DIR / dataset
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {source}")

    final_dir = TOKENIZED_DIR / f"{dataset}__{tokenizer_name}"
    if final_dir.exists():
        raise FileExistsError(f"{final_dir} already exists. Delete it to rebuild.")
    out_dir = TOKENIZED_DIR / f"{dataset}__{tokenizer_name}.partial"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    def batches():
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=10_000, columns=["game_id", "moves"]):
                yield batch.column("game_id").to_pylist(), batch.column("moves").to_pylist(), val_per_mille

    bins = {s: open(out_dir / f"{s}.bin", "wb") for s in SPLITS}
    offsets = {s: [] for s in SPLITS}
    stats = {s: {"games": 0, "tokens": 0, "moves": 0} for s in SPLITS}
    start = time.monotonic()
    # spawn, not fork: pyarrow starts threads, and forking a threaded process can deadlock.
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(tokenizer_name,)) as pool:
        for i, result in enumerate(pool.imap(_encode_batch, batches()), 1):
            for s, (tokens, lengths, plies) in result.items():
                tokens.tofile(bins[s])
                lengths = np.array(lengths, dtype=np.int64)
                offsets[s].append(stats[s]["tokens"] + np.cumsum(lengths) - lengths)  # start of each game
                stats[s]["games"] += len(lengths)
                stats[s]["tokens"] += len(tokens)
                stats[s]["moves"] += plies
            if i % 50 == 0:
                games = sum(st["games"] for st in stats.values())
                print(f"{games:,} games  ({games / (time.monotonic() - start):,.0f} games/s)", file=sys.stderr)
    for s in SPLITS:
        bins[s].close()
        np.save(out_dir / f"{s}_idx.npy", np.concatenate(offsets[s]))
        stats[s]["tokens_per_move"] = round(stats[s]["tokens"] / max(stats[s]["moves"], 1), 3)

    meta = {
        "dataset": dataset,
        "tokenizer": tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "itos": tokenizer.itos,
        "val_per_mille": val_per_mille,
        "splits": stats,
        "seconds": round(time.monotonic() - start),
        "git_commit": git_commit(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    out_dir.rename(final_dir)
    print(f"wrote {final_dir}: {json.dumps(stats)}", file=sys.stderr)
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="name of a dataset in data/processed/")
    parser.add_argument("tokenizer", help="pgn-char, uci-move or uci-square")
    parser.add_argument("--val-per-mille", type=int, default=10, help="validation games per 1000 (default 10 = 1%%)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    pretokenize(args.dataset, args.tokenizer, args.val_per_mille, args.workers)


if __name__ == "__main__":
    main()
