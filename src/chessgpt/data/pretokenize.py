"""Tokenize a processed dataset once, so training can read token ids straight from disk.

    uv run python -m chessgpt.data.pretokenize lichess-2026-09-rapid uci-move

Writes data/tokenized/<dataset>__<tokenizer>/:
    train.bin, val.bin          every game as <bos> ... <eos>, concatenated, uint16 token ids
    train_idx.npy, val_idx.npy  start offset of each game in the matching .bin (int64)
    meta.json                   tokenizer vocab and token/move counts

The validation split is chosen by hashing game IDs, so it holds the same games
for every tokenizer and experiments stay comparable.

Streaming datasets (written as chunk-NNNN.parquet by `chessgpt.data.lichess` with
`chunk_games`) are tokenized by a follower that keeps up with the pull:

    uv run python -m chessgpt.data.pretokenize lichess-stream --follow

It writes data/tokenized/<dataset>__<tokenizer>/ for every tokenizer at once:
    meta.json                     written first, so trainers can start
    val.bin, val_idx.npy          the held-out games of the FIRST chunk only (a fixed validation set)
    chunk-NNNN.bin, chunk-NNNN_idx.npy   the training games of each chunk, appearing as chunks land
    status.json                   chunks done, for the dashboard
and exits once the pull is finished and every chunk is tokenized.
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
_tokenizers: dict[str, Tokenizer] = {}
TOKENIZERS = ("pgn-char", "uci-move", "uci-square")


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


def _init_multi(names: tuple[str, ...]) -> None:
    global _tokenizers
    _tokenizers = {n: get_tokenizer(n) for n in names}


def _encode_batch_multi(args: tuple[list[str], list[str], int, bool]) -> dict:
    """Encode a batch with every tokenizer at once: {tokenizer: {split: (tokens, lengths, plies)}}.
    With keep_val=False the validation games are dropped (only the first chunk keeps them)."""
    game_ids, moves, val_per_mille, keep_val = args
    out = {n: {s: ([], [], 0) for s in SPLITS} for n in _tokenizers}
    for game_id, game in zip(game_ids, moves):
        sans = game.split()
        split = "val" if is_val(game_id, val_per_mille) else "train"
        if split == "val" and not keep_val:
            continue
        for n, tok in _tokenizers.items():
            tokens, lengths, plies = out[n][split]
            ids = tok.encode(sans)
            tokens.extend(ids)
            lengths.append(len(ids))
            out[n][split] = (tokens, lengths, plies + len(sans))
    return {n: {s: (np.array(t, dtype=np.uint16), l, p) for s, (t, l, p) in d.items()} for n, d in out.items()}


def _write_split(out_dir: Path, name: str, tokens: list[np.ndarray], lengths: list[np.ndarray]) -> dict:
    """Write <name>.bin then <name>_idx.npy atomically (the idx file last: its presence means 'ready')."""
    data = np.concatenate(tokens) if tokens else np.zeros(0, dtype=np.uint16)
    lens = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
    starts = np.cumsum(lens) - lens
    tmp = out_dir / f"{name}.bin.partial"
    data.tofile(tmp)
    tmp.replace(out_dir / f"{name}.bin")
    tmp = out_dir / f"{name}_idx.partial.npy"  # np.save insists on the .npy suffix
    np.save(tmp, starts)
    tmp.replace(out_dir / f"{name}_idx.npy")
    return {"games": int(len(lens)), "tokens": int(len(data))}


def follow(dataset: str, tokenizers: tuple[str, ...] = TOKENIZERS, val_per_mille: int = 10, workers: int = 6, poll: int = 15) -> None:
    """Tokenize the chunks of a streaming dataset as they appear, for every tokenizer, until the pull is done."""
    source = PROCESSED_DIR / dataset
    dirs = {n: TOKENIZED_DIR / f"{dataset}__{n}" for n in tokenizers}
    for n, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        tok = get_tokenizer(n)
        if not (d / "meta.json").exists():
            (d / "meta.json").write_text(json.dumps({"dataset": dataset, "tokenizer": n, "vocab_size": tok.vocab_size, "itos": tok.itos, "val_per_mille": val_per_mille, "streaming": True, "git_commit": git_commit()}, indent=2) + "\n")

    def done_chunks(n: str) -> set[int]:
        return {int(p.stem.split("-")[1].split("_")[0]) for p in dirs[n].glob("chunk-*_idx.npy")}

    def write_status(n: str, extra: dict) -> None:
        done = done_chunks(n)
        st = {"chunks": len(done), "last_chunk": max(done, default=0), "updated": time.time(), **extra}
        tmp = dirs[n] / "status.json.tmp"
        tmp.write_text(json.dumps(st))
        tmp.replace(dirs[n] / "status.json")

    start = time.monotonic()
    games_total = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_multi, initargs=(tuple(tokenizers),)) as pool:
        while True:
            status = json.loads((source / "status.json").read_text()) if (source / "status.json").exists() else {}
            available = sorted(int(p.stem.split("-")[1]) for p in source.glob("chunk-*.parquet"))
            pending = [c for c in available if any(c not in done_chunks(n) for n in tokenizers)]
            if not pending:
                if status.get("done"):
                    break
                time.sleep(poll)
                continue
            c = pending[0]
            keep_val = c == 1
            pf = pq.ParquetFile(source / f"chunk-{c:04d}.parquet")
            jobs = ((b.column("game_id").to_pylist(), b.column("moves").to_pylist(), val_per_mille, keep_val) for b in pf.iter_batches(batch_size=5_000, columns=["game_id", "moves"]))
            acc = {n: {s: ([], []) for s in SPLITS} for n in tokenizers}
            games = 0
            for result in pool.imap(_encode_batch_multi, jobs):
                for n in tokenizers:
                    for s in SPLITS:
                        tokens, lengths, _ = result[n][s]
                        acc[n][s][0].append(tokens)
                        acc[n][s][1].append(np.array(lengths, dtype=np.int64))
                games += len(result[tokenizers[0]]["train"][1])
            for n in tokenizers:
                if keep_val and not (dirs[n] / "val_idx.npy").exists():
                    _write_split(dirs[n], "val", *acc[n]["val"])
                info = _write_split(dirs[n], f"chunk-{c:04d}", *acc[n]["train"])
                write_status(n, {"games_per_s": round(games_total / max(1e-9, time.monotonic() - start))})
            games_total += games
            print(f"chunk {c:04d}: {games:,} train games  ({games_total / (time.monotonic() - start):,.0f} games/s overall)", file=sys.stderr)
    print("follower done: pull finished and every chunk tokenized", file=sys.stderr)


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
    parser.add_argument("tokenizer", nargs="?", help="pgn-char, uci-move or uci-square (all three with --follow)")
    parser.add_argument("--val-per-mille", type=int, default=10, help="validation games per 1000 (default 10 = 1%%)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--follow", action="store_true", help="streaming dataset: tokenize chunks as they land, every tokenizer at once")
    args = parser.parse_args()
    if args.follow:
        follow(args.dataset, val_per_mille=args.val_per_mille, workers=args.workers)
    else:
        pretokenize(args.dataset, args.tokenizer, args.val_per_mille, args.workers)


if __name__ == "__main__":
    main()
