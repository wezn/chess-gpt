"""Training batches: whole games packed into fixed-size rows.

A row of length T holds as many complete games as fit (games longer than T are
truncated). Each game contributes (input, target) pairs shifted by one token, so
no target ever crosses into the next game. Unused space is padding.
"""

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chessgpt.tokenizers import PAD

IGNORE = -100  # target value for padding, ignored by the loss


@dataclass
class Batch:
    x: np.ndarray  # (R, T) input tokens
    y: np.ndarray  # (R, T) target tokens, IGNORE on padding
    pos: np.ndarray  # (R, T) position within the game
    doc: np.ndarray  # (R, T) game index within the row, -1 on padding


class TokenizedSplit:
    """One split (train or val) of data/tokenized/<dataset>__<tokenizer>/, memory-mapped."""

    def __init__(self, path: Path, split: str):
        self.data = np.memmap(path / f"{split}.bin", dtype=np.uint16, mode="r")
        self.starts = np.load(path / f"{split}_idx.npy")
        self.ends = np.append(self.starts[1:], len(self.data))
        self.meta = json.loads((path / "meta.json").read_text())

    def __len__(self) -> int:
        return len(self.starts)

    def pair_lengths(self, block_size: int) -> np.ndarray:
        """(input, target) pairs each game contributes: its length minus one, capped at block_size."""
        return np.minimum(self.ends - self.starts - 1, block_size)


def pack_rows(
    pair_lengths: np.ndarray, order: np.ndarray, block_size: int, pool_size: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Pack games into rows of at most block_size tokens, mostly following `order`.

    Best-fit with a lookahead pool: the next `pool_size` games are kept in a pool and
    each row is filled with the largest pooled game that still fits. This leaves far
    less padding than filling rows strictly in order.

    Returns (packed, bounds): row r holds games packed[bounds[r]:bounds[r + 1]].
    """
    lengths = pair_lengths[order].tolist()
    buckets: dict[int, list[int]] = {}  # length -> indices into order
    sizes: list[int] = []  # sorted lengths that have a non-empty bucket
    packed: list[int] = []
    bounds = [0]
    nxt, pooled = 0, 0
    while nxt < len(order) or pooled:
        while pooled < pool_size and nxt < len(order):
            n = lengths[nxt]
            if n in buckets:
                buckets[n].append(nxt)
            else:
                buckets[n] = [nxt]
                bisect.insort(sizes, n)
            nxt += 1
            pooled += 1
        room = block_size
        while (k := bisect.bisect_right(sizes, room) - 1) >= 0:
            n = sizes[k]
            bucket = buckets[n]
            packed.append(bucket.pop())
            if not bucket:
                del buckets[n]
                sizes.pop(k)
            room -= n
            pooled -= 1
        bounds.append(len(packed))
    return order[np.array(packed, dtype=np.int64)], np.array(bounds, dtype=np.int64)


def make_batch(split: TokenizedSplit, games: list[np.ndarray], block_size: int) -> Batch:
    """Build a batch where row r contains the games listed in games[r]."""
    R, T = len(games), block_size
    x = np.full((R, T), PAD, dtype=np.int64)
    y = np.full((R, T), IGNORE, dtype=np.int64)
    pos = np.zeros((R, T), dtype=np.int64)
    doc = np.full((R, T), -1, dtype=np.int64)
    for r, row in enumerate(games):
        c = 0
        for j, g in enumerate(row.tolist()):
            s = int(split.starts[g])
            n = min(int(split.ends[g]) - s - 1, T)
            tokens = split.data[s : s + n + 1]
            x[r, c : c + n] = tokens[:-1]
            y[r, c : c + n] = tokens[1:]
            pos[r, c : c + n] = np.arange(n)
            doc[r, c : c + n] = j
            c += n
    return Batch(x, y, pos, doc)


class EpochPlan:
    """Shuffled, packed order of the training games for one epoch, split into batches of `rows` rows.

    Deterministic given (seed, epoch), so a resumed run sees the same batches, and
    experiments with different tokenizers see the games in the same order.
    """

    def __init__(self, split: TokenizedSplit, block_size: int, rows: int, seed: int, epoch: int):
        order = np.random.default_rng([seed, epoch]).permutation(len(split))
        self.order, self.bounds = pack_rows(split.pair_lengths(block_size), order, block_size)
        self.rows = rows
        self.num_batches = (len(self.bounds) - 1) // rows  # drop the last partial batch

    def batch_games(self, b: int) -> list[np.ndarray]:
        r0 = b * self.rows
        return [self.order[self.bounds[r] : self.bounds[r + 1]] for r in range(r0, r0 + self.rows)]
