"""Stockfish analysis, run in parallel, and move accuracy scoring.

Accuracy uses the formulas Lichess publishes (https://lichess.org/page/accuracy):
an evaluation in centipawns becomes a win probability, and a move's accuracy
depends on how much it lowers the mover's win probability.
"""

import math
import multiprocessing
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass

import chess
import chess.engine

STOCKFISH = shutil.which("stockfish") or os.path.expanduser("~/.local/bin/stockfish")
ANALYSIS_NODES = 50_000  # fixed per position; calibration and model evals must use the same value
MATE_CP = 10_000
CP_CAP = 1_000  # for centipawn loss, so one blunder in a lost position doesn't dominate the average

_engine: chess.engine.SimpleEngine


def _init_worker() -> None:
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    _engine.configure({"Threads": 1, "Hash": 16})


def _evaluate(fen: str) -> int:
    """Evaluation of a position in centipawns from White's point of view."""
    board = chess.Board(fen)
    if board.is_checkmate():
        return -MATE_CP if board.turn == chess.WHITE else MATE_CP
    if board.is_game_over():
        return 0
    info = _engine.analyse(board, chess.engine.Limit(nodes=ANALYSIS_NODES))
    return info["score"].white().score(mate_score=MATE_CP)


class Analyzer:
    """A pool of Stockfish processes that evaluates many positions in parallel."""

    def __init__(self, workers: int = 16):
        self.pool = multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker)

    def evaluate(self, fens: Sequence[str]) -> list[int]:
        return self.pool.map(_evaluate, fens, chunksize=16)

    def close(self) -> None:
        self.pool.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def win_percent(cp: float) -> float:
    """Lichess win probability (0-100) for the side whose point of view `cp` is from."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(cp_before: float, cp_after: float) -> float:
    """Accuracy (0-100) of a move, given evaluations before and after it from the mover's point of view."""
    drop = win_percent(cp_before) - win_percent(cp_after)
    return min(100.0, max(0.0, 103.1668 * math.exp(-0.04354 * drop) - 3.1669))


@dataclass
class MoveScore:
    ply: int  # 0 = White's first move
    accuracy: float
    cp_loss: float  # centipawns lost, evaluations capped at +-CP_CAP
    win_loss: float  # win probability lost, in percentage points


def _clamp(cp: float) -> float:
    return max(-CP_CAP, min(CP_CAP, cp))


def score_game(evals: Sequence[int], first_ply: int = 0) -> list[MoveScore]:
    """Score every move of a game from the White-POV evaluations of each position (len = moves + 1)."""
    scores = []
    for ply in range(first_ply, len(evals) - 1):
        sign = 1 if ply % 2 == 0 else -1
        before, after = sign * evals[ply], sign * evals[ply + 1]
        cp_loss = max(0, _clamp(before) - _clamp(after))
        win_loss = max(0.0, win_percent(before) - win_percent(after))
        scores.append(MoveScore(ply, move_accuracy(before, after), cp_loss, win_loss))
    return scores


def game_fens(moves: Sequence[chess.Move]) -> list[str]:
    """FEN of every position in a game from the start, including the final one."""
    board = chess.Board()
    fens = [board.fen()]
    for move in moves:
        board.push(move)
        fens.append(board.fen())
    return fens
