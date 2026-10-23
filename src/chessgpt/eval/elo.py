"""Estimate the Elo a player plays at from the accuracy of its moves, calibrated on Lichess games.

This is the idea behind chess.com's "Game Review" rating, built from public pieces:
Stockfish evaluates every position, each move gets a Lichess-style accuracy, and the
accuracy is mapped to a rating.

Calibration: analyse held-out Lichess games whose players' ratings we know and
measure the average accuracy at each rating. A model's estimate is the rating whose
average accuracy matches the model's average accuracy over many moves.

    uv run python -m chessgpt.eval.elo calibrate    # writes results/elo_calibration.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import chess
import numpy as np

from chessgpt.data.common import REPO_ROOT
from chessgpt.eval.games import OPENING_PLIES, val_games
from chessgpt.eval.stockfish import ANALYSIS_NODES, Analyzer, MoveScore, game_fens, score_game

CALIBRATION_PATH = REPO_ROOT / "results" / "elo_calibration.json"
BIN_WIDTH = 100
MAX_ELO_GAP = 150  # calibration games are between similarly rated players, like self-play


def rate_games(analyzer: Analyzer, games: Sequence[tuple[Sequence[chess.Move], int]]) -> list[list[MoveScore]]:
    """Score every move after the opening of each (moves, opening_plies) game, analysing all positions at once."""
    fens = [game_fens(moves) for moves, _ in games]
    evals = analyzer.evaluate([f for game in fens for f in game])
    out, i = [], 0
    for (moves, opening), game in zip(games, fens):
        out.append(score_game(evals[i : i + len(game)], first_ply=opening))
        i += len(game)
    return out


def summarize(scores: Sequence[MoveScore]) -> dict:
    return {
        "accuracy": float(np.mean([s.accuracy for s in scores])),
        "acpl": float(np.mean([s.cp_loss for s in scores])),
        "moves": len(scores),
    }


def calibrate(dataset: str, games_per_bin: int, lo: int, hi: int, workers: int) -> dict:
    pool = val_games(dataset, n=10**9, min_plies=OPENING_PLIES + 20)
    by_bin: dict[int, list[dict]] = defaultdict(list)
    for g in pool:
        if abs(g["white_elo"] - g["black_elo"]) <= MAX_ELO_GAP:
            b = (g["white_elo"] + g["black_elo"]) // 2 // BIN_WIDTH * BIN_WIDTH
            if lo <= b < hi and len(by_bin[b]) < games_per_bin:
                by_bin[b].append(g)
    games = [g for b in sorted(by_bin) for g in by_bin[b]]
    print(f"analysing {len(games)} games: " + ", ".join(f"{b}:{len(by_bin[b])}" for b in sorted(by_bin)), file=sys.stderr)

    parsed = []
    for g in games:
        board = chess.Board()
        parsed.append(([board.push_san(san) for san in g["moves"].split()], OPENING_PLIES))
    start = time.monotonic()
    with Analyzer(workers) as analyzer:
        scores = rate_games(analyzer, parsed)

    # Pool every move by the rating of the player who made it.
    moves_by_bin: dict[int, list[MoveScore]] = defaultdict(list)
    elos_by_bin: dict[int, list[int]] = defaultdict(list)
    for g, game_scores in zip(games, scores):
        for color, elo in ((0, g["white_elo"]), (1, g["black_elo"])):
            b = elo // BIN_WIDTH * BIN_WIDTH
            mine = [s for s in game_scores if s.ply % 2 == color]
            moves_by_bin[b] += mine
            elos_by_bin[b] += [elo] * len(mine)
    bins = [
        {"elo": float(np.mean(elos_by_bin[b])), **summarize(moves_by_bin[b])}
        for b in sorted(moves_by_bin)
        if len(moves_by_bin[b]) >= 500
    ]
    result = {
        "dataset": dataset,
        "analysis_nodes": ANALYSIS_NODES,
        "opening_plies": OPENING_PLIES,
        "max_elo_gap": MAX_ELO_GAP,
        "games": len(games),
        "seconds": round(time.monotonic() - start),
        "bins": bins,
    }
    CALIBRATION_PATH.parent.mkdir(exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(result, indent=2) + "\n")
    return result


@dataclass
class EloCalibration:
    elos: np.ndarray
    accuracies: np.ndarray

    @classmethod
    def load(cls) -> "EloCalibration":
        cal = json.loads(CALIBRATION_PATH.read_text())
        if cal["analysis_nodes"] != ANALYSIS_NODES:
            raise ValueError("calibration used different Stockfish settings; recalibrate")
        elos = np.array([b["elo"] for b in cal["bins"]])
        # Accuracy should rise with rating; enforce it so the curve can be inverted.
        acc = np.maximum.accumulate(np.array([b["accuracy"] for b in cal["bins"]]))
        return cls(elos, acc)

    def estimate(self, accuracy: float) -> dict:
        """Elo whose average accuracy equals `accuracy`. Outside the calibrated range, it is clamped and flagged."""
        elo = float(np.interp(accuracy, self.accuracies, self.elos))
        flag = "below_range" if accuracy < self.accuracies[0] else "above_range" if accuracy > self.accuracies[-1] else "ok"
        return {"elo_estimate": round(elo), "elo_range": flag}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="analyse held-out Lichess games to map accuracy to Elo")
    c.add_argument("--dataset", default="lichess-2026-09-rapid")
    c.add_argument("--games-per-bin", type=int, default=80)
    c.add_argument("--lo", type=int, default=600)
    c.add_argument("--hi", type=int, default=2700)
    c.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    result = calibrate(args.dataset, args.games_per_bin, args.lo, args.hi, args.workers)
    for b in result["bins"]:
        print(f"elo {b['elo']:6.0f}  accuracy {b['accuracy']:5.1f}  acpl {b['acpl']:5.1f}  moves {b['moves']}")


if __name__ == "__main__":
    main()
