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
import math
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chessgpt.data.common import REPO_ROOT
from chessgpt.eval.games import OPENING_PLIES, val_games
from chessgpt.eval.stockfish import ANALYSIS_NODES, Analyzer, MoveScore, game_fens, score_game

CALIBRATION_PATH = REPO_ROOT / "results" / "elo_calibration.json"
MOVES_PATH = REPO_ROOT / "data" / "calibration" / "moves.parquet"  # every scored move, for refitting
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


STATS = ("accuracy", "acpl", "win_loss")  # per-move statistics that can be mapped to Elo
SE_MOVES = 5000  # typical number of scored moves in one evaluation, for the noise estimate


def summarize(scores: Sequence[MoveScore]) -> dict:
    return {
        "accuracy": float(np.mean([s.accuracy for s in scores])),
        "acpl": float(np.mean([s.cp_loss for s in scores])),
        "win_loss": float(np.mean([s.win_loss for s in scores])),
        "moves": len(scores),
    }


def calibrate(dataset: str, games_per_bin: int, lo: int, hi: int, workers: int) -> None:
    """Analyse held-out games and save every scored move with its player's rating, then fit."""
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

    rows = []
    for g, game_scores in zip(games, scores):
        for m in game_scores:
            elo = g["white_elo"] if m.ply % 2 == 0 else g["black_elo"]
            rows.append({"game_id": g["game_id"], "elo": elo, "ply": m.ply, "accuracy": m.accuracy, "cp_loss": m.cp_loss, "win_loss": m.win_loss})
    MOVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), MOVES_PATH)
    print(f"saved {len(rows):,} scored moves to {MOVES_PATH} ({time.monotonic() - start:.0f}s)", file=sys.stderr)
    fit(dataset)


def fit(dataset: str) -> dict:
    """Fit each statistic as a smooth function of Elo, from the saved calibration moves.

    The fit is stat = a + b*elo + c*elo^2, weighted by moves per bin. A model's Elo is
    found by inverting the curve. The statistic used for the main estimate is the one
    with the least Elo noise for a typical evaluation (SE_MOVES moves).
    """
    t = pq.read_table(MOVES_PATH).to_pandas()
    t["acpl"] = t["cp_loss"]
    t["bin"] = t["elo"] // BIN_WIDTH * BIN_WIDTH
    bins = t.groupby("bin").filter(lambda g: len(g) >= 500).groupby("bin")
    table = bins.agg(elo=("elo", "mean"), moves=("elo", "size"), **{k: (k, "mean") for k in STATS}, **{f"{k}_std": (k, "std") for k in STATS})
    fits = {}
    for stat in STATS:
        coef = np.polyfit(table["elo"], table[stat], 2, w=np.sqrt(table["moves"]))
        grid = np.arange(table["elo"].min(), table["elo"].max() + 1)
        curve = np.polyval(coef, grid)
        slope = np.abs(np.gradient(curve)).mean()  # stat units per Elo point
        noise = table[f"{stat}_std"].mean() / np.sqrt(SE_MOVES)
        residual = float(np.sqrt(np.mean((np.polyval(coef, table["elo"]) - table[stat]) ** 2)) / slope)
        monotonic = bool(np.all(np.diff(curve) > 0) or np.all(np.diff(curve) < 0))
        fits[stat] = {"coef": coef.tolist(), "elo_noise": float(noise / slope), "elo_residual": residual, "monotonic": monotonic}
    usable = [s for s in STATS if fits[s]["monotonic"]]
    primary = min(usable, key=lambda s: math.hypot(fits[s]["elo_noise"], fits[s]["elo_residual"]))
    result = {
        "dataset": dataset,
        "analysis_nodes": ANALYSIS_NODES,
        "opening_plies": OPENING_PLIES,
        "max_elo_gap": MAX_ELO_GAP,
        "games": int(t["game_id"].nunique()),
        "primary_stat": primary,
        "elo_range": [float(table["elo"].min()), float(table["elo"].max())],
        "fits": fits,
        "bins": table.reset_index(drop=True).round(3).to_dict(orient="records"),
    }
    CALIBRATION_PATH.parent.mkdir(exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(result, indent=2) + "\n")
    return result


@dataclass
class EloCalibration:
    fits: dict
    primary: str
    lo: float
    hi: float

    @classmethod
    def load(cls) -> "EloCalibration":
        cal = json.loads(CALIBRATION_PATH.read_text())
        if cal["analysis_nodes"] != ANALYSIS_NODES or cal["opening_plies"] != OPENING_PLIES:
            raise ValueError("calibration used different settings; recalibrate")
        return cls(cal["fits"], cal["primary_stat"], *cal["elo_range"])

    def elo_for(self, stat: str, value: float) -> tuple[float, str]:
        grid = np.arange(self.lo, self.hi + 1)
        curve = np.polyval(self.fits[stat]["coef"], grid)
        order = np.argsort(curve)
        elo = float(np.interp(value, curve[order], grid[order]))
        if curve.min() <= value <= curve.max():
            flag = "ok"
        else:  # clamped; which end is it beyond?
            flag = "below_range" if abs(value - curve[0]) < abs(value - curve[-1]) else "above_range"
        return elo, flag

    def estimate(self, summary: dict) -> dict:
        """Elo estimates from a summarize() result: the primary one, plus one per statistic for reference."""
        elo, flag = self.elo_for(self.primary, summary[self.primary])
        out = {"elo_estimate": round(elo), "elo_range": flag, "elo_stat": self.primary}
        for stat in STATS:
            out[f"elo_by_{stat}"] = round(self.elo_for(stat, summary[stat])[0])
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="analyse held-out Lichess games, then fit")
    c.add_argument("--dataset", default="lichess-2026-09-rapid")
    c.add_argument("--games-per-bin", type=int, default=160)
    c.add_argument("--lo", type=int, default=600)
    c.add_argument("--hi", type=int, default=2500)
    c.add_argument("--workers", type=int, default=16)
    f = sub.add_parser("fit", help="refit from the saved calibration moves (no new analysis)")
    f.add_argument("--dataset", default="lichess-2026-09-rapid")
    args = parser.parse_args()
    if args.cmd == "calibrate":
        calibrate(args.dataset, args.games_per_bin, args.lo, args.hi, args.workers)
    result = fit(args.dataset)
    for b in result["bins"]:
        print("  ".join(f"{k} {b[k]:7.1f}" for k in ("elo", *STATS)) + f"  moves {b['moves']}")
    for stat, info in result["fits"].items():
        print(f"{stat:9s} Elo noise +-{info['elo_noise']:.0f}  fit residual +-{info['elo_residual']:.0f}  monotonic {info['monotonic']}")
    print(f"primary statistic: {result['primary_stat']}")


if __name__ == "__main__":
    main()
