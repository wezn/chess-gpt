"""Evaluate a run's snapshots as training produces them.

    uv run python -m chessgpt.eval.watch runs/large-uci-move/<timestamp>

For every snapshot in <run>/snapshots/ this appends one line to <run>/eval.jsonl:
  - legal-move rate at every position of held-out games (overall and per phase)
  - illegal-move stats in raw self-play games (the model writes its moves freely)
  - strength: many legal-only self-play games from a fixed set of openings, every move
    scored by Stockfish (accuracy, ACPL, win% lost, blunder rate) and kept per game, so
    `chessgpt.monitor` can turn them into an Elo estimate with a confidence interval
and saves the games to <run>/games/step-NNNNNN.pgn (raw) and step-NNNNNN-legal.pgn.

Order: the newest snapshot first, then always the pending snapshot farthest from any
evaluated one, so the curve is coarse but complete early and fills in. Exits once
the final snapshot is evaluated.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
import yaml

from chessgpt.eval.elo import per_game, pooled, rate_games
from chessgpt.eval.games import OPENING_PLIES, Game, legality_on_games, self_play, self_play_legality, val_games
from chessgpt.eval.player import ModelPlayer
from chessgpt.eval.stockfish import Analyzer

MAX_PLIES = 180  # fits in every tokenizer's context (~190 moves)


def snapshot_step(path: Path) -> int:
    return int(path.stem.split("-")[1])


class Status:
    """Writes <run>/eval-status.json after every step of an evaluation, for the dashboard."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "eval-status.json"
        self.step = None
        self.started = 0.0

    def begin(self, step: int) -> None:
        self.step, self.started = step, time.time()
        self("starting")

    def __call__(self, phase: str) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"step": self.step, "phase": phase, "started": self.started, "updated": time.time()}))
        os.replace(tmp, self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def write_pgn(path: Path, games: list[Game], name: str) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        for i, g in enumerate(games):
            board = chess.Board()
            for m in g.moves:
                board.push(m)
            pgn = chess.pgn.Game.from_board(board)
            pgn.headers.update(Event=f"self-play {i}", White=name, Black=name, Result=board.result(claim_draw=True), Termination=g.end)
            print(pgn, file=f, end="\n\n")


def evaluate(
    snapshot: Path, run_dir: Path, legality_set: list[dict], raw_openings: list[list[str]], strength_openings: list[list[str]], analyzer: Analyzer
) -> dict:
    start = time.monotonic()
    status = Status(run_dir)
    status.begin(snapshot_step(snapshot))
    player = ModelPlayer(snapshot)
    name = f"{player.name}@{player.step}"
    legality = legality_on_games(player, legality_set, progress=lambda d, n: status(f"legality on real games: {d:,}/{n:,} positions"))
    raw = self_play(player, raw_openings, max_plies=MAX_PLIES, progress=lambda ply, a, n: status(f"raw self-play: move {ply // 2 + 1}, {a}/{n} games going"))
    write_pgn(run_dir / "games" / f"{snapshot.stem}.pgn", raw, name)
    strong = self_play(
        player, strength_openings, max_plies=MAX_PLIES, legal_only=True,
        progress=lambda ply, a, n: status(f"legal-only self-play: move {ply // 2 + 1}, {a}/{n} games going"),
    )
    write_pgn(run_dir / "games" / f"{snapshot.stem}-legal.pgn", strong, name)
    del player
    torch.cuda.empty_cache()

    games = per_game(rate_games(analyzer, [(g.moves, g.opening_plies) for g in strong], progress=lambda d, n: status(f"Stockfish scoring: {d:,}/{n:,} positions")))
    status.clear()
    quality = pooled(games)
    ends = Counter(g.end for g in strong)
    ckpt_meta = torch.load(snapshot, map_location="cpu", weights_only=False, mmap=True)
    return {
        "version": 2,
        "step": ckpt_meta["step"],
        "progress": ckpt_meta.get("progress"),
        "val_loss_move": ckpt_meta.get("val_loss_move"),
        **legality,
        **self_play_legality(raw),
        "strength_games": len(strong),
        "strength_avg_plies": float(np.mean([len(g.moves) for g in strong])),
        **{f"strength_end_{k}": v / len(strong) for k, v in ends.items()},
        **{f"strength_{k}": quality[k] for k in ("accuracy", "acpl", "win_loss", "blunder_rate")},
        "strength_scored_moves": quality["moves"],
        "strength_per_game": games,
        "eval_seconds": round(time.monotonic() - start),
    }


def next_pending(pending: list[Path], done: set[int]) -> Path:
    """Newest first; afterwards the snapshot farthest from anything evaluated (coarse-to-fine)."""
    if not done:
        return pending[-1]
    return max(pending, key=lambda p: (min(abs(snapshot_step(p) - d) for d in done), snapshot_step(p)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--legality-games", type=int, default=100, help="held-out games for the legal-move rate")
    parser.add_argument("--selfplay-games", type=int, default=64, help="raw self-play games, for illegal-move stats")
    parser.add_argument("--strength-games", type=int, default=256, help="legal-only self-play games scored for Elo")
    parser.add_argument("--workers", type=int, default=12, help="Stockfish processes")
    parser.add_argument("--once", action="store_true", help="evaluate existing snapshots, then exit")
    parser.add_argument("--poll", type=int, default=30, help="seconds between checks for new snapshots")
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    dataset = cfg["data"].split("__")[0]
    n_raw, n_strong = args.selfplay_games, args.strength_games
    held_out = val_games(dataset, args.legality_games + n_raw + n_strong, min_plies=OPENING_PLIES + 10, seed=1)
    legality_set = held_out[: args.legality_games]
    openings = [g["moves"].split()[:OPENING_PLIES] for g in held_out[args.legality_games :]]
    raw_openings, strength_openings = openings[:n_raw], openings[n_raw:]
    out = run_dir / "eval.jsonl"

    with Analyzer(args.workers) as analyzer:
        while True:
            records = [json.loads(line) for line in open(out)] if out.exists() else []
            records = [r for r in records if r.get("version") == 2]  # older rows are re-evaluated
            done = {r["step"] for r in records}
            pending = [p for p in sorted((run_dir / "snapshots").glob("step-*.pt"), key=snapshot_step) if snapshot_step(p) not in done]
            if not pending:
                if args.once or any((r.get("progress") or 0) >= 1 for r in records):
                    break
                time.sleep(args.poll)
                continue
            record = evaluate(next_pending(pending, done), run_dir, legality_set, raw_openings, strength_openings, analyzer)
            with open(out, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"step {record['step']:,} ({record['progress']:.0%})  ACPL {record['strength_acpl']:.1f}  ".replace(f" ({record['progress']:.0%})" if record['progress'] is not None else "", "") if record['progress'] is not None else f"step {record['step']:,}  ACPL {record['strength_acpl']:.1f}  "
                f"blunders {record['strength_blunder_rate']:.2%}  legal {record['legal_rate']:.2%}  "
                f"self-play illegal {record['selfplay_first_try_illegal_rate']:.2%}  [{record['eval_seconds']}s]",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
