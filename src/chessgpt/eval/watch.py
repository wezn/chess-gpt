"""Evaluate a run's snapshots as training produces them.

    uv run python -m chessgpt.eval.watch runs/large-uci-move/<timestamp>

For every snapshot in <run>/snapshots/ this appends one line to <run>/eval.jsonl:
  - legal-move rate at every position of held-out games (overall and per phase)
  - illegal-move stats in self-play games, where the model reaches its own positions
  - estimated Elo, from the accuracy of the model's self-play moves (see eval/elo.py)
and saves the self-play games to <run>/games/step-NNNNNN.pgn.

The newest snapshot is evaluated first so results keep up with training; older
ones are filled in afterwards. Exits once the final snapshot is evaluated.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import torch
import yaml

from chessgpt.eval.elo import EloCalibration, rate_games, summarize
from chessgpt.eval.games import OPENING_PLIES, Game, legality_on_games, self_play, self_play_legality, val_games
from chessgpt.eval.player import ModelPlayer
from chessgpt.eval.stockfish import Analyzer

MAX_PLIES = 180  # fits in every tokenizer's context (~190 moves)


def snapshot_step(path: Path) -> int:
    return int(path.stem.split("-")[1])


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


def evaluate(snapshot: Path, run_dir: Path, legality_set: list[dict], openings: list[list[str]], analyzer: Analyzer, calibration: EloCalibration) -> dict:
    start = time.monotonic()
    player = ModelPlayer(snapshot)
    legality = legality_on_games(player, legality_set)
    games = self_play(player, openings, max_plies=MAX_PLIES)
    write_pgn(run_dir / "games" / f"{snapshot.stem}.pgn", games, f"{player.name}@{player.step}")
    del player
    torch.cuda.empty_cache()

    scores = rate_games(analyzer, [(g.moves, g.opening_plies) for g in games])
    quality = summarize([s for game in scores for s in game])
    ckpt_meta = torch.load(snapshot, map_location="cpu", weights_only=False, mmap=True)
    return {
        "step": ckpt_meta["step"],
        "progress": ckpt_meta.get("progress"),
        "val_loss_move": ckpt_meta.get("val_loss_move"),
        **legality,
        **self_play_legality(games),
        "selfplay_accuracy": quality["accuracy"],
        "selfplay_acpl": quality["acpl"],
        "selfplay_scored_moves": quality["moves"],
        **calibration.estimate(quality["accuracy"]),
        "eval_seconds": round(time.monotonic() - start),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--legality-games", type=int, default=100)
    parser.add_argument("--selfplay-games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12, help="Stockfish processes")
    parser.add_argument("--once", action="store_true", help="evaluate existing snapshots, then exit")
    parser.add_argument("--poll", type=int, default=30, help="seconds between checks for new snapshots")
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    dataset = cfg["data"].split("__")[0]
    held_out = val_games(dataset, args.legality_games + args.selfplay_games, min_plies=OPENING_PLIES + 10, seed=1)
    legality_set = held_out[: args.legality_games]
    openings = [g["moves"].split()[:OPENING_PLIES] for g in held_out[args.legality_games :]]
    calibration = EloCalibration.load()
    out = run_dir / "eval.jsonl"

    with Analyzer(args.workers) as analyzer:
        while True:
            records = [json.loads(line) for line in open(out)] if out.exists() else []
            done = {r["step"] for r in records}
            pending = [p for p in sorted((run_dir / "snapshots").glob("step-*.pt"), key=snapshot_step) if snapshot_step(p) not in done]
            if not pending:
                if args.once or any((r.get("progress") or 0) >= 1 for r in records):
                    break
                time.sleep(args.poll)
                continue
            record = evaluate(pending[-1], run_dir, legality_set, openings, analyzer, calibration)
            with open(out, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"step {record['step']:,} ({record['progress']:.0%})  Elo ~{record['elo_estimate']} ({record['elo_range']})  "
                f"accuracy {record['selfplay_accuracy']:.1f}  legal {record['legal_rate']:.2%}  "
                f"self-play illegal {record['selfplay_first_try_illegal_rate']:.2%}  [{record['eval_seconds']}s]",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
