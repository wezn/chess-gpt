"""Evaluations that use a ModelPlayer: legality on real games, and self-play games.

Openings: self-play games start from the first OPENING_PLIES moves of held-out
Lichess games, so games are varied. Those opening moves are never scored, both
here and in the Elo calibration.
"""

import glob
import random
from collections import Counter
from dataclasses import dataclass, field

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chessgpt.data.common import PROCESSED_DIR
from chessgpt.data.pretokenize import is_val
from chessgpt.eval.player import ModelPlayer

OPENING_PLIES = 6
PHASES = [("opening", 0, 20), ("middlegame", 20, 60), ("endgame", 60, 10_000)]  # by ply


def phase_of(ply: int) -> str:
    return next(name for name, lo, hi in PHASES if lo <= ply < hi)


def val_games(dataset: str, n: int, min_plies: int = 0, seed: int = 0) -> list[dict]:
    """n held-out games (the validation split, never trained on), deterministic."""
    files = sorted(glob.glob(str(PROCESSED_DIR / dataset / "*.parquet")))
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["game_id", "white_elo", "black_elo", "num_plies", "moves"])
        mask = [is_val(g, 10) for g in t.column("game_id").to_pylist()]
        rows += [r for r in t.filter(pa.array(mask)).to_pylist() if r["num_plies"] >= min_plies]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def legality_on_games(player: ModelPlayer, games: list[dict]) -> dict:
    """Ask the model for a move at every position of real games (one attempt, greedy).

    Reports the legal-move rate overall and per game phase, and how often the
    model picked the move the human actually played.
    """
    boards, human = [], []
    for g in games:
        board = chess.Board()
        for san in g["moves"].split():
            boards.append(board.copy())
            move = board.parse_san(san)
            human.append(move)
            board.push(move)
    choices = player.choose_moves(boards, max_attempts=1)
    stats: Counter[str] = Counter()
    for board, choice, move in zip(boards, choices, human):
        if choice.status == "context":
            continue
        phase = phase_of(len(board.move_stack))
        legal = choice.status == "legal"
        stats["positions"] += 1
        stats[f"{phase}_positions"] += 1
        stats["legal"] += legal
        stats[f"{phase}_legal"] += legal
        stats["match"] += choice.move == move
    out = {
        "legal_rate": stats["legal"] / stats["positions"],
        "human_match_rate": stats["match"] / stats["positions"],
        "positions": stats["positions"],
    }
    for name, _, _ in PHASES:
        if stats[f"{name}_positions"]:
            out[f"legal_rate_{name}"] = stats[f"{name}_legal"] / stats[f"{name}_positions"]
    return out


@dataclass
class Game:
    moves: list[chess.Move]
    opening_plies: int
    illegal_attempts: list[int] = field(default_factory=list)  # per model move
    random_fallback: list[bool] = field(default_factory=list)  # per model move: every attempt was illegal
    end: str = ""  # "checkmate", "draw", "max_plies" or "context"


def self_play(player: ModelPlayer, openings: list[list[str]], max_plies: int = 200, seed: int = 0) -> list[Game]:
    """The model plays both sides of each game, all games batched together.

    Illegal moves are retried (sampling); if every attempt fails, a random legal
    move is played so the game can continue, and that is recorded.
    """
    rng = random.Random(seed)
    boards, games = [], []
    for sans in openings:
        board = chess.Board()
        for san in sans[:OPENING_PLIES]:
            board.push_san(san)
        boards.append(board)
        games.append(Game([], OPENING_PLIES))
    active = list(range(len(boards)))
    while active:
        choices = player.choose_moves([boards[i] for i in active])
        still = []
        for i, choice in zip(active, choices):
            board, game = boards[i], games[i]
            if choice.status == "context":
                game.end = "context"
                continue
            fallback = choice.move is None
            board.push(rng.choice(list(board.legal_moves)) if fallback else choice.move)
            game.illegal_attempts.append(choice.illegal_attempts)
            game.random_fallback.append(fallback)
            if board.is_checkmate():
                game.end = "checkmate"
            elif board.is_game_over(claim_draw=True):
                game.end = "draw"
            elif len(board.move_stack) >= max_plies:
                game.end = "max_plies"
            else:
                still.append(i)
        active = still
    for board, game in zip(boards, games):
        game.moves = list(board.move_stack)
    return games


def self_play_legality(games: list[Game]) -> dict:
    """Illegal-move statistics in the model's own games, where it reaches positions it created itself."""
    attempts = np.concatenate([g.illegal_attempts for g in games])
    fallback = np.concatenate([g.random_fallback for g in games])
    # Moves played before the first illegal first-attempt, per game.
    until_first = [next((k for k, a in enumerate(g.illegal_attempts) if a > 0), len(g.illegal_attempts)) for g in games]
    ends = Counter(g.end for g in games)
    return {
        "selfplay_first_try_illegal_rate": float(np.mean(attempts > 0)),
        "selfplay_random_fallback_rate": float(np.mean(fallback)),
        "selfplay_moves_before_first_illegal": float(np.mean(until_first)),
        "selfplay_games_without_illegal": float(np.mean([u == len(g.illegal_attempts) for u, g in zip(until_first, games)])),
        "selfplay_avg_plies": float(np.mean([len(g.moves) for g in games])),
        **{f"selfplay_end_{k}": v / len(games) for k, v in ends.items()},
    }
