import io

import chess
import pytest

from chessgpt.data.lichess import reject_reason
from chessgpt.data.pgn import clean_moves, iter_games, speed

PGN = """\
[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "2010"]
[BlackElo "1950"]
[TimeControl "180+2"]
[Termination "Normal"]

1. e4 { [%eval 0.2] [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Qh5?! Nc6 3. Bc4 Nf6?? 4. Qxf7# 1-0

[Event "Rated Bullet game"]
[Site "https://lichess.org/efgh5678"]
[Result "0-1"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "60+0"]
[Termination "Time forfeit"]
[WhiteTitle "BOT"]

1. d4 d5 0-1
"""

FILTERS = {"min_elo": 1800, "speeds": ["blitz", "rapid"], "terminations": ["Normal", "Time forfeit"]}


def test_iter_games_splits_games_and_headers():
    games = list(iter_games(io.StringIO(PGN)))
    assert len(games) == 2
    assert games[0].headers["Site"] == "https://lichess.org/abcd1234"
    assert games[1].headers["WhiteTitle"] == "BOT"


def test_clean_moves_strips_comments_numbers_and_annotations():
    game = next(iter_games(io.StringIO(PGN)))
    moves = clean_moves(game.movetext)
    assert moves == ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]
    board = chess.Board()
    for san in moves:
        board.push_san(san)  # raises if any move is illegal
    assert board.is_checkmate()


@pytest.mark.parametrize("movetext", ["1. e4 e5 (1... c5) 2. Nf3", "1. e4 garbage"])
def test_clean_moves_rejects_unexpected_tokens(movetext):
    assert clean_moves(movetext) is None


def test_clean_moves_handles_promotion_and_castling():
    assert clean_moves("30. exd8=Q+ O-O-O 31. O-O 1/2-1/2") == ["exd8=Q+", "O-O-O", "O-O"]


@pytest.mark.parametrize(
    "tc,expected",
    [("15+0", "ultrabullet"), ("60+0", "bullet"), ("180+2", "blitz"), ("600+5", "rapid"), ("1800+0", "classical"), ("-", "correspondence")],
)
def test_speed(tc, expected):
    assert speed(tc) == expected


def test_reject_reason():
    good, bot = iter_games(io.StringIO(PGN))
    assert reject_reason(good, FILTERS) is None
    assert reject_reason(bot, FILTERS) == "bot"
    assert reject_reason(good, {**FILTERS, "min_elo": 2000}) == "elo_too_low"
