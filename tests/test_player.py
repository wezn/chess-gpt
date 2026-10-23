import chess
import pytest
import torch

from chessgpt.eval.player import ModelPlayer, parse_partial, prompt_ids
from chessgpt.eval.stockfish import move_accuracy, score_game, win_percent
from chessgpt.model import GPT, GPTConfig
from chessgpt.tokenizers import BOS, EOS, get_tokenizer


def board_after(sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans.split():
        board.push_san(san)
    return board


def ids(tok, tokens):
    return [tok.stoi[t] for t in tokens]


def test_pgn_char_prompt_ends_with_move_number_for_white():
    tok = get_tokenizer("pgn-char")
    text = lambda b: "".join(tok.itos[i] for i in prompt_ids(tok, b)[1:])
    assert text(chess.Board()) == "1."
    assert text(board_after("e4")) == "1.e4 "
    assert text(board_after("e4 e5")) == "1.e4 e5 2."


def test_parse_pgn_char():
    tok = get_tokenizer("pgn-char")
    board = board_after("e4 e5")
    assert parse_partial(tok, board, ids(tok, "Nf")) == ("more", None)
    assert parse_partial(tok, board, ids(tok, "Nf3 ")) == ("move", chess.Move.from_uci("g1f3"))
    assert parse_partial(tok, board, ids(tok, "Nf6 "))[0] == "illegal"
    assert parse_partial(tok, board, [EOS])[0] == "illegal"


def test_parse_uci_move():
    tok = get_tokenizer("uci-move")
    assert parse_partial(tok, chess.Board(), ids(tok, ["e2e4"])) == ("move", chess.Move.from_uci("e2e4"))
    assert parse_partial(tok, chess.Board(), ids(tok, ["e2e5"]))[0] == "illegal"


def test_parse_uci_square_with_promotion():
    tok = get_tokenizer("uci-square")
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert parse_partial(tok, board, ids(tok, ["a7"])) == ("more", None)
    assert parse_partial(tok, board, ids(tok, ["a7", "a8"])) == ("more", None)  # needs the piece
    assert parse_partial(tok, board, ids(tok, ["a7", "a8", "n"])) == ("move", chess.Move.from_uci("a7a8n"))
    assert parse_partial(tok, board, ids(tok, ["a1", "a2"])) == ("move", chess.Move.from_uci("a1a2"))
    assert parse_partial(tok, board, ids(tok, ["a7", "a8", "e4"]))[0] == "illegal"
    assert parse_partial(tok, board, ids(tok, ["q"]))[0] == "illegal"


@pytest.mark.parametrize("name", ["pgn-char", "uci-move", "uci-square"])
def test_choose_moves_with_untrained_model(name):
    """An untrained model mostly fails, but every move it returns must be legal."""
    tok = get_tokenizer(name)
    player = ModelPlayer.__new__(ModelPlayer)
    player.tokenizer, player.block_size, player.device = tok, 256, "cpu"
    player.model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=256, n_layer=1, n_head=1, n_embd=16)).eval()
    boards = [chess.Board(), board_after("e4 e5 Nf3")]
    choices = player.choose_moves(boards, max_attempts=3)
    for board, c in zip(boards, choices):
        assert c.status in ("legal", "illegal")
        if c.move is not None:
            assert board.is_legal(c.move)


def test_accuracy_formulas():
    assert win_percent(0) == pytest.approx(50)
    assert move_accuracy(100, 100) == pytest.approx(100, abs=0.01)
    assert move_accuracy(0, -500) < 30
    # Evals are White-POV: Black (ply 1) keeps the balance, then White (ply 2) blunders 600cp.
    scores = score_game([0, 0, 0, -600])
    assert scores[1].accuracy == pytest.approx(100, abs=0.01) and scores[1].cp_loss == 0
    assert scores[2].accuracy < 20 and scores[2].cp_loss == 600
