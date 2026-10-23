import random

import chess
import pytest

from chessgpt.tokenizers import BOS, EOS, TOKENIZERS, all_uci_moves, get_tokenizer, san_to_uci

# A legal (random) game with castling, captures, a queen promotion and a knight underpromotion with check.
GAME = (
    "e4 a6 Qg4 g6 Nc3 e5 a3 b6 Ra2 f5 Qh3 Nf6 exf5 Bh6 a4 O-O Qd3 e4 Nf3 g5 a5 Kf7 Nd1 Ke8 Qb5 e3 dxe3 Rg8 axb6 Bb7 Qxa6 cxb6 h4 Kf8 Bc4 Rg6 Qa5 Ra7 Qa4 gxh4 Qa6 d6 Bd3 Bd5 Bb5 Bc6 Qa3 Be8 Bxe8 Nd5 Qxa7 Rg7 Qa8 Bf4 Nd4 b5 f3 h3 Bh5 Qf6 g4 Rg6 Ra6 Bg3+ Kd2 Be5 fxg6 Qf5 Ra5 Qe6 Rg1 Ke7 Qa6 h2 Rg3 Nb4 Ne2 "
    "Ke8 Ra2 Nxa2 Rg2 Nb4 g5 Bg3 Qc6+ Qd7 f4 d5 Qb7 Qg4 Qxh7 Qc8 b3 Qd7 Kc3 Qe7 Qxe7+ Kxe7 g7 Nxc2 Rxg3 Na3 Rg4 h1=Q b4 Qh3 Nf2 Qh2 Nd4 Qxf4 g8=N+ Kd8 Ne6+ Kd7 Rg2 Qd6 Rg4 Qf4 Nd8 Qd6"
).split()


def random_game(seed: int, max_plies: int = 300) -> list[str]:
    rng = random.Random(seed)
    board = chess.Board()
    sans = []
    while not board.is_game_over() and len(sans) < max_plies:
        move = rng.choice(list(board.legal_moves))
        sans.append(board.san(move))
        board.push(move)
    return sans


@pytest.fixture(params=sorted(TOKENIZERS))
def tok(request):
    return get_tokenizer(request.param)


def expected_moves(tok, sans):
    return sans if tok.notation == "san" else san_to_uci(sans)


def test_roundtrip_known_game(tok):
    ids = tok.encode(GAME)
    assert ids[0] == BOS and ids[-1] == EOS
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == expected_moves(tok, GAME)


def test_roundtrip_random_games(tok):
    # Random games hit odd moves (underpromotions, long king walks) that human games rarely contain.
    for seed in range(50):
        sans = random_game(seed)
        assert tok.decode(tok.encode(sans)) == expected_moves(tok, sans)


def test_vocab_sizes():
    assert len(all_uci_moves()) == len(set(all_uci_moves())) == 1968
    assert get_tokenizer("pgn-char").vocab_size == 34
    assert get_tokenizer("uci-move").vocab_size == 1971
    assert get_tokenizer("uci-square").vocab_size == 71


def test_tokens_per_move():
    n = len(GAME)
    assert len(get_tokenizer("uci-move").encode_moves(GAME)) == n
    assert len(get_tokenizer("uci-square").encode_moves(GAME)) == 2 * n + 2  # 2 promotions
    assert len(get_tokenizer("pgn-char").encode_moves(GAME[:2])) == len("1.e4 a6")


def test_pgn_char_format():
    tok = get_tokenizer("pgn-char")
    assert "".join(tok.itos[i] for i in tok.encode_moves(["e4", "e5", "Nf3"])) == "1.e4 e5 2.Nf3"


def test_illegal_move_raises():
    with pytest.raises(ValueError):
        get_tokenizer("uci-move").encode(["e4", "e4"])


def test_unknown_tokenizer():
    with pytest.raises(KeyError):
        get_tokenizer("nope")
