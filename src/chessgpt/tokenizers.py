"""Tokenizers: turn a game (a list of SAN moves) into token ids and back.

All tokenizers share the same special tokens, so every encoded game looks like
    <bos> move tokens ... <eos>

    pgn-char    one token per character of "1.e4 e5 2.Nf3 ..."     (~5 tokens/move)
    uci-move    one token per move, e.g. "g1f3"                     (1 token/move)
    uci-square  one token per square (+ promotion piece), "g1" "f3" (2 tokens/move)
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import chess

from chessgpt.data.pgn import MOVE_NUMBER_RE

SPECIALS = ["<pad>", "<bos>", "<eos>"]
PAD, BOS, EOS = 0, 1, 2
PROMOTIONS = ["q", "r", "b", "n"]


def san_to_uci(san_moves: Sequence[str]) -> list[str]:
    """Replay a game to convert its SAN moves to UCI (raises ValueError on an illegal move)."""
    board = chess.Board()
    uci = []
    for san in san_moves:
        move = board.parse_san(san)
        uci.append(move.uci())
        board.push(move)
    return uci


class Tokenizer(ABC):
    name: str
    notation: str  # notation of the moves returned by decode(): "san" or "uci"

    def __init__(self, tokens: list[str]):
        self.itos = SPECIALS + tokens
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, san_moves: Sequence[str]) -> list[int]:
        """Encode a whole game as <bos> ... <eos>."""
        return [BOS, *self.encode_moves(san_moves), EOS]

    @abstractmethod
    def encode_moves(self, san_moves: Sequence[str]) -> list[int]:
        """Encode moves from the start of the game, without special tokens."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> list[str]:
        """Decode ids to moves in `self.notation`, skipping special tokens. Malformed output is kept as-is."""

    def _tokens(self, ids: Sequence[int]) -> list[str]:
        return [self.itos[i] for i in ids if i >= len(SPECIALS)]


class PgnCharTokenizer(Tokenizer):
    """Characters of PGN movetext with move numbers: "1.e4 e5 2.Nf3 Nc6"."""

    name = "pgn-char"
    notation = "san"

    def __init__(self):
        super().__init__(list(" .0123456789abcdefghKQRBNOx+#=-"))

    def encode_moves(self, san_moves):
        text = " ".join(f"{i // 2 + 1}.{san}" if i % 2 == 0 else san for i, san in enumerate(san_moves))
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        text = "".join(self._tokens(ids))
        return [m for m in (MOVE_NUMBER_RE.sub("", tok) for tok in text.split()) if m]


def all_uci_moves() -> list[str]:
    """Every move that can ever be legal: queen and knight moves between squares, plus promotions (1968)."""
    moves = []
    for a in chess.SQUARES:
        for b in chess.SQUARES:
            df = abs(chess.square_file(a) - chess.square_file(b))
            dr = abs(chess.square_rank(a) - chess.square_rank(b))
            if a != b and (df == 0 or dr == 0 or df == dr or {df, dr} == {1, 2}):
                moves.append(chess.square_name(a) + chess.square_name(b))
    for from_rank, to_rank in [(6, 7), (1, 0)]:  # white, black
        for file in range(8):
            for to_file in range(max(file - 1, 0), min(file + 1, 7) + 1):
                for piece in PROMOTIONS:
                    moves.append(
                        chess.square_name(chess.square(file, from_rank))
                        + chess.square_name(chess.square(to_file, to_rank))
                        + piece
                    )
    return moves


class UciMoveTokenizer(Tokenizer):
    """One token per UCI move: "e2e4" "e7e5" "g1f3"."""

    name = "uci-move"
    notation = "uci"

    def __init__(self):
        super().__init__(all_uci_moves())

    def encode_moves(self, san_moves):
        return [self.stoi[uci] for uci in san_to_uci(san_moves)]

    def decode(self, ids):
        return self._tokens(ids)


class UciSquareTokenizer(Tokenizer):
    """From-square, to-square and optional promotion piece: "e2" "e4" ... "e7" "e8" "q"."""

    name = "uci-square"
    notation = "uci"

    def __init__(self):
        super().__init__(list(chess.SQUARE_NAMES) + PROMOTIONS)

    def encode_moves(self, san_moves):
        ids = []
        for uci in san_to_uci(san_moves):
            ids += [self.stoi[uci[:2]], self.stoi[uci[2:4]]]
            if len(uci) == 5:
                ids.append(self.stoi[uci[4]])
        return ids

    def decode(self, ids):
        moves: list[str] = []
        pending = None
        for tok in self._tokens(ids):
            if tok in PROMOTIONS and pending is None and moves:
                moves[-1] += tok
            elif pending is None:
                pending = tok
            else:
                moves.append(pending + tok)
                pending = None
        if pending is not None:
            moves.append(pending)
        return moves


TOKENIZERS: dict[str, type[Tokenizer]] = {
    t.name: t for t in (PgnCharTokenizer, UciMoveTokenizer, UciSquareTokenizer)
}


def get_tokenizer(name: str) -> Tokenizer:
    if name not in TOKENIZERS:
        raise KeyError(f"unknown tokenizer {name!r}, choose from {sorted(TOKENIZERS)}")
    return TOKENIZERS[name]()
