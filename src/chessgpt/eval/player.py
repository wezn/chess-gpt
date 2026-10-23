"""Let a trained model choose moves, whatever its tokenizer.

The model writes its move token by token. Each tokenizer has its own rule for
when a move is complete (pgn-char: a space; uci-square: two squares, plus a piece
on promotion; uci-move: one token), and the result is checked for legality.

The first attempt is greedy (the model's single most likely continuation). If it
is illegal, later attempts sample from the model's distribution, so a retry can
produce something different.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from chessgpt.data.pretokenize import TOKENIZED_DIR
from chessgpt.model import GPT, GPTConfig
from chessgpt.tokenizers import BOS, PAD, PROMOTIONS, SPECIALS, Tokenizer, get_tokenizer

MAX_NEW_TOKENS = {"pgn-char": 9, "uci-square": 3, "uci-move": 1}  # longest SAN is 7 chars + space


@dataclass
class MoveChoice:
    move: chess.Move | None  # None if no legal move was produced
    status: str  # "legal", "illegal" (every attempt failed) or "context" (game longer than the model's context)
    illegal_attempts: int  # attempts that failed before success (0 = first try was legal)
    first_output: str  # what the model wrote on its first attempt, for inspecting mistakes


def san_history(board: chess.Board) -> list[str]:
    replay = chess.Board()
    sans = []
    for move in board.move_stack:
        sans.append(replay.san(move))
        replay.push(move)
    return sans


def prompt_ids(tokenizer: Tokenizer, board: chess.Board) -> list[int]:
    """Tokens for the game so far, ending where the model should start writing its move."""
    sans = san_history(board)
    ids = [BOS, *tokenizer.encode_moves(sans)]
    if tokenizer.name == "pgn-char":
        # The model writes only the move itself; separators and move numbers are given.
        suffix = (" " if sans else "") + (f"{board.fullmove_number}." if board.turn == chess.WHITE else "")
        ids += [tokenizer.stoi[c] for c in suffix]
    return ids


def parse_partial(tokenizer: Tokenizer, board: chess.Board, generated: list[int]) -> tuple[str, chess.Move | None]:
    """Decide whether the tokens generated so far form a move: ("more" | "move" | "illegal", move)."""
    if generated[-1] < len(SPECIALS):
        return "illegal", None
    tokens = [tokenizer.itos[i] for i in generated]
    if tokenizer.name == "uci-move":
        move = chess.Move.from_uci(tokens[0])
        return ("move", move) if board.is_legal(move) else ("illegal", None)
    if tokenizer.name == "uci-square":
        if tokens[0] in PROMOTIONS or (len(tokens) >= 2 and tokens[1] in PROMOTIONS):
            return "illegal", None
        if len(tokens) == 1:
            return "more", None
        frm, to = chess.parse_square(tokens[0]), chess.parse_square(tokens[1])
        candidates = [m for m in board.legal_moves if m.from_square == frm and m.to_square == to]
        if not candidates:
            return "illegal", None
        if len(tokens) == 2:
            return ("more", None) if candidates[0].promotion else ("move", candidates[0])
        if tokens[2] not in PROMOTIONS:
            return "illegal", None
        move = chess.Move(frm, to, promotion=chess.Piece.from_symbol(tokens[2]).piece_type)
        return ("move", move) if move in candidates else ("illegal", None)
    # pgn-char: a move ends with a space
    if tokens[-1] != " ":
        return ("illegal", None) if len(tokens) >= MAX_NEW_TOKENS["pgn-char"] else ("more", None)
    try:
        return "move", board.parse_san("".join(tokens[:-1]))
    except ValueError:
        return "illegal", None


class ModelPlayer:
    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        meta = json.loads((TOKENIZED_DIR / cfg["data"] / "meta.json").read_text())
        self.tokenizer = get_tokenizer(meta["tokenizer"])
        self.block_size = cfg["block_size"]
        model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
        self.model = GPT(GPTConfig(vocab_size=meta["vocab_size"], block_size=self.block_size, **model_cfg))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        self.device = device
        self.step, self.progress = ckpt["step"], ckpt.get("progress")
        self.name = cfg["name"]

    @torch.no_grad()
    def _next_tokens(self, seqs: list[list[int]], sample: list[bool], batch_size: int = 256) -> list[int]:
        out = []
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i : i + batch_size]
            lengths = torch.tensor([len(s) for s in chunk], device=self.device)
            x = torch.full((len(chunk), int(lengths.max())), PAD, dtype=torch.long)
            for r, s in enumerate(chunk):
                x[r, : len(s)] = torch.tensor(s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(x.to(self.device))  # right padding is harmless with a causal mask
            last = logits[torch.arange(len(chunk), device=self.device), lengths - 1].float()
            greedy = last.argmax(-1)
            sampled = torch.multinomial(torch.softmax(last, -1), 1).squeeze(-1)
            use_sample = torch.tensor(sample[i : i + batch_size], device=self.device)
            out += torch.where(use_sample, sampled, greedy).tolist()
        return out

    def choose_moves(self, boards: list[chess.Board], max_attempts: int = 4) -> list[MoveChoice]:
        """Choose a move in each position (batched). Boards must start from the standard initial position."""
        prompts = [prompt_ids(self.tokenizer, b) for b in boards]
        generated: list[list[int]] = [[] for _ in boards]
        attempts = [0] * len(boards)
        first_output = [""] * len(boards)
        results: list[MoveChoice | None] = [None] * len(boards)
        max_new = MAX_NEW_TOKENS[self.tokenizer.name]
        for i, p in enumerate(prompts):
            if len(p) + max_new > self.block_size + 1:
                results[i] = MoveChoice(None, "context", 0, "")

        while active := [i for i in range(len(boards)) if results[i] is None]:
            tokens = self._next_tokens([prompts[i] + generated[i] for i in active], [attempts[i] > 0 for i in active])
            for i, tok in zip(active, tokens):
                generated[i].append(tok)
                status, move = parse_partial(self.tokenizer, boards[i], generated[i])
                if status == "more":
                    continue
                if attempts[i] == 0:
                    first_output[i] = "".join(self.tokenizer.itos[t] for t in generated[i])
                if status == "move":
                    results[i] = MoveChoice(move, "legal", attempts[i], first_output[i])
                    continue
                attempts[i] += 1
                generated[i] = []
                if attempts[i] >= max_attempts:
                    results[i] = MoveChoice(None, "illegal", attempts[i], first_output[i])
        return results

    def move_tokens(self, board: chess.Board, move: chess.Move) -> list[int]:
        """The tokens the model would write for `move` in this position."""
        tok = self.tokenizer
        if tok.name == "uci-move":
            return [tok.stoi[move.uci()]]
        if tok.name == "uci-square":
            uci = move.uci()
            return [tok.stoi[uci[:2]], tok.stoi[uci[2:4]]] + ([tok.stoi[uci[4]]] if len(uci) == 5 else [])
        return [tok.stoi[c] for c in board.san(move) + " "]

    @torch.no_grad()
    def rank_legal_moves(self, board: chess.Board) -> list[tuple[chess.Move, float]] | None:
        """Every legal move with the model's probability for it, best first (None if the game is past the context).

        A move's probability is the product of its tokens' probabilities, so multi-token
        moves (pgn-char, uci-square) are scored fairly. Probabilities are renormalised over
        the legal moves.
        """
        prompt = prompt_ids(self.tokenizer, board)
        moves = list(board.legal_moves)
        seqs = [prompt + self.move_tokens(board, m) for m in moves]
        if not moves or max(map(len, seqs)) > self.block_size:
            return None
        x = torch.full((len(seqs), max(map(len, seqs))), PAD, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, : len(s)] = torch.tensor(s)
        x = x.to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logp = torch.log_softmax(self.model(x).float(), -1)
        scores = []
        for r, s in enumerate(seqs):
            positions = torch.arange(len(prompt) - 1, len(s) - 1, device=self.device)
            scores.append(logp[r, positions, x[r, positions + 1]].sum())
        probs = torch.softmax(torch.stack(scores), 0).tolist()
        return sorted(zip(moves, probs), key=lambda mp: -mp[1])
