---
license: mit
library_name: pytorch
pipeline_tag: text-generation
tags:
  - chess
  - gpt
  - lichess
datasets:
  - lichess/standard-chess-games
---

# chess-gpt-{tokenizer} ({letter})

A {params}M-parameter GPT that plays chess by predicting the next move token. There is no
search and no evaluation function: one forward pass per move. It is model {letter} of three
models trained the same way and differing only in their tokenizer:

| Model | Tokenizer | A token is | Vocab |
|---|---|---|---|
| [A](https://huggingface.co/{ns}/chess-gpt-pgn-char) | `pgn-char` | one character of `1.e4 e5 2.Nf3` | 34 |
| [B](https://huggingface.co/{ns}/chess-gpt-uci-move) | `uci-move` | one move, `g1f3` | 1971 |
| [C](https://huggingface.co/{ns}/chess-gpt-uci-square) | `uci-square` | one square (+ promotion piece), `g1` `f3` | 71 |

Write-up: [blog.wezn.dev/chess-gpt](https://blog.wezn.dev/chess-gpt/) · Code: [github.com/wezn/chess-gpt](https://github.com/wezn/chess-gpt)

## This model

- Architecture: decoder-only transformer, {n_layer} layers, {n_head} heads, width {n_embd}, context {block_size} tokens, tied embeddings
- Training data: {games_m}M Lichess games (rapid and slower, all ratings), each seen once
- Training: {step:,} steps on one NVIDIA GB10
- Validation loss per move: {val_loss:.3f}

Mean over the final 11 snapshots, from the write-up:

| | A | B | C |
|---|---|---|---|
| Elo (Lichess scale, estimated from self-play scored by Stockfish) | {elo_A} | {elo_B} | {elo_C} |
| Legal moves on held-out human games | {legal_A}% | {legal_B}% | {legal_C}% |
| Average centipawn loss | {acpl_A} | {acpl_B} | {acpl_C} |

The Elo figures are estimates, not results from rated play. The write-up describes how they
were measured.

## Usage

```bash
git clone https://github.com/wezn/chess-gpt && cd chess-gpt && uv sync
uv run hf download {ns}/chess-gpt-{tokenizer} --local-dir models/chess-gpt-{tokenizer}
```

```python
import chess
from chessgpt.eval.player import ModelPlayer

player = ModelPlayer("models/chess-gpt-{tokenizer}", device="cpu")
board = chess.Board()
for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]:
    board.push_san(san)
print(player.choose_moves([board])[0].move)                         # the model's move
print([(board.san(m), round(p, 3)) for m, p in player.rank_legal_moves(board)[:3]])
```

`config.json` holds the architecture and tokenizer. `model.safetensors` holds the weights only.
