# Chess GPT

Three 50M-parameter GPTs trained on the same 35M Lichess games. They differ only in how a
move is tokenized. None of them searches: each move is one forward pass. The two best play at
about 1,900 Elo on the Lichess scale, estimated from self-play scored by Stockfish.

- **Write-up:** [blog.wezn.dev/chess-gpt](https://blog.wezn.dev/chess-gpt/)
- **Models:** [chess-gpt-pgn-char](https://huggingface.co/ftaz/chess-gpt-pgn-char) (A) ·
  [chess-gpt-uci-move](https://huggingface.co/ftaz/chess-gpt-uci-move) (B) ·
  [chess-gpt-uci-square](https://huggingface.co/ftaz/chess-gpt-uci-square) (C)

| Tokenizer | Token | Vocab | Tokens/move |
|-----------|-------|-------|-------------|
| `pgn-char` (A) | one character of `1.e4 e5 2.Nf3` | 34 | ~5.5 |
| `uci-move` (B) | one move, `g1f3` | 1971 | 1 |
| `uci-square` (C) | one square (+ promotion piece), `g1` `f3` | 71 | ~2 |

All share `<pad>`, `<bos>`, `<eos>`. The validation split (1%) is picked by hashing game IDs,
so every tokenizer is validated on the same games.

## Play a trained model

```bash
uv sync
uv run hf download ftaz/chess-gpt-uci-move --local-dir models/chess-gpt-uci-move
```

```python
import chess
from chessgpt.eval.player import ModelPlayer

player = ModelPlayer("models/chess-gpt-uci-move", device="cpu")
board = chess.Board()
print(player.choose_moves([board])[0].move)           # greedy move, retries if illegal
print(player.rank_legal_moves(board)[:3])             # probabilities over the legal moves
```

`uv run chess-play` plays your own training runs in the browser.

PyTorch comes from the CUDA 13.0 wheel index (`pyproject.toml`), which is what the NVIDIA GB10 we
trained on needs. On other hardware, delete the `[tool.uv.sources]` entry for torch to get the
default wheel.

## Train your own

```bash
uv run pytest

# 5M rapid+ games, all ratings -> data/processed/<name>/ (Parquet + meta.json)
# --max-mbps N caps download bandwidth
uv run python -m chessgpt.data.lichess configs/data/lichess-2026-09-rapid.yaml

# tokenize once per tokenizer -> data/tokenized/<dataset>__<tokenizer>/
uv run python -m chessgpt.data.pretokenize lichess-2026-09-rapid uci-square

uv run python -m chessgpt.train configs/experiments/large-uci-square.yaml
uv run python -m chessgpt.eval.watch runs/large-uci-square/<timestamp>   # evaluates every snapshot
uv run chess-monitor --all                                               # live dashboard
```

Experiments are named `<size>-<tokenizer>`; the sizes are in `configs/model/` (tiny, small,
medium, large). Each training row packs several whole games, and an attention mask stops games
from attending to each other, so every game is scored as if it were alone. Validation loss is
reported per move, so it can be compared across tokenizers.

The final models were trained by a streaming pipeline: Lichess monthly dumps are pulled,
filtered and tokenized in 250k-game chunks while A, B and C train round-robin on the same
chunks. `scripts/install-services.sh` installs it as systemd user services:

```bash
uv run python -m chessgpt.data.lichess configs/data/lichess-stream.yaml   # pull
uv run python -m chessgpt.data.pretokenize lichess-stream --follow        # tokenize
scripts/stream-train.sh                                                   # train
uv run python -m chessgpt.eval.watch runs/stream-large-uci-move/<stamp>   # evaluate
```

`scripts/export_hf.py` turns a checkpoint into the Hub format: `config.json` plus
`model.safetensors`, without optimizer state.

## Layout

```
configs/data/         how each dataset is built (one file = one immutable dataset)
configs/model/        model sizes
configs/experiments/  dataset + model + training hyperparameters
src/chessgpt/data/    dataset builders and pre-tokenization
src/chessgpt/tokenizers.py  pgn-char, uci-move, uci-square
src/chessgpt/model.py       the GPT
src/chessgpt/batches.py     packing whole games into training rows
src/chessgpt/train.py       training loop (stream_train.py: the streaming version)
src/chessgpt/eval/          legality, Stockfish move quality, Elo, snapshot watcher
data/, runs/          datasets and checkpoints (gitignored, regenerable)
```

## License

MIT. The training data is the [Lichess database](https://database.lichess.org/) (CC0).
