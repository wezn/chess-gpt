# Chess GPT

Educational project: train small GPT models on chess move sequences and compare
how the training data changes how they play.

## Datasets compared

| # | Dataset | Teaches | Status |
|---|---------|---------|--------|
| 1 | Lichess human games (`configs/data/lichess-*.yaml`) | realistic, varied play | ✅ builder ready |
| 2 | Lichess puzzles + solutions (`configs/data/puzzles.yaml`) | tactics | planned |
| 3 | Engine vs engine games (`configs/data/engine.yaml`) | strong play | planned |

For a fair comparison, every model uses the same tokenizer, the same model size,
the same number of training tokens and the same eval suite.

## Layout

```
configs/data/         how each dataset is built (one file = one immutable dataset)
configs/model/        model sizes
configs/experiments/  dataset + model + training hyperparameters
src/chessgpt/data/    dataset builders and pre-tokenization
src/chessgpt/tokenizers.py  pgn-char, uci-move, uci-square
tests/
notebooks/            analysis only
results/              small summaries and plots (committed)
data/                 datasets and tokenized datasets (gitignored, regenerable)
runs/                 checkpoints and metrics (gitignored)
```

## Where things live

- **Git:** code, configs, small results.
- **Local disk:** `data/` and `runs/`. Raw Lichess dumps are streamed, never stored.
- **Hugging Face Hub:** checkpoints (and datasets) worth keeping.

## Quickstart

```bash
uv sync
uv run pytest
uv run python -m chessgpt.data.lichess configs/data/lichess-smoke.yaml

# full dataset 1: 5M rapid+ games, all ratings
# optional: --max-mbps N caps download bandwidth (megabits/s)
uv run python -m chessgpt.data.lichess configs/data/lichess-2026-09-rapid.yaml

# tokenize a dataset once per tokenizer -> data/tokenized/<dataset>__<tokenizer>/
uv run python -m chessgpt.data.pretokenize lichess-2026-09-rapid uci-move
```

## Tokenizers

| Tokenizer | Token | Vocab | Tokens/move |
|-----------|-------|-------|-------------|
| `pgn-char` (A) | one character of `1.e4 e5 2.Nf3` | 34 | ~5.5 |
| `uci-move` (B) | one move, `g1f3` | 1971 | 1 |
| `uci-square` (C) | one square (+ promotion piece), `g1` `f3` | 71 | ~2 |

All share `<pad>`, `<bos>`, `<eos>`. The validation split (1%) is picked by hashing
game IDs, so it is the same games for every tokenizer.

See `TODO.md` for planned work.

PyTorch is installed from the CUDA 13.0 wheel index (see `pyproject.toml`) for the GB10 GPU.

Each dataset is written to `data/processed/<name>/` as Parquet shards plus a
`meta.json` recording the config, filter statistics and git commit.
