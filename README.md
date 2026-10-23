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
src/chessgpt/data/    dataset builders
tests/
notebooks/            analysis only
results/              small summaries and plots (committed)
data/                 datasets (gitignored, regenerable from configs)
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
```

PyTorch is installed from the CUDA 13.0 wheel index (see `pyproject.toml`) for the GB10 GPU.

Each dataset is written to `data/processed/<name>/` as Parquet shards plus a
`meta.json` recording the config, filter statistics and git commit.
