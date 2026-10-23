"""Helpers shared by all dataset builders: config loading and immutable, sharded Parquet output."""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class DatasetWriter:
    """Writes rows to data/processed/<name>/shard-NNNNN.parquet plus a meta.json.

    Datasets are immutable: writing into an existing dataset fails. Output goes to
    <name>.partial/ and is renamed only on success, so a crashed build never looks complete.
    """

    def __init__(self, name: str, schema: pa.Schema, rows_per_shard: int = 1_000_000, batch_size: int = 50_000):
        self.final_dir = PROCESSED_DIR / name
        if self.final_dir.exists():
            raise FileExistsError(
                f"{self.final_dir} already exists. Datasets are immutable: pick a new name or delete it."
            )
        self.dir = PROCESSED_DIR / f"{name}.partial"
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.batch_size = batch_size
        self.batch: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.shard = 0
        self.rows_in_shard = 0
        self.rows = 0

    def write(self, row: dict) -> None:
        self.batch.append(row)
        if len(self.batch) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self.batch:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.dir / f"shard-{self.shard:05d}.parquet", self.schema)
        self.writer.write_table(pa.Table.from_pylist(self.batch, schema=self.schema))
        self.rows += len(self.batch)
        self.rows_in_shard += len(self.batch)
        self.batch = []
        if self.rows_in_shard >= self.rows_per_shard:
            self.writer.close()
            self.writer = None
            self.shard += 1
            self.rows_in_shard = 0

    def finish(self, meta: dict) -> Path:
        self._flush()
        if self.writer is not None:
            self.writer.close()
        meta = {
            "rows": self.rows,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            **meta,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        self.dir.rename(self.final_dir)
        return self.final_dir
