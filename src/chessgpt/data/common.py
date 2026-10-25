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


class ChunkWriter:
    """Writes a dataset as independent chunks that become readable as soon as each is closed.

    data/processed/<name>/chunk-NNNN.parquet appears atomically (written as .partial, then
    renamed) every `rows_per_chunk` rows, and status.json is refreshed with every chunk, so
    a tokenizer and a trainer can consume the dataset while it is still being pulled.
    `finish(meta)` writes meta.json and marks the status done.

    Reopening an unfinished dataset resumes it: chunks of months listed as done are kept,
    chunks of the month that was in progress are deleted and that month starts over.
    """

    def __init__(self, name: str, schema: pa.Schema, rows_per_chunk: int = 250_000, batch_size: int = 50_000):
        self.dir = PROCESSED_DIR / name
        self.schema = schema
        self.rows_per_chunk, self.batch_size = rows_per_chunk, batch_size
        self.batch: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.rows_in_chunk = 0
        self.extra: dict = {}
        if (self.dir / "meta.json").exists():
            raise FileExistsError(f"{self.dir} is complete. Datasets are immutable: pick a new name or delete it.")
        self.dir.mkdir(parents=True, exist_ok=True)
        status_path = self.dir / "status.json"
        self.status = json.loads(status_path.read_text()) if status_path.exists() else {"rows": 0, "chunks": 0, "months_done": [], "month_chunks": {}}
        for p in self.dir.glob("*.partial"):
            p.unlink()
        unfinished = [m for m in self.status["month_chunks"] if m not in self.status["months_done"]]
        for m in unfinished:  # the month that was being written: start it over
            first, _ = self.status["month_chunks"].pop(m)
            for k in range(first, self.status["chunks"] + 1):
                (self.dir / f"chunk-{k:04d}.parquet").unlink(missing_ok=True)
            self.status["chunks"] = first - 1
            self.status["rows"] = sum(pq.ParquetFile(self.dir / f"chunk-{k:04d}.parquet").metadata.num_rows for k in range(1, first))
        self.chunk = self.status["chunks"]

    @property
    def months_done(self) -> list[str]:
        return list(self.status["months_done"])

    def begin_month(self, month: str) -> None:
        self.status["month_chunks"][month] = [self.chunk + 1, None]
        self.status["month"] = month

    def end_month(self, month: str) -> None:
        self._close_chunk()
        self.status["month_chunks"][month][1] = self.chunk
        self.status["months_done"].append(month)
        self.write_status()

    def write(self, row: dict) -> None:
        self.batch.append(row)
        if len(self.batch) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self.batch:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.dir / f"chunk-{self.chunk + 1:04d}.parquet.partial", self.schema)
        self.writer.write_table(pa.Table.from_pylist(self.batch, schema=self.schema))
        self.rows_in_chunk += len(self.batch)
        self.status["rows"] += len(self.batch)
        self.batch = []
        if self.rows_in_chunk >= self.rows_per_chunk:
            self._close_chunk()

    def _close_chunk(self) -> None:
        self._flush_pending()
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        self.chunk += 1
        (self.dir / f"chunk-{self.chunk:04d}.parquet.partial").rename(self.dir / f"chunk-{self.chunk:04d}.parquet")
        self.status["chunks"] = self.chunk
        self.rows_in_chunk = 0
        self.write_status()

    def _flush_pending(self) -> None:
        if self.batch:
            if self.writer is None:
                self.writer = pq.ParquetWriter(self.dir / f"chunk-{self.chunk + 1:04d}.parquet.partial", self.schema)
            self.writer.write_table(pa.Table.from_pylist(self.batch, schema=self.schema))
            self.rows_in_chunk += len(self.batch)
            self.status["rows"] += len(self.batch)
            self.batch = []

    def write_status(self, **extra) -> None:
        self.extra.update(extra)
        self.status.update(self.extra, updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=2) + "\n")
        tmp.replace(self.dir / "status.json")

    def finish(self, meta: dict) -> Path:
        self._close_chunk()
        self.status["done"] = True
        self.write_status()
        meta = {"rows": self.status["rows"], "chunks": self.chunk, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(), **meta}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return self.dir
