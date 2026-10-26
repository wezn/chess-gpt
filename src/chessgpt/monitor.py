"""Live terminal dashboard: system, data pipeline, every model on one screen.

    uv run chess-monitor                 # active runs (anything that moved in the last 2 hours)
    uv run chess-monitor --all           # every run
    uv run chess-monitor runs/large-uci-move/20261023-133700 [more runs...]
    uv run chess-monitor --chart loss    # the shared chart: elo (default), loss or legal

Layout, top to bottom:
    system     GPU utilisation / temperature / power, CPU utilisation / temperatures, RAM, disk
    pipeline   puller, tokenizer follower, round-robin trainer, snapshot watchers
    models     one row per run: status, step, games, losses, Elo with interval, what changed
               recently (Δ over the last 5 snapshots), legality, evaluation in progress
    chart      Elo (or loss / legality) of every model over snapshot number, so models are
               compared at equal numbers of games seen whatever their step counts

Reads <run>/metrics.jsonl, <run>/eval.jsonl, <run>/eval-status.json, runs/<group>/status.json
and data/processed/<dataset>/status.json. Elo is computed here from the stored per-game
statistics with the current calibration, so recalibrating updates the whole history.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from chessgpt.eval.elo import EloCalibration

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
SPARK = "▁▂▃▄▅▆▇█"
MARKERS = ["●", "▲", "■", "◆", "★", "✚", "▼", "○"]
COLOURS = ["bright_blue", "dark_orange", "green3", "magenta", "cyan", "yellow", "red", "white"]


# ----------------------------------------------------------------------------- helpers

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written right now
            pass
    return out


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def sparkline(values: list[float], width: int = 24) -> str:
    values = values[-width:]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    return "".join(SPARK[int((v - lo) / (hi - lo + 1e-12) * (len(SPARK) - 1))] for v in values)


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds >= 86400:
        return f"{seconds // 86400}d{seconds % 86400 // 3600:02d}h"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def fmt_games(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else f"{n:.0f}"


def delta(value: float | None, unit: str = "", invert: bool = False, fmt: str = "{:+.0f}") -> Text:
    """A coloured ▲/▼ change; `invert` when going down is good (loss)."""
    if value is None:
        return Text("—", style="dim")
    good = value < 0 if invert else value > 0
    arrow = "▲" if value > 0 else "▼" if value < 0 else "="
    return Text(f"{arrow}{fmt.format(value)}{unit}", style="green" if good and value else "red" if value else "dim")


def load_calibration() -> EloCalibration | None:
    try:
        return EloCalibration.load()
    except Exception:  # missing, or being refitted
        return None


def temperature_c() -> int:
    return 0


# ----------------------------------------------------------------------------- system

class SystemStats:
    """GPU and CPU load/temperatures; CPU utilisation needs two samples, so it keeps the previous one."""

    def __init__(self):
        self.prev_cpu = self._cpu_times()

    @staticmethod
    def _cpu_times() -> tuple[int, int]:
        parts = open("/proc/stat").readline().split()[1:]
        vals = list(map(int, parts))
        return sum(vals), vals[3] + vals[4]  # total, idle+iowait

    def cpu_percent(self) -> float:
        total, idle = self._cpu_times()
        dt, di = total - self.prev_cpu[0], idle - self.prev_cpu[1]
        self.prev_cpu = (total, idle)
        return 100 * (1 - di / dt) if dt > 0 else 0.0

    @staticmethod
    def cpu_temps() -> list[int]:
        temps = []
        for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                temps.append(int(z.read_text()) // 1000)
            except (OSError, ValueError):
                pass
        return temps

    @staticmethod
    def gpu() -> dict:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip().split(", ")
            return {"util": float(out[0]), "temp": float(out[1]), "power": float(out[2]), "clock": float(out[3])}
        except Exception:
            return {}

    def render(self) -> Text:
        g = self.gpu()
        cpu = self.cpu_percent()
        temps = self.cpu_temps()
        load = os.getloadavg()[0]
        mem = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            mem[k] = int(v.split()[0]) // 1024 // 1024  # GB
        disk = shutil.disk_usage(REPO_ROOT).free / 1e12
        t = Text()
        if g:
            t.append("GPU ", style="bold").append(f"{g['util']:3.0f}%  ").append(f"{g['temp']:.0f}°C", style=heat(g["temp"], 75, 85)).append(f"  {g['power']:.0f} W  {g['clock']:.0f} MHz")
        else:
            t.append("GPU n/a", style="dim")
        t.append("   │   ")
        t.append("CPU ", style="bold").append(f"{cpu:3.0f}%  load {load:.1f}/{os.cpu_count()}  ")
        if temps:
            t.append(f"{min(temps)}–{max(temps)}°C", style=heat(max(temps), 85, 95))
        t.append("   │   ").append("RAM ", style="bold").append(f"{mem.get('MemTotal', 0) - mem.get('MemAvailable', 0)}/{mem.get('MemTotal', 0)} GB")
        t.append("   │   ").append("disk ", style="bold").append(f"{disk:.1f} TB free")
        t.append("   │   ").append(datetime.now().strftime("%H:%M:%S"), style="dim")
        return t


def heat(value: float, warn: float, hot: float) -> str:
    return "red bold" if value >= hot else "yellow" if value >= warn else "green"


# ----------------------------------------------------------------------------- pipeline

def alive(pattern: str) -> int:
    try:
        return int(subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True).stdout.strip() or 0)
    except Exception:
        return 0


def dot(on: bool) -> Text:
    return Text("● ", style="green" if on else "red")


def pipeline_panel(groups: dict[str, dict]) -> Panel | None:
    parts: list[Text] = []
    for status_path in sorted((REPO_ROOT / "data" / "processed").glob("*/status.json")):
        st = read_json(status_path)
        if not st:
            continue
        name = status_path.parent.name
        age = time.time() - datetime.fromisoformat(st["updated"]).timestamp() if "updated" in st else 1e9
        running = not st.get("done") and age < 300
        t = dot(running or st.get("done", False))
        t.append(f"pull {name}: ", style="bold")
        t.append("done, " if st.get("done") else f"{st.get('month', '?')}, ")
        t.append(f"{st.get('chunks', 0)} chunks · {fmt_games(st.get('rows', 0))} games")
        if running:
            t.append(f" · {st.get('kept_rate', 0):,}/s")
        if st.get("months_done"):
            t.append(f" · months done {', '.join(st['months_done'])}", style="dim")
        parts.append(t)
        toks = []
        for tdir in sorted((REPO_ROOT / "data" / "tokenized").glob(f"{name}__*")):
            ts = read_json(tdir / "status.json")
            if ts:
                toks.append(f"{tdir.name.split('__')[1]} {ts['chunks']}/{st.get('chunks', 0)}")
        if toks:
            t2 = dot(alive("bin/python[0-9.]* -m chessgpt[.]data[.]pretokenize.*--follow") > 0 or st.get("done", False))
            t2.append("tokenizer: ", style="bold").append("  ".join(toks))
            parts.append(t2)
    for name, g in groups.items():
        fresh = time.time() - g["updated"] < 300
        t = dot(fresh)
        t.append(f"trainer {name}: ", style="bold")
        if all(r.get("done") for r in g["runs"].values()):
            t.append("all models done")
        else:
            now = g["runs"].get(g["now"], {}) if g["now"] else {}
            t.append(f"round {g['round']} · now {g['now'] or '—'}" + (f" · {now['games_per_s']:,.0f} games/s" if now.get("games_per_s") else "") + ("  (waiting for data)" if now.get("waiting_for_data") else ""))
            if (RUNS_DIR / name / "STOP").exists():
                t.append("  STOP requested, cooling down", style="magenta")
        parts.append(t)
    watchers, engines = alive("bin/python[0-9.]* -m chessgpt[.]eval[.]watch"), alive("^[^ ]*/stockfish$")
    if watchers or engines:
        t = dot(watchers > 0)
        t.append("watchers: ", style="bold").append(f"{watchers} running · {engines} Stockfish engines")
        parts.append(t)
    if not parts:
        return None
    return Panel(Group(*parts), title="[bold]pipeline[/]", title_align="left", padding=(0, 1))


# ----------------------------------------------------------------------------- runs

class RunInfo:
    """Everything the table and the chart need about one run, read fresh on each refresh (Elo cached per snapshot)."""

    elo_cache: dict[tuple[str, int], dict] = {}

    def __init__(self, run: Path, calibration: EloCalibration | None, groups: dict[str, dict]):
        self.run = run
        self.name = run.parent.name
        metrics = read_jsonl(run / "metrics.jsonl")
        self.train = [m for m in metrics if m["type"] == "train"]
        self.val = [m for m in metrics if m["type"] == "eval"]
        starts = [m for m in metrics if m["type"] == "start"]
        self.total = starts[-1]["total_steps"] if starts else None
        self.evals = sorted(read_jsonl(run / "eval.jsonl"), key=lambda r: r["step"])
        self.snapshots = len(list((run / "snapshots").glob("step-*.pt")))
        self.mtime = max((p.stat().st_mtime for p in (run / "metrics.jsonl", run / "eval.jsonl", run / "eval-status.json") if p.exists()), default=0)
        self.group, self.me = None, None
        gy = run / "group.yaml"
        if gy.exists():
            gname = yaml.safe_load(gy.read_text())["name"]
            self.group = groups.get(gname)
            if self.group:
                self.me = next((v for v in self.group["runs"].values() if Path(v["run_dir"]).resolve() == run.resolve()), None)
        self.elos: list[dict] = []
        if calibration:
            for r in self.evals:
                key = (str(run), r["step"])
                if key not in self.elo_cache:
                    if "strength_per_game" in r:
                        self.elo_cache[key] = calibration.estimate_ci(r["strength_per_game"])
                    else:
                        e = calibration.estimate({"accuracy": r["selfplay_accuracy"], "acpl": r["selfplay_acpl"], "win_loss": r["selfplay_win_loss"]})
                        self.elo_cache[key] = {**e, "elo_lo": None, "elo_hi": None}
                self.elos.append(self.elo_cache[key])
        self.status_file = read_json(run / "eval-status.json")

    # --- derived
    @property
    def step(self) -> int:
        return self.train[-1]["step"] if self.train else 0

    @property
    def games(self) -> float | None:
        if self.me:
            return self.me.get("games_seen")
        if self.train and "games_seen" in self.train[-1]:
            return self.train[-1]["games_seen"]
        return None

    def status(self) -> Text:
        age = time.time() - self.mtime
        if self.me:
            fresh = time.time() - self.group["updated"] < 300
            if self.me.get("done"):
                return Text("finished", style="green")
            if not fresh:
                return Text("trainer stopped?", style="yellow")
            me_key = next(k for k, v in self.group["runs"].items() if v is self.me)
            if self.group["now"] == me_key:
                t = Text("▶ training", style="bold cyan")
                if self.me.get("waiting_for_data"):
                    t.append(" (waiting for data)", style="yellow")
            else:
                t = Text("queued", style="blue")
            if self.me.get("cooldown"):
                t.append(" · cooling down", style="magenta")
            return t
        if self.total and self.val and self.val[-1]["step"] >= self.total:
            return Text("finished", style="green")
        if age < 180:
            return Text("▶ training", style="bold cyan")
        return Text("idle", style="dim")

    def rate(self) -> str:
        """Steps per minute and games per second over the current burst of log records, when the run is moving."""
        if len(self.train) < 2 or time.time() - self.mtime > 300:
            return ""
        if self.me and self.group.get("now") and self.group["runs"][self.group["now"]] is not self.me:
            return ""  # queued behind another model: its last records are stale
        recent = [self.train[-1]]
        for rec in reversed(self.train[:-1]):  # walk back while the records are consecutive (no queued gap)
            if recent[-1]["elapsed_s"] - rec["elapsed_s"] > 120 or len(recent) >= 20:
                break
            recent.append(rec)
        recent.reverse()
        dt = recent[-1]["elapsed_s"] - recent[0]["elapsed_s"]
        out = f"{(recent[-1]['step'] - recent[0]['step']) / dt * 60:.0f} st/min" if dt > 0 and len(recent) > 1 else ""
        if "games_per_s" in self.train[-1]:
            out += f" · {self.train[-1]['games_per_s']:,.0f} games/s"
        elif dt > 0 and len(recent) > 1 and "games_seen" in recent[-1]:
            out += f" · {(recent[-1]['games_seen'] - recent[0]['games_seen']) / dt:,.0f} games/s"
        return out

    def eval_now(self) -> Text:
        st = self.status_file
        if st and time.time() - st["updated"] < 600:
            return Text(f"step {st['step']:,} · {st['phase']} ({fmt_duration(time.time() - st['started'])})", style="dim")
        pending = self.snapshots - len(self.evals)
        if pending > 0:
            return Text(f"{pending} snapshot{'s' if pending > 1 else ''} waiting", style="yellow")
        return Text("up to date", style="dim") if self.evals else Text("—", style="dim")


def table_row(table: Table, info: RunInfo, series_index: int) -> None:
    style = COLOURS[series_index % len(COLOURS)]
    name = Text(f"{MARKERS[series_index % len(MARKERS)]} ", style=style) + Text(info.name)
    step = f"{info.step:,}" + (f"/{info.total:,}" if info.total else "")
    games = fmt_games(info.games) if info.games is not None else "—"
    loss = f"{info.train[-1]['loss']:.3f}" if info.train else "—"
    val = f"{info.val[-1]['val_loss_move']:.3f}" if info.val else "—"
    d_val = info.val[-1]["val_loss_move"] - info.val[-2]["val_loss_move"] if len(info.val) >= 2 else None
    if info.elos:
        e = info.elos[-1]
        elo = f"{e['elo_estimate']}" + (f" ±{(e['elo_hi'] - e['elo_lo']) / 2:.0f}" if e.get("elo_lo") is not None else "") + {"below_range": "↓", "above_range": "↑"}.get(e["elo_range"], "")
        d_elo = e["elo_estimate"] - info.elos[-6]["elo_estimate"] if len(info.elos) >= 6 else None
    else:
        elo, d_elo = "—", None
    last = info.evals[-1] if info.evals else {}
    legal = f"{last['legal_rate']:.1%}" if last else "—"
    illegal = f"{last['selfplay_first_try_illegal_rate']:.1%}" if last else "—"
    status = info.status()
    if rate := info.rate():
        status.append(f"  {rate}", style="dim")
    table.add_row(
        name, status, step, games, loss,
        Text(val) + Text("  ") + delta(d_val, invert=True, fmt="{:+.3f}"),
        Text(elo) + Text("  ") + delta(d_elo),
        legal, illegal, f"{len(info.evals)}", info.eval_now(),
    )


def models_panel(infos: list[RunInfo]) -> Panel:
    table = Table(box=None, header_style="bold", pad_edge=False, expand=True, collapse_padding=True)
    for col, justify, ratio in (("model", "left", 3), ("status", "left", 3), ("step", "right", 1), ("games", "right", 1), ("loss", "right", 1),
                                ("val/move  Δ", "right", 2), ("Elo (95% CI)  Δ5", "right", 2),
                                ("legal", "right", 1), ("sp-illegal", "right", 1), ("evals", "right", 1), ("evaluation now", "left", 4)):
        table.add_column(col, justify=justify, no_wrap=True, ratio=ratio, overflow="ellipsis")
    for i, info in enumerate(infos):
        table_row(table, info, i)
    return Panel(table, title="[bold]models[/]  [dim](Δ: val loss vs previous snapshot · Elo vs 5 snapshots ago)[/]", title_align="left", padding=(0, 1))


# ----------------------------------------------------------------------------- chart

def chart_panel(infos: list[RunInfo], kind: str, width: int, height: int) -> Panel:
    """All models' Elo / loss / legality over snapshot number, drawn with one marker per model."""
    series = []
    for i, info in enumerate(infos):
        if kind == "elo":
            ys = [e["elo_estimate"] for e in info.elos]
        elif kind == "loss":
            ys = [r["val_loss_move"] for r in info.evals]
        else:
            ys = [r["legal_rate"] * 100 for r in info.evals]
        if ys:
            series.append((i, ys))
    titles = {"elo": "Elo over snapshots (legal-only self-play)", "loss": "validation loss per move over snapshots", "legal": "legal-move rate on real games, % over snapshots"}
    if not series:
        return Panel(Text("no evaluations yet", style="dim"), title=f"[bold]{titles[kind]}[/]", title_align="left")
    all_y = [y for _, ys in series for y in ys]
    lo, hi = min(all_y), max(all_y)
    if kind == "elo":
        lo, hi = min(lo, 550), max(hi, 1500)
    pad = (hi - lo) * 0.05 or 1
    lo, hi = lo - pad, hi + pad
    n_max = max(len(ys) for _, ys in series)
    H = max(6, height - 3)
    W = max(20, width - 10)
    grid = [[None] * W for _ in range(H)]
    for i, ys in series:
        for k, y in enumerate(ys):
            col = round(k / max(1, n_max - 1) * (W - 1))
            row = round((hi - y) / (hi - lo) * (H - 1))
            grid[row][col] = i
    lines = []
    for r in range(H):
        y_val = hi - (hi - lo) * r / (H - 1)
        label = f"{y_val:6.0f} ┤" if kind != "loss" else f"{y_val:6.2f} ┤"
        line = Text(label, style="dim")
        for c in range(W):
            i = grid[r][c]
            line.append(MARKERS[i % len(MARKERS)], style=COLOURS[i % len(COLOURS)]) if i is not None else line.append(" ")
        lines.append(line)
    axis = Text("       └" + "─" * W, style="dim")
    legend = Text("        snapshot 1" + " " * max(0, W - 24) + f"snapshot {n_max}   ", style="dim")
    for i, _ in series:
        legend.append(f" {MARKERS[i % len(MARKERS)]} ", style=COLOURS[i % len(COLOURS)]).append(infos[i].name, style="dim")
    return Panel(Group(*lines, axis, legend), title=f"[bold]{titles[kind]}[/]  [dim](equal snapshot = equal games seen)[/]", title_align="left", padding=(0, 1))


# ----------------------------------------------------------------------------- main

def find_runs(explicit: list[Path], all_runs: bool) -> list[Path]:
    if explicit:
        return [r.resolve() for r in explicit]
    runs = sorted(RUNS_DIR.glob("*/*/metrics.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = [p.parent for p in runs]
    if all_runs:
        return runs
    active = [r for r in runs if time.time() - max((p.stat().st_mtime for p in (r / "metrics.jsonl", r / "eval.jsonl") if p.exists())) < 7200]
    return active or runs[:3]


def load_groups() -> dict[str, dict]:
    groups = {}
    for p in RUNS_DIR.glob("*/status.json"):
        st = read_json(p)
        if st and "runs" in st:
            groups[p.parent.name] = st
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="show every run, not just active ones")
    parser.add_argument("--chart", choices=["elo", "loss", "legal"], default="elo", help="what the shared chart plots")
    parser.add_argument("--interval", type=float, default=5, help="seconds between refreshes")
    parser.add_argument("--once", action="store_true", help="print once and exit")
    args = parser.parse_args()

    console = Console(width=180 if args.once else None, height=50 if args.once else None)
    system = SystemStats()

    def render():
        groups = load_groups()
        calibration = load_calibration()
        runs = find_runs(args.runs, args.all)
        infos = [RunInfo(r, calibration, groups) for r in runs]
        width, height = console.size
        layout = Layout()
        pipeline = pipeline_panel(groups)
        models_h = len(infos) + 4
        sections = [Layout(Panel(system.render(), title="[bold]system[/]", title_align="left", padding=(0, 1)), name="system", size=3)]
        if pipeline:
            sections.append(Layout(pipeline, name="pipeline", size=pipeline.renderable.renderables.__len__() + 2))
        sections.append(Layout(models_panel(infos), name="models", size=models_h))
        used = sum(s.size for s in sections)
        chart_h = max(9, height - used)
        sections.append(Layout(chart_panel(infos, args.chart, width - 4, chart_h - 2), name="chart"))
        layout.split_column(*sections)
        return layout

    if args.once:
        console.print(render())
        return
    with Live(render(), console=console, screen=True, auto_refresh=False) as live:
        while True:
            time.sleep(args.interval)
            live.update(render(), refresh=True)


if __name__ == "__main__":
    main()
