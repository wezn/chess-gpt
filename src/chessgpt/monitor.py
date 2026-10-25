"""Live terminal dashboard for training runs.

    uv run chess-monitor                                 # the most recently active run
    uv run python -m chessgpt.monitor runs/large-uci-move/20261023-133700 [more runs...]
    uv run python -m chessgpt.monitor --all              # one panel per run, newest first

Reads <run>/metrics.jsonl (training) and <run>/eval.jsonl (snapshot evaluations,
written by `chessgpt.eval.watch`) and refreshes every few seconds. Elo is computed
here from the stored move statistics with the current calibration, so recalibrating
updates the whole history.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from chessgpt.eval.elo import EloCalibration

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
SPARK = "▁▂▃▄▅▆▇█"


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


def sparkline(values: list[float], width: int = 40) -> str:
    values = values[-width:]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    return "".join(SPARK[int((v - lo) / (hi - lo + 1e-12) * (len(SPARK) - 1))] for v in values)


def total_steps(run: Path, metrics: list[dict]) -> int | None:
    start = [m for m in metrics if m["type"] == "start"]
    if start:
        return start[-1]["total_steps"]  # None for streaming runs
    # Older runs: derive it from a snapshot's step and progress.
    snaps = sorted((run / "snapshots").glob("step-*.pt"))
    if snaps:
        ckpt = torch.load(snaps[-1], map_location="cpu", weights_only=False, mmap=True)
        return round(ckpt["step"] / ckpt["progress"]) if ckpt.get("progress") else None
    return None


def group_status(run: Path) -> dict | None:
    """For a run that belongs to a round-robin streaming group: that group's status.json, if fresh."""
    group = run / "group.yaml"
    if not group.exists():
        return None
    name = yaml.safe_load(group.read_text())["name"]
    path = RUNS_DIR / name / "status.json"
    if not path.exists():
        return None
    try:
        st = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    st["fresh"] = time.time() - st["updated"] < 300
    st["mine"] = next((k for k, v in st["runs"].items() if Path(v["run_dir"]).resolve() == run.resolve()), None)
    return st


def stream_panel() -> Panel | None:
    """Puller and tokenizer progress for streaming datasets (data/processed/<name>/status.json)."""
    lines = []
    for status_path in sorted((REPO_ROOT / "data" / "processed").glob("*/status.json")):
        try:
            st = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            continue
        name = status_path.parent.name
        age = time.time() - datetime.fromisoformat(st["updated"]).timestamp() if "updated" in st else 0
        state = "[green]done[/]" if st.get("done") else "[cyan]pulling[/]" if age < 300 else "[yellow]stalled?[/]"
        months = ", ".join(st.get("months_done", [])) or "—"
        lines.append(Text.from_markup(
            f"[bold]{name}[/]  {state}  month {st.get('month', '—')}  chunks {st.get('chunks', 0)}  games {st.get('rows', 0):,}"
            + (f"  {st.get('kept_rate', 0):,} kept/s" if not st.get("done") else "") + f"  months done: {months}"
        ))
        toks = []
        for tdir in sorted((REPO_ROOT / "data" / "tokenized").glob(f"{name}__*")):
            ts = tdir / "status.json"
            if ts.exists():
                try:
                    t = json.loads(ts.read_text())
                    toks.append(f"{tdir.name.split('__')[1]} {t['chunks']}/{st.get('chunks', 0)}")
                except (json.JSONDecodeError, KeyError):
                    pass
        if toks:
            lines.append(Text("tokenized chunks: " + "  ".join(toks), style="dim"))
    return Panel(Group(*lines), title="[bold]data stream[/]", title_align="left") if lines else None


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def load_calibration() -> EloCalibration | None:
    try:
        return EloCalibration.load()
    except Exception:  # missing, or still the old format while recalibrating
        return None


def row_elo(r: dict, calibration: EloCalibration) -> tuple[int, str]:
    """(point estimate, display string) for one eval row; new rows carry per-game sums, old rows pooled stats."""
    if "strength_per_game" in r:
        e = calibration.estimate_ci(r["strength_per_game"])
        text = f"{e['elo_estimate']} ±{(e['elo_hi'] - e['elo_lo']) / 2:.0f}"
    else:
        e = calibration.estimate({"accuracy": r["selfplay_accuracy"], "acpl": r["selfplay_acpl"], "win_loss": r["selfplay_win_loss"]})
        text = f"{e['elo_estimate']} (v1)"
    return e["elo_estimate"], text + {"below_range": "↓", "above_range": "↑"}.get(e["elo_range"], "")


def eval_progress(run: Path, evals: list[dict]) -> Text | None:
    """One line on the snapshot evaluation: done/total, pace, ETA and the snapshot being evaluated now."""
    snaps = sorted((run / "snapshots").glob("step-*.pt"))
    if not snaps:
        return None
    done, total = len(evals), len(snaps)
    out_path = run / "eval.jsonl"
    last_write = out_path.stat().st_mtime if out_path.exists() else 0
    now = ""
    status_path = run / "eval-status.json"
    if status_path.exists():  # written by chessgpt.eval.watch after every step of the current evaluation
        try:
            st = json.loads(status_path.read_text())
            if time.time() - st["updated"] < 600:
                now = f"\n  now: step {st['step']:,} ({fmt_duration(time.time() - st['started'])} in) · {st['phase']}"
            else:
                now = "  [yellow]watcher stalled?[/]"
        except (json.JSONDecodeError, KeyError):
            pass
    elif done < total and time.time() - last_write > 1800:
        now = "  [yellow]no watcher running?[/]"
    pace = ""
    if evals:
        recent = [e["eval_seconds"] for e in read_jsonl(out_path) if e.get("version") == 2][-10:]  # file order = chronological
        per = sum(recent) / len(recent)
        pace = f"  {fmt_duration(per)} per snapshot" + (f"  ETA {fmt_duration(per * (total - done))}" if done < total else "")
    colour = "green" if done >= total else "cyan"
    return Text.from_markup(f"eval [{colour}]{done}/{total}[/] snapshots{pace}{now}")


def run_panel(run: Path, totals: dict, calibration: EloCalibration | None, rows: int) -> Panel:
    metrics = read_jsonl(run / "metrics.jsonl")
    evals = sorted(read_jsonl(run / "eval.jsonl"), key=lambda r: r["step"])
    train = [m for m in metrics if m["type"] == "train"]
    val = [m for m in metrics if m["type"] == "eval"]
    if run not in totals:
        totals[run] = total_steps(run, metrics)
    total = totals[run]

    lines = []
    group = group_status(run)
    if train:
        last = train[-1]
        step = last["step"]
        age = time.time() - (run / "metrics.jsonl").stat().st_mtime
        finished = total is not None and val and val[-1]["step"] >= total
        status = "[green]finished[/]" if finished else "[yellow]stalled?[/]" if age > 120 else "[cyan]training[/]"
        extra = ""
        if group and group["mine"]:
            me = group["runs"][group["mine"]]
            if me.get("done"):
                status = "[green]finished[/]"
            elif not group["fresh"]:
                status = "[yellow]trainer stopped?[/]"
            elif group["now"] == group["mine"]:
                status = "[cyan]training[/]" + ("  [dim](waiting for data)[/]" if me.get("waiting_for_data") else "")
            else:
                status = f"[blue]queued[/] [dim](round {group['round']}, {group['now']} is training)[/]"
            if me.get("cooldown"):
                status += "  [magenta]cooling down[/]"
            extra = f"  games seen {me.get('games_seen', 0):,}  snapshots {me.get('snapshots', 0)}"
        progress = f"{step:,}/{total:,} ({step / total:.0%})" if total else f"{step:,}"
        eta = ""
        if total and len(train) >= 2 and not finished:
            recent = train[-min(len(train), 20):]
            rate = (recent[-1]["step"] - recent[0]["step"]) / max(1, recent[-1]["elapsed_s"] - recent[0]["elapsed_s"])
            eta = f"  ETA {fmt_duration((total - step) / rate)}" if rate > 0 else ""
        lines.append(Text.from_markup(f"{status}  step {progress}  elapsed {fmt_duration(last['elapsed_s'])}{eta}{extra}"))
        lines.append(Text.from_markup(
            f"train loss [bold]{last['loss']:.4f}[/]  lr {last['lr']:.2e}  {last['tokens_per_s'] / 1e3:,.0f}k tok/s  {last['tflops']:.1f} TFLOPS"
        ))
    elif group and group["mine"]:
        lines.append(Text.from_markup(f"[blue]starting[/]  (round-robin group {group.get('round', 0)}, {group['now']} is training)"))
    if val:
        losses = [v["val_loss_move"] for v in val]
        lines.append(Text.from_markup(f"val loss/move [bold]{losses[-1]:.4f}[/]  (best {min(losses):.4f})  {sparkline(losses)}"))
    if progress_line := eval_progress(run, evals):
        lines.append(progress_line)

    if evals:
        table = Table(box=None, header_style="bold", pad_edge=False)
        for col in ("progress", "step", "val loss", "Elo (95% CI)", "legal (real games)", "opening/mid/end", "self-play illegal", "moves to 1st illegal", "blunders", "ACPL"):
            table.add_column(col, justify="right")
        elos = [row_elo(r, calibration) for r in evals] if calibration else [(0, "—")] * len(evals)
        for r, (_, elo) in list(zip(evals, elos))[-rows:]:
            acpl = r.get("strength_acpl", r.get("selfplay_acpl", float("nan")))
            blunders = f"{r['strength_blunder_rate']:.1%}" if "strength_blunder_rate" in r else "—"
            table.add_row(
                f"{r['progress']:.0%}" if r.get("progress") is not None else "—", f"{r['step']:,}", f"{r['val_loss_move']:.3f}", elo,
                f"{r['legal_rate']:.2%}",
                "/".join(f"{r.get(f'legal_rate_{p}', float('nan')):.1%}" for p in ("opening", "middlegame", "endgame")),
                f"{r['selfplay_first_try_illegal_rate']:.2%}", f"{r['selfplay_moves_before_first_illegal']:.0f}",
                blunders, f"{acpl:.1f}",
            )
        lines.append(Text(""))
        lines.append(table)
        if calibration:
            lines.append(Text(f"Elo over training (legal-only self-play, {len(evals)} snapshots): {sparkline([e for e, _ in elos])}"))
        else:
            lines.append(Text("Elo: calibration not available yet (python -m chessgpt.eval.elo calibrate)", style="dim"))
    else:
        lines.append(Text("no snapshot evaluations yet (start: python -m chessgpt.eval.watch <run>)", style="dim"))
    return Panel(Group(*lines), title=f"[bold]{run.parent.name}[/] / {run.name}", title_align="left")


def find_runs(all_runs: bool) -> list[Path]:
    runs = sorted(RUNS_DIR.glob("*/*/metrics.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = [p.parent for p in runs]
    return runs if all_runs else runs[:1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="show every run")
    parser.add_argument("--interval", type=float, default=5, help="seconds between refreshes")
    parser.add_argument("--rows", type=int, help="snapshot evaluations to show per run (default: fit the terminal)")
    parser.add_argument("--once", action="store_true", help="print once and exit")
    args = parser.parse_args()

    totals: dict = {}
    console = Console()

    def render():
        runs = [r.resolve() for r in args.runs] or find_runs(args.all)
        calibration = load_calibration()
        # Fit the terminal: each panel needs ~9 lines besides its table rows.
        rows = args.rows or max(3, (console.size.height - 9 * max(1, len(runs))) // max(1, len(runs)))
        panels = [run_panel(r, totals, calibration, rows) for r in runs]
        if (stream := stream_panel()) is not None:
            panels.insert(0, stream)
        return Group(*panels) if panels else Text("no runs found")

    if args.once:
        Console(width=160).print(render())
        return
    # The alternate screen is double-buffered, so redraws don't flicker (also over ssh); Ctrl+C restores the terminal.
    with Live(render(), console=console, screen=True, auto_refresh=False) as live:
        while True:
            time.sleep(args.interval)
            live.update(render(), refresh=True)


if __name__ == "__main__":
    main()
