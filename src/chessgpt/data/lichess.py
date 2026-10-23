"""Build a filtered dataset of human games from a Lichess monthly dump.

The dump is streamed (download -> zstd decompress -> parse -> filter) so the ~30 GB
raw file never touches disk. Only games passing the filters are written.

    uv run python -m chessgpt.data.lichess configs/data/lichess-smoke.yaml [--max-mbps 7]
"""

import argparse
import io
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow as pa
import zstandard

from chessgpt.data.common import DatasetWriter, load_config
from chessgpt.data.pgn import RawGame, clean_moves, iter_games, speed

URL_TEMPLATE = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"

SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),  # Lichess game ID, used to match puzzles to their source game
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("time_control", pa.string()),
        ("speed", pa.string()),
        ("result", pa.string()),
        ("termination", pa.string()),
        ("eco", pa.string()),
        ("opening", pa.string()),
        ("num_plies", pa.int16()),
        ("moves", pa.string()),  # space-separated SAN, e.g. "e4 e5 Nf3 Nc6"
    ]
)


class Throttled(io.RawIOBase):
    """Wraps a binary stream and sleeps so reads average at most `max_mbps` megabits per second.

    Reading slowly makes TCP slow the sender down, so this limits actual network usage.
    """

    def __init__(self, raw, max_mbps: float):
        self.raw = raw
        self.bytes_per_sec = max_mbps * 1_000_000 / 8
        self.total = 0
        self.start = time.monotonic()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        n = self.raw.readinto(buffer)
        self.total += n
        ahead = self.total / self.bytes_per_sec - (time.monotonic() - self.start)
        if ahead > 0:
            time.sleep(ahead)
        return n

    def close(self) -> None:
        self.raw.close()
        super().close()


def open_source(source: dict, max_mbps: float | None = None) -> io.TextIOBase:
    """Open a dump as a text stream, from a local file (`path`) or by streaming it from Lichess (`month`)."""
    if "path" in source:
        raw = open(source["path"], "rb")
    else:
        raw = urllib.request.urlopen(URL_TEMPLATE.format(month=source["month"]))
        if max_mbps:
            raw = Throttled(raw, max_mbps)
    # Lichess dumps may use a large zstd window; allow up to 2 GB.
    reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(raw)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def reject_reason(game: RawGame, f: dict) -> str | None:
    """Return why a game fails the header filters, or None if it passes."""
    h = game.headers
    if f.get("exclude_bots", True) and "BOT" in (h.get("WhiteTitle"), h.get("BlackTitle")):
        return "bot"
    if h.get("Termination") not in f["terminations"]:
        return "termination"
    try:
        white_elo, black_elo = int(h["WhiteElo"]), int(h["BlackElo"])
    except (KeyError, ValueError):
        return "missing_elo"
    if min(white_elo, black_elo) < f.get("min_elo", 0):
        return "elo_too_low"
    if f.get("max_elo") is not None and max(white_elo, black_elo) > f["max_elo"]:
        return "elo_too_high"
    try:
        if speed(h["TimeControl"]) not in f["speeds"]:
            return "speed"
    except (KeyError, ValueError):
        return "bad_time_control"
    return None


def build(config: dict, max_mbps: float | None = None) -> Path:
    f = config["filters"]
    max_games = config.get("max_games")
    writer = DatasetWriter(config["name"], SCHEMA)
    stats: Counter[str] = Counter()
    start = time.monotonic()

    with open_source(config["source"], max_mbps) as stream:
        for game in iter_games(stream):
            stats["scanned"] += 1
            reason = reject_reason(game, f)
            if reason is None:
                moves = clean_moves(game.movetext)
                if moves is None:
                    reason = "unparseable_moves"
                elif len(moves) < f.get("min_plies", 0):
                    reason = "too_short"
            if reason:
                stats[f"rejected_{reason}"] += 1
            else:
                h = game.headers
                writer.write(
                    {
                        "game_id": h.get("Site", "").rsplit("/", 1)[-1],
                        "white_elo": int(h["WhiteElo"]),
                        "black_elo": int(h["BlackElo"]),
                        "time_control": h["TimeControl"],
                        "speed": speed(h["TimeControl"]),
                        "result": h.get("Result"),
                        "termination": h.get("Termination"),
                        "eco": h.get("ECO"),
                        "opening": h.get("Opening"),
                        "num_plies": len(moves),
                        "moves": " ".join(moves),
                    }
                )
                stats["kept"] += 1
                stats["plies"] += len(moves)
                if max_games and stats["kept"] >= max_games:
                    break
            if stats["scanned"] % 500_000 == 0:
                rate = stats["scanned"] / (time.monotonic() - start)
                print(f"scanned {stats['scanned']:,}  kept {stats['kept']:,}  ({rate:,.0f} games/s)", file=sys.stderr)

    stats["seconds"] = round(time.monotonic() - start)
    out = writer.finish({"config": config, "stats": dict(stats)})
    print(f"wrote {stats['kept']:,} games ({stats['plies']:,} plies) to {out}", file=sys.stderr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a configs/data/lichess-*.yaml file")
    parser.add_argument("--max-mbps", type=float, help="limit download speed, in megabits per second")
    args = parser.parse_args()
    build(load_config(args.config), args.max_mbps)


if __name__ == "__main__":
    main()
