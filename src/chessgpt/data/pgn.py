"""Fast, streaming PGN parsing for Lichess dumps.

python-chess is too slow to parse every game in a monthly dump (~100M games), so
this module does plain text parsing. Headers are parsed for every game; movetext
is only cleaned for games that pass the header filters.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]$')
COMMENT_RE = re.compile(r"\{[^}]*\}")
MOVE_NUMBER_RE = re.compile(r"^\d+\.+")
SAN_RE = re.compile(
    r"^(?:O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8](?:=[QRBN])?)[+#]?$"
)
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


@dataclass
class RawGame:
    headers: dict[str, str]
    movetext: str


def iter_games(lines: Iterable[str]) -> Iterator[RawGame]:
    """Split a stream of PGN lines into games. Every Lichess game starts with an [Event] header."""
    headers: dict[str, str] = {}
    movetext: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            if line.startswith("[Event ") and headers:
                yield RawGame(headers, " ".join(movetext))
                headers, movetext = {}, []
            m = HEADER_RE.match(line)
            if m:
                headers[m[1]] = m[2]
        else:
            movetext.append(line)
    if headers:
        yield RawGame(headers, " ".join(movetext))


def clean_moves(movetext: str) -> list[str] | None:
    """Strip comments (clocks, evals), move numbers, annotations and the result.

    Returns the SAN moves, or None if anything unexpected is left (the game is then skipped).
    """
    moves = []
    for tok in COMMENT_RE.sub(" ", movetext).split():
        tok = MOVE_NUMBER_RE.sub("", tok)
        if not tok or tok in RESULTS or tok.startswith("$"):
            continue
        tok = tok.rstrip("?!")
        if not SAN_RE.match(tok):
            return None
        moves.append(tok)
    return moves


def speed(time_control: str) -> str:
    """Lichess speed category from a TimeControl header like "180+2" (estimated duration = base + 40 * increment)."""
    if time_control == "-":
        return "correspondence"
    base, _, inc = time_control.partition("+")
    estimate = int(base) + 40 * int(inc or 0)
    if estimate < 30:
        return "ultrabullet"
    if estimate < 180:
        return "bullet"
    if estimate < 480:
        return "blitz"
    if estimate < 1500:
        return "rapid"
    return "classical"
