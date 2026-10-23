"""Play against a trained model in your browser.

    uv run chess-play                      # then open http://127.0.0.1:8000
    uv run chess-play --host 0.0.0.0       # reachable from other machines on your network

Pick any run's best checkpoint or latest snapshot, your colour, and a mode:
  raw         the model writes its move like in the evaluations; illegal attempts are
              shown, and if every attempt fails a random legal move is played
  legal-only  the model's most likely *legal* move (never illegal)
The panel shows the model's top candidate moves with their probabilities.
"""

import argparse
import json
import random
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess

from chessgpt.eval.player import ModelPlayer

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def list_models() -> list[dict]:
    models = []
    for run in sorted(RUNS_DIR.glob("*/*/")):
        if (run / "best.pt").exists():
            models.append({"id": str(run / "best.pt"), "label": f"{run.parent.name} · best ({run.name})"})
        snaps = sorted((run / "snapshots").glob("step-*.pt"))
        if snaps:
            models.append({"id": str(snaps[-1]), "label": f"{run.parent.name} · latest snapshot {snaps[-1].stem} ({run.name})"})
    return models


class Games:
    def __init__(self):
        self.games: dict[str, dict] = {}
        self.players: dict[tuple[str, float], ModelPlayer] = {}
        self.lock = threading.Lock()  # one model call at a time on the GPU

    def player(self, path: str) -> ModelPlayer:
        allowed = {m["id"] for m in list_models()}
        if path not in allowed:
            raise ValueError("unknown model")
        key = (path, Path(path).stat().st_mtime)  # best.pt changes while a run trains
        if key not in self.players:
            self.players = {k: v for k, v in self.players.items() if k[0] != path}
            self.players[key] = ModelPlayer(path)
        return self.players[key]

    def new(self, model: str, color: str, mode: str) -> dict:
        if model not in {m["id"] for m in list_models()} or mode not in ("raw", "legal-only"):
            raise ValueError("unknown model or mode")
        game_id = uuid.uuid4().hex
        self.games[game_id] = {"board": chess.Board(), "model": model, "human": chess.WHITE if color == "white" else chess.BLACK, "mode": mode, "last": None}
        g = self.games[game_id]
        if g["human"] == chess.BLACK:
            self.model_move(g)
        return self.state(game_id)

    def human_move(self, game_id: str, uci: str) -> dict:
        g = self.games[game_id]
        board = g["board"]
        move = chess.Move.from_uci(uci)
        if board.turn != g["human"] or move not in board.legal_moves:
            raise ValueError("illegal move")
        board.push(move)
        if not board.is_game_over(claim_draw=True):
            self.model_move(g)
        return self.state(game_id)

    def model_move(self, g: dict) -> None:
        board = g["board"]
        with self.lock:
            player = self.player(g["model"])
            ranked = player.rank_legal_moves(board)
            info = {"mode": g["mode"], "top": [[board.san(m), p] for m, p in (ranked or [])[:5]]}
            if ranked is None:
                move = random.choice(list(board.legal_moves))
                info |= {"status": "context", "note": "game is longer than the model's context; random move"}
            elif g["mode"] == "legal-only":
                move = ranked[0][0]
                info |= {"status": "legal"}
            else:
                choice = player.choose_moves([board])[0]
                move = choice.move or random.choice(list(board.legal_moves))
                info |= {"status": choice.status, "illegal_attempts": choice.illegal_attempts, "first_output": choice.first_output}
                if choice.move is None:
                    info["note"] = "every attempt was illegal; played a random legal move"
        info["san"] = board.san(move)
        board.push(move)
        g["last"] = info

    def state(self, game_id: str) -> dict:
        g = self.games[game_id]
        board = g["board"]
        replay = chess.Board()
        sans = []
        for m in board.move_stack:
            sans.append(replay.san(m))
            replay.push(m)
        over = board.is_game_over(claim_draw=True)
        return {
            "game": game_id,
            "fen": board.fen(),
            "pieces": [(board.piece_at(sq).symbol() if board.piece_at(sq) else "") for sq in chess.SQUARES],
            "human": "white" if g["human"] == chess.WHITE else "black",
            "turn": "white" if board.turn == chess.WHITE else "black",
            "legal": [m.uci() for m in board.legal_moves] if not over else [],
            "moves": sans,
            "last_move": board.peek().uci() if board.move_stack else None,
            "check": board.is_check(),
            "result": board.result(claim_draw=True) if over else None,
            "model_info": g["last"],
        }


GAMES = Games()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/models":
            self._json(list_models())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/new":
                self._json(GAMES.new(data["model"], data["color"], data["mode"]))
            elif self.path == "/api/move":
                self._json(GAMES.human_move(data["game"], data["uci"]))
            else:
                self._json({"error": "not found"}, 404)
        except (KeyError, ValueError) as e:
            self._json({"error": str(e)}, 400)

    def log_message(self, *args):
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Chess GPT</title>
<style>
 body { font-family: system-ui, sans-serif; background: #1e1e1e; color: #ddd; margin: 24px; }
 .wrap { display: flex; gap: 28px; flex-wrap: wrap; }
 #board { display: grid; grid-template-columns: repeat(8, 64px); border: 3px solid #555; user-select: none; }
 .sq { width: 64px; height: 64px; display: flex; align-items: center; justify-content: center; font-size: 46px; cursor: pointer; position: relative; }
 .light { background: #eed7b4; } .dark { background: #b58863; }
 .last { box-shadow: inset 0 0 0 64px rgba(255, 230, 0, .35); }
 .sel { box-shadow: inset 0 0 0 4px #2b7de9; }
 .target::after { content: ""; position: absolute; width: 18px; height: 18px; border-radius: 50%; background: rgba(20, 90, 200, .55); }
 .check { box-shadow: inset 0 0 18px 6px #e33; }
 .w { color: #fff; text-shadow: 0 0 2px #000, 0 0 2px #000; } .b { color: #111; }
 .coord { position: absolute; font-size: 11px; color: #555; left: 3px; bottom: 1px; }
 .panel { width: 420px; }
 select, button { font-size: 14px; margin: 3px 0; padding: 4px 6px; }
 #moves { font-family: monospace; line-height: 1.6; max-height: 220px; overflow-y: auto; background: #2a2a2a; padding: 8px; }
 .bar { background: #2b7de9; height: 12px; display: inline-block; vertical-align: middle; }
 .muted { color: #999; } .warn { color: #f0a040; } .bad { color: #f06060; }
 h3 { margin: 16px 0 6px; }
</style></head>
<body>
<h2>Chess GPT</h2>
<div class="wrap">
 <div><div id="board"></div><div id="result" style="margin-top:10px;font-size:18px"></div></div>
 <div class="panel">
  <div><label>Model<br><select id="model" style="width:420px"></select></label></div>
  <div><label>You play <select id="color"><option>white</option><option>black</option></select></label>
   <label>Mode <select id="mode"><option value="raw">raw (can be illegal)</option><option value="legal-only">legal-only</option></select></label>
   <label>Promote to <select id="promo"><option value="q">queen</option><option value="r">rook</option><option value="b">bishop</option><option value="n">knight</option></select></label></div>
  <button id="newgame">New game</button>
  <h3>Model's last move</h3><div id="info" class="muted">—</div>
  <h3>Moves</h3><div id="moves"></div>
 </div>
</div>
<script>
const GLYPH = {K:"♚",Q:"♛",R:"♜",B:"♝",N:"♞",P:"♟",k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"};
let state = null, selected = null, busy = false;
const files = "abcdefgh";
const sqName = i => files[i % 8] + (Math.floor(i / 8) + 1);

async function api(path, body) {
  const r = await fetch(path, body ? {method: "POST", body: JSON.stringify(body)} : {});
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

function render() {
  const el = document.getElementById("board");
  el.innerHTML = "";
  const flip = state && state.human === "black";
  for (let row = 0; row < 8; row++) for (let col = 0; col < 8; col++) {
    const rank = flip ? row : 7 - row, file = flip ? 7 - col : col, i = rank * 8 + file;
    const d = document.createElement("div");
    d.className = "sq " + ((rank + file) % 2 ? "light" : "dark");
    const p = state ? state.pieces[i] : "";
    if (p) { d.textContent = GLYPH[p]; d.classList.add(p === p.toUpperCase() ? "w" : "b"); }
    const name = sqName(i);
    if (state && state.last_move && (state.last_move.slice(0, 2) === name || state.last_move.slice(2, 4) === name)) d.classList.add("last");
    if (selected === name) d.classList.add("sel");
    if (selected && state.legal.some(m => m.startsWith(selected + name))) d.classList.add("target");
    if (state && state.check && (p === (state.turn === "white" ? "K" : "k"))) d.classList.add("check");
    if (col === 0) { const c = document.createElement("span"); c.className = "coord"; c.textContent = rank + 1; d.appendChild(c); }
    d.onclick = () => click(name);
    el.appendChild(d);
  }
  if (!state) return;
  let html = "";
  state.moves.forEach((m, k) => { html += (k % 2 === 0 ? `${k / 2 + 1}. ` : "") + m + " "; });
  document.getElementById("moves").textContent = html;
  document.getElementById("result").textContent = state.result ? `Game over: ${state.result}` : (busy ? "Model is thinking…" : `${state.turn} to move`);
  const info = state.model_info;
  if (info) {
    let s = `<b>${info.san}</b> `;
    if (info.mode === "raw") {
      if (info.status === "legal" && info.illegal_attempts === 0) s += `<span class="muted">(first try)</span>`;
      else if (info.status === "legal") s += `<span class="warn">(after ${info.illegal_attempts} illegal attempt(s); first wrote "${info.first_output}")</span>`;
      else s += `<span class="bad">(model wrote "${info.first_output}", illegal)</span>`;
    }
    if (info.note) s += `<br><span class="bad">${info.note}</span>`;
    s += "<br><br>Model's top legal candidates:<br>" + info.top.map(([m, p]) =>
      `<span style="display:inline-block;width:70px">${m}</span><span class="bar" style="width:${Math.round(p * 250)}px"></span> ${(p * 100).toFixed(1)}%`).join("<br>");
    document.getElementById("info").innerHTML = s;
  }
}

async function click(name) {
  if (!state || busy || state.result || state.turn !== state.human) return;
  const own = state.legal.some(m => m.startsWith(name));
  if (selected) {
    const cands = state.legal.filter(m => m.startsWith(selected + name));
    if (cands.length) {
      const uci = cands.length > 1 ? selected + name + document.getElementById("promo").value : cands[0];
      selected = null; busy = true;
      // show the human move immediately
      const from = files.indexOf(uci[0]) + (uci[1] - 1) * 8, to = files.indexOf(uci[2]) + (uci[3] - 1) * 8;
      state.pieces[to] = state.pieces[from]; state.pieces[from] = ""; state.last_move = uci; render();
      try { state = await api("/api/move", {game: state.game, uci}); } catch (e) { alert(e.message); }
      busy = false; render(); return;
    }
  }
  selected = own ? name : null;
  render();
}

document.getElementById("newgame").onclick = async () => {
  busy = true; selected = null;
  document.getElementById("info").textContent = "—";
  try {
    state = await api("/api/new", {model: document.getElementById("model").value, color: document.getElementById("color").value, mode: document.getElementById("mode").value});
  } catch (e) { alert(e.message); }
  busy = false; render();
};

(async () => {
  const models = await api("/api/models");
  const sel = document.getElementById("model");
  for (const m of models) { const o = document.createElement("option"); o.value = m.id; o.textContent = m.label; sel.appendChild(o); }
  const preferred = models.findIndex(m => m.label.startsWith("large-uci-square · best"));
  if (preferred >= 0) sel.selectedIndex = preferred;
  render();
})();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"open http://{args.host}:{args.port}  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
