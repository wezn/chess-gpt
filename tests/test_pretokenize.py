import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chessgpt.data import pretokenize as pt
from chessgpt.tokenizers import BOS, EOS, TOKENIZERS, get_tokenizer, san_to_uci

GAMES = [
    ("g0", "e4 e5 Nf3 Nc6 Bb5 a6"),
    ("g1", "d4 d5 c4 e6"),
    ("g2", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6"),
    ("g3", "c4 e5"),
] * 50  # 200 games so both splits get some


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    (processed / "tiny").mkdir(parents=True)
    ids = [f"{gid}-{i}" for i, (gid, _) in enumerate(GAMES)]
    pq.write_table(pa.table({"game_id": ids, "moves": [m for _, m in GAMES]}), processed / "tiny" / "shard-00000.parquet")
    monkeypatch.setattr(pt, "PROCESSED_DIR", processed)
    monkeypatch.setattr(pt, "TOKENIZED_DIR", tmp_path / "tokenized")
    return ids


@pytest.mark.parametrize("name", sorted(TOKENIZERS))
def test_pretokenize_roundtrip(dirs, name):
    out = pt.pretokenize("tiny", name, val_per_mille=200, workers=2)
    tok = get_tokenizer(name)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["splits"]["train"]["games"] + meta["splits"]["val"]["games"] == len(GAMES)
    assert meta["splits"]["val"]["games"] > 0

    by_split = {"train": [], "val": []}
    for gid, (_, moves) in zip(dirs, GAMES):
        by_split["val" if pt.is_val(gid, 200) else "train"].append(moves.split())
    for split, games in by_split.items():
        data = np.fromfile(out / f"{split}.bin", dtype=np.uint16)
        idx = np.load(out / f"{split}_idx.npy")
        bounds = [*idx, len(data)]
        decoded = []
        for a, b in zip(bounds, bounds[1:]):
            assert data[a] == BOS and data[b - 1] == EOS
            decoded.append(tok.decode(data[a:b].tolist()))
        assert decoded == [g if tok.notation == "san" else san_to_uci(g) for g in games]


def test_pretokenize_refuses_to_overwrite(dirs):
    pt.pretokenize("tiny", "uci-move", workers=1)
    with pytest.raises(FileExistsError):
        pt.pretokenize("tiny", "uci-move", workers=1)
