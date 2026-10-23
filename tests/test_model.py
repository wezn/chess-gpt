import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from chessgpt.batches import IGNORE, EpochPlan, TokenizedSplit, make_batch, pack_rows
from chessgpt.model import GPT, GPTConfig, attention_mask
from chessgpt.tokenizers import BOS, EOS


@pytest.fixture
def split(tmp_path):
    rng = np.random.default_rng(0)
    games = [[BOS, *rng.integers(3, 50, size=n).tolist(), EOS] for n in (5, 9, 3, 12, 7, 30, 4)]
    data = np.array([t for g in games for t in g], dtype=np.uint16)
    data.tofile(tmp_path / "train.bin")
    np.save(tmp_path / "train_idx.npy", np.cumsum([0] + [len(g) for g in games[:-1]]))
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 50}))
    return TokenizedSplit(tmp_path, "train"), games


def test_pack_rows_fills_rows_without_overflow():
    lengths = np.array([5, 9, 3, 12, 7, 4, 16, 1])
    packed, bounds = pack_rows(lengths, np.arange(8), block_size=16)
    assert sorted(packed.tolist()) == list(range(8))
    rows = [lengths[packed[a:b]].sum() for a, b in zip(bounds, bounds[1:])]
    assert all(r <= 16 for r in rows) and sum(rows) == lengths.sum()
    assert len(rows) == 4  # 57 tokens in rows of 16: best fit reaches the minimum


def test_make_batch_targets_never_cross_games(split):
    s, games = split
    batch = make_batch(s, [np.array([0, 1, 2])], block_size=32)
    c = 0
    for j, g in enumerate(games[:3]):
        n = len(g) - 1
        assert batch.x[0, c : c + n].tolist() == g[:-1]
        assert batch.y[0, c : c + n].tolist() == g[1:]
        assert batch.pos[0, c : c + n].tolist() == list(range(n))
        assert (batch.doc[0, c : c + n] == j).all()
        c += n
    assert (batch.y[0, c:] == IGNORE).all() and (batch.doc[0, c:] == -1).all()


def test_long_games_are_truncated(split):
    s, games = split
    batch = make_batch(s, [np.array([5])], block_size=16)  # game 5 has 32 tokens
    assert batch.x[0].tolist() == games[5][:16] and batch.y[0].tolist() == games[5][1:17]


def test_epoch_plan_covers_every_game_once(split):
    s, _ = split
    plan = EpochPlan(s, block_size=16, rows=1, seed=0, epoch=0)
    seen = np.concatenate([plan.batch_games(b)[0] for b in range(plan.num_batches)])
    assert sorted(seen.tolist()) == list(range(len(s)))


def test_packed_logits_match_unpacked(split):
    """The mask and positions must make a packed game score exactly as if it were alone."""
    s, games = split
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=32)).eval()
    b = make_batch(s, [np.array([0, 1, 2])], block_size=32)
    x, pos, doc = (torch.from_numpy(a) for a in (b.x, b.pos, b.doc))
    packed = model(x, pos, attention_mask(doc))
    c = 0
    for g in games[:3]:
        n = len(g) - 1
        alone = model(torch.tensor([g[:-1]]))
        torch.testing.assert_close(packed[0, c : c + n], alone[0], atol=1e-5, rtol=1e-5)
        c += n
    loss = F.cross_entropy(packed[0], torch.from_numpy(b.y[0]), ignore_index=IGNORE)
    assert torch.isfinite(loss)
