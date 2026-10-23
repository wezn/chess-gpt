# TODO

## Now: basic models
- [x] Dataset 1: 5M Lichess rapid+ games (`lichess-2026-09-rapid`)
- [x] Tokenizers A/B/C: `pgn-char`, `uci-move`, `uci-square`
- [x] Model + training loop
- [ ] Full tiny runs (A, B, C) + time estimates for every size
- [ ] Eval suite (legal-move rate, puzzles, vs Stockfish)
- [ ] Experiments: one model per tokenizer (A, B, C), compared at equal compute, loss reported per move

## After the basic models
- [ ] Conditioning tokens: prefix games with `<elo_1800>`, `<rapid>` etc. and test "play like a 2200"
- [ ] Result token: `<1-0>` at the end (predict the winner) or at the start (condition on winning)
- [ ] BPE tokenizer on PGN text: does it discover moves as tokens?
- [ ] Board-state tokens (64 squares, FEN-like): position -> move models, good fit for puzzles
- [ ] Dataset 2: puzzles (game prefix + solution)
- [ ] Dataset 3: engine vs engine games
- [ ] Dataset with a higher Elo filter, to compare against the all-ratings one
