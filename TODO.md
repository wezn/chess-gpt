# TODO

## Done: basic models
- [x] Dataset 1: 5M Lichess rapid+ games (`lichess-2026-09-rapid`)
- [x] Tokenizers A/B/C: `pgn-char`, `uci-move`, `uci-square`
- [x] Model + training loop, snapshots every 2%
- [x] Throughput benchmarks and epoch-time estimates for every size
- [x] Eval: legal-move rate (real games by phase, self-play), ACPL/accuracy via Stockfish, Elo calibration
- [x] Live dashboard (`chess-monitor`), play in the browser (`chess-play`)
- [x] Large A/B/C trained for one epoch on the same games, evaluated every 2%

## Next: settle the comparison
- [ ] Final evaluation of the three best checkpoints: ≥512 self-play games each, raw and legal-only
- [ ] Matches against Stockfish at fixed `UCI_Elo` levels as a ground-truth Elo check
- [ ] Legal-only mode in the evaluation suite (how much Elo do illegal moves cost?)
- [ ] Puzzle accuracy as a third strength measure
- [ ] Second epoch / next Lichess month: strength was still rising at 100%

## After the basic models
- [ ] Conditioning tokens: prefix games with `<elo_1800>`, `<rapid>` etc. and test "play like a 2200"
- [ ] Result token: `<1-0>` at the end (predict the winner) or at the start (condition on winning)
- [ ] BPE tokenizer on PGN text: does it discover moves as tokens?
- [ ] Board-state tokens (64 squares, FEN-like): position -> move models, good fit for puzzles
- [ ] Dataset 2: puzzles (game prefix + solution)
- [ ] Dataset 3: engine vs engine games
- [ ] Dataset with a higher Elo filter, to compare against the all-ratings one
