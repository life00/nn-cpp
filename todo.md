# Todo

- [ ] `Net::setup`: sizes buffers, validates shapes, initializes weights from `rng`
- [ ] `forward`: `xi`/`y` for a batch, output in `y.back()`
- [ ] `backward`: delta from output, propagate, fill `gw`/`gb` (1/n averaged)
- [ ] `step`: `w -= lr * gw`
- [ ] Loss + batch loop in `main`: shuffle order, epoch timing, val accuracy
- [ ] Gate 2: XOR, 4/4, fixed seed
- [ ] Gate 3: finite-difference gradient check on a tiny net
- [ ] Gate 4: overfit 100 samples to ~100 % train accuracy
- [ ] Gate 1: real prediction writers — 60 000 train / 10 000 test lines from the model
- [ ] Gate 5: val accuracy >= 88 % before any test pass
- [ ] Profile one epoch; confirm total well under 600 s
- [ ] Gate 6: `evaluate.py` on test >= 0.8800 (single final test pass)
- [ ] Gate 7: clean unzip, rerun, same accuracy
- [ ] Consultation prep: explain every line; live modification
