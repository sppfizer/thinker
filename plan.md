# PRM — Project Plan

> **Physical Routing Model** — a non-neural-network LLM alternative where tokens are
> "balls" falling through a diamond-shaped nail grid with forward-only magnetic training.

---

## ✅ Completed

- [x] Concept design & math model (model-insight.md, thinking-process.md)
- [x] C# prototype: `PRM.Core` (BallSimulator, DiamondGrid, VocabularyBuilder, MagnetField)
- [x] C# app: `PRM.App` (train / val / test / autooptimize modes)
- [x] 2D nail grid: unit-circle offsets, staggered hex layout, position-aware routing tables
- [x] Auto-optimizer: hill-climbing over 15 hyperparams (rows, epochs, LR, alpha, width ratio, TrainPasses…)
- [x] Multi-pass training: `TrainPasses` hyperparameter with decayed LR per pass
- [x] Dynamic diamond ratio: optimizer finds best widening/narrowing split
- [x] IDF voting bug fixed: flat weights in `Score()` — IDF stays in training nudge only
- [x] Scale-with-vocab: `LoadBest` and `RandomRestart` scale params to vocabulary size
- [x] WebSocket visualizer: `PRM.Viz` — balls + nails + trails + output slots in browser
- [x] Diamond shape fixed in README (`model-insight.md`) and PPT (`PRM-Design.pptx`)
- [x] VizServer HTTP loop fix: `WaitForClientAsync` handles favicon → WebSocket connects reliably
- [x] Coordinate fix: browser uses `grid.Config.MaxWidth` (post-scale) matching C# physics
- [x] Fair causal context training: PRM now receives NN-comparable multi-token next-token samples, not fixed 3-token windows
- [x] OpenCL GPU diagnostics and training: ILGPU backend with CPU/GPU parity checks, persistent GPU session, grouped per-ball sample kernel, mini-batch GPU training, and `train --gpu` / `autooptimize --gpu`
- [x] All artifacts pushed to `github.com/sppfizer/thinker`

---

## 🔲 TODO — Next Session

### 🔴 Critical — GPU throughput and learning quality

- [x] **Implement GPU mini-batch training**
  - Samples are packed into one OpenCL launch with one workgroup per sample.
  - Offset updates use averaged atomic deltas and a projection pass to avoid unsafe direct write races.
  - CLI: `dotnet run --project src/PRM/PRM.App -- train --gpu --batch 32`

- [ ] **Re-run fair-context `autooptimize --gpu` with mini-batches**
  - Old 3-token-window best is invalid for NN comparison.
  - The current fair-context baseline is lower but comparable.
  - Use: `dotnet run --project src/PRM/PRM.App -- autooptimize 50 0.60 --gpu`

- [x] **Large corpus (209 tokens) overfitting reduced** — retry/contact/inertia sweep improved val to 22.2%
  - The corpus is still small, but the new training loop changes moved the benchmark off the 0% plateau.
  - Best current sweep on the 39-sample corpus: inverse-IDF deflection, val 22.2%, test 14.3%.
  - Remaining work: more data, data augmentation, and a larger-scale benchmark.

### 🟡 Visualizer — polish & robustness

- [ ] **Verify viz with controlled Chrome instance** (Puppeteer via Node.js or Playwright):
  - Launch `PRM.Viz`, then drive a headless/headed Chrome programmatically to
    open `http://localhost:5050/`, confirm WebSocket connects, confirm frames arrive,
    take a screenshot to verify balls are visible. This replaces "should work now" guessing.
- [ ] **Make balls more visible**: increase min radius, brighter colors, stronger glow
- [ ] **Lighter theme option**: `#1a1a2e` background was preferred — add a toggle or make default lighter
- [ ] **WebSocket reconnect after new viz** — after `VisualiseSingle` completes and user types new tokens,
  verify browser updates cleanly without needing a page refresh
- [ ] **Show nail offset arrows** more clearly — scale arrow length, color by direction magnitude
- [ ] **Add visualizer speed slider** — control fall-through speed in the browser so stuck/dropped balls are easier to inspect
- [ ] **Show trained nail pattern** — render trained nail positions and sizes clearly so the learned routing pattern is visible at a glance

### 🟢 Model improvements — ideas to explore

- [x] **IDF in deflection revisited**: tested flat vs sqrt vs inverse-mass on the current corpus
- [x] **Ball interaction (gravity/collision)**: supported by the GPU training subset
- [ ] **Specialist diamonds**: train 2–3 role configs (Analyst, Generator, Synthesizer) on same corpus and compare
- [ ] **Seq-to-seq inference**: chain predictions (output ball → next input) to generate multi-token completions

### 📄 Documentation

- [x] Update `model-insight.md` section on IDF (describe the voting vs deflection split)
- [x] Add benchmark table: tiny corpus accuracy history (was 33%, current state after IDF fix)
- [x] Update `thinking-process.md` with IDF bug discovery and visualizer work
- [x] Document GPU mini-batch training insight and current bottlenecks

---

## Architecture at a Glance

```
Input tokens (balls)
        ↓
  ┌──────────┐    ← entryWidth (= vocab × NailSpacing, auto-scaled)
  │ ⊕ ⊕ ⊕  │
 / ⊕  ⊕  ⊕  \
/  ⊕  ⊕  ⊕   \   WIDENING — divergent thinking
\  ⊕  ⊕  ⊕   /   NARROWING — convergent summarising
 \ ⊕  ⊕  ⊕  /
  │ ⊕ ⊕ ⊕  │
  └──────────┘    ← vocab output slots (winner = predicted token)
```

**Training**: magnetic force nudges nails toward the correct output slot.  
**Inference**: nails fixed, balls fall freely, winner slot = next token.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/PRM/PRM.Core/Engine/BallSimulator.cs` | Physics engine (nail deflection, ball interaction, trace) |
| `src/PRM/PRM.Core/Engine/DiamondGrid.cs` | Grid owner: train / predict / save / load |
| `src/PRM/PRM.App/AutoOptimizer.cs` | Hill-climbing optimizer (15 hyperparams) |
| `src/PRM/PRM.Viz/VizServer.cs` | HTTP + WebSocket server |
| `src/PRM/PRM.Viz/HtmlPage.cs` | Browser Canvas visualizer (JS) |
| `data/tiny_corpus.txt` | 15 sentences, 40 tokens — primary test set |
| `data/simple_corpus.txt` | 100 sentences, 209 tokens — larger corpus |
| `model-insight.md` | Full design document |
| `PRM-Design.pptx` | Presentation (generated via `py generate_pptx.py`) |
