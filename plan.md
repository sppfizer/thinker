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
- [x] All artifacts pushed to `github.com/sppfizer/thinker`

---

## 🔲 TODO — Next Session

### 🔴 Critical — Learning performance broken

- [ ] **Re-run `autooptimize` on tiny corpus** after IDF changes
  - Before fix: 33% val. After flat-voting + idf=1f in deflection: stuck at 1.9% train.
  - Optimizer needs fresh sweep with current settings to find new good params.
  - Command: `cd src/PRM/PRM.App && dotnet run -- autooptimize --corpus tiny_corpus.txt`

- [ ] **Large corpus (209 tokens) still 0% val** — overfitting
  - Only ~1.7 training samples/token → model memorises noise, doesn't generalise.
  - Options to try:
    - (a) Increase corpus size (add more sentences to `simple_corpus.txt`)
    - (b) Data augmentation: shuffle token order within samples
    - (c) Revisit IDF in deflection: try `sqrt(mass)` instead of flat — was 3% train / 0.8% val

### 🟡 Visualizer — polish & robustness

- [ ] **Make balls more visible**: increase min radius, brighter colors, stronger glow
- [ ] **Lighter theme option**: `#1a1a2e` background was preferred — add a toggle or make default lighter
- [ ] **WebSocket reconnect after new viz** — after `VisualiseSingle` completes and user types new tokens,
  verify browser updates cleanly without needing a page refresh
- [ ] **Show nail offset arrows** more clearly — scale arrow length, color by direction magnitude

### 🟢 Model improvements — ideas to explore

- [ ] **IDF in deflection revisited**: test `idf = 1/sqrt(mass)` systematically across tiny + large corpus
- [ ] **Ball interaction (gravity)**: re-enable `GravityG > 0` in optimizer sweep — currently 0.0
- [ ] **Collision radius**: re-enable `CollisionRadius > 0` — balls pushing each other could help routing
- [ ] **Specialist diamonds**: train 2–3 role configs (Analyst, Generator, Synthesizer) on same corpus and compare
- [ ] **Seq-to-seq inference**: chain predictions (output ball → next input) to generate multi-token completions

### 📄 Documentation

- [ ] Update `model-insight.md` section on IDF (describe the voting vs deflection split)
- [ ] Add benchmark table: tiny corpus accuracy history (was 33%, current state after IDF fix)
- [ ] Update `thinking-process.md` with IDF bug discovery and visualizer work

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
