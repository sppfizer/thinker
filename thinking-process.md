# 🧠 Alternative LLM Architecture — Thinking Process

> Started: 2026-07-08
> Participants: sppfizer + Copilot (Claude Sonnet 4.6)
> Goal: Design a non-neural-network LLM that trains forward-only with active correction force

---

## 🎯 Core Objective

Build a model that:
- Can be **trained** (learns from data)
- Produces **next-token predictions** from input tokens
- Builds **contextual answers** via token loop
- Trains in a **fraction of current time** (no backpropagation)
- Uses **forward-only, forced training** — the correct answer acts like a magnet during training

---

## 💡 The Nail Grid Metaphor (Seed Idea — Turn 1)

> *"Imagine a grid of nails and balls falling through."*

| Concept | Physical Object | Meaning |
|---|---|---|
| Token | Ball | A unit of meaning |
| Token importance/frequency | Ball size | How often it appears in the corpus |
| Model internals | Grid of nails | Processing / routing layer |
| Nail position | Tilt/angle | Routing direction = learned weight |
| Nail diameter | Thickness | Bias — thicker nail = more resistant to change |
| Training signal | Magnet below grid | Known correct output token = active force |
| Inference | Ball freely falling | No magnet, nails guide ball to predicted output |
| Training update | Nail nudging | Each forward pass nudges nails toward correct output |

### Key Insight — Forced Forward Training
- Classical NN: forward pass → compute loss → **backward** propagation → update weights
- Proposed: forward pass → at each nail, a **live correction force** steers the ball → **single-pass update**
- No gradient needs to travel backward — the target exerts a **pull force** throughout the forward path

---

## 🔤 Vocabulary Pre-Training (Token Weighting)

- Vocabulary is built **before** main training (like BPE/SentencePiece)
- But tokens are ordered/weighted by **corpus-wide frequency**
- High-frequency token = large ball = hits nails harder = influences nail adjustment more
- This embeds **importance** into the physical property of the token, not learned separately

---

## 🔩 Nail Properties

Each nail has two physical properties:

1. **Position / tilt** — which direction does the ball bounce?
   - Adjustable during training
   - Nudges in direction of expected output (magnetic pull)

2. **Diameter / thickness** — how resistant is the nail to being moved?
   - High diameter = high bias = nail stays stable = dominant routing rule
   - Low diameter = easily nudged = flexible / context-sensitive routing

---

## 🎯 Output Layer — Weighted Vocabulary Slots

- Slot width at the bottom ∝ **token frequency** in vocabulary
- **Winner = slot that collects most ball-mass** (weight × count)
- Common tokens: wide slots, easy to hit → natural language prior
- Rare tokens: narrow slots, require precise nail routing → learned precision
- Rare correct answer = stronger magnet force → larger nail nudge → model learns harder on surprises

```
Bottom:
┌──────────┬──┬────┬───────────────┬──────┐
│  "the"   │"a│"is"│   "quantum"   │"cat" │
│ (wide)   │  │    │   (medium)    │(thin)│
└──────────┴──┴────┴───────────────┴──────┘
  high freq              medium       rare
```


| Range | Force | What it encodes |
|---|---|---|
| Far apart in grid | Gravitational attraction (∝ size × proximity) | Semantic affinity ("Paris" pulls "France") |
| Crossing paths | Elastic collision (deflect by mass ratio) | Token dominance (subject overrides preposition) |

### Prediction Slot Ball
- A special near-zero-mass ball dropped at the "next token" position
- Has no inherent identity — gets pulled/shaped entirely by context balls
- Where it lands = predicted next token
- Maps to the "query" in attention — but physical, no dot-product


- [ ] What does "position" of a nail map to mathematically?
- [ ] How do multiple balls (sequence of input tokens) interact with the same grid?
- [ ] Is the grid layered (depth) or flat (single pass)?
- [ ] How does the magnet (target token) exert force if it's at the end — does each layer "feel" it?
- [ ] How does context accumulate across a sequence?
- [ ] What is the inference-time equivalent (no magnet)?
- [ ] How does this relate to Hinton's Forward-Forward Algorithm (2022)?
- [ ] Can nail positions encode **relational** meaning (token A near token B = similar routing)?

---

## 📚 Related Prior Art (for reference, not to copy)

| Concept | Relevance |
|---|---|
| Hinton's Forward-Forward Algorithm (2022) | Forward-only learning, two-pass (positive/negative) |
| Direct Feedback Alignment | Bypasses backprop with random feedback |
| Reservoir Computing / Echo State Networks | Fixed random internal grid, only output layer trained |
| Hopfield Networks / Associative Memory | Content-addressable, attractor states |
| Decision Routing Networks | Ball-routing analogy |
| Trie / Prefix Trees | Token routing by sequence |

---

## 🗣️ Conversation Log

### Turn 1 — sppfizer (seed idea)
- Proposed nail grid + ball metaphor
- Forward-only training with magnetic correction force
- Ball size = token frequency weight
- Nail diameter = bias / resistance
- Vocabulary pre-ordered by frequency

### Turn 1 — Copilot response
- Framed this as a **routing problem** (not regression)
- Proposed: nail grid = weighted DAG, training = path-bending toward target
- Asked: how does the sequence of input tokens enter the grid?

### Turn 2 — sppfizer
- **Simultaneously** — balls interact mid-fall (parallel, like attention)
- Multiple balls in the air at once, they can interact with each other and with nails

### Turn 2 — Copilot response
- Large balls deflect small balls → dominant token shapes context
- Two balls on same nail → joint weight response → emergent context
- Natural token-to-token relationships WITHOUT explicit attention matrix
- Open question: how does position in sequence encode into the ball/entry point?

### Turn 3 — sppfizer
- Do balls also have direct impact on each other (not just via nails)?

### Turn 4 — sppfizer
- Chose **both** interaction modes: attraction at distance + collision when crossing

### Turn 5 — sppfizer
- Output = full weighted vocabulary at bottom, slot size ∝ token frequency
- Slot that collects most balls = winning token (natural weighted vote)

### Turn 5 — Copilot response
- Confirmed: slot width = frequency = built-in language prior
- Rare tokens need precise routing → larger nail nudge during training
- Training on rare/surprising tokens has strongest learning signal naturally

### Turn 6 — sppfizer
- Proposed **diamond grid shape** over rectangle
- Open borders preferred (balls fall off = natural forgetting)
- Diamond: widening = divergent thinking, narrowing = summarization
- Asymmetric W:N ratio is a tunable depth parameter
- Magnet must follow the diamond shape (fan out in widening, converge in narrowing)
- More layers = more memory capacity

### Turn 7 — sppfizer
- Proposed multiple specialist diamonds trained on categorized data
- Problem: how to pick the best specialist efficiently?
- Two approaches: **pre-selector diamond** (routes first, then activates winner) vs **all parallel** (run all, pick highest output weight)
- Pre-selector is a cost if only 3 specialists → better to run all 3 parallel
- Pre-selector makes sense when specialist count is large
- Pre-selector re-feeds exact same input tokens to the winning specialist

### Turn 8 — sppfizer
- Proposed: start with 5 randomly initialized diamonds, train all on same first 10% data
- Router also gets same input but only has 5 output slots
- Identified the core problem: without a magnet target for the router, you can't train it
- Concluded: manual category labeling is likely required — or is there another way?

### Turn 8 — Copilot response
- Identified the chicken-and-egg loop: router needs specialist winners → specialists need training → training needs categories → categories need router
- Proposed **EM Bootstrap** as a third option:
  1. Blind parallel training (all specialists, no router) on first N% data
  2. Measure retained ball-mass ratio per specialist per sample → auto-generate labels
  3. Use auto-labels to bootstrap router training (magnet = auto-winner slot)
  4. Re-specialize each specialist on its assigned samples
  5. Iterate until convergence
- Caveat: EM produces performatively different specialists, not necessarily domain-interpretable ones
- Three approaches summarized: manual labeling (best quality), EM bootstrap (no labels, emergent), auto-topic-modeling (LDA pre-pass, semi-interpretable)
- Manual labeling preferred if domain-interpretable specialists are desired

### Turn 9 — sppfizer
- Key reframe: specialists should not be "category-based" (science, math) but "role-based"
- A role = a way of thinking/processing, not a domain of knowledge
- Analogy: a person can be good at analytical thinking across many domains — that's a role, not a category
- Role transfers across domains; category is brittle at domain borders

### Turn 9 — Copilot response
- Role = cognitive style; category = content domain
- Role-trained specialist generalizes better (analytical thinking applies to math AND code AND law)
- Router asks "what thinking style does this need?" not "what topic is this?"
- KEY INSIGHT: the diamond shape itself physically encodes the role:
  - Generator role → wide W:N ratio, thin nails (explore broadly)
  - Precisionist role → narrow W:N ratio, thick nails (stable, direct)
  - Synthesizer role → deep narrowing phase (compression-heavy)
- The role IS the shape — geometry + training data together define the specialist

### Turn 10 — sppfizer
- Requested a PowerPoint version of the brainstorm in the repo
- Asked to translate the physical model into fast GPU/CPU math
- Asked to create a C# application with training, test, tune, and val modes
- Inference stays straightforward because the diamond grids are fixed and balls just fall through

### Turn 10 — Copilot response
- Generated `PRM-Design.pptx` and synced it with the repository contents
- Added a formal math model section to `model-insight.md` describing state, diamond geometry, deflection, gravity, collisions, magnet force, scoring, and GPU parallelism
- Created a C# solution scaffold with `PRM.Core` and `PRM.App`
- Implemented the app modes: training, test, tune, and val
- Kept the inference path fixed-grid and forward-only: no magnet, no updates, just ball routing

### Turn 11 — sppfizer
- Asked for a more visual PowerPoint: diamond structure, ball size, nail size, and magnet-path effects
- Asked for a very simple LLM-free training set and a train/test/tune/val/benchmark flow

### Turn 11 — Copilot response
- Added a new visual mechanics slide to the PPT deck with big/small ball and thick/thin nail comparisons plus magnet path arrows
- Switched the app to load a tiny corpus from `data/simple_corpus.txt`
- Added train/test/tune/val splits and a benchmark mode that reports timing plus accuracy
- Verified the toy corpus runs through the full pipeline end to end

### Turn 12 — sppfizer
- Asked for an automatic iterative optimization loop that improves routing/magnet math
- Wanted live evolution plus rollback when candidates get worse

### Turn 12 — Copilot response
- Added `optimize` mode with a candidate sweep over magnet strength and diamond geometry
- Kept the best candidate by validation accuracy and rolled back the others automatically
- Fixed config persistence so the winning nails and the winning diamond geometry stay together
- Reran the benchmark on the persisted best state: validation improved from 22.2% to 27.8%

### Turn 13 — sppfizer
- Asked to implement contact-memory, angle-aware nail correction, nail inertia, widening bounce-back, stuck detection, and 5-retry training

### Turn 13 — Copilot response
- Added nail-contact memory to balls so successful low-contact paths can reinforce the nails they touched
- Made nail updates depend on angle disagreement with the target direction, ball mass, resistance, and density inertia
- Added widening-phase drop-off, narrowing-phase bounce-back, stuck detection, and retry-based re-training per sample
- Ran the optimizer sweep on the corpus again: flat baseline val=0.0%, sqrt-IDF val=11.1%, inverse-IDF val=22.2%
- Found and fixed an optimizer bug where the IDF power case was unreachable in `AutoOptimizer.Perturb`; the next sweep should now actually explore `DeflectionIdfPower`
- Found and fixed the legacy diameter alias so config-specified radius values actually apply, and aligned the widening/narrowing boundary so geometry, magnet, and bounds all agree
- Found and fixed the auto-scale clone so `DeflectionIdfPower` survives resize, which keeps the IDF sweep live during optimization




---

## Implementation Learnings

### Critical Bugs Discovered (and Fixed)

1. **Grid too narrow** — EntryWidth defaulted to a small fixed value, giving ~10 nails for a 40-token vocabulary.  The model structurally could not differentiate most tokens because many shared the same output slot.  *Fix*: auto-scale `EntryWidth = vocab.Length × NailSpacing`.

2. **Mass-weighted IDF voting** — common tokens ("the", "a", "is") were heavy balls → won most slot votes → model predicted "the" for almost everything.  *Fix*: invert to IDF weighting (vote weight = 1/mass); rare tokens are more discriminative.

3. **Probe ball IDF explosion** — the prediction probe ball has mass 0.001 → 1/mass = 1000 → massively outweighed real tokens.  *Fix*: exclude probe ball (tokenId = -1) from IDF voting entirely.

4. **Gravity frozen at 0** — the hill-climbing optimizer used multiplicative perturbations; multiplying 0 by any factor stays 0.  G and CollisionRadius could never escape their zero initial values.  *Fix*: switched to additive perturbation for these parameters.

5. **Sequential train/val split** — with a small corpus sorted by sentence order, rare tokens only appeared toward the end and were split entirely into val/test.  Training never saw them.  *Fix*: shuffle the dataset before splitting.

6. **Ball velocity explosion** — elastic collisions between many balls can add velocity repeatedly per row, making velocities grow without bound → positions become Infinity/NaN → `NailColumn` returned garbage array indices → `IndexOutOfRangeException`.  *Fix*: clamp velocity to ±5 after each integration step; guard `NailColumn` against NaN/Infinity inputs.

7. **MagnetField zero force at widest rows** — midpoint (widest rows, most nails) returned force=0; those rows never trained. *Fix*: constant 0.4 in widening, linear 0.4→1.0 ramp in narrowing.

8. **Stiffening ratchet** — every nail touched during training gained Resistance/Density permanently. After ~10 epochs all nails saturated: inertia reached 9.0× initial, effective LR collapsed 12×, model froze at 33%. *Fix*: removed per-step stiffening; added SoftenContacts on miss; 2% per-epoch DecayNailStiffness toward InitNails baseline.

9. **Wrong nail trained** — `ApplyNailUpdates` recomputed column from post-deflection ball position; a different nail than the one that actually deflected the ball got the update. *Fix*: store `LastNailCol`/`LastNailTIdx` before deflecting; use stored values in update.

10. **Force-accumulation (no equilibrium)** — nail update `newX += lr * force` had no natural rest point; nails oscillated to ±1 boundary. *Fix*: error-correction form `newX = currentX + scale*(idealX − currentX)`; scale clamped to [0,1] to prevent overshoot.

11. **Optimizer restart regression** — `bestVal` stayed at global best during restarts; the new local region could never improve its own starting point. *Fix*: separated `currentVal`/`currentParams` (local hill-climb) from `globalBestVal`/`globalBestParams` (never regresses).

12. **Deflection scale invariant** — `rawStepX = offX * alpha` (constant ≤0.6 units/row regardless of grid width). For simple_corpus (418-unit output zone, 35 rows): max total deflection = 21 units = 5% of output zone → balls structurally unable to reach target slots → 0.8% accuracy (barely above random). *Fix*: `rawStepX = offX * maxStepX * idf` where `maxStepX = rowWidth/TotalRows * alpha`. Also normalised training ideal: `normForceX = forceX * TotalRows / rowWidth` so idealX saturates at 1 when ball is ≥one row-budget away from target.

---

### What the Optimizer Found

After fixing the additive-perturbation bug (so G and collision could actually be explored):

- **WideningRatio converges to 79–83%** across independent restarts — more thinking rows = better routing capacity
- **NailSpacing prefers 1.8–1.9** (slightly tighter than the default 2.0) — denser nail coverage improves discrimination
- **More epochs needed**: convergence at this scale requires 80–130 epochs, not the earlier default of 30
- Gravity (G ≈ 0.01) and collision (R ≈ 0.5) both contribute; disabling either slightly hurts accuracy

---

### Phase 1 Accuracy Milestone (tiny_corpus, 40 tokens)

| Metric | Value |
|---|---|
| Best validation accuracy | **33.3%** on 40-token toy corpus |
| Training samples | 87 |
| Random baseline (1/40 tokens) | 2.5% |
| Model performance | **13× above random** |

Ceiling analysis: tiny_corpus has ~67% hapax legomena (tokens appearing once) → theoretical ceiling ~33-44%. The model hit the structural ceiling. All further optimization requires the 209-token simple_corpus.

---

### Phase 2 Status — Blocking Bugs Fixed, Ready for Real Run

After the 15-bug audit-and-fix cycle (bugs 8–12 above + methodology fixes), the model is ready for its first meaningful optimization run on `simple_corpus.txt`. The deflection-scaling fix (bug #12) was the last critical blocker: without it, balls could never reach their target output slots on a 209-token grid regardless of hyperparameters.

**How to run:**
```
cd C:\src\Claude\Thinker
dotnet run --project src/PRM/PRM.App -- --corpus data/simple_corpus.txt autooptimize
```

---

### Next Steps

- First real optimizer run on simple_corpus.txt — expect >33% val once deflection scaling works
- Investigate: should `anglePenalty = 1 + (1-dot)` be damped? (Opus flagged it amplifies oscillations for conflicting samples)
- Larger corpus: more tokens and more samples to test routing capacity at scale
- Multi-token generation loop (predict → append → predict again)
- Eventually: real text corpora beyond the toy 209-token set
