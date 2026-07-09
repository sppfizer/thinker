# 🧠 Thinker — Physical Routing Model (PRM)

> An experimental design for a non-neural-network language model that trains entirely forward, without backpropagation.

---

## What Is This?

This repository documents the design thinking behind **PRM — the Physical Routing Model**, a from-scratch alternative architecture for language models. The core idea: instead of neural networks with backpropagation, use a *physical routing metaphor* — balls (tokens) falling through a grid of adjustable nails — where the correct answer acts as a **magnet** guiding the system during training.

No gradients. No backward pass. One forward pass, real-time correction.

---

## The Core Metaphor

```
  Input tokens → balls of different sizes (size = frequency weight)
  Model internals → grid of nails (tilt = routing weight, diameter = bias)
  Training signal → magnet at the correct output slot
  Output → vocabulary slot collecting the most ball-mass wins
```

Balls fall simultaneously, interact with each other (gravitational attraction at distance, elastic collision up close), and are routed by nails that nudge themselves toward the correct answer as each ball passes — during the forward pass itself.

---

## Key Innovations

| Feature | How It Works |
|---|---|
| **No backpropagation** | Nails update locally as balls pass — correction force is applied forward |
| **No learned embeddings** | Token weight = corpus frequency, pre-computed once |
| **No attention matrix** | Token relationships emerge from physical ball-to-ball forces |
| **No positional encoding** | Sequence position = ball entry x-position; proximity = interaction strength |
| **Natural language prior** | Output slot width ∝ token frequency — common words are geometrically easier to produce |
| **Natural sparsity** | Open-border diamond grid lets irrelevant balls fall off the edges |

---

## The Diamond Grid

The grid is **diamond-shaped** with open borders:

```
  ● ● ● ● ●         ← input tokens (balls)
      ↓
     ● ● ●
    /  ↓  \
   / nails  \
  /  grid    \
  /            \
  \   routes   /
  \          /
   \   ↓    /
    \______/
  narrow → wide → narrow
```

- **Widening phase** = brainstorming, exploring meaning space
- **Narrowing phase** = summarizing, committing to an answer
- **Open edges** = natural forgetting; irrelevant context balls fall off
- **W:N ratio** (widening rows : narrowing rows) is a tunable hyperparameter

---

## Role-Based Specialists

Multiple diamonds can run as **role specialists** — each trained on a *cognitive style*, not a knowledge domain:

| Role | Diamond Shape | Cognitive Style |
|---|---|---|
| Analyst | W:N = 1:1, thick nails | Logical, step-by-step |
| Generator | W:N = 3:1, thin nails | Creative, exploratory |
| Synthesizer | W:N = 1:3, medium nails | Compression, abstraction |
| Precisionist | W:N = 1:4, very thick nails | Exact, minimal inference |
| Narrator | W:N = 2:2, mixed nails | Sequential explanation |
| Conversationalist | W:N = 2:1, thin nails | Context-aware, adaptive |

A **router diamond** (shallower, cheaper) selects the best role for each input. For ≤3 specialists, run all in parallel and pick the highest retained-ball-mass winner. For >3, the pre-selector is more efficient.

---

## Files

| File | Contents |
|---|---|
| [`model-insight.md`](./model-insight.md) | Full design document — all architecture decisions, diagrams, training loops, open questions, prior art comparison |
| [`thinking-process.md`](./thinking-process.md) | Raw conversation log — how each design decision was reached, turn by turn |
| [`PRM-Design.pptx`](./PRM-Design.pptx) | Slide deck version of the brainstorm and model summary |
| [`PRM.sln`](./PRM.sln) | C# solution containing the PRM.Core library and PRM.App console app |
| [`src/PRM`](./src/PRM) | Forward-only PRM prototype in C# with training, test, tune, val, benchmark, optimize, autooptimize, and viz modes |
| [`data/simple_corpus.txt`](./data/simple_corpus.txt) | Larger toy corpus used for training and optimizer sweeps |
| [`prm_config.json`](./prm_config.json) | Persisted best diamond configuration from the optimizer |

---

## Status

🟡 **All blocking bugs fixed — ready for first real optimizer run.**

### How to run the optimizer (fresh start):

```powershell
cd C:\src\Claude\Thinker
# Delete stale state first — these were from tiny_corpus (40 tokens, incompatible)
Remove-Item prm_best_params.json, prm_config.json -ErrorAction SilentlyContinue
# Start optimizer on the 209-token corpus
dotnet run --project src/PRM/PRM.App -- --corpus data/simple_corpus.txt autooptimize
```

### How to visualize the trained model:

```powershell
cd C:\src\Claude\Thinker
dotnet run --project src/PRM/PRM.App -- viz the cat sat
# Browser opens automatically at http://localhost:5000
# Use the combobox to select word combinations and watch balls route
```

### Key training facts

| Parameter | Value | Why |
|---|---|---|
| Corpus | `data/simple_corpus.txt` | 209 tokens / 604 samples — large enough to stress-test routing |
| Split | 70/10/20 | 422 train / 60 tune / 122 val, genuinely separate |
| Default rows | vocab/6 capped at 80 | 4× was too sparse (27M params, 2 updates/param avg) |
| Default MaxWidth | 2× entryWidth | balance routing capacity with training coverage |
| Deflection | `offX * maxStepX * idf` | scaled to per-row budget; output-slot reachable from anywhere |
| Nail update | error-correction form | natural equilibrium, no drift to ±1 |

### Bug-fix history summary

15 blocking and methodology bugs were found and fixed in a joint audit by GPT-5.5 and Claude Opus 4.8. See `model-insight.md §3.1` and `thinking-process.md §Implementation Learnings` for the full table.

The most critical fix (bug #20 in model-insight.md) was deflection scaling: `rawStepX = offX * alpha` gave a maximum travel of 21 grid units over 35 rows on a 418-unit-wide output zone — structurally impossible to route correctly. Fixed to `rawStepX = offX * maxStepX * idf` so the per-row budget scales with the actual row width.

---

*Co-designed by sppfizer + GitHub Copilot — 2026-07-09*
