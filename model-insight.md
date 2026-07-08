# 🧠 Physical Routing Model (PRM) — Full Design Insight

> Version: 0.3 — 2026-07-08
> Authors: sppfizer + Copilot (Claude Sonnet 4.6)
> Status: Conceptual — active design thinking

---

## 1. Vision & Motivation

### What We Want to Build

A language model that:
- Can be **trained** on text corpora and learns to predict the next token
- Produces **contextual answers** by looping token predictions
- Trains in a **fraction of current LLM training time**
- Uses **no backpropagation** — training is entirely forward-pass driven
- Has an **active correction force** during training: the known correct answer pulls the model toward it *while* the forward pass is happening, not after

### What We Reject

| Classical NN Approach | Why We Step Away |
|---|---|
| Backpropagation | Two-phase (forward + backward), expensive, requires storing all activations |
| Gradient descent | Requires computing global loss and propagating it backward through all layers |
| Learned embeddings | Separate expensive pre-training step just to encode token identity |
| Attention matrices | Quadratic dot-product over all token pairs every pass |

### The Core Intuition

> *"Instead of finishing a shot, computing how far off it was, and adjusting — you put a magnet at the target and let it guide the shot in real time."*

---

## 2. The Physical Metaphor — Nail Grid & Balls

The entire model is understood through a physical metaphor that maps 1:1 to the computational concepts.

### 2.1 The Grid

A **two-dimensional grid of nails** — deep (many rows) and wide (vocabulary-scale width). Each nail can tilt/rotate slightly and has a physical diameter.

```
  [INPUT TOKENS — balls dropped from top]
  ●    ●●      ●        ●    ○  ← different sizes = token weights
  |    ||      |        |    |
  ↓    ↓↓      ↓        ↓    ↓
  ╔════════════════════════════════╗
  ║  ⊕   ⊕   ⊕   ⊕   ⊕   ⊕   ⊕  ║  ← row of nails
  ║    ⊕   ⊕   ⊕   ⊕   ⊕   ⊕    ║
  ║  ⊕   ⊕   ⊕   ⊕   ⊕   ⊕   ⊕  ║
  ║    ⊕   ⊕   ⊕   ⊕   ⊕   ⊕    ║
  ║  ⊕   ⊕   ⊕   ⊕   ⊕   ⊕   ⊕  ║
  ╚════════════════════════════════╝
  [ "the" ]["a"]["is"]["quantum"]["cat"]  ← vocabulary slots
     wide    sm  sm     medium    thin
```

---

### 2.2 The Balls (Tokens)

| Ball Property | What It Represents | How It's Set |
|---|---|---|
| **Size / mass** | Token frequency weight | Pre-computed from corpus frequency before training |
| **Entry x-position** | Position in the input sequence | Left-to-right = token order in context |
| **Identity** | Which token it is | Determines starting x-position and mass |

**Key design choice:** Token importance is **pre-embedded** into the ball's physical size — not learned. High-frequency tokens ("the", "a", "is") are large, heavy balls. Rare tokens ("Constantinople", "photosynthesis") are small, light balls.

This means the vocabulary pre-training step produces a **frequency-ranked token table** — the only required pre-processing step before main training begins.

---

### 2.3 The Nails (Internal Model State)

Each nail has **two independently adjustable physical properties**:

#### Property 1 — Tilt / Position (Routing Weight)
- Determines which direction a ball bounces off the nail
- Adjusted during every training pass
- The magnetic force toward the correct output token nudges the tilt
- This is the primary **learned parameter** of the model

#### Property 2 — Diameter / Thickness (Bias / Stability)
- Determines how **resistant** the nail is to being moved
- Thick nail = high bias = stable routing rule = dominant directional preference
- Thin nail = low bias = easily nudged = flexible, context-sensitive routing
- Diameter can itself be initialized and optionally learned
- High-diameter nails create **anchored routing lanes** — structural knowledge that resists noise
- Low-diameter nails create **soft routing** — subtle contextual adjustments

```
Thin nail:   |    → ball deflects it easily, nail tilts a lot during training
Thick nail:  ▐▌   → ball barely moves it, nail holds its routing direction
```

---

### 2.4 Ball-to-Ball Interaction (Context = Physics)

All input tokens fall **simultaneously**. Balls interact with each other in two distinct force regimes:

#### Force 1 — Gravitational Attraction (Long Range)
- All balls in the grid pull each other proportionally to `size × proximity`
- Heavier balls exert stronger pull
- Naturally produces **semantic clustering**: "Paris" and "France" falling near each other will curve toward similar routing paths
- This encodes **token-to-token affinity without a learned attention matrix**

#### Force 2 — Elastic Collision (Short Range)
- When two ball paths cross in the grid, they collide
- **Larger ball deflects less; smaller ball deflects more** (mass ratio determines deflection magnitude)
- Produces **token dominance**: subject noun overrides preposition, verb overrides filler word
- Creates naturally **asymmetric influence** — high-importance tokens steer the routing landscape

```
Long range:
  ●          ●        →    ●       ●
                            ↖     ↗   (paths curve toward each other)

Short range (collision):
  ●  →←  ●   →    ●→  ←●   (larger wins, smaller deflects)
```

#### Why Both Forces Together
- Attraction at distance creates **semantic neighborhoods**
- Collision at proximity creates **local dominance hierarchies**
- Together they produce **rich context representations** from pure physics — no learned attention mechanism needed

---

### 2.5 The Prediction Slot Ball (The Query)

For next-token prediction, a special **zero-mass or near-zero-mass ball** is dropped at the position representing "the token to be predicted":

- Has **no inherent identity** — no size bias toward any specific token
- Gets **pulled entirely** by the context balls around it (gravitational attraction)
- Gets **deflected** by collisions with context balls (elastic collisions)
- Where it **lands** at the bottom = the predicted next token

This is the physical analog of the **query vector** in transformer attention:
- Context balls = keys and values
- Prediction slot ball = query
- No dot-product, no softmax — just physics

---

### 2.6 The Training Magnet (Forward Correction Force)

During training, the correct next token is known. A **magnetic force** is placed at the bottom of the grid, centered on the **correct output token's slot**.

#### How It Works
- The magnet exerts an upward pull-field **through the entire grid depth**
- Every ball in the grid — especially the prediction slot ball — feels a tilt toward the correct output
- As a ball passes each nail, the magnet's residual force tells the nail: *"tilt slightly more in this direction"*
- The nail nudges by an amount proportional to: `ball_mass × magnet_force_at_depth × nail_flexibility (1/diameter)`

#### Why This Is Faster Than Backpropagation
| Backpropagation | Forward Magnetic Training |
|---|---|
| Full forward pass first | Single pass — update happens **as the ball falls** |
| Store all intermediate activations | No storage needed — nail updates are local and immediate |
| Compute global loss | Loss is implicit — magnet strength IS the correction signal |
| Propagate gradients backward through all layers | Each nail adjusts locally in the forward direction |
| Two computational phases | **One computational phase** |

#### Magnet Strength and Rarity
- When the correct answer is a **rare token** (narrow slot), the magnet pulls harder to concentrate the prediction ball into a tiny zone
- Result: **larger nail nudges for surprising/rare correct answers**
- Naturally mirrors language: the model learns most from unusual, information-dense outputs

---

### 2.7 The Output Layer — Weighted Vocabulary Slots

The bottom of the grid is divided into **vocabulary-sized collection slots**:

```
┌──────────┬───┬──────┬───────────────┬───┬──────┐
│  "the"   │"a"│ "is" │   "quantum"   │"…"│"cat" │
│ (wide)   │(sm│(med) │   (medium)    │   │(thin)│
└──────────┴───┴──────┴───────────────┴───┴──────┘
  high freq   ↑            medium           rare
          common
```

#### Slot Rules
| Slot Property | Value | Effect |
|---|---|---|
| Width | ∝ token corpus frequency | Common tokens easier to hit → natural language prior |
| Winner selection | Slot with highest collected ball-mass wins | Weighted vote — heavier balls count more |
| Rare token slot | Narrow | Requires precise routing to win → model must learn exactly |
| Inference | No magnet, balls fall freely | Nails guide prediction ball to highest-mass slot |

#### Why Slot Width = Frequency Is Elegant
- No explicit probability distribution needed at output
- The prior is **built into the geometry**
- Model still learns to override the prior when context demands it (routing beats width for precisely trained scenarios)
- Rare tokens require more training examples to reliably route balls into their narrow slot — which matches empirical LLM training observations

---

## 3. Full Architecture Summary

### Components

| Component | Physical Analog | Computational Role |
|---|---|---|
| Ball mass | Token frequency rank | Token importance weight (pre-computed) |
| Ball x-entry position | Sequence position | Positional encoding (implicit, geometric) |
| Grid width | Vocabulary size | Routing space |
| Grid depth | Model depth / layers | Abstraction level |
| Nail tilt | Routing weight | Primary learned parameter |
| Nail diameter | Routing bias / stability | Resistance to weight update |
| Ball-to-ball attraction | Token affinity | Semantic relationship (no attention matrix) |
| Ball-to-ball collision | Token dominance | Context hierarchy |
| Prediction slot ball | Query token | Zero-mass ball shaped by context |
| Training magnet | Known correct output | Forward correction force |
| Vocabulary slots | Output distribution | Weighted collection zones (width = frequency) |

### Training Loop (Single Forward Pass)

```
For each training example (context_tokens → target_token):

  1. Drop all context balls simultaneously from their x-positions (= sequence positions)
  2. Drop prediction slot ball (zero mass) at "next position"
  3. Balls fall — interacting via:
       a. Gravitational attraction to each other
       b. Elastic collisions when paths cross
       c. Deflection off nails (nail tilt determines direction)
  4. Magnet at bottom pulls toward target_token's slot — force felt throughout fall
  5. At each nail a passing ball hits:
       nail_tilt += ball_mass × magnet_force_at_depth × (1 / nail_diameter)
  6. Winner = slot with highest ball-mass collected
  7. If winner == target_token → reinforce (small positive nail lock)
     If winner ≠ target_token → already corrected by magnet (no backward pass needed)
```

### Inference Loop

```
Given input sequence → predict next token:

  1. Drop all context balls + prediction slot ball
  2. No magnet — balls fall freely through trained nail grid
  3. Balls interact (attraction + collision)
  4. Winner slot = predicted next token
  5. Append predicted token → repeat for next position
```

---

## 4. Grid Shape Architecture

### 4.1 Why Diamond Over Rectangle

Three grid shape options were considered and ranked:

| Shape | Border | Verdict |
|---|---|---|
| Rectangle, closed borders | Balls bounce back in | ❌ No information pruning, cluttered routing |
| Rectangle, open borders | Balls fall off edges | ⚠️ Better, but no think/summarize structure |
| **Diamond, open borders** | Balls fall off expanding then narrowing edges | ✅ **Preferred — natural think + summarize + forget** |

---

### 4.2 The Diamond Grid — Structure

```
        Input tokens (balls drop here)
           ● ● ○ ● ●  ○  ●
           | | | | |  |  |
    ┌──────┴─┴─┴─┴─┴──┴──┴──────┐   ← narrow entry
    │   ⊕   ⊕   ⊕   ⊕   ⊕   ⊕  │
     \  ⊕  ⊕   ⊕   ⊕   ⊕  ⊕   /
      \ ⊕   ⊕   ⊕   ⊕   ⊕  ⊕  /    ← WIDENING PHASE
       \⊕  ⊕   ⊕   ⊕   ⊕  ⊕  /       (divergent thinking)
        ⊕   ⊕   ⊕   ⊕   ⊕  ⊕         balls can fall off
        ⊕   ⊕   ⊕   ⊕   ⊕            at widest = max exploration
       /⊕  ⊕   ⊕   ⊕   ⊕  \
      / ⊕   ⊕   ⊕   ⊕   ⊕  \    ← NARROWING PHASE
     /  ⊕  ⊕   ⊕   ⊕   ⊕   \       (convergent summarization)
    │   ⊕   ⊕   ⊕   ⊕   ⊕   │       weak balls fall off edges
    └────────────────────────┘   ← narrow output
       [vocab slots — winner]
```

**Balls that drift to the open edges fall out** — they are gone. Only balls that remain in active routing lanes contribute to the output.

---

### 4.3 Phase Meanings

| Phase | Shape | What Happens | Cognitive Analog |
|---|---|---|---|
| **Widening** | Grid expands outward | Balls spread into larger nail space, more interactions possible | Brainstorming, exploring meaning space |
| **Midpoint** | Widest row | Maximum ball-to-ball interaction surface, maximum context mixing | Full contextual awareness |
| **Narrowing** | Grid contracts inward | Balls compete for fewer lanes, weak signals crowd out | Summarizing, deciding, focusing |
| **Output** | Narrowest exit | Surviving high-mass ball routes hit vocabulary slots | Committed answer |

---

### 4.4 Asymmetric Diamond — Tunable Depth Ratio

The widening and narrowing phases do **not** have to be equal. This is a key design parameter:

```
Ratio W:N = widening rows : narrowing rows

W:N = 1:1  →  balanced think and summarize
W:N = 3:1  →  long exploration, quick decision (creative / generative tasks)
W:N = 1:3  →  quick spread, long convergence (analytical / precise tasks)
W:N = 5:1  →  very deep thinking before any compression
```

The total depth (W + N rows) determines **memory capacity** — more rows = more nail interactions = more context the model can "hold" while routing.

---

### 4.5 Open Borders — Natural Information Pruning

Balls that reach the lateral edge of the grid fall out. This creates **emergent sparsity**:

- Trained nails will learn to keep **relevant balls in** (route them inward or keep them central)
- **Irrelevant context balls are allowed to drift off** — nails don't fight to retain them
- This replaces: dropout, regularization, attention masking — all emergent from physics
- The more training passes, the more aggressively nails route important balls to center, unimportant ones toward edges

**Critical property:** A ball that falls off still nudged every nail it touched before leaving. Its training contribution is proportional to how far it got before exiting. Short-lived balls (fell off quickly) had minimal influence on internal routing.

---

### 4.6 The Diamond Magnet Field

During training, the magnet (correct output token) does not simply pull straight down. Its field shape **follows the diamond**:

```
WIDENING PHASE — magnet field fans outward:
       ↙ ↓ ↘           ← force spreads
     ↙   ↓   ↘         ← encourages balls to explore
   ↙     ↓     ↘

MIDPOINT — field is maximally diffuse:
   ← ← ← ↓ → → →     ← neutral, exploration maximum

NARROWING PHASE — field contracts inward:
   ↘     ↓     ↙
     ↘   ↓   ↙         ← force focuses
       ↘ ↓ ↙           ← converges to target slot
         ⬛  ← target token slot (magnet center)
```

**Why this is important:** Without the fan-shaped field in the widening phase, the magnet would fight against the widening and compress routing too early. The diamond magnet field *cooperates* with the grid shape — it lets the model think wide first, then focuses the correction force for the narrowing phase.

Nail update rule per phase:
- **Widening rows**: `nail_tilt += ball_mass × magnet_fan_force × (1 / nail_diameter)` — fan angle increases with row depth
- **Narrowing rows**: `nail_tilt += ball_mass × magnet_focus_force × (1 / nail_diameter)` — converging force increases as output approaches

---

### 4.7 Diamond + Open Borders as a Memory Mechanism

> *"The more layers, the more the whole can remember."*

More rows = more nails = more routing decisions = **more stored relational context**. Each nail's tilt encodes a fragment of learned routing knowledge. The total number of nails is the model's **parameter count** (analogous to weights in a neural network).

| Grid Size | Rough Analogy |
|---|---|
| Small diamond (e.g. 10W + 10N rows) | Small model, limited context retention |
| Medium diamond (50W + 50N rows) | Mid-scale model |
| Large diamond (200W + 100N rows, asymmetric) | Large model, rich exploration, strong summarization |
| Very deep diamond | Long-context model — distant tokens can still interact meaningfully |

The open borders mean that adding more rows doesn't necessarily add noise — irrelevant routing paths simply produce more balls falling off edges, keeping the output clean.

---

## 5. Parallel Specialist Architecture (Mixture of Expert Diamonds)

### 5.1 Core Concept — Role Specialists, Not Category Specialists

A critical distinction: specialists are trained on **roles** (cognitive styles / ways of processing), not **categories** (domains of knowledge).

| Category Specialist | Role Specialist |
|---|---|
| Trained on: science data | Trained on: analytical content across ALL domains |
| Fails on: new science not in training | Generalizes to: any task requiring analytical thinking |
| Router asks: "what topic is this?" | Router asks: "what thinking style does this need?" |
| Brittle at domain borders | Robust — roles overlap comfortably |

**Why roles generalize better:** An "analyst" trained on math problems, legal arguments, code reviews, and scientific papers learns *analytical thinking itself* — not math, law, code, or science. When it encounters a new domain with analytical structure, it applies the same routing patterns.

### Example Role Set

| Role | Cognitive Style | Trained On |
|---|---|---|
| **Analyst** | Step-by-step logical deduction | Math, code, logic puzzles, arguments across all domains |
| **Generator** | Broad exploration, novelty | Creative writing, brainstorming, analogies, fiction |
| **Synthesizer** | Compression, abstraction | Summaries, abstracts, conclusions, overviews |
| **Precisionist** | Exact recall, minimal inference | Definitions, citations, factual Q&A, specifications |
| **Narrator** | Sequential explanation, teaching | Tutorials, storytelling, step-by-step guides |
| **Conversationalist** | Context-aware, tone-matching | Dialogue, interviews, back-and-forth exchanges |

---

### 5.2 The Role IS the Diamond Shape

The physical geometry of the diamond encodes the cognitive style of the role:

| Role | W:N Ratio | Nail Diameter | Physical Meaning |
|---|---|---|---|
| **Analyst** | 1:1 balanced | Thick (stable) | Methodical, resists noise, precise routing |
| **Generator** | 3:1 wide | Thin (flexible) | Explores broadly before committing |
| **Synthesizer** | 1:3 deep narrow | Medium | Quick spread, long convergence = compression |
| **Precisionist** | 1:4 very narrow | Very thick | Almost no exploration, direct to answer |
| **Conversationalist** | 2:1 moderate | Thin | Context-aware, easily shaped by input balls |
| **Narrator** | 2:2 balanced | Mixed | Broad spread then structured convergence |

A role specialist is therefore defined by **two things together**:
1. Its **diamond geometry** (W:N ratio, width, nail diameter distribution) — the physical shape of thinking
2. Its **training data** — role-appropriate content regardless of knowledge domain



### 5.3 Winner Determination — Parallel Mode

When all N specialists run simultaneously with the same input tokens, the winner is the specialist whose output has the **highest retained ball-mass ratio**:

```
retained_mass_ratio = (ball-mass reaching output slots) / (total ball-mass dropped in)
```

- **High ratio** → specialist kept most balls in → input was in its domain → confident answer
- **Low ratio** → specialist lost most balls off edges → input was out of domain → uncertain answer

This is elegant: **balls falling off the open borders are the confidence signal**. No separate scoring mechanism needed — the physics already encodes domain confidence.

```
Input: "Solve the integral of x² dx"

  Specialist A (Narrator):    ratio = 0.12  ← lost most balls
  Specialist B (Generator):   ratio = 0.19  ← lost most balls
  Specialist C (Synthesizer): ratio = 0.31
  Specialist D (Analyst):     ratio = 0.78  ← winner ✓
  Specialist E (Precisionist):ratio = 0.61  ← strong but Analyst wins
```

---

### 5.4 Break-Even: Parallel vs Pre-Selector

Let **R** = cost of routing diamond, **S** = cost of one specialist diamond:

| Approach | Total Cost | Preferred When |
|---|---|---|
| **All parallel** | N × S | N ≤ 3 (small specialist pool) |
| **Pre-selector + 1 winner** | R + S | N > (R/S) + 1 |

**Key insight:** The pre-selector diamond does NOT need vocabulary output slots — its output slots are **specialist IDs**. This means it can be significantly shallower and cheaper than a full specialist diamond:

- If R ≈ 0.3 × S → pre-selector wins for any **N ≥ 2**
- If R ≈ S → pre-selector wins for **N ≥ 3**

**Rule of thumb:** ≤ 3 specialists → run parallel; > 3 specialists → use pre-selector.

---

### 5.5 The Pre-Selector Diamond

A special **smaller diamond** whose job is purely classification, not generation:

```
         Input tokens (same balls, same drop)
              ● ● ○ ● ●
                  ↓
         ┌── ROUTING DIAMOND ──┐
         │  (shallower, cheaper│
         │   no vocab output)  │
         └─────────────────────┘
              ↓         ↓
         [Spec. A]  [Spec. D]  [Spec. B] ...   ← output slots = specialist IDs
              ↓
         Specialist D activated
              ↓
         Input tokens re-dropped into Specialist D diamond
              ↓
         [vocabulary output slots → next token]
```

**Properties of the pre-selector:**
- Shallower diamond (fewer rows needed — classification is a simpler task than generation)
- Output slots at bottom = specialist IDs, not vocabulary tokens
- Slot width ∝ how commonly that specialist is activated (frequent categories = wider slots)
- Same ball-mass retention ratio signals routing confidence
- The original input tokens are **not consumed** — they are re-dropped identically into the winning specialist

---

### 5.6 Soft Routing (Near-Border Landings)

When the pre-selector ball lands near the **border between two specialist slots**, both specialists can be partially activated:

```
  [Spec. A | Spec. D]
       ↑ ball lands here → activate both, weighted by distance from border

  Spec. A weight = 0.35
  Spec. D weight = 0.65
  → Both run, outputs combined proportionally by weight
```

This is **soft routing** — more expensive (two specialists run) but handles ambiguous domain inputs gracefully. It activates naturally from the physics without a separate mechanism.

---

### 5.7 Hierarchical Routing (Many Specialists)

For very large specialist pools, a **tree of pre-selectors** can route inputs progressively:

```
Level 1:  [Broad roles: Analyst | Generator | Synthesizer | Precisionist | Narrator]
                                  ↓ (winner selected)
Level 2:  [Sub-roles within Analyst: Deductive | Inductive | Comparative | ...]
                                  ↓ (winner selected)
Level 3:  [Specialist diamond runs and produces token output]
```

Each routing level adds cost R but eliminates (branching_factor - 1) × S of specialist cost.
Optimal tree depth depends on total specialist count and R/S ratio.

---

### 5.8 Training the Pre-Selector

The routing diamond is trained separately with **category-labeled data**:
- Input: any text sample
- Target output slot: the specialist ID that produced the highest-quality output for this sample
- The same magnetic training mechanism applies — target slot is the known correct specialist

Over time the pre-selector learns to route by **distributional patterns in the input token balls** — not by semantic understanding, but by the physical signature of which tokens arrive and in what configuration.

### 5.9 Specialist Self-Organization vs Manual Labeling

Three approaches to defining specialist categories:

#### Option A — Manual Category Labeling (Recommended for interpretability)
- Human defines categories (science, code, math, language, facts…)
- Training data pre-sorted into categories
- Each specialist trained on its category data with normal magnetic training
- Router trained with category ID as magnet target
- **Result**: clean, domain-interpretable specialists

#### Option B — EM Bootstrap (No labels needed)
The chicken-and-egg problem: router needs specialist winners → specialists need domain training → domain training needs categories → categories need router.

**Broken by Expectation-Maximization:**
```
Phase 1: Train all N specialists blindly on same data (no router, no categories)
         Random init causes gradual divergence in nail configurations

Phase 2: For each training sample — run all specialists, measure retained ball-mass ratio
         Auto-label: sample → specialist with highest ratio = natural winner

Phase 3: Use auto-labels as router magnet targets → train router

Phase 4: Re-train each specialist on only its assigned samples → deepen specialization

Phase 5: Repeat phases 2-4 until convergence
```
**Caveat**: produces performatively different specialists (not guaranteed domain-interpretable)

#### Option C — Auto-Topic Modeling Pre-Pass
- Run LDA or vocabulary-clustering on training corpus before any diamond training
- Use cluster IDs as category labels → train router and specialists on these
- Semi-interpretable, no human labeling needed, less clean than manual

| Approach | Labels | Specialist Type | Recommended For |
|---|---|---|---|
| Manual labeling | Human effort | Domain-interpretable | Production models |
| EM Bootstrap | None | Emergent/performative | Research, exploration |
| Auto-topic modeling | Algorithmic | Semi-interpretable | Compromise |

---

## 6. Key Properties and Advantages

### ⚡ Speed
- **No backward pass** → ~50% less compute per training step minimum
- **Local nail updates** → no need to store full activation graph
- **No global loss computation** → magnet force replaces gradient signal

### 🧲 Forward Training Force
- Correct answer actively steers the model *during* the forward pass
- No delayed feedback — the nail learns *at the moment the ball passes*
- Conceptually: the model is **corrected in real time**, not retrospectively

### 🔤 Natural Positional Encoding
- Ball entry x-position = sequence position
- Nearby tokens have closer balls → stronger gravitational interaction
- Distant tokens have weaker interaction → natural positional decay
- **No sinusoidal or learned positional encodings needed**

### 📊 Natural Language Prior
- Slot widths encode corpus frequency distribution
- Model must work against the prior only when context is precise
- Common-sense defaults are **geometrically embedded**

### 🔗 Context Without Attention
- Token relationships emerge from physics (attraction + collision)
- No O(n²) attention matrix
- Interaction strength naturally decays with distance in the grid

---

## 7. Open Questions for Further Design

- [ ] **Grid depth** — does depth = abstraction level? Is there a minimum depth for coherent language?
- [ ] **Nail initialization** — random, uniform, or frequency-informed starting tilts?
- [ ] **Magnet field shape** — point magnet at target slot, or gradient field across the full vocabulary bottom?
- [ ] **Ball-to-ball force decay** — how quickly does gravitational attraction drop off with grid distance?
- [ ] **Collision elasticity** — perfectly elastic (energy conserved) or damped (energy absorbed = information loss)?
- [ ] **Multi-layer architecture** — are multiple stacked grids needed, or is one deep grid sufficient?
- [ ] **Training stabilization** — what prevents nails from drifting to extreme tilts (exploding routing)?
- [ ] **Nail diameter learning** — should diameter be fixed or also trainable (meta-learning the stability)?
- [ ] **Vocabulary size scalability** — with 50k+ tokens the slot layer is enormous; does this need compression?
- [ ] **Long context** — balls from distant past tokens will have very weak gravitational influence; is that sufficient or does a memory mechanism need to be added?

---

## 8. Relation to Prior Art

| Prior Concept | Similarity | Key Difference |
|---|---|---|
| **Hinton's Forward-Forward Algorithm (2022)** | Forward-only training, no backprop | FF uses two forward passes (positive + negative data); PRM uses one pass with magnetic force |
| **Direct Feedback Alignment** | Bypasses gradient backprop | DFA still requires a backward signal; PRM has no backward signal at all |
| **Reservoir Computing** | Fixed internal routing grid | Reservoir has fixed (non-learned) internals; PRM nails are fully learned |
| **Hopfield Networks** | Attractor states, energy minimization | Hopfield is recurrent; PRM is purely feed-forward |
| **Routing Networks / Mixture of Experts** | Token routing through the model | MoE routing is learned by backprop; PRM routing is learned by magnetic nudge |
| **Physical Hash Maps** | Deterministic routing of items | Hash maps don't generalize; PRM generalizes via continuous nail adjustment |
| **Galton Board (bean machine)** | Physical ball-through-nails | Galton board is static and probabilistic; PRM nails are adaptive and trained |

---

## 9. Name Proposal

**PRM — Physical Routing Model**

- Emphasizes the routing nature (not regression)
- References the physical metaphor without being too whimsical
- Distinguishes clearly from Neural Networks, Transformers, and all prior art

---

*Document continues as thinking progresses — see `thinking-process.md` for conversation log.*
