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




---
