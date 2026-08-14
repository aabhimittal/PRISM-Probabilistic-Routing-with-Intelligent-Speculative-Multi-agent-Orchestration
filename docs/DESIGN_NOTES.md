# Design Notes — the *why* behind PRISM

Architecture tells you *what* the pieces are. This document records the
reasoning: the choices that had real alternatives, why PRISM went the way it
did, and what it consciously gave up. If you're evaluating the idea, this is the
honest part.

---

## 1. Why speculative execution at all?

**The problem it solves.** Routing under uncertainty is a commitment problem. A
router that must pick one agent per step is forced to gamble exactly when it's
least able to — when two agents look equally good. Classic bandits handle the
*long-run* learning ("which arm is best on average") but say nothing about the
*single high-stakes decision in front of you right now*. You can be perfectly
calibrated on average and still ship a bad answer on the task that mattered.

**The insight from hardware.** CPUs face the identical dilemma at branches and
resolved it decades ago: don't stall, don't guess-and-pray — execute
speculatively and commit the survivor. The cost is wasted work on the wrong path;
the benefit is you never stall on an unresolved condition. For agents, "stall"
becomes "commit to a coin-flip," and the trade is compute for quality.

**Why it's not just ensembling.** An ensemble runs everything, every time, and
votes. PRISM runs multiple branches *only when the uncertainty math says it's
worth it*, and it keeps a single winner rather than averaging (averaging LLM
outputs is usually meaningless). Ensembling is a fixed cost; speculation is a
*demand-driven* cost that vanishes as the router learns. The convergence table in
the README (spec rate 0.70 → 0.00) is the difference made visible.

---

## 2. Why Beta distributions for beliefs?

Alternatives considered: point estimates (a running average per agent), Gaussians,
a learned neural value model.

- **Point estimates** throw away the one thing that matters here — *how sure are
  we?* Two agents at 0.75 are indistinguishable to a point estimate, but if one
  has 5 observations and the other 500, they should be treated completely
  differently. Rejected.
- **Gaussians** have support on all of ℝ; quality lives on `[0,1]`. You'd be
  constantly clamping, and the conjugacy that makes updates free is gone.
- **A learned value model** is the "serious ML" answer, but it needs training
  data, a training loop, and it turns a transparent decision into a black box —
  antithetical to the causal-tracing goal.
- **Beta** wins on every axis that matters for *this* system: it's conjugate
  (closed-form online updates, no training), it lives on `[0,1]`, it natively
  carries epistemic uncertainty, it composes (§4), and Thompson sampling over it
  is a provably strong explore/exploit strategy with **zero tuning knobs**.

The cost: Beta assumes rewards are (soft) Bernoulli and independent across draws.
For agent quality that's an approximation — but a defensible one, and a
transparent one.

---

## 3. Why Thompson sampling instead of UCB / ε-greedy?

All three are legitimate bandit strategies. The deciding factors:

- **ε-greedy** explores uniformly at random — it wastes exploration on arms
  already known to be bad, and needs a hand-tuned, usually-scheduled ε. Ugly.
- **UCB** is solid but needs a confidence-width constant `c` that you tune, and
  its optimism is deterministic (same decision every time given the same counts),
  which makes it awkward to compose with speculation's Monte-Carlo `p_best`.
- **Thompson sampling** explores *in exact proportion to uncertainty* with no
  knob, and — critically — it's **already sampling-based**, so the very same
  draws that route a single arm also power `probability_best` and the
  value-of-information estimate. One mechanism, three uses. That coherence is
  worth a lot.

A subtle but important fix lives here: the *non-speculative* path also routes via
a Thompson draw, not `argmax(mean)`. An early version committed greedily when it
chose not to speculate, and the router **never explored** — the first arm to get
lucky was picked forever and the others stayed at their priors. Routing must
explore on the cheap path too, or the whole policy stalls. (This is now a test.)

---

## 4. Why multiply qualities for uncertainty propagation?

The claim is "a chain is only as strong as its weakest link, and the uncertainty
compounds." Options for composing stage `i`'s quality into an end-to-end number:

- **min()** — "weakest link" literally. Correct in spirit but non-smooth and it
  ignores all-but-one stage, so it can't express "three mediocre stages are worse
  than one."
- **mean()** — smooth but *wrong direction*: it lets a great final stage paper
  over a broken early one, which is exactly the failure mode we want to surface.
- **product** — smooth, monotone, penalizes every weak stage, and has a clean
  probabilistic reading (probability all stages "succeed" if quality ≈ P(good)).
  And the moments of a product of independent `[0,1]` variables are exact and
  cheap, so we can moment-match back to a Beta and keep composing indefinitely.

Product it is. The honest caveat, stated in the README and here: it assumes
**independence** between stages. Correlated failures — the same underlying model
powering several stages and failing on the same input — will be *under*-estimated
by this. That's a known, deliberate simplification; a fuller model would carry a
covariance structure, at a large cost in complexity and interpretability. For a
system whose selling point is transparency, the simple conservative choice is the
right one — *as long as it's disclosed*, which it is.

---

## 5. Why calibrate scores against the prior before picking a winner?

The naive winner rule is "highest raw score." It's a trap, because the scorer is
noisy (that's the whole reason `SimulatedScorer` injects noise). Under a noisy
judge, a genuinely weak agent will *occasionally* post a high reading, and
"highest raw score" will crown it — precisely on the speculative stages where you
were hoping to do better than a coin flip.

Fusing the raw score with the agent's prior (empirical-Bayes shrinkage) fixes
this: an outlier reading from an agent we have strong reason to think is mediocre
gets pulled back toward its prior, while the same reading from a proven agent is
taken more at face value. The shrinkage strength scales with scorer reliability —
a perfect judge is trusted fully, a useless one is ignored. And it reuses the
*exact same* Beta update the bandit uses to learn, so there's one coherent notion
of "evidence" throughout.

---

## 6. Why an explicit cost model (`cost_weight`) instead of a fixed rule?

A fixed "always speculate when the top two arms are within X" rule can't express
the thing that actually determines whether speculation is worth it: **the ratio
of quality-upside to compute-cost, scaled by how much the task matters.**
Speculating three expensive frontier models on a low-stakes logging step is
obviously wrong; speculating two cheap retrievers on the user-facing final answer
is obviously right. Only an economic model captures both, and it collapses the
whole decision to a single interpretable knob (`cost_weight` = quality-units you'll
trade per compute-unit). Tuning aggressiveness becomes a one-number dial rather
than a policy rewrite.

The risk, stated plainly: get `cost_weight` wrong and you over- or
under-speculate. It's a knob, not magic. But it's the *right* knob — it's the one
a cost-conscious operator actually thinks in.

---

## 7. v0.2 — the industrial features, and why they're shaped this way

### 7.1 Latency hedging: why a *quantile* fence, not mean + kσ

The first implementation used the obvious estimator — EMA of latency plus three
EW standard deviations. It never fired. The failure is instructive: production
agent latency is **bimodal** (a fast mode plus a fat tail), and the tail
inflates both the mean and σ, hoisting the fence *above* the very stragglers it
exists to catch. The estimator self-defeats on exactly the distribution it's
for. The fix is what the hedged-request literature actually specifies: an
empirical **quantile** over a sliding window — exact, O(window) memory,
drift-adapting by construction. We kept the discovery in the docstring because
"mean + kσ intuition fails on bimodal data" is a lesson worth the space.

### 7.2 Why hedging is a separate mechanism from speculation

They look similar (both run extra branches) but answer different questions.
Speculation answers *"who will produce the better output?"* — it must launch
branches **up front**, because its value comes from comparing finished outputs.
Hedging answers *"will my chosen arm answer in time?"* — it must launch
**lazily**, because its value comes from *not* paying the duplicate cost on the
92% of calls that stay fast. Same family (spend compute to buy down
uncertainty), different uncertainty, different launch schedule. Folding them
into one mechanism would force one schedule and ruin one of the two economics.

### 7.3 Drift: why *reset* beliefs instead of decaying them

Exponential forgetting (`max_evidence`) is included and helps, but on its own
it's a compromise: decay fast enough to track drift and you're permanently
noisy; slow enough to be stable and you're slow to react. A change-point
detector breaks the trade-off — keep long, stable memory *until there is
specific evidence of a regime change*, then discard it wholesale. The reset is
also the on-thesis move: it re-creates the cold-start condition, and PRISM's
cold-start machinery (Thompson exploration + speculation) reignites
automatically. Detection and recovery are decoupled — Page–Hinkley only pulls a
trigger; the existing uncertainty machinery does all the actual adapting.

Two subtleties found while building it:

* **Anchor the reset on the *recent* mean, not the stream mean.** Page–Hinkley
  tracks the running mean of the whole stream, which at alarm time is dominated
  by the dead regime — anchoring there resets the belief optimistically high.
  The detector now keeps a 10-observation window; its mean is the only honest
  estimate of the *new* regime.
* **A quarantined arm drifts silently.** Detection needs rewards; rewards need
  the arm to be chosen. This is a real blind spot shared by every
  passive-observation scheme, and it interacts with the circuit breaker (which
  deliberately stops choosing an arm). Disclosed rather than papered over.

### 7.4 Failure semantics: failures are *evidence*, not just errors

Every failure path funnels into the same principle: an exception or timeout
becomes a **reward-0 belief update** — an outage is the strongest possible
evidence about an agent's current usefulness, and throwing it away would waste
exactly the signal that routing needs. The ladder (absorb failed branch →
failover across untried arms → `StageFailure` with the partial trace attached)
exists because each rung is cheaper than the next; most incidents should
resolve on the first rung, invisibly. The circuit breaker sits on top because a
bandit alone keeps *probing* a hard-down agent (Thompson still samples it
occasionally) — the breaker makes "stop calling it entirely, then probe on a
schedule" explicit, with exponential backoff because flapping services punish
naive fixed cooldowns.

One interaction surfaced by the demo and worth stating: with the breaker
holding an arm out, that arm's *call count* freezes — so failure windows
expressed in calls never end. Outages are wall-clock phenomena; `FlakyAgent`
grew a `clock` hook for exactly this reason, and real deployments should think
in the same terms.

### 7.5 Budgets: a governor, not a veto

The budget doesn't just refuse speculation — it degrades it in a controlled
ladder (full width → narrower → greedy-only → clean `BudgetExhausted`), and
every rung is recorded in the decision's explanation. The `reserve_fraction`
encodes the operational instinct "don't let early stages feast and starve the
pipeline": speculative *extra* compute may only spend the unreserved slice,
while baseline greedy work may dip into the reserve, because finishing
cheaply beats speculating early and not finishing at all.

### 7.6 Persistence: state is what's learned, nothing else

`save_policy` serializes belief tables, edge beliefs, and breaker state — and
deliberately *not* agents, scorers, or policy knobs. Those are code and config;
persisting them invites version skew between the saved object and the running
system. Loading tolerates roster drift (intersection semantics) because in
production the roster *will* change between save and load, and refusing to
load anything because one agent was renamed is the wrong failure mode.

---

## 8. Things deliberately left out (scope discipline)

- **A real LLM backend in the core.** Kept out so the repo runs with zero keys
  and the *ideas* stay the star. Adapters are a ~10-line `CallableAgent`.
- **Persistence / distributed execution.** The belief table is in-memory. A real
  deployment would back it with a store; the interface (`router.snapshot()` /
  reload) makes that a small addition, not a rewrite.
- **Learned schedulers / RL over the graph.** Tempting, but it would trade the
  system's transparency for marginal gains. PRISM's thesis is that *principled
  uncertainty + a hardware-inspired hedge* gets you most of the way with full
  explainability. That trade is the whole point.

---

## 9. If you're going to poke holes

Good — here's where to aim, honestly:

1. **The benchmark gains look modest.** They are, *in this scenario*, because
   Thompson routing is already strong and the simulated scorer is only 0.85
   reliable. Speculation's value climbs with judge reliability, near-tie
   frequency, and stakes. The mechanism is designed to *do nothing* when it can't
   help — which is a feature, but it does mean the headline number is scenario
   dependent. Don't trust one number; run `prism benchmark` under your own
   competence profiles.
2. **Independence in uncertainty composition** (§4) is the softest assumption.
3. **`scorer_efficiency` is a scalar** standing in for a genuinely complex thing
   (how often your judge picks the true best of a branch set). A real deployment
   should measure it, not guess it.

None of these sink the idea; all of them are disclosed in code and docs. That
combination — a genuinely novel mechanism, implemented cleanly, and *honest about
its edges* — is what this project is trying to demonstrate.
