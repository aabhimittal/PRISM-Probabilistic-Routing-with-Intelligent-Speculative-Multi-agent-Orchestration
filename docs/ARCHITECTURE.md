# PRISM Architecture

This document is the deep dive: the data model, the control flow, and the exact
role each module plays. Read the [README](../README.md) first for the intuition.

## Mental model

PRISM is a **stage machine over a probabilistic graph**. A task enters at the
graph's entry node and is pushed through stages. At each stage the orchestrator
answers three questions in order:

1. **Who could do this?** — the candidate agents (bandit arms) registered for
   this task type.
2. **Should I commit to one, or hedge across several?** — the speculation
   decision, made on an economic (value-of-information) basis.
3. **Which output do I keep, and what did I learn?** — calibrated winner
   selection, then a Bayesian update of *every* branch that ran.

The winner's output becomes the next stage's input; the winner's *belief* is
composed into a running compound belief. When the graph reaches a terminal node,
the orchestrator returns a `CausalTrace` — the entire decision history.

## The data model

```
Task            task_type · payload · stakes · context
  │
  ▼  (orchestrator picks agents for task_type)
Belief[]        one Beta(α,β) per candidate agent   ← the learned policy
  │
  ▼  SpeculationPolicy.decide(beliefs, costs, stakes)
SpeculationDecision   mode · branches · p_best · VOI · cost · explanation
  │
  ▼  SpeculativeExecutor.run_stage(...)
StageResult     winner · outcomes[] · latency · cost
  │             (each BranchOutcome: agent · output · score · prior/posterior · squashed?)
  ▼  router.update(...) for EVERY outcome
Belief[]'       sharpened posteriors
  │
  ▼  propagate(compound, winner_belief)
CausalTrace     grows one StageTrace; final_belief = compound epistemic uncertainty
```

Every object here is a plain, serializable dataclass. That is a deliberate
constraint: it's what makes `CausalTrace.to_json()` able to snapshot a full run,
and what lets beliefs be recorded *at decision time* without later mutation
aliasing the record (beliefs are immutable — `updated()` returns a new one).

## Module responsibilities

### `uncertainty.py` — the belief calculus
The foundation. A `Belief` is `Beta(α, β)` over a quality in `[0,1]`. It provides
summary stats (`mean`, `variance`, `entropy`, `evidence`), a closed-form Bayesian
`updated(reward, weight)`, and `sample()` (the primitive behind Thompson
sampling). Module-level helpers:
- `from_mean_std` — moment-match a Beta from a target mean/std (seeding priors).
- `propagate` — compose two stage beliefs into the belief about their chained
  output (product of qualities, moment-matched back to Beta).
- `probability_best` — Monte-Carlo `P(arm is best)` across arms.
- `selection_entropy` — how torn the router is, as a scalar.

Kept numpy/scipy-free on purpose (`_digamma`, `_log_beta_fn` are hand-rolled) so
the core installs with zero dependencies.

### `routing.py` — the learned policy
`BanditRouter` owns the belief table: `(task_type, agent) -> Belief`. This table
**is** the policy. It exposes `thompson_select` / `greedy_select`, `p_best`, and
`update(...)` which returns `(before, after)` so callers can record the exact
belief delta into the trace. `snapshot()` dumps the whole policy as plain numbers
for plotting convergence.

### `speculation.py` — the headline mechanism
Two pieces:
- `SpeculationPolicy.decide(...)` — the economic brain. Computes `p_best`, picks a
  viable branch set, estimates `R_greedy` and `VOI` in one Monte-Carlo pass, and
  returns a `SpeculationDecision` with a human-readable rationale. When it decides
  *not* to speculate, it still routes the single arm via a **Thompson draw** so
  under-explored arms keep getting evidence (avoiding the bandit cold-start trap).
- `SpeculativeExecutor.run_stage(...)` — the mechanism. Runs the branch set in
  parallel (`ThreadPoolExecutor`), scores + calibrates each output, commits the
  winner, marks losers `squashed`, and invokes an optional `on_squash`
  compensation hook for external side effects.

### `scoring.py` — calibrated selection
`calibrate(agent, raw_score, prior, reliability)` treats a raw score as one soft
observation and fuses it with the agent's prior belief, weighted by scorer
reliability. `select_winner` takes the argmax **posterior** (not raw). This is
empirical-Bayes shrinkage: it stops a noisy judge from crowning a weak agent that
got a lucky reading.

### `graph.py` — the probabilistic task graph
`TaskGraph` holds weighted, optionally-conditional edges between task types.
`next_node(...)` samples the successor by `weight × edge_belief.mean` among
edges whose predicate passes — so traversal is itself probabilistic and the edge
beliefs are learnable via `update_edge(...)`. `TaskGraph.linear(...)` builds the
common straight-line pipeline.

### `orchestrator.py` — the conductor
`Orchestrator.run(payload, stakes)` implements the per-stage loop above and
assembles the `CausalTrace`. `Orchestrator.build(...)` is the batteries-included
constructor that wires router + policy + executor + scorer from an
`agents_by_stage` map. Deterministic given a seed.

### `trace.py` — provenance
`CausalTrace` / `StageTrace` / `RoutingRationale` capture everything. `explain()`
renders the annotated decision tree you paste into a PR; `to_json()` serializes
it for storage, diffing, or handing to an auditing agent. `CausalTrace.events`
carries the operational log: breaker trips, failovers, hedges, drift alarms.

### `resilience.py`, `drift.py`, `hedging.py`, `budget.py`, `errors.py` — the industrial layer
Circuit breaking, change-point detection with uncertainty resurrection,
tail-latency hedging, compute budgets, and the exception hierarchy. Covered in
detail in "The industrial layer" below.

## Control flow (one stage, precise)

```
def _run_stage(task):
    beliefs = router.beliefs(task.task_type)          # current policy for this stage
    costs   = {name: agent.cost for ...}

    decision      = policy.decide(beliefs, costs, task.stakes)   # greedy? speculate?
    branch_agents = [agents[n] for n in decision.branches]
    result        = executor.run_stage(task, branch_agents, priors)  # parallel + score + pick

    for outcome in result.outcomes:                    # learn from ALL branches
        before, after = router.update(task.task_type, outcome.agent_name, outcome.score)
        # record before/after means into the outcome for the trace

    return StageTrace(...), result.winner.output, winner_belief_after
```

Then the caller (`run`) composes `compound = propagate(compound, winner_belief)`,
routes to the next node via the graph, and reinforces the taken edge by the
stage's realized quality.

## Concurrency

Speculative branches run on a `ThreadPoolExecutor`. Real agents are I/O-bound
(network LLM calls), so threads give true wall-clock overlap and the GIL is a
non-issue — the Python code between calls is negligible. Wall-clock latency for a
speculative stage is therefore the **slowest single branch**, not the sum, which
is the entire point: speculation buys quality without a latency penalty
proportional to width. Determinism is preserved because scoring/selection happens
after the barrier, in a fixed order.

## The industrial layer (v0.2)

Each of these is opt-in via `Orchestrator.build(...)`; leaving them off yields
the simple, deterministic v0.1 behaviour.

### Failure path, precisely

```
branch raises/times out ──► BranchOutcome(failed=True, score=0)  ─┐
                            (winner chosen among survivors)       │ reward-0
ALL branches fail ────────► AllBranchesFailed ─► failover over    ├─ updates +
                            untried arms (one round)              │ breaker
failover fails too ───────► StageFailure(.trace = partial trace)  ┘ records
```

Failures are evidence: every failed call is a reward-0 bandit update *and* a
`CircuitBreaker.record()`. The breaker's state machine per (stage, agent):
CLOSED → (N consecutive failures) → OPEN for `cooldown` runs → HALF-OPEN probe
→ success closes / failure re-opens with doubled cooldown (capped). A roster
that is entirely OPEN is overridden — degraded routing beats refusing to route.

### Hedged execution path

`run_stage_hedged` (executor) is used when the policy commits a single arm and
a `HedgePolicy` is armed: run primary → `wait(hedge_after)` → if still pending,
launch backup → commit **first successful** result → straggler's completion
fires `on_late`, which scores it and updates the bandit from a worker thread
(router updates are lock-serialized). The fence comes from `LatencyTracker`, a
sliding-window empirical quantile per (stage, agent) — see design notes §7.1
for why it is *not* mean + kσ.

### Drift path

`BanditRouter.update()` → `DriftMonitor.observe()` → Page–Hinkley per arm → on
alarm, the arm's belief is *replaced* with a wide Beta anchored on the
detector's recent-window mean (uncertainty resurrection). The orchestrator
drains `DriftMonitor` events into `CausalTrace.events` each stage. `max_evidence`
(exponential forgetting via rescaling α, β before each update) complements it:
the cap keeps the policy movable at all; the detector makes big shifts fast.

### Budget path

`Orchestrator.run(budget=...)` gates each stage on `can_afford_base(cheapest)`
(raising `BudgetExhausted` with the partial trace when even greedy work is
unaffordable) and passes the budget into `SpeculationPolicy.decide`, which
narrows or suppresses the branch set when `can_afford_extra` fails — recording
the pressure in the decision's explanation. Stage costs are spent after
execution; `reserve_fraction` shields headroom for later stages.

### Persistence

`Orchestrator.save_policy(path)` → one JSON document: router `state_dict`
(α, β per arm), graph edge beliefs, breaker state, run counter.
`load_policy` restores the intersection with the current roster.

## Extension points

| You want to... | Do this |
|----------------|---------|
| Use a real LLM | Wrap it in `CallableAgent` (or implement the `Agent` protocol) |
| Use a real verifier | Implement the `Scorer` protocol (LLM-judge, tests, reward model) |
| Change speculation aggressiveness | Tune `SpeculationPolicy.cost_weight` / `max_branches` |
| Seed domain knowledge | Pass `priors` to `Orchestrator.build` |
| Branch on content | Add conditional edges with predicates in `TaskGraph` |
| Undo side effects of squashed branches | Pass `on_squash` to the executor |
| Survive flaky agents | `breaker=CircuitBreaker(...)`, `timeout=...` in `build` |
| Collapse the latency tail | `hedge=HedgePolicy(...)` in `build` |
| Track non-stationary agents | `drift=DriftMonitor(...)`, `max_evidence=...` in `build` |
| Cap spend | `orch.run(..., budget=ComputeBudget(total=...))` |
| Keep learning across restarts | `orch.save_policy(p)` / `orch.load_policy(p)` |
