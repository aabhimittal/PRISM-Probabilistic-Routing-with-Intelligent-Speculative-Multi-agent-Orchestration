"""08 · Latency hedging — speculation on the *time* axis.

PRISM's core mechanism handles *quality* uncertainty. This one handles the
uncertainty that dominates production LLM systems: **latency**. Agent calls are
heavy-tailed — p50 is fine, p99 is a disaster — and a pipeline is as slow as
its slowest stage.

Borrowed idea #2, from Dean & Barroso's "The Tail at Scale": send the request
to one replica; if it hasn't answered by ~the p95 latency, duplicate it to
another replica and take the first answer. PRISM generalises the "replica" to a
*different agent*: commit the best arm, but if it blows through its learned
latency fence, launch the runner-up and race them. It's lazy speculation — the
second branch materialises only when the first one stalls — and the straggler's
result still updates the bandit when it lands ("squashed work still teaches",
applied to time).

This demo uses agents that REALLY sleep, so the latencies below are measured
wall-clock, not simulated numbers.

Run:  python examples/08_latency_hedging.py       (~10 seconds)
"""

import time

from prism import (
    HedgePolicy,
    LatencyTracker,
    Orchestrator,
    SimulatedAgent,
    SpeculationPolicy,
    TaskGraph,
)

RUNS = 60


def build(hedged: bool):
    graph = TaskGraph.linear("answer")
    # 'sharp' is the better agent but has a nasty tail: 8% of calls take 300ms
    # instead of ~20ms (rate limits, retries, a bad shard — pick your poison).
    sharp = SimulatedAgent("sharp", {"answer": (0.85, 0.08)}, base_latency=0.020,
                           simulate_latency=True, tail=(0.08, 0.300)).seed(5)
    quick = SimulatedAgent("quick", {"answer": (0.78, 0.08)}, base_latency=0.010,
                           simulate_latency=True).seed(5)
    hedge = HedgePolicy(
        tracker=LatencyTracker(quantile=0.90),  # fence ≈ recent p90 latency
        fallback_fence=0.050,                   # until the profile warms up
    ) if hedged else None
    # Speculation disabled on purpose: this isolates hedging as the only
    # multi-branch mechanism, so the latency win is unambiguously *its* doing.
    return Orchestrator.build(
        graph, {"answer": [sharp, quick]}, seed=5,
        policy=SpeculationPolicy(max_branches=1), hedge=hedge,
    )


def bench(hedged: bool):
    orch = build(hedged)
    lats, hedges, quality = [], 0, 0.0
    for i in range(RUNS):
        t0 = time.perf_counter()
        tr = orch.run(f"q{i}", stakes=0.6)
        lats.append(time.perf_counter() - t0)
        hedges += sum(1 for e in tr.events if "hedge fired" in e)
        quality += tr.mean_realized_quality
    time.sleep(0.5)          # let straggler results land and teach the bandit
    lats.sort()
    q = lambda p: lats[min(int(p * len(lats)), len(lats) - 1)] * 1000
    return {
        "p50": q(0.50), "p90": q(0.90), "p99": q(0.99), "max": lats[-1] * 1000,
        "hedges": hedges, "quality": quality / RUNS,
        "late": len(orch.late_events),
    }


print(f"{RUNS} runs each, one stage, agents actually sleeping (wall-clock ms):\n")
plain = bench(hedged=False)
hedged = bench(hedged=True)

print(f"{'':14}{'p50':>7}{'p90':>7}{'p99':>7}{'max':>7}   hedges  avg quality")
print(f"{'no hedging':<14}{plain['p50']:>7.0f}{plain['p90']:>7.0f}"
      f"{plain['p99']:>7.0f}{plain['max']:>7.0f}   {plain['hedges']:>6}  "
      f"{plain['quality']:.3f}")
print(f"{'hedged':<14}{hedged['p50']:>7.0f}{hedged['p90']:>7.0f}"
      f"{hedged['p99']:>7.0f}{hedged['max']:>7.0f}   {hedged['hedges']:>6}  "
      f"{hedged['quality']:.3f}")

print(f"""
The tail collapses (p99 {plain['p99']:.0f}ms → {hedged['p99']:.0f}ms) while
median and quality hold steady — the hedge costs nothing on the {100 - 8}% of
calls that stay fast, and only duplicates work inside the tail.
{hedged['late']} straggler results landed late and were still folded into the
bandit's beliefs (see orch.late_events) — no observation is wasted.""")
