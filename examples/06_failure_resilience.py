"""06 · Industrial resilience — failures, failover, and the circuit breaker.

Real agents fail: APIs 500, models time out, providers have outages. This
example stages an outage and shows PRISM's full resilience ladder:

  1. a failed *branch* is absorbed — survivors compete, the failure is scored
     as a zero reward so the bandit learns from it;
  2. a failed *stage* (every branch down) triggers automatic failover to the
     arms not yet tried;
  3. an agent that keeps failing trips a **circuit breaker** — it's removed
     from the candidate set entirely (no calls, no cost, no latency), then
     probed after a cooldown and re-admitted once it heals.

Run:  python examples/06_failure_resilience.py
"""

from prism import (
    CircuitBreaker,
    FlakyAgent,
    Orchestrator,
    SimulatedAgent,
    TaskGraph,
)

graph = TaskGraph.linear("answer")

# 'premium' is the best agent — when it's up. We give it a hard outage during
# runs 5..21 (keyed to the orchestrator's run counter via `clock`, because an
# outage is a property of the world, not of how often we happened to call the
# agent). 'steady' is worse but dependable.
premium_inner = SimulatedAgent("premium", {"answer": (0.88, 0.08)}, cost=2.0).seed(1)
premium = FlakyAgent(premium_inner, fail_between=(5, 22), seed=1)
steady = SimulatedAgent("steady", {"answer": (0.74, 0.10)}, cost=1.0).seed(1)

orch = Orchestrator.build(
    graph,
    {"answer": [premium, steady]},
    seed=1,
    breaker=CircuitBreaker(failure_threshold=3, base_cooldown=8),
)
premium.clock = lambda: orch._run_counter   # outage measured in runs

print("running 60 tasks through an outage window (premium is down runs 5-21):\n")
seen: set[str] = set()
for i in range(60):
    trace = orch.run(f"task {i}", stakes=0.8)
    for ev in trace.events:
        if ev not in seen:            # print each distinct event once
            seen.add(ev)
            print(f"  run {i:>2}: {ev}")

print("\nwhat the bandit believes now (failures taught it, breaker protected it):")
for name, (mean, std) in sorted(orch.router.snapshot()["answer"].items()):
    print(f"    {name:<8} {mean:.2f} ± {std:.2f}")

print(
    "\nReading the event stream top to bottom you can see the whole arc:\n"
    "failures absorbed → breaker OPEN (premium quarantined, zero wasted calls)\n"
    "→ cooldown → half-open probe → probe succeeds → breaker CLOSED → premium\n"
    "earns its way back via ordinary bandit updates. No task ever failed."
)
