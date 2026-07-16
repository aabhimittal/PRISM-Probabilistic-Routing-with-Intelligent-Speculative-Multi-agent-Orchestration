"""``prism`` command-line entry point.

Usage
-----
    prism demo            # run one task, print its full causal trace
    prism learn [N]       # run N tasks, show the bandit converging on the truth
    prism benchmark [N]   # compare PRISM vs greedy-only routing on quality/cost

Everything is simulated — no API keys, no network.
"""

from __future__ import annotations

import sys

from .scenarios import ground_truth_best, research_pipeline


def _demo() -> None:
    orch = research_pipeline(seed=3)
    # Warm the policy up a little so speculation has non-trivial beliefs to act on.
    for _ in range(12):
        orch.run("Summarize the causes of the 2008 financial crisis.", stakes=0.7)
    trace = orch.run("Explain how mRNA vaccines work.", stakes=0.9)
    print(trace.explain())


def _learn(n: int) -> None:
    orch = research_pipeline(seed=1)
    truth = ground_truth_best()
    print(f"ground-truth best per stage: {truth}\n")
    checkpoints = sorted({1, 5, 10, 25, 50, n} & set(range(1, n + 1)))
    for i in range(1, n + 1):
        orch.run("query", stakes=0.6)
        if i in checkpoints:
            snap = orch.router.snapshot()
            agree = sum(
                1 for stage in truth
                if max(snap[stage], key=lambda k: snap[stage][k][0]) == truth[stage]
            )
            print(f"after {i:>4} tasks:  router agrees with oracle on "
                  f"{agree}/{len(truth)} stages")
            for stage in truth:
                ranked = sorted(snap[stage].items(), key=lambda kv: -kv[1][0])
                lead = ", ".join(f"{k}={m:.2f}±{s:.2f}" for k, (m, s) in ranked[:3])
                print(f"    {stage:<11} {lead}")
            print()


def _benchmark(n: int) -> None:
    from .speculation import SpeculationPolicy

    def run_variant(label: str, policy) -> None:
        orch = research_pipeline(seed=2, policy=policy)
        q_sum = c_sum = spec = 0.0
        for _ in range(n):
            tr = orch.run("query", stakes=0.8)
            q_sum += tr.mean_realized_quality   # ground-truth output quality
            c_sum += tr.total_cost
            spec += tr.speculations
        print(f"  {label:<22} realized_quality={q_sum / n:.3f}  "
              f"avg_cost={c_sum / n:5.2f}  speculations/run={spec / n:.2f}")

    print(f"benchmark over {n} tasks — realized_quality is the winners' TRUE\n"
          f"mean quality (higher better); cost is compute units (lower better):\n")
    # Greedy-only: cap branches at 1 so the policy can never speculate.
    run_variant("greedy-only", SpeculationPolicy(max_branches=1))
    run_variant("PRISM (balanced)", SpeculationPolicy(max_branches=3, cost_weight=0.03))
    run_variant("PRISM (aggressive)", SpeculationPolicy(max_branches=3, cost_weight=0.01))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "demo"
    arg = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else None
    if cmd == "demo":
        _demo()
    elif cmd == "learn":
        _learn(arg or 50)
    elif cmd == "benchmark":
        _benchmark(arg or 200)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
