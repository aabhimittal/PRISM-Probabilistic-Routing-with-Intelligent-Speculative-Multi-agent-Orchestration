"""03 · Online bandit routing — the policy learns, and speculation self-tapers.

Two things to watch as tasks stream in:
  1. The router's posterior for each agent converges toward its (hidden) true
     competence, and its top pick per stage locks onto the ground-truth best.
  2. Speculation is frequent early (high uncertainty ⇒ high value of information)
     and fades as the beliefs sharpen — PRISM stops paying for hedges it no
     longer needs. This is the whole thesis in one curve.

Run:  python examples/03_bandit_convergence.py
"""

from prism.scenarios import ground_truth_best, research_pipeline

orch = research_pipeline(seed=5)
truth = ground_truth_best()

print("hidden ground-truth best agent per stage:")
for stage, name in truth.items():
    print(f"    {stage:<11} -> {name}")
print()

N = 400
window = []
print(f"{'tasks':>6} | {'oracle agree':>12} | {'spec rate (last 40)':>20} | "
      f"{'realized quality':>16}")
print("-" * 66)

recent_q = []
for i in range(1, N + 1):
    tr = orch.run("query", stakes=0.8)
    window.append(tr.speculations / max(len(tr.stages), 1))
    recent_q.append(tr.mean_realized_quality)
    window = window[-40:]
    recent_q = recent_q[-40:]

    if i in (5, 10, 25, 50, 100, 200, 400):
        snap = orch.router.snapshot()
        agree = sum(
            1 for s in truth
            if max(snap[s], key=lambda k: snap[s][k][0]) == truth[s]
        )
        spec_rate = sum(window) / len(window)
        print(f"{i:>6} | {agree:>10}/4 | {spec_rate:>20.2f} | "
              f"{sum(recent_q) / len(recent_q):>16.3f}")

print("\nfinal learned posteriors:")
for stage, arms in orch.router.snapshot().items():
    ranked = sorted(arms.items(), key=lambda kv: -kv[1][0])
    star = lambda n: " ★" if n == truth[stage] else "  "
    print(f"  {stage}:")
    for n, (m, s) in ranked:
        print(f"      {star(n)} {n:<17} {m:.3f} ± {s:.3f}")

print("\nNote how 'spec rate' starts high and decays toward ~0 as the router "
      "grows confident — speculation was buying information, and once bought it "
      "stops paying for it.")
