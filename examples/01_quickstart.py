"""01 · Quickstart — build a pipeline in ~15 lines and read its causal trace.

Run:  python examples/01_quickstart.py
No API keys, no network — everything is simulated and reproducible.
"""

from prism import Orchestrator, SimulatedAgent, SimulatedScorer, TaskGraph

# 1. Define the pipeline shape: a straight-line graph of task types (stages).
graph = TaskGraph.linear("draft", "polish")

# 2. Give each stage a roster of candidate agents. Each SimulatedAgent has a
#    hidden competence per task type: (mean_quality, spread). PRISM does NOT see
#    these numbers — it has to *learn* them from scored outcomes.
agents = {
    "draft": [
        SimulatedAgent("fast-writer", {"draft": (0.72, 0.15)}, cost=0.5),
        SimulatedAgent("careful-writer", {"draft": (0.80, 0.12)}, cost=2.0),
    ],
    "polish": [
        SimulatedAgent("grammar-bot", {"polish": (0.75, 0.13)}, cost=0.5),
        SimulatedAgent("style-llm", {"polish": (0.83, 0.11)}, cost=2.0),
    ],
}

# 3. Wire it up. The scorer stands in for your verifier / LLM-judge.
orch = Orchestrator.build(graph, agents, scorer=SimulatedScorer(reliability=0.9), seed=0)

# 4. Run a few tasks so the bandit gathers evidence, then trace one.
for _ in range(15):
    orch.run("Write a paragraph about tardigrades.", stakes=0.6)

trace = orch.run("Write a paragraph about tardigrades.", stakes=0.9)
print(trace.explain())

print("\nWhat PRISM learned (posterior mean ± std per agent):")
for stage, arms in orch.router.snapshot().items():
    ranked = sorted(arms.items(), key=lambda kv: -kv[1][0])
    print(f"  {stage:<7}", ", ".join(f"{n}={m:.2f}±{s:.2f}" for n, (m, s) in ranked))
