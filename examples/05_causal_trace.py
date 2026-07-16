"""05 · Causal tracing — every output carries its full decision provenance.

The trace is a first-class, serializable object. You can render it for humans,
dump it to JSON for storage/diffing, or hand it to another agent to audit the
decision. This example shows all three.

Run:  python examples/05_causal_trace.py
"""

import json

from prism.scenarios import research_pipeline

orch = research_pipeline(seed=8)
for _ in range(20):
    orch.run("warm up", stakes=0.7)

trace = orch.run("Draft a launch email for a new database product.", stakes=0.9)

# 1. Human-readable decision tree.
print("=" * 70)
print("HUMAN VIEW")
print("=" * 70)
print(trace.explain())

# 2. Machine-readable provenance (excerpt) — diffable, storable, auditable.
print("\n" + "=" * 70)
print("MACHINE VIEW (stage 1 provenance, JSON)")
print("=" * 70)
d = trace.to_dict()
stage1 = d["stages"][1]
print(json.dumps({
    "task_type": stage1["task_type"],
    "mode": stage1["rationale"]["mode"],
    "p_best": {k: round(v, 3) for k, v in stage1["rationale"]["p_best"].items()},
    "value_of_speculation": round(stage1["rationale"]["value_of_speculation"], 4),
    "branches_run": stage1["rationale"]["branches"],
    "winner": stage1["winner"],
    "per_branch": [
        {"agent": o["agent_name"], "score": round(o["score"], 3),
         "winner": o["is_winner"], "squashed": o["squashed"]}
        for o in stage1["outcomes"]
    ],
}, indent=2))

# 3. The kind of question a trace lets you answer AFTER the fact.
print("\n" + "=" * 70)
print("QUERYING THE TRACE")
print("=" * 70)
spec_stages = [s for s in trace.stages if s.rationale.mode.value == "speculative"]
squashed = [o.agent_name for s in trace.stages for o in s.outcomes if o.squashed]
print(f"  · stages that speculated: {[s.task_type for s in spec_stages]}")
print(f"  · agents whose work was squashed (ran but discarded): {squashed}")
print(f"  · total compute spent: {trace.total_cost:.1f} units "
      f"({trace.speculations} speculations)")
print(f"  · final answer confidence: {trace.final_belief.mean:.3f} "
      f"± {trace.final_belief.std:.3f}")
print("\n  Full trace serializes to JSON via trace.to_json() — store it next to "
      "the\n  output and you can always reconstruct *why* the system answered "
      "the way it did.")
