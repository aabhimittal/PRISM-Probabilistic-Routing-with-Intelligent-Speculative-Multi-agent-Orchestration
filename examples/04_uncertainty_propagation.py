"""04 · Compound epistemic uncertainty — why a 4-stage answer is less certain
than any single stage.

Most frameworks report a per-step confidence and quietly forget it at the next
step. PRISM composes the belief at every hop (see prism.uncertainty.propagate):
end-to-end quality is modelled as the PRODUCT of per-stage qualities, so the
final confidence is honestly gated by the whole chain — and its uncertainty
*compounds* rather than resetting.

Run:  python examples/04_uncertainty_propagation.py
"""

from prism import propagate
from prism.uncertainty import from_mean_std
from prism.scenarios import research_pipeline

# --- Part A: the math, in isolation ---------------------------------------
print("Part A — composing four fairly-confident stages (each ~0.85 ± 0.08):\n")

compound = None
for k in range(1, 5):
    s = from_mean_std(0.85, 0.08)
    compound = s if compound is None else propagate(compound, s)
    print(f"  after {k} stage(s): mean={compound.mean:.3f}  std={compound.std:.3f}  "
          f"(a single stage was mean=0.850, std=0.080)")

print(
    "\n  Each stage is individually strong, but four of them chained give a mean\n"
    "  of ~0.52 — because quality multiplies. And the std does NOT shrink toward\n"
    "  zero; the uncertainty of the composite is real and is surfaced, not hidden.\n"
)

# --- Part B: the same effect inside a real run ----------------------------
print("Part B — watch confidence decay stage-by-stage inside an actual pipeline:\n")
orch = research_pipeline(seed=2)
for _ in range(30):
    orch.run("query", stakes=0.7)  # warm up

trace = orch.run("Explain CRISPR base editing.", stakes=0.9)
for i, st in enumerate(trace.stages):
    b = st.propagated_belief
    bar = "█" * int(b.mean * 30)
    print(f"  stage {i} [{st.task_type:<10}] compound mean={b.mean:.3f} "
          f"± {b.std:.3f}  {bar}")

fb = trace.final_belief
print(f"\n  FINAL end-to-end confidence: {fb.mean:.3f} ± {fb.std:.3f} "
      f"(evidence n={fb.evidence:.1f})")
lo, hi = fb.credible_interval(0.9)
print(f"  90% credible interval on answer quality: [{lo:.2f}, {hi:.2f}]")
print(
    "\n  This is the number you'd threshold on before shipping an answer to a\n"
    "  user — or use to decide whether to escalate to a human."
)
