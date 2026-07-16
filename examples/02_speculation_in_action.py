"""02 · Speculation in action — watch the greedy-vs-speculate decision, live.

This example isolates a SINGLE decision so you can see the economic model work.
We hand the policy two hand-crafted belief states and print exactly why it does
or doesn't speculate.

Run:  python examples/02_speculation_in_action.py
"""

import random

from prism import Belief, SpeculationPolicy

policy = SpeculationPolicy(max_branches=3, cost_weight=0.03,
                           scorer_efficiency=0.9, rng=random.Random(0))
costs = {"agent-A": 1.0, "agent-B": 1.0, "agent-C": 1.0}


def show(title, beliefs, stakes):
    d = policy.decide(beliefs, costs, stakes)
    print(f"\n### {title}   (stakes={stakes})")
    for name, b in beliefs.items():
        print(f"    {name}: mean={b.mean:.2f}  std={b.std:.2f}  (n={b.evidence:.0f})")
    print(f"    p_best = {{" + ", ".join(f'{k}:{v:.2f}' for k, v in d.p_best.items()) + "}")
    print(f"    DECISION: {d.mode.value.upper()}  branches={d.branches}")
    print(f"    → {d.explanation}")


# CASE 1: one arm clearly dominates. Speculation would be wasted compute.
show(
    "Clear winner — commit greedily",
    {
        "agent-A": Belief(45, 8),    # mean ~0.85, tight
        "agent-B": Belief(20, 25),   # mean ~0.44
        "agent-C": Belief(15, 30),   # mean ~0.33
    },
    stakes=0.9,
)

# CASE 2: two strong arms with genuine EPISTEMIC uncertainty — similar means
# (~0.75) but *wide, overlapping* beliefs (std ~0.12, few observations). We
# don't yet know which is actually better, and either could turn out much
# better than its mean. On a high-stakes task, running both and keeping the
# winner is worth the compute.
#
# Subtlety worth internalising: it is NOT "close means" that make speculation
# pay — it's uncertainty about which is better *with real upside*. Two arms
# both tightly pinned at 0.77 (Belief(40,12)) would produce ~zero value: you're
# confident they're equal, so it doesn't matter which you pick. Compare Case 2b.
show(
    "Near-tie with WIDE beliefs, high stakes — speculate",
    {
        "agent-A": Belief(9, 3),     # mean ~0.75, std ~0.12  (only ~12 obs)
        "agent-B": Belief(8.5, 3),   # mean ~0.74, std ~0.12  (overlaps A heavily)
        "agent-C": Belief(10, 20),   # mean ~0.33
    },
    stakes=0.95,
)

# CASE 2b: the SAME means, but now TIGHT beliefs (lots of evidence). We're
# confident both are ~0.75, so which one we pick barely matters → don't
# speculate. This is the distinction Case 2 is making concrete.
show(
    "Near-tie with TIGHT beliefs — indifferent, so commit",
    {
        "agent-A": Belief(90, 30),   # mean ~0.75, std ~0.04  (well characterised)
        "agent-B": Belief(85, 30),   # mean ~0.74, std ~0.04
        "agent-C": Belief(10, 20),
    },
    stakes=0.95,
)

# CASE 3: the wide near-tie again, but now the task barely matters (low stakes).
# The value of getting it exactly right no longer beats the extra compute.
show(
    "Wide near-tie, low stakes — don't bother",
    {
        "agent-A": Belief(9, 3),
        "agent-B": Belief(8.5, 3),
        "agent-C": Belief(10, 20),
    },
    stakes=0.15,
)

# CASE 4: total cold-start. Everything is Beta(1,1) — maximal uncertainty.
# High stakes + cheap agents → speculate broadly to learn fast.
show(
    "Cold start — explore under maximal uncertainty",
    {
        "agent-A": Belief(1, 1),
        "agent-B": Belief(1, 1),
        "agent-C": Belief(1, 1),
    },
    stakes=0.9,
)

print("\nTakeaway: PRISM speculates precisely when uncertainty is high AND the "
      "stakes justify the compute — and settles into cheap single-arm routing "
      "once it knows what works.")
