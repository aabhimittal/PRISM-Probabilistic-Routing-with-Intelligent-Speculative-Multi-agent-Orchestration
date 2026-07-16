"""Calibrated winner selection.

When several speculative branches come back, we must pick a winner. The naive
approach — "take the highest raw score" — is a trap: the scorer is noisy, and a
weak agent can get a lucky high reading. PRISM instead treats each branch's raw
score as *evidence* and combines it with the *prior* belief about that agent,
producing a **posterior** quality estimate. The winner is the argmax posterior.

This is shrinkage / empirical-Bayes in miniature: an outlier score from an agent
we have strong reason to think is mediocre gets pulled back toward its prior,
while the same score from a proven agent is taken more at face value. It makes
selection robust to exactly the judge noise that :class:`SimulatedScorer`
injects on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from .uncertainty import Belief


@dataclass
class CalibratedScore:
    agent_name: str
    raw_score: float          # what the scorer said
    posterior_mean: float     # score fused with the prior belief
    prior_mean: float


def calibrate(agent_name: str, raw_score: float, prior: Belief,
              scorer_reliability: float) -> CalibratedScore:
    """Fuse a single raw score with the agent's prior belief.

    We model the raw score as one soft observation and update the prior with it,
    but weighted by how much we trust the scorer. A flawless scorer
    (reliability=1) contributes a full unit of evidence; a useless one
    contributes ~nothing, so the posterior stays at the prior. Concretely we add
    ``w`` pseudo-observations, where ``w`` scales with reliability::

        posterior = prior.updated(raw_score, weight = w)

    and read off the posterior mean. This is the same conjugate update the
    bandit uses to *learn*, reused here to *decide* — one coherent mechanism.
    """
    # Map reliability in [0,1] to an evidence weight. Reliability 1 -> weight ~4
    # (a strong single vote), reliability 0.5 -> ~1, reliability 0 -> ~0.
    weight = max(0.0, 4.0 * (scorer_reliability - 0.25) / 0.75)
    posterior = prior.updated(raw_score, weight=weight)
    return CalibratedScore(
        agent_name=agent_name,
        raw_score=raw_score,
        posterior_mean=posterior.mean,
        prior_mean=prior.mean,
    )


def select_winner(scored: list[CalibratedScore]) -> CalibratedScore:
    """Pick the branch with the highest *posterior* quality (calibrated), with
    a deterministic tie-break on agent name for reproducibility."""
    return max(scored, key=lambda s: (s.posterior_mean, s.raw_score, -_ord(s.agent_name)))


def _ord(name: str) -> int:
    # Stable numeric key from a name, only used for deterministic tie-breaking.
    return sum(ord(c) for c in name)
