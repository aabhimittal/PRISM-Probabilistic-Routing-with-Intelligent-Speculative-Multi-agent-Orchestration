"""Uncertainty primitives for PRISM.

Everything in PRISM that has to reason "how good is this, and how *sure* am I?"
is expressed as a :class:`Belief` — a Beta distribution over a quality value in
``[0, 1]``.

Why Beta?
---------
1. **Conjugacy.** The Beta distribution is the conjugate prior of the Bernoulli
   / Binomial likelihood. Every time an agent produces an output that we score,
   we get a (soft) success/failure signal, and updating the posterior is just
   ``alpha += reward`` / ``beta += (1 - reward)``. No gradient steps, no
   retraining — pure closed-form online learning. That is exactly what a
   routing policy needs to adapt on the fly.

2. **It carries its own uncertainty.** The *mean* ``alpha / (alpha + beta)`` is
   our point estimate of quality; the *variance* shrinks as ``alpha + beta``
   (the "pseudo-count" of evidence) grows. A brand-new agent we've never run is
   ``Beta(1, 1)`` — mean 0.5, maximal variance — i.e. "could be anything." An
   agent we've run 500 times has a razor-thin posterior. This epistemic vs.
   aleatoric distinction is *the* thing most orchestration frameworks throw
   away, and it is what lets PRISM decide when it is worth speculating.

3. **It composes.** When agent B consumes agent A's output, the end-to-end
   quality is bounded by both. We approximate the product of two independent
   Beta variables and moment-match back to a Beta, giving *compound epistemic
   uncertainty* that grows sensibly along a chain (see :func:`propagate`).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# A floor we add to variance estimates so moment matching never blows up when a
# distribution collapses to (almost) a point mass.
_EPS = 1e-9


@dataclass(frozen=True)
class Belief:
    """A Beta(alpha, beta) belief over a latent quality in ``[0, 1]``.

    Immutable on purpose: updates return a *new* ``Belief`` so a causal trace
    can snapshot the exact belief state at the moment a decision was made,
    without worrying about later mutation aliasing the record.
    """

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"Beta parameters must be > 0, got ({self.alpha}, {self.beta})")

    # --- summary statistics -------------------------------------------------

    @property
    def mean(self) -> float:
        """Expected quality. This is the point estimate the greedy router uses."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the Beta. Shrinks like 1/(evidence) — this is *epistemic*
        uncertainty: how much we still don't know about this agent."""
        a, b = self.alpha, self.beta
        n = a + b
        return (a * b) / (n * n * (n + 1.0))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def evidence(self) -> float:
        """Pseudo-count of observations backing this belief (``alpha + beta``).

        Two agents can share the same mean (0.8) yet differ wildly in evidence
        — Beta(4, 1) vs Beta(400, 100). PRISM treats those very differently."""
        return self.alpha + self.beta

    @property
    def entropy(self) -> float:
        """Differential entropy of the Beta — a scalar 'how unsure am I' knob
        used when reporting an output's confidence."""
        a, b = self.alpha, self.beta
        return (
            _log_beta_fn(a, b)
            - (a - 1.0) * _digamma(a)
            - (b - 1.0) * _digamma(b)
            + (a + b - 2.0) * _digamma(a + b)
        )

    # --- online update ------------------------------------------------------

    def updated(self, reward: float, weight: float = 1.0) -> "Belief":
        """Bayesian update from a soft reward in ``[0, 1]``.

        ``reward`` is treated as a fractional success: it splits ``weight`` units
        of evidence between ``alpha`` (success mass) and ``beta`` (failure mass).
        A perfect output (reward=1) adds a full unit to alpha; a total failure
        adds it to beta; a 0.7 adds 0.7 / 0.3. ``weight`` lets a *confident*
        score move the posterior more than a hedged one.

        Industrial hardening: non-finite rewards (NaN/inf from a broken scorer)
        are treated as total failures (0.0) rather than poisoning the posterior;
        negative weights are ignored. Garbage in must never corrupt the policy.
        """
        if not math.isfinite(reward):
            reward = 0.0
        if not math.isfinite(weight) or weight < 0.0:
            weight = 0.0
        reward = _clamp(reward, 0.0, 1.0)
        return Belief(
            alpha=self.alpha + weight * reward,
            beta=self.beta + weight * (1.0 - reward),
        )

    # --- sampling -----------------------------------------------------------

    def sample(self, rng: random.Random) -> float:
        """Draw a plausible quality value. This is the primitive behind Thompson
        sampling: sampling each arm's belief and picking the argmax is provably
        an excellent explore/exploit strategy."""
        return rng.betavariate(self.alpha, self.beta)

    def credible_interval(self, mass: float = 0.9, rng: random.Random | None = None,
                          samples: int = 2000) -> tuple[float, float]:
        """Approximate central credible interval via sampling (kept dependency
        free — with numpy you'd use the exact inverse-CDF)."""
        rng = rng or random.Random(0)
        draws = sorted(self.sample(rng) for _ in range(samples))
        lo = (1.0 - mass) / 2.0
        hi = 1.0 - lo
        return draws[int(lo * samples)], draws[min(int(hi * samples), samples - 1)]

    def __repr__(self) -> str:
        return (f"Belief(mean={self.mean:.3f}, std={self.std:.3f}, "
                f"n={self.evidence:.1f})")


def from_mean_std(mean: float, std: float) -> Belief:
    """Construct a Belief from a target mean and standard deviation via moment
    matching. Handy for seeding priors when you have a rough sense of an agent's
    quality but no observations yet."""
    mean = _clamp(mean, 1e-6, 1.0 - 1e-6)
    var = min(std * std, mean * (1.0 - mean) - _EPS)
    var = max(var, _EPS)
    # concentration := mean(1-mean)/var - 1  (inverts the Beta variance formula)
    concentration = mean * (1.0 - mean) / var - 1.0
    concentration = max(concentration, _EPS)
    return Belief(alpha=mean * concentration, beta=(1.0 - mean) * concentration)


def propagate(upstream: Belief, downstream: Belief) -> Belief:
    """Compose two stage beliefs into the belief about their *chained* output.

    The intuition: a pipeline's quality is gated by every stage. If stage A
    produces mediocre input, even a great stage B is working with mediocre raw
    material — "garbage in, garbage out." We model the end-to-end latent quality
    as the **product** of the per-stage latent qualities (both in ``[0, 1]``),
    which is a conservative, monotone composition.

    For two *independent* variables X, Y on ``[0, 1]``::

        E[XY]   = E[X] E[Y]
        Var(XY) = (Var X + E[X]^2)(Var Y + E[Y]^2) - E[X]^2 E[Y]^2

    We compute those exact moments, then moment-match back onto a Beta so the
    result is still a first-class :class:`Belief` and can be propagated again
    down an arbitrarily long chain. Crucially, the composed variance is *larger*
    than either input's (relative to its mean) — uncertainty **compounds**, it
    never silently vanishes. That compounded variance is what PRISM surfaces as
    the confidence of a multi-stage answer.
    """
    mx, my = upstream.mean, downstream.mean
    vx, vy = upstream.variance, downstream.variance

    mean = mx * my
    second_moment = (vx + mx * mx) * (vy + my * my)
    var = max(second_moment - (mx * my) ** 2, _EPS)

    # Keep the variance inside the Beta's feasible region [0, mean(1-mean)).
    var = min(var, mean * (1.0 - mean) - _EPS)
    return from_mean_std(mean, math.sqrt(var))


def probability_best(beliefs: dict[str, Belief], rng: random.Random,
                     samples: int = 512) -> dict[str, float]:
    """Monte-Carlo estimate of ``P(arm i is the best arm)`` for each arm.

    This is the quantity that drives *both* routing and the decision to
    speculate. We draw a joint Thompson sample of every arm's quality and tally
    who wins; repeating gives a distribution over "who is actually best." If one
    arm wins ~99% of draws, routing is easy and we commit. If the top two split
    55/45, the greedy choice is a coin flip in disguise — a textbook case for
    speculative execution.
    """
    names = list(beliefs)
    if not names:
        return {}
    wins = {name: 0 for name in names}
    for _ in range(samples):
        best_name, best_val = None, -1.0
        for name in names:
            val = beliefs[name].sample(rng)
            if val > best_val:
                best_name, best_val = name, val
        wins[best_name] += 1  # type: ignore[index]
    return {name: wins[name] / samples for name in names}


def selection_entropy(p_best: dict[str, float]) -> float:
    """Shannon entropy (nats) of the 'who is best' distribution. 0 = certain,
    ln(k) = maximally torn among k arms. A compact scalar for dashboards/traces."""
    return -sum(p * math.log(p) for p in p_best.values() if p > 0.0)


# ---------------------------------------------------------------------------
# Small special-function helpers so the core stays numpy/scipy free.
# ---------------------------------------------------------------------------

def _log_beta_fn(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _digamma(x: float) -> float:
    """Digamma (psi) via recurrence + asymptotic expansion. Accurate to ~1e-8
    for x > 0, which is all we need for entropy reporting."""
    result = 0.0
    # Push x up to >= 6 using psi(x) = psi(x+1) - 1/x for stability.
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic series.
    inv = 1.0 / x
    inv2 = inv * inv
    result += (
        math.log(x)
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    )
    return result


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
