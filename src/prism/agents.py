"""Agents and scorers.

PRISM is agnostic about what an "agent" actually is — an LLM call, a tool, a
whole sub-pipeline. It only requires the :class:`Agent` protocol: a ``name``, a
``cost``, and a ``run(task) -> AgentOutput`` method.

To keep the repo runnable with **zero API keys and zero network**, we ship
:class:`SimulatedAgent`: a deterministic stand-in with a hidden "true
competence" per task type. It lets the benchmarks measure PRISM's routing
regret against ground truth. Swapping in a real backend is a ~10-line adapter
(see ``LLMAgent`` docstring / examples).
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .types import AgentOutput, Task
from .uncertainty import _clamp


@runtime_checkable
class Agent(Protocol):
    """The one interface PRISM cares about. Implement this and you're routable."""

    name: str
    cost: float

    def run(self, task: Task) -> AgentOutput: ...


class SimulatedAgent:
    """A reproducible fake agent for demos, tests, and benchmarks.

    Each agent has a hidden *competence* per task type — a ``(mean, spread)``
    pair. When asked to run, it draws a latent quality from a Beta shaped by
    that competence (so a "strong" agent still occasionally whiffs, and a weak
    one occasionally shines — realistic variance is what makes speculation pay
    off). The latent quality is stashed on the output so the simulated scorer
    can observe it noisily, exactly as a real LLM-judge would observe a real
    answer.

    Parameters
    ----------
    name : str
    competence : dict[task_type -> (mean, spread)]
        Hidden skill profile. ``spread`` is the std of the per-run quality; a
        specialist has high mean + low spread on its home task, a generalist
        has middling mean across many tasks.
    default_competence : (mean, spread)
        Used for task types not explicitly listed.
    cost : float
        Relative compute cost of one invocation (drives the speculation budget).
    base_latency : float
        Simulated seconds per call (never actually slept unless you ask).
    """

    def __init__(
        self,
        name: str,
        competence: Optional[dict[str, tuple[float, float]]] = None,
        default_competence: tuple[float, float] = (0.5, 0.18),
        cost: float = 1.0,
        base_latency: float = 0.02,
    ) -> None:
        self.name = name
        self.competence = competence or {}
        self.default_competence = default_competence
        self.cost = cost
        self.base_latency = base_latency
        # Per-agent RNG stream so runs are reproducible *and* agents don't
        # correlate with each other through a shared global seed.
        self._rng = random.Random(hash(name) & 0xFFFFFFFF)

    def seed(self, seed: int) -> "SimulatedAgent":
        self._rng = random.Random(seed ^ (hash(self.name) & 0xFFFFFFFF))
        return self

    def _draw_quality(self, task_type: str) -> float:
        mean, spread = self.competence.get(task_type, self.default_competence)
        mean = _clamp(mean, 1e-3, 1.0 - 1e-3)
        var = min(spread * spread, mean * (1 - mean) - 1e-6)
        var = max(var, 1e-6)
        conc = mean * (1 - mean) / var - 1.0
        conc = max(conc, 1e-6)
        return self._rng.betavariate(mean * conc, (1 - mean) * conc)

    def run(self, task: Task) -> AgentOutput:
        quality = self._draw_quality(task.task_type)
        # A synthetic "answer" whose text encodes provenance so multi-stage
        # demos read sensibly. Real agents return real content here.
        text = f"[{self.name}:{task.task_type} q={quality:.2f}] {task.payload}"
        return AgentOutput(
            output=text,
            self_report=_clamp(quality + self._rng.gauss(0, 0.05), 0, 1),
            latent_quality=quality,
            cost=self.cost,
            latency=self.base_latency * (0.8 + 0.4 * self._rng.random()),
        )


@runtime_checkable
class Scorer(Protocol):
    """Judges an agent output, returning a calibrated quality score in [0, 1].

    In production this is your verifier: an LLM-as-judge, a unit-test pass rate,
    a reward model, a schema validator — anything that maps an output to a
    quality signal. PRISM's winner selection and bandit updates are only as good
    as this signal, which is *why* PRISM also folds in the prior (see
    :mod:`prism.scoring`) instead of trusting the raw score blindly.
    """

    reliability: float

    def score(self, task: Task, output: AgentOutput) -> float: ...


class SimulatedScorer:
    """A noisy observer of the hidden latent quality.

    ``reliability`` in ``[0, 1]`` interpolates between a perfect oracle (1.0)
    and a coin flip (0.0). We add Gaussian noise with std ``0.25 * (1 -
    reliability)``. This models the real, uncomfortable fact that your judge is
    itself imperfect — and it's precisely why calibrating the score against the
    prior matters."""

    def __init__(self, reliability: float = 0.85, rng: Optional[random.Random] = None) -> None:
        self.reliability = _clamp(reliability, 0.0, 1.0)
        self._rng = rng or random.Random(1234)

    def score(self, task: Task, output: AgentOutput) -> float:
        if output.latent_quality is None:
            # No ground truth to observe (real agent). Fall back to the agent's
            # self-reported confidence, which is better than nothing.
            return _clamp(output.self_report, 0.0, 1.0)
        noise_std = 0.25 * (1.0 - self.reliability)
        observed = output.latent_quality + self._rng.gauss(0.0, noise_std)
        return _clamp(observed, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Real-backend adapter sketch (kept import-light on purpose).
# ---------------------------------------------------------------------------

class CallableAgent:
    """Wrap any ``f(task) -> str`` (or richer) as a PRISM agent.

    Example — an Anthropic-backed agent::

        from anthropic import Anthropic
        client = Anthropic()

        def summarize(task):
            msg = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=1024,
                messages=[{"role": "user", "content": task.payload}],
            )
            return msg.content[0].text

        agent = CallableAgent("claude-opus", summarize, cost=3.0)

    PRISM will route to it, speculate with it, learn its competence online, and
    trace every decision — no other changes required.
    """

    def __init__(self, name: str, fn: Callable[[Task], Any],
                 cost: float = 1.0, self_report: float = 0.5) -> None:
        self.name = name
        self._fn = fn
        self.cost = cost
        self._self_report = self_report

    def run(self, task: Task) -> AgentOutput:
        result = self._fn(task)
        if isinstance(result, AgentOutput):
            return result
        return AgentOutput(output=result, self_report=self._self_report,
                           latent_quality=None, cost=self.cost)
