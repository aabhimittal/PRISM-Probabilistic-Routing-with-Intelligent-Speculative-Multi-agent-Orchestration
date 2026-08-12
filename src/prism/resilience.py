"""Circuit breaking for flaky agents.

A bandit already learns that a *bad* agent is bad — but slowly, one pulled arm
at a time, and it keeps probing. That's the right behaviour for "mediocre" and
the wrong behaviour for "actively failing": an agent whose backing service is
down will fail every call, and each probe costs money and latency. Industrial
systems solved this long ago with the **circuit breaker** pattern; PRISM adapts
it per (task_type, agent) arm:

  * CLOSED   — normal operation; failures are counted.
  * OPEN     — after ``failure_threshold`` *consecutive* failures the arm is
               removed from the candidate set for ``cooldown`` runs. No calls,
               no cost, no latency.
  * HALF-OPEN — when the cooldown expires the arm is re-admitted for a single
               probe. Success closes the breaker; failure re-opens it with an
               exponentially longer cooldown (capped), the classic backoff.

The breaker composes with, not replaces, the bandit: failures also feed
reward-0 belief updates, so even after a breaker closes, Thompson sampling is
appropriately skeptical of the arm until it re-earns trust.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class _ArmState:
    consecutive_failures: int = 0
    open_until: int = -1          # run-counter tick at which cooldown expires
    cooldown: int = 0             # current cooldown length (grows on re-trips)
    half_open: bool = False       # probe in flight
    trips: int = 0                # lifetime count, for observability


@dataclass
class CircuitBreaker:
    """Per-arm circuit breaker keyed by ``(task_type, agent_name)``.

    ``tick`` is a monotonically increasing run counter supplied by the
    orchestrator (one tick per pipeline run) — cooldowns are measured in runs,
    not wall-clock, so behaviour is deterministic and testable.
    """

    failure_threshold: int = 3
    base_cooldown: int = 10
    max_cooldown: int = 160
    _states: dict[tuple[str, str], _ArmState] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _state(self, stage: str, agent: str) -> _ArmState:
        return self._states.setdefault((stage, agent), _ArmState())

    # --- candidate filtering -----------------------------------------------

    def filter(self, stage: str, names: list[str], tick: int) -> tuple[list[str], list[str]]:
        """Split candidates into (allowed, blocked) for this tick.

        An OPEN arm whose cooldown has expired transitions to HALF-OPEN and is
        allowed through as a probe. If *every* arm is blocked we allow them all
        — a fully-open roster means the breaker has no useful signal left and
        refusing to route at all would turn a degraded state into an outage.
        """
        with self._lock:
            allowed, blocked = [], []
            for n in names:
                st = self._state(stage, n)
                if st.open_until > tick:
                    blocked.append(n)
                else:
                    if st.open_until != -1 and not st.half_open:
                        st.half_open = True   # cooldown just expired: probe mode
                    allowed.append(n)
            if not allowed:
                return names, []   # breaker overridden: degraded > dead
            return allowed, blocked

    # --- outcome recording --------------------------------------------------

    def record(self, stage: str, agent: str, failed: bool, tick: int) -> str | None:
        """Record a call outcome. Returns a human-readable event string when
        the breaker changes state (for the causal trace), else None."""
        with self._lock:
            st = self._state(stage, agent)
            if not failed:
                healed = st.half_open or st.consecutive_failures > 0
                st.consecutive_failures = 0
                st.half_open = False
                st.open_until = -1
                st.cooldown = 0
                return (f"breaker CLOSED for ({stage}, {agent}) — probe succeeded"
                        if healed and st.trips else None)

            st.consecutive_failures += 1
            if st.half_open:
                # Failed probe: re-open with exponential backoff.
                st.cooldown = min(max(st.cooldown * 2, self.base_cooldown),
                                  self.max_cooldown)
                st.open_until = tick + st.cooldown
                st.half_open = False
                st.trips += 1
                return (f"breaker RE-OPENED for ({stage}, {agent}) — probe failed; "
                        f"cooldown {st.cooldown} runs")
            if st.consecutive_failures >= self.failure_threshold and st.open_until <= tick:
                st.cooldown = self.base_cooldown if st.cooldown == 0 else min(
                    st.cooldown * 2, self.max_cooldown)
                st.open_until = tick + st.cooldown
                st.trips += 1
                return (f"breaker OPEN for ({stage}, {agent}) after "
                        f"{st.consecutive_failures} consecutive failures; "
                        f"cooldown {st.cooldown} runs")
            return None

    # --- persistence / observability ---------------------------------------

    def state_dict(self) -> dict[str, dict]:
        with self._lock:
            return {
                f"{stage}::{agent}": {
                    "consecutive_failures": st.consecutive_failures,
                    "open_until": st.open_until,
                    "cooldown": st.cooldown,
                    "half_open": st.half_open,
                    "trips": st.trips,
                }
                for (stage, agent), st in self._states.items()
            }

    def load_state_dict(self, data: dict[str, dict]) -> None:
        with self._lock:
            for key, st in data.items():
                stage, _, agent = key.partition("::")
                self._states[(stage, agent)] = _ArmState(**st)
