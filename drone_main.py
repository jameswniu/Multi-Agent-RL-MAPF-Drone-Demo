"""
Drone control loop with alignment-based decision making.

Big idea:
- We replace delayed external rewards (classic RL) with a present-time "alignment score".
- Each candidate action gets scored by a Constitution (weighted principles like safety).
- The policy picks the action with the highest current alignment score.
- A SafetyController (outside the ML path) can override risky actions.
- A Supervisor wires everything, monitors health, and restarts agents if they crash.

Mapping to RL / MAPF:
- RL: the "reward" is the alignment score (continuous, low-variance). You can still log it
  and fine-tune a policy later, but runtime decisions don't wait for future treats.
- MAPF: add a "planner consistency" principle so actions that conflict with the global
  plan score lower; SafetyController enforces hard no-collision rules.
"""

from __future__ import annotations
import asyncio, time, random, math
from dataclasses import dataclass
from typing import Optional, Literal, Callable, Sequence
import contextlib

# ---------- messages ----------
# These dataclasses are the typed "messages" passed between agents via asyncio queues.

@dataclass(frozen=True)
class Frame:
    """Raw sensor sample produced by the IngestionAgent."""
    seq: int            # sequence number (monotonic)
    ts: float           # capture timestamp (seconds)
    payload: bytes      # placeholder for sensor bytes (IMU/LiDAR/etc.)

@dataclass(frozen=True)
class Preprocessed:
    """Compact features extracted by the PreprocessAgent."""
    seq: int
    ts: float
    features: list[float]  # deterministic, small feature vector

# Actions the drone can take (you can extend this list later).
Action = Literal["hover","climb","turn_left","turn_right","forward"]

@dataclass(frozen=True)
class Decision:
    """Chosen action with an alignment score (0..1) and optional notes."""
    seq: int
    ts: float
    action: Action
    alignment_reward: float  # present-time reward from internal values (no delay)
    notes: str = ""          # debugging breadcrumbs (e.g., per-action scores)

@dataclass
class Health:
    """Minimal health snapshot an agent can report to its supervisor."""
    ok: bool
    latency_ms: float
    drops: int = 0
    reason: str = ""

# ---------- constitution and alignment scoring ----------
# The "Constitution" encodes internal values as small scoring functions (principles).
# Each principle scores (state, action) -> [0..1]. We blend them with weights.
# We also run a tiny "inversion test": perturb the state slightly and ensure the score
# doesn't flip dramatically. This encourages smooth, robust decisions.

@dataclass
class Constitution:
    """
    A compact set of internal values and checks.

    principles: list of (name, weight, fn)
      - name: display name for debugging
      - weight: relative importance (sums are normalized)
      - fn(state, action) -> score in [0,1]
    """
    principles: list[tuple[str, float, Callable[[Sequence[float], Action], float]]]

    def score(self, state: Sequence[float], action: Action) -> float:
        # No principles -> no preference (reward 0)
        if not self.principles:
            return 0.0

        # 1) Weighted average of principle scores
        weighted = 0.0
        wsum = 0.0
        for name, w, fn in self.principles:
            # Clamp each principle's output to [0,1] just to be safe.
            s = max(0.0, min(1.0, fn(state, action)))
            weighted += w * s
            wsum += w
        base = weighted / max(1e-6, wsum)  # avoid divide-by-zero

        # 2) Structural inversion test: score should not change wildly
        #    for tiny perturbations in state (helps robustness).
        eps = 0.05  # ±5% nudge
        inv1 = [x * (1.0 + eps) for x in state]
        inv2 = [x * (1.0 - eps) for x in state]
        s1 = sum(w * max(0.0, min(1.0, fn(inv1, action))) for _, w, fn in self.principles) / wsum
        s2 = sum(w * max(0.0, min(1.0, fn(inv2, action))) for _, w, fn in self.principles) / wsum
        # stability near 1.0 means small state nudges didn't flip the score
        stability = 1.0 - min(1.0, abs(s1 - s2))

        # 3) Blend base preference with stability to form the present reward
        reward = 0.85 * base + 0.15 * stability
        return max(0.0, min(1.0, reward))

class AlignmentScorer:
    """Thin wrapper to obtain a present-time reward for (state, action)."""
    def __init__(self, constitution: Constitution):
        self.constitution = constitution

    def step_reward(self, state: Sequence[float], action: Action) -> float:
        return self.constitution.score(state, action)

# ---------- simple policy that maximizes present alignment ----------
# The policy tries a small set of candidate actions and picks the one
# with the highest current alignment reward (no delayed returns).

class AlignmentPolicy:
    def __init__(self, scorer: AlignmentScorer):
        self.scorer = scorer
        self.candidate_actions: list[Action] = ["hover","climb","turn_left","turn_right","forward"]

    def choose(self, state: Sequence[float]) -> tuple[Action, float, str]:
        best_a = None
        best_r = -1.0
        notes = []
        for a in self.candidate_actions:
            r = self.scorer.step_reward(state, a)
            notes.append(f"{a}:{r:.2f}")  # keep human-debuggable traces
            if r > best_r:
                best_r = r
                best_a = a
        # If everything fails, fall back to the safest idle-ish action.
        return best_a or "hover", float(best_r), " | ".join(notes)

# ---------- Safety Controller (outside ML path) ----------
# This is the final arbiter. Even if the policy picks an action, the safety layer
# can override it immediately (e.g., geofence, collision cones, altitude floors).

class SafetyController:
    def __init__(self):
        self._near_no_fly = False  # stub condition (replace with real check)

    def check(self, d: Decision) -> Decision:
        # Example rule: if near a no-fly zone, never go "forward"
        if self._near_no_fly and d.action == "forward":
            return Decision(d.seq, d.ts, "hover", d.alignment_reward, d.notes + " | safety_override")
        return d

    async def run(self, inq: asyncio.Queue[Decision]):
        # Reads decisions from PredictionAgent, enforces rules, and (in real life)
        # would forward the final command to the flight computer.
        while True:
            d = await inq.get()
            safe = self.check(d)
            # Light telemetry print; real systems would send to a logger/black box.
            if safe.seq % 100 == 0:
                print(f"[safety] seq={safe.seq} action={safe.action} reward={safe.alignment_reward:.2f}")

# ---------- agents ----------
# Agents are small "workers" supervised by the Supervisor. Each has:
# - its own run() loop,
# - a health snapshot,
# - bounded queues connecting stages (for backpressure).

class Agent:
    def __init__(self, name: str):
        self.name = name
        self._task: Optional[asyncio.Task] = None
        self._health = Health(ok=True, latency_ms=0.0)

    async def start(self):
        # Start the agent's asynchronous loop.
        self._task = asyncio.create_task(self.run(), name=self.name)

    def health(self) -> Health:
        # Supervisor can read this quickly (no await) to make decisions.
        return self._health

    async def run(self):
        raise NotImplementedError  # subclasses must implement

class IngestionAgent(Agent):
    """Reads raw sensor data at a fixed rate and pushes frames downstream."""
    def __init__(self, outq: asyncio.Queue[Frame], hz: float = 60.0):
        super().__init__("ingestion")
        self.outq = outq
        self.period = 1.0 / hz
        self.seq = 0

    async def run(self):
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(self.period)  # simulate sensor tick
            # Simulate occasional hardware/bus faults to exercise supervision.
            if random.random() < 0.01:
                raise RuntimeError("sensor bus fault")
            frame = Frame(self.seq, time.time(), b"\x01\x02")
            self.seq += 1
            try:
                self.outq.put_nowait(frame)  # bounded queue applies backpressure
            except asyncio.QueueFull:
                self._health.drops += 1  # downstream is slow; we dropped a frame
            self._health.latency_ms = (time.perf_counter() - t0) * 1e3
            self._health.ok = True

class PreprocessAgent(Agent):
    """Converts raw frames into small, deterministic feature vectors."""
    def __init__(self, inq: asyncio.Queue[Frame], outq: asyncio.Queue[Preprocessed]):
        super().__init__("preprocess")
        self.inq = inq
        self.outq = outq

    async def run(self):
        while True:
            t0 = time.perf_counter()
            frame = await self.inq.get()  # blocks until a frame arrives
            # Deterministic feature extraction (toy logic for example).
            # In real life: calibration, filtering, sensor fusion, normalization.
            features = [len(frame.payload), (frame.seq % 17) / 17.0, 1.0]
            await asyncio.sleep(0.002)  # pretend DSP time
            pp = Preprocessed(frame.seq, frame.ts, features)
            try:
                self.outq.put_nowait(pp)
            except asyncio.QueueFull:
                self._health.drops += 1
            self._health.latency_ms = (time.perf_counter() - t0) * 1e3
            self._health.ok = True

class PredictionAgent(Agent):
    """Selects the best action *right now* using the alignment-based policy."""
    def __init__(self, inq: asyncio.Queue[Preprocessed], outq: asyncio.Queue[Decision], policy: AlignmentPolicy):
        super().__init__("prediction")
        self.inq = inq
        self.outq = outq
        self.policy = policy

    async def run(self):
        while True:
            t0 = time.perf_counter()
            pp = await self.inq.get()
            # Choose action by current alignment score, not by delayed returns.
            action, reward, notes = self.policy.choose(pp.features)
            dec = Decision(pp.seq, pp.ts, action, reward, notes)
            self.outq.put_nowait(dec)
            self._health.latency_ms = (time.perf_counter() - t0) * 1e3
            self._health.ok = True

# ---------- supervisor and wiring ----------
# The Supervisor owns the queues, constructs the Constitution, starts all agents,
# and monitors them (restarts on crashes, simple latency alerts).

class Supervisor:
    def __init__(self):
        # Bounded queues: prevent one slow stage from overflowing another.
        self.q_ing_to_prep: asyncio.Queue[Frame] = asyncio.Queue(maxsize=64)
        self.q_prep_to_pred: asyncio.Queue[Preprocessed] = asyncio.Queue(maxsize=64)
        self.q_pred_to_safety: asyncio.Queue[Decision] = asyncio.Queue(maxsize=128)

        # Define constitutional "principles" (toy examples):
        # - safety_margin: avoid aggressive moves when load/congestion is high
        # - energy_thrift: hovering conserves energy more than moving
        # - smoothness: prefer consistent motions over jittery zig-zags
        def safety_margin(state, action):
            load = state[1]  # here, feature[1] stands for "load/congestion" in [0,1]
            if action == "forward":
                return 1.0 - load  # less forward when congested
            if action in ("turn_left","turn_right"):
                return 0.7
            if action == "climb":
                return 0.8
            return 0.9  # hover is generally safe

        def energy_thrift(state, action):
            # Simplified: hovering is cheapest, turning moderate, others costly.
            return 1.0 if action == "hover" else 0.7 if action in ("turn_left","turn_right") else 0.5

        def smoothness(state, action):
            # Prefer an action consistent with a simple hash of the state.
            # This discourages frequent direction changes.
            pref = ["forward","hover","turn_left","turn_right","climb"][int(state[1]*10) % 5]
            return 1.0 if action == pref else 0.6

        constitution = Constitution(principles=[
            ("safety_margin", 0.5, safety_margin),
            ("energy_thrift", 0.2, energy_thrift),
            ("smoothness",    0.3, smoothness),
        ])
        scorer = AlignmentScorer(constitution)
        policy = AlignmentPolicy(scorer)

        # Agents
        self.ing = IngestionAgent(self.q_ing_to_prep, hz=60.0)
        self.prep = PreprocessAgent(self.q_ing_to_prep, self.q_prep_to_pred)
        self.pred = PredictionAgent(self.q_prep_to_pred, self.q_pred_to_safety, policy)
        self.safety = SafetyController()

        # Simple supervision settings (restart windows/cooldowns)
        self._tasks: list[asyncio.Task] = []
        self._restart_log: dict[str, list[float]] = {"ingestion":[], "preprocess":[], "prediction":[]}
        self._restart_limit = 5   # quarantine after too many restarts
        self._window_s = 30       # count restarts within this time window
        self._cooldown_s = 2      # wait before restarting

    async def start(self):
        # Start each agent and a monitoring coroutine for it.
        for agent in (self.ing, self.prep, self.pred):
            await agent.start()
            self._tasks.append(asyncio.create_task(self._monitor(agent)))
        # Start the SafetyController loop that consumes decisions.
        self._tasks.append(asyncio.create_task(self.safety.run(self.q_pred_to_safety), name="safety"))

    async def _monitor(self, agent: Agent):
        # Very small supervisor: restart crashed agents, print latency warnings.
        while True:
            await asyncio.sleep(0.2)
            task = agent._task
            if task and task.done():
                err = task.exception()
                when = time.time()
                self._restart_log[agent.name].append(when)
                window = [t for t in self._restart_log[agent.name] if when - t <= self._window_s]
                self._restart_log[agent.name] = window
                if len(window) > self._restart_limit:
                    print(f"[sup] {agent.name} quarantined; degrade")
                    await asyncio.sleep(self._cooldown_s)
                else:
                    print(f"[sup] restart {agent.name} after: {err!r}")
                    await asyncio.sleep(self._cooldown_s)
                    await agent.start()

            # Simple SLI check: warn if stage is getting slow.
            h = agent.health()
            if h.latency_ms > 25.0:
                print(f"[sup] {agent.name} latency high {h.latency_ms:.1f} ms")

    async def run(self):
        # Boot the system and keep it alive.
        await self.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            for t in self._tasks:
                t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*self._tasks)

# ---------- boot ----------

async def main():
    # Create the supervisor and run forever (Ctrl-C to stop in a console).
    sup = Supervisor()
    await sup.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
