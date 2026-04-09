"""
env.py
------
FocusAI Task Scheduling RL Environment.

Models a knowledge worker's day: the agent must complete tasks before
their deadlines while managing cognitive energy to avoid burnout.

OpenEnv interface
-----------------
    reset()  -> Observation
    step()   -> (Observation, Reward, done, info)
    state    -> Observation  (@property, read-only)

Actions
-------
    start_task('<id>')   — work on a pending task  (costs time + energy)
    take_break(<hours>)  — recover energy          (costs time)
    switch_task('<id>')  — change active focus      (0.5h context-switch cost)
    noop()               — do nothing              (penalised)

Episode ends when
-----------------
    - All tasks completed
    - Energy reaches 0 (burnout)
    - Time reaches 24h (end of day)
    - STAGNATION_LIMIT consecutive unproductive steps
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from models import Observation, Reward, Task
from reward_and_tasks import (
    GRADERS,
    TASK_LOADERS,
    build_env_tasks,
    calculate_reward,
    numeric_to_level,
    safe_score,
)

logger = logging.getLogger(__name__)

# Episode terminates after this many consecutive invalid / noop steps.
STAGNATION_LIMIT = 5


class FocusEnv:
    """
    FocusAI Task Scheduling RL Environment.

    Implements the OpenEnv interface: reset(), step(), state property.
    """

    def __init__(self, difficulty: str = "easy") -> None:
        if difficulty not in GRADERS:
            raise ValueError(
                f"difficulty must be one of {list(GRADERS)}; got {difficulty!r}"
            )
        self.difficulty: str = difficulty
        self._state: Optional[Dict[str, Any]] = None
        self.history: list = []
        self.metrics: Dict[str, Any] = {}
        self._stagnation: int = 0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Start a fresh episode.  Returns the initial Observation."""
        scenario = TASK_LOADERS[self.difficulty]()
        tasks = build_env_tasks(scenario["tasks"])

        self._state = {
            "time":         9.0,
            "energy":       scenario["energy"],
            "energy_level": scenario["energy_level"],
            "tasks":        tasks,
            "current_task": None,
        }
        self.history = []
        self._stagnation = 0
        self.metrics = {
            "completed_tasks":       0,
            "total_tasks":           len(tasks),
            "on_time":               0,
            "good_energy_usage":     0,
            "high_priority_choices": 0,
            "total_steps":           0,
        }

        logger.info(
            "reset | difficulty=%s tasks=%d energy=%d (%s)",
            self.difficulty, len(tasks),
            self._state["energy"], self._state["energy_level"],
        )
        return self._build_obs()

    @property
    def state(self) -> Observation:
        """Current observation (OpenEnv @property requirement)."""
        if self._state is None:
            raise RuntimeError("Call reset() before accessing state.")
        return self._build_obs()

    def step(
        self, action: str,
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply one action and advance the environment.

        Returns (observation, reward, done, info).
        info contains 'metrics'; 'score' is set when done=True.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        if self._is_done():
            return (
                self._build_obs(),
                Reward(reward=0.0),
                True,
                {"error": "Episode already finished."},
            )

        logger.debug(
            "step | action=%r energy=%d time=%.1f",
            action, self._state["energy"], self._state["time"],
        )

        name, args = self._parse_action(action)

        # ---- Invalid parse -----------------------------------------------
        if name == "invalid":
            logger.warning("step | unparseable action: %r", action)
            return self._penalise(f"Unparseable action: {action!r}")

        # Snapshot energy level BEFORE state mutation
        energy_before = self._state["energy_level"]

        # ---- Dispatch ----------------------------------------------------
        self.history.append(action)

        if name == "start_task":
            result = self._do_start_task(args[0]) if args else {"invalid": True}
        elif name == "take_break":
            result = self._do_take_break(int(args[0])) if args else {"invalid": True}
        elif name == "switch_task":
            result = self._do_switch_task(args[0]) if args else {"invalid": True}
        elif name == "noop":
            result = self._empty_result()
        else:
            result = {"invalid": True}

        # Keep energy_level in sync with numeric value
        self._state["energy_level"] = numeric_to_level(self._state["energy"])

        # ---- Invalid result ----------------------------------------------
        if result.get("invalid"):
            logger.warning("step | invalid result for %r", action)
            return self._penalise(f"Invalid action: {action!r}")

        # ---- Stagnation counter ------------------------------------------
        if name == "noop":
            self._stagnation += 1
        else:
            self._stagnation = 0

        # ---- Reward ------------------------------------------------------
        reward_value = calculate_reward(
            {"energy": energy_before}, action, result,
        )

        if self._state["energy"] <= 0:
            reward_value = max(-20.0, reward_value - 10.0)

        # ---- Metrics -----------------------------------------------------
        self.metrics["total_steps"] += 1

        if result.get("task_completed"):
            self.metrics["completed_tasks"] += 1
            if result.get("before_deadline"):
                self.metrics["on_time"] += 1
            if result.get("priority") == "high":
                self.metrics["high_priority_choices"] += 1

        if name == "start_task" and energy_before in ("medium", "high"):
            self.metrics["good_energy_usage"] += 1
        elif name == "take_break" and energy_before == "low":
            self.metrics["good_energy_usage"] += 1

        # ---- Build output ------------------------------------------------
        done = self._is_done()
        obs = self._build_obs()
        reward_obj = Reward(
            reward=reward_value,
            task_completed=result.get("task_completed", False),
            before_deadline=result.get("before_deadline", False),
            missed_deadline=result.get("missed_deadline", False),
            priority=result.get("priority"),
        )

        info: Dict[str, Any] = {"metrics": dict(self.metrics), "score": None}
        if done:
            info["score"] = GRADERS[self.difficulty](self.metrics)
            logger.info("step | final score=%.6f", info["score"])

        logger.debug(
            "step | reward=%+.1f done=%s time=%.1fh energy=%d (%s)",
            reward_value, done,
            self._state["time"], self._state["energy"],
            self._state["energy_level"],
        )
        return obs, reward_obj, done, info

    # ------------------------------------------------------------------
    # Done condition (single authoritative location)
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        if self._state is None:
            return False
        if all(t["completed"] for t in self._state["tasks"]):
            return True
        if self._state["energy"] <= 0:
            return True
        if self._state["time"] >= 24.0:
            return True
        if self._stagnation >= STAGNATION_LIMIT:
            return True
        return False

    # ------------------------------------------------------------------
    # Penalty helper
    # ------------------------------------------------------------------

    def _penalise(self, reason: str):
        """Flat -3 reward for invalid actions; episode may end via stagnation."""
        self._stagnation += 1
        self.metrics["total_steps"] += 1

        done = self._is_done()
        obs = self._build_obs()
        reward_obj = Reward(reward=-3.0)
        info: Dict[str, Any] = {
            "metrics": dict(self.metrics),
            "score":   None,
            "error":   reason,
        }
        if done:
            info["score"] = GRADERS[self.difficulty](self.metrics)
        return obs, reward_obj, done, info

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _do_start_task(self, task_id: str) -> dict:
        for task in self._state["tasks"]:
            if task["id"] == task_id:
                if task["completed"]:
                    return {"invalid": True}
                self._state["current_task"] = task_id
                self._state["time"] = min(24.0, self._state["time"] + task["duration"])
                self._state["energy"] = max(0, self._state["energy"] - task["duration"] * 10)
                task["completed"] = True
                on_time = self._state["time"] <= task["deadline"]
                return {
                    "task_completed":  True,
                    "before_deadline": on_time,
                    "missed_deadline": not on_time,
                    "priority":        task["priority"],
                }
        return {"invalid": True}

    def _do_take_break(self, hours: int) -> dict:
        if hours <= 0:
            return {"invalid": True}
        self._state["energy"] = min(100, self._state["energy"] + hours * 10)
        self._state["time"] = min(24.0, self._state["time"] + hours)
        return self._empty_result()

    def _do_switch_task(self, task_id: str) -> dict:
        for task in self._state["tasks"]:
            if task["id"] == task_id:
                if task["completed"]:
                    return {"invalid": True}
                if self._state["current_task"] == task_id:
                    return {"invalid": True}
                self._state["current_task"] = task_id
                self._state["time"] = min(24.0, self._state["time"] + 0.5)
                return self._empty_result()
        return {"invalid": True}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict:
        return {
            "task_completed":  False,
            "before_deadline": False,
            "missed_deadline": False,
            "priority":        None,
        }

    @staticmethod
    def _parse_action(action: str):
        """
        Parse an action string into (name, args).

        Examples:
            "start_task('report')"  → ("start_task", ["report"])
            "take_break(2)"         → ("take_break", ["2"])
            "noop()"                → ("noop", [])
        """
        try:
            action = action.strip()
            if "(" not in action or not action.endswith(")"):
                return "invalid", []
            name = action.split("(")[0].strip()
            args_str = action[action.index("(") + 1 : action.rindex(")")]
            if not args_str.strip():
                return name, []
            args = [a.strip().strip("'\"") for a in args_str.split(",")]
            return name, args
        except Exception as exc:
            logger.error("parse_action failed on %r: %s", action, exc)
            return "invalid", []

    def _build_legal_actions(self) -> list:
        """Dynamically compute valid actions for the current state."""
        actions = []
        pending = [t for t in self._state["tasks"] if not t["completed"]]

        for t in pending:
            actions.append(f"start_task('{t['id']}')")

        if self._state["current_task"] is not None:
            for t in pending:
                if t["id"] != self._state["current_task"]:
                    actions.append(f"switch_task('{t['id']}')")

        actions += ["take_break(1)", "take_break(2)", "noop()"]
        return actions

    def _build_obs(self) -> Observation:
        return Observation(
            time=self._state["time"],
            energy=self._state["energy"],
            energy_level=self._state["energy_level"],
            tasks=[
                Task(**{k: v for k, v in t.items()
                        if k in ("id", "name", "priority", "deadline",
                                 "duration", "category", "completed")})
                for t in self._state["tasks"]
            ],
            current_task=self._state["current_task"],
            recent_actions=list(self.history[-3:]),
            goal="Complete all tasks before their deadlines while managing energy to avoid burnout.",
            legal_actions=self._build_legal_actions(),
        )

    def get_observation_text(self) -> str:
        """Human-readable observation string optimised for LLM prompts."""
        time   = self._state["time"]
        energy = self._state["energy"]
        level  = self._state["energy_level"]

        advice = {
            "high":   "Energy HIGH — ideal for difficult tasks.",
            "medium": "Energy MEDIUM — manageable, consider task order.",
            "low":    "Energy LOW — burnout risk. Consider a short break first.",
        }[level]

        lines = []
        for t in self._state["tasks"]:
            if t["completed"]:
                lines.append(f"  [DONE] {t['id']:14s}  {t['name']}")
            else:
                time_left = t["deadline"] - time
                urgency = "⚠ URGENT" if time_left <= t["duration"] + 1 else "OK"
                lines.append(
                    f"  [TODO] {t['id']:14s}  {t['name']}\n"
                    f"           priority={t['priority']:6s}  "
                    f"deadline={t['deadline']}h  "
                    f"duration={t['duration']}h  "
                    f"time_left={time_left:.1f}h  {urgency}"
                )

        history_str = ", ".join(self.history[-3:]) or "none"

        return (
            f"=== FOCUS AI — TASK SCHEDULER ===\n"
            f"Time    : {time}h / 24h\n"
            f"Energy  : {energy}/100 ({level.upper()})\n"
            f"Advice  : {advice}\n"
            f"Pending : {sum(1 for t in self._state['tasks'] if not t['completed'])}"
            f"/{len(self._state['tasks'])} tasks\n\n"
            f"TASKS:\n" + "\n".join(lines) + "\n\n"
            f"Recent actions : {history_str}\n\n"
            f"GOAL: Complete all tasks before their deadlines.\n"
            f"VALID ACTIONS: start_task('<id>') | take_break(<hours>) | "
            f"switch_task('<id>') | noop()\n"
            f"NOTE: Use exact task IDs above. Do NOT repeat a [DONE] task."
        )


# ---------------------------------------------------------------------------
# Smart baseline agent (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

def smart_agent(observation: Observation) -> str:
    """
    Energy-aware, deadline-aware, priority-aware deterministic baseline.

    Strategy:
      1. Critical energy (< 30) → take_break(2) immediately.
      2. Sort pending by priority (high first), then deadline, then duration.
      3. For each candidate: check feasibility (time + energy).
      4. Fallback: attempt the most important task regardless.
    """
    energy = observation.energy

    pending = [t for t in observation.tasks if not t.completed]
    if not pending:
        return "noop()"

    if energy < 30:
        return "take_break(2)"

    level = observation.energy_level
    pri = {"high": 3, "medium": 2, "low": 1}
    candidates = sorted(
        pending,
        key=lambda t: (
            -pri.get(t.priority, 0),
            t.deadline,
            t.duration if level in ("medium", "low") else 0,
        ),
    )

    for task in candidates:
        if (observation.time + task.duration) > task.deadline:
            continue
        if energy >= task.duration * 10:
            return f"start_task('{task.id}')"
        else:
            needed = -(-((task.duration * 10) - energy) // 10)
            return f"take_break({needed})"

    # Fallback — best task regardless of deadline feasibility
    best = candidates[0]
    if energy < best.duration * 10:
        needed = -(-((best.duration * 10) - energy) // 10)
        return f"take_break({needed})"
    return f"start_task('{best.id}')"


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    for diff in ("easy", "medium", "hard"):
        print(f"\n{'='*60}")
        print(f"  {diff.upper()}")
        print("=" * 60)

        env = FocusEnv(difficulty=diff)
        obs = env.reset()
        print(env.get_observation_text())

        for step_n in range(20):
            action = smart_agent(obs)
            obs, rew, done, info = env.step(action)
            print(
                f"  Step {step_n+1:2d} | {action:<35} "
                f"| reward={rew.reward:+.1f} | done={done}"
            )
            if done:
                score = info["score"]
                print(f"\n  Score ({diff}): {score}")
                assert 0 < score < 1, f"Score out of range: {score}"
                break

    print("\nAll scores in (0, 1). ✓")
