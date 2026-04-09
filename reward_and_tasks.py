"""
reward_and_tasks.py
-------------------
Task definitions, reward shaping, and deterministic graders for FocusAI.

SCORE CONTRACT  (HIGHEST PRIORITY)
----------------------------------
Every public grader returns a float STRICTLY inside the open interval (0, 1).

    safe_score(raw) = LOWER + (UPPER - LOWER) * clamp(raw, 0, 1)

    where LOWER = 0.01, UPPER = 0.99

This guarantees:
    raw = 0.0  →  0.01   (> 0, never equals 0)
    raw = 1.0  →  0.99   (< 1, never equals 1)
    raw = 0.5  →  0.50

The formula is a linear bijection on [0, 1] → [0.01, 0.99].  It is
mathematically impossible for any IEEE-754 float in [0, 1] to produce
exactly 0.0 or 1.0 under this mapping.

An additional assertion guards every call site at runtime.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENERGY_LEVELS = ("low", "medium", "high")
PRIORITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}

# Score boundaries — strictly inside (0, 1)
_SCORE_LOWER = 0.01
_SCORE_UPPER = 0.99


# ---------------------------------------------------------------------------
# Score safety  [HIGHEST PRIORITY — must return strictly (0, 1)]
# ---------------------------------------------------------------------------

def safe_score(raw: float) -> float:
    """
    Convert a raw score in [0, 1] to a value STRICTLY inside (0, 1).

    Output range: [0.01, 0.99]
        raw=0.0 → 0.01  (> 0, never 0)
        raw=1.0 → 0.99  (< 1, never 1)

    The result is verified with an assertion before returning so that
    any floating-point anomaly is caught immediately rather than at
    submission time.
    """
    raw = float(raw)
    # Clamp input to [0, 1]
    if raw < 0.0:
        raw = 0.0
    elif raw > 1.0:
        raw = 1.0

    # Linear map [0, 1] → [LOWER, UPPER]
    result = _SCORE_LOWER + (_SCORE_UPPER - _SCORE_LOWER) * raw

    # Round to 6 decimal places to kill floating-point noise
    result = round(result, 6)

    # CRITICAL SAFETY NET — catch any anomaly before submission
    assert 0.0 < result < 1.0, (
        f"safe_score VIOLATION: raw={raw!r} produced result={result!r} "
        f"which is not strictly inside (0, 1)"
    )

    return result


# ---------------------------------------------------------------------------
# Energy helpers
# ---------------------------------------------------------------------------

def numeric_to_level(energy: int) -> str:
    """
    Convert numeric energy (0–100) to a qualitative level.

        > 70  → "high"
        > 40  → "medium"
        ≤ 40  → "low"
    """
    if energy > 70:
        return "high"
    if energy > 40:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def calculate_reward(state: dict, action: str, result: dict) -> float:
    """
    Compute the shaped step reward for one (state, action, result) triple.

    Design principles:
      1. Dense — every step produces a non-zero signal.
      2. Priority-aware — high-priority completions earn more.
      3. Deadline-sensitive — early finish is rewarded; missing is penalised.
      4. Energy-aware — working at low energy is costly; recovering is good.
      5. Anti-spam — noop and random actions are penalised.

    Return value is clamped to [-20, +20] for stability.
    """
    reward = 0.0

    # Determine energy level (accept string or int)
    raw_energy = state.get("energy", "medium")
    if isinstance(raw_energy, int):
        energy = numeric_to_level(raw_energy)
    elif raw_energy in ENERGY_LEVELS:
        energy = raw_energy
    else:
        energy = "medium"

    # ---- Task completion ---------------------------------------------------
    if result.get("task_completed"):
        priority = result.get("priority", "low")
        base_reward = {"high": 15, "medium": 12, "low": 10}.get(priority, 10)
        reward += base_reward

        # Priority bonus/penalty for start_task actions
        if action.startswith("start_task"):
            if priority == "high":
                reward += 2
            elif priority == "low":
                reward -= 2

        # Early-finish bonus
        if result.get("before_deadline"):
            reward += 2

    # ---- Deadline outcome -------------------------------------------------
    if result.get("before_deadline"):
        reward += 5
    elif result.get("missed_deadline"):
        reward -= 10

    # ---- Energy management ------------------------------------------------
    if action.startswith("start_task"):
        energy_delta = {"high": 3, "medium": 1, "low": -8}.get(energy, 0)
        reward += energy_delta
        if energy == "low":
            reward -= 3  # Extra burnout-risk penalty

    elif action.startswith("take_break"):
        energy_delta = {"low": 5, "medium": 1, "high": -3}.get(energy, 0)
        reward += energy_delta
        if energy == "low":
            reward += 2  # Extra recovery bonus

    # ---- Anti-spam --------------------------------------------------------
    if action.startswith("noop"):
        reward -= 3

    # ---- Clamp to [-20, +20] ---------------------------------------------
    clamped = max(-20.0, min(20.0, reward))
    return float(clamped)


# ---------------------------------------------------------------------------
# Task format conversion
# ---------------------------------------------------------------------------

def build_env_tasks(raw_tasks: list) -> list:
    """
    Convert raw task definitions into env-internal task dicts.
    Maps 'difficulty' field → 'duration' (1 difficulty = 1 hour of work).
    """
    return [
        {
            "id":        t["id"],
            "name":      t.get("name", t["id"]),
            "priority":  t["priority"],
            "deadline":  t["deadline"],
            "duration":  t.get("difficulty", 1),
            "category":  t.get("category", "general"),
            "completed": False,
        }
        for t in raw_tasks
    ]


# ---------------------------------------------------------------------------
# Task generators (3 difficulty levels)
# ---------------------------------------------------------------------------

def get_easy_task() -> dict:
    """
    Easy scenario: high starting energy, 2 tasks, clear priority ordering.
    Tests: can the agent pick the right task first?
    """
    return {
        "energy":       80,
        "energy_level": "high",
        "tasks": [
            {
                "id":         "report",
                "name":       "Write Q3 status report (due to manager)",
                "priority":   "high",
                "deadline":   12,
                "difficulty": 1,
                "category":   "writing",
            },
            {
                "id":         "inbox",
                "name":       "Clear email inbox (low urgency)",
                "priority":   "low",
                "deadline":   18,
                "difficulty": 1,
                "category":   "communication",
            },
        ],
    }


def get_medium_task() -> dict:
    """
    Medium scenario: medium energy, 3 tasks with mixed priorities and
    tighter deadlines.  Requires energy management + prioritisation.
    """
    return {
        "energy":       60,
        "energy_level": "medium",
        "tasks": [
            {
                "id":         "pr_review",
                "name":       "Review pull request blocking team deploy",
                "priority":   "high",
                "deadline":   11,
                "difficulty": 2,
                "category":   "engineering",
            },
            {
                "id":         "client_call",
                "name":       "Prepare slides for client presentation",
                "priority":   "medium",
                "deadline":   14,
                "difficulty": 2,
                "category":   "communication",
            },
            {
                "id":         "docs",
                "name":       "Update internal documentation",
                "priority":   "low",
                "deadline":   18,
                "difficulty": 1,
                "category":   "writing",
            },
        ],
    }


def get_hard_task() -> dict:
    """
    Hard scenario: medium-low energy, 6 competing tasks with two critical
    high-priority blockers on tight deadlines.
    Tests: genuine multi-step planning under constraints.
    """
    return {
        "energy":       55,
        "energy_level": "medium",
        "tasks": [
            {
                "id":         "incident",
                "name":       "Respond to production incident report",
                "priority":   "high",
                "deadline":   12,
                "difficulty": 2,
                "category":   "engineering",
            },
            {
                "id":         "security",
                "name":       "Apply critical security patch before audit",
                "priority":   "high",
                "deadline":   13,
                "difficulty": 3,
                "category":   "engineering",
            },
            {
                "id":         "interview",
                "name":       "Complete candidate interview feedback form",
                "priority":   "medium",
                "deadline":   15,
                "difficulty": 2,
                "category":   "hr",
            },
            {
                "id":         "budget",
                "name":       "Submit team budget request",
                "priority":   "medium",
                "deadline":   16,
                "difficulty": 2,
                "category":   "finance",
            },
            {
                "id":         "onboarding",
                "name":       "Set up new hire workstation",
                "priority":   "low",
                "deadline":   20,
                "difficulty": 1,
                "category":   "hr",
            },
            {
                "id":         "retro",
                "name":       "Write sprint retrospective notes",
                "priority":   "low",
                "deadline":   22,
                "difficulty": 1,
                "category":   "process",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: float, denominator: float) -> float:
    """Safe division clamped to [0.0, 1.0]. Returns 0.0 when denominator ≤ 0."""
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


# ---------------------------------------------------------------------------
# Graders  (all return strictly (0, 1) via safe_score)
# ---------------------------------------------------------------------------

def grade_easy(metrics: dict) -> float:
    """
    Easy grader.
    Weights: 60% completion + 40% on-time.
    """
    total = max(1, metrics.get("total_tasks", 1))
    raw = (
        0.60 * _safe_ratio(metrics.get("completed_tasks", 0), total)
        + 0.40 * _safe_ratio(metrics.get("on_time", 0), total)
    )
    return safe_score(raw)


def grade_medium(metrics: dict) -> float:
    """
    Medium grader.
    Weights: 40% completion + 35% on-time + 25% energy efficiency.
    """
    total = max(1, metrics.get("total_tasks", 1))
    steps = max(1, metrics.get("total_steps", 1))
    raw = (
        0.40 * _safe_ratio(metrics.get("completed_tasks", 0), total)
        + 0.35 * _safe_ratio(metrics.get("on_time", 0), total)
        + 0.25 * _safe_ratio(metrics.get("good_energy_usage", 0), steps)
    )
    return safe_score(raw)


def grade_hard(metrics: dict) -> float:
    """
    Hard grader.
    Weights: 35% completion + 30% on-time + 20% energy + 15% priority.
    """
    total = max(1, metrics.get("total_tasks", 1))
    steps = max(1, metrics.get("total_steps", 1))
    completed = max(1, metrics.get("completed_tasks", 1))
    raw = (
        0.35 * _safe_ratio(metrics.get("completed_tasks", 0), total)
        + 0.30 * _safe_ratio(metrics.get("on_time", 0), total)
        + 0.20 * _safe_ratio(metrics.get("good_energy_usage", 0), steps)
        + 0.15 * _safe_ratio(metrics.get("high_priority_choices", 0), completed)
    )
    return safe_score(raw)


# ---------------------------------------------------------------------------
# Registry maps
# ---------------------------------------------------------------------------

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

TASK_LOADERS = {
    "easy":   get_easy_task,
    "medium": get_medium_task,
    "hard":   get_hard_task,
}
