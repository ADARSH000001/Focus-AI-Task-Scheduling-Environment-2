"""
inference.py
------------
Inference runner for the FocusAI Task Scheduling Environment.

Supports two agent modes:
  1. LLM agent   — uses API_BASE_URL + MODEL_NAME + HF_TOKEN env vars
  2. Smart agent  — built-in deterministic fallback (no API needed)

Structured stdout logs (machine-parseable):
  [START] {...}
  [STEP]  {...}
  [END]   {...}

Usage:
    python inference.py                      # all 3 difficulties, smart agent
    python inference.py --difficulty easy     # single difficulty
    python inference.py --strict-env          # fail if LLM env vars are missing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

from env import FocusEnv, smart_agent
from models import Observation
from reward_and_tasks import GRADERS, safe_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 10
ALLOWED_ACTIONS = {"start_task", "take_break", "switch_task", "noop"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured logging helpers  (key=value format — required by OpenEnv validator)
# ---------------------------------------------------------------------------

def emit_start(task_id: str, env_name: str, model: str) -> None:
    """[START] task=<id> env=<name> model=<model>"""
    print(f"[START] task={task_id} env=focus_ai model={model}", flush=True)


def emit_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str = "null",
) -> None:
    """[STEP] step=N action=ACTION reward=R done=true|false error=null"""
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.4f} done={str(done).lower()} error={error}",
        flush=True,
    )


def emit_end(
    task_id: str,
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """[END] task=<id> success=true|false steps=N score=S rewards=r1,r2,..."""
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] task={task_id} success={str(success).lower()} "
        f"steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FocusAI inference runner")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Run one difficulty or all three (default: all)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum steps per episode (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="Fail if API_BASE_URL / MODEL_NAME / HF_TOKEN are missing",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

_OLD_HF_URL = "https://api-inference.huggingface.co"
_NEW_HF_URL = "https://router.huggingface.co/v1"


def get_runtime_config(strict_env: bool = False) -> Dict[str, Optional[str]]:
    """Read API configuration from environment variables."""
    api_base = os.getenv("API_BASE_URL", _NEW_HF_URL)
    model = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    token = os.getenv("HF_TOKEN")

    # Auto-correct the deprecated api-inference endpoint
    if api_base and _OLD_HF_URL in api_base:
        print(
            f"[WARN] API_BASE_URL points to deprecated endpoint ({api_base}).\n"
            f"       Auto-correcting to {_NEW_HF_URL}",
            file=sys.stderr,
        )
        api_base = _NEW_HF_URL

    if strict_env and not all([api_base, model, token]):
        missing = [
            k
            for k, v in {
                "API_BASE_URL": api_base,
                "MODEL_NAME": model,
                "HF_TOKEN": token,
            }.items()
            if not v
        ]
        print(
            f"[ERROR] Missing required environment variables: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    return {"api_base_url": api_base, "model_name": model, "hf_token": token}


# ---------------------------------------------------------------------------
# LLM System Prompt  ← THE CORE PROMPT, tuned to the grader's scoring weights
# ---------------------------------------------------------------------------
#
# Grader scoring weights (from reward_and_tasks.py):
#
#   grade_easy   → 60% task completion  + 40% on-time
#   grade_medium → 40% completion + 35% on-time + 25% energy efficiency
#   grade_hard   → 35% completion + 30% on-time + 20% energy + 15% priority order
#
# Step reward signals (from calculate_reward):
#   ✅ Complete high-priority task          → +17  (base 15 + bonus 2)
#   ✅ Complete medium-priority task        → +12
#   ✅ Complete low-priority task           → +10  (base 10 - penalty 2 = 8)
#   ✅ Finish before deadline               → +5
#   ✅ Take break when energy is LOW        → +7   (energy delta 5 + recovery bonus 2)
#   ✅ Start task when energy is HIGH       → +3   (energy delta)
#   ❌ Miss a deadline                      → -10
#   ❌ Start task when energy is LOW        → -11  (delta -8 + burnout penalty -3)
#   ❌ Take break when energy is HIGH       → -3   (energy delta)
#   ❌ noop at any time                     → -3
#
# The prompt teaches the agent to:
#   1. Always prioritise HIGH → MEDIUM → LOW tasks
#   2. Never use noop (strong penalty)
#   3. Recover energy with take_break only when energy is LOW (≤ 40)
#   4. Work when energy is HIGH or MEDIUM (> 40)
#   5. Respect deadlines — choose task with earliest deadline first among equals
#   6. Reply with ONLY the action string — nothing else

_SYSTEM_PROMPT = """\
You are FocusAI, an expert productivity scheduling agent managing a knowledge \
worker's day in a deterministic simulation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Maximise the episode score by:
  1. Completing ALL tasks before their deadlines.
  2. Finishing HIGH-priority tasks first.
  3. Maintaining energy to avoid burnout penalties.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RULES (grader weights you must optimise for)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Easy   → 60% task completion   + 40% on-time delivery
  Medium → 40% completion + 35% on-time + 25% energy efficiency
  Hard   → 35% completion + 30% on-time + 20% energy + 15% priority ordering

  This means: completing tasks on time and in priority order is ALWAYS worth more
  than taking extra breaks or doing low-priority work first.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWARD SIGNALS (to guide your choices)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRONGLY POSITIVE:
    ✅  Complete a HIGH-priority task         → +17 points
    ✅  Complete any task before deadline     → +5 extra points
    ✅  take_break when energy is LOW (≤40)   → +7 points (recovery bonus)
    ✅  start_task when energy is HIGH (>70)  → +3 energy bonus

  STRONGLY NEGATIVE:
    ❌  Miss a deadline                       → −10 points (catastrophic)
    ❌  start_task when energy is LOW (≤40)   → −11 points (burnout risk)
    ❌  take_break when energy is HIGH (>70)  → −3 points (wasted time)
    ❌  noop()                                → −3 points (always penalised)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION ALGORITHM — follow this order every step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 1 — CHECK ENERGY FIRST:
    If energy_level == "low"  (energy ≤ 40):
        → take_break(2)          # Always recover before working. Never start a task.

  STEP 2 — PICK THE BEST TASK (energy is medium or high):
    Among all incomplete tasks visible in legal_actions:
        a) Sort by priority:     high > medium > low
        b) Break ties by:        earliest deadline first
        c) Use:                  start_task('<id>') or switch_task('<id>')

  STEP 3 — NO TASKS LEFT:
    If all tasks are completed and no legal start/switch actions remain:
        → take_break(1)          # Recover energy while waiting for episode end
        # NEVER use noop() — it always loses 3 points with no benefit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEGAL ACTIONS FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  start_task('<task_id>')          — begin a new task (requires high/medium energy)
  switch_task('<task_id>')         — switch to a different task mid-step
  take_break(<hours>)             — rest to recover energy (use 1–3 hours)
  noop()                          — do nothing (AVOID — always penalised −3)

  You will be given legal_actions listing exactly which actions are valid right now.
  You MUST only choose from that list. Do NOT invent actions not in the list.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIORITY REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  high   → Do this FIRST — worth +17 points when done before deadline
  medium → Do this SECOND — worth +12 points
  low    → Do this LAST — worth only +8 points (base 10 − penalty 2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Reply with ONLY the action string. No explanation. No reasoning. No JSON.
  No markdown. No extra words. Just the action.

  ✅ CORRECT:   start_task('report')
  ✅ CORRECT:   take_break(2)
  ✅ CORRECT:   switch_task('incident')
  ❌ WRONG:     "I will start the report task" → INVALID — the parser will fail
  ❌ WRONG:     {"action": "start_task('report')"} → INVALID — no JSON
  ❌ WRONG:     noop() → only if no other legal action exists

Examples of ideal responses:
  start_task('report')
  switch_task('pr_review')
  take_break(2)
"""


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def _extract_action(text: str, legal: List[str]) -> Optional[str]:
    """Extract a valid action from LLM output (robust multi-strategy parser)."""
    text = text.strip()

    # 1. Direct exact match
    if text in legal:
        return text

    # 2. Strip markdown fences or quotes the LLM might add
    text_clean = text.strip("`\"'")
    if text_clean in legal:
        return text_clean

    # 3. Regex search for known action patterns inside the text
    pattern = r"(start_task|take_break|switch_task|noop)\s*\([^)]*\)"
    for match in re.finditer(pattern, text):
        candidate = match.group(0)
        if candidate in legal:
            return candidate

    # 4. Partial name match — first legal action whose base name appears in text
    for action in legal:
        name = action.split("(")[0]
        if name in text:
            return action

    return None


def llm_agent(
    observation: Observation,
    client,
    model_name: str,
    conversation: List[Dict[str, str]],
) -> str:
    """
    Call the LLM and return a valid action string.

    Builds a compact, information-dense observation string so the LLM has
    all the data it needs to apply the decision algorithm in the system prompt.
    """
    # Format pending tasks with priority + deadline for easy sorting by LLM
    pending = [
        f"  [{t.priority.upper()}] {t.id!r} — deadline hour {t.deadline}, "
        f"duration {t.duration}h"
        for t in observation.tasks
        if not t.completed
    ]
    completed_ids = [t.id for t in observation.tasks if t.completed]

    obs_text = (
        f"=== CURRENT STATE ===\n"
        f"Time        : {observation.time:.1f}h\n"
        f"Energy      : {observation.energy}/100  [{observation.energy_level.upper()}]\n"
        f"Active task : {observation.current_task or 'none'}\n"
        f"Recent      : {observation.recent_actions}\n"
        f"\nPENDING TASKS (choose from these, high priority first):\n"
        + ("\n".join(pending) if pending else "  (all tasks completed)")
        + f"\n\nCOMPLETED: {completed_ids}\n"
        f"\nLEGAL ACTIONS: {observation.legal_actions}\n"
        f"\nGOAL: {observation.goal}\n"
        f"\nRemember: energy={observation.energy_level.upper()} → "
        + (
            "TAKE A BREAK FIRST (energy is low ≤ 40, starting a task costs −11 pts)"
            if observation.energy_level == "low"
            else "WORK ON HIGHEST PRIORITY TASK (energy is good)"
        )
    )

    conversation.append({"role": "user", "content": obs_text})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + conversation,
            max_tokens=32,       # Action strings are short — keep context tight
            temperature=0.0,     # Deterministic — graders reward consistency
        )
        reply = response.choices[0].message.content or ""
        conversation.append({"role": "assistant", "content": reply})

        action = _extract_action(reply, observation.legal_actions)
        if action:
            return action

        logger.warning("LLM returned no valid action (%r) — falling back to smart_agent", reply)
    except Exception as exc:
        logger.error("LLM error: %s — falling back to smart_agent", exc)

    return smart_agent(observation)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    difficulty: str,
    max_steps: int,
    config: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """
    Run one full episode.

    Uses the LLM if credentials are available; falls back to smart_agent.
    """
    use_llm = bool(config.get("hf_token"))
    client = None

    if use_llm:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=config["hf_token"],
                base_url=config["api_base_url"],
            )
        except Exception as exc:
            logger.warning("Could not create LLM client: %s", exc)
            use_llm = False

    env = FocusEnv(difficulty=difficulty)
    obs = env.reset()

    model_name = config.get("model_name") if use_llm else "smart_agent"
    task_id = difficulty  # task_id matches openenv.yaml task ids

    emit_start(task_id=task_id, env_name="focus_ai", model=model_name)

    conversation: List[Dict[str, str]] = []
    step_rewards: List[float] = []
    total_reward = 0.0
    final_score = None

    for step_n in range(max_steps):
        if use_llm and client:
            action = llm_agent(obs, client, config["model_name"], conversation)
        else:
            action = smart_agent(obs)

        obs, reward, done, info = env.step(action)
        total_reward += reward.reward
        step_rewards.append(reward.reward)

        emit_step(
            step=step_n + 1,
            action=action,
            reward=reward.reward,
            done=done,
        )

        if done:
            final_score = info.get("score")
            break

    # Compute score if episode didn't terminate naturally
    if final_score is None:
        final_score = GRADERS[difficulty](env.metrics)

    # CRITICAL: verify score is strictly in (0, 1) before returning
    assert 0 < final_score < 1, (
        f"Score {final_score} is outside (0, 1) for difficulty={difficulty}"
    )

    success = final_score >= 0.5

    emit_end(
        task_id=task_id,
        success=success,
        steps=len(step_rewards),
        score=final_score,
        rewards=step_rewards,
    )

    return {
        "difficulty":   difficulty,
        "score":        final_score,
        "total_reward": total_reward,
        "metrics":      dict(env.metrics),
        "agent":        "llm" if use_llm else "smart",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    config = get_runtime_config(strict_env=args.strict_env)

    difficulties = (
        ["easy", "medium", "hard"]
        if args.difficulty == "all"
        else [args.difficulty]
    )

    results = []
    for diff in difficulties:
        result = run_episode(diff, args.max_steps, config)
        results.append(result)

    # Summary
    if len(results) > 1:
        avg_score = sum(r["score"] for r in results) / len(results)
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        print(
            f"\nSummary: avg_score={avg_score:.4f}  avg_reward={avg_reward:.1f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()