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
# Structured logging helpers
# ---------------------------------------------------------------------------

def _emit(tag: str, payload: Dict[str, Any]) -> None:
    """Print a structured JSON log line and flush."""
    print(
        f"[{tag}] {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}"
    )
    sys.stdout.flush()


def emit_start(payload: Dict[str, Any]) -> None:
    _emit("START", payload)


def emit_step(payload: Dict[str, Any]) -> None:
    _emit("STEP", payload)


def emit_end(payload: Dict[str, Any]) -> None:
    _emit("END", payload)


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
# LLM agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a productivity agent managing a knowledge worker's day.

Your goal: complete all tasks before their deadlines while managing energy \
to avoid burnout.

At each step you receive the current state and must choose ONE action from \
the legal_actions list. Reply with ONLY the action string, nothing else.

Examples:
  start_task('report')
  take_break(2)
  switch_task('inbox')
  noop()
"""


def _extract_action(text: str, legal: List[str]) -> Optional[str]:
    """Extract a valid action from LLM output."""
    text = text.strip()

    # 1. Direct exact match
    if text in legal:
        return text

    # 2. Regex search for known action patterns
    pattern = r"(start_task|take_break|switch_task|noop)\s*\([^)]*\)"
    for match in re.finditer(pattern, text):
        candidate = match.group(0)
        if candidate in legal:
            return candidate

    # 3. Partial match — first legal action whose name appears in text
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
    """Call the LLM and return a valid action string."""
    obs_text = (
        f"Time: {observation.time}h | "
        f"Energy: {observation.energy}/100 ({observation.energy_level})\n"
        f"Pending tasks: {[t.id for t in observation.tasks if not t.completed]}\n"
        f"Legal actions: {observation.legal_actions}\n"
        f"Goal: {observation.goal}"
    )
    conversation.append({"role": "user", "content": obs_text})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + conversation,
            max_tokens=64,
            temperature=0.0,
        )
        reply = response.choices[0].message.content or ""
        conversation.append({"role": "assistant", "content": reply})

        action = _extract_action(reply, observation.legal_actions)
        if action:
            return action

        logger.warning("LLM returned no valid action — falling back to smart_agent")
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
    Run one episode.

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

    emit_start({
        "difficulty": difficulty,
        "agent":      "llm" if use_llm else "smart",
        "max_steps":  max_steps,
        "model":      config.get("model_name") if use_llm else "smart_agent",
    })

    conversation: List[Dict[str, str]] = []
    total_reward = 0.0
    final_score = None

    for step_n in range(max_steps):
        if use_llm and client:
            action = llm_agent(obs, client, config["model_name"], conversation)
        else:
            action = smart_agent(obs)

        obs, reward, done, info = env.step(action)
        total_reward += reward.reward

        emit_step({
            "step":        step_n + 1,
            "action":      action,
            "reward":      reward.reward,
            "done":        done,
            "energy":      obs.energy,
            "time":        obs.time,
            "completed":   info["metrics"]["completed_tasks"],
            "total_tasks": info["metrics"]["total_tasks"],
        })

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

    result = {
        "difficulty":   difficulty,
        "score":        final_score,
        "total_reward": total_reward,
        "metrics":      dict(env.metrics),
        "agent":        "llm" if use_llm else "smart",
    }
    emit_end(result)

    return result


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
