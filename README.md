---
title: Focus AI Task Scheduling Environment
sdk: docker
app_port: 7860
---

# 🧠 Focus AI — Task Scheduling RL Environment

> A real-world reinforcement learning environment where an AI agent manages tasks, deadlines, and energy to maximize productivity while avoiding burnout.

---

## 1. Project Overview

Focus AI is an **OpenEnv-compatible RL environment** where an agent must:

- Complete tasks before their deadlines
- Manage cognitive energy to avoid burnout
- Prioritize high-importance work over low-priority tasks
- Recover efficiently when exhausted before continuing

The environment simulates a real knowledge-worker's day — making it directly useful for agent benchmarking, policy evaluation, and LLM productivity research.

---

## 2. Problem Statement

Knowledge workers constantly balance urgent tasks, deadlines, context switching, and cognitive fatigue. Focus AI models that decision process as a structured RL environment where every action affects future productivity and wellbeing.

---

## 3. Environment Design

### Action Space

String-based actions passed as plain text:

| Action | Description |
|---|---|
| `start_task('<id>')` | Work on a task for its full duration (costs time + energy) |
| `take_break(1)` or `take_break(2)` | Rest to recover energy (+10 per hour). Max 2 hours. |
| `switch_task('<id>')` | Change active focus with a 0.5h context-switch overhead |
| `noop()` | Do nothing — always penalized -3 |

> `take_break(N)` is capped at N=2. Any value above 2 is treated as an invalid action.

### Observation Space

Typed Pydantic `Observation` model returned from `reset()` and `step()`:

| Field | Type | Description |
|---|---|---|
| `time` | float | Current hour (starts at 9.0, ends at 24.0) |
| `energy` | int | Energy level 0–100 |
| `energy_level` | str | `low` (<=40) / `medium` (41–70) / `high` (>70) |
| `tasks` | List[Task] | All tasks with current status |
| `current_task` | str or None | Active task id |
| `recent_actions` | List[str] | Last 3 actions taken |
| `goal` | str | Episode objective text |
| `legal_actions` | List[str] | All valid actions right now |

---

## 4. Reward Function

Reward is calculated per step and clamped to `[-20, +20]`. These values reflect the actual code in `reward_and_tasks.py`:

| Condition | Reward |
|---|---|
| Complete high-priority task | +15 base +2 priority bonus = **+17** |
| Complete medium-priority task | **+12** base |
| Complete low-priority task | +10 base -2 priority penalty = **+8** |
| Finish before deadline | **+5** |
| Miss deadline | **-10** |
| Work at high energy | **+3** |
| Work at medium energy | **+1** |
| Work at low energy | **-5** |
| Break at low energy | **+4** |
| Break at medium energy | **+1** |
| Unnecessary break at high energy | **-3** |
| `noop()` | **-3** |
| Invalid action | **-3** (flat, no state mutation) |
| Burnout (energy reaches 0) | **-10** added on top of step reward |

> The `Reward` model auto-computes `normalized_reward` in `[-1, +1]` from the raw reward via a Pydantic model_validator. You do not need to pass it in manually.

---

## 5. Task Difficulty Levels

### Easy
- 2 tasks, **high starting energy (80/100)**
- Generous deadlines, 1–2 hour task durations
- Scoring: 60% task completion + 40% on-time delivery

### Medium
- 3 tasks, **medium starting energy (60/100)**
- Tighter deadlines, requires some energy management
- Scoring: 40% completion + 35% on-time + 25% energy efficiency

### Hard
- 6 tasks, **low starting energy (~38/100)** — already in the `low` band from the start
- Tight deadlines (10–16h range), task durations up to 3h
- Requires genuine multi-step planning under severe resource constraints
- Scoring: 35% completion + 30% on-time + 20% energy + 15% priority quality

> **Hard design intent:** With 6 tasks, a 10-step cap, and energy starting in the `low` band, full completion is not realistically achievable. The grader rewards proportional progress. Expect scores in the 0.30–0.65 range even for strong agents.

> All difficulties use randomized task generation by default via `get_random_task()`, preventing LLMs from memorizing fixed scenarios. Pass a `seed` integer to `reset()` for reproducible episodes.

---

## 6. Scoring System

Each difficulty has a deterministic grader. Raw scores are mapped to **strictly inside (0, 1)** via `safe_score()`:

```
safe_score(raw) = 0.01 + 0.98 x clamp(raw, 0, 1)
```

- **Minimum possible score**: 0.01 (nothing done)
- **Maximum possible score**: 0.99 (perfect episode)
- **Never exactly 0.0 or 1.0** — guaranteed by the formula + runtime assertion

| Grader | Weights |
|---|---|
| `grade_easy` | 60% completion + 40% on-time |
| `grade_medium` | 40% completion + 35% on-time + 25% energy efficiency |
| `grade_hard` | 35% completion + 30% on-time + 20% energy + 15% priority quality |
| `grade_performance` | Cross-difficulty aggregate used in `baseline_scores.json` |

> **grade_hard fix:** If an agent completes 0 tasks, the priority quality component now correctly scores 0. A prior bug used `max(1, completed_tasks)` as the denominator, which inflated scores for agents that did nothing. This is fixed.

---

## 7. Agent Modes

### Smart Agent (default)
A built-in deterministic baseline. No API keys required. Follows priority + deadline ordering with energy-aware break decisions. Run immediately with `python inference.py`.

### LLM Agent
Uses any OpenAI-compatible API. Configured via environment variables. Falls back to the smart agent automatically on API errors or invalid outputs.

> **Important:** Use `--strict-env` if you actually intend to test an LLM. Without it, a missing or wrong `HF_TOKEN` causes a silent fallback to the smart agent — you will get smart agent scores and think your LLM performed well.

**Fallback behaviour on API failure:**
- If the LLM API raises an exception (e.g. expired token, network error), exactly **one** error is logged on the step it fails.
- `llm_failed` is set to `True` and the episode silently continues with the smart agent for all remaining steps — no repeated error spam.
- The `model` label in `[END]` changes to `<model_name>→smart_agent` to make the switch visible in logs.
- If the LLM responds successfully but returns an unparseable or illegal action, the smart agent handles that single step and the LLM is retried on the next step (not a permanent fallback).

The LLM receives a structured system prompt covering:
- Energy-first decision rules
- Priority + deadline sorting strategy
- Reward penalties to avoid
- Hard override cases (e.g., work through low energy if deadline is imminent)
- A sliding conversation window (last 6 turn-pairs / 12 messages) enforced both inside `llm_agent()` and at the top of each step loop in `run_episode()`, to stay within small-model context limits

---

## 8. Episode Termination

An episode ends when ANY of these is true:

1. All tasks completed
2. Energy reaches 0 (burnout)
3. Time reaches 24h (end of day)
4. 5 consecutive invalid or noop steps (stagnation guard)

Maximum steps per episode: **10**

---

## 9. OpenEnv Compliance

| Requirement | Status |
|---|---|
| `reset()` returns typed `Observation` | Yes |
| `step()` returns `(obs, reward, done, info)` | Yes |
| `state` is a `@property` returning `Observation` | Yes |
| Typed Pydantic models (`Observation`, `Action`, `Reward`) | Yes |
| `legal_actions` field in `Observation` | Yes |
| `openenv.yaml` with `entry_point: env:FocusEnv` | Yes |
| 3 tasks with graders returning strictly `(0, 1)` | Yes |
| `inference.py` with OpenAI client | Yes |
| Structured logs `[START]` `[STEP]` `[END]` in `key=value` format | Yes |
| Dockerfile builds and runs | Yes |
| HF Space deployable | Yes |

---

## 10. How to Run

### Local — Smart Agent (no API key needed)
```bash
pip install -r requirements.txt
python inference.py
```

### Local — Single Difficulty
```bash
python inference.py --difficulty easy     # or medium / hard
python inference.py --difficulty all      # runs all three (default)
```

### Local — Custom Episode Count
```bash
python inference.py --num-episodes 5      # run 5 seeded episodes per difficulty
```

### Local — LLM Agent
```bash
pip install -r requirements.txt
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_xxx"
python inference.py --strict-env
```

### Docker
```bash
docker build -t focus-ai .
docker run --rm -p 7860:7860 focus-ai
```

> `.dockerignore` is included. It excludes `uv.lock`, `__pycache__/`, `.git/`, and `.md` files from the image, keeping the build lean.

### Self-test (env only)
```bash
python env.py
```

### With uv (recommended for reproducible installs)
```bash
uv sync          # installs from uv.lock — exact versions
uv run python inference.py
```

> **Python version note:** `uv.lock` was generated on Python 3.14 (dev). The Dockerfile runs Python 3.10. If you hit import errors in Docker, install from `requirements.txt` directly rather than `uv sync`.

---

## 11. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | No | `meta-llama/Llama-3.1-8B-Instruct` | Model to use |
| `HF_TOKEN` | For LLM mode | — | Hugging Face API token |

> If `HF_TOKEN` is not set, the runner falls back to the smart agent silently. Use `--strict-env` to make it fail loudly instead.

> Deprecated `api-inference.huggingface.co` URLs are auto-corrected to `router.huggingface.co/v1` at startup with a warning.

---

## 12. Structured Log Format

All episode output follows `key=value` format for machine parsing. All reward values are clamped to `[-20, +20]`.

**Smart agent (no LLM token):**
```
[START] task=easy env=focus_ai model=smart_agent
[STEP]  step=1 action=start_task('report') reward=20.0000 done=false error=null
[STEP]  step=2 action=start_task('inbox') reward=12.0000 done=true error=null
[END]   task=easy success=true steps=2 score=0.990 rewards=20.0000,12.0000
```

**LLM agent running cleanly:**
```
[START] task=medium env=focus_ai model=meta-llama/Llama-3.1-8B-Instruct
[STEP]  step=1 action=start_task('report') reward=20.0000 done=false error=null
...
[END]   task=medium success=true steps=5 score=0.875 rewards=...
```

**LLM token expires mid-episode (exactly one error, then silent fallback):**
```
[START] task=hard env=focus_ai model=meta-llama/Llama-3.1-8B-Instruct
[STEP]  step=1 action=start_task('report') reward=17.0000 done=false error=null
[STEP]  step=2 action=take_break(1) reward=4.0000 done=false error=null
ERROR   LLM error: 401 Unauthorized — switching episode to smart_agent
[STEP]  step=3 action=start_task('inbox') reward=12.0000 done=false error=null
...
[END]   task=hard success=false steps=10 score=0.431 model=meta-llama/Llama-3.1-8B-Instruct→smart_agent rewards=...
```

> The `→smart_agent` suffix in `model=` makes it immediately visible in logs that an LLM fallback occurred and at which episode it happened.

Stagnation termination also emits a dedicated line:
```
[END] task=hard termination_reason=stagnation step=7
```

---

## 13. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Status check |
| GET | `/health` | Health check |
| GET | `/tasks` | List task definitions from openenv.yaml |
| POST | `/reset` | Start new episode (body: `{"difficulty": "easy"}`) |
| POST | `/step` | Apply action (body: `{"action": "start_task('id')"}`) |
| GET | `/state` | Current observation + metrics + human-readable text |
| GET | `/grade/{difficulty}` | Score for the currently running episode |
| GET | `/validate` | OpenEnv compliance check (all must pass) |
| GET | `/score/{difficulty}` | Run one smart-agent episode and return its score |
| GET | `/benchmark` | Run smart-agent across all 3 difficulties with fixed seeds |
| GET | `/history` | Last 5 completed episode trajectories |

---

## 14. Project Structure

```
project/
├── env.py                  # FocusEnv class + smart_agent baseline
├── reward_and_tasks.py     # Reward function, task generators, graders, safe_score
├── inference.py            # Inference runner (LLM + smart agent)
├── models.py               # Pydantic models (Observation, Action, Reward, Task)
├── server/
│   └── app.py              # FastAPI server
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Project metadata + entry points
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies (pip)
├── uv.lock                 # Locked dependency tree (uv)
├── baseline_scores.json    # Reproducible benchmark results (smart agent)
└── README.md
```

---

## 15. HF Space Deployment

1. Create HF Space with SDK type **Docker**
2. Upload this folder — exclude `__pycache__/` and `.git/`
3. Set secrets: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`
4. Verify: `GET /health` returns `{"status":"healthy"}`

> Make sure `__pycache__/` is in your `.gitignore` before pushing. Committing compiled `.pyc` files causes unnecessary repo bloat and exposes your local Python version.

---

## 16. Baseline Scores (Smart Agent)

Results from 3 seeded evaluation episodes per difficulty. Seeds are fixed in `inference.py` (`EVAL_SEEDS`) to guarantee reproducibility. Run `python inference.py` to reproduce.

```json
{
  "overall_score": 0.630,
  "eval_seeds": {
    "easy":   [42, 43, 44],
    "medium": [7,  8,  9],
    "hard":   [13, 14, 15]
  },
  "difficulties": [
    { "difficulty": "easy",   "score": 0.990, "total_reward": 111.0 },
    { "difficulty": "medium", "score": 0.875, "total_reward": 125.0 },
    { "difficulty": "hard",   "score": 0.431, "total_reward":  95.0 }
  ]
}
```

Full per-episode breakdown is in `baseline_scores.json`.

> Hard scores vary significantly across seeds (0.32–0.61). This reflects genuine scenario variance — different seeds produce different priority mixes from the 12-task pool, not agent inconsistency.

---

## 17. Benchmarking Your LLM

To evaluate any OpenAI-compatible model against these baselines:

```bash
export HF_TOKEN="hf_xxx"
export MODEL_NAME="your-model-id"
export API_BASE_URL="https://router.huggingface.co/v1"
python inference.py --difficulty all --strict-env --num-episodes 3
```

Compare the resulting `baseline_scores.json` to the smart agent scores above. Targets to beat:

- **Easy (0.990):** Smart agent nearly maxes this. Beating it requires zero mistakes. Realistic.
- **Medium (0.875):** A well-prompted LLM should match or exceed this. Look for improvements in energy efficiency.
- **Hard (0.431):** Any LLM consistently above 0.55 is meaningfully better than the smart agent. This is where multi-step reasoning earns its keep.

---

## 18. Known Limitations

- **Hard difficulty is intentionally near-unsolvable at full completion.** 6 tasks + low starting energy + 10-step cap. The grader rewards proportional progress. This is by design, not a bug.
- **Task pool has 12 entries.** Hard samples 6. With 4 tasks of each priority level, random seeds can produce very unbalanced scenarios, causing high variance in hard episode scores.
- **No unit tests.** The environment is validated via `env.py` self-test and the `/validate` endpoint. If you modify `reward_and_tasks.py` or graders, manually verify the `safe_score()` contract holds.
- **Python version mismatch.** `uv.lock` was generated on Python 3.14 (dev). The Dockerfile runs Python 3.10. Use `pip install -r requirements.txt` in Docker, not `uv sync`.
- **`_current_trajectory` is unbounded within an episode.** Normal usage (max 10 steps) is fine. If you build a wrapper that calls `step()` in a loop without checking `done`, this list grows uncapped.