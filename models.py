"""
models.py
---------
Pydantic v2 data models for the FocusAI Task Scheduling Environment.

Three core models:
  - Task         : a single work item in the agent's day
  - Observation  : full state visible to the agent each step
  - Reward       : step reward with diagnostic breakdown

One utility model:
  - Action       : thin wrapper around a raw action string
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single schedulable task in the agent's workday."""

    id: str = Field(description="Unique task identifier used in action strings")
    name: str = Field(description="Human-readable task description")
    priority: str = Field(description="Priority level: 'high' | 'medium' | 'low'")
    deadline: int = Field(description="Hour of day by which the task must be done")
    duration: int = Field(description="Hours of work required to complete")
    category: str = Field(default="general", description="Task category")
    completed: bool = Field(default=False, description="Whether the task is finished")


class Observation(BaseModel):
    """Everything the agent can see at one timestep."""

    time: float = Field(description="Current hour of day (9.0 to 24.0)")
    energy: int = Field(description="Cognitive energy level (0 to 100)")
    energy_level: str = Field(description="Qualitative energy: 'low' | 'medium' | 'high'")
    tasks: List[Task] = Field(description="All tasks with current status")
    current_task: Optional[str] = Field(default=None, description="ID of the active task")
    recent_actions: List[str] = Field(default_factory=list, description="Last 3 actions")
    goal: str = Field(default="", description="Episode objective text")
    legal_actions: List[str] = Field(default_factory=list, description="Valid actions now")


class Action(BaseModel):
    """Wrapper for a single agent action string."""

    action: str = Field(description="Raw action string, e.g. start_task('report')")


class Reward(BaseModel):
    """Step reward with diagnostic breakdown."""

    reward: float = Field(description="Numeric reward for this step")
    task_completed: bool = Field(default=False, description="A task was finished")
    before_deadline: bool = Field(default=False, description="Finished before deadline")
    missed_deadline: bool = Field(default=False, description="Deadline was missed")
    priority: Optional[str] = Field(default=None, description="Priority of completed task")
