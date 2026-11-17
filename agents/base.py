"""Abstract base classes and data structures for background agent bridges.

Designed so Hermes Agent (Nous Research), OpenCode, or other autonomous
coding/research agents can be plugged behind a uniform async interface.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Coroutine, Dict, List, Optional


class TaskStatus(str, Enum):
    """Lifecycle states of a delegated background task."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class AgentEvent:
    """Streaming event emitted by an agent task."""
    task_id: str
    event_type: str  # "task.started", "tool.started", "tool.completed", "tool.failed", "thinking.delta", "message.delta", "artifact.created", "task.completed", "task.failed", "task.cancelled"
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            **self.data,
        }


@dataclass
class ArtifactInfo:
    """Metadata about a file in the shared agent workspace."""
    path: str
    name: str
    size: int
    modified_at: float
    mime_type: str
    previewable: bool
    task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "modified_at": self.modified_at,
            "mime_type": self.mime_type,
            "previewable": self.previewable,
            "task_id": self.task_id,
        }


@dataclass
class AgentTask:
    """Represents an in-flight or completed background agent task."""
    task_id: str
    agent_id: str
    task: str
    context: str = ""
    category: str = "general"  # "coding", "research", "project", "general"
    status: TaskStatus = TaskStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    remote_run_id: Optional[str] = None
    session_id: Optional[str] = None
    output: str = ""
    spoken_summary: str = ""
    error: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    workspace_dir: str = ""

    @classmethod
    def create(
        cls,
        agent_id: str,
        task: str,
        context: str = "",
        category: str = "general",
        workspace_dir: str = "",
    ) -> "AgentTask":
        return cls(
            task_id=f"task_{uuid.uuid4().hex[:10]}",
            agent_id=agent_id,
            task=task,
            context=context,
            category=category,
            workspace_dir=workspace_dir,
        )

    @property
    def duration_ms(self) -> Optional[int]:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return int((end - self.started_at) * 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "task": self.task,
            "context": self.context,
            "category": self.category,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "remote_run_id": self.remote_run_id,
            "session_id": self.session_id,
            "output": self.output,
            "spoken_summary": self.spoken_summary,
            "error": self.error,
            "steps": self.steps,
            "artifacts": self.artifacts,
            "workspace_dir": self.workspace_dir,
        }


EventCallback = Callable[[AgentEvent], Coroutine[Any, Any, None]]


class AgentBridge(ABC):
    """Abstract interface for background agent backends (Hermes, OpenCode, etc.)."""

    agent_id: str = "base"
    display_name: str = "Base Agent"

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if the agent backend is configured via environment/config."""

    @abstractmethod
    async def check_status(self) -> Dict[str, Any]:
        """Return current availability, transport mode, and active task count."""

    @abstractmethod
    async def warmup(self) -> None:
        """Warm up persistent HTTP/RPC session and probe capabilities."""

    @abstractmethod
    async def submit_task(
        self,
        task: str,
        context: str = "",
        category: str = "general",
        session_key: Optional[str] = None,
        on_event: Optional[EventCallback] = None,
    ) -> AgentTask:
        """Dispatch a task asynchronously in the background and return immediately."""

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task by ID."""

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[AgentTask]:
        """Get a task by ID."""

    @abstractmethod
    def list_tasks(self) -> List[AgentTask]:
        """List all tracked tasks (newest first)."""

    @abstractmethod
    def list_artifacts(self) -> List[ArtifactInfo]:
        """List generated files in the shared workspace."""
