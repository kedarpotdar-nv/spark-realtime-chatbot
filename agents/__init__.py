"""Background agent bridges for Spark Realtime."""

from .base import AgentBridge, AgentEvent, AgentTask, ArtifactInfo, TaskStatus
from .hermes import HermesAgentBridge, get_hermes_bridge

__all__ = [
    "AgentBridge",
    "AgentEvent",
    "AgentTask",
    "ArtifactInfo",
    "TaskStatus",
    "HermesAgentBridge",
    "get_hermes_bridge",
]
