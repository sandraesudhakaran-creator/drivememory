"""A minimal implementation of the Agent2Agent (A2A) protocol's core objects.

Agents publish an ``AgentCard`` describing their skills and exchange
``Message`` objects (made of text/data ``Part``s). Each request produces a
``Task`` with a status and output ``artifacts``. In-process calls go through
the ``AgentBus``; the FastAPI app exposes the same agents over HTTP as
JSON-RPC ``message/send`` endpoints with agent cards at
``/a2a/<agent>/.well-known/agent-card.json``.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class AgentSkill:
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    name: str
    description: str
    skills: list[AgentSkill]
    version: str = "0.1.0"
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocolVersion": "0.3.0",
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": ["application/json", "text/plain"],
            "defaultOutputModes": ["application/json", "text/plain"],
            "skills": [asdict(s) for s in self.skills],
        }


@dataclass
class Part:
    kind: Literal["text", "data"]
    text: str | None = None
    data: dict | None = None

    @classmethod
    def of_text(cls, text: str) -> Part:
        return cls("text", text=text)

    @classmethod
    def of_data(cls, data: dict) -> Part:
        return cls("data", data=data)


@dataclass
class Message:
    role: Literal["user", "agent"]
    parts: list[Part]
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    context_id: str | None = None

    def data(self) -> dict:
        for p in self.parts:
            if p.kind == "data" and p.data is not None:
                return p.data
        return {}

    def text(self) -> str:
        return " ".join(p.text for p in self.parts if p.kind == "text" and p.text)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "messageId": self.message_id,
            "contextId": self.context_id,
            "parts": [{k: v for k, v in asdict(p).items() if v is not None} for p in self.parts],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Message:
        return cls(
            role=d.get("role", "user"),
            parts=[Part(p["kind"], p.get("text"), p.get("data")) for p in d.get("parts", [])],
            message_id=d.get("messageId") or uuid.uuid4().hex,
            context_id=d.get("contextId"),
        )


@dataclass
class Task:
    id: str
    context_id: str | None
    state: Literal["submitted", "working", "completed", "failed"] = "submitted"
    history: list[Message] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    error: str | None = None

    def result(self) -> dict:
        """Merged data of all artifacts (convenience for in-process callers)."""
        out: dict[str, Any] = {}
        for a in self.artifacts:
            for p in a.get("parts", []):
                if p.get("kind") == "data":
                    out.update(p["data"])
        return out

    def to_dict(self) -> dict:
        return {
            "kind": "task",
            "id": self.id,
            "contextId": self.context_id,
            "status": {"state": self.state, **({"message": self.error} if self.error else {})},
            "history": [m.to_dict() for m in self.history],
            "artifacts": self.artifacts,
        }


class Agent(ABC):
    card: AgentCard

    def handle(self, message: Message) -> Task:
        task = Task(id=uuid.uuid4().hex, context_id=message.context_id, state="working", history=[message])
        try:
            data = self.run(message)
            task.artifacts.append({"artifactId": uuid.uuid4().hex, "name": f"{self.card.name}-result",
                                   "parts": [{"kind": "data", "data": data}]})
            task.state = "completed"
        except Exception as e:  # surface failures as A2A task errors, not exceptions
            task.state = "failed"
            task.error = f"{type(e).__name__}: {e}"
        return task

    @abstractmethod
    def run(self, message: Message) -> dict: ...


class AgentBus:
    """Routes A2A messages to registered agents (in-process transport)."""

    def __init__(self):
        self.agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self.agents[agent.card.name] = agent

    def send(self, agent_name: str, message: Message) -> Task:
        if agent_name not in self.agents:
            raise KeyError(f"unknown agent {agent_name!r}")
        return self.agents[agent_name].handle(message)

    def cards(self) -> list[dict]:
        return [a.card.to_dict() for a in self.agents.values()]
