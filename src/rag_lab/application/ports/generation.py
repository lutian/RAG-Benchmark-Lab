"""Chat model port: the LLM that writes the answer from the numbered sources."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    model: str
    # Token usage when the provider reports it (some local servers do not).
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatModel(Protocol):
    """Implementations raise ``GenerationProviderError`` on failure."""

    @property
    def name(self) -> str: ...

    def complete(self, messages: Sequence[ChatMessage]) -> Completion: ...
