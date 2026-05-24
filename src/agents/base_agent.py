from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import AnnotationBundle, ConversationMessage


class BaseAgent(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def should_respond(self, bundle: AnnotationBundle) -> bool:
        raise NotImplementedError

    @abstractmethod
    def respond(self, bundle: AnnotationBundle) -> ConversationMessage:
        raise NotImplementedError
