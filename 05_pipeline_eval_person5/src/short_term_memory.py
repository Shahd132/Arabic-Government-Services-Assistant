from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, List, Tuple


@dataclass
class ConversationMemory:
    """
    Short-term conversation memory.

    Stores the most recent N conversation turns for each session.

    Each turn contains:
        (user_question, assistant_answer)
    """

    window_size: int = 5

    _store: Dict[str, Deque[Tuple[str, str]]] = field(
        default_factory=dict
    )

    _lock: Lock = field(
        default_factory=Lock
    )

    def get_history(
        self,
        session_id: str
    ) -> List[Tuple[str, str]]:
        """
        Return recent conversation history.

        Each item is:
            (question, answer)
        """

        with self._lock:
            turns = self._store.get(
                session_id,
                deque()
            )

            return list(turns)

    def get_questions(
        self,
        session_id: str
    ) -> List[str]:
        """
        Return only the user's previous questions.

        Useful when only questions are needed.
        """

        with self._lock:
            turns = self._store.get(
                session_id,
                deque()
            )

            return [
                question
                for question, _answer in turns
            ]

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str
    ) -> None:
        """
        Add a new conversation turn.

        The deque automatically removes the oldest turn
        when window_size is reached.
        """

        if not session_id:
            raise ValueError(
                "session_id cannot be empty"
            )

        question = question.strip()
        answer = answer.strip()

        if not question:
            raise ValueError(
                "question cannot be empty"
            )

        with self._lock:

            if session_id not in self._store:
                self._store[session_id] = deque(
                    maxlen=self.window_size
                )

            self._store[session_id].append(
                (question, answer)
            )

    def clear(
        self,
        session_id: str
    ) -> None:
        """
        Clear all conversation history
        for a specific session.
        """

        with self._lock:
            self._store.pop(
                session_id,
                None
            )

    def has_history(
        self,
        session_id: str
    ) -> bool:
        """
        Check whether a session has conversation history.
        """

        with self._lock:
            return bool(
                self._store.get(
                    session_id,
                    deque()
                )
            )

    def size(
        self,
        session_id: str
    ) -> int:
        """
        Return number of stored turns for a session.
        """

        with self._lock:
            return len(
                self._store.get(
                    session_id,
                    deque()
                )
            )


# One shared memory instance for the whole process.
memory = ConversationMemory(
    window_size=5
)