from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A - Baseline Agent.

    Requirements:
    - Within-session memory only (keyed strictly by thread_id)
    - No persistent User.md
    - Forgets all facts across new threads
    - Zero compactions
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to live langchain agent if available, otherwise deterministic offline."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Live fallback path
                session = self.sessions.setdefault(thread_id, SessionState())
                current_prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
                session.prompt_tokens_processed += current_prompt_tokens
                session.messages.append({"role": "user", "content": message})

                ai_msg = self.langchain_agent.invoke(message)
                resp_text = getattr(ai_msg, "content", str(ai_msg))
                resp_tokens = estimate_tokens(resp_text)
                session.token_usage += resp_tokens
                session.messages.append({"role": "assistant", "content": resp_text})
                return {
                    "response": resp_text,
                    "tokens": resp_tokens,
                    "prompt_tokens": current_prompt_tokens,
                }
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count generated in one thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate cumulative prompt context kept and processed across turns in thread."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory, always returns 0."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic within-thread memory behavior.

        - Stores user message in thread session.
        - Calculates prompt tokens from all previous messages in this thread + new message.
        - Responds only based on current thread context.
        - Forgets all facts across different thread_ids.
        """
        session = self.sessions.setdefault(thread_id, SessionState())

        # Measure prompt context load for this turn
        current_prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += current_prompt_tokens

        # Record incoming message
        session.messages.append({"role": "user", "content": message})

        # Generate response based STRICTLY on current thread's messages
        lower_msg = message.lower()
        is_question = any(q in lower_msg for q in ["?", "gì", "ai", "đâu", "nào", "nhắc lại", "nhớ", "biết"])

        # Check if the requested info was mentioned in THIS CURRENT THREAD
        prior_thread_text = " ".join(m["content"] for m in session.messages[:-1])

        if is_question:
            if not prior_thread_text:
                # Fresh thread: Baseline has no memory of past threads
                response_text = "Tôi không có thông tin về bạn trong phiên trò chuyện mới này. Bạn có thể chia sẻ lại không?"
            else:
                # Answer from current thread context if available
                response_text = f"Dựa trên các tin nhắn trong phiên này, tôi ghi nhận yêu cầu của bạn: {message[:60]}."
        else:
            response_text = "Tôi đã ghi nhận thông tin của bạn trong phiên làm việc hiện tại."

        resp_tokens = estimate_tokens(response_text)
        session.token_usage += resp_tokens
        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "tokens": resp_tokens,
            "prompt_tokens": current_prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire live chat model when valid config and keys are available."""
        if self.force_offline:
            return None
        try:
            if not self.config.model or not self.config.model.api_key:
                return None
            return build_chat_model(self.config.model)
        except Exception:
            return None

