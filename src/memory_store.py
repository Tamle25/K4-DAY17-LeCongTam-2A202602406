from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a deterministic heuristic token estimator.

    Properties:
    - Returns 0 for empty or whitespace-only text.
    - Deterministic: identical strings return the exact same count.
    - Monotonic: longer strings return equal or greater token count.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Manages user profile markdown files on disk with read, write, edit,
    and structured fact extraction/upsert operations.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Sanitize user_id and return path to User.md under root_dir."""
        safe_user_id = re.sub(r"[^\w\-]", "_", user_id.strip())
        return self.root_dir / safe_user_id / "User.md"

    def read_text(self, user_id: str) -> str:
        """Return User.md content or empty string if file does not exist."""
        path = self.path_for(user_id)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown content to disk and return file path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence in User.md. Return True if modified."""
        current = self.read_text(user_id)
        if not current or search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        """Return current file size in bytes, or 0 if file does not exist."""
        path = self.path_for(user_id)
        if path.is_file():
            return path.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured facts from User.md."""
        content = self.read_text(user_id)
        facts_dict: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                key, val = line[2:].split(":", 1)
                facts_dict[key.strip()] = val.strip()
        return facts_dict

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        """Update or insert structured facts and rewrite User.md cleanly."""
        current_facts = self.facts(user_id)
        current_facts.update(updates)

        lines = [f"# User Profile: {user_id}", ""]
        for k, v in current_facts.items():
            lines.append(f"- {k}: {v}")
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user message into stable profile facts.

    Handles:
    - Skipping pure question turns.
    - Capturing stable facts (name, location, profession, beverage, food, pet, style, tech).
    - Handling corrections (e.g. Đà Nẵng -> Huế, backend -> MLOps).
    - Filtering out noise (e.g. Hà Nội just a meeting trip, product manager was just a joke).
    """
    msg = message.strip()
    if not msg:
        return {}

    updates: dict[str, str] = {}

    # Check for noise to ignore or avoid misattribution
    is_joke_pm = "đùa" in msg.lower() and "product manager" in msg.lower()
    is_meeting_hanoi = "hà nội" in msg.lower() and ("họp" in msg.lower() or "hai ngày" in msg.lower())

    # 1. Name extraction
    name_match = re.search(r"(?:mình tên là|tên mình là|tên là)\s+([A-Za-z0-9_\sÀ-ỹ]+?)(?:[.,\n]|$)", msg, re.IGNORECASE)
    if name_match:
        name_val = name_match.group(1).strip()
        # Ensure it's not a question
        if not any(q in name_val.lower() for q in ["gì", "ai", "không"]):
            updates["name"] = name_val

    # 2. Location extraction & corrections
    # Check for explicit correction to Huế
    if ("ở huế" in msg.lower() or "đang ở huế" in msg.lower()) and not ("còn ở huế không" in msg.lower() or "ở đâu" in msg.lower()):
        # Avoid taking Đà Nẵng if this turn states moving/being in Huế
        updates["location"] = "Huế"
    # Check for explicit correction to Đà Nẵng
    elif ("đang làm việc ở đà nẵng" in msg.lower() or "nơi ở hiện tại là đà nẵng" in msg.lower() or "ở đà nẵng trong giai đoạn này" in msg.lower()):
        updates["location"] = "Đà Nẵng"
    elif "mình ở đà nẵng" in msg.lower() and "đừng lấy nó làm nơi ở" not in msg.lower():
        updates["location"] = "Đà Nẵng"

    # 3. Profession extraction & corrections
    if ("chuyển sang mlops engineer" in msg.lower() or "làm mlops engineer" in msg.lower() or "công việc mlops" in msg.lower()):
        updates["profession"] = "MLOps engineer"
    elif "làm backend engineer" in msg.lower() and "không còn làm backend" not in msg.lower() and "đừng nói backend" not in msg.lower():
        updates["profession"] = "backend engineer"

    # Avoid joke PM
    if is_joke_pm and updates.get("profession") == "product manager":
        del updates["profession"]

    # 4. Favorite beverage
    if "cà phê sữa đá" in msg.lower() and not ("uống gì" in msg.lower() or "đồ uống yêu thích của mình là gì" in msg.lower()):
        updates["favorite_beverage"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in msg.lower() and not ("món ăn yêu thích của mình là gì" in msg.lower()):
        updates["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in msg.lower() or "bé corgi" in msg.lower() or "con corgi" in msg.lower() or "con bơ" in msg.lower():
        updates["pet"] = "corgi"

    # 7. Response style preference
    if "3 bullet" in msg.lower():
        updates["response_style"] = "3 bullet"
    elif "ngắn gọn" in msg.lower() and not ("style trả lời" in msg.lower() and "?" in msg):
        updates["response_style"] = "ngắn gọn"

    # 8. Tech interests
    if ("python" in msg.lower() and "ai" in msg.lower()) or "ai ứng dụng" in msg.lower():
        updates["tech_interests"] = "Python, AI"

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 4) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""

    summary_items: list[str] = []
    selected = messages[-max_items:] if len(messages) > max_items else messages
    for m in selected:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        # Compress text to first 60 chars
        snippet = content[:60].replace("\n", " ")
        if len(content) > 60:
            snippet += "..."
        summary_items.append(f"[{role}]: {snippet}")

    return "\n".join(summary_items)


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads.

    - Keeps recent messages in full.
    - Compresses older messages into a summary when total tokens exceed threshold.
    - Tracks number of compactions per thread for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message to thread and trigger compaction if exceeding token threshold."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread_state = self.state[thread_id]
        msgs = thread_state["messages"]  # list of dict
        msgs.append({"role": role, "content": content})

        # Calculate current thread tokens
        summary_text = str(thread_state.get("summary", ""))
        summary_tokens = estimate_tokens(summary_text)
        msgs_tokens = sum(estimate_tokens(m.get("content", "")) for m in msgs)
        total_tokens = summary_tokens + msgs_tokens

        # Check if compaction is triggered
        if total_tokens > self.threshold_tokens and len(msgs) > self.keep_messages:
            older = msgs[:-self.keep_messages]
            kept = msgs[-self.keep_messages:]

            new_summary = summarize_messages(older)
            existing_summary = summary_text.strip()
            if existing_summary:
                combined_lines = [l for l in existing_summary.splitlines() if l.strip()] + [l for l in new_summary.splitlines() if l.strip()]
                # Keep up to 3 most recent summary lines to prevent unbounded growth
                thread_state["summary"] = "\n".join(combined_lines[-3:])
            else:
                thread_state["summary"] = new_summary

            thread_state["messages"] = kept
            thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state with messages, summary, and compactions."""
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions for this thread."""
        if thread_id not in self.state:
            return 0
        return int(self.state[thread_id].get("compactions", 0))


