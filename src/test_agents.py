from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests pointing state_dir to tmp_path and low threshold."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=60,
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="gpt-4o-mini"),
        judge_model=ProviderConfig(provider="openai", model_name="gpt-4o-mini"),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, read, and edited with corrections."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # 1. Initial state (non-existent file returns empty string)
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # 2. Write initial profile
    initial_content = "# User Profile: test_user\n\n- name: DũngCT\n- location: Đà Nẵng\n"
    path = store.write_text(user_id, initial_content)
    assert path.is_file()
    assert store.file_size(user_id) > 0
    assert "Đà Nẵng" in store.read_text(user_id)

    # 3. Edit profile with correction
    edited = store.edit_text(user_id, "location: Đà Nẵng", "location: Huế")
    assert edited is True
    updated_text = store.read_text(user_id)
    assert "location: Huế" in updated_text
    assert "Đà Nẵng" not in updated_text


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction and preserve recent messages."""
    manager = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    thread_id = "test_compact_thread"

    # Append long messages to exceed the 60-token threshold
    for i in range(8):
        manager.append(
            thread_id,
            "user",
            f"Tin tức dài số {i}: NASA Artemis III bay quanh mặt trăng với kế hoạch chuẩn bị tích hợp kỹ thuật dài.",
        )

    assert manager.compaction_count(thread_id) > 0
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 2
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify Advanced remembers across sessions and Baseline strictly does not."""
    config = make_config(tmp_path)
    user_id = "dungct"

    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    # Thread 1: Introduction turn
    intro_msg = "Chào bạn, mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá."
    advanced.reply(user_id, "thread_1", intro_msg)
    baseline.reply(user_id, "thread_1", intro_msg)

    # Thread 2: Recall in a fresh session
    recall_msg = "Mình tên gì và đồ uống yêu thích là gì?"
    adv_resp = advanced.reply(user_id, "thread_2", recall_msg)["response"]
    base_resp = baseline.reply(user_id, "thread_2", recall_msg)["response"]

    # Advanced MUST remember facts from previous thread via User.md
    assert "DũngCT" in adv_resp
    assert "cà phê sữa đá" in adv_resp

    # Baseline MUST forget across different threads
    assert "DũngCT" not in base_resp
    assert "cà phê sữa đá" not in base_resp


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Quantitatively prove compact memory reduces prompt context load on long threads."""
    config = make_config(tmp_path)
    user_id = "dungct_stress"
    thread_id = "stress_thread"

    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    long_turns = [
        "Chào bạn, đây là thử nghiệm hội thoại dài về hệ thống AI và quản trị dữ liệu lớn.",
        "Bài học từ NASA Artemis III là quản trị phụ thuộc kỹ thuật qua các cột mốc có thể kiểm chứng.",
        "Máy bay X-59 nghiên cứu giảm tiếng nổ siêu thanh để người dùng cuối cảm thấy thoải mái hơn.",
        "Cảnh báo khí hậu của WMO cho thấy xác suất El Nino tăng cao đòi hỏi chuẩn bị kịch bản ứng phó.",
        "Kế hoạch năng lượng British Columbia kết hợp tăng trưởng nhu cầu với tối ưu tiết kiệm điện.",
        "Ở công việc MLOps, giữ toàn bộ log và context thô sẽ khiến hệ thống chậm và đội chi phí token.",
        "Nén ngữ cảnh cũ thành summary và giữ lại recent messages là cách làm bền vững hơn nhiều.",
        "Mình muốn câu trả lời luôn ngắn gọn, có cấu trúc và phân tích rõ trade-off hệ thống.",
    ]

    for turn in long_turns:
        advanced.reply(user_id, thread_id, turn)
        baseline.reply(user_id, thread_id, turn)

    # 1. Verify compaction actually happened
    assert advanced.compaction_count(thread_id) > 0

    # 2. Verify prompt load of Advanced is strictly lower than Baseline
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)
    base_prompt_tokens = baseline.prompt_token_usage(thread_id)

    assert adv_prompt_tokens < base_prompt_tokens, (
        f"Advanced prompt tokens ({adv_prompt_tokens}) should be strictly less than "
        f"Baseline prompt tokens ({base_prompt_tokens})"
    )

