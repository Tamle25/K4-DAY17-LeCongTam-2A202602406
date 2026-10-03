from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0, 0.5, or 1.0 depending on how many expected facts appear.

    - 1.0: all expected items appear in the answer.
    - 0.5: some (at least 1) expected items appear in the answer.
    - 0.0: none of the expected items appear.
    """
    if not expected:
        return 1.0
    lower_ans = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in lower_ans)
    ratio = matches / len(expected)
    if ratio == 1.0:
        return 1.0
    elif ratio > 0.0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode based on recall, structure and conciseness."""
    if not answer.strip():
        return 0.0
    base_recall = recall_points(answer, expected)
    has_structure = "- " in answer or "\n" in answer
    length_ok = 20 <= len(answer) <= 500
    structure_score = 0.2 if has_structure else 0.1
    length_score = 0.2 if length_ok else 0.1
    total_score = min(1.0, base_recall * 0.6 + structure_score + length_score)
    return round(total_score, 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over conversations and recall questions.

    1. Feed all turns to the agent in conversation threads.
    2. Track cumulative agent tokens and prompt tokens.
    3. Ask recall questions in fresh threads.
    4. Compute average recall and quality.
    5. Record memory file growth and compaction count.
    """
    user_ids = {c["user_id"] for c in conversations}
    initial_mem_size = 0
    if hasattr(agent, "memory_file_size"):
        initial_mem_size = sum(agent.memory_file_size(u) for u in user_ids)

    recall_scores: list[float] = []
    quality_scores: list[float] = []
    all_threads: list[str] = []

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        chat_thread = f"{conv_id}_chat"
        all_threads.append(chat_thread)

        # 1. Feed dialogue turns
        for turn in conv.get("turns", []):
            agent.reply(user_id, chat_thread, turn)

        # 2. Ask recall questions in a FRESH thread
        for idx, q_item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{conv_id}_recall_{idx}"
            all_threads.append(recall_thread)
            question = q_item["question"]
            expected = q_item.get("expected_contains", [])

            resp_dict = agent.reply(user_id, recall_thread, question)
            ans_text = resp_dict.get("response", "")

            rec_score = recall_points(ans_text, expected)
            qual_score = heuristic_quality(ans_text, expected)
            recall_scores.append(rec_score)
            quality_scores.append(qual_score)

    # Aggregate token metrics
    agent_tokens_only = sum(agent.token_usage(t) for t in all_threads)
    prompt_tokens_processed = sum(agent.prompt_token_usage(t) for t in all_threads)
    compactions = sum(agent.compaction_count(t) for t in all_threads)

    final_mem_size = 0
    if hasattr(agent, "memory_file_size"):
        final_mem_size = sum(agent.memory_file_size(u) for u in user_ids)
    memory_growth_bytes = max(0, final_mem_size - initial_mem_size)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens_only,
        prompt_tokens_processed=prompt_tokens_processed,
        recall_score=round(avg_recall, 2),
        response_quality=round(avg_quality, 2),
        memory_growth_bytes=memory_growth_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print tabulated benchmark results with 6 core metric columns."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            f"{r.memory_growth_bytes:,}",
            r.compactions,
        ])
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both Standard Benchmark and Long-Context Stress Benchmark."""
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    std_convs = load_conversations(std_data_path)
    stress_convs = load_conversations(stress_data_path)

    # Clean state before benchmark runs to measure memory growth cleanly
    profiles_dir = config.state_dir / "profiles"
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)

    print("=" * 80)
    print("STANDARD BENCHMARK (data/conversations.json)")
    print("=" * 80)
    baseline_std = BaselineAgent(config, force_offline=True)
    advanced_std = AdvancedAgent(config, force_offline=True)
    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, std_convs, config),
        run_agent_benchmark("Advanced", advanced_std, std_convs, config),
    ]
    print(format_rows(std_rows))
    print()

    print("=" * 80)
    print("LONG-CONTEXT STRESS BENCHMARK (data/advanced_long_context.json)")
    print("=" * 80)
    # Clean state for stress test isolation
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    baseline_stress = BaselineAgent(config, force_offline=True)
    advanced_stress = AdvancedAgent(config, force_offline=True)
    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print(format_rows(stress_rows))
    print()


if __name__ == "__main__":
    main()

