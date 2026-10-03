from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Contains:
    - Paths for the repo root, dataset directory, and state directory.
    - Compact-memory settings (threshold and keep messages).
    - Provider settings for model and judge_model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int = 800
    compact_keep_messages: int = 4
    model: ProviderConfig = None  # type: ignore
    judge_model: ProviderConfig = None  # type: ignore


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    Steps:
    1. Resolve the repo root or default to the parent of src/.
    2. Load values from `.env` via python-dotenv if present.
    3. Ensure `state/` directory exists.
    4. Populate and return LabConfig with sane defaults.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env file from repo root
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Ensure state directory exists
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    # Compact memory knobs
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Provider knobs
    raw_provider = os.getenv("LLM_PROVIDER", "openai")
    try:
        provider = normalize_provider(raw_provider)
    except ValueError:
        provider = "openai"

    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("CUSTOM_API_KEY")
    )
    base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        api_key=api_key,
        base_url=base_url,
    )

    judge_raw_provider = os.getenv("JUDGE_PROVIDER", provider)
    try:
        judge_provider = normalize_provider(judge_raw_provider)
    except ValueError:
        judge_provider = provider

    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_cfg,
        judge_model=judge_cfg,
    )

