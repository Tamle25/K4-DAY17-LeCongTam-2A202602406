from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider name or aliases (e.g. `anthorpic` -> `anthropic`) to standard name."""
    norm = value.strip().lower()
    alias_map = {
        "openai": "openai",
        "gpt": "openai",
        "custom": "custom",
        "local": "custom",
        "compatible": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "anthorpic": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
    }
    if norm in alias_map:
        return alias_map[norm]
    raise ValueError(f"Unsupported provider: '{value}'. Expected one of {list(set(alias_map.values()))}")


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider.

    Supported:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("CUSTOM_API_KEY") or "EMPTY",
            base_url=config.base_url or os.getenv("CUSTOM_BASE_URL"),
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("GEMINI_API_KEY"),
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model_name=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or os.getenv("OLLAMA_BASE_URL"),
        )
    elif provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("OPENROUTER_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

