from bidmate_rag.config.settings import ProviderConfig
from bidmate_rag.providers.llm.registry import build_embedding_provider, build_llm_provider


def test_registry_builds_openai_and_local_llm_providers() -> None:
    openai_provider = build_llm_provider(
        ProviderConfig(provider="openai", model="gpt-5-mini", api_base="https://api.openai.com/v1")
    )
    local_provider = build_llm_provider(
        ProviderConfig(
            provider="local-openai-compat",
            model="local-model",
            api_base="http://localhost:8000/v1",
        )
    )

    assert openai_provider.provider_name == "openai"
    assert local_provider.provider_name == "local-openai-compat"


def test_registry_builds_local_hf_llm_provider() -> None:
    provider = build_llm_provider(
        ProviderConfig(
            provider="huggingface",
            model="Qwen/Qwen2.5-3B-Instruct",
        )
    )

    assert provider.provider_name == "huggingface"


def test_registry_builds_hf_and_openai_embedding_providers() -> None:
    hf_provider = build_embedding_provider(
        ProviderConfig(
            provider="huggingface",
            model="Qwen/Qwen2.5-3B-Instruct",
            embedding_model="BAAI/bge-m3",
        )
    )
    openai_provider = build_embedding_provider(
        ProviderConfig(
            provider="openai",
            model="gpt-5-mini",
            embedding_model="text-embedding-3-small",
        )
    )

    assert hf_provider.model_name == "BAAI/bge-m3"
    assert openai_provider.model_name == "text-embedding-3-small"
