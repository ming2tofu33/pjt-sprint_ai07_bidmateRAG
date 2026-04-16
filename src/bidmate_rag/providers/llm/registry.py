"""Provider registry."""

from __future__ import annotations

from bidmate_rag.config.settings import ProviderConfig
from bidmate_rag.providers.embeddings.hf_embedder import HFEmbedder
from bidmate_rag.providers.embeddings.openai_embedder import OpenAIEmbedder
from bidmate_rag.providers.llm.hf_local import HFLocalLLM
from bidmate_rag.providers.llm.openai_compat import OpenAICompatibleLLM


def build_llm_provider(config: ProviderConfig, adapter_path = None):
    if config.provider in {"openai", "local-openai-compat", "openai-compat"}:
        return OpenAICompatibleLLM(
            provider_name=config.provider,
            model_name=config.model,
            api_base=config.api_base,
        )
    if config.provider in {"huggingface", "local-hf"}:
        return HFLocalLLM(
            provider_name=config.provider,
            model_name=config.model,
            adapter_path=adapter_path,
        )
    raise ValueError(f"Unsupported llm provider: {config.provider}")


def build_embedding_provider(config: ProviderConfig):
    embedding_model = config.embedding_model or config.model
    if config.provider in {"openai", "openai-compat"}:
        return OpenAIEmbedder(
            provider_name=config.provider,
            model_name=embedding_model,
            api_base=config.api_base,
        )
    if config.provider in {"huggingface", "local-hf", "local-openai-compat"}:
        return HFEmbedder(
            provider_name=config.provider,
            model_name=embedding_model,
        )
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
