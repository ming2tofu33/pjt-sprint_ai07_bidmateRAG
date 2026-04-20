"""Common interfaces for embedding providers.

모든 임베딩 프로바이더(OpenAI, HuggingFace 등)가 상속하는 추상 클래스.
embed_documents와 embed_query 두 메서드를 구현해야 한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """임베딩 프로바이더 추상 클래스."""

    provider_name: str   # 프로바이더 이름 (예: "openai", "huggingface")
    model_name: str      # 모델 이름 (예: "text-embedding-3-small")

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록을 벡터 목록으로 변환한다.

        Args:
            texts: 임베딩할 텍스트 목록.

        Returns:
            각 텍스트에 대응하는 벡터(float 리스트)의 리스트.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """질문 하나를 벡터로 변환한다.

        Args:
            query: 임베딩할 질문 문자열.

        Returns:
            질문에 대응하는 벡터(float 리스트).
        """
        raise NotImplementedError
