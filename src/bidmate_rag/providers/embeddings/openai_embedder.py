"""OpenAI embedding adapter.

OpenAI API(text-embedding-3-small 등)를 호출하여 텍스트를 벡터로 변환한다.
배치당 30만 토큰 한도가 있으므로 build_index에서 배치 크기를 조절해야 한다.
"""

from __future__ import annotations

import os

from openai import OpenAI

from bidmate_rag.providers.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbedder(BaseEmbeddingProvider):
    """OpenAI API 기반 임베딩 프로바이더."""

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        client: OpenAI | None = None,
        provider_name: str = "openai",
    ) -> None:
        """OpenAIEmbedder를 초기화한다.

        Args:
            model_name: OpenAI 임베딩 모델명 (예: "text-embedding-3-small").
            api_base: API 베이스 URL (커스텀 엔드포인트 사용 시).
            api_key_env: API 키를 읽을 환경변수 이름.
            client: 외부 OpenAI 클라이언트 주입 (테스트용).
            provider_name: 프로바이더 식별자.
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_base = api_base
        self.client = client or OpenAI(
            api_key=os.getenv(api_key_env, "EMPTY"),
            base_url=api_base,
        )
        self.cumulative_tokens: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록을 OpenAI API로 임베딩한다.

        Args:
            texts: 임베딩할 텍스트 목록.

        Returns:
            각 텍스트에 대응하는 벡터(float 리스트)의 리스트.
        """
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        usage = getattr(response, "usage", None)
        self.cumulative_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """질문 하나를 임베딩 벡터로 변환한다.

        Args:
            query: 임베딩할 질문 문자열.

        Returns:
            질문에 대응하는 벡터(float 리스트).
        """
        return self.embed_documents([query])[0]
