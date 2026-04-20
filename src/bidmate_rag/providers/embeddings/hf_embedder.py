"""Hugging Face embedding adapter.

sentence-transformers 라이브러리를 사용하여 로컬에서 임베딩을 생성한다.
API 호출 없이 로컬 GPU/CPU로 동작하므로 비용이 들지 않는다.
"""

from __future__ import annotations

from bidmate_rag.providers.embeddings.base import BaseEmbeddingProvider


class HFEmbedder(BaseEmbeddingProvider):
    """HuggingFace sentence-transformers 기반 임베딩 프로바이더."""

    def __init__(self, model_name: str, encode_fn=None, provider_name: str = "huggingface") -> None:
        """HFEmbedder를 초기화한다.

        Args:
            model_name: HuggingFace 모델 이름 (예: "jhgan/ko-sroberta-multitask").
            encode_fn: 외부 인코딩 함수 주입 (테스트용). None이면 자동 로딩.
            provider_name: 프로바이더 식별자.
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self._encode_fn = encode_fn

    def _get_encode_fn(self):
        """인코딩 함수를 반환한다. 없으면 SentenceTransformer 모델을 로딩.

        Returns:
            SentenceTransformer.encode 함수.

        Raises:
            ModuleNotFoundError: sentence-transformers가 설치되지 않은 경우.
        """
        if self._encode_fn is not None:
            return self._encode_fn
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "sentence-transformers is required for HF embedding. "
                "Install the ml dependency group."
            ) from exc
        model = SentenceTransformer(self.model_name)
        self._encode_fn = model.encode
        return self._encode_fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록을 로컬 모델로 임베딩한다.

        Args:
            texts: 임베딩할 텍스트 목록.

        Returns:
            각 텍스트에 대응하는 벡터(float 리스트)의 리스트.
        """
        embeddings = self._get_encode_fn()(texts)
        return [
            embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            for embedding in embeddings
        ]

    def embed_query(self, query: str) -> list[float]:
        """질문 하나를 임베딩 벡터로 변환한다.

        Args:
            query: 임베딩할 질문 문자열.

        Returns:
            질문에 대응하는 벡터(float 리스트).
        """
        return self.embed_documents([query])[0]
