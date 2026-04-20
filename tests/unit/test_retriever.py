from unittest.mock import MagicMock

from bidmate_rag.retrieval.filters import extract_matched_agencies, extract_project_clues
from bidmate_rag.retrieval.memory import ConversationMemory
from bidmate_rag.retrieval.retriever import RAGRetriever
from bidmate_rag.schema import Chunk, RetrievedChunk


class FakeEmbedder:
    provider_name = "fake-embedder"
    model_name = "fake-embedding-model"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, query_results: list[RetrievedChunk] | None = None):
        self.last_kwargs = None
        self.calls = []
        self.query_results = query_results

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls.append(kwargs)
        if self.query_results is not None:
            return self.query_results
        return [
            RetrievedChunk(
                rank=1,
                score=0.9,
                chunk=Chunk(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    text="요구사항",
                    text_with_meta="[발주기관: 기관 | 사업명: 사업]\n요구사항",
                    char_count=4,
                    section="요구사항",
                    content_type="table",
                    chunk_index=0,
                    metadata={"사업명": "사업", "발주 기관": "국민연금공단", "파일명": "doc-1.hwp"},
                ),
            )
        ]


class FakeMetadataStore:
    agency_list = ["국민연금공단", "기초과학연구원"]

    def find_relevant_docs(self, query: str, top_n: int = 3):
        return ["doc-1.hwp", "doc-2.hwp"]


class ProjectAwareFakeMetadataStore(FakeMetadataStore):
    def __init__(self, mapping: dict[str, list[str]]):
        self.mapping = mapping

    def find_relevant_docs(self, query: str, top_n: int = 3):
        for key, docs in self.mapping.items():
            if key in query:
                return docs[:top_n]
        return super().find_relevant_docs(query, top_n=top_n)


class EmptyShortlistMetadataStore:
    agency_list = ["국민연금공단", "기초과학연구원"]

    def find_relevant_docs(self, query: str, top_n: int = 3):
        return []


class FakeReranker:
    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.calls.append(pairs)
        return [self.scores_by_text[text] for _, text in pairs]


class FakeSparseStore:
    def __init__(self, query_results: list[RetrievedChunk] | None = None):
        self.calls: list[dict] = []
        self.query_results = query_results or []

    def query(self, *, query: str, top_k: int = 5, where: dict | None = None):
        self.calls.append({"query": query, "top_k": top_k, "where": where})
        return self.query_results[:top_k]


def _make_mock_llm(rewritten_text: str) -> MagicMock:
    from bidmate_rag.providers.llm.base import RewriteResponse

    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=rewritten_text,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )
    mock_llm.model_name = "gpt-5-mini"
    return mock_llm


def _retrieved_chunk(
    chunk_id: str,
    score: float,
    *,
    agency: str,
    section: str = "",
    content_type: str = "text",
    doc_id: str | None = None,
    file_name: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        score=score,
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id or f"{chunk_id}-doc",
            text=chunk_id,
            text_with_meta=chunk_id,
            char_count=len(chunk_id),
            section=section,
            content_type=content_type,
            chunk_index=0,
            metadata={
                "사업명": f"{chunk_id}-사업",
                "발주 기관": agency,
                "파일명": file_name or f"{chunk_id}.hwp",
            },
        ),
    )


class ScopedFakeVectorStore:
    def __init__(self, results_by_agency: dict[str, list[RetrievedChunk]]):
        self.results_by_agency = results_by_agency
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        agency = kwargs["where"]["발주 기관"]
        return self.results_by_agency[agency][: kwargs["top_k"]]


class ScopedFieldFakeVectorStore:
    def __init__(self, field_name: str, results_by_value: dict[str, list[RetrievedChunk]]):
        self.field_name = field_name
        self.results_by_value = results_by_value
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        value = kwargs["where"][self.field_name]
        return self.results_by_value[value][: kwargs["top_k"]]


class SequenceFakeVectorStore:
    def __init__(self, responses: list[list[RetrievedChunk]]):
        self.responses = responses
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return []


def test_retriever_merges_metadata_range_and_section_filters() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve("국민연금공단 2024년 5억 이상 보안 요구사항 알려줘", top_k=3)

    main_call = vector_store.calls[0]
    assert main_call["where"] == {
        "발주 기관": "국민연금공단",
        "사업 금액": {"$gte": 500000000},
        "공개연도": 2024,
    }
    assert main_call["where_document"] is None
    assert main_call["top_k"] == 9


def test_retriever_expands_candidate_pool_without_reranker() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        reranker_model=None,
    )

    retriever.retrieve("국민연금공단 보안 요구사항 알려줘", top_k=4)

    assert vector_store.last_kwargs["top_k"] == 12
    assert vector_store.last_kwargs["where_document"] is None


def test_retriever_trims_back_to_requested_top_k_without_reranker() -> None:
    vector_store = FakeVectorStore(
        [
            _retrieved_chunk("chunk-1", 0.95, agency="국민연금공단"),
            _retrieved_chunk("chunk-2", 0.94, agency="국민연금공단"),
            _retrieved_chunk("chunk-3", 0.93, agency="국민연금공단"),
        ]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        reranker_model=None,
    )

    results = retriever.retrieve("국민연금공단 요구사항 알려줘", top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["chunk-1", "chunk-2"]
    assert [result.rank for result in results] == [1, 2]


def test_retriever_uses_hybrid_rrf_when_sparse_store_enabled() -> None:
    vector_store = FakeVectorStore(
        [
            _retrieved_chunk("dense-a", 0.99, agency="국민연금공단"),
            _retrieved_chunk("shared", 0.97, agency="국민연금공단"),
        ]
    )
    sparse_store = FakeSparseStore(
        [
            _retrieved_chunk("shared", 1.0, agency="국민연금공단"),
            _retrieved_chunk("sparse-b", 0.9, agency="국민연금공단"),
        ]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        sparse_store=sparse_store,
        hybrid_config={
            "enabled": True,
            "dense_pool_multiplier": 3,
            "sparse_pool_multiplier": 3,
            "rrf_k": 60,
        },
    )

    results = retriever.retrieve("국민연금공단 shared", top_k=3)

    assert vector_store.last_kwargs["top_k"] == 9
    assert sparse_store.calls[0]["top_k"] == 9
    assert [result.chunk.chunk_id for result in results] == ["shared", "dense-a", "sparse-b"]
    assert results[0].chunk.metadata["retrieval_source"] == "hybrid"


def test_retriever_fans_out_metadata_store_shortlist_when_query_has_comparison_intent() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve("비슷한 사업 비교해줘", top_k=2)

    assert len(vector_store.calls) == 2
    assert [call["where"]["파일명"] for call in vector_store.calls] == [
        "doc-1.hwp",
        "doc-2.hwp",
    ]
    assert results[0].score > 0


def test_retriever_fans_out_generic_shortlist_when_query_requests_per_source_summary() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve("요구사항을 각각 정리해줘", top_k=2)

    assert len(vector_store.calls) == 2
    assert [call["where"]["파일명"] for call in vector_store.calls] == [
        "doc-1.hwp",
        "doc-2.hwp",
    ]


def test_retriever_merges_multi_agency_filter_results_round_robin() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [
                _retrieved_chunk("nps-1", 0.99, agency="국민연금공단"),
                _retrieved_chunk("nps-2", 0.98, agency="국민연금공단"),
            ],
            "기초과학연구원": [
                _retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원"),
                _retrieved_chunk("ibs-2", 0.96, agency="기초과학연구원"),
            ],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve(
        "국민연금공단과 기초과학연구원 사업을 비교해줘",
        top_k=4,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "국민연금공단",
        "기초과학연구원",
    ]
    assert [call["top_k"] for call in vector_store.calls] == [8, 8]
    assert [result.chunk.chunk_id for result in results] == ["nps-1", "ibs-1", "nps-2", "ibs-2"]
    assert [result.rank for result in results] == [1, 2, 3, 4]


def test_retriever_keeps_explicit_multi_source_filter_single_query_without_comparison_intent() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve(
        "요구사항 알려줘",
        top_k=4,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert len(vector_store.calls) == 1
    assert vector_store.last_kwargs["where"] == {
        "발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}
    }


def test_retriever_fans_out_multi_agency_filter_for_per_source_each_query() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [_retrieved_chunk("nps-1", 0.99, agency="국민연금공단")],
            "기초과학연구원": [_retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원")],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve(
        "각 기관 요구사항을 각각 정리해줘",
        top_k=2,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "국민연금공단",
        "기초과학연구원",
    ]
    assert [call["top_k"] for call in vector_store.calls] == [4, 4]
    assert [result.chunk.chunk_id for result in results] == ["nps-1", "ibs-1"]


def test_retriever_auto_fans_out_when_query_mentions_two_agencies() -> None:
    class LocalComparisonMetadataStore:
        agency_list = ["경기도 안양시", "경기도사회서비스원"]

        def find_relevant_docs(self, query: str, top_n: int = 3):
            return []

    vector_store = ScopedFakeVectorStore(
        {
            "경기도 안양시": [_retrieved_chunk("anyang-1", 0.99, agency="경기도 안양시")],
            "경기도사회서비스원": [
                _retrieved_chunk("ggsw-1", 0.97, agency="경기도사회서비스원")
            ],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=LocalComparisonMetadataStore(),
    )

    results = retriever.retrieve(
        "안양시 사업이랑 경기도사회서비스원 사업은 규정과 관련된 접근 통제를 명시하고 있습니다. 각각 알려줘",
        top_k=2,
    )

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "경기도 안양시",
        "경기도사회서비스원",
    ]
    assert [call["top_k"] for call in vector_store.calls] == [4, 4]
    assert [result.chunk.chunk_id for result in results] == ["anyang-1", "ggsw-1"]


def test_retriever_fans_out_file_scopes_for_multi_project_comparison() -> None:
    vector_store = ScopedFieldFakeVectorStore(
        "파일명",
        {
            "mobile-1.hwp": [
                _retrieved_chunk(
                    "mobile-1",
                    0.99,
                    agency="한국철도공사 (용역)",
                    doc_id="mobile-1",
                    file_name="mobile-1.hwp",
                )
            ],
            "mobile-2.hwp": [
                _retrieved_chunk(
                    "mobile-2",
                    0.98,
                    agency="한국철도공사 (용역)",
                    doc_id="mobile-2",
                    file_name="mobile-2.hwp",
                )
            ],
            "reserve-1.hwp": [
                _retrieved_chunk(
                    "reserve-1",
                    0.97,
                    agency="한국철도공사 (용역)",
                    doc_id="reserve-1",
                    file_name="reserve-1.hwp",
                )
            ],
            "reserve-2.hwp": [
                _retrieved_chunk(
                    "reserve-2",
                    0.96,
                    agency="한국철도공사 (용역)",
                    doc_id="reserve-2",
                    file_name="reserve-2.hwp",
                )
            ],
        },
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=ProjectAwareFakeMetadataStore(
            {
                "모바일오피스 시스템 고도화 용역": ["mobile-1.hwp", "mobile-2.hwp"],
                "예약발매시스템 개량 ISMP 용역": ["reserve-1.hwp", "reserve-2.hwp"],
            }
        ),
    )

    results = retriever.retrieve(
        '"모바일오피스 시스템 고도화 용역"과 "예약발매시스템 개량 ISMP 용역"을 비교해줘',
        top_k=4,
        metadata_filter={"발주 기관": "한국철도공사 (용역)"},
    )

    assert [call["where"]["파일명"] for call in vector_store.calls] == [
        "mobile-1.hwp",
        "mobile-2.hwp",
        "reserve-1.hwp",
        "reserve-2.hwp",
    ]
    assert [result.chunk.chunk_id for result in results] == [
        "mobile-1",
        "mobile-2",
        "reserve-1",
        "reserve-2",
    ]


def test_retriever_prefers_document_diversity_for_scoped_comparison() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [
                _retrieved_chunk(
                    "nps-doc1-1",
                    0.99,
                    agency="국민연금공단",
                    doc_id="nps-doc1",
                    file_name="nps-doc1.hwp",
                ),
                _retrieved_chunk(
                    "nps-doc1-2",
                    0.98,
                    agency="국민연금공단",
                    doc_id="nps-doc1",
                    file_name="nps-doc1.hwp",
                ),
                _retrieved_chunk(
                    "nps-doc2-1",
                    0.97,
                    agency="국민연금공단",
                    doc_id="nps-doc2",
                    file_name="nps-doc2.hwp",
                ),
            ],
            "기초과학연구원": [
                _retrieved_chunk(
                    "ibs-doc1-1",
                    0.96,
                    agency="기초과학연구원",
                    doc_id="ibs-doc1",
                    file_name="ibs-doc1.hwp",
                ),
                _retrieved_chunk(
                    "ibs-doc1-2",
                    0.95,
                    agency="기초과학연구원",
                    doc_id="ibs-doc1",
                    file_name="ibs-doc1.hwp",
                ),
                _retrieved_chunk(
                    "ibs-doc2-1",
                    0.94,
                    agency="기초과학연구원",
                    doc_id="ibs-doc2",
                    file_name="ibs-doc2.hwp",
                ),
            ],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve(
        "국민연금공단과 기초과학연구원의 사업을 비교해줘",
        top_k=4,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert [result.chunk.chunk_id for result in results] == [
        "nps-doc1-1",
        "ibs-doc1-1",
        "nps-doc2-1",
        "ibs-doc2-1",
    ]


def test_retriever_does_not_apply_where_document_filter_for_scoped_comparison() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [_retrieved_chunk("nps-1", 0.99, agency="국민연금공단", section="예산")],
            "기초과학연구원": [_retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원", section="예산")],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve(
        "국민연금공단과 기초과학연구원의 예산을 비교해줘",
        top_k=2,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert [call["where_document"] for call in vector_store.calls] == [None, None]


def test_extract_matched_agencies_supports_koica_korean_alias() -> None:
    matched = extract_matched_agencies(
        "코이카 전자조달의 우즈베키스탄 방송시스템 사업과 아시아물위원회 사무국 사업을 비교해줘",
        ["KOICA 전자조달", "사단법인아시아물위원회사무국"],
    )

    assert matched == ["KOICA 전자조달", "사단법인아시아물위원회사무국"]


def test_retriever_fans_out_query_named_multi_agency_comparison_without_explicit_filter() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [_retrieved_chunk("nps-1", 0.99, agency="국민연금공단")],
            "기초과학연구원": [_retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원")],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve("국민연금공단과 기초과학연구원 사업을 비교해줘", top_k=2)

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "국민연금공단",
        "기초과학연구원",
    ]
    assert [result.chunk.chunk_id for result in results] == ["nps-1", "ibs-1"]


def test_retriever_fans_out_multi_agency_listing_query_without_explicit_filter() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [_retrieved_chunk("nps-1", 0.99, agency="국민연금공단")],
            "기초과학연구원": [_retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원")],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve(
        "국민연금공단과 기초과학연구원의 사업 정보를 각각 나열해줘",
        top_k=2,
    )

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "국민연금공단",
        "기초과학연구원",
    ]
    assert [result.chunk.chunk_id for result in results] == ["nps-1", "ibs-1"]


def test_retriever_adds_project_doc_shortlist_for_quoted_multi_project_query() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=ProjectAwareFakeMetadataStore(
            {
                "호계체육관 예약 시스템 구축": ["anyang.hwp"],
                "평택시 버스정보시스템(BIS) 구축": ["pyeongtaek.hwp"],
                "경기도사회서비스원 통합사회정보시스템 운영 지원": ["ggsw.hwp"],
            }
        ),
    )

    retriever.retrieve(
        '"호계체육관 예약 시스템 구축", "평택시 버스정보시스템(BIS) 구축", "경기도사회서비스원 통합사회정보시스템 운영 지원" 사업 중에서 소요예산이 가장 큰 사업과 가장 작은 사업은 각각 무엇입니까?',
        top_k=5,
        metadata_filter={"공개연도": 2024},
    )

    assert [call["where"] for call in vector_store.calls] == [
        {"공개연도": 2024, "파일명": "anyang.hwp"},
        {"공개연도": 2024, "파일명": "pyeongtaek.hwp"},
        {"공개연도": 2024, "파일명": "ggsw.hwp"},
    ]


def test_retriever_applies_cross_encoder_after_multi_agency_fan_out_merge() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [
                _retrieved_chunk("nps-1", 0.99, agency="국민연금공단"),
                _retrieved_chunk("nps-2", 0.98, agency="국민연금공단"),
            ],
            "기초과학연구원": [
                _retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원"),
                _retrieved_chunk("ibs-2", 0.96, agency="기초과학연구원"),
            ],
        }
    )
    reranker = FakeReranker(
        {
            "[발주기관: 국민연금공단 | 사업명: nps-1-사업]\nnps-1": 0.20,
            "[발주기관: 기초과학연구원 | 사업명: ibs-1-사업]\nibs-1": 0.95,
            "[발주기관: 국민연금공단 | 사업명: nps-2-사업]\nnps-2": 0.90,
            "[발주기관: 기초과학연구원 | 사업명: ibs-2-사업]\nibs-2": 0.10,
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        reranker_model=reranker,
    )

    results = retriever.retrieve(
        "국민연금공단과 기초과학연구원 사업을 비교해줘",
        top_k=2,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "국민연금공단",
        "기초과학연구원",
    ]
    assert [call["top_k"] for call in vector_store.calls] == [8, 8]
    assert reranker.calls == [
        [
            [
                "국민연금공단과 기초과학연구원 사업을 비교해줘",
                "[발주기관: 국민연금공단 | 사업명: nps-1-사업]\nnps-1",
            ],
            [
                "국민연금공단과 기초과학연구원 사업을 비교해줘",
                "[발주기관: 기초과학연구원 | 사업명: ibs-1-사업]\nibs-1",
            ],
            [
                "국민연금공단과 기초과학연구원 사업을 비교해줘",
                "[발주기관: 국민연금공단 | 사업명: nps-2-사업]\nnps-2",
            ],
            [
                "국민연금공단과 기초과학연구원 사업을 비교해줘",
                "[발주기관: 기초과학연구원 | 사업명: ibs-2-사업]\nibs-2",
            ],
        ]
    ]
    assert [result.chunk.chunk_id for result in results] == ["ibs-1", "nps-2"]
    assert [result.rank for result in results] == [1, 2]
    # CE 점수가 score에 반영돼 boost가 CE 기준으로 재정렬한다.
    assert [result.score for result in results] == [0.95, 0.9]
    assert [result.rerank_score for result in results] == [0.95, 0.9]


def test_retriever_preserves_doc_diversity_after_rerank_for_comparison_queries() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [
                _retrieved_chunk("nps-1", 0.99, agency="국민연금공단"),
                _retrieved_chunk("nps-2", 0.98, agency="국민연금공단"),
            ],
            "기초과학연구원": [
                _retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원"),
                _retrieved_chunk("ibs-2", 0.96, agency="기초과학연구원"),
            ],
        }
    )
    reranker = FakeReranker(
        {
            "[발주기관: 국민연금공단 | 사업명: nps-1-사업]\nnps-1": 0.99,
            "[발주기관: 국민연금공단 | 사업명: nps-2-사업]\nnps-2": 0.98,
            "[발주기관: 기초과학연구원 | 사업명: ibs-1-사업]\nibs-1": 0.50,
            "[발주기관: 기초과학연구원 | 사업명: ibs-2-사업]\nibs-2": 0.40,
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        reranker_model=reranker,
    )

    results = retriever.retrieve(
        "국민연금공단과 기초과학연구원 사업을 비교해줘",
        top_k=2,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert [result.chunk.chunk_id for result in results] == ["nps-1", "ibs-1"]
    assert [result.rank for result in results] == [1, 2]


def test_retriever_reranks_table_and_section_matches_over_higher_raw_score_text() -> None:
    vector_store = FakeVectorStore(
        query_results=[
            _retrieved_chunk("overview-text", 0.91, agency="국민연금공단", section="사업개요"),
            _retrieved_chunk(
                "budget-table",
                0.8,
                agency="국민연금공단",
                section="예산",
                content_type="table",
            ),
        ]
    )
    mock_llm = _make_mock_llm(
        '{"rewritten_query": "국민연금공단 예산 표를 알려줘", "section_hint": "예산"}'
    )
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    results = retriever.retrieve(
        "국민연금공단 예산 표를 알려줘",
        top_k=2,
        chat_history=[{"role": "user", "content": "예산 관련 질문"}],
    )

    assert results[0].chunk.chunk_id == "budget-table"
    assert results[0].rank == 1


def test_retriever_preserves_original_order_when_no_rerank_hints_present() -> None:
    vector_store = FakeVectorStore(
        query_results=[
            _retrieved_chunk("first-text", 0.75, agency="국민연금공단", section="사업개요"),
            _retrieved_chunk("second-text", 0.95, agency="국민연금공단", section="일반"),
        ]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve("국민연금공단 사업 알려줘", top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["first-text", "second-text"]
    assert [result.rank for result in results] == [1, 2]


def test_retriever_rewrites_follow_up_query_and_inherits_recent_agency_filter() -> None:
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve(
        "그 사업 예산은?",
        top_k=2,
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
    )

    assert embedder.queries == ["국민연금공단 차세대 ERP 사업 예산은?", "그 사업 예산은?"]
    assert vector_store.last_kwargs["where"]["발주 기관"] == "국민연금공단"
    assert vector_store.last_kwargs["where"]["파일명"] == {"$in": ["doc-1.hwp", "doc-2.hwp"]}
    assert retriever._last_debug["rewritten_query"] == "국민연금공단 차세대 ERP 사업 예산은?"
    assert retriever._last_debug["retrieved_chunks_before_rerank"]
    assert retriever._last_debug["retrieved_chunks_after_rerank"]


def test_retriever_uses_llm_rewrite_for_implicit_followup() -> None:
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    mock_llm = _make_mock_llm("국민연금공단 차세대 ERP 사업의 평가기준")
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    retriever.retrieve(
        "평가기준은?",
        top_k=2,
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
            {"role": "assistant", "content": "해당 사업은 국민연금공단의 차세대 ERP 구축 사업입니다."},
        ],
    )

    prompt = mock_llm.rewrite.call_args.args[0]
    assert embedder.queries == ["국민연금공단 차세대 ERP 사업의 평가기준", "평가기준은?"]
    mock_llm.rewrite.assert_called_once()
    assert "발주기관: 국민연금공단" in prompt
    assert "사업명" in prompt
    assert retriever._last_debug["rewrite_reason"] == "llm"
    assert retriever._last_debug["rewritten_query"] == "국민연금공단 차세대 ERP 사업의 평가기준"
    assert retriever._last_debug["rewrite_slot_memory"]["발주기관"] == "국민연금공단"


def test_retriever_filters_numeric_slots_before_passing_memory_to_rewrite() -> None:
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    mock_llm = _make_mock_llm(
        '{"rewritten_query": "국민연금공단 차세대 ERP 사업의 예산", "section_hint": "예산"}'
    )
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    retriever.retrieve(
        "예산은?",
        top_k=2,
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
            {"role": "assistant", "content": "예산은 5억원입니다."},
        ],
    )

    prompt = mock_llm.rewrite.call_args.args[0]
    assert "발주기관: 국민연금공단" in prompt
    assert "사업명: 국민연금공단 차세대 ERP 사업" in prompt
    assert "관심속성: 예산" in prompt
    assert "예산: 5억원" not in prompt
    assert retriever._last_debug["rewrite_slot_memory"] == {
        "발주기관": "국민연금공단",
        "사업명": "국민연금공단 차세대 ERP 사업",
        "관심속성": "예산",
    }


def test_retriever_keeps_full_generation_memory_when_rewrite_slots_are_filtered() -> None:
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    mock_llm = _make_mock_llm(
        '{"rewritten_query": "국민연금공단 차세대 ERP 사업의 예산", "section_hint": "예산"}'
    )
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    retriever.retrieve(
        "예산은?",
        top_k=2,
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
            {"role": "assistant", "content": "예산은 5억원입니다."},
        ],
    )

    assert retriever._last_debug["memory_state"]["slot_memory"]["예산"] == "5억원"


def test_retriever_apply_experimental_rerank_returns_original_results_when_disabled() -> None:
    results = [_retrieved_chunk("chunk-1", 0.9, agency="국민연금공단")]
    retriever = RAGRetriever(
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        reranker_model=None,
    )

    reranked = retriever._apply_experimental_rerank("질문", results, 1)

    assert reranked is results


def test_retriever_keeps_minimal_runtime_state_when_debug_trace_disabled() -> None:
    from bidmate_rag.providers.llm.base import RewriteResponse

    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text="국민연금공단 차세대 ERP 사업의 평가기준",
        prompt_tokens=12,
        completion_tokens=7,
        total_tokens=19,
    )
    mock_llm.model_name = "gpt-5-mini"
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
        debug_trace_enabled=False,
    )

    retriever.retrieve(
        "평가기준은?",
        top_k=2,
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
            {
                "role": "assistant",
                "content": "해당 사업은 국민연금공단의 차세대 ERP 구축 사업입니다.",
            },
        ],
    )

    assert retriever._last_debug["rewritten_query"] == "국민연금공단 차세대 ERP 사업의 평가기준"
    assert retriever._last_debug["rewrite_prompt_tokens"] == 12
    assert retriever._last_debug["rewrite_completion_tokens"] == 7
    assert retriever._last_debug["rewrite_total_tokens"] == 19
    assert retriever._last_debug["rewrite_cost_usd"] > 0.0
    assert retriever._last_debug["memory_state"]["slot_memory"]["발주기관"] == "국민연금공단"
    assert "retrieved_chunks_before_rerank" not in retriever._last_debug
    assert "where" not in retriever._last_debug


def test_retriever_can_disable_multiturn_rewrite_and_history_filter() -> None:
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=EmptyShortlistMetadataStore(),
        enable_multiturn=False,
    )

    retriever.retrieve(
        "그 사업 예산은?",
        top_k=2,
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
    )

    assert embedder.queries == ["그 사업 예산은?"]
    assert vector_store.last_kwargs["where"] is None


def test_retriever_history_aware_query_supports_user_assistant_shape() -> None:
    retriever = RAGRetriever(
        vector_store=FakeVectorStore(),
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    query = retriever._build_history_aware_query(
        "follow-up question",
        [{"user": "first question", "assistant": "first answer"}],
    )

    assert query == "first question first answer follow-up question"


def test_extract_project_clues_ignores_generic_quoted_phrases() -> None:
    clues = extract_project_clues(
        '"오류 메시지 처리 시간"과 "참여인력 개별 보안서약서 징구" 절차를 설명해줘'
    )

    assert clues == []


def test_retriever_retries_without_shortlist_when_augmented_where_returns_no_results() -> None:
    vector_store = SequenceFakeVectorStore(
        responses=[
            [],
            [],
            [_retrieved_chunk("fallback-hit", 0.88, agency="한국철도공사 (용역)")],
        ]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=ProjectAwareFakeMetadataStore(
            {
                "모바일오피스 시스템 고도화 용역": ["wrong-doc-1.hwp"],
                "예약발매시스템 개량 ISMP 용역": ["wrong-doc-2.hwp"],
            }
        ),
    )

    results = retriever.retrieve(
        '"모바일오피스 시스템 고도화 용역"과 "예약발매시스템 개량 ISMP 용역" 차이를 알려줘',
        top_k=5,
        metadata_filter={"발주 기관": "한국철도공사 (용역)"},
    )

    assert len(vector_store.calls) == 3
    assert vector_store.calls[0]["where"] == {
        "발주 기관": "한국철도공사 (용역)",
        "파일명": "wrong-doc-1.hwp",
    }
    assert vector_store.calls[1]["where"] == {
        "발주 기관": "한국철도공사 (용역)",
        "파일명": "wrong-doc-2.hwp",
    }
    assert vector_store.calls[2]["where"] == {"발주 기관": "한국철도공사 (용역)"}
    assert [result.chunk.chunk_id for result in results] == ["fallback-hit"]


def test_retriever_keeps_where_document_none_without_strong_hint() -> None:
    """Soft boost 전환 후 where_document은 항상 None이어야 한다."""
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve(
        "이 사업의 예산 규모를 알려줘",
        top_k=5,
        metadata_filter={"발주 기관": "한국원자력연구원"},
    )

    assert vector_store.last_kwargs["where_document"] is None


def test_retriever_uses_dynamic_where_document_anchor_from_query_phrase() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve(
        '경상북도 봉화군의 "재난통합관리시스템 고도화 사업"의 사업기간은 며칠로 명시되어 있습니까?',
        top_k=3,
    )

    assert vector_store.last_kwargs["where_document"] == {"$contains": "사업기간"}


def test_retriever_forwards_rewrite_section_hint_to_rerank_boost() -> None:
    """rewrite trace의 section_hint가 rerank_with_boost까지 전달돼야 한다."""
    vector_store = FakeVectorStore(
        query_results=[
            _retrieved_chunk("overview", 0.91, agency="국민연금공단", section="사업개요"),
            _retrieved_chunk("budget", 0.80, agency="국민연금공단", section="예산"),
        ]
    )
    mock_llm = _make_mock_llm(
        '{"rewritten_query": "국민연금공단 차세대 ERP 예산", "section_hint": "예산"}'
    )
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    results = retriever.retrieve(
        "예산은?",
        top_k=2,
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
    )

    # section_hint가 "예산"으로 들어오면 budget chunk가 boost로 1위가 됨
    assert vector_store.last_kwargs["where_document"] is None
    assert results[0].chunk.chunk_id == "budget"


def test_retriever_does_not_use_section_hint_as_where_document_anchor() -> None:
    vector_store = FakeVectorStore(
        query_results=[
            _retrieved_chunk("overview", 0.91, agency="국민연금공단", section="사업개요"),
            _retrieved_chunk("budget", 0.80, agency="국민연금공단", section="예산"),
        ]
    )
    mock_llm = _make_mock_llm(
        '{"rewritten_query": "국민연금공단 차세대 ERP 예산", "section_hint": "예산"}'
    )
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    retriever.retrieve(
        "예산은?",
        top_k=2,
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
    )

    assert vector_store.last_kwargs["where_document"] is None


def test_retriever_queries_original_text_as_secondary_variant_when_rewrite_applied() -> None:
    vector_store = SequenceFakeVectorStore(
        responses=[
            [_retrieved_chunk("rewritten-hit", 0.91, agency="국민연금공단")],
            [_retrieved_chunk("original-hit", 0.89, agency="국민연금공단")],
        ]
    )
    embedder = FakeEmbedder()
    mock_llm = _make_mock_llm(
        '{"rewritten_query": "국민연금공단 보안 규정", "section_hint": "보안 요구사항"}'
    )
    memory = ConversationMemory(
        max_recent_turns=4,
        max_summary_chars=120,
        agency_list=["국민연금공단"],
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=FakeMetadataStore(),
        rewrite_llm=mock_llm,
        memory=memory,
    )

    results = retriever.retrieve(
        "USB 반입 반출해도 되나요?",
        chat_history=[{"role": "user", "content": "국민연금공단 보안 요구사항 알려줘"}],
        top_k=2,
    )

    assert embedder.queries == ["국민연금공단 보안 규정", "USB 반입 반출해도 되나요?"]
    assert [result.chunk.chunk_id for result in results] == ["rewritten-hit", "original-hit"]


def test_anchor_auxiliary_runs_for_direct_phrase_fact_query() -> None:
    """직접 구문 앵커(사업기간)가 있으면 보조 검색 경로가 1회 추가 실행된다."""
    vector_store = SequenceFakeVectorStore(
        responses=[
            [_retrieved_chunk("main", 0.9, agency="국민연금공단")],
            [_retrieved_chunk("aux", 0.85, agency="국민연금공단")],
        ]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        hybrid_config={"enabled": False, "anchor_auxiliary": True},
    )

    retriever.retrieve("국민연금공단 사업기간 알려줘", top_k=3)

    assert len(vector_store.calls) == 2
    assert vector_store.calls[0]["where_document"] is None
    assert vector_store.calls[1]["where_document"] == {"$contains": "사업기간"}


def test_anchor_auxiliary_runs_for_numeric_anchor_only() -> None:
    """직접 구문이 없어도 숫자 앵커(12억원)만 있으면 보조 경로가 실행된다."""
    vector_store = SequenceFakeVectorStore(
        responses=[
            [_retrieved_chunk("main", 0.9, agency="국민연금공단")],
            [_retrieved_chunk("aux", 0.85, agency="국민연금공단")],
        ]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        hybrid_config={"enabled": False, "anchor_auxiliary": True},
    )

    retriever.retrieve("12억원 규모 사업 찾아줘", top_k=3)

    assert len(vector_store.calls) == 2
    assert vector_store.calls[1]["where_document"] == {"$contains": "12억원"}


def test_anchor_auxiliary_skipped_when_no_anchor() -> None:
    """앵커가 없는 일반 질의에서는 보조 경로가 돌지 않는다."""
    vector_store = SequenceFakeVectorStore(
        responses=[[_retrieved_chunk("only", 0.9, agency="국민연금공단")]]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        hybrid_config={"enabled": False, "anchor_auxiliary": True},
    )

    retriever.retrieve("국민연금공단 보안 요구사항 알려줘", top_k=3)

    assert len(vector_store.calls) == 1


def test_anchor_auxiliary_disabled_by_flag() -> None:
    """anchor_auxiliary: False면 fact형 질의에도 보조 경로가 돌지 않는다."""
    vector_store = SequenceFakeVectorStore(
        responses=[[_retrieved_chunk("only", 0.9, agency="국민연금공단")]]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        hybrid_config={"enabled": False, "anchor_auxiliary": False},
    )

    retriever.retrieve("국민연금공단 사업기간 알려줘", top_k=3)

    assert len(vector_store.calls) == 1


def test_anchor_auxiliary_skipped_when_force_scoped() -> None:
    """multi-source fan-out(force_scoped=True) 중이면 앵커 보조 경로를 스킵한다."""
    vector_store = FakeVectorStore(
        [_retrieved_chunk("chunk-1", 0.9, agency="국민연금공단")]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        hybrid_config={"enabled": False, "anchor_auxiliary": True},
    )

    retriever.retrieve(
        "국민연금공단과 기초과학연구원의 사업기간 알려줘",
        top_k=3,
    )

    anchor_aux_calls = [
        call
        for call in vector_store.calls
        if call.get("where_document") == {"$contains": "사업기간"}
    ]
    assert anchor_aux_calls == []


def test_anchor_auxiliary_merges_main_and_aux_results() -> None:
    """메인과 보조 결과가 score 기준으로 중복 제거되며 합쳐진다."""
    vector_store = SequenceFakeVectorStore(
        responses=[
            [_retrieved_chunk("main-only", 0.9, agency="국민연금공단")],
            [_retrieved_chunk("aux-only", 0.95, agency="국민연금공단")],
        ]
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        hybrid_config={"enabled": False, "anchor_auxiliary": True},
    )

    results = retriever.retrieve("국민연금공단 사업기간 알려줘", top_k=5)

    chunk_ids = {result.chunk.chunk_id for result in results}
    assert {"main-only", "aux-only"}.issubset(chunk_ids)
