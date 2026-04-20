import pandas as pd

from bidmate_rag.storage.metadata_store import MetadataStore


def test_find_relevant_docs_prefers_title_and_summary_matches() -> None:
    frame = pd.DataFrame(
        [
            {
                "파일명": "doc-a.hwp",
                "사업명": "평택시 버스정보시스템 구축사업",
                "사업 요약": "버스정보시스템 구축",
                "텍스트": "일반 본문",
                "본문_마크다운": "",
                "발주 기관": "경기도 평택시",
            },
            {
                "파일명": "doc-b.hwp",
                "사업명": "기타 사업",
                "사업 요약": "일반 시스템",
                "텍스트": "버스정보시스템이라는 단어만 본문에 있음",
                "본문_마크다운": "",
                "발주 기관": "기타 기관",
            },
        ]
    )
    store = MetadataStore(frame)

    docs = store.find_relevant_docs("평택시 버스정보시스템 구축", top_n=2)

    assert docs == ["doc-a.hwp", "doc-b.hwp"]


def test_find_relevant_docs_can_match_body_clause_keywords() -> None:
    frame = pd.DataFrame(
        [
            {
                "파일명": "doc-a.hwp",
                "사업명": "일반 사업",
                "사업 요약": "일반 요약",
                "텍스트": "본 문서에는 법제도 준수 여부 점검표와 과업심의 종합결과서가 포함되어 있습니다.",
                "본문_마크다운": "",
                "발주 기관": "한국보건산업진흥원",
            },
            {
                "파일명": "doc-b.hwp",
                "사업명": "다른 사업",
                "사업 요약": "다른 요약",
                "텍스트": "일반적인 제안 요청 내용입니다.",
                "본문_마크다운": "",
                "발주 기관": "기타 기관",
            },
        ]
    )
    store = MetadataStore(frame)

    docs = store.find_relevant_docs("법제도 준수 여부 점검표를 포함한 사업은 무엇입니까?", top_n=2)

    assert docs[0] == "doc-a.hwp"


def test_find_relevant_docs_strips_common_particles_from_query_tokens() -> None:
    frame = pd.DataFrame(
        [
            {
                "파일명": "doc-a.hwp",
                "사업명": "원격지 접속 보안 관리",
                "사업 요약": "로그기록 보관",
                "텍스트": "원격지 접속 시 지정 단말기 로그기록 1년 이상 보관",
                "본문_마크다운": "",
                "발주 기관": "경기도 안양시",
            },
            {
                "파일명": "doc-b.hwp",
                "사업명": "일반 사업",
                "사업 요약": "기타",
                "텍스트": "무관한 본문",
                "본문_마크다운": "",
                "발주 기관": "기타 기관",
            },
        ]
    )
    store = MetadataStore(frame)

    docs = store.find_relevant_docs("로그기록은 얼마 동안 보관해야 합니까?", top_n=2)

    assert docs[0] == "doc-a.hwp"
