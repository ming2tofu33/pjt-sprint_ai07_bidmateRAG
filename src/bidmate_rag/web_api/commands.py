"""Slash command registry for BidMate web UI.

각 커맨드는:
1. 쿼리 증강 (query_augmentation) — ChromaDB 임베딩 검색이 더 관련 있는 청크를 끌어오도록 키워드 추가
2. content_type 선호 — table 중심 커맨드는 표 청크를 우선
3. 특화된 system_prompt — 답변 형식(bullet/표/체크리스트) 강제
4. requires_doc / requires_multi_doc — Send 버튼 validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlashCommand:
    id: str
    label: str
    description: str
    icon: str
    query_augmentation: str = ""
    system_prompt: str = ""
    top_k: int = 10
    content_type_preference: str | None = None  # "text" | "table" | None
    requires_doc: bool = False
    requires_multi_doc: bool = False
    static_response: bool = False
    static_payload: dict[str, Any] | None = None


_SUMMARY_PROMPT = """당신은 RFP 분석 전문가입니다. 제공된 문서 컨텍스트에서 사업 개요를 추출해 주세요.

반드시 다음 형식으로 답변:
- 5~7개의 bullet point
- 각 bullet 끝에 [n] 인용 번호
- 목적 → 배경 → 범위 순서

컨텍스트에 없는 내용은 절대 추측하지 마세요."""

_REQUIREMENTS_PROMPT = """당신은 RFP 요구사항 추출 전문가입니다.

반드시 다음 형식으로 답변:
## 기능 요구사항
| 항목 | 내용 | 근거 |
|---|---|---|
| ... | ... | [n] |

## 비기능 요구사항
| 항목 | 내용 | 근거 |
|---|---|---|

컨텍스트에 기능/비기능이 명시적으로 구분되지 않았다면, 내용을 보고 판단해 분류하세요."""

_SCHEDULE_PROMPT = """당신은 RFP 일정 추출 전문가입니다.

반드시 다음 형식으로 답변:
| 시점 | 마일스톤 | 산출물 | 근거 |
|---|---|---|---|
| ... | ... | ... | [n] |

계약기간, 착수일, 중간보고, 납품 같은 주요 이벤트를 빠짐없이 추출하세요."""

_BUDGET_PROMPT = """당신은 RFP 예산 분석 전문가입니다.

반드시 다음 형식으로 답변:
**총 사업비**: N억 N천만원 [n]

**세부 내역**:
| 항목 | 금액 | 근거 |
|---|---|---|

**지급 방식**: ... [n]"""

_COMPARE_PROMPT = """당신은 RFP 비교 분석 전문가입니다. 사용자가 멘션한 여러 문서를 항목별로 나란히 비교하세요.

반드시 다음 형식으로 답변:
| 항목 | 문서 A | 문서 B | (문서 C) |
|---|---|---|---|
| 발주기관 | ... | ... | ... |
| 사업 금액 | ... | ... | ... |
| 일정 | ... | ... | ... |
| 주요 요구사항 | ... | ... | ... |
| 차이점 | ... | ... | ... |

각 셀 끝에 [n] 인용 번호. 질문 의도에 맞는 행만 남기고 불필요한 행은 제외."""

_QUALIFICATION_PROMPT = """당신은 RFP 참가 자격 추출 전문가입니다.

반드시 다음 형식으로 답변:
## 참가 자격 체크리스트
- [ ] 법인/사업자 유형: ... [n]
- [ ] 실적 요건: ... [n]
- [ ] 인증/자격: ... [n]
- [ ] 기타: ... [n]

컨텍스트에 없으면 "명시되지 않음"으로 표시."""

_EVALUATION_PROMPT = """당신은 RFP 평가 기준 추출 전문가입니다.

반드시 다음 형식으로 답변:
| 평가 영역 | 세부 항목 | 배점 | 근거 |
|---|---|---|---|
| 기술 | ... | XX점 | [n] |
| 가격 | ... | XX점 | [n] |

총점과 평가 방식(절대/상대 평가)도 명시."""

_RISK_PROMPT = """당신은 RFP 리스크 분석 전문가입니다. 컨설턴트가 수주 전 확인해야 할 독소조항·위약·분쟁 조항을 찾으세요.

반드시 다음 형식으로 답변:
## ⚠️ 주요 리스크
- **[위험도: 상/중/하] 항목명** — 설명 [n]

## 계약 종료/해지 조건
- ... [n]

## 권장 대응
- ..."""

_BASIC_INFO_PROMPT = """당신은 RFP 기본 정보 요약 전문가입니다.

반드시 다음 형식으로 답변:
- **발주기관**: ... [n]
- **사업명**: ... [n]
- **사업 금액**: ... [n]
- **입찰 마감**: ... [n]
- **담당자**: ... [n]
- **문서번호**: ... [n]"""

_SUBMISSION_PROMPT = """당신은 RFP 제출 서류 체크리스트 전문가입니다.

반드시 다음 형식으로 답변:
## 필수 제출 서류
- [ ] 서류명 — 양식/조건 [n]

## 선택 제출 서류
- [ ] 서류명 — 양식/조건 [n]

기한이 명시된 경우 함께 표시."""


_HELP_PAYLOAD = {
    "answer": """## 슬래시 커맨드 안내

| 커맨드 | 설명 |
|---|---|
| `/요약` | 사업 개요 bullet 요약 |
| `/요구사항` | 기능/비기능 요구사항 표 |
| `/일정` | 마일스톤 타임라인 |
| `/예산` | 금액 breakdown |
| `/비교` | 여러 문서 비교표 (2개+ 멘션 필요) |
| `/자격요건` | 참가 자격 체크리스트 |
| `/평가기준` | 평가 배점표 |
| `/리스크` | 독소조항/위약 경고 |
| `/기본정보` | 핵심 정보 요약 |
| `/제출서류` | 서류 체크리스트 |
| `/도움말` | 이 안내 |
| `/초기화` | 현재 컨텍스트 초기화 |

`@` 입력 후 문서명을 선택하면 해당 문서만 검색 대상이 됩니다.""",
    "citations": [],
}

_RESET_PAYLOAD = {
    "answer": "컨텍스트가 초기화되었습니다. 멘션된 문서와 활성 커맨드를 모두 해제했습니다.",
    "citations": [],
    "client_action": "clear_context",
}


COMMAND_REGISTRY: dict[str, SlashCommand] = {
    "요약": SlashCommand(
        id="요약",
        label="/요약",
        description="사업 개요를 bullet로 요약",
        icon="📋",
        query_augmentation="사업 개요 목적 배경 범위",
        system_prompt=_SUMMARY_PROMPT,
        top_k=8,
        content_type_preference="text",
    ),
    "요구사항": SlashCommand(
        id="요구사항",
        label="/요구사항",
        description="기능/비기능 요구사항 표 추출",
        icon="📐",
        query_augmentation="요구사항 기능 비기능 상세 과업 업무 내용",
        system_prompt=_REQUIREMENTS_PROMPT,
        top_k=12,
        content_type_preference="text",
    ),
    "일정": SlashCommand(
        id="일정",
        label="/일정",
        description="마일스톤 타임라인",
        icon="📅",
        query_augmentation="일정 기간 납기 마일스톤 스케줄 착수 완료",
        system_prompt=_SCHEDULE_PROMPT,
        top_k=8,
        content_type_preference="table",
    ),
    "예산": SlashCommand(
        id="예산",
        label="/예산",
        description="사업비 상세 breakdown",
        icon="💰",
        query_augmentation="사업비 예산 금액 비용 단가 지급",
        system_prompt=_BUDGET_PROMPT,
        top_k=8,
        content_type_preference="table",
    ),
    "비교": SlashCommand(
        id="비교",
        label="/비교",
        description="2개 이상 문서 항목별 비교",
        icon="📊",
        query_augmentation="비교 차이점 대조 공통점",
        system_prompt=_COMPARE_PROMPT,
        top_k=15,
        requires_doc=True,
        requires_multi_doc=True,
    ),
    "자격요건": SlashCommand(
        id="자격요건",
        label="/자격요건",
        description="참가 자격·실적·인증 추출",
        icon="✅",
        query_augmentation="참가 자격 요건 실적 인증 입찰 자격",
        system_prompt=_QUALIFICATION_PROMPT,
        top_k=8,
        content_type_preference="text",
    ),
    "평가기준": SlashCommand(
        id="평가기준",
        label="/평가기준",
        description="평가 배점표",
        icon="⚖️",
        query_augmentation="평가 기준 배점 심사 항목 기술 가격",
        system_prompt=_EVALUATION_PROMPT,
        top_k=8,
        content_type_preference="table",
    ),
    "리스크": SlashCommand(
        id="리스크",
        label="/리스크",
        description="독소조항·위약·분쟁 조항",
        icon="⚠️",
        query_augmentation="독소조항 위약 분쟁 해지 계약 조건 책임",
        system_prompt=_RISK_PROMPT,
        top_k=10,
        content_type_preference="text",
    ),
    "기본정보": SlashCommand(
        id="기본정보",
        label="/기본정보",
        description="핵심 항목 요약",
        icon="📌",
        query_augmentation="발주 기관 담당자 사업 금액 마감 공고",
        system_prompt=_BASIC_INFO_PROMPT,
        top_k=5,
    ),
    "제출서류": SlashCommand(
        id="제출서류",
        label="/제출서류",
        description="필요 서류 체크리스트",
        icon="📋",
        query_augmentation="제출 서류 양식 첨부 필수",
        system_prompt=_SUBMISSION_PROMPT,
        top_k=8,
        content_type_preference="text",
    ),
    "도움말": SlashCommand(
        id="도움말",
        label="/도움말",
        description="슬래시 커맨드 안내",
        icon="❓",
        static_response=True,
        static_payload=_HELP_PAYLOAD,
    ),
    "초기화": SlashCommand(
        id="초기화",
        label="/초기화",
        description="현재 컨텍스트 초기화",
        icon="🔄",
        static_response=True,
        static_payload=_RESET_PAYLOAD,
    ),
}
