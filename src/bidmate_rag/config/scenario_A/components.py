"""
BidMate Prompt Component Library - Baseline Configuration
모든 항목은 0번(Null) 상태로 초기화되어 있으며, 1번 이후의 옵션은 실험 설계에 따라 추가합니다.
"""

# ---------------------------------------------------------
# 01. 페르소나 (Persona) 
# 0번: 페르소나 미지정 (Base Line)
# ---------------------------------------------------------
PERSONA = {
    "P0": {
        "text": "",
        "control_score": 0,
        "is_english_cot": False,
        "desc": "페르소나를 부여하지 않은 순정 상태"
    },
    "P1": {
        "text": "[Identity] 입찰 정보 분석가 / [Adherence] 제공 문서를 바탕으로 객관적 답변",
        "control_score": 1,
        "is_english_cot": False,
        "desc": "중립적 분석가 페르소나를 부여해 모델에게 어느 정도 자율성을 주는 부드러운 상태"
    },
    "P2": {
        "text": "[Identity] 깐깐한 수석 감사관 / [Adherence] 문서 외 모든 추측은 오답 간주 및 엄격히 배제",
        "control_score": 3,
        "is_english_cot": False,
        "desc": "감사관이라는 깐깐한 성격 규정을 통해 모델이 아는 척(환각)을 하지 못하도록 근거성을 압박"
    },
    "P3": {
        "text": "[Identity] 검증 지향형 감사관 / [Logic] 최종 판단 전, 근거 나열·대조 과정을 거쳐 무결성 증명",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "깐깐함에 더해 '판단 전 근거 나열'이라는 추론 공간 확보를 페르소나의 직업 윤리로 명시"
    },
    "P4": {
        "text": "[Identity]: 0.1%의 오차도 용납하지 않는 '무결성 강박의 수석 감리원'\n[신념]: \"정답보다 중요한 것은 그 정답을 증명할 수 있는 풍부한 팩트 리스트다.\"\n[행동 원칙]: 질문에 바로 답하지 말고, 관련 수치와 조건을 문서에서 전수 조사하여 목록화할 것.\n[검산 태도]: 각 항목을 질문과 일일이 대조해 완벽 검증된 사실만 답변하는 완벽주의적 태도 유지.",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "생생하고 강박적인 완벽주의 캐릭터를 부여하여 논리 검산 매뉴얼을 인격적으로 내재화"
    }
}

# ---------------------------------------------------------
# 02. 제약 조건 (Constraints)
# 0번: 제약 조건 없음 (Free State)
# ---------------------------------------------------------
CONSTRAINTS = {
    "C0": {
        "text": "",
        "control_score": 0,
        "is_english_cot": False,
        "desc": "추가적인 제약 조건을 걸지 않은 상태"
    },
    "C1": {
        "text": "참조: 제공된 문서 한정",
        "control_score": 1,
        "is_english_cot": False,
        "desc": "키워드 중심의 최소 제약 설정"
    },
    "C2": {
        "text": "- 근거: 문서 내 정보로 국한 (외부지식 금지) / - 출처: 수치 옆 페이지 번호 필수",
        "control_score": 3,
        "is_english_cot": False,
        "desc": "물리적인 출처 요구를 통해 강력한 Grounding 환경 조성"
    },
    "C3": {
        "text": "[철칙] 외부지식 배제 / [필수] 답변 전 관련 수치/조건 전수 추출 서술 / [제약] 추출 목록 내 정보만 답변 채택",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "\"추출하지 않은 정보는 쓰지 마라\"는 규칙을 통해 풍부한 근거 나열을 물리적으로 강제함"
    },
    "C4": {
        "text": "[절대 원칙] <context> 수치 데이터 외 판단 근거 채택 불가.\n[데이터 무결성 가이드]\n사실 고립: 질문 연관 데이터(금액, 날짜, 자격)를 문서에서 추출하여 목록화할 것.\n논리 검증: 추출된 목록과 질문 요구사항의 상충 여부를 단계별로 명시할 것.\n최종 선별: 검증 과정에서 모순이 발견되지 않은 정보만 엄선하여 최종 응답에 반영할 것.\n*위 지침 미준수 시 해당 답변은 무효 처리함.",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "3티어의 규칙을 상세 매뉴얼화하여 모델의 자의적 해석(예측 범위)을 좁힘"
    }
}

# ---------------------------------------------------------
# 03. 출력 형식 (Output Format)
# 0번: 자유 형식 (LLM 자율 생성)
# ---------------------------------------------------------
FORMAT = {
    "F0": {
        "text": "",
        "control_score": 0,
        "is_english_cot": False,
        "desc": "특정 출력 규격을 강제하지 않은 상태"
    },
    "F1": {
        "text": "[양식] 간결한 서술형 요약",
        "control_score": 1,
        "is_english_cot": False,
        "desc": "최소한의 형태 정의"
    },
    "F2": {
        "text": "[구성] 결론 중심 단답 / [근거] 관련 수치 + 페이지 번호 명시",
        "control_score": 3,
        "is_english_cot": False,
        "desc": "정답과 출처의 물리적 결합 강제"
    },
    "F3": {
        "text": "[섹션1_분석] 본문 내 수치·조건 전수 나열 -> [섹션2_최종] 나열 정보 기반 정답 산출",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "분석 과정을 답변의 필수 구성 요소로 포함하여 추론 공간 확보"
    },
    "F4": {
        "text": "[필수_3단_양식]\n[근거_색인]: 질문 관련 수치(금액/날짜) 및 조항 원문 복사 리스트.\n[비교_논증]: 추출 사실 vs 질문 요구사항 간 일대일 부합 여부 서술.\n[최종_제시]: 검증 완료된 팩트 중심의 압축 결론.\n*누락 섹션 존재 시 오답 처리.",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "추론 공간을 '색인-비교-결론'으로 세분화하여 모델이 데이터를 누락할 틈을 주지 않음"
    },
    "F5": {
        "text": "[Bilingual_Format]\n<thought_en>: Step-by-step reasoning and fact listing in English.\n<answer_ko>: 한국어 최종 답변 및 근거 페이지 표기.",
        "control_score": 5,
        "is_english_cot": True,
        "desc": "출력 구조에 영문 사고 공간을 명시적으로 할당하여 모델의 최고 지능 유도"
    }
}

# ---------------------------------------------------------
# 04. 예시 (Few-shot Examples)
# 0번: 제로샷 (Zero-shot)
# ---------------------------------------------------------
FEW_SHOT = {
    "S0": {
        "text": "",
        "control_score": 0,
        "is_english_cot": False,
        "desc": "예시 데이터를 주입하지 않은 제로샷 상태"
    },
    "S1": {
        "text": "[예시 1] Q. 사업목적? A. 시스템 구축\n[예시 2] Q. 총예산? A. 50억",
        "control_score": 1,
        "is_english_cot": False,
        "desc": "극초단기 예시 2개를 통해 약한 Grounding 형식만 환기"
    },
    "S2": {
        "text": "[예시 1] - Q: 예산은? / - A: 제안요청서(5p) 근거, 50억 원입니다.\n[예시 2] - Q: 제출처는? / - A: 공고문(2p) 근거, e-발주 시스템입니다.",
        "control_score": 3,
        "is_english_cot": False,
        "desc": "구체적인 페이지 인용이 삽입된 강제적 단답형 예시 2개 제공"
    },
    "S3": {
        "text": "[성공 예시] Q: 참가자격? / [분석] 자본 10억, 실적 필요 명시(3p) / [정답] 위 요건 충족 기업\n[거절 예시] Q: 야간수당? / [분석] 본문 내 관련 수당 언급 없음 / [정답] 정보 없음",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "'정보 없음' 거절 케이스를 주입하여 환각 발생을 방지함"
    },
    "S4": {
        "text": "[성공 예시]\nQ. 제출 기한?\n[색인] 1p: 시작 11/1, 3p: 마감 11/10 확인.\n[비교] 질문의 '전체 기간' 조건과 대조 완료.\n[정답] 11/1 ~ 11/10.\n\n[거절 예시]\nQ. 담당자 연락처?\n[색인] 본문 전수 조사 결과 연락처 항목 누락.\n[비교] 질문의 요구 정보가 문서 내에 존재하지 않음.\n[정답] 정보 없음.",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "모델이 데이터를 검색하고 필터링하는 과정을 상세한 성공/거절 케이스로 매뉴얼화함"
    },
    "S5": {
        "text": "[성공 예시]\nQ. 예산 규모 및 VAT 포함 여부?\n1. [Logic_EN] Context p.5: \"Total 5B KRW\". p.12: \"VAT excluded\".\n2. [Logic_EN] Compare with query: Amount matches, VAT status identified.\n3. [Answer_KO] 총액 50억 원 (VAT 별도).\n\n[거절 예시]\nQ. 사업 담당자 연락처?\n1. [Logic_EN] Search all context fields for \"contact\", \"phone\", \"manager\".\n2. [Logic_EN] Result: No relevant information found in the provided text.\n3. [Answer_KO] 정보 없음.",
        "control_score": 5,
        "is_english_cot": True,
        "desc": "[Logic_EN] -> [Answer_KO]의 엔진 최적화 방식에 성공/거절 케이스를 결합해 정확도 극대화"
    }
}

# ---------------------------------------------------------
# 05. 추론 유도 (Thought Trigger / CoT)
# 0번: 추론 유도 없음 (Direct Answer)
# ---------------------------------------------------------
COT_STRATEGY = {
    "T0": {
        "text": "",
        "control_score": 0,
        "is_english_cot": False,
        "desc": "단계별 사고를 유도하지 않고 바로 답하게 하는 상태"
    },
    "T1": {
        "text": "[Trigger] Think step by step.",
        "control_score": 1,
        "is_english_cot": False,
        "desc": "모델의 사고 모드만 살짝 깨우는 최소한의 트리거"
    },
    "T2": {
        "text": "1. 근거추출: 관련 문장 확보 / 2. 사실확인: 데이터 대조 / 3. 답변",
        "control_score": 3,
        "is_english_cot": False,
        "desc": "1. 2. 3. 숫자 체계를 도입하여 논리적 비약을 방지하는 기초 공사"
    },
    "T3": {
        "text": "1. 전수조사: 질문 연관 수치·조건 리스팅 / 2. 검증: 항목별 일치 여부 대조 / 3. 답변",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "정답을 내기 전 '리스팅' 단계를 강제하여 정답이 얻어걸릴 확률을 극대화"
    },
    "T4": {
        "text": "[무결성 가이드]\n리스팅: 질문 관련 본문 내 예산, 날짜, 수치 항목을 누락 없이 나열.\n필터링: 질문의 제한 조건과 나열된 데이터 간의 모순점(VAT, 마감일 등) 상세 대조.\n확정: 검증 통과한 수치만을 최종 답변에 반영.",
        "control_score": 5,
        "is_english_cot": False,
        "desc": "3티어의 로직을 '데이터 필드' 단위로 상세화하여 로컬 모델의 실수를 원천 봉쇄"
    },
    "T5": {
        "text": "[English_Logic_Engine]\nAnalysis: Extract and list all relevant data in English.\nVerification: Cross-check conditions and values in English.\nOutput: Provide final answer in Korean.",
        "control_score": 5,
        "is_english_cot": True,
        "desc": "언어 엔진 교체. 영어 토큰의 논리 밀도를 활용해 지능을 쥐어짜는 최고 단계"
    }
}