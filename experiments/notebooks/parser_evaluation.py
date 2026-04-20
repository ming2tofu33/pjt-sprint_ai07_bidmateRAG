import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. API 클라이언트 및 경로 설정
client = OpenAI()

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base_dir = os.path.join(project_root, "data", "test")
modern_dir = os.path.join(base_dir, "hwp-hwpx-parser", "hwp")
raw_dir = os.path.join(base_dir, "olefile", "hwp")
kordoc_dir = os.path.join(base_dir, "kordoc", "hwp")
output_report = os.path.join(project_root, "experiments", "tracking", "hwp_evaluation_report.csv")

# 2. 평가 대상 파일 목록 확보
modern_files = [f for f in os.listdir(modern_dir) if f.endswith(".txt") and "_ERROR" not in f]


# 3. 평가 메인 로직
def evaluate_hwp_parsers(filename, text_modern, text_raw, text_kordoc):
    """LLM을 심판으로 사용하여 세 파서의 품질 비교"""

    # 토큰 절약을 위해 앞 3000자만 사용
    text_modern_clipped = text_modern[:3000]
    text_raw_clipped = text_raw[:3000]
    text_kordoc_clipped = text_kordoc[:3000]

    system_prompt = """당신은 RAG 시스템의 데이터 전처리 품질을 검증하는 10년차 AI 엔지니어입니다.
이 RAG 시스템은 공공 입찰 공고 문서(HWP)에서 핵심 요구사항을 검색하고 LLM이 답변을 생성하는 데 사용됩니다.
동일한 HWP 파일에서 추출된 세 가지 텍스트 데이터를 비교하여 어떤 것이 이 시스템에 더 유리한지 판정하세요.

[평가 대상]
- Parser A (hwp-hwpx-parser): hwp-hwpx-parser 라이브러리로 추출한 결과
- Parser B (olefile raw text): OLE 구조에서 순수 텍스트만 강제로 뽑아낸 결과
- Parser C (kordoc): Node.js 기반 CLI 파서로 변환한 마크다운 결과

[평가 기준]
1. 표 보존력(Table): 표의 내용이 깨지지 않고 마크다운(|---|) 등으로 잘 구조화되었는가? (최우선순위)
2. 구조 보존력(Structure): 제목, 섹션, 항목 번호, 목록 구조가 원문 의미를 유지한 채 잘 보존되었는가?
3. 가독성(Readability): 문장 간 연결이 자연스럽고 텍스트 흐름이 잘 읽히는가?
4. 노이즈 수준(Noise): 목차 점선, 페이지 번호, 반복 구분선, 태그 잔재, 의미 없는 특수문자가 적은가?
5. RAG 적합성: 이 데이터를 chunking 및 Vector DB에 넣었을 때 입찰 요구사항 검색 정확도와 LLM 답변 품질에 유리한가?

[평가 원칙]
- 표 보존력은 가장 중요한 기준으로 반영합니다.
- 단순히 텍스트 길이가 길다고 높은 점수를 주지 마세요.
- 특수문자가 많더라도 문서 구조 보존에 필요한 경우는 무조건 감점하지 마세요.
- 최종 판단은 실제 RAG 검색 및 생성에 더 유리한 결과인지 기준으로 내리세요.
- 이 평가는 전처리 이전 상태의 추출 결과를 기준으로 합니다.
- 표가 많이 보인다고 무조건 높은 점수를 주지 말고, 정보형 표와 목차/레이아웃형 표를 구분하여 평가하세요.
- 특히 요구사항, 예산, 제출 방식, 추진 목적, 수행 범위와 같은 핵심 입찰 정보를 추출하기에 유리한지를 중점적으로 판단하세요.

[출력 포맷 (쉼표로 구분된 1줄의 CSV)]
파일명, hwp-hwpx-parser점수(1-10), olefile점수(1-10), kordoc점수(1-10), 승자(hwp-hwpx-parser/olefile/kordoc/무승부), \"핵심근거(1문장)\""""

    user_prompt = f"""
[파일명]: {filename}

--- [hwp-hwpx-parser 결과물] ---
{text_modern_clipped}

--- [olefile 결과물] ---
{text_raw_clipped}

--- [kordoc 결과물] ---
{text_kordoc_clipped}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


# 4. 실행 루프
print(f"🚀 총 {len(modern_files)}개 HWP 결과물에 대한 최종 평가를 시작합니다.")
results = ["파일명,hwp-hwpx-parser_점수,olefile_점수,kordoc_점수,승자,선택_근거"]

for filename in modern_files:
    try:
        base_name = filename.replace(".txt", "")  # e.g. 한국수자원공사.hwp

        with open(os.path.join(modern_dir, filename), "r", encoding="utf-8") as f:
            text_modern = f.read()
        with open(os.path.join(raw_dir, filename), "r", encoding="utf-8") as f:
            text_raw = f.read()
        with open(os.path.join(kordoc_dir, f"{base_name}_kordoc.md"), "r", encoding="utf-8") as f:
            text_kordoc = f.read()

        print(f"🤖 판독 중: {filename} ...", end=" ")

        eval_result = evaluate_hwp_parsers(filename, text_modern, text_raw, text_kordoc)
        results.append(eval_result)

        print("완료")

    except FileNotFoundError:
        print("건너뜀 (대조 파일 없음)")
    except Exception as e:
        print(f"에러 발생: {e}")

# 5. 리포트 저장
with open(output_report, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\n✅ HWP 최종 평가 리포트 생성 완료: {output_report}")
