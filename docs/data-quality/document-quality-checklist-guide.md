# Document Quality Checklist Guide

## 목적

`cleaned_documents.parquet` 기준으로 파일별 점검표를 자동 생성해,
파싱 결과가 비정상인 문서나 중복 처리 상태를 빠르게 확인하기 위한 가이드다.

## 생성 명령

```powershell
uv run python scripts/build_document_quality_report.py
```

기본 출력 경로:

- `data/processed/document_quality_checklist.csv`

## 주요 컬럼

- `파일명`
  - 메타데이터 기준 원본 파일명
- `ingest_file`
  - 실제 파싱에 사용된 파일명
- `canonical_file`
  - 중복 그룹 기준 정본 파일명
- `is_duplicate`
  - 중복 행 여부
- `ingest_enabled`
  - 실제 ingest 대상 여부
- `본문_글자수`
  - 파싱 직후 본문 길이
- `정제_글자수`
  - cleaner 적용 후 본문 길이
- `품질상태`
  - `ok`, `canonical_redirect`, `duplicate_skip`, `review_required`
- `품질플래그`
  - `duplicate_skip`, `canonical_redirect`, `empty_text`, `short_text`, `format_variant_docx`
- `점검사유`
  - 사람이 바로 읽을 수 있는 품질 설명
- `검토상태`, `검토메모`
  - 팀 수동 점검용 컬럼

## 해석 기준

### ok

- 별도 조치 없이 사용 가능

### canonical_redirect

- 현재 메타데이터 행은 유지되지만 실제 파싱은 정본 파일로 수행됨
- 예: 중복본만 CSV에 있고 정본 파일로 redirect되는 경우

### duplicate_skip

- 같은 문서의 정본 행이 별도로 존재해 현재 행은 ingest에서 제외됨

### review_required

- 아래 중 하나라도 해당하면 우선 검토 대상이다.
  - `empty_text`
  - `short_text`

## 운영 팁

1. `review_required`부터 먼저 확인한다.
2. 그 다음 `canonical_redirect`, `duplicate_skip`을 보며 중복 처리 상태를 점검한다.
3. threshold가 너무 빡빡하거나 느슨하면 `--short-threshold` 값으로 조정한다.
