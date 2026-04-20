# Metadata Resolution Design

## 목적

`data_list.csv`에 있는 메타데이터와 `duplicates_map.csv`에 정의한 중복 관계를 결합해,
ingest 이전 단계에서 "실제로 어떤 파일을 파싱할지"와 "어떤 메타데이터를 대표값으로 쓸지"
명확하게 결정하는 구조를 정의한다.

이 설계의 목표는 다음과 같다.

- 동일 문서가 중복으로 인덱싱되지 않도록 한다.
- CSV에만 있는 메타데이터를 무조건 버리지 않는다.
- 정본 파일과 메타데이터 대표값의 출처를 추적 가능하게 남긴다.

## 대상 파일

- 입력 메타데이터
  - `data/raw/metadata/data_list.csv`
- 중복 관리 기준
  - `configs/data_quality/duplicates_map.csv`
- 적용 대상 코드
  - `src/bidmate_rag/loaders/metadata_loader.py`
  - `src/bidmate_rag/pipelines/ingest.py`

## metadata_loader 역할

`metadata_loader`는 기존의 단순 CSV 로더에서 다음 역할까지 맡는다.

1. `data_list.csv` 로드
2. `duplicates_map.csv` 로드
3. 파일명 기준으로 duplicate / canonical 관계 부여
4. ingest 전용 제어 컬럼 생성
5. 대표 메타데이터 컬럼 생성

즉, ingest 단계는 "파일을 실제로 읽는 역할"만 하고, 중복 여부 판단은
`metadata_loader`에서 끝내는 구조로 가져간다.

## 추가할 컬럼

### duplicate_group_id

- 의미: 같은 문서 그룹에 속하는 행을 식별하는 ID
- 예: `DUP-001`, `DUP-002`

### canonical_file

- 의미: 해당 행이 속한 그룹에서 정본으로 간주하는 파일명
- 중복이 아닌 경우에는 자기 자신의 파일명을 그대로 사용

### is_duplicate

- 의미: 해당 CSV 행이 중복 행인지 여부
- 타입: boolean
- 중복이 아닌 경우 `false`

### ingest_enabled

- 의미: 해당 CSV 행을 실제 ingest 대상으로 사용할지 여부
- 타입: boolean
- canonical 행만 `true`
- duplicate 행은 `false`

### ingest_file

- 의미: 실제로 raw 폴더에서 열어야 하는 파일명
- canonical 행은 자기 자신의 정본 파일명
- duplicate 행도 canonical 파일명을 가리킬 수 있지만, `ingest_enabled=false`이면 실제 파싱은 수행하지 않음

### resolved_agency

- 의미: 대표 발주기관명
- canonical-selection-policy에서 정의한 정본 기준에 따라 선택

### original_agency

- 의미: CSV 원본의 `발주 기관` 값을 보존하는 컬럼
- 기존 `발주 기관`은 downstream 호환을 위해 `resolved_agency`로 치환 가능

### alias_files_json

- 의미: 같은 duplicate group 안에 속한 다른 파일명 목록
- 타입: JSON 문자열
- 검색 별칭 또는 감사 추적용으로 사용

### metadata_sources_json

- 의미: 어떤 메타데이터가 어느 파일 행에서 왔는지 기록하는 JSON 문자열
- 초기 버전에서는 최소한 `agency_source`, `canonical_source` 수준만 남겨도 충분

## 동작 규칙

### 1. duplicates_map.csv가 없는 경우

- 기존과 동일하게 동작한다.
- 즉, 모든 행은 다음 기본값을 가진다.
  - `duplicate_group_id = None`
  - `canonical_file = 파일명`
  - `is_duplicate = false`
  - `ingest_enabled = true`
  - `ingest_file = 파일명`
  - `resolved_agency = 발주 기관`

### 2. duplicates_map.csv에 등록된 경우

- `source_file` 기준으로 중복 정보를 붙인다.
- `canonical_file`은 map 기준으로 채운다.
- `is_duplicate=true`이면 기본적으로 `ingest_enabled=false`
- `is_duplicate=false`이면 `ingest_enabled=true`

### 3. CSV에 duplicate 행만 있고 canonical 행은 없는 경우

예: `BioIN_...` 행은 CSV에 있는데 `한국보건산업진흥원_...` 행은 CSV에 없는 상황

- 해당 행은 `is_duplicate=true` 상태이더라도 CSV에 존재하는 유일한 대표 행이므로
  `ingest_enabled=true`로 승격한다.
- 대신 실제로 읽는 파일은 `ingest_file=canonical_file`로 설정한다.
- 즉, 메타데이터는 현재 행에서 가져오되 파싱은 정본 파일로 수행한다.

### 4. CSV에 duplicate 행과 canonical 행이 모두 있는 경우

예: `국가과학기술지식정보서비스_...` / `한국한의학연구원_...`

- canonical 행만 `ingest_enabled=true`
- duplicate 행은 `ingest_enabled=false`
- duplicate 행의 메타데이터는 `metadata_sources_json` 보강 출처로 보존 가능

## 발주 기관 처리 규칙

- `original_agency`에 CSV 원본값을 저장한다.
- `resolved_agency`는 정본 기준으로 결정한다.
- downstream에서 발주기관 필터가 `발주 기관` 컬럼을 계속 사용하고 있다면,
  `발주 기관` 값을 `resolved_agency`로 덮어쓴다.

이렇게 해야 retrieval 필터와 UI가 별도 수정 없이 정본 기준 기관명으로 동작한다.

## ingest에서 기대하는 입력 상태

`metadata_loader`가 끝난 뒤 ingest는 다음만 믿고 동작하면 된다.

- `ingest_enabled=true`인 행만 실제 파싱
- 실제 파싱 경로는 `ingest_file`
- 로그에 `파일명`과 `ingest_file`을 함께 출력

즉 ingest는 "중복 판단"을 하지 않고,
"이미 판단된 결과를 실행하는 단계"로 단순화한다.

## 구현 순서

1. `metadata_loader.py`
   - duplicates map 읽기 함수 추가
   - resolution 컬럼 생성
2. `ingest.py`
   - `ingest_enabled` 필터 적용
   - `ingest_file` 기준으로 raw 파일 열기
3. 테스트
   - duplicates map 없음
   - duplicate만 있는 경우
   - duplicate와 canonical이 같이 있는 경우

## 현재 기준 예시

### DUP-001

- CSV 대표 행: `BioIN_...`
- 실제 파싱 파일: `한국보건산업진흥원_...`
- 기대 상태:
  - `is_duplicate = true`
  - `ingest_enabled = true`
  - `ingest_file = 한국보건산업진흥원_...`
  - `resolved_agency = 한국보건산업진흥원`

### DUP-002

- duplicate 행: `국가과학기술지식정보서비스_...`
- canonical 행: `한국한의학연구원_...`
- 기대 상태:
  - 국가과학기술지식정보서비스 행
    - `is_duplicate = true`
    - `ingest_enabled = false`
  - 한국한의학연구원 행
    - `is_duplicate = false`
    - `ingest_enabled = true`
    - `ingest_file = 한국한의학연구원_...`
    - `resolved_agency = 한국한의학연구원`
