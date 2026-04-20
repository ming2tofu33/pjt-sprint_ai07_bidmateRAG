# Data Quality Current Status

## 목적

이 문서는 현재 데이터 품질 관련 작업이 어디까지 반영되어 있는지 빠르게 공유하기 위한 상태 요약 문서다.
특히 아래 3가지를 구분해서 본다.

- 전체 원본 데이터 기준으로 공통 적용되는 작업
- 일부 확인된 케이스를 중심으로 반영된 작업
- 아직 전수 보강이 필요한 작업

관련 상세 정책은 아래 문서를 본다.

- `docs/data-quality/canonical-selection-policy.md`
- `docs/data-quality/metadata-resolution-design.md`
- `docs/data-quality/project-alias-policy.md`

## 현재 결론

현재 반영된 데이터 작업은 크게 4축이다.

1. 중복 문서 정본 처리
2. 기관명 정규화 및 alias 매칭
3. 사업명 alias 기반 메타데이터 매칭 보강
4. retrieval 단계의 과한 필터링 방지

이 중에서 기관명 정규화와 retrieval 안정화는 거의 공통 로직으로 반영되어 있다.
반면 중복 문서 목록과 사업명 alias 사전은 아직 확인된 고위험 케이스 중심으로 채워져 있어 전수 보강 여지가 남아 있다.

## 항목별 상태

### 1. 중복 문서 정본 처리

상태: 부분 반영

적용 파일:

- `configs/data_quality/duplicates_map.csv`
- `src/bidmate_rag/loaders/metadata_loader.py`
- `src/bidmate_rag/pipelines/ingest.py`

현재 반영 내용:

- `duplicates_map.csv`에 등록된 중복 문서는 `canonical_file` 기준으로 정본을 선택한다.
- `metadata_loader`에서 `canonical_file`, `is_duplicate`, `ingest_enabled`, `ingest_file`, `resolved_agency`를 생성한다.
- ingest 단계에서는 정본 기준 파일만 파싱 대상으로 삼는다.

해석:

- 구조 자체는 전체 데이터에 적용되는 방식이다.
- 다만 실제로 `duplicates_map.csv`에 등록된 문서는 현재까지 확인된 중복 후보 중심이다.
- 따라서 "정본 처리 로직"은 공통 적용이지만 "중복 후보 목록"은 아직 전수 확정 상태가 아니다.

### 2. 기관명 정규화 및 alias 매칭

상태: 공통 반영

적용 파일:

- `src/bidmate_rag/retrieval/agency_matching.py`
- `src/bidmate_rag/evaluation/dataset.py`

현재 반영 내용:

- 법인 표기 `(사)`, `(재)`, `(주)` 등을 제거한 비교용 기관명을 만든다.
- `입찰공고`, `전자조달` 같은 꼬리 표현을 걷어낸다.
- `KOICA -> 코이카` 같은 일부 약어 alias를 처리한다.
- eval metadata_filter 정규화 시 전체 metadata를 읽어 기관 alias 후보를 만든다.

해석:

- 이 로직은 특정 배치 전용이 아니라 전체 원본 데이터에 공통으로 적용된다.
- 새로운 질문이나 새로운 평가셋에도 같은 방식으로 작동한다.
- 다만 acronym alias 사전은 앞으로 더 넓힐 수 있다.

### 3. 사업명 alias 기반 매칭

상태: 부분 반영

적용 파일:

- `configs/data_quality/project_alias_map.csv`
- `src/bidmate_rag/evaluation/dataset.py`

현재 반영 내용:

- metadata의 `사업명`, `파일명`, `canonical_file`, `ingest_file`에서 사업명 후보를 자동 수집한다.
- `project_alias_map.csv`에 수동 alias를 추가로 넣을 수 있다.
- eval metadata_filter 정규화 시 기관 범위를 먼저 좁힌 뒤 사업명 alias를 매칭한다.

해석:

- 자동 수집 부분은 전체 metadata 기준이라 공통 적용이다.
- 수동 alias 사전은 현재 eval에서 자주 깨진 케이스 위주로 우선 반영된 상태다.
- 따라서 사업명 alias는 아직 "전체 100개 문서 전수 커버"라고 보기는 어렵다.

### 4. retrieval 안정화 로직

상태: 공통 반영

적용 파일:

- `src/bidmate_rag/retrieval/retriever.py`
- `src/bidmate_rag/retrieval/filters.py`

현재 반영 내용:

- history가 `content`뿐 아니라 `user`, `assistant` 형태로 들어와도 읽는다.
- project clue가 너무 일반적인 따옴표 문구까지 잡지 않도록 제한한다.
- `where_document`가 결과를 0건으로 만들면 문서 필터 없이 한 번 더 조회한다.
- 문서 shortlist가 과하게 좁혀져 0건이 되면 base metadata filter로 다시 조회한다.

해석:

- 이 부분은 특정 문항 대응이라기보다 검색기의 기본 안정성을 높이는 공통 수정이다.
- 시나리오 A, B 모두 retrieval 공통 경로에서 의미가 있다.

## 현재 기준에서 "전체 적용"과 "부분 적용" 구분

### 전체 적용으로 봐도 되는 것

- metadata_loader 기반 정본/중복 처리 구조
- ingest 시 canonical 기준 파싱 흐름
- 기관명 정규화 및 기관 alias 매칭 구조
- eval metadata_filter 정규화 로직
- retrieval fallback, history 보강, 과한 섹션 필터 방지

### 아직 부분 적용인 것

- `duplicates_map.csv`에 등록된 중복 후보 목록
- `project_alias_map.csv`에 수동 등록한 사업명 alias
- acronym, 축약형, 영문 사업명에 대한 사전 coverage

## 지금 시점에서 남은 데이터 담당 우선순위

1. `project_alias_map.csv`를 전체 사업명 기준으로 전수 보강
2. `duplicates_map.csv` 추가 후보 점검
3. metadata와 원본 문서의 기관명/사업명 일치성 재검수
4. `eval_batch_01`, `eval_batch_31`, `eval_batch_33` 교차 확인

## 실무적으로 보는 현재 상태

지금 상태는 "데이터 품질 작업의 뼈대는 반영되었고, 운영용 사전은 계속 넓혀야 하는 단계"에 가깝다.
즉 기반 로직은 이미 코드에 들어가 있으나, 사전형 자산인 `duplicates_map.csv`와 `project_alias_map.csv`는 전수 점검을 거치며 점진적으로 완성도를 높여야 한다.
