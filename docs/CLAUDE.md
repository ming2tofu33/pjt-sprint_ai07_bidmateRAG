# docs/ — 문서 작성 규칙

## 구조

```
docs/
├── architecture.md                    # 전체 파이프라인 아키텍처 (Mermaid 차트)
├── decision-log.md                    # 10개 핵심 의사결정 근거
├── collaboration/
│   ├── branch-strategy.md             # Git 브랜치 전략
│   └── git-worktree-workflow.md       # Worktree 워크플로우
└── superpowers/specs/                 # 설계 스펙 문서
```

## 문서 작성 원칙

### architecture.md
- Mermaid 차트로 파이프라인 시각화
- GitHub에서 렌더링되므로 ```mermaid 코드블록 사용
- 변경 시 파이프라인 수정 사항 반영

### decision-log.md
- 설계 결정을 내릴 때마다 기록
- 형식: 결정 → 배경 → 대안 → 선정 이유 → 근거 데이터
- 노트북 인사이트와 연결 (참조 노트북 번호 명시)

### 실험 결과 기록
- 노트북(experiments/notebooks/) 안에 인사이트 마크다운 셀로 기록
- 핵심 결정은 decision-log.md에도 반영
- 수치 데이터는 노트북에, 의사결정 근거는 decision-log에

### 비공개 데이터 주의
- 원본 RFP 문서 내용을 문서에 직접 포함하지 말 것
- 통계/분석 결과만 기록 (2차 가공 형태)
