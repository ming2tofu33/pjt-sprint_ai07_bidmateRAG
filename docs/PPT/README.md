# BidMate RAG — PPT 웹사이트

20분 발표용 30장 PPT를 HTML 슬라이드 + iframe 네비게이션으로 구성한 웹사이트.

## 🚀 로컬에서 보기

프로젝트 루트에서:

```bash
make ppt
```

브라우저로 `http://localhost:8000` 접속. Ctrl+C로 종료.

**make를 못 쓰는 환경**이면:

```bash
cd docs/PPT
python3 -m http.server 8000
```

## ⌨️ 키보드 단축키

| 키 | 동작 |
|---|---|
| `←` / `PageUp` | 이전 슬라이드 |
| `→` / `PageDown` / `Space` | 다음 슬라이드 |
| `Home` | 1번 슬라이드 |
| `End` | 30번 슬라이드 |
| `F` | 풀스크린 토글 |
| `ESC` | 풀스크린 해제 (브라우저 기본) |

## 📝 슬라이드 수정하기

1. `make ppt`로 서버 실행 (또는 VSCode Live Server 확장)
2. 브라우저에서 수정할 슬라이드로 이동
3. **nav bar 중앙의 파일명** (`slide05.html` 등) 을 확인
4. 해당 파일을 편집기에서 열어 수정
5. **저장 후 브라우저 F5 (새로고침)** — 즉시 반영

### 수정 범위
- 슬라이드 내용: `slide01.html` ~ `slide30.html` 각각 편집
- 네비게이션·스케일·키 바인딩: `index.html`
- 실측 수치·메시지 원본: `PPT_콘텐츠_브리프.md` / `PPT 슬라이드 설계 v2.md`

## 🔗 특정 슬라이드 직접 링크

URL 해시로 특정 슬라이드 바로 열기:

```
http://localhost:8000/#15   ← 슬라이드 15로 바로 이동
```

GitHub Pages 배포 후에는 `https://.../PPT/#15` 같은 식으로 공유 가능.

## 📐 슬라이드 디자인 규격

모든 슬라이드가 **1280×720** 고정 크기. `index.html`은 CSS `transform: scale()`로 뷰포트에 맞게 자동 축소·확대 (비율 유지). 슬라이드 원본 파일은 수정할 필요 없음.

## 🗂 참고 문서

- `PPT_콘텐츠_브리프.md` — 30장 콘텐츠 원본 (Genspark.ai 투입용)
- `PPT 슬라이드 설계 v2.md` — 팀 내부 관리용 설계서 (발표자 배분·시간·R&R)
- `../superpowers/specs/2026-04-19-ppt-content-brief-design.md` — 콘텐츠 브리프 설계 스펙
