import type {
  DocumentSummary,
  DocumentDetail,
  SlashCommandMeta,
  QueryRequest,
  QueryResponse,
  QueryStreamEvent,
} from "./types";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // ignore — use statusText
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export async function listDocuments(): Promise<{
  documents: DocumentSummary[];
  total: number;
}> {
  return fetchJson("/api/documents");
}

export async function getDocument(docId: string): Promise<DocumentDetail> {
  return fetchJson(`/api/documents/${encodeURIComponent(docId)}`);
}

export async function listCommands(): Promise<{ commands: SlashCommandMeta[] }> {
  return fetchJson("/api/commands");
}

export async function postQuery(request: QueryRequest): Promise<QueryResponse> {
  return fetchJson("/api/query", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/**
 * POST /api/query/stream — SSE 스트리밍. 각 이벤트를 `onEvent`로 전달한다.
 *
 * 이벤트 순서:
 *   1. `retrieval` (1회) — 검색 완료 직후, 근거 카드 즉시 표시
 *   2. `token` (여러 번) — LLM delta가 올 때마다
 *   3. `done` (1회) — 최종 metadata
 *   또는 `error` (1회) — 중간 실패 시
 *
 * `signal`로 요청 취소 가능 (추후 "중지" 버튼 연동 시 사용).
 */
export async function postQueryStream(
  request: QueryRequest,
  onEvent: (event: QueryStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok || !response.body) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // ignore — use statusText
    }
    throw new Error(`${response.status}: ${detail}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE 프레임은 `\n\n`으로 구분된다. 마지막 조각은 다음 chunk에 이어질 수 있으므로 보류.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        // 프레임은 여러 라인일 수 있고, data: 로 시작하는 라인만 JSON 페이로드.
        // (현재 백엔드는 data: 한 줄만 보내므로 단순화 가능하지만 방어적으로 처리.)
        const dataLines = frame
          .split("\n")
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice(6));
        if (dataLines.length === 0) continue;
        const payload = dataLines.join("\n");
        try {
          const event = JSON.parse(payload) as QueryStreamEvent;
          onEvent(event);
        } catch (err) {
          console.error("SSE parse failed", err, payload);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
