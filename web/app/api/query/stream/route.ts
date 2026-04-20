// Route Handler: proxies POST /api/query/stream to the FastAPI backend at
// :8100 with a long timeout and pass-through streaming. We can't use
// `next.config.ts` rewrites for this path because the rewrites proxy buffers
// responses and drops long-running connections (~10s) before the RAG
// pipeline (gpt-5-mini) finishes streaming.

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
// 최대 실행 시간 (초). 자체 호스팅이라 실질적 제한은 없지만 명시적으로 넉넉히.
export const maxDuration = 300;

const BACKEND_URL = "http://localhost:8100/api/query/stream";

export async function POST(req: Request): Promise<Response> {
  const body = await req.text(); // raw passthrough — schema는 백엔드가 검증

  let upstream: Response;
  try {
    upstream = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      // Node 18+ fetch는 기본적으로 스트리밍 body를 지원.
      // duplex:"half"는 Node에서 요청 body를 스트림으로 보낼 때만 필요.
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return new Response(
      JSON.stringify({ detail: `backend unreachable: ${message}` }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => upstream.statusText);
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  // 스트림 body를 그대로 전달. SSE 헤더를 명시해서 프록시/브라우저가 버퍼링하지 않게 함.
  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
