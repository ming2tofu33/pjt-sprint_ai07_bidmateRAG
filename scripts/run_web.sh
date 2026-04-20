#!/usr/bin/env bash
# FastAPI + Next.js 동시 실행 헬퍼 (BidMate 사용자용 웹 UI)
set -e

cd "$(git rev-parse --show-toplevel)"

MODE="${1:-dev}"
API_PORT="${API_PORT:-8100}"
WEB_PORT="${WEB_PORT:-3000}"

cleanup() {
  echo ""
  echo "Shutting down services..."
  kill "${UVICORN_PID:-}" "${WEB_PID:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI on :${API_PORT}..."
PYTHONPATH=. uv run uvicorn bidmate_rag.web_api.main:app \
  --port "${API_PORT}" --reload &
UVICORN_PID=$!

sleep 2

if [[ "${MODE}" == "prod" ]]; then
  echo "Building Next.js..."
  (cd web && npm run build)
  echo "Starting Next.js (prod) on :${WEB_PORT}..."
  (cd web && PORT="${WEB_PORT}" npm run start) &
else
  echo "Starting Next.js (dev) on :${WEB_PORT}..."
  (cd web && PORT="${WEB_PORT}" npm run dev) &
fi
WEB_PID=$!

echo ""
echo "BidMate web UI  — http://localhost:${WEB_PORT}"
echo "FastAPI docs    — http://localhost:${API_PORT}/docs"
echo "Press Ctrl+C to stop"
wait
