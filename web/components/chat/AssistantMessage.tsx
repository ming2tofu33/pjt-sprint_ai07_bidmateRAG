import { RotateCcw } from "lucide-react";
import type { Message } from "@/lib/types";
import { MarkdownWithCitations } from "./MarkdownWithCitations";

interface Props {
  message: Message;
  showTail?: boolean;
  showMetadata?: boolean;
  /** 제공되면 에러 버블 아래에 "다시 시도" 버튼이 렌더된다. */
  onRetry?: () => void;
}

export function AssistantMessage({
  message,
  showTail = true,
  showMetadata = true,
  onRetry,
}: Props) {
  const radius = showTail
    ? "rounded-[20px_20px_20px_4px]"
    : "rounded-[20px]";

  if (message.error) {
    return (
      <div className="flex flex-col items-start gap-1.5 px-2">
        <div
          className={`imessage-bubble-enter max-w-[80%] border border-destructive/40 bg-destructive/10 px-[14px] py-[10px] text-[15px] text-destructive ${radius}`}
        >
          {message.content}
        </div>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="ml-3 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[12px] font-medium text-[var(--imessage-blue)] transition-colors hover:bg-[color-mix(in_oklab,var(--imessage-blue)_10%,transparent)]"
          >
            <RotateCcw className="size-3" strokeWidth={2.25} />
            다시 시도
          </button>
        )}
      </div>
    );
  }

  // content가 비어있으면 "생성 중" 상태 — 버블 안에 iMessage 타이핑 점만 표시.
  // 토큰이 도착해서 content가 생기면 점이 자연스럽게 답변으로 치환된다.
  const isPending = message.content.length === 0;

  return (
    <div className="flex flex-col items-start gap-1 px-2">
      <div
        className={`imessage-bubble-assistant imessage-bubble-enter max-w-[80%] ${radius} ${
          isPending
            ? "flex items-center gap-1 px-4 py-3"
            : "px-[14px] py-[10px] text-[15px] leading-[1.4]"
        }`}
        aria-live={isPending ? "polite" : undefined}
        role={isPending ? "status" : undefined}
        aria-label={isPending ? "답변 생성 중" : undefined}
      >
        {isPending ? (
          <>
            <span className="imessage-dot h-2 w-2 rounded-full bg-neutral-500" />
            <span className="imessage-dot h-2 w-2 rounded-full bg-neutral-500" />
            <span className="imessage-dot h-2 w-2 rounded-full bg-neutral-500" />
          </>
        ) : (
          <MarkdownWithCitations content={message.content} />
        )}
      </div>
      {showMetadata && message.metadata && (
        <div className="ml-3 flex gap-2 text-[10px] text-muted-foreground">
          <span>{message.metadata.model}</span>
          <span>·</span>
          <span>{(message.metadata.latency_ms / 1000).toFixed(1)}s</span>
          <span>·</span>
          <span>{message.metadata.token_usage.total ?? 0} tokens</span>
        </div>
      )}
    </div>
  );
}
