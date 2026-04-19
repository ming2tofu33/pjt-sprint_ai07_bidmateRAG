import { RotateCcw } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { Message } from "@/lib/types";

import { MarkdownWithCitations } from "./MarkdownWithCitations";

interface Props {
  message: Message;
  showTail?: boolean;
  showMetadata?: boolean;
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

  const isPending = message.content.length === 0;
  const isCalculation = message.metadata?.answer_source === "calculation";
  const sourceLabel =
    message.metadata?.answer_source_label ??
    (isCalculation ? "SQL 계산 응답" : "LLM 생성 응답");
  const calculationLabel =
    message.metadata?.calculation_label ?? "구조화 계산 응답";

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
          <div className="space-y-2">
            {isCalculation && (
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary" className="text-[10px]">
                  {calculationLabel}
                </Badge>
                <span className="text-[11px] text-muted-foreground">
                  문서 구조화 필드와 SQL 계산 결과를 먼저 사용한 응답입니다.
                </span>
              </div>
            )}
            <MarkdownWithCitations content={message.content} />
          </div>
        )}
      </div>
      {showMetadata && message.metadata && (
        <div className="ml-3 flex gap-2 text-[10px] text-muted-foreground">
          {isCalculation ? (
            <>
              <span>{sourceLabel}</span>
              <span aria-hidden="true">·</span>
              <span>{(message.metadata.latency_ms / 1000).toFixed(1)}s</span>
              {message.citations && message.citations.length > 0 && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>근거 {message.citations.length}건</span>
                </>
              )}
            </>
          ) : (
            <>
              <span>{message.metadata.model}</span>
              <span aria-hidden="true">·</span>
              <span>{(message.metadata.latency_ms / 1000).toFixed(1)}s</span>
              <span aria-hidden="true">·</span>
              <span>{message.metadata.token_usage.total ?? 0} tokens</span>
            </>
          )}
        </div>
      )}
    </div>
  );
}
