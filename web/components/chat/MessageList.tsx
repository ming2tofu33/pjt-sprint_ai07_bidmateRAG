"use client";

import { Fragment } from "react";
import { MessageCircle } from "lucide-react";
import { useStore } from "@/store/useStore";
import type { Message } from "@/lib/types";
import { UserMessage } from "./UserMessage";
import { AssistantMessage } from "./AssistantMessage";
import { TimestampDivider } from "./TimestampDivider";

const FIVE_MIN_MS = 5 * 60 * 1000;

export function MessageList() {
  const messages = useStore((s) => s.messages);
  const isLoading = useStore((s) => s.isLoading);
  const retryLastMessage = useStore((s) => s.retryLastMessage);

  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-center">
        <div>
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-[var(--imessage-bubble)] text-foreground/70">
            <MessageCircle className="size-7" strokeWidth={1.75} />
          </div>
          <div className="mt-4 text-[17px] font-semibold">BidMate</div>
          <div className="mt-1 text-[13px] text-muted-foreground">
            iMessage · 방금 전
          </div>
          <div className="mt-6 text-[13px] text-muted-foreground">
            사이드바에서 문서를 클릭하거나{" "}
            <code className="rounded bg-[var(--imessage-bubble)] px-1">@</code>
            {" "}·{" "}
            <code className="rounded bg-[var(--imessage-bubble)] px-1">/</code>
            를 입력해 시작하세요.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col py-3">
      {messages.map((m, i) => {
        const prev: Message | undefined = messages[i - 1];
        const next: Message | undefined = messages[i + 1];

        const showDivider =
          !prev || m.createdAt - prev.createdAt > FIVE_MIN_MS;

        const isLastInGroup =
          !next ||
          next.role !== m.role ||
          next.createdAt - m.createdAt > FIVE_MIN_MS;

        const isFirstInGroup =
          !prev ||
          prev.role !== m.role ||
          m.createdAt - prev.createdAt > FIVE_MIN_MS;

        const spacing = showDivider
          ? ""
          : isFirstInGroup
          ? "mt-[8px]"
          : "mt-[2px]";

        // 마지막 메시지 & assistant & 에러 상태 & 스트리밍 중이 아닐 때만 retry 노출
        const isLastMessage = i === messages.length - 1;
        const canRetry =
          isLastMessage && m.role === "assistant" && !!m.error && !isLoading;

        return (
          <Fragment key={m.id}>
            {showDivider && <TimestampDivider ts={m.createdAt} />}
            <div className={spacing}>
              {m.role === "user" ? (
                <UserMessage message={m} showTail={isLastInGroup} />
              ) : (
                <AssistantMessage
                  message={m}
                  showTail={isLastInGroup}
                  showMetadata={isLastInGroup}
                  onRetry={canRetry ? retryLastMessage : undefined}
                />
              )}
            </div>
          </Fragment>
        );
      })}
    </div>
  );
}
