"use client";

import { Trash2 } from "lucide-react";
import type { Chat } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  chat: Chat;
  active: boolean;
  onOpen: () => void;
  onRequestDelete: () => void;
}

function formatRelativeTime(ts: number): string {
  const diffMs = Date.now() - ts;
  const minute = 60_000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diffMs < minute) return "방금 전";
  if (diffMs < hour) return `${Math.floor(diffMs / minute)}분 전`;
  if (diffMs < day) return `${Math.floor(diffMs / hour)}시간 전`;
  const d = new Date(ts);
  const now = new Date();
  const isSameYear = d.getFullYear() === now.getFullYear();
  return new Intl.DateTimeFormat("ko-KR", {
    month: "long",
    day: "numeric",
    year: isSameYear ? undefined : "numeric",
  }).format(d);
}

export function ChatListItem({ chat, active, onOpen, onRequestDelete }: Props) {
  // 사용자 메시지 수만 카운트 (assistant 빈 placeholder 제외해서 체감과 일치)
  const msgCount = chat.messages.filter((m) => m.role === "user").length;

  return (
    <div
      className={cn(
        "group relative rounded-md border border-transparent transition-colors",
        active
          ? "border-[var(--imessage-blue)]/40 bg-accent"
          : "hover:bg-accent"
      )}
    >
      <button
        type="button"
        onClick={onOpen}
        className="w-full rounded-md px-3 py-2 pr-10 text-left"
      >
        <div className="truncate text-[13px] font-medium text-foreground">
          {chat.title || "새 대화"}
        </div>
        <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <span>{formatRelativeTime(chat.updatedAt)}</span>
          <span>·</span>
          <span>{msgCount}개 메시지</span>
        </div>
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onRequestDelete();
        }}
        title="이 대화 삭제"
        aria-label={`${chat.title} 대화 삭제`}
        className={cn(
          "absolute right-1.5 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground transition-all",
          // 기본: 숨김 (hover/focus-within에서만 노출)
          "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
          "hover:bg-muted hover:text-destructive"
        )}
      >
        <Trash2 className="size-3.5" />
      </button>
    </div>
  );
}
