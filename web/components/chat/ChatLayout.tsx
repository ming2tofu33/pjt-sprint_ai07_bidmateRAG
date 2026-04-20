"use client";

import { useEffect, useRef } from "react";
import { useStore } from "@/store/useStore";
import { MessageList } from "./MessageList";
import { EvidencePanel } from "./EvidencePanel";

// 스트리밍 중 사용자가 "여전히 맨 아래를 보고 있는가"를 판단하는 임계값.
// 이 픽셀 이내에 있으면 새 토큰마다 자동으로 따라 내려간다.
const STICKY_THRESHOLD_PX = 100;

export function ChatLayout() {
  const messages = useStore((s) => s.messages);
  const isLoading = useStore((s) => s.isLoading);
  const latestAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const citations = latestAssistant?.citations ?? [];

  const scrollRef = useRef<HTMLDivElement>(null);
  // 사용자가 "맨 아래에 붙어있는가"를 기억 — 위로 스크롤해서 읽는 중이면
  // 새 토큰이 와도 방해하지 않는다. 다시 맨 아래로 내리면 복귀.
  const isStickyRef = useRef(true);

  // 스크롤 위치 추적
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const handleScroll = () => {
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      isStickyRef.current = distanceFromBottom < STICKY_THRESHOLD_PX;
    };
    el.addEventListener("scroll", handleScroll, { passive: true });
    return () => el.removeEventListener("scroll", handleScroll);
  }, []);

  // `contentTick`은 모든 메시지 content 길이 합 — 토큰이 추가될 때마다 바뀌어서
  // 이 effect가 재실행된다. messages.length만 보면 스트리밍 중 발동 X.
  const contentTick = messages.reduce((acc, m) => acc + m.content.length, 0);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (!isStickyRef.current) return;
    // 스트리밍 중엔 'auto'(즉시) — 'smooth' 애니메이션이 겹치면 떨림.
    el.scrollTo({ top: el.scrollHeight, behavior: "auto" });
  }, [messages.length, contentTick, isLoading]);

  return (
    <div className="flex h-full overflow-hidden">
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[640px] px-4">
          <MessageList />
        </div>
      </div>
      <aside className="hidden w-[300px] shrink-0 overflow-y-auto border-l border-border/60 bg-[var(--imessage-chrome)] px-4 py-4 lg:block">
        <EvidencePanel citations={citations} />
      </aside>
    </div>
  );
}
