"use client";

import { useEffect, useRef } from "react";
import { useStore } from "@/store/useStore";
import { MessageList } from "./MessageList";
import { EvidencePanel } from "./EvidencePanel";

export function ChatLayout() {
  const messages = useStore((s) => s.messages);
  const isLoading = useStore((s) => s.isLoading);
  const latestAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const citations = latestAssistant?.citations ?? [];

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom < 120) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [messages.length, isLoading]);

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
