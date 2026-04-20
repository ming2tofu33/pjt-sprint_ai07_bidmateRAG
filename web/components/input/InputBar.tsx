"use client";

import { useEffect, useRef, useState } from "react";
import { useStore } from "@/store/useStore";
import { ChipTray } from "./ChipTray";
import { QuickCommandBar } from "./QuickCommandBar";
import { MentionTextarea } from "./MentionTextarea";

function stripMentionMarkup(raw: string): string {
  // react-mentions stores markup like `@[display](id)` or `/[display](id)`.
  // On submit we strip all mention/command tokens since they are captured
  // separately in the Zustand store as pinnedDocs / activeCommand.
  return raw
    .replace(/@\[[^\]]+\]\([^)]+\)/g, "")
    .replace(/\/\[[^\]]+\]\([^)]+\)/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function InputBar() {
  const [text, setText] = useState("");

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const inputFocusToken = useStore((s) => s.inputFocusToken);

  useEffect(() => {
    if (inputFocusToken === 0) return;
    const raf1 = requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        textareaRef.current?.focus();
      });
    });
    return () => cancelAnimationFrame(raf1);
  }, [inputFocusToken]);

  const pinnedDocs = useStore((s) => s.pinnedDocs);
  const activeCommand = useStore((s) => s.activeCommand);
  const isLoading = useStore((s) => s.isLoading);
  const sendMessage = useStore((s) => s.sendMessage);
  const abortCurrentRequest = useStore((s) => s.abortCurrentRequest);

  const requiresDoc = activeCommand?.requires_doc && pinnedDocs.length === 0;
  const requiresMulti =
    activeCommand?.requires_multi_doc && pinnedDocs.length < 2;
  // 스트리밍 중에는 버튼이 "중지"로 변신하므로 isLoading은 disabled 조건에서 제외.
  const sendDisabled =
    !isLoading &&
    (!text.trim() || Boolean(requiresDoc) || Boolean(requiresMulti));

  const disabledReason = requiresMulti
    ? `${activeCommand?.label}는 2개 이상 문서 멘션이 필요합니다`
    : requiresDoc
    ? `${activeCommand?.label}는 문서 멘션이 필요합니다`
    : undefined;

  const handleSend = () => {
    if (sendDisabled) return;
    const cleaned = stripMentionMarkup(text);
    if (!cleaned) return;
    sendMessage(cleaned);
    setText("");
  };

  const handleButtonClick = () => {
    if (isLoading) {
      abortCurrentRequest();
      return;
    }
    handleSend();
  };

  return (
    <div className="border-t border-border/60 bg-[var(--imessage-chrome)]/95 pb-[calc(env(safe-area-inset-bottom,0px)+8px)] pt-2 backdrop-blur-md">
      <ChipTray />
      <QuickCommandBar />
      <div className="px-3">
        <div className="imessage-input flex items-end gap-2 py-1.5 pl-4 pr-1.5">
          <div className="flex-1">
            <MentionTextarea
              value={text}
              onChange={setText}
              onEnter={handleSend}
              disabled={isLoading}
              inputRef={textareaRef}
            />
          </div>
          <button
            type="button"
            onClick={handleButtonClick}
            disabled={sendDisabled}
            title={isLoading ? "생성 중지" : disabledReason}
            aria-label={isLoading ? "생성 중지" : "전송"}
            className="relative flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[var(--imessage-blue)] text-white transition-all duration-150 ease-out before:absolute before:-inset-1.5 before:content-[''] active:scale-90 disabled:opacity-30 disabled:active:scale-100"
          >
            {isLoading ? (
              // 정사각형 stop 아이콘
              <svg
                viewBox="0 0 24 24"
                width="12"
                height="12"
                aria-hidden="true"
              >
                <rect
                  x="6"
                  y="6"
                  width="12"
                  height="12"
                  rx="2"
                  fill="currentColor"
                />
              </svg>
            ) : (
              <svg
                viewBox="0 0 24 24"
                width="18"
                height="18"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M12 19V5" />
                <path d="M5 12l7-7 7 7" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
