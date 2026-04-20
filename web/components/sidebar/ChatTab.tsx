"use client";

import { useMemo, useState } from "react";
import { Plus } from "lucide-react";
import { useStore } from "@/store/useStore";
import { ChatListItem } from "./ChatListItem";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";

export function ChatTab() {
  const chats = useStore((s) => s.chats);
  const currentChatId = useStore((s) => s.currentChatId);
  const newChat = useStore((s) => s.newChat);
  const openChat = useStore((s) => s.openChat);
  const deleteChat = useStore((s) => s.deleteChat);

  // 삭제 확인 다이얼로그: null이면 닫힘, 아니면 해당 id 대화를 지울 예정.
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);

  const sortedChats = useMemo(
    () => [...chats].sort((a, b) => b.updatedAt - a.updatedAt),
    [chats]
  );

  const pendingDeleteChat = pendingDeleteId
    ? chats.find((c) => c.id === pendingDeleteId)
    : null;

  return (
    <div className="flex h-full flex-col">
      <div className="p-3 pb-2">
        <button
          type="button"
          onClick={newChat}
          className="flex w-full items-center justify-center gap-1.5 rounded-md bg-[var(--imessage-blue)] px-3 py-2 text-sm font-semibold text-white transition-all hover:opacity-90 active:scale-[0.98]"
        >
          <Plus className="size-4" />새 대화
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 pb-4">
        {sortedChats.length === 0 ? (
          <div className="mt-8 text-center text-[12px] text-muted-foreground">
            아직 대화가 없습니다.
            <br />첫 메시지를 보내면 여기 표시됩니다.
          </div>
        ) : (
          <>
            <div className="mb-1.5 mt-1 px-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              최근 대화
            </div>
            <div className="flex flex-col gap-0.5">
              {sortedChats.map((chat) => (
                <ChatListItem
                  key={chat.id}
                  chat={chat}
                  active={chat.id === currentChatId}
                  onOpen={() => openChat(chat.id)}
                  onRequestDelete={() => setPendingDeleteId(chat.id)}
                />
              ))}
            </div>
          </>
        )}
      </div>

      <ConfirmDialog
        open={pendingDeleteId !== null}
        title="이 대화를 삭제할까요?"
        description={
          pendingDeleteChat
            ? `"${pendingDeleteChat.title}" 대화가 영구 삭제됩니다. 이 작업은 되돌릴 수 없습니다.`
            : undefined
        }
        confirmLabel="삭제"
        destructive
        onConfirm={() => {
          if (pendingDeleteId) deleteChat(pendingDeleteId);
          setPendingDeleteId(null);
        }}
        onCancel={() => setPendingDeleteId(null)}
      />
    </div>
  );
}
