"use client";

import { Button } from "@/components/ui/button";
import { useStore } from "@/store/useStore";

export function ChatTab() {
  const newChat = useStore((s) => s.newChat);
  return (
    <div className="p-4">
      <Button
        type="button"
        variant="default"
        className="w-full"
        onClick={newChat}
      >
        + 새 대화
      </Button>
      <p className="mt-4 text-xs text-muted-foreground">
        대화 히스토리는 탭을 닫으면 사라집니다 (F5는 유지).
      </p>
    </div>
  );
}
