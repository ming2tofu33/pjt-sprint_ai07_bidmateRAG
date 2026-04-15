"use client";

import { ChevronLeft, ChevronRight, MessageSquare } from "lucide-react";
import { useStore } from "@/store/useStore";
import { TabBar } from "./TabBar";
import { ChatTab } from "./ChatTab";
import { DocumentsTab } from "./DocumentsTab";
import { useCmdBShortcut, useCmdKShortcut } from "@/lib/keyboard";
import { Button } from "@/components/ui/button";

export function Sidebar() {
  const collapsed = useStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useStore((s) => s.toggleSidebar);
  const activeTab = useStore((s) => s.activeTab);
  useCmdBShortcut();
  useCmdKShortcut();

  if (collapsed) {
    return (
      <aside className="w-12 border-r border-border">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="mt-2 w-full"
          onClick={toggleSidebar}
          title="사이드바 펼치기 (⌘B)"
          aria-label="사이드바 펼치기"
        >
          <ChevronRight className="size-4" />
        </Button>
      </aside>
    );
  }

  return (
    <aside className="flex w-80 flex-col border-r border-border">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <div className="flex items-center gap-2 text-sm font-bold">
          <MessageSquare className="size-4 text-[var(--imessage-blue)]" />
          BidMate
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          title="사이드바 접기 (⌘B)"
          aria-label="사이드바 접기"
        >
          <ChevronLeft className="size-4" />
        </Button>
      </div>
      <TabBar />
      <div className="flex-1 overflow-hidden">
        {activeTab === "chat" ? <ChatTab /> : <DocumentsTab />}
      </div>
    </aside>
  );
}
