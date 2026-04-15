"use client";

import { Sidebar } from "@/components/sidebar/Sidebar";
import { DocumentPreviewModal } from "@/components/sidebar/DocumentPreviewModal";
import { ContextHeader } from "@/components/header/ContextHeader";
import { ChatLayout } from "@/components/chat/ChatLayout";
import { InputBar } from "@/components/input/InputBar";

export default function Page() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        <ContextHeader />
        <div className="flex-1 overflow-hidden">
          <ChatLayout />
        </div>
        <InputBar />
      </main>
      <DocumentPreviewModal />
    </div>
  );
}
