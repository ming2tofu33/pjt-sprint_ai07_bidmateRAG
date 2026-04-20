"use client";

import { FileText } from "lucide-react";
import { useStore } from "@/store/useStore";

export function ChipTray() {
  const pinnedDocs = useStore((s) => s.pinnedDocs);
  const activeCommand = useStore((s) => s.activeCommand);
  const unpinDoc = useStore((s) => s.unpinDoc);
  const setCommand = useStore((s) => s.setCommand);

  if (pinnedDocs.length === 0 && !activeCommand) return null;

  const tooMany = pinnedDocs.length >= 4;

  return (
    <div className="flex flex-wrap gap-2 px-4 pb-2">
      {pinnedDocs.map((doc) => (
        <span
          key={doc.id}
          className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-xs ${
            tooMany
              ? "border-red-400 bg-red-50 text-red-700"
              : "border-primary/40 bg-primary/10 text-primary"
          }`}
          title={tooMany ? "4개 이상 멘션 시 답변 품질이 떨어질 수 있어요" : undefined}
        >
          <FileText className="size-3.5" />
          {doc.title.slice(0, 20)}
          {doc.title.length > 20 ? "..." : ""}
          <button
            type="button"
            onClick={() => unpinDoc(doc.id)}
            className="ml-1 opacity-60 hover:opacity-100"
            aria-label="멘션 해제"
          >
            ×
          </button>
        </span>
      ))}
      {activeCommand && (
        <span className="inline-flex items-center gap-1 rounded-full border border-amber-400 bg-amber-50 px-2 py-1 text-xs text-amber-800">
          {activeCommand.icon} {activeCommand.label}
          <button
            type="button"
            onClick={() => setCommand(null)}
            className="ml-1 opacity-60 hover:opacity-100"
            aria-label="커맨드 해제"
          >
            ×
          </button>
        </span>
      )}
    </div>
  );
}
