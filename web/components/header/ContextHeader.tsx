"use client";

import { useStore } from "@/store/useStore";
import { QuickFactsCard } from "./QuickFactsCard";
import { Badge } from "@/components/ui/badge";

export function ContextHeader() {
  const pinnedDocs = useStore((s) => s.pinnedDocs);
  const activeCommand = useStore((s) => s.activeCommand);

  const title =
    pinnedDocs.length === 0
      ? "BidMate"
      : pinnedDocs.length === 1
      ? pinnedDocs[0].title
      : `${pinnedDocs[0].title} 외 ${pinnedDocs.length - 1}건`;

  return (
    <div className="border-b border-border/60 bg-[var(--imessage-chrome)]/95 backdrop-blur-md">
      <div className="flex flex-col items-center px-4 pt-3 pb-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--imessage-blue)] text-[13px] font-semibold text-white">
          BM
        </div>
        <div className="mt-1 max-w-[80%] truncate text-[13px] font-semibold text-foreground">
          {title}
        </div>
        {activeCommand && (
          <div className="mt-1">
            <Badge variant="default" className="text-[10px]">
              {activeCommand.label}
            </Badge>
          </div>
        )}
      </div>
      {pinnedDocs.length === 1 && (
        <div className="px-4 pb-3">
          <QuickFactsCard docId={pinnedDocs[0].id} />
        </div>
      )}
      {pinnedDocs.length > 1 && (
        <div className="flex flex-wrap justify-center gap-1.5 px-4 pb-3">
          {pinnedDocs.map((d) => (
            <Badge key={d.id} variant="secondary" className="text-[10px]">
              {d.title}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
