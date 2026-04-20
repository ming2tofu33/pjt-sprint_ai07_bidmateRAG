"use client";

import { Check, Plus } from "lucide-react";
import type { DocumentSummary } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { useStore } from "@/store/useStore";
import { cn } from "@/lib/utils";

interface Props {
  doc: DocumentSummary;
}

export function DocumentCard({ doc }: Props) {
  const pinDoc = useStore((s) => s.pinDoc);
  const unpinDoc = useStore((s) => s.unpinDoc);
  const pinnedDocs = useStore((s) => s.pinnedDocs);
  const openPreview = useStore((s) => s.openPreview);
  const isPinned = pinnedDocs.some((d) => d.id === doc.id);

  return (
    <div
      className={cn(
        "relative rounded-md border border-border bg-card transition-colors hover:bg-accent",
        isPinned && "border-[var(--imessage-blue)] bg-accent"
      )}
    >
      <button
        type="button"
        onClick={() => openPreview(doc.id)}
        title="클릭하여 문서 보기"
        className="w-full rounded-md p-3 pr-10 text-left"
      >
        <div className="truncate text-sm font-medium">{doc.title}</div>
        <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
          <span className="truncate">{doc.agency}</span>
          <span>·</span>
          <span>{doc.budget_label}</span>
        </div>
        <div className="mt-2 flex gap-1">
          {doc.domain && (
            <Badge variant="secondary" className="text-[10px]">
              {doc.domain}
            </Badge>
          )}
          {doc.agency_type && (
            <Badge variant="outline" className="text-[10px]">
              {doc.agency_type}
            </Badge>
          )}
        </div>
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          if (isPinned) {
            unpinDoc(doc.id);
          } else {
            pinDoc(doc);
          }
        }}
        title={isPinned ? "추가됨 · 클릭하여 해제" : "문서 추가 (멘션)"}
        aria-label={isPinned ? `${doc.title} 추가 해제` : `${doc.title} 추가`}
        aria-pressed={isPinned}
        className={cn(
          "absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-md transition-colors",
          isPinned
            ? "bg-[var(--imessage-blue)] text-white hover:bg-[var(--imessage-blue-dark)]"
            : "text-muted-foreground hover:bg-muted hover:text-[var(--imessage-blue)]"
        )}
      >
        {isPinned ? <Check className="size-4" /> : <Plus className="size-4" />}
      </button>
    </div>
  );
}
