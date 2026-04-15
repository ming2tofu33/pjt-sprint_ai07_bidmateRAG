import { Paperclip } from "lucide-react";
import type { Citation } from "@/lib/types";
import { CitationCard } from "./CitationCard";

export function EvidencePanel({ citations }: { citations: Citation[] }) {
  if (!citations || citations.length === 0) {
    return (
      <div className="sticky top-4 rounded-md border border-dashed border-border p-4 text-xs text-muted-foreground">
        답변이 생성되면 근거 카드가 여기에 표시됩니다.
      </div>
    );
  }
  return (
    <div className="sticky top-4 flex flex-col gap-2">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
        <Paperclip className="size-3.5" />
        근거 ({citations.length})
      </div>
      {citations.map((c) => (
        <CitationCard key={c.id} citation={c} />
      ))}
    </div>
  );
}
