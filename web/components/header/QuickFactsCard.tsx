"use client";

import { useEffect, useState } from "react";
import type { DocumentDetail } from "@/lib/types";
import { getDocument } from "@/lib/api";

export function QuickFactsCard({ docId }: { docId: string }) {
  const [detail, setDetail] = useState<DocumentDetail | null>(null);

  useEffect(() => {
    setDetail(null);
    getDocument(docId)
      .then(setDetail)
      .catch((err) => console.error("quick facts load failed", err));
  }, [docId]);

  if (!detail) {
    return (
      <div className="rounded-md border border-border bg-card px-4 py-3 text-xs text-muted-foreground">
        문서 정보 로딩 중...
      </div>
    );
  }

  return (
    <div className="rounded-md border border-border bg-card px-4 py-3">
      <div className="mb-2 truncate text-sm font-semibold">{detail.title}</div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {detail.quick_facts.map((f) => (
          <div key={f.label} className="flex gap-1">
            <span className="text-muted-foreground">{f.label}:</span>
            <span className="font-medium">{f.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
