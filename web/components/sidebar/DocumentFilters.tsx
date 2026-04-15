"use client";

import { useStore } from "@/store/useStore";
import type { DocumentSummary } from "@/lib/types";
import { useMemo } from "react";

export function DocumentFilters({
  documents,
}: {
  documents: DocumentSummary[];
}) {
  const filters = useStore((s) => s.documentFilters);
  const setDocumentFilters = useStore((s) => s.setDocumentFilters);

  const domains = useMemo(
    () => Array.from(new Set(documents.map((d) => d.domain).filter(Boolean))).sort(),
    [documents]
  );
  const agencyTypes = useMemo(
    () =>
      Array.from(new Set(documents.map((d) => d.agency_type).filter(Boolean))).sort(),
    [documents]
  );

  const budgetOptions = [
    { value: "", label: "전체 금액" },
    { value: "under_1", label: "1억 이하" },
    { value: "1_5", label: "1~5억" },
    { value: "5_10", label: "5~10억" },
    { value: "over_10", label: "10억 이상" },
  ];

  return (
    <div className="flex flex-col gap-2 px-4 pb-2 text-xs">
      <select
        className="rounded border border-border bg-background px-2 py-1"
        value={filters.domain ?? ""}
        onChange={(e) =>
          setDocumentFilters({ ...filters, domain: e.target.value || undefined })
        }
      >
        <option value="">전체 도메인</option>
        {domains.map((d) => (
          <option key={d} value={d}>{d}</option>
        ))}
      </select>
      <select
        className="rounded border border-border bg-background px-2 py-1"
        value={filters.agencyType ?? ""}
        onChange={(e) =>
          setDocumentFilters({
            ...filters,
            agencyType: e.target.value || undefined,
          })
        }
      >
        <option value="">전체 기관유형</option>
        {agencyTypes.map((t) => (
          <option key={t} value={t}>{t}</option>
        ))}
      </select>
      <select
        className="rounded border border-border bg-background px-2 py-1"
        value={filters.budgetRange ?? ""}
        onChange={(e) =>
          setDocumentFilters({
            ...filters,
            budgetRange: e.target.value || undefined,
          })
        }
      >
        {budgetOptions.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}
