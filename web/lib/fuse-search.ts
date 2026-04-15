import Fuse from "fuse.js";
import type { DocumentSummary } from "./types";

export function searchDocuments(
  documents: DocumentSummary[],
  query: string
): DocumentSummary[] {
  if (!query.trim()) return documents;
  const fuse = new Fuse(documents, {
    keys: ["title", "agency", "domain"],
    threshold: 0.35,
    ignoreLocation: true,
  });
  return fuse.search(query).map((r) => r.item);
}

export function filterDocuments(
  documents: DocumentSummary[],
  filters: {
    domain?: string;
    agencyType?: string;
    budgetRange?: string;
  }
): DocumentSummary[] {
  return documents.filter((d) => {
    if (filters.domain && d.domain !== filters.domain) return false;
    if (filters.agencyType && d.agency_type !== filters.agencyType) return false;
    if (filters.budgetRange) {
      const b = d.budget;
      switch (filters.budgetRange) {
        case "under_1":
          if (b >= 100_000_000) return false;
          break;
        case "1_5":
          if (b < 100_000_000 || b > 500_000_000) return false;
          break;
        case "5_10":
          if (b < 500_000_000 || b > 1_000_000_000) return false;
          break;
        case "over_10":
          if (b < 1_000_000_000) return false;
          break;
      }
    }
    return true;
  });
}
