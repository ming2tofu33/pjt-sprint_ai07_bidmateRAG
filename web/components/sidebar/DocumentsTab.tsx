"use client";

import { useEffect, useMemo, useRef } from "react";
import { Search } from "lucide-react";
import { useStore } from "@/store/useStore";
import { listDocuments } from "@/lib/api";
import { searchDocuments, filterDocuments } from "@/lib/fuse-search";
import { DocumentCard } from "./DocumentCard";
import { DocumentFilters } from "./DocumentFilters";
import { Input } from "@/components/ui/input";

export function DocumentsTab() {
  const documents = useStore((s) => s.documents);
  const setDocuments = useStore((s) => s.setDocuments);
  const searchQuery = useStore((s) => s.documentSearchQuery);
  const setSearchQuery = useStore((s) => s.setDocumentSearchQuery);
  const filters = useStore((s) => s.documentFilters);
  const searchFocusToken = useStore((s) => s.searchFocusToken);

  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (documents.length === 0) {
      listDocuments()
        .then((data) => setDocuments(data.documents))
        .catch((err) => console.error("failed to load documents", err));
    }
  }, [documents.length, setDocuments]);

  // ⌘K 단축키: 전역 핸들러(Sidebar)가 searchFocusToken 을 증가시키면
  // DocumentsTab 이 마운트된 직후 이 effect 가 발동해 입력창에 포커스.
  useEffect(() => {
    if (searchFocusToken === 0) return;
    // DocumentsTab 마운트 직후에는 input ref 가 아직 안 붙었을 수 있으므로
    // 다음 paint 뒤로 미룬다. requestAnimationFrame 두 번이 가장 안전.
    const raf1 = requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
      });
    });
    return () => cancelAnimationFrame(raf1);
  }, [searchFocusToken]);

  const visible = useMemo(() => {
    const filtered = filterDocuments(documents, filters);
    return searchDocuments(filtered, searchQuery);
  }, [documents, filters, searchQuery]);

  return (
    <div className="flex h-full flex-col">
      <div className="p-4 pb-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            ref={searchInputRef}
            type="search"
            placeholder="문서 검색 (⌘K)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8"
          />
        </div>
      </div>
      <DocumentFilters documents={documents} />
      <div className="mt-2 flex-1 overflow-y-auto px-4 pb-4">
        <div className="mb-2 text-xs text-muted-foreground">
          {visible.length}건 / 총 {documents.length}건
        </div>
        <div className="flex flex-col gap-2">
          {visible.map((doc) => (
            <DocumentCard key={doc.id} doc={doc} />
          ))}
        </div>
      </div>
    </div>
  );
}
