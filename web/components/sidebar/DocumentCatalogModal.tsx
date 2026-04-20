"use client";

import { useEffect, useMemo, useState } from "react";
import {
  X,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Eye,
  MessageSquare,
  LayoutGrid,
} from "lucide-react";
import type { DocumentSummary, SlashCommandMeta } from "@/lib/types";
import { useStore } from "@/store/useStore";
import { searchDocuments, filterDocuments } from "@/lib/fuse-search";
import { getCachedCommands } from "@/lib/commands";
import { Input } from "@/components/ui/input";

type SortKey =
  | "title"
  | "agency"
  | "budget"
  | "agency_type"
  | "domain"
  | "char_count";

type SortDir = "asc" | "desc";

interface SortRule {
  key: SortKey;
  dir: SortDir;
}

export function DocumentCatalogModal() {
  const catalogOpen = useStore((s) => s.catalogOpen);
  const closeCatalog = useStore((s) => s.closeCatalog);
  const documents = useStore((s) => s.documents);
  const searchQuery = useStore((s) => s.documentSearchQuery);
  const setSearchQuery = useStore((s) => s.setDocumentSearchQuery);
  const filters = useStore((s) => s.documentFilters);
  const setFilters = useStore((s) => s.setDocumentFilters);
  const pinDoc = useStore((s) => s.pinDoc);
  const setCommand = useStore((s) => s.setCommand);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const requestInputFocus = useStore((s) => s.requestInputFocus);
  const openPreview = useStore((s) => s.openPreview);

  const [sortRules, setSortRules] = useState<SortRule[]>([
    { key: "title", dir: "asc" },
  ]);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!catalogOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeCatalog();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [catalogOpen, closeCatalog]);

  // 모달이 닫히면 선택 초기화
  useEffect(() => {
    if (!catalogOpen) setSelected(new Set());
  }, [catalogOpen]);

  const visible = useMemo(() => {
    const filtered = filterDocuments(documents, filters);
    const searched = searchDocuments(filtered, searchQuery);
    const sorted = [...searched].sort((a, b) => {
      for (const rule of sortRules) {
        const dir = rule.dir === "asc" ? 1 : -1;
        let cmp = 0;
        switch (rule.key) {
          case "budget":
            cmp = (a.budget - b.budget) * dir;
            break;
          case "char_count":
            cmp = (a.char_count - b.char_count) * dir;
            break;
          default:
            cmp = a[rule.key].localeCompare(b[rule.key], "ko") * dir;
        }
        if (cmp !== 0) return cmp;
      }
      return 0;
    });
    return sorted;
  }, [documents, filters, searchQuery, sortRules]);

  const toggleSort = (key: SortKey, shiftKey: boolean) => {
    setSortRules((rules) => {
      const existingIdx = rules.findIndex((r) => r.key === key);

      if (!shiftKey) {
        // 단일 정렬: 같은 키면 방향 토글, 다른 키면 교체
        if (existingIdx === 0 && rules.length === 1) {
          return [
            { key, dir: rules[0].dir === "asc" ? "desc" : "asc" },
          ];
        }
        return [{ key, dir: "asc" }];
      }

      // Shift+Click: 보조 정렬 추가 / 방향 토글
      if (existingIdx >= 0) {
        // 이미 룰에 있으면 방향만 토글
        const next = [...rules];
        next[existingIdx] = {
          key,
          dir: next[existingIdx].dir === "asc" ? "desc" : "asc",
        };
        return next;
      }
      // 새로 추가
      return [...rules, { key, dir: "asc" }];
    });
  };

  const toggleSelect = (docId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) {
        next.delete(docId);
      } else {
        next.add(docId);
      }
      return next;
    });
  };

  const selectAll = () => {
    setSelected(new Set(visible.map((d) => d.id)));
  };

  const clearSelection = () => {
    setSelected(new Set());
  };

  const handleStartCompare = () => {
    if (selected.size === 0) return;
    const selectedDocs: DocumentSummary[] = visible.filter((d) =>
      selected.has(d.id),
    );
    // 모든 선택 문서를 핀
    for (const doc of selectedDocs) {
      pinDoc(doc);
    }
    // 2개 이상 선택했으면 /비교 커맨드 자동 활성화
    if (selectedDocs.length >= 2) {
      const compareCmd: SlashCommandMeta | undefined = getCachedCommands().find(
        (c) => c.id === "비교",
      );
      if (compareCmd) {
        setCommand(compareCmd);
      }
    }
    closeCatalog();
    setActiveTab("chat");
    requestInputFocus();
  };

  if (!catalogOpen) return null;

  const domains = Array.from(
    new Set(documents.map((d) => d.domain).filter(Boolean)),
  ).sort();
  const agencyTypes = Array.from(
    new Set(documents.map((d) => d.agency_type).filter(Boolean)),
  ).sort();

  return (
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
      onClick={closeCatalog}
      role="dialog"
      aria-modal="true"
      aria-label="문서 카탈로그"
    >
      <div
        className="flex h-[92vh] w-full max-w-7xl flex-col overflow-hidden rounded-2xl border border-border bg-background shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between gap-3 border-b border-border bg-[var(--imessage-chrome)]/90 px-5 py-3 backdrop-blur-md">
          <div className="flex items-center gap-2">
            <LayoutGrid className="size-5 text-[var(--imessage-blue)]" />
            <div className="text-[15px] font-semibold">RFP 문서 카탈로그</div>
            <div className="text-[11px] text-muted-foreground">
              총 {documents.length}건 · 필터 후 {visible.length}건
            </div>
          </div>
          <button
            type="button"
            onClick={closeCatalog}
            aria-label="닫기"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <X className="size-4" />
          </button>
        </div>

        {/* Search + Filters */}
        <div className="flex flex-wrap items-center gap-2 border-b border-border bg-muted/30 px-5 py-3">
          <Input
            type="search"
            placeholder="사업명 또는 발주기관 키워드..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="max-w-md"
          />
          <select
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={filters.domain ?? ""}
            onChange={(e) =>
              setFilters({ ...filters, domain: e.target.value || undefined })
            }
          >
            <option value="">도메인 전체</option>
            {domains.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
          <select
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={filters.agencyType ?? ""}
            onChange={(e) =>
              setFilters({
                ...filters,
                agencyType: e.target.value || undefined,
              })
            }
          >
            <option value="">기관유형 전체</option>
            {agencyTypes.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <select
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={filters.budgetRange ?? ""}
            onChange={(e) =>
              setFilters({
                ...filters,
                budgetRange: e.target.value || undefined,
              })
            }
          >
            <option value="">금액 전체</option>
            <option value="under_1">1억 이하</option>
            <option value="1_5">1~5억</option>
            <option value="5_10">5~10억</option>
            <option value="over_10">10억 이상</option>
          </select>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto">
          <table className="w-full border-collapse text-sm">
            <thead className="sticky top-0 z-10 bg-muted text-left text-[11px] uppercase tracking-wider text-muted-foreground">
              <tr>
                <th className="w-10 border-b border-border px-3 py-2">
                  <input
                    type="checkbox"
                    aria-label="전체 선택"
                    checked={
                      visible.length > 0 && selected.size === visible.length
                    }
                    onChange={(e) =>
                      e.target.checked ? selectAll() : clearSelection()
                    }
                  />
                </th>
                <SortableTh
                  label="사업명"
                  sortKey="title"
                  rules={sortRules}
                  onToggle={toggleSort}
                />
                <SortableTh
                  label="발주기관"
                  sortKey="agency"
                  rules={sortRules}
                  onToggle={toggleSort}
                />
                <SortableTh
                  label="사업금액"
                  sortKey="budget"
                  rules={sortRules}
                  onToggle={toggleSort}
                  align="right"
                />
                <SortableTh
                  label="기관유형"
                  sortKey="agency_type"
                  rules={sortRules}
                  onToggle={toggleSort}
                />
                <SortableTh
                  label="사업도메인"
                  sortKey="domain"
                  rules={sortRules}
                  onToggle={toggleSort}
                />
                <SortableTh
                  label="문서크기"
                  sortKey="char_count"
                  rules={sortRules}
                  onToggle={toggleSort}
                  align="right"
                />
                <th className="w-14 border-b border-border px-3 py-2" />
              </tr>
            </thead>
            <tbody>
              {visible.length === 0 && (
                <tr>
                  <td
                    colSpan={8}
                    className="px-3 py-12 text-center text-sm text-muted-foreground"
                  >
                    검색 조건과 일치하는 문서가 없습니다.
                  </td>
                </tr>
              )}
              {visible.map((doc) => {
                const isSelected = selected.has(doc.id);
                return (
                  <tr
                    key={doc.id}
                    onClick={() => toggleSelect(doc.id)}
                    className={`cursor-pointer border-b border-border/60 transition-colors ${
                      isSelected
                        ? "bg-[var(--imessage-blue)]/10"
                        : "hover:bg-muted/50"
                    }`}
                  >
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        aria-label={`${doc.title} 선택`}
                        checked={isSelected}
                        onChange={() => toggleSelect(doc.id)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </td>
                    <td className="max-w-[320px] truncate px-3 py-2 font-medium">
                      {doc.title}
                    </td>
                    <td className="max-w-[160px] truncate px-3 py-2 text-muted-foreground">
                      {doc.agency}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">
                      {doc.budget_label}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {doc.agency_type}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {doc.domain}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                      {doc.char_count.toLocaleString()}자
                    </td>
                    <td className="px-3 py-2">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          openPreview(doc.id);
                        }}
                        aria-label={`${doc.title} 미리보기`}
                        title="PDF 미리보기"
                        className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                      >
                        <Eye className="size-4" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Selection bar + CTA */}
        <div className="flex items-center justify-between gap-3 border-t border-border bg-[var(--imessage-chrome)]/90 px-5 py-3 backdrop-blur-md">
          <div className="text-[13px] text-muted-foreground">
            {selected.size === 0
              ? "체크박스로 문서를 선택하세요"
              : `${selected.size}건 선택됨${
                  selected.size >= 2 ? " · /비교 커맨드가 자동 적용됩니다" : ""
                }`}
          </div>
          <div className="flex items-center gap-2">
            {selected.size > 0 && (
              <button
                type="button"
                onClick={clearSelection}
                className="text-[12px] text-muted-foreground hover:text-foreground"
              >
                선택 해제
              </button>
            )}
            <button
              type="button"
              onClick={handleStartCompare}
              disabled={selected.size === 0}
              className="flex items-center gap-2 rounded-full bg-[var(--imessage-blue)] px-4 py-2 text-sm font-semibold text-white shadow-sm transition-all duration-150 ease-out hover:opacity-90 active:scale-[0.98] disabled:opacity-40 disabled:active:scale-100"
            >
              <MessageSquare className="size-4" />
              {selected.size >= 2
                ? "비교 시작"
                : selected.size === 1
                  ? "이 문서로 질문 시작"
                  : "선택 후 질문 시작"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

interface SortableThProps {
  label: string;
  sortKey: SortKey;
  rules: SortRule[];
  onToggle: (key: SortKey, shiftKey: boolean) => void;
  align?: "left" | "right";
}

function SortableTh({
  label,
  sortKey,
  rules,
  onToggle,
  align = "left",
}: SortableThProps) {
  const ruleIdx = rules.findIndex((r) => r.key === sortKey);
  const active = ruleIdx >= 0;
  const rule = active ? rules[ruleIdx] : null;
  const Icon = !active
    ? ArrowUpDown
    : rule!.dir === "asc"
      ? ArrowUp
      : ArrowDown;

  // 정렬 규칙이 여러 개일 때만 우선순위 번호 표시
  const showBadge = active && rules.length > 1;

  return (
    <th
      onClick={(e) => onToggle(sortKey, e.shiftKey)}
      title="클릭 = 단일 정렬 · Shift+클릭 = 보조 정렬 추가"
      className={`cursor-pointer select-none border-b border-border px-3 py-2 font-semibold transition-colors hover:bg-muted/70 ${
        align === "right" ? "text-right" : "text-left"
      }`}
    >
      <span
        className={`inline-flex items-center gap-1 ${
          align === "right" ? "flex-row-reverse" : ""
        }`}
      >
        {label}
        <span className="inline-flex items-center gap-0.5">
          <Icon
            className={`size-3 ${
              active ? "text-[var(--imessage-blue)]" : "text-muted-foreground/50"
            }`}
          />
          {showBadge && (
            <span className="inline-flex h-3.5 min-w-[14px] items-center justify-center rounded-full bg-[var(--imessage-blue)] px-1 text-[9px] font-bold leading-none text-white">
              {ruleIdx + 1}
            </span>
          )}
        </span>
      </span>
    </th>
  );
}
