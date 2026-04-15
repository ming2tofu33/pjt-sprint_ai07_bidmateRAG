"use client";

import { useEffect, useState } from "react";
import {
  X,
  ExternalLink,
  MessageSquare,
  Building2,
  Wallet,
  Calendar,
  Tag,
  FileText,
} from "lucide-react";
import type { DocumentDetail } from "@/lib/types";
import { getDocument } from "@/lib/api";
import { useStore } from "@/store/useStore";

export function DocumentPreviewModal() {
  const previewDocId = useStore((s) => s.previewDocId);
  const closePreview = useStore((s) => s.closePreview);
  const pinDoc = useStore((s) => s.pinDoc);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const requestInputFocus = useStore((s) => s.requestInputFocus);
  const cachedDoc = useStore((s) =>
    s.previewDocId ? s.documents.find((d) => d.id === s.previewDocId) : undefined
  );

  const [detail, setDetail] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!previewDocId) {
      setDetail(null);
      setFetchError(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setFetchError(null);
    setDetail(null);
    getDocument(previewDocId)
      .then((data) => {
        if (cancelled) return;
        setDetail(data);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setFetchError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [previewDocId]);

  useEffect(() => {
    if (!previewDocId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closePreview();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [previewDocId, closePreview]);

  if (!previewDocId) return null;

  const displayDoc = detail ?? cachedDoc;
  const pdfUrl = `/api/documents/${encodeURIComponent(previewDocId)}/pdf`;

  const handleStartChat = () => {
    const target = detail ?? cachedDoc;
    if (!target) return;
    pinDoc(target);
    closePreview();
    setActiveTab("chat");
    requestInputFocus();
  };

  return (
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
      onClick={closePreview}
      role="dialog"
      aria-modal="true"
      aria-label="문서 미리보기"
    >
      <div
        className="flex h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-border bg-background shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between gap-3 border-b border-border bg-[var(--imessage-chrome)]/90 px-5 py-3 backdrop-blur-md">
          <div className="min-w-0 flex-1">
            <div className="truncate text-[15px] font-semibold">
              {displayDoc?.title ?? "문서 미리보기"}
            </div>
            <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground">
              {displayDoc && (
                <>
                  <span className="truncate">{displayDoc.agency}</span>
                  <span>·</span>
                  <span>{displayDoc.budget_label}</span>
                  {displayDoc.domain && (
                    <>
                      <span>·</span>
                      <span>{displayDoc.domain}</span>
                    </>
                  )}
                  <span>·</span>
                </>
              )}
              <span className="whitespace-nowrap opacity-70">
                💡 PDF 영역 클릭 후{" "}
                <kbd className="rounded border border-border bg-muted px-1 py-px text-[10px] font-mono">
                  ⌘F
                </kbd>
                {" / "}
                <kbd className="rounded border border-border bg-muted px-1 py-px text-[10px] font-mono">
                  Ctrl+F
                </kbd>
                로 본문 검색
              </span>
            </div>
          </div>
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            title="새 탭에서 열기"
            aria-label="새 탭에서 열기"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <ExternalLink className="size-4" />
          </a>
          <button
            type="button"
            onClick={closePreview}
            aria-label="닫기"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <X className="size-4" />
          </button>
        </div>

        {/* Body: 2-column grid */}
        <div className="grid flex-1 grid-cols-[minmax(260px,30%)_1fr] overflow-hidden">
          {/* Left: metadata panel */}
          <aside className="flex flex-col overflow-y-auto border-r border-border bg-muted/30 px-5 py-4">
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              문서 정보
            </div>

            {loading && !cachedDoc && (
              <div className="space-y-2">
                <div className="h-4 w-24 rounded bg-muted" />
                <div className="h-4 w-32 rounded bg-muted" />
                <div className="h-4 w-20 rounded bg-muted" />
              </div>
            )}

            {fetchError && (
              <div className="rounded border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
                불러오기 실패: {fetchError}
              </div>
            )}

            {(detail?.quick_facts ?? []).length > 0 && (
              <dl className="space-y-3">
                {detail!.quick_facts.map((fact) => {
                  const Icon = iconForLabel(fact.label);
                  return (
                    <div key={fact.label}>
                      <dt className="flex items-center gap-1 text-[11px] text-muted-foreground">
                        <Icon className="size-3" aria-hidden="true" />
                        {fact.label}
                      </dt>
                      <dd className="mt-0.5 text-sm font-medium text-foreground">
                        {fact.value || "-"}
                      </dd>
                    </div>
                  );
                })}
              </dl>
            )}

            {detail?.summary_oneline && (
              <>
                <div className="my-4 border-t border-border/60" />
                <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                  사업 요약
                </div>
                <p className="mt-1.5 whitespace-pre-wrap text-[13px] leading-relaxed text-foreground/90">
                  {detail.summary_oneline}
                </p>
              </>
            )}

            <div className="mt-auto pt-4">
              <button
                type="button"
                onClick={handleStartChat}
                disabled={!displayDoc}
                className="flex w-full items-center justify-center gap-2 rounded-full bg-[var(--imessage-blue)] px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-all duration-150 ease-out hover:opacity-90 active:scale-[0.98] disabled:opacity-40 disabled:active:scale-100"
              >
                <MessageSquare className="size-4" />
                이 문서로 질문 시작
              </button>
              <p className="mt-1.5 text-center text-[10px] text-muted-foreground">
                클릭하면 채팅 탭으로 이동하며 이 문서가 자동으로 첨부됩니다
              </p>
            </div>
          </aside>

          {/* Right: PDF iframe */}
          <iframe
            src={pdfUrl}
            title={displayDoc?.title ?? "문서 미리보기"}
            className="h-full w-full border-0 bg-muted"
          />
        </div>
      </div>
    </div>
  );
}

function iconForLabel(label: string) {
  switch (label) {
    case "발주기관":
      return Building2;
    case "사업금액":
      return Wallet;
    case "마감일":
      return Calendar;
    case "도메인":
      return Tag;
    case "문서크기":
      return FileText;
    default:
      return FileText;
  }
}
