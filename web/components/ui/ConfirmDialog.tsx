"use client";

import { useEffect } from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  open: boolean;
  title: string;
  description?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * 가벼운 공용 확인 다이얼로그. 백드롭 클릭/ESC로 취소, Enter로 확인.
 * `destructive`가 true면 확인 버튼이 빨간색 (삭제 류).
 */
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "확인",
  cancelLabel = "취소",
  destructive = false,
  onConfirm,
  onCancel,
}: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter") {
        e.preventDefault();
        onConfirm();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel, onConfirm]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[1200] flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
      onClick={onCancel}
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="confirm-title"
    >
      <div
        className="w-full max-w-sm overflow-hidden rounded-2xl border border-border bg-background shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 px-5 pt-5">
          {destructive && (
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-destructive/15 text-destructive">
              <AlertTriangle className="size-5" />
            </div>
          )}
          <div className="min-w-0 flex-1">
            <h3 id="confirm-title" className="text-[15px] font-semibold">
              {title}
            </h3>
            {description && (
              <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">
                {description}
              </p>
            )}
          </div>
        </div>
        <div className="mt-5 flex items-center justify-end gap-2 border-t border-border bg-muted/30 px-4 py-3">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md px-3 py-1.5 text-sm text-foreground transition-colors hover:bg-muted"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            autoFocus
            className={
              destructive
                ? "rounded-md bg-destructive px-3 py-1.5 text-sm font-medium text-white transition-opacity hover:opacity-90"
                : "rounded-md bg-[var(--imessage-blue)] px-3 py-1.5 text-sm font-medium text-white transition-opacity hover:opacity-90"
            }
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
