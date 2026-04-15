"use client";

import { useEffect } from "react";
import { useStore } from "@/store/useStore";

/** ⌘K / Ctrl+K 단축키: 사이드바 → 보관 문서 탭 → 검색창 포커스 요청
 *
 * 항상 마운트된 컴포넌트(Sidebar)에서 호출해야 한다. DocumentsTab이
 * 언마운트된 채팅 탭에서도 동작하려면 이 훅이 탭과 독립적으로 살아 있어야
 * 한다. 실제 포커스는 Store 의 `searchFocusToken` 변화를 DocumentsTab 이
 * 감지해서 처리한다. */
export function useCmdKShortcut(): void {
  const setSidebarCollapsed = useStore((s) => s.setSidebarCollapsed);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const requestSearchFocus = useStore((s) => s.requestSearchFocus);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSidebarCollapsed(false);
        setActiveTab("documents");
        requestSearchFocus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [setSidebarCollapsed, setActiveTab, requestSearchFocus]);
}

/** ⌘B / Ctrl+B 단축키: 사이드바 토글 */
export function useCmdBShortcut(): void {
  const toggleSidebar = useStore((s) => s.toggleSidebar);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        toggleSidebar();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [toggleSidebar]);
}
