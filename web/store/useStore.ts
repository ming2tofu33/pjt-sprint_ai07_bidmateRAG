import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type {
  DocumentSummary,
  SlashCommandMeta,
  Message,
  QueryMetadata,
} from "@/lib/types";
import { postQueryStream } from "@/lib/api";

function newId(prefix: string): string {
  // crypto.randomUUID는 secure context(https/localhost)에서만 동작.
  // fallback으로 타임스탬프 + random을 사용.
  try {
    return `${prefix}-${crypto.randomUUID()}`;
  } catch {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

interface DocumentFilters {
  domain?: string;
  agencyType?: string;
  budgetRange?: string;
}

interface Store {
  sidebarCollapsed: boolean;
  activeTab: "chat" | "documents";

  documents: DocumentSummary[];
  documentSearchQuery: string;
  documentFilters: DocumentFilters;

  pinnedDocs: DocumentSummary[];
  activeCommand: SlashCommandMeta | null;

  previewDocId: string | null;

  catalogOpen: boolean;

  searchFocusToken: number;
  inputFocusToken: number;

  messages: Message[];
  isLoading: boolean;
  lastError: string | null;

  setSidebarCollapsed: (v: boolean) => void;
  toggleSidebar: () => void;
  setActiveTab: (tab: "chat" | "documents") => void;

  setDocuments: (docs: DocumentSummary[]) => void;
  setDocumentSearchQuery: (q: string) => void;
  setDocumentFilters: (f: DocumentFilters) => void;

  pinDoc: (doc: DocumentSummary) => void;
  unpinDoc: (docId: string) => void;
  setCommand: (cmd: SlashCommandMeta | null) => void;
  clearContext: () => void;

  openPreview: (docId: string) => void;
  closePreview: () => void;

  openCatalog: () => void;
  closeCatalog: () => void;

  requestSearchFocus: () => void;
  requestInputFocus: () => void;

  newChat: () => void;
  sendMessage: (text: string) => Promise<void>;
}

export const useStore = create<Store>()(
  persist(
    (set, get) => ({
      sidebarCollapsed: false,
      activeTab: "documents",

      documents: [],
      documentSearchQuery: "",
      documentFilters: {},

      pinnedDocs: [],
      activeCommand: null,

      previewDocId: null,

      catalogOpen: false,

      searchFocusToken: 0,
      inputFocusToken: 0,

      messages: [],
      isLoading: false,
      lastError: null,

      setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),
      toggleSidebar: () =>
        set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setActiveTab: (tab) => set({ activeTab: tab }),

      setDocuments: (docs) => set({ documents: docs }),
      setDocumentSearchQuery: (q) => set({ documentSearchQuery: q }),
      setDocumentFilters: (f) => set({ documentFilters: f }),

      pinDoc: (doc) => {
        const existing = get().pinnedDocs;
        if (existing.some((d) => d.id === doc.id)) return;
        set({ pinnedDocs: [...existing, doc] });
      },
      unpinDoc: (docId) =>
        set((s) => ({
          pinnedDocs: s.pinnedDocs.filter((d) => d.id !== docId),
        })),
      setCommand: (cmd) => set({ activeCommand: cmd }),
      clearContext: () => set({ pinnedDocs: [], activeCommand: null }),

      openPreview: (docId) => set({ previewDocId: docId }),
      closePreview: () => set({ previewDocId: null }),

      openCatalog: () => set({ catalogOpen: true }),
      closeCatalog: () => set({ catalogOpen: false }),

      requestSearchFocus: () =>
        set((s) => ({ searchFocusToken: s.searchFocusToken + 1 })),

      requestInputFocus: () =>
        set((s) => ({ inputFocusToken: s.inputFocusToken + 1 })),

      newChat: () =>
        set({
          messages: [],
          lastError: null,
          pinnedDocs: [],
          activeCommand: null,
        }),

      sendMessage: async (text) => {
        const state = get();
        const userMessage: Message = {
          id: newId("user"),
          role: "user",
          content: text,
          createdAt: Date.now(),
        };
        const assistantId = newId("assistant");
        const assistantMessage: Message = {
          id: assistantId,
          role: "assistant",
          content: "",
          createdAt: Date.now(),
          citations: [],
        };
        set({
          messages: [...state.messages, userMessage, assistantMessage],
          isLoading: true,
          lastError: null,
        });

        // eslint-disable-next-line prefer-const
        let finalMetadata = null as QueryMetadata | null;

        const updateAssistant = (patch: Partial<Message>) =>
          set((s) => ({
            messages: s.messages.map((m) =>
              m.id === assistantId ? { ...m, ...patch } : m
            ),
          }));

        const appendDelta = (delta: string) =>
          set((s) => ({
            messages: s.messages.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + delta } : m
            ),
          }));

        try {
          await postQueryStream(
            {
              question: text,
              mentioned_doc_ids: state.pinnedDocs.map((d) => d.id),
              command: state.activeCommand?.id ?? null,
            },
            (event) => {
              switch (event.type) {
                case "retrieval":
                  updateAssistant({ citations: event.citations });
                  break;
                case "token":
                  appendDelta(event.delta);
                  break;
                case "done":
                  finalMetadata = event.metadata;
                  updateAssistant({ metadata: event.metadata });
                  break;
                case "error":
                  updateAssistant({
                    content: `오류: ${event.message}`,
                    error: event.message,
                  });
                  set({ lastError: event.message });
                  break;
              }
            }
          );

          set({ isLoading: false });

          if (finalMetadata?.command_applied === "초기화") {
            set({ pinnedDocs: [], activeCommand: null });
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          updateAssistant({
            content: `오류: ${message}`,
            error: message,
          });
          set({ isLoading: false, lastError: message });
        }
      },
    }),
    {
      name: "bidmate-session",
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        messages: state.messages,
        pinnedDocs: state.pinnedDocs,
        activeCommand: state.activeCommand,
        sidebarCollapsed: state.sidebarCollapsed,
        activeTab: state.activeTab,
      }),
    }
  )
);
