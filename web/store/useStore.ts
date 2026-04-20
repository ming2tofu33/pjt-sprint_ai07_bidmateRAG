import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type {
  Chat,
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

/** 첫 유저 메시지로 대화 제목을 생성. 없으면 "새 대화". */
function autoTitle(messages: Message[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser || !firstUser.content.trim()) return "새 대화";
  const text = firstUser.content.trim().replace(/\s+/g, " ");
  return text.length > 40 ? `${text.slice(0, 40)}…` : text;
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

  // ---- 대화 히스토리 ----
  // 모든 저장된 대화. 활성 대화도 여기 포함되며, `currentChatId`로 식별된다.
  // 최초 진입 시 빈 배열이고, 사용자가 첫 메시지를 보내는 순간 Chat이 생성된다.
  chats: Chat[];
  // 활성 대화 id. null이면 "아직 아무 메시지도 없는 새 대화" 상태 —
  // 첫 메시지 전송 시 자동으로 Chat을 생성하고 이 값을 채운다.
  currentChatId: string | null;

  // ---- 활성 대화의 working copy (top-level에서 편집하고 chats[currentChatId]에 동기화) ----
  pinnedDocs: DocumentSummary[];
  activeCommand: SlashCommandMeta | null;
  messages: Message[];

  // ---- UI state ----
  previewDocId: string | null;
  catalogOpen: boolean;
  searchFocusToken: number;
  inputFocusToken: number;

  // Citation 클릭 시 Evidence 카드를 하이라이트. 토큰은 빠른 연속 클릭에서
  // 이전 setTimeout을 무효화하기 위한 카운터.
  highlightedCitationId: number | null;
  highlightToken: number;

  isLoading: boolean;
  lastError: string | null;

  // 스트리밍 중 취소용. persist 제외 (런타임 전용).
  abortController: AbortController | null;

  // ---- Actions ----
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

  highlightCitation: (id: number) => void;

  // 신규: 대화 히스토리 액션
  newChat: () => void;
  openChat: (id: string) => void;
  deleteChat: (id: string) => void;

  sendMessage: (text: string) => Promise<void>;
  retryLastMessage: () => Promise<void>;
  abortCurrentRequest: () => void;
}

/** 현재 working state를 chats[] 안으로 반영 (또는 신규 생성). */
function syncWorkingToChats(
  chats: Chat[],
  currentChatId: string | null,
  patch: {
    messages?: Message[];
    pinnedDocs?: DocumentSummary[];
    activeCommand?: SlashCommandMeta | null;
  }
): { chats: Chat[]; currentChatId: string } {
  const now = Date.now();
  // 새 chat 생성이 필요한 경우
  if (currentChatId === null) {
    const newChatId = newId("chat");
    const messages = patch.messages ?? [];
    const newChatEntry: Chat = {
      id: newChatId,
      title: autoTitle(messages),
      createdAt: now,
      updatedAt: now,
      messages,
      pinnedDocs: patch.pinnedDocs ?? [],
      activeCommand: patch.activeCommand ?? null,
    };
    return {
      chats: [newChatEntry, ...chats],
      currentChatId: newChatId,
    };
  }
  // 기존 chat 업데이트
  const updatedChats = chats.map((c) => {
    if (c.id !== currentChatId) return c;
    const nextMessages = patch.messages ?? c.messages;
    return {
      ...c,
      messages: nextMessages,
      pinnedDocs: patch.pinnedDocs ?? c.pinnedDocs,
      activeCommand:
        patch.activeCommand !== undefined ? patch.activeCommand : c.activeCommand,
      title: autoTitle(nextMessages),
      updatedAt: now,
    };
  });
  return { chats: updatedChats, currentChatId };
}

function toChatHistory(messages: Message[]): Array<{ role: Message["role"]; content: string }> {
  return messages
    .filter(
      (message) =>
        (message.role === "user" || message.role === "assistant") &&
        message.content.trim().length > 0
    )
    .map((message) => ({ role: message.role, content: message.content }));
}

export const useStore = create<Store>()(
  persist(
    (set, get) => {
      /**
       * 한 번의 RAG 질의를 스트리밍으로 실행하고 지정된 assistant 메시지로
       * 이벤트를 흘려보낸다. `sendMessage`와 `retryLastMessage`가 공유.
       *
       * 호출자가 준비해야 할 것:
       * - `assistantId`를 가진 빈 assistant 메시지가 이미 `messages`에 추가돼 있어야 함.
       * - `isLoading=true`, `abortController`가 store에 설정돼 있어야 함.
       *
       * 이 헬퍼가 처리:
       * - postQueryStream 호출 + event 별 state 업데이트
       * - abort / error / done 분기 처리
       * - 종료 시 isLoading=false, abortController=null로 정리
       */
      const runQuery = async (params: {
        question: string;
        assistantId: string;
        pinnedDocs: DocumentSummary[];
        activeCommand: SlashCommandMeta | null;
        activeChatId: string;
        history: Array<{ role: Message["role"]; content: string }>;
        signal: AbortSignal;
      }) => {
        const {
          question,
          assistantId,
          pinnedDocs,
          activeCommand,
          activeChatId,
          history,
          signal,
        } = params;

        const updateAssistant = (patch: Partial<Message>) =>
          set((s) => {
            const nextMsgs = s.messages.map((m) =>
              m.id === assistantId ? { ...m, ...patch } : m
            );
            const syncedChats = syncWorkingToChats(s.chats, activeChatId, {
              messages: nextMsgs,
            });
            return { messages: nextMsgs, chats: syncedChats.chats };
          });

        const appendDelta = (delta: string) =>
          set((s) => {
            const nextMsgs = s.messages.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + delta } : m
            );
            const syncedChats = syncWorkingToChats(s.chats, activeChatId, {
              messages: nextMsgs,
            });
            return { messages: nextMsgs, chats: syncedChats.chats };
          });

        let finalMetadata = null as QueryMetadata | null;

        try {
          await postQueryStream(
            {
              question,
              mentioned_doc_ids: pinnedDocs.map((d) => d.id),
              history,
              command: activeCommand?.id ?? null,
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
            },
            signal
          );

          if (finalMetadata?.command_applied === "초기화") {
            const cleared = syncWorkingToChats(get().chats, activeChatId, {
              pinnedDocs: [],
              activeCommand: null,
            });
            set({
              pinnedDocs: [],
              activeCommand: null,
              chats: cleared.chats,
            });
          }
        } catch (err) {
          // AbortError: 사용자가 중단 버튼을 눌렀을 때 — 오류 버블 없이 조용히 종료.
          // 현재까지 받은 토큰과 citations는 유지한다.
          const isAbort = err instanceof DOMException && err.name === "AbortError";
          if (!isAbort) {
            const message = err instanceof Error ? err.message : String(err);
            updateAssistant({
              content: `오류: ${message}`,
              error: message,
            });
            set({ lastError: message });
          }
        } finally {
          set({ isLoading: false, abortController: null });
        }
      };

      return {
        sidebarCollapsed: false,
        activeTab: "documents",

        documents: [],
        documentSearchQuery: "",
        documentFilters: {},

        chats: [],
        currentChatId: null,

        pinnedDocs: [],
        activeCommand: null,
        messages: [],

        previewDocId: null,
        catalogOpen: false,

        searchFocusToken: 0,
        inputFocusToken: 0,

        highlightedCitationId: null,
        highlightToken: 0,

        isLoading: false,
        lastError: null,

        abortController: null,

        setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),
        toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
        setActiveTab: (tab) => set({ activeTab: tab }),

        setDocuments: (docs) => set({ documents: docs }),
        setDocumentSearchQuery: (q) => set({ documentSearchQuery: q }),
        setDocumentFilters: (f) => set({ documentFilters: f }),

        pinDoc: (doc) => {
          const existing = get().pinnedDocs;
          if (existing.some((d) => d.id === doc.id)) return;
          const nextPinned = [...existing, doc];
          set({ pinnedDocs: nextPinned });
          // chat이 이미 존재하면 즉시 sync — 새 chat은 sendMessage에서 생성됨
          const { chats, currentChatId } = get();
          if (currentChatId) {
            const synced = syncWorkingToChats(chats, currentChatId, {
              pinnedDocs: nextPinned,
            });
            set({ chats: synced.chats });
          }
        },
        unpinDoc: (docId) => {
          const nextPinned = get().pinnedDocs.filter((d) => d.id !== docId);
          set({ pinnedDocs: nextPinned });
          const { chats, currentChatId } = get();
          if (currentChatId) {
            const synced = syncWorkingToChats(chats, currentChatId, {
              pinnedDocs: nextPinned,
            });
            set({ chats: synced.chats });
          }
        },
        setCommand: (cmd) => {
          set({ activeCommand: cmd });
          const { chats, currentChatId } = get();
          if (currentChatId) {
            const synced = syncWorkingToChats(chats, currentChatId, {
              activeCommand: cmd,
            });
            set({ chats: synced.chats });
          }
        },
        clearContext: () => {
          set({ pinnedDocs: [], activeCommand: null });
          const { chats, currentChatId } = get();
          if (currentChatId) {
            const synced = syncWorkingToChats(chats, currentChatId, {
              pinnedDocs: [],
              activeCommand: null,
            });
            set({ chats: synced.chats });
          }
        },

        openPreview: (docId) => set({ previewDocId: docId }),
        closePreview: () => set({ previewDocId: null }),

        openCatalog: () => set({ catalogOpen: true }),
        closeCatalog: () => set({ catalogOpen: false }),

        requestSearchFocus: () =>
          set((s) => ({ searchFocusToken: s.searchFocusToken + 1 })),

        requestInputFocus: () =>
          set((s) => ({ inputFocusToken: s.inputFocusToken + 1 })),

        /**
         * 본문의 `[n]` 클릭 시 Evidence 패널의 해당 카드를 1.8초간 하이라이트.
         * 빠른 연속 클릭 시 이전 타이머를 무효화하기 위해 토큰을 증가시킴.
         */
        highlightCitation: (id) => {
          const token = get().highlightToken + 1;
          set({ highlightedCitationId: id, highlightToken: token });
          setTimeout(() => {
            // 이후 다른 citation이 클릭되어 토큰이 증가했다면 이 cleanup은 무시.
            if (get().highlightToken === token) {
              set({ highlightedCitationId: null });
            }
          }, 1900);
        },

        newChat: () =>
          // 현재 chat은 이미 chats[]에 저장돼 있음 — working 상태만 초기화.
          // 새 Chat은 첫 sendMessage에서 lazy-create.
          set({
            messages: [],
            pinnedDocs: [],
            activeCommand: null,
            lastError: null,
            currentChatId: null,
            isLoading: false,
          }),

        openChat: (id) => {
          const chat = get().chats.find((c) => c.id === id);
          if (!chat) return;
          set({
            currentChatId: id,
            messages: chat.messages,
            pinnedDocs: chat.pinnedDocs,
            activeCommand: chat.activeCommand,
            lastError: null,
            isLoading: false,
          });
        },

        deleteChat: (id) => {
          const { chats, currentChatId } = get();
          const nextChats = chats.filter((c) => c.id !== id);
          // 현재 대화를 지우면 working 초기화
          if (id === currentChatId) {
            set({
              chats: nextChats,
              currentChatId: null,
              messages: [],
              pinnedDocs: [],
              activeCommand: null,
              lastError: null,
              isLoading: false,
            });
          } else {
            set({ chats: nextChats });
          }
        },

        sendMessage: async (text) => {
          const state = get();
          // 이미 스트리밍 중이면 중복 호출 무시 (InputBar 레벨에서 막지만 방어적)
          if (state.isLoading) return;

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
          const nextMessages = [...state.messages, userMessage, assistantMessage];
          const history = toChatHistory(state.messages);

          // 첫 메시지라면 Chat을 생성 (syncWorkingToChats가 처리)
          const synced = syncWorkingToChats(state.chats, state.currentChatId, {
            messages: nextMessages,
            pinnedDocs: state.pinnedDocs,
            activeCommand: state.activeCommand,
          });
          const controller = new AbortController();
          set({
            messages: nextMessages,
            chats: synced.chats,
            currentChatId: synced.currentChatId,
            isLoading: true,
            lastError: null,
            abortController: controller,
          });

          await runQuery({
            question: text,
            assistantId,
            pinnedDocs: state.pinnedDocs,
            activeCommand: state.activeCommand,
            activeChatId: synced.currentChatId,
            history,
            signal: controller.signal,
          });
        },

        /**
         * 마지막 assistant 메시지(오류 포함)를 리셋하고 직전 user 메시지로 재질의.
         * 유저 메시지는 중복 append하지 않고 기존 assistant bubble을 재사용.
         */
        retryLastMessage: async () => {
          const state = get();
          if (state.isLoading) return;
          if (state.messages.length === 0) return;

          // 뒤에서부터 마지막 assistant / 그 직전 user를 찾는다.
          let assistantIdx = -1;
          for (let i = state.messages.length - 1; i >= 0; i--) {
            if (state.messages[i].role === "assistant") {
              assistantIdx = i;
              break;
            }
          }
          if (assistantIdx < 0) return;
          // 직전 user 메시지 (같은 exchange의 질문)
          let userIdx = -1;
          for (let i = assistantIdx - 1; i >= 0; i--) {
            if (state.messages[i].role === "user") {
              userIdx = i;
              break;
            }
          }
          if (userIdx < 0) return;

          const question = state.messages[userIdx].content;
          const assistantId = state.messages[assistantIdx].id;

          // assistant 메시지를 pending 상태로 리셋 (ID 유지 → bubble-enter 재발동 X,
          // content 비우면 AssistantMessage가 자동으로 타이핑 점 상태 표시)
          const resetMessages = state.messages.map((m, i) =>
            i === assistantIdx
              ? {
                  ...m,
                  content: "",
                  citations: [],
                  metadata: undefined,
                  error: undefined,
                  createdAt: Date.now(),
                }
              : m
          );
          const history = toChatHistory(resetMessages);
          const synced = syncWorkingToChats(state.chats, state.currentChatId, {
            messages: resetMessages,
          });
          const controller = new AbortController();
          set({
            messages: resetMessages,
            chats: synced.chats,
            isLoading: true,
            lastError: null,
            abortController: controller,
          });

          await runQuery({
            question,
            assistantId,
            pinnedDocs: state.pinnedDocs,
            activeCommand: state.activeCommand,
            activeChatId: synced.currentChatId,
            history,
            signal: controller.signal,
          });
        },

        /**
         * 스트리밍 중인 요청을 취소. abortController가 없으면 no-op.
         * runQuery의 catch에서 AbortError를 감지해 오류 버블을 만들지 않는다.
         */
        abortCurrentRequest: () => {
          const controller = get().abortController;
          if (!controller) return;
          controller.abort();
          // isLoading은 runQuery의 finally에서 정리됨
        },
      };
    },
    {
      name: "bidmate-session",
      // localStorage로 이주 — 탭을 닫아도 대화 히스토리 유지.
      storage: createJSONStorage(() => localStorage),
      version: 2,
      partialize: (state) => ({
        chats: state.chats,
        currentChatId: state.currentChatId,
        messages: state.messages,
        pinnedDocs: state.pinnedDocs,
        activeCommand: state.activeCommand,
        sidebarCollapsed: state.sidebarCollapsed,
        activeTab: state.activeTab,
      }),
      // v1(sessionStorage 기반) → v2 마이그레이션: 기존 messages를 단일 chat으로 래핑
      migrate: (persisted: unknown, version: number) => {
        const source = persisted as Partial<{
          chats: Chat[];
          currentChatId: string | null;
          messages: Message[];
          pinnedDocs: DocumentSummary[];
          activeCommand: SlashCommandMeta | null;
          sidebarCollapsed: boolean;
          activeTab: "chat" | "documents";
        }> | null;
        if (!source) return source as never;
        if (version >= 2) return source as never;
        // v1에는 chats 개념이 없었음 — 기존 messages가 있으면 하나의 chat으로 감싼다.
        const legacyMessages = source.messages ?? [];
        if (legacyMessages.length === 0) {
          return {
            ...source,
            chats: [],
            currentChatId: null,
          } as never;
        }
        const now = Date.now();
        const migrated: Chat = {
          id: newId("chat"),
          title: autoTitle(legacyMessages),
          createdAt: now,
          updatedAt: now,
          messages: legacyMessages,
          pinnedDocs: source.pinnedDocs ?? [],
          activeCommand: source.activeCommand ?? null,
        };
        return {
          ...source,
          chats: [migrated],
          currentChatId: migrated.id,
        } as never;
      },
    }
  )
);
