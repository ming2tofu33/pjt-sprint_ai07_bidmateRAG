export interface DocumentSummary {
  id: string;
  title: string;
  agency: string;
  agency_type: string;
  domain: string;
  budget: number;
  budget_label: string;
  deadline: string | null;
  char_count: number;
}

export interface DocumentDetail extends DocumentSummary {
  summary_oneline: string | null;
  quick_facts: Array<{ label: string; value: string }>;
}

export interface SlashCommandMeta {
  id: string;
  label: string;
  description: string;
  icon: string;
  requires_doc: boolean;
  requires_multi_doc: boolean;
}

export interface Citation {
  id: number;
  doc_id: string;
  doc_title: string;
  section: string;
  content_type: string;
  text: string;
  score: number;
}

export interface QueryMetadata {
  model: string;
  token_usage: Record<string, number>;
  latency_ms: number;
  cost_usd: number;
  command_applied: string | null;
  filter_applied: Record<string, unknown> | null;
  retrieval_strategy: "single" | "per_doc_split" | "static";
  per_doc_k: number | null;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  metadata: QueryMetadata;
}

export interface QueryRequest {
  question: string;
  mentioned_doc_ids: string[];
  history: Array<{ role: MessageRole; content: string }>;
  command: string | null;
  // provider_config, chunking_config, top_k, max_context_chars 은
  // 서버의 configs/web.yaml 기본값을 사용 — 프론트에서 보내지 않는다.
}

export type QueryStreamEvent =
  | {
      type: "retrieval";
      citations: Citation[];
      retrieval_strategy: "single" | "per_doc_split" | "static";
    }
  | { type: "token"; delta: string }
  | { type: "done"; metadata: QueryMetadata }
  | { type: "error"; message: string };

export type MessageRole = "user" | "assistant";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  citations?: Citation[];
  metadata?: QueryMetadata;
  error?: string;
}

export interface Chat {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
  pinnedDocs: DocumentSummary[];
  activeCommand: SlashCommandMeta | null;
}
