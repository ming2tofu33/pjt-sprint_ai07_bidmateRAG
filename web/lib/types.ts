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
  provider_config: string;
  chunking_config: string | null;
  mentioned_doc_ids: string[];
  command: string | null;
  top_k: number;
  max_context_chars: number;
}

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
