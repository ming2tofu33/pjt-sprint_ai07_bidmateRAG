import type {
  DocumentSummary,
  DocumentDetail,
  SlashCommandMeta,
  QueryRequest,
  QueryResponse,
} from "./types";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // ignore — use statusText
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export async function listDocuments(): Promise<{
  documents: DocumentSummary[];
  total: number;
}> {
  return fetchJson("/api/documents");
}

export async function getDocument(docId: string): Promise<DocumentDetail> {
  return fetchJson(`/api/documents/${encodeURIComponent(docId)}`);
}

export async function listCommands(): Promise<{ commands: SlashCommandMeta[] }> {
  return fetchJson("/api/commands");
}

export async function postQuery(request: QueryRequest): Promise<QueryResponse> {
  return fetchJson("/api/query", {
    method: "POST",
    body: JSON.stringify(request),
  });
}
