const API_BASE_URL = import.meta.env.VITE_RAG_API_URL ?? 'http://127.0.0.1:8000';

export interface Source {
  citation_id: number;
  document_id: number;
  relative_path: string;
  location: string;
  preview: string;
  page_number?: number | null;
  source_type: string;
  view_url: string;
  download_url: string;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  used_sources?: Source[];
  retrieved_sources: Source[];
  meta: {
    llm_model: string;
    embedding_model: string;
    top_k: number;
    source_count: number;
    retrieved_source_count: number;
  };
}

export interface AskRequest {
  question: string;
  top_k?: number;
  llm_model?: string;
  num_predict?: number;
}

export async function askQuestion(request: AskRequest): Promise<AskResponse> {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`API request failed with status ${response.status}`);
  }

  return response.json() as Promise<AskResponse>;
}

export function apiUrl(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  return `${API_BASE_URL}${path}`;
}

export function sourceViewHref(source: Source): string {
  const baseUrl = apiUrl(source.view_url);
  if (source.source_type === 'pdf' && source.page_number) {
    return `${baseUrl}#page=${source.page_number}`;
  }
  return baseUrl;
}

export function sourceDownloadHref(source: Source): string {
  return apiUrl(source.download_url);
}
