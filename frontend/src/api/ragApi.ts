const API_BASE_URL = import.meta.env.VITE_RAG_API_URL ?? 'http://127.0.0.1:8000';

export interface Source {
  document_id: number;
  relative_path: string;
  location: string;
  preview: string;
  page_number?: number | null;
  source_type: string;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  meta: {
    llm_model: string;
    embedding_model: string;
    top_k: number;
    source_count: number;
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
