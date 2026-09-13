export type User = {
  id: number;
  email: string;
  auth_provider: string;
  is_active: boolean;
  first_name?: string | null;
  last_name?: string | null;
  display_name?: string | null;
};

export type DeveloperAccessResult = {
  authorized: boolean;
  message: string;
};

export type AccessConfig = {
  accessPassword: string;
  userOpenAiApiKey: string;
};

export type UploadResult = {
  status: string;
  filename: string;
  saved_filename: string;
  file_size_mb: number;
  total_pages: number;
  extracted_characters: number;
  num_chunks: number;
  selected_chunk_size: number;
  selected_chunk_overlap: number;
  preview_chunk?: string | null;
  message?: string | null;
};

export type SummaryResult = {
  status: string;
  active_document: string | null;
  summary: string;
  key_topics: string[];
  message: string;
};

export type SuggestedQuestionsResult = {
  status: string;
  active_document: string | null;
  suggested_questions: string[];
  message: string;
};

export type RetrievedChunk = {
  chunk_id: number;
  file_name: string;
  page_number: number;
  text: string;
  score: number;
};

export type AskEvidence = {
  page_number: number;
  snippet: string;
  image_path?: string;
  chunk_id?: number;
  file_name?: string;
};

export type EvaluationDecision = "AUTO_APPROVE" | "REVIEW_RECOMMENDED" | "HUMAN_REVIEW";

export type EvaluationEvidence = {
  page_number: number;
  snippet: string;
  chunk_id?: number | null;
  file_name?: string | null;
};

export type EvaluationClaim = {
  claim: string;
  supported: boolean;
  evidence: EvaluationEvidence[];
  reason?: string | null;
};

export type EvaluationResult = {
  evaluation_id: string;
  user_id?: number | null;
  question: string;
  answer: string | null;
  retrieval_quality: number;
  source_relevance: number;
  grounding: number;
  answer_correctness: number;
  citation_quality: number;
  overall_confidence: number;
  decision: EvaluationDecision;
  reasons: string[];
  claims: EvaluationClaim[];
};

export type ReviewDecision = "AUTO_APPROVE" | "REVIEW_RECOMMENDED" | "HUMAN_REVIEW";
export type FeedbackCategory = "incorrect_retrieval" | "insufficient_evidence" | "unsupported_claim" | "hallucination" | "incorrect_answer" | "poor_source" | "citation_problem" | "other";
export type HumanReviewResult = {
  status: string;
  evaluation_id: string;
  review: {
    evaluation_id: string;
    decision: ReviewDecision;
    feedback_category?: FeedbackCategory | null;
    feedback_text?: string | null;
    corrected_answer?: string | null;
  };
  message: string;
};

export type SecurityCheckResult = {
  decision: "ALLOW" | "BLOCK" | "REVIEW";
  is_safe: boolean;
  reasons: string[];
  risk_score: number;
};

export type AskResult = {
  status: string;
  active_document: string | null;
  question: string;
  question_type: string;
  answer: string | null;
  confidence: string | null;
  best_score: number | null;
  average_score: number | null;
  evidence: AskEvidence[];
  retrieved_chunks: RetrievedChunk[];
  evaluation?: EvaluationResult | null;
  message: string;
};

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api").replace(/\/$/, "");
const SESSION_HINT = "intelliground_session";

export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

function messageFromPayload(payload: unknown, fallback: string) {
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (typeof detail === "object" && detail !== null && "message" in detail) {
      const message = (detail as { message?: unknown }).message;
      if (typeof message === "string") return message;
    }
    if (Array.isArray(detail)) {
      const messages = detail.map((item) => {
        if (typeof item === "object" && item !== null && "msg" in item) return (item as { msg?: unknown }).msg;
        return null;
      }).filter((message): message is string => typeof message === "string" && message.length > 0);
      if (messages.length > 0) return messages.join(" ");
    }
  }
  return fallback;
}

async function request<T>(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
    credentials: "include",
  });
  const contentType = response.headers.get("content-type") ?? "";
  const payload: unknown = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) throw new ApiError(messageFromPayload(payload, "The request could not be completed."), response.status, payload);
  return payload as T;
}

export const authApi = {
  async login(email: string, password: string) {
    await request<User>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
  },
  register(firstName: string, lastName: string, email: string, password: string) {
    return request<User>("/auth/register", { method: "POST", body: JSON.stringify({ first_name: firstName, last_name: lastName, email, password }) });
  },
  forgotPassword(email: string) {
    return request<{ message: string }>("/auth/forgot-password", { method: "POST", body: JSON.stringify({ email }) });
  },
  resetPassword(token: string, newPassword: string) {
    return request<{ message: string }>("/auth/reset-password", { method: "POST", body: JSON.stringify({ token, new_password: newPassword }) });
  },
  me() {
    return request<User>("/auth/me");
  },
  developerAccess(password: string) {
    return request<DeveloperAccessResult>("/auth/developer-access", {
      method: "POST",
      body: JSON.stringify({ password }),
    });
  },
  async logout() {
    try {
      await request<{ message: string }>("/auth/logout", { method: "POST" });
    } finally {
      document.cookie = `${SESSION_HINT}=; Max-Age=0; path=/`;
    }
  },
  googleLoginUrl() {
    return `${API_BASE_URL}/auth/google/login`;
  },
};

export const documentApi = {
  upload(file: File, access: AccessConfig) {
    const body = new FormData();
    body.append("file", file);
    body.append("access_password", access.accessPassword);
    body.append("user_openai_api_key", access.userOpenAiApiKey);
    return request<UploadResult>("/upload-pdf", { method: "POST", body });
  },
  summary(access: AccessConfig) {
    return request<SummaryResult>("/summary", {
      method: "POST",
      body: JSON.stringify({ question: "", access_password: access.accessPassword, user_openai_api_key: access.userOpenAiApiKey }),
    });
  },
  suggestedQuestions(access: AccessConfig) {
    return request<SuggestedQuestionsResult>("/suggested-questions", {
      method: "POST",
      body: JSON.stringify({ question: "", access_password: access.accessPassword, user_openai_api_key: access.userOpenAiApiKey }),
    });
  },
  ask(question: string, access: AccessConfig) {
    return request<AskResult>("/ask", {
      method: "POST",
      body: JSON.stringify({ question, access_password: access.accessPassword, user_openai_api_key: access.userOpenAiApiKey }),
    });
  },
  getEvaluation(evaluationId: string) {
    return request<EvaluationResult>(`/evaluation/${encodeURIComponent(evaluationId)}`);
  },
  submitReview(evaluationId: string, review: { decision: ReviewDecision; feedback_category?: FeedbackCategory; feedback_text?: string; corrected_answer?: string }) {
    return request<HumanReviewResult>(`/evaluation/${encodeURIComponent(evaluationId)}/review`, {
      method: "POST",
      body: JSON.stringify(review),
    });
  },
};

export async function fetchProtectedImage(imagePath: string) {
  const url = imagePath.startsWith("http") ? imagePath : `${API_BASE_URL}${imagePath.replace(/^\/api/, "")}`;
  const response = await fetch(url, { credentials: "include" });
  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    const payload: unknown = contentType.includes("application/json") ? await response.json() : await response.text();
    throw new ApiError(messageFromPayload(payload, "The source page could not be loaded."), response.status, payload);
  }
  return response.blob();
}

export function hasSession() {
  return document.cookie.split("; ").some((cookie) => cookie.startsWith(`${SESSION_HINT}=1`));
}
