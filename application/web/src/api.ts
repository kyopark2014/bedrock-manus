import type { AppConfig, DashboardStats, Message, StreamEvent, Task, TaskRun } from "./types";
import { uiError, uiLog } from "./debug";

export interface RagUploadResult {
  ok: boolean;
  file_name: string;
  s3_key: string;
  url?: string | null;
  message: string;
  sync?: {
    ingestion_job_id?: string;
    status?: string;
  };
}

export interface RagUploadPresignResult {
  ok: boolean;
  file_name: string;
  s3_key: string;
  content_type?: string;
  upload_url: string;
  headers: Record<string, string>;
  expires_in?: number;
  url?: string | null;
}

export interface FileUploadResult {
  ok: boolean;
  file_name: string;
  s3_key: string;
  url: string;
  content_type?: string;
}

export interface LlmGatewayConfig {
  url: string;
  configured: boolean;
  key_configured?: boolean;
}

export interface LlmGatewayVerifyResult {
  ok: boolean;
  message: string;
  models: string[];
  ui_models?: string[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET";
  uiLog(`api:${method} ${path}`);
  const res = await fetch(path, {
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    uiError(`api:${method} ${path} failed`, { status: res.status, body: text });
    let message = text || res.statusText;
    try {
      const parsed = JSON.parse(text) as { detail?: string | { msg?: string }[] };
      if (typeof parsed.detail === "string" && parsed.detail) {
        message = parsed.detail;
      }
    } catch {
      // keep raw text
    }
    throw new Error(message);
  }
  if (res.status === 204) {
    uiLog(`api:${method} ${path} -> 204`);
    return undefined as T;
  }
  const text = await res.text();
  if (!text) {
    uiLog(`api:${method} ${path} -> empty`);
    return undefined as T;
  }
  const data = JSON.parse(text) as T;
  uiLog(`api:${method} ${path} -> ok`);
  return data;
}

export interface GraphStatus {
  user_id: string;
  exists: boolean;
  path: string | null;
  status: "idle" | "queued" | "running" | "ready" | "error" | "skipped_cooldown" | "disabled" | string;
  enabled?: boolean;
  error?: string | null;
  last_success_at?: string | null;
  cooldown_seconds?: number;
  next_eligible_at?: string | null;
}

export type GraphPattern = "pattern1" | "pattern2" | "pattern3";

export interface SessionInfo {
  user_id: string;
  name?: string | null;
  picture?: string | null;
  llm_gateway_ready?: boolean;
  knowledge_graph_enabled?: boolean;
  graph_pattern?: GraphPattern | string;
}

export const api = {
  getSession: () => request<SessionInfo | null>("/api/session"),
  setSession: (credential: string) =>
    request<SessionInfo>("/api/session", {
      method: "POST",
      body: JSON.stringify({ credential }),
    }),
  setSessionWithAccessToken: (access_token: string) =>
    request<SessionInfo>("/api/session", {
      method: "POST",
      body: JSON.stringify({ access_token }),
    }),
  setLocalSession: (user_id: string) =>
    request<SessionInfo>("/api/session", {
      method: "POST",
      body: JSON.stringify({ user_id }),
    }),
  clearSession: () => request<void>("/api/session", { method: "DELETE" }),
  patchSessionSettings: (body: {
    knowledge_graph_enabled?: boolean;
    graph_pattern?: GraphPattern | string;
  }) =>
    request<SessionInfo>("/api/session/settings", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  getGraphStatus: () => request<GraphStatus>("/api/graph/status"),
  rebuildGraph: (force = false) =>
    request<GraphStatus>(`/api/graph/rebuild${force ? "?force=1" : ""}`, {
      method: "POST",
    }),
  getConfig: () => request<AppConfig>("/api/config"),
  getAdminDashboard: () => request<DashboardStats>("/api/admin/dashboard"),
  getLlmGateway: () =>
    request<LlmGatewayConfig>("/api/config/llm-gateway"),
  verifyLlmGateway: (body: { url: string; key?: string }) =>
    request<LlmGatewayVerifyResult>("/api/config/llm-gateway/verify", {
      method: "POST",
      body: JSON.stringify({ url: body.url, key: body.key ?? "" }),
    }),
  listTasks: () => request<{ tasks: Task[] }>("/api/tasks"),
  createTask: (body: Partial<Task>) =>
    request<Task>("/api/tasks", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getTask: (id: string) => request<Task>(`/api/tasks/${id}`),
  patchTask: (id: string, body: Partial<Task>) =>
    request<Task>(`/api/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  deleteTask: (id: string) =>
    request<{ ok: boolean }>(`/api/tasks/${id}`, { method: "DELETE" }),
  getMessages: (id: string) =>
    request<{ messages: Message[] }>(`/api/tasks/${id}/messages`),
  getTaskRun: (id: string) => request<TaskRun>(`/api/tasks/${id}/run`),
  cancelTaskRun: (id: string) =>
    request<{ ok: boolean; task_id: string; cancelled: boolean }>(
      `/api/tasks/${id}/cancel`,
      { method: "POST" },
    ),
  addMessage: (
    id: string,
    body: {
      role?: "assistant" | "user";
      content: string;
      images?: string[];
      tool_events?: Message["tool_events"];
    },
  ) =>
    request<Message>(`/api/tasks/${id}/messages`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  uploadToRag: async (
    file: File,
    options?: { sync?: boolean },
  ): Promise<RagUploadResult> => {
    // Presigned PUT: browser → S3 directly (avoids ECS/ALB ~80MB body limits).
    const sync = options?.sync !== false;
    uiLog("rag:upload start", { name: file.name, size: file.size, sync });

    const presign = await request<RagUploadPresignResult>("/api/rag/upload/presign", {
      method: "POST",
      body: JSON.stringify({
        file_name: file.name,
        size: file.size,
        content_type: file.type || undefined,
      }),
    });
    if (!presign.upload_url || !presign.s3_key) {
      throw new Error("Presign succeeded but no upload URL was returned");
    }

    uiLog("rag:upload put start", {
      name: presign.file_name,
      s3_key: presign.s3_key,
      size: file.size,
      host: (() => {
        try {
          return new URL(presign.upload_url).host;
        } catch {
          return "";
        }
      })(),
    });

    const putHeaders = new Headers(presign.headers || {});
    if (!putHeaders.has("Content-Type")) {
      putHeaders.set(
        "Content-Type",
        presign.content_type || "application/octet-stream",
      );
    }
    let putRes: Response;
    try {
      putRes = await fetch(presign.upload_url, {
        method: "PUT",
        body: file,
        headers: putHeaders,
      });
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      uiError("rag:upload put network error", { detail });
      throw new Error(`S3 직접 업로드 네트워크 오류: ${detail}`);
    }
    if (!putRes.ok) {
      const text = await putRes.text();
      uiError("rag:upload put failed", { status: putRes.status, body: text });
      const codeMatch = text.match(/<Code>([^<]+)<\/Code>/i);
      const msgMatch = text.match(/<Message>([^<]+)<\/Message>/i);
      const s3Detail =
        codeMatch || msgMatch
          ? [codeMatch?.[1], msgMatch?.[1]].filter(Boolean).join(": ")
          : "";
      throw new Error(
        s3Detail ||
          text.slice(0, 200) ||
          putRes.statusText ||
          `Direct S3 upload failed (HTTP ${putRes.status})`,
      );
    }

    const data = await request<RagUploadResult>("/api/rag/upload/complete", {
      method: "POST",
      body: JSON.stringify({
        file_name: presign.file_name,
        s3_key: presign.s3_key,
        size: file.size,
        sync,
      }),
    });
    uiLog("rag:upload complete", data);
    return data;
  },
  uploadFile: async (file: File): Promise<FileUploadResult> => {
    uiLog("file:upload start", { name: file.name, size: file.size, type: file.type });
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/files/upload", {
      method: "POST",
      credentials: "include",
      body: form,
    });
    if (!res.ok) {
      const text = await res.text();
      uiError("file:upload failed", { status: res.status, body: text });
      throw new Error(text || res.statusText);
    }
    const data = (await res.json()) as FileUploadResult;
    if (!data.url) {
      throw new Error("Upload succeeded but no URL was returned");
    }
    uiLog("file:upload complete", data);
    return data;
  },
  streamChat: async function* (
    taskId: string,
    prompt: string,
    files: string[] = [],
    signal?: AbortSignal,
  ): AsyncGenerator<StreamEvent> {
    uiLog("chat:stream start", { taskId, prompt, files });
    const res = await fetch(`/api/tasks/${taskId}/chat`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, files }),
      signal,
    });
    if (!res.ok || !res.body) {
      const body = await res.text();
      uiError("chat:stream request failed", { status: res.status, body });
      throw new Error("Chat request failed. Please try again.");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let eventCount = 0;

    try {
      while (true) {
        if (signal?.aborted) {
          throw new DOMException("Aborted", "AbortError");
        }
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          const line = part
            .split("\n")
            .find((l) => l.startsWith("data:"));
          if (!line) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          const event = JSON.parse(payload) as StreamEvent;
          eventCount += 1;
          if (event.type === "token") {
            const text = event.data ?? "";
            uiLog("chat:sse token", { chars: text.length, preview: text.slice(0, 80) });
          } else if (event.type === "error") {
            uiError("chat:sse error", event);
          } else {
            uiLog(`chat:sse ${event.type}`, event);
          }
          yield event;
        }
      }
    } catch (err) {
      try {
        await reader.cancel();
      } catch {
        /* ignore cancel errors */
      }
      throw err;
    }

    uiLog("chat:stream end", { taskId, eventCount });
  },
};
