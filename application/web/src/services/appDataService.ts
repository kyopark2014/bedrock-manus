import { api } from "../api";
import type { AppConfig, Message, Task, TaskRun } from "../types";

export type CreateTaskDefaults = {
  model_name?: string;
  skills?: string[];
  mcp_servers?: string[];
  guardrail_enabled?: boolean;
  memory_enabled?: boolean;
  llm_gateway_enabled?: boolean;
};

function sanitizeError(error: unknown, fallback: string): Error {
  if (error instanceof Error) {
    console.error(fallback, error);
  } else if (error !== undefined) {
    console.error(fallback, error);
  }
  return new Error(fallback);
}

/** Orchestrates session, config, and task API calls for the App shell. */
export const appDataService = {
  async loadBootState(): Promise<{
    config: AppConfig;
    userId: string | null;
    llmGatewayReady: boolean;
    knowledgeGraphEnabled: boolean;
    graphPattern: string;
  }> {
    try {
      const config = await api.getConfig();
      const session = await api.getSession();
      const userId = session?.user_id?.trim() || null;
      return {
        config,
        userId,
        llmGatewayReady: Boolean(session?.llm_gateway_ready),
        knowledgeGraphEnabled: session?.knowledge_graph_enabled ?? true,
        graphPattern: session?.graph_pattern || "pattern1",
      };
    } catch (error) {
      throw sanitizeError(error, "Failed to load application configuration.");
    }
  },

  async setSession(credential: string) {
    try {
      return await api.setSession(credential);
    } catch (error) {
      throw sanitizeError(error, "Google login failed.");
    }
  },

  async setSessionWithAccessToken(accessToken: string) {
    try {
      return await api.setSessionWithAccessToken(accessToken);
    } catch (error) {
      throw sanitizeError(error, "Google login failed.");
    }
  },

  async setLocalSession(userId: string) {
    try {
      return await api.setLocalSession(userId);
    } catch (error) {
      throw sanitizeError(error, "Login failed.");
    }
  },

  async logout(): Promise<void> {
    try {
      await api.clearSession();
    } catch (error) {
      throw sanitizeError(error, "Logout failed.");
    }
  },

  async listTasksSorted(sort: (tasks: Task[]) => Task[]): Promise<Task[]> {
    try {
      const { tasks } = await api.listTasks();
      return sort(tasks);
    } catch (error) {
      throw sanitizeError(error, "Failed to load tasks.");
    }
  },

  async getMessages(taskId: string): Promise<Message[]> {
    try {
      const { messages } = await api.getMessages(taskId);
      return messages;
    } catch (error) {
      throw sanitizeError(error, "Failed to load messages.");
    }
  },

  async getTaskRun(taskId: string): Promise<TaskRun> {
    try {
      return await api.getTaskRun(taskId);
    } catch (error) {
      throw sanitizeError(error, "Failed to load task run status.");
    }
  },

  async cancelTaskRun(taskId: string): Promise<void> {
    try {
      await api.cancelTaskRun(taskId);
    } catch (error) {
      throw sanitizeError(error, "Failed to cancel task run.");
    }
  },

  async addMessage(
    taskId: string,
    body: {
      role?: "assistant" | "user";
      content: string;
      images?: string[];
      tool_events?: Message["tool_events"];
    },
  ): Promise<Message> {
    try {
      return await api.addMessage(taskId, body);
    } catch (error) {
      throw sanitizeError(error, "Failed to save message.");
    }
  },

  async createTask(defaults: CreateTaskDefaults): Promise<Task> {
    try {
      return await api.createTask(defaults);
    } catch (error) {
      throw sanitizeError(error, "Failed to create task.");
    }
  },

  async ensureInitialTask(
    config: AppConfig,
    sort: (tasks: Task[]) => Task[],
    llmGatewayReady = false,
  ): Promise<Task> {
    try {
      const latest = sort((await api.listTasks()).tasks);
      if (latest.length > 0) return latest[0];
      return await api.createTask({
        model_name: config.default_model,
        skills: config.default_skills,
        mcp_servers: config.default_mcp_servers,
        memory_enabled: true,
        llm_gateway_enabled:
          Boolean(config.llm_gateway_configured) && llmGatewayReady,
      });
    } catch (error) {
      throw sanitizeError(error, "Failed to initialize task.");
    }
  },

  async patchTask(taskId: string, patch: Partial<Task>): Promise<Task> {
    try {
      return await api.patchTask(taskId, patch);
    } catch (error) {
      if (error instanceof Error && error.message) throw error;
      throw sanitizeError(error, "Failed to update task.");
    }
  },

  async deleteTask(taskId: string): Promise<void> {
    try {
      await api.deleteTask(taskId);
    } catch (error) {
      throw sanitizeError(error, "Failed to delete task.");
    }
  },
};
