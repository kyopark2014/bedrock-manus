export interface Task {
  id: string;
  user_id: string;
  title: string;
  runtime_session_id: string;
  model_name: string;
  skills: string[];
  mcp_servers: string[];
  guardrail_enabled: boolean;
  llm_gateway_enabled: boolean;
  memory_enabled: boolean;
  pinned: boolean;
  created_at: string;
  updated_at: string;
}

export interface ToolEvent {
  type: "text" | "tool" | "tool_result" | "info" | "plan" | "status";
  tool?: string;
  input?: unknown;
  toolUseId?: string;
  data?: string;
}

export interface Message {
  id: string;
  task_id: string;
  role: "user" | "assistant";
  content: string;
  images: string[];
  tool_events: ToolEvent[];
  created_at: string;
}

export interface AppConfig {
  projectName: string;
  google_client_id?: string;
  local_auth_bypass?: boolean;
  is_admin?: boolean;
  skills: string[];
  mcp_servers: string[];
  models: string[];
  gateway_models?: string[];
  default_model: string;
  default_gateway_model?: string;
  default_skills: string[];
  default_mcp_servers: string[];
  llm_gateway_configured?: boolean;
}

export interface DashboardSummary {
  total_users: number;
  google_users: number;
  legacy_users: number;
  total_tasks: number;
  total_messages: number;
  total_logins: number;
  logins_today: number;
  active_users_today: number;
  logins_7d: number;
  active_users_7d: number;
}

export interface DashboardUser {
  user_id: string;
  task_count: number;
  message_count: number;
  login_count: number;
  first_seen?: string | null;
  last_active?: string | null;
  last_login?: string | null;
  auth_methods: string[];
  is_google: boolean;
}

export interface DashboardLogin {
  id: string;
  user_id: string;
  method: string;
  name?: string | null;
  picture?: string | null;
  logged_at: string;
}

export interface DashboardDailyLogin {
  date: string;
  logins: number;
  unique_users: number;
}

export interface DashboardStats {
  summary: DashboardSummary;
  users: DashboardUser[];
  recent_logins: DashboardLogin[];
  daily_logins: DashboardDailyLogin[];
}

export interface StreamEvent {
  type: "token" | "text" | "tool" | "tool_result" | "info" | "plan" | "status" | "done" | "error";
  data?: string;
  content?: string;
  images?: string[];
  tool_events?: ToolEvent[];
  tool?: string;
  mcpServer?: string;
  input?: unknown;
  toolUseId?: string;
}

export type TaskRunStatus =
  | "idle"
  | "running"
  | "pending"
  | "done"
  | "error"
  | "missing";

export interface TaskRun {
  task_id: string;
  status: TaskRunStatus;
  content: string;
  images: string[];
  error?: string | null;
  source?: "messages" | "registry" | "checkpoint" | null;
  hydrated: boolean;
}
