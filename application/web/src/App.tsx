import { useCallback, useEffect, useRef, useState } from "react";
import { uiError, uiLog } from "./debug";
import { formatBrandTitle } from "./formatBrandTitle";
import { useChatStream } from "./hooks/useChatStream";
import { appDataService } from "./services/appDataService";
import {
  applyTaskTitleFromPrompt,
  buildFallbackTaskDefaults,
  buildNewTaskDefaults,
  mostRecentlyUpdatedTask,
  sortTasks,
} from "./services/taskService";
import {
  buildDisplayPrompt,
  buildOptimisticUserMessage,
  buildPendingAssistantMessage,
  buildRagUploadNotice,
  shouldAppendAssistantMessage,
  stabilizeMessageKeys,
} from "./services/messageService";
import { api } from "./api";
import type { AppConfig, Message, Task } from "./types";
import { Sidebar } from "./components/Sidebar";
import { ChatThread } from "./components/ChatThread";
import { ChatInput } from "./components/ChatInput";
import { GoogleLoginModal } from "./components/GoogleLoginModal";
import { Dashboard } from "./components/Dashboard";

type DrawerKind = "skill" | "mcp" | "model" | "appearance" | null;

type QueuedMessage = {
  id: string;
  text: string;
  files: string[];
};

const MOBILE_BREAKPOINT_PX = 768;
const BOOT_ERROR_MESSAGE =
  "Failed to load application configuration. Please try again.";
const TASK_ERROR_MESSAGE = "Task operation failed. Please try again.";
const CHAT_ERROR_MESSAGE = "Failed to send message. Please try again.";
const LOAD_MESSAGES_ERROR_MESSAGE =
  "Failed to load messages. Please try again.";
/** Poll interval while waiting for a background agent after refresh. */
const RUN_RECOVERY_POLL_MS = 2500;
/** Stop polling after this many attempts (~50 min at 2.5s). */
const RUN_RECOVERY_MAX_ATTEMPTS = 1200;

export default function App() {
  const [userId, setUserId] = useState<string | null>(null);
  const [llmGatewayReady, setLlmGatewayReady] = useState(false);
  const [knowledgeGraphEnabled, setKnowledgeGraphEnabled] = useState(true);
  const [authReady, setAuthReady] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [drawer, setDrawer] = useState<DrawerKind>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [view, setView] = useState<"chat" | "dashboard">("chat");
  const [queuedByTaskId, setQueuedByTaskId] = useState<
    Record<string, QueuedMessage[]>
  >({});
  const [queuePausedByTaskId, setQueuePausedByTaskId] = useState<
    Record<string, boolean>
  >({});
  /** True while polling /run after refresh when messages end on user. */
  const [recoveringTaskId, setRecoveringTaskId] = useState<string | null>(null);
  const { getStreamForTask, sendMessage, stopMessage } = useChatStream();
  const emptyTaskBootstrapRef = useRef<Promise<Task> | null>(null);
  const tasksBootstrappedForUserRef = useRef<string | null>(null);
  const activeTaskIdRef = useRef<string | null>(null);
  const queuedByTaskIdRef = useRef(queuedByTaskId);
  const pendingSteerRef = useRef<{
    taskId: string;
    text: string;
    files: string[];
  } | null>(null);

  const activeTask = tasks.find((t) => t.id === activeTaskId) ?? null;
  const activeStream = getStreamForTask(activeTaskId);
  const lastMessageRole = messages[messages.length - 1]?.role;
  const needsRecovery =
    Boolean(activeTaskId) &&
    !activeStream.streaming &&
    lastMessageRole === "user";
  const isRecovering =
    Boolean(activeTaskId) && recoveringTaskId === activeTaskId;
  const showWaiting = activeStream.streaming || isRecovering;
  const activeQueuedMessages = activeTaskId
    ? (queuedByTaskId[activeTaskId] ?? [])
    : [];
  const activeQueuePaused = Boolean(
    activeTaskId && queuePausedByTaskId[activeTaskId],
  );

  useEffect(() => {
    activeTaskIdRef.current = activeTaskId;
  }, [activeTaskId]);

  useEffect(() => {
    queuedByTaskIdRef.current = queuedByTaskId;
  }, [queuedByTaskId]);

  const loadMessages = useCallback(async (taskId: string) => {
    uiLog("messages:load start", { taskId });
    const rows = await appDataService.getMessages(taskId);
    uiLog("messages:load complete", {
      taskId,
      count: rows.length,
      roles: rows.map((m) => m.role),
    });
    setMessages((prev) => stabilizeMessageKeys(prev, rows));
    return rows;
  }, []);

  const refreshTasks = useCallback(async () => {
    const rows = await appDataService.listTasksSorted(sortTasks);
    setTasks(rows);
    return rows;
  }, []);

  const refreshConfig = useCallback(async () => {
    const latest = await api.getConfig();
    setConfig(latest);
    return latest;
  }, []);

  /** After refresh: if last message is user, poll /run until assistant arrives. */
  useEffect(() => {
    if (!activeTaskId) {
      setRecoveringTaskId(null);
      return;
    }
    if (!needsRecovery) {
      setRecoveringTaskId((prev) => (prev === activeTaskId ? null : prev));
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let attempts = 0;
    const taskId = activeTaskId;

    setRecoveringTaskId(taskId);
    uiLog("run:recovery start", { taskId });

    async function poll() {
      if (cancelled || activeTaskIdRef.current !== taskId) return;
      attempts += 1;
      try {
        const run = await appDataService.getTaskRun(taskId);
        if (cancelled || activeTaskIdRef.current !== taskId) return;

        if (
          run.status === "done" ||
          run.status === "error" ||
          run.status === "idle" ||
          run.hydrated
        ) {
          uiLog("run:recovery complete", {
            taskId,
            status: run.status,
            source: run.source,
            hydrated: run.hydrated,
          });
          await loadMessages(taskId);
          if (!cancelled && activeTaskIdRef.current === taskId) {
            setRecoveringTaskId((prev) => (prev === taskId ? null : prev));
          }
          return;
        }

        // running / pending — keep waiting
        if (attempts >= RUN_RECOVERY_MAX_ATTEMPTS) {
          uiLog("run:recovery timeout", { taskId, attempts });
          setRecoveringTaskId((prev) => (prev === taskId ? null : prev));
          return;
        }
      } catch (err) {
        uiError("run:recovery poll failed", err);
        if (attempts >= 5) {
          setRecoveringTaskId((prev) => (prev === taskId ? null : prev));
          return;
        }
      }

      timer = setTimeout(poll, RUN_RECOVERY_POLL_MS);
    }

    // Delay first poll so an in-flight sendMessage can mark streaming
    // before we treat a freshly appended user bubble as abandoned.
    timer = setTimeout(poll, RUN_RECOVERY_POLL_MS);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [activeTaskId, needsRecovery, loadMessages]);

  useEffect(() => {
    (async () => {
      try {
        const boot = await appDataService.loadBootState();
        setConfig(boot.config);
        if (boot.userId) {
          setUserId(boot.userId);
          setLlmGatewayReady(boot.llmGatewayReady);
          setKnowledgeGraphEnabled(boot.knowledgeGraphEnabled);
        }
      } catch (err) {
        uiError("boot failed", err);
        setBootError(BOOT_ERROR_MESSAGE);
      } finally {
        setAuthReady(true);
      }
    })();
  }, []);

  useEffect(() => {
    if (!config?.projectName || !userId) return;
    document.title = formatBrandTitle(config.projectName, userId);
  }, [config?.projectName, userId]);

  useEffect(() => {
    if (!userId || !config) return;
    if (tasksBootstrappedForUserRef.current === userId) return;

    let cancelled = false;

    (async () => {
      const rows = await refreshTasks();
      if (cancelled) return;

      if (rows.length === 0) {
        if (!emptyTaskBootstrapRef.current) {
          emptyTaskBootstrapRef.current = appDataService.ensureInitialTask(
            config,
            sortTasks,
            llmGatewayReady,
          );
        }
        const task = await emptyTaskBootstrapRef.current;
        if (cancelled) return;
        setTasks([task]);
        setActiveTaskId(task.id);
        setMessages([]);
      } else {
        const latest = mostRecentlyUpdatedTask(rows)!;
        setActiveTaskId(latest.id);
        await loadMessages(latest.id);
      }
      if (!cancelled) {
        tasksBootstrappedForUserRef.current = userId;
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [userId, config, refreshTasks, loadMessages, llmGatewayReady]);

  useEffect(() => {
    if (activeTaskId) {
      loadMessages(activeTaskId).catch((err) => {
        uiError("task:select failed", err);
        setBootError(LOAD_MESSAGES_ERROR_MESSAGE);
      });
    }
  }, [activeTaskId, loadMessages]);

  useEffect(() => {
    if (!sidebarOpen) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setSidebarOpen(false);
    }
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [sidebarOpen]);

  useEffect(() => {
    function onResize() {
      if (window.innerWidth > MOBILE_BREAKPOINT_PX) setSidebarOpen(false);
    }
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const handleGoogleCredential = useCallback(
    async (credential: string) => {
      setBootError(null);
      try {
        const session = await appDataService.setSession(credential);
        setUserId(session.user_id.trim());
        setLlmGatewayReady(Boolean(session.llm_gateway_ready));
        setKnowledgeGraphEnabled(session.knowledge_graph_enabled ?? true);
        await refreshConfig();
      } catch (err) {
        setBootError(err instanceof Error ? err.message : String(err));
      }
    },
    [refreshConfig],
  );

  const handleGoogleAccessToken = useCallback(
    async (accessToken: string) => {
      setBootError(null);
      try {
        const session =
          await appDataService.setSessionWithAccessToken(accessToken);
        setUserId(session.user_id.trim());
        setLlmGatewayReady(Boolean(session.llm_gateway_ready));
        setKnowledgeGraphEnabled(session.knowledge_graph_enabled ?? true);
        await refreshConfig();
      } catch (err) {
        setBootError(err instanceof Error ? err.message : String(err));
      }
    },
    [refreshConfig],
  );

  const handleLocalUserId = useCallback(
    async (id: string) => {
      setBootError(null);
      try {
        const session = await appDataService.setLocalSession(id.trim());
        setUserId(session.user_id.trim());
        setLlmGatewayReady(Boolean(session.llm_gateway_ready));
        setKnowledgeGraphEnabled(session.knowledge_graph_enabled ?? true);
        await refreshConfig();
      } catch (err) {
        setBootError(err instanceof Error ? err.message : String(err));
      }
    },
    [refreshConfig],
  );

  async function handleLogout() {
    setBootError(null);
    try {
      await appDataService.logout();
    } catch (err) {
      uiError("logout failed", err);
    }
    try {
      window.google?.accounts?.id?.disableAutoSelect();
    } catch {
      // GSI may not be loaded yet
    }
    tasksBootstrappedForUserRef.current = null;
    emptyTaskBootstrapRef.current = null;
    setUserId(null);
    setLlmGatewayReady(false);
    setKnowledgeGraphEnabled(true);
    setTasks([]);
    setActiveTaskId(null);
    setMessages([]);
    setDrawer(null);
    setView("chat");
    setQueuedByTaskId({});
    setQueuePausedByTaskId({});
    if (config?.projectName) {
      document.title = formatBrandTitle(config.projectName);
    }
  }

  async function handleNewTask() {
    if (!config) return;
    try {
      const defaults = buildNewTaskDefaults(config, activeTask);
      const canEnableGateway =
        Boolean(config.llm_gateway_configured) && llmGatewayReady;
      const task = await appDataService.createTask({
        ...defaults,
        llm_gateway_enabled: canEnableGateway
          ? (activeTask?.llm_gateway_enabled ?? true)
          : false,
      });
      setTasks((prev) => [task, ...prev]);
      setActiveTaskId(task.id);
      setMessages([]);
      setConfig((prev) =>
        prev
          ? {
              ...prev,
              default_skills: task.skills,
              default_mcp_servers: task.mcp_servers,
            }
          : prev,
      );
    } catch (err) {
      uiError("task:create failed", err);
      setBootError(TASK_ERROR_MESSAGE);
    }
  }

  async function handleSelectTask(id: string) {
    setView("chat");
    setActiveTaskId(id);
    setSidebarOpen(false);
    try {
      await loadMessages(id);
    } catch (err) {
      uiError("task:select failed", err);
      setBootError(LOAD_MESSAGES_ERROR_MESSAGE);
    }
  }

  async function handlePatchTask(taskId: string, patch: Partial<Task>) {
    try {
      const updated = await appDataService.patchTask(taskId, patch);
      setTasks((prev) =>
        sortTasks(prev.map((t) => (t.id === updated.id ? updated : t))),
      );
      // Keep config defaults in sync with settings.json (server persists on patch).
      if (patch.skills !== undefined || patch.mcp_servers !== undefined) {
        setConfig((prev) =>
          prev
            ? {
                ...prev,
                ...(patch.skills !== undefined
                  ? { default_skills: updated.skills }
                  : {}),
                ...(patch.mcp_servers !== undefined
                  ? { default_mcp_servers: updated.mcp_servers }
                  : {}),
              }
            : prev,
        );
      }
    } catch (err) {
      uiError("task:patch failed", err);
      const message =
        err instanceof Error && err.message
          ? err.message
          : TASK_ERROR_MESSAGE;
      setBootError(message);
      // Let LLM Gateway enable flow surface the error in the modal.
      if (patch.llm_gateway_enabled === true) {
        throw err instanceof Error ? err : new Error(message);
      }
    }
  }

  function handleRemoveQueued(queueId: string) {
    if (!activeTaskId) return;
    const taskId = activeTaskId;
    const nextList = (queuedByTaskIdRef.current[taskId] ?? []).filter(
      (item) => item.id !== queueId,
    );
    const next = { ...queuedByTaskIdRef.current, [taskId]: nextList };
    queuedByTaskIdRef.current = next;
    setQueuedByTaskId(next);
    if (nextList.length === 0) {
      clearQueuePaused(taskId);
    }
  }

  function clearQueuePaused(taskId: string) {
    setQueuePausedByTaskId((prev) => {
      if (!prev[taskId]) return prev;
      const next = { ...prev };
      delete next[taskId];
      return next;
    });
  }

  function handleStop() {
    if (!activeTaskId) return;
    stopMessage(activeTaskId);
  }

  async function handleResumeQueue() {
    if (!activeTaskId) return;
    const taskId = activeTaskId;
    if (getStreamForTask(taskId).streaming) return;
    clearQueuePaused(taskId);
    await drainQueue(taskId);
  }

  async function handleSteerQueued(queueId: string) {
    if (!activeTaskId) return;
    const taskId = activeTaskId;
    const queue = queuedByTaskIdRef.current[taskId] ?? [];
    const item = queue.find((entry) => entry.id === queueId);
    if (!item) return;

    const rest = queue.filter((entry) => entry.id !== queueId);
    const updated = { ...queuedByTaskIdRef.current, [taskId]: rest };
    queuedByTaskIdRef.current = updated;
    setQueuedByTaskId(updated);
    clearQueuePaused(taskId);
    uiLog("chat:queue steer", { taskId, queueId });

    if (getStreamForTask(taskId).streaming) {
      pendingSteerRef.current = {
        taskId,
        text: item.text,
        files: item.files,
      };
      stopMessage(taskId);
      return;
    }

    await dispatchSend(taskId, item.text, item.files);
  }

  async function handleDeleteTask(taskId: string) {
    try {
      await appDataService.deleteTask(taskId);
      setQueuedByTaskId((prev) => {
        if (!(taskId in prev)) return prev;
        const next = { ...prev };
        delete next[taskId];
        queuedByTaskIdRef.current = next;
        return next;
      });
      setQueuePausedByTaskId((prev) => {
        if (!prev[taskId]) return prev;
        const next = { ...prev };
        delete next[taskId];
        return next;
      });
      const rows = await refreshTasks();
      if (activeTaskId !== taskId) return;
      if (rows.length > 0) {
        const latest = mostRecentlyUpdatedTask(rows)!;
        setActiveTaskId(latest.id);
        await loadMessages(latest.id);
        return;
      }
      if (!config) return;
      const task = await appDataService.createTask({
        ...buildFallbackTaskDefaults(config),
        llm_gateway_enabled:
          Boolean(config.llm_gateway_configured) && llmGatewayReady,
      });
      setTasks([task]);
      setActiveTaskId(task.id);
      setMessages([]);
      setConfig((prev) =>
        prev
          ? {
              ...prev,
              default_skills: task.skills,
              default_mcp_servers: task.mcp_servers,
            }
          : prev,
      );
    } catch (err) {
      uiError("task:delete failed", err);
      setBootError(TASK_ERROR_MESSAGE);
    }
  }

  async function handleRagUploadComplete(message: string) {
    if (!activeTaskId) return;
    setMessages((prev) => [
      ...prev,
      buildRagUploadNotice(activeTaskId, message),
    ]);
  }

  async function dispatchSend(
    taskId: string,
    prompt: string,
    files: string[] = [],
  ) {
    const displayPrompt = buildDisplayPrompt(prompt, files);
    uiLog("chat:handleSend", { taskId, prompt: displayPrompt, files });
    if (activeTaskIdRef.current === taskId) {
      setMessages((prev) => [
        ...prev,
        buildOptimisticUserMessage(taskId, displayPrompt, files),
      ]);
    }
    setTasks((prev) => applyTaskTitleFromPrompt(prev, taskId, displayPrompt));

    try {
      await sendMessage(
        taskId,
        displayPrompt,
        async (final) => {
          if (activeTaskIdRef.current === taskId) {
            if (shouldAppendAssistantMessage(final)) {
              setMessages((prev) => [
                ...prev,
                buildPendingAssistantMessage(
                  taskId,
                  final!.content,
                  final!.images,
                  final!.tool_events,
                ),
              ]);
            }
            if (final?.stopped && shouldAppendAssistantMessage(final)) {
              try {
                // Belt-and-suspenders: ensure worker cancel even if stop raced.
                await appDataService.cancelTaskRun(taskId).catch(() => undefined);
                await appDataService.addMessage(taskId, {
                  role: "assistant",
                  content: final!.content,
                  images: final!.images,
                  tool_events: final!.tool_events,
                });
                await loadMessages(taskId);
              } catch (err) {
                uiError("chat:persist stop message failed", err);
              }
            } else if (!final?.stopped) {
              await loadMessages(taskId);
            }
          }
          await refreshTasks();
          if (final?.stopped) {
            const pending = pendingSteerRef.current;
            if (pending && pending.taskId === taskId) {
              pendingSteerRef.current = null;
              clearQueuePaused(taskId);
              await dispatchSend(pending.taskId, pending.text, pending.files);
              return;
            }
            if ((queuedByTaskIdRef.current[taskId] ?? []).length > 0) {
              setQueuePausedByTaskId((prev) => ({ ...prev, [taskId]: true }));
            }
            return;
          }
          clearQueuePaused(taskId);
          await drainQueue(taskId);
        },
        files,
      );
    } catch (err) {
      uiError("chat:send failed", err);
      setBootError(CHAT_ERROR_MESSAGE);
    }
  }

  async function drainQueue(taskId: string) {
    const queue = queuedByTaskIdRef.current[taskId] ?? [];
    if (queue.length === 0) return;

    const [next, ...rest] = queue;
    const updated = { ...queuedByTaskIdRef.current, [taskId]: rest };
    queuedByTaskIdRef.current = updated;
    setQueuedByTaskId(updated);

    uiLog("chat:queue drain", { taskId, remaining: rest.length });
    await dispatchSend(taskId, next.text, next.files);
  }

  async function handleSend(prompt: string, files: string[] = []) {
    if (!activeTaskId) {
      uiError("chat:send skipped — no active task");
      return;
    }

    const taskId = activeTaskId;
    if (getStreamForTask(taskId).streaming) {
      const item: QueuedMessage = {
        id: crypto.randomUUID(),
        text: prompt,
        files,
      };
      uiLog("chat:queue enqueue", { taskId, queueId: item.id });
      setQueuedByTaskId((prev) => {
        const next = {
          ...prev,
          [taskId]: [...(prev[taskId] ?? []), item],
        };
        queuedByTaskIdRef.current = next;
        return next;
      });
      return;
    }

    await dispatchSend(taskId, prompt, files);
  }

  async function handleNewTaskAndCloseSidebar() {
    await handleNewTask();
    setSidebarOpen(false);
  }

  if (!authReady) {
    return <div className="boot-loading">불러오는 중…</div>;
  }

  if (!userId) {
    return (
      <GoogleLoginModal
        clientId={config?.google_client_id ?? ""}
        onCredential={handleGoogleCredential}
        onAccessToken={handleGoogleAccessToken}
        onLocalUserId={handleLocalUserId}
        localAuthBypass={Boolean(config?.local_auth_bypass)}
        error={bootError}
        projectName={config?.projectName}
      />
    );
  }

  return (
    <div className={`app-shell${sidebarOpen ? " sidebar-open" : ""}`}>
      {sidebarOpen && (
        <button
          type="button"
          className="sidebar-backdrop"
          aria-label="메뉴 닫기"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <Sidebar
        userId={userId}
        tasks={tasks}
        activeTask={activeTask}
        config={config}
        drawer={drawer}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewTask={handleNewTaskAndCloseSidebar}
        onSelectTask={handleSelectTask}
        onOpenDrawer={setDrawer}
        onCloseDrawer={() => setDrawer(null)}
        onPatchTask={handlePatchTask}
        onDeleteTask={handleDeleteTask}
        onLogout={handleLogout}
        knowledgeGraphEnabled={knowledgeGraphEnabled}
        onPatchKnowledgeGraphEnabled={async (enabled) => {
          setKnowledgeGraphEnabled(enabled);
          try {
            const session = await api.patchSessionSettings({
              knowledge_graph_enabled: enabled,
            });
            setKnowledgeGraphEnabled(session.knowledge_graph_enabled ?? enabled);
          } catch (err) {
            setKnowledgeGraphEnabled(!enabled);
            uiError("knowledge graph setting failed", err);
            throw err;
          }
        }}
        onOpenDashboard={
          config?.is_admin
            ? () => {
                setView("dashboard");
                setSidebarOpen(false);
              }
            : undefined
        }
        onRefreshConfig={refreshConfig}
      />
      <div className="main-panel">
        {view === "dashboard" ? (
          <Dashboard onBack={() => setView("chat")} />
        ) : (
          <ChatThread
            messages={messages}
            streaming={showWaiting}
            streamText={activeStream.streamText}
            streamEvents={activeStream.streamEvents}
            taskTitle={activeTask?.title ?? "New task"}
            onMenuClick={() => setSidebarOpen(true)}
            footer={
              <ChatInput
                disabled={!activeTask}
                waiting={showWaiting}
                queuedMessages={activeQueuedMessages}
                queuePaused={activeQueuePaused}
                onRemoveQueued={handleRemoveQueued}
                onSteerQueued={handleSteerQueued}
                onResumeQueue={handleResumeQueue}
                onStop={handleStop}
                onSend={handleSend}
                onRagUploadComplete={handleRagUploadComplete}
              />
            }
          />
        )}
      </div>
    </div>
  );
}
