"""Task/message store with per-user SQLite DBs and a global legacy DB."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from application.task_store_persistence import (
    durable_user_db_path,
    flush_persist,
    restore_user_db,
    schedule_persist,
    working_db_path,
    working_user_db_path,
)
from application.utils import sanitize_user_path_segment

logger = logging.getLogger(__name__)

_APPLICATION_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_APPLICATION_DIR, "data")
_GLOBAL_DB_PATH = working_db_path()

DEFAULT_MODEL = "Claude 4.6 Sonnet"

_USER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT,
  runtime_session_id TEXT NOT NULL UNIQUE,
  model_name TEXT,
  skills_json TEXT,
  mcp_servers_json TEXT,
  guardrail_enabled INTEGER DEFAULT 0,
  llm_gateway_enabled INTEGER DEFAULT 0,
  memory_enabled INTEGER DEFAULT 1,
  pinned INTEGER DEFAULT 0,
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT,
  images_json TEXT,
  tool_events_json TEXT,
  created_at TEXT,
  FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_updated
  ON tasks(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_task_created
  ON messages(task_id, created_at ASC);
"""

_GLOBAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT,
  runtime_session_id TEXT NOT NULL UNIQUE,
  model_name TEXT,
  skills_json TEXT,
  mcp_servers_json TEXT,
  guardrail_enabled INTEGER DEFAULT 0,
  llm_gateway_enabled INTEGER DEFAULT 0,
  memory_enabled INTEGER DEFAULT 1,
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT,
  images_json TEXT,
  tool_events_json TEXT,
  created_at TEXT,
  FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_updated
  ON tasks(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_task_created
  ON messages(task_id, created_at ASC);

CREATE TABLE IF NOT EXISTS login_events (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  method TEXT NOT NULL,
  name TEXT,
  picture TEXT,
  logged_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_login_events_logged
  ON login_events(logged_at DESC);
CREATE INDEX IF NOT EXISTS idx_login_events_user
  ON login_events(user_id, logged_at DESC);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass
    return conn


def _connect_path(db_path: str) -> sqlite3.Connection:
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
    return _configure_connection(conn)


def _connect_global() -> sqlite3.Connection:
    global _GLOBAL_DB_PATH
    _GLOBAL_DB_PATH = working_db_path()
    return _connect_path(_GLOBAL_DB_PATH)


def _apply_task_column_migrations(conn: sqlite3.Connection) -> None:
    for stmt in (
        "ALTER TABLE tasks ADD COLUMN pinned INTEGER DEFAULT 0",
        "ALTER TABLE tasks ADD COLUMN memory_enabled INTEGER DEFAULT 1",
        "ALTER TABLE tasks ADD COLUMN llm_gateway_enabled INTEGER DEFAULT 0",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass


def _init_user_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_USER_SCHEMA_SQL)
    _apply_task_column_migrations(conn)


def _init_global_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_GLOBAL_SCHEMA_SQL)
    _apply_task_column_migrations(conn)


def init_db() -> None:
    """Initialize the global DB (login_events + legacy tables)."""
    global _GLOBAL_DB_PATH
    _GLOBAL_DB_PATH = working_db_path()
    with _connect_global() as conn:
        _init_global_schema(conn)


@contextmanager
def _user_db_lock(user_id: str) -> Iterator[None]:
    lock_path = working_user_db_path(user_id) + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _db_ready(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if "." in table:
        schema, name = table.split(".", 1)
        rows = conn.execute(f"PRAGMA {schema}.table_info({name})").fetchall()
    else:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in rows}


def _migrate_user_from_legacy(user_db_path: str, user_id: str) -> None:
    """Create user DB and copy this user's tasks/messages from legacy global DB."""
    legacy = working_db_path()
    parent = os.path.dirname(user_db_path)
    os.makedirs(parent, exist_ok=True)
    tmp_path = user_db_path + f".migrating-{os.getpid()}"
    for suffix in ("", "-wal", "-shm"):
        side = tmp_path + suffix
        if os.path.isfile(side):
            os.remove(side)

    task_rows: list[sqlite3.Row] = []
    message_rows: list[sqlite3.Row] = []
    task_cols: list[str] = []
    msg_cols: list[str] = []

    if _db_ready(legacy):
        with _connect_path(legacy) as legacy_conn:
            legacy_task_cols = _table_columns(legacy_conn, "tasks")
            legacy_msg_cols = _table_columns(legacy_conn, "messages")
            task_cols = [
                c
                for c in (
                    "id",
                    "user_id",
                    "title",
                    "runtime_session_id",
                    "model_name",
                    "skills_json",
                    "mcp_servers_json",
                    "guardrail_enabled",
                    "llm_gateway_enabled",
                    "memory_enabled",
                    "pinned",
                    "created_at",
                    "updated_at",
                )
                if c in legacy_task_cols
            ]
            msg_cols = [
                c
                for c in (
                    "id",
                    "task_id",
                    "role",
                    "content",
                    "images_json",
                    "tool_events_json",
                    "created_at",
                )
                if c in legacy_msg_cols
            ]
            if task_cols:
                col_csv = ", ".join(task_cols)
                task_rows = legacy_conn.execute(
                    f"SELECT {col_csv} FROM tasks WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            if msg_cols and task_rows:
                col_csv = ", ".join(msg_cols)
                message_rows = legacy_conn.execute(
                    f"""
                    SELECT {col_csv}
                    FROM messages
                    WHERE task_id IN (
                      SELECT id FROM tasks WHERE user_id = ?
                    )
                    """,
                    (user_id,),
                ).fetchall()

    conn = _connect_path(tmp_path)
    try:
        _init_user_schema(conn)
        dest_task_cols = _table_columns(conn, "tasks")
        dest_msg_cols = _table_columns(conn, "messages")
        task_cols = [c for c in task_cols if c in dest_task_cols]
        msg_cols = [c for c in msg_cols if c in dest_msg_cols]
        if task_cols and task_rows:
            col_csv = ", ".join(task_cols)
            placeholders = ", ".join("?" for _ in task_cols)
            conn.executemany(
                f"INSERT OR IGNORE INTO tasks ({col_csv}) VALUES ({placeholders})",
                [tuple(row[c] for c in task_cols) for row in task_rows],
            )
        if msg_cols and message_rows:
            col_csv = ", ".join(msg_cols)
            placeholders = ", ".join("?" for _ in msg_cols)
            conn.executemany(
                f"INSERT OR IGNORE INTO messages ({col_csv}) VALUES ({placeholders})",
                [tuple(row[c] for c in msg_cols) for row in message_rows],
            )
        conn.commit()
    finally:
        conn.close()

    os.replace(tmp_path, user_db_path)
    for suffix in ("-wal", "-shm"):
        src = tmp_path + suffix
        dst = user_db_path + suffix
        if os.path.isfile(src):
            os.replace(src, dst)
        elif os.path.isfile(dst):
            os.remove(dst)
    logger.info(
        "Migrated user tasks DB for %s -> %s (tasks=%s messages=%s)",
        user_id,
        user_db_path,
        len(task_rows),
        len(message_rows),
    )


def ensure_user_db(user_id: str) -> str:
    """Return working path for the user's tasks/messages DB, creating/migrating if needed."""
    if not sanitize_user_path_segment(user_id):
        raise ValueError(f"Invalid user_id: {user_id!r}")

    working = working_user_db_path(user_id)
    with _user_db_lock(user_id):
        if _db_ready(working):
            with _connect_path(working) as conn:
                _init_user_schema(conn)
            return working

        restored = False
        try:
            restored = restore_user_db(user_id)
        except Exception:
            logger.exception("Failed to restore user DB for %s", user_id)

        if restored and _db_ready(working):
            with _connect_path(working) as conn:
                _init_user_schema(conn)
            return working

        durable = durable_user_db_path(user_id)
        if _db_ready(durable) and not _db_ready(working):
            try:
                restore_user_db(user_id)
            except Exception:
                logger.exception("Failed durable→working restore for %s", user_id)
            if _db_ready(working):
                with _connect_path(working) as conn:
                    _init_user_schema(conn)
                return working

        _migrate_user_from_legacy(working, user_id)
        with _connect_path(working) as conn:
            _init_user_schema(conn)
        flush_persist(user_id)
        return working


def _connect_user(user_id: str) -> sqlite3.Connection:
    path = ensure_user_db(user_id)
    return _connect_path(path)


def _after_write(user_id: str | None = None) -> None:
    schedule_persist(user_id)


def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "title": row["title"] or "New task",
        # Checkpoints must be isolated per task; use task id as the stable fallback.
        "runtime_session_id": row["runtime_session_id"] or row["id"],
        "model_name": row["model_name"] or DEFAULT_MODEL,
        "skills": json.loads(row["skills_json"] or "[]"),
        "mcp_servers": json.loads(row["mcp_servers_json"] or "[]"),
        "guardrail_enabled": bool(row["guardrail_enabled"]),
        "llm_gateway_enabled": (
            bool(row["llm_gateway_enabled"]) if "llm_gateway_enabled" in keys else False
        ),
        "memory_enabled": bool(row["memory_enabled"]) if "memory_enabled" in keys else True,
        "pinned": bool(row["pinned"]) if "pinned" in keys else False,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "role": row["role"],
        "content": row["content"] or "",
        "images": json.loads(row["images_json"] or "[]"),
        "tool_events": json.loads(row["tool_events_json"] or "[]"),
        "created_at": row["created_at"],
    }


def list_tasks(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    with _connect_user(user_id) as conn:
        rows = conn.execute(
            """
            SELECT * FROM tasks
            WHERE user_id = ?
            ORDER BY pinned DESC, updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [_row_to_task(r) for r in rows]


def get_task(task_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    if not user_id:
        raise ValueError("user_id is required to resolve the per-user task DB")
    with _connect_user(user_id) as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ? AND user_id = ?",
            (task_id, user_id),
        ).fetchone()
    return _row_to_task(row) if row else None


def get_task_refreshing(task_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    """Return a task, reloading this user's durable DB once if missing locally."""
    if not user_id:
        return None
    task = get_task(task_id, user_id)
    if task:
        return task
    try:
        restore_user_db(user_id)
    except Exception:
        return None
    if not _db_ready(working_user_db_path(user_id)):
        return None
    return get_task(task_id, user_id)


def create_task(
    user_id: str,
    *,
    model_name: str | None = None,
    skills: list[str] | None = None,
    mcp_servers: list[str] | None = None,
    guardrail_enabled: bool = False,
    llm_gateway_enabled: bool = False,
    memory_enabled: bool = True,
    title: str = "New task",
) -> dict[str, Any]:
    task_id = str(uuid.uuid4())
    # Keep the checkpoint namespace aligned with the task identity.
    runtime_session_id = task_id
    now = _now_iso()
    with _connect_user(user_id) as conn:
        conn.execute(
            """
            INSERT INTO tasks (
              id, user_id, title, runtime_session_id, model_name,
              skills_json, mcp_servers_json, guardrail_enabled, llm_gateway_enabled,
              memory_enabled, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                user_id,
                title,
                runtime_session_id,
                model_name or DEFAULT_MODEL,
                json.dumps(skills or [], ensure_ascii=False),
                json.dumps(mcp_servers or [], ensure_ascii=False),
                1 if guardrail_enabled else 0,
                1 if llm_gateway_enabled else 0,
                1 if memory_enabled else 0,
                now,
                now,
            ),
        )
    # Flush immediately so sibling ECS tasks / replacements can see the row
    # (debounced persist alone loses creates during rolling deploys).
    flush_persist(user_id)
    return get_task(task_id, user_id)  # type: ignore[return-value]


def update_task(task_id: str, user_id: str, **fields: Any) -> dict[str, Any] | None:
    allowed = {
        "title": "title",
        "model_name": "model_name",
        "guardrail_enabled": "guardrail_enabled",
        "llm_gateway_enabled": "llm_gateway_enabled",
        "memory_enabled": "memory_enabled",
        "pinned": "pinned",
    }
    sets: list[str] = []
    values: list[Any] = []

    for key, column in allowed.items():
        if key in fields and fields[key] is not None:
            value = fields[key]
            if key in (
                "guardrail_enabled",
                "llm_gateway_enabled",
                "memory_enabled",
                "pinned",
            ):
                value = 1 if value else 0
            sets.append(f"{column} = ?")
            values.append(value)

    if "skills" in fields and fields["skills"] is not None:
        sets.append("skills_json = ?")
        values.append(json.dumps(fields["skills"], ensure_ascii=False))

    if "mcp_servers" in fields and fields["mcp_servers"] is not None:
        sets.append("mcp_servers_json = ?")
        values.append(json.dumps(fields["mcp_servers"], ensure_ascii=False))

    if not sets:
        return get_task(task_id, user_id)

    sets.append("updated_at = ?")
    values.append(_now_iso())
    values.extend([task_id, user_id])

    with _connect_user(user_id) as conn:
        conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE id = ? AND user_id = ?",
            values,
        )
    _after_write(user_id)
    return get_task(task_id, user_id)


def delete_task(task_id: str, user_id: str) -> bool:
    with _connect_user(user_id) as conn:
        conn.execute("DELETE FROM messages WHERE task_id = ?", (task_id,))
        cur = conn.execute(
            "DELETE FROM tasks WHERE id = ? AND user_id = ?",
            (task_id, user_id),
        )
    if cur.rowcount > 0:
        _after_write(user_id)
    return cur.rowcount > 0


def list_messages(task_id: str, user_id: str) -> list[dict[str, Any]]:
    with _connect_user(user_id) as conn:
        rows = conn.execute(
            """
            SELECT * FROM messages
            WHERE task_id = ?
            ORDER BY created_at ASC
            """,
            (task_id,),
        ).fetchall()
    return [_row_to_message(r) for r in rows]


def add_message(
    task_id: str,
    role: str,
    content: str,
    *,
    user_id: str,
    images: list[str] | None = None,
    tool_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message_id = str(uuid.uuid4())
    now = _now_iso()
    with _connect_user(user_id) as conn:
        conn.execute(
            """
            INSERT INTO messages (
              id, task_id, role, content, images_json, tool_events_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                task_id,
                role,
                content,
                json.dumps(images or [], ensure_ascii=False),
                json.dumps(tool_events or [], ensure_ascii=False),
                now,
            ),
        )
        title_update = None
        if role == "user":
            row = conn.execute(
                "SELECT title FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row and (row["title"] or "New task") in ("New task", ""):
                title_update = content.strip()[:50] or "New task"

        conn.execute(
            "UPDATE tasks SET updated_at = ?"
            + (", title = ?" if title_update else "")
            + " WHERE id = ?",
            ([now, title_update, task_id] if title_update else [now, task_id]),
        )
    _after_write(user_id)
    with _connect_user(user_id) as conn:
        row = conn.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
    return _row_to_message(row) if row else {}


def update_message_tool_events(
    message_id: str,
    task_id: str,
    user_id: str,
    tool_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Backfill tool_events on an existing assistant message (refresh recovery)."""
    if not message_id or not tool_events:
        return None
    with _connect_user(user_id) as conn:
        row = conn.execute(
            "SELECT * FROM messages WHERE id = ? AND task_id = ?",
            (message_id, task_id),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            """
            UPDATE messages
            SET tool_events_json = ?
            WHERE id = ? AND task_id = ?
            """,
            (
                json.dumps(tool_events, ensure_ascii=False),
                message_id,
                task_id,
            ),
        )
        conn.execute(
            "UPDATE tasks SET updated_at = ? WHERE id = ?",
            (_now_iso(), task_id),
        )
        updated = conn.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
    _after_write(user_id)
    return _row_to_message(updated) if updated else None
