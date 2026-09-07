import logging
import sys
import traceback
import asyncio
try:
    from application import chat
    from application import run_cancel
except ImportError:  # script-style runs with application/ on sys.path
    import chat
    import run_cancel
import utils
import agentcore_sigv4_auth
import sys
import subprocess

from langgraph.prebuilt import ToolNode
from typing import Literal
from langgraph.graph import START, END, StateGraph
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langgraph.prebuilt import ToolNode
from typing import Literal
from langgraph.graph import START, END, StateGraph
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

logging.basicConfig(
    level=logging.INFO,  
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("langgraph_agent")

config = utils.load_config()
sharing_url = config["sharing_url"] if "sharing_url" in config else None
s3_prefix = "docs"

import io, os, sys, json, traceback
import subprocess as _subprocess, pathlib as _pathlib, shutil as _shutil
import tempfile as _tempfile, glob as _glob, datetime as _datetime
import math as _math, re as _re, requests as _requests
from urllib.parse import quote
from langchain_core.tools import tool

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
# Per-user artifacts/skills under SESSION_STORAGE_DIR (set via set_user_workspace).
ARTIFACTS_DIR = utils.get_user_artifacts_dir("default")
USER_SKILLS_DIR = utils.get_user_skills_dir("default")

_py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
_user_bin = os.path.expanduser(f"~/Library/Python/{_py_ver}/bin")
if os.path.isdir(_user_bin) and _user_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _user_bin + os.pathsep + os.environ.get("PATH", "")

ARTIFACT_EXT = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".js",  # e.g. generated scripts; still offer download when created
})

_mpl_runtime_ready = False

_EXCLUDED_SNAPSHOT_DIRS = frozenset({
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "site-packages",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".session_storage",  # scan only the active user's ARTIFACTS_DIR below
})


def set_user_artifacts(user_id: str | None) -> str:
    """Point ARTIFACTS_DIR at {SESSION_STORAGE_DIR}/{user_id}/artifacts."""
    global ARTIFACTS_DIR
    artifacts_dir = utils.ensure_user_artifacts_dir(user_id)
    ARTIFACTS_DIR = artifacts_dir
    exec_globals = globals().get("_exec_globals")
    if isinstance(exec_globals, dict):
        exec_globals["ARTIFACTS_DIR"] = artifacts_dir
        if "USER_SKILLS_DIR" in globals():
            exec_globals["USER_SKILLS_DIR"] = USER_SKILLS_DIR
    logger.info(f"ARTIFACTS_DIR set for user {user_id!r}: {artifacts_dir}")
    return artifacts_dir


def set_user_skills(user_id: str | None) -> str:
    """Point USER_SKILLS_DIR and ensure per-user skills.list exists."""
    global USER_SKILLS_DIR
    skills_dir = utils.ensure_user_skills_dir(user_id)
    USER_SKILLS_DIR = skills_dir
    utils.ensure_user_skills_list(user_id)
    exec_globals = globals().get("_exec_globals")
    if isinstance(exec_globals, dict):
        exec_globals["USER_SKILLS_DIR"] = skills_dir
    logger.info(f"USER_SKILLS_DIR set for user {user_id!r}: {skills_dir}")
    return skills_dir


def set_user_workspace(user_id: str | None) -> tuple[str, str]:
    """Configure per-user artifacts + skills dirs; create skills.list if missing."""
    artifacts_dir = set_user_artifacts(user_id)
    skills_dir = set_user_skills(user_id)
    return artifacts_dir, skills_dir


def _expand_user_skills_token(raw: str) -> str:
    """Expand $USER_SKILLS_DIR / ${USER_SKILLS_DIR} using the current workspace path."""
    if not USER_SKILLS_DIR or "$" not in raw:
        return raw
    expanded = raw
    for token in ("${USER_SKILLS_DIR}", "$USER_SKILLS_DIR", "${user_skills_dir}", "$user_skills_dir"):
        expanded = expanded.replace(token, USER_SKILLS_DIR)
    return expanded


def _path_is_under(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        norm_path = os.path.normpath(path)
        norm_root = os.path.normpath(root)
        return os.path.commonpath([norm_path, norm_root]) == norm_root
    except ValueError:
        return False


def _resolve_workdir_path(filepath: str) -> str:
    """Resolve filepath; map relative artifacts/ onto ARTIFACTS_DIR; allow USER_SKILLS_DIR."""
    if not filepath:
        return filepath
    filepath = _expand_user_skills_token(filepath)
    if os.path.isabs(filepath):
        return filepath
    normalized = filepath.replace("\\", "/").lstrip("./")
    if normalized == "artifacts" or normalized.startswith("artifacts/"):
        suffix = normalized[len("artifacts") :].lstrip("/")
        return os.path.join(ARTIFACTS_DIR, suffix) if suffix else ARTIFACTS_DIR
    return os.path.join(WORKING_DIR, filepath)


def _working_dir_files_mtime_snapshot() -> dict:
    """Relative path -> mtime for files under WORKING_DIR (vendor/cache dirs excluded).

    Code often writes under artifacts/ but may also write to the working dir root;
    scanning only artifacts/ missed those files and left download lists empty.
    Other users' session folders are excluded; the active ARTIFACTS_DIR is scanned.
    """
    snap = {}
    if not os.path.isdir(WORKING_DIR):
        return snap
    for dirpath, dirnames, filenames in os.walk(WORKING_DIR):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_SNAPSHOT_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                rel = os.path.relpath(full, WORKING_DIR)
                snap[rel] = os.path.getmtime(full)
            except OSError:
                pass
    if os.path.isdir(ARTIFACTS_DIR):
        for dirpath, dirnames, filenames in os.walk(ARTIFACTS_DIR):
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_SNAPSHOT_DIRS]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                try:
                    try:
                        rel = os.path.relpath(full, WORKING_DIR)
                    except ValueError:
                        rel = full
                    snap[rel] = os.path.getmtime(full)
                except OSError:
                    pass
    return snap


def _ensure_node_path():
    """Expose /app/node_modules to Node require() for bash and execute_code."""
    node_modules = os.path.join(WORKING_DIR, "node_modules")
    if not os.path.isdir(node_modules):
        return
    existing = os.environ.get("NODE_PATH", "")
    if node_modules not in existing.split(os.pathsep):
        os.environ["NODE_PATH"] = (
            f"{node_modules}{os.pathsep}{existing}" if existing else node_modules
        )


def _ensure_cli_scripts_on_path() -> None:
    """Prepend pip user script dir so CLIs (e.g. browser-use) resolve in subprocess."""
    import site
    import sysconfig

    extra: list[str] = []
    user_base = getattr(site, "USER_BASE", None)
    if user_base:
        user_bin = os.path.join(user_base, "bin")
        if os.path.isdir(user_bin):
            extra.append(user_bin)
    try:
        scripts = sysconfig.get_path("scripts")
        if scripts and os.path.isdir(scripts):
            extra.append(scripts)
    except Exception:
        pass
    path = os.environ.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p]
    for d in reversed(extra):
        if d and d not in parts:
            parts.insert(0, d)
    os.environ["PATH"] = os.pathsep.join(parts)


def _ensure_user_site_on_sys_path() -> None:
    """Expose pip --user site-packages to this process (and PYTHONPATH for children).

    Runtime runs as appuser, so `pip install` lands under ~/.local. site.py only
    adds USER_SITE at interpreter startup if that directory already exists; packages
    installed later (bash/execute_code) stay invisible until we addsitedir here.
    """
    import site

    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if not user_site or not os.path.isdir(user_site):
        return

    # addsitedir also honors .pth files; skip if already on path
    if user_site not in sys.path:
        site.addsitedir(user_site)

    py_path = os.environ.get("PYTHONPATH", "")
    parts = [p for p in py_path.split(os.pathsep) if p]
    if user_site not in parts:
        parts.insert(0, user_site)
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)


class _UserSiteRefreshFinder:
    """Refresh USER_SITE on import so mid-exec `pip install` + `import` works."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        import site

        try:
            user_site = site.getusersitepackages()
        except Exception:
            return None
        if user_site and os.path.isdir(user_site) and user_site not in sys.path:
            _ensure_user_site_on_sys_path()
        return None


def _install_user_site_import_hook() -> None:
    if any(isinstance(f, _UserSiteRefreshFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _UserSiteRefreshFinder())


_install_user_site_import_hook()


def _touched_artifact_paths(before: dict, after: dict) -> list:
    """Return files that were newly created or modified between two snapshots."""
    touched = []
    for rel, mt in after.items():
        if rel not in before or before[rel] != mt:
            touched.append(rel)
    return sorted(touched)


def _paths_for_ui(relative_paths: list) -> list:
    """Return public URLs if sharing_url is set, otherwise absolute local paths."""
    out = []
    base = sharing_url.rstrip("/") if sharing_url else ""
    for rel in relative_paths:
        if base:
            out.append(f"{base}/{quote(rel)}")
        else:
            out.append(os.path.abspath(os.path.join(WORKING_DIR, rel)))
    return out

def _ensure_matplotlib_runtime():
    """Use non-interactive Agg backend, prefer CJK-capable fonts, silence headless/show noise."""
    global _mpl_runtime_ready
    if _mpl_runtime_ready:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")

        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r"Glyph .* missing from font",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"FigureCanvasAgg is non-interactive.*",
            category=UserWarning,
        )

        import matplotlib.font_manager as fm
        import matplotlib as mpl

        mpl.rcParams["axes.unicode_minus"] = False
        cjk_candidates = (
            "AppleGothic",
            "Apple SD Gothic Neo",
            "Malgun Gothic",
            "NanumGothic",
            "NanumBarunGothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
        )
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = list(cjk_candidates) + ["DejaVu Sans", "sans-serif"]

        _mpl_runtime_ready = True
    except Exception as e:
        logger.info(f"matplotlib runtime setup skipped: {e}")
        _mpl_runtime_ready = True


def register_korean_font() -> str:
    """Register a Korean-capable font for ReportLab (execute_code tool).

    Prefer ``WORKING_DIR/assets/NanumGothic-Regular.ttf``, then common system paths,
    then built-in CID ``HYGothic-Medium``. Returns the font name to pass as
    ``fontName`` / ``bulletFontName`` on ParagraphStyle and table styles.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    except ImportError:
        return "Helvetica"

    ttf_candidates = [
        os.path.join(WORKING_DIR, "assets", "NanumGothic-Regular.ttf"),
        os.path.join("assets", "NanumGothic-Regular.ttf"),
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "/Library/Fonts/NanumGothic.ttf",
    ]
    for path in ttf_candidates:
        if not os.path.isfile(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont("KoreanFont", path))
            return "KoreanFont"
        except Exception:
            continue

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
        return "HYGothic-Medium"
    except Exception:
        pass

    return "Helvetica"


_exec_globals = {
    "__builtins__": __builtins__,
    "subprocess": _subprocess,
    "json": json,
    "os": os,
    "sys": sys,
    "io": io,
    "pathlib": _pathlib,
    "shutil": _shutil,
    "tempfile": _tempfile,
    "glob": _glob,
    "datetime": _datetime,
    "math": _math,
    "re": _re,
    "requests": _requests,
    "WORKING_DIR": WORKING_DIR,
    "ARTIFACTS_DIR": ARTIFACTS_DIR,  # updated by set_user_workspace()
    "USER_SKILLS_DIR": USER_SKILLS_DIR,  # updated by set_user_workspace()
    "register_korean_font": register_korean_font,
}

import datetime
from pytz import timezone

_KOREAN_WEEKDAYS = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")

@tool
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in Asia/Seoul, including the Korean weekday.

    Example: "2026-08-08 15:51:06 (토요일)"
    """
    format = format.replace('\'','')
    now = datetime.datetime.now(timezone('Asia/Seoul'))
    timestr = f"{now.strftime(format)} ({_KOREAN_WEEKDAYS[now.weekday()]})"
    logger.info(f"timestr: {timestr}")
    
    return timestr

@tool
def execute_code(code: str) -> str:
    """Execute Python code and return stdout/stderr output.

    Use this tool to run Python code for tasks such as processing data,
    processing data, or performing computations. The execution environment
    has access to common libraries: pandas, numpy, matplotlib, seaborn, etc.
    json, csv, os, requests, etc.

    Variables and imports from previous calls persist across invocations.
    Generated files should be saved under ARTIFACTS_DIR (per-user workspace).

    Document types (do not confuse extensions):
    - Word / 한글 보고서 산출물 → 반드시 '.docx' (권장: Python python-docx). '.js'는 자바스크립트 소스용이며 Word 본문 보고서 파일명으로 쓰지 마세요.
    - PDF → '.pdf', Excel → '.xlsx' 등 실제 형식에 맞는 확장자를 사용하세요.

    Path variables (pre-defined, do NOT redefine):
    - WORKING_DIR: absolute path to application directory
    - ARTIFACTS_DIR: absolute path to this user's artifacts ({SESSION_STORAGE_DIR}/{user_id}/artifacts)
    - USER_SKILLS_DIR: absolute path to this user's skills ({SESSION_STORAGE_DIR}/{user_id}/skills)
    - register_korean_font(): registers Nanum TTF or CID fallback for ReportLab; returns font name str

    Args:
        code: Python code to execute.

    Returns:
        Captured stdout output, or error traceback if execution failed.
        If there is a result file, return the path of the file.            
    """
    logger.info(f"###### execute_code ######")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    _exec_globals["ARTIFACTS_DIR"] = ARTIFACTS_DIR
    _exec_globals["USER_SKILLS_DIR"] = USER_SKILLS_DIR
    before_files = _working_dir_files_mtime_snapshot()

    old_cwd = os.getcwd()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        os.chdir(WORKING_DIR)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

        _ensure_cli_scripts_on_path()
        _ensure_user_site_on_sys_path()
        _ensure_matplotlib_runtime()
        _ensure_node_path()
        
        exec(code, _exec_globals)

        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        result = ""
        if output:
            result += output
        if errors:
            result += f"\n[stderr]\n{errors}"
        if not result.strip():
            result = "Code executed successfully (no output)."

        after_files = _working_dir_files_mtime_snapshot()
        touched = _touched_artifact_paths(before_files, after_files)
        artifact_rels = [
            r
            for r in touched
            if os.path.splitext(r)[1].lower() in ARTIFACT_EXT
        ]
        other_rels = [r for r in touched if r not in artifact_rels]
        if other_rels:
            lines = "\n".join(
                os.path.abspath(os.path.join(WORKING_DIR, r)) for r in other_rels
            )
            result += f"\n[artifacts]\n{lines}"

        if artifact_rels:
            payload = {"output": result.strip()}
            payload["path"] = _paths_for_ui(artifact_rels)
            return json.dumps(payload, ensure_ascii=False)

        return result

    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)
        tb = traceback.format_exc()
        logger.error(f"Code execution error: {tb}")
        return f"Error executing code:\n{tb}"

@tool
def write_file(filepath: str, content: str = "") -> str:
    """Write text content to a file.

    CRITICAL: content must always be passed. Calling without content will fail.
    Never call without content. Both filepath and content are required in a single call.

    Args:
        filepath: Absolute path or path relative to WORKING_DIR. Use the real file extension
            (e.g. '.docx' for Word, '.md' for Markdown). Do not save report bodies as '.js'.
        content: The text content to write. REQUIRED - must not be omitted. Must include full file content.

    Returns:
        A success or failure message.
    """
    if not content:
        return (
            "Error: content parameter is required. "
            "Pass the full content to save in the form write_file(filepath='path', content='content_to_save')."
        )
    logger.info(f"###### write_file: {filepath} ######")
    try:
        full_path = _resolve_workdir_path(filepath)
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        rel = os.path.relpath(full_path, WORKING_DIR)
        result_msg = f"File saved: {filepath}"
        payload = {"output": result_msg, "path": _paths_for_ui([rel])}
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return f"Failed to save file: {str(e)}"


@tool
def read_file(filepath: str) -> str:
    """Read the contents of a local file.

    Args:
        filepath: Absolute path or path relative to WORKING_DIR.
            Relative paths under artifacts/ map to the active user's ARTIFACTS_DIR.

    Returns:
        The file contents as text, or an error message.
    """
    logger.info(f"###### read_file: {filepath} ######")
    try:
        full_path = _resolve_workdir_path(filepath)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {str(e)}"


@tool
def upload_file_to_s3(filepath: str) -> str:
    """Upload a local file to S3 and return the download URL.

    Args:
        filepath: Absolute path or path relative to WORKING_DIR
            (e.g. 'artifacts/report.pdf' → active user's ARTIFACTS_DIR).

    Returns:
        The download URL, or an error message.
    """
    logger.info(f"###### upload_file_to_s3: {filepath} ######")
    try:
        import boto3
        from urllib import parse as url_parse

        s3_bucket = config.get("s3_bucket")
        if not s3_bucket:
            return "S3 bucket is not configured."

        full_path = _resolve_workdir_path(filepath)
        if not os.path.exists(full_path):
            return f"File not found: {filepath}"

        s3_key = os.path.relpath(full_path, WORKING_DIR).replace("\\", "/")
        content_type = utils.get_contents_type(s3_key)
        s3 = boto3.client("s3", region_name=config.get("region", "us-west-2"))

        with open(full_path, "rb") as f:
            s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=f.read(), ContentType=content_type)

        if sharing_url:
            url = f"{sharing_url}/{url_parse.quote(s3_key)}"
            return f"Upload complete: {url}"
        return f"Upload complete: {chat.s3_uri_to_console_url(f"s3://{s3_bucket}/{s3_key}", config.get("region", "us-west-2"))}"

    except Exception as e:
        return f"Upload failed: {str(e)}"

@tool
def bash(command: str) -> str:
    """Execute a bash command and return the result"""
    logger.info(f"###### bash: {command} ######")
    _ensure_cli_scripts_on_path()
    _ensure_user_site_on_sys_path()
    _ensure_node_path()
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        cwd=WORKING_DIR, timeout=300,
        env=os.environ,
    )
    # pip install may have just created ~/.local/.../site-packages
    _ensure_user_site_on_sys_path()
    parts = []
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if result.returncode != 0:
        parts.append(f"Return code: {result.returncode}")
    return "\n".join(parts) if parts else "(no output)"

def get_builtin_tools() -> list:
    """Return the list of built-in tools for the skill-aware agent."""

    if sharing_url:
        return [execute_code, write_file, read_file, bash, upload_file_to_s3, get_current_time]
    else:
        return [execute_code, write_file, read_file, bash, get_current_time]

def _assistant_text_content(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content) if content else ""


def sanitize_messages_for_bedrock(messages: list) -> list:
    """Bedrock requires every assistant tool_use to be followed by tool_result for each id.

    Checkpoint/history can contain AIMessage(tool_calls) without matching ToolMessage
    (e.g. interrupted turn). Strip broken tool rounds and drop orphan tool results.
    """
    msgs = list(messages)
    out: list = []
    i = 0
    n = len(msgs)
    while i < n:
        msg = msgs[i]
        if isinstance(msg, ToolMessage):
            logger.warning(
                "Bedrock compatibility: dropping orphan ToolMessage (no preceding tool_use)"
            )
            i += 1
            continue
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            needed = {tc["id"] for tc in msg.tool_calls}
            tool_msgs: list = []
            j = i + 1
            while j < n and isinstance(msgs[j], ToolMessage):
                tool_msgs.append(msgs[j])
                j += 1
            got = {tm.tool_call_id for tm in tool_msgs}
            if needed <= got:
                out.append(msg)
                out.extend(tool_msgs)
                i = j
                continue
            logger.warning(
                "Bedrock compatibility: stripping tool_calls (expected ids %s, got %s)",
                needed,
                got,
            )
            text = _assistant_text_content(msg)
            if text.strip():
                out.append(AIMessage(content=text))
            i = j
            continue
        out.append(msg)
        i += 1
    return out


def message_chunk_to_message(chunk: BaseMessage) -> BaseMessage:
    """Convert a message chunk to a `Message`.

    Args:
        chunk: Message chunk to convert.

    Returns:
        Message.
    """
    if not isinstance(chunk, BaseMessageChunk):
        return chunk
    # chunk classes always have the equivalent non-chunk class as their first parent
    ignore_keys = ["type"]
    if isinstance(chunk, AIMessageChunk):
        ignore_keys.extend(["tool_call_chunks", "chunk_position"])
    return chunk.__class__.__mro__[1](
        **{k: v for k, v in chunk.__dict__.items() if k not in ignore_keys}
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    artifacts: list

BASE_SYSTEM_PROMPT = (
    "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다.\n"
    "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다.\n"
    "모르는 질문을 받으면 솔직히 모른다고 말합니다.\n"
    "한국어로 답변하세요.\n"
)

MAX_CONTEXT_TURNS = 5

# Bedrock Anthropic/Nova prompt caching (ephemeral, 1h TTL).
PROMPT_CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}


def _supports_prompt_caching(model_type: str | None) -> bool:
    # Bedrock-specific cache_control; skip when routing via LiteLLM gateway.
    if getattr(chat, "llm_gateway_enabled", False):
        return False
    return model_type in ("claude", "nova")


def _system_message_with_cache(system: str) -> SystemMessage:
    """Build a SystemMessage for Bedrock prompt caching.

    Do **not** embed ``cache_control`` on the system content block.

    ``ChatBedrock`` strips system ``cache_control`` ttl to the ``5m``
    default, which clashes with ``model.bind(cache_control=… ttl=1h)``
    on the last message and raises ``ValidationException``. Let bind
    alone own the 1h breakpoint(s).
    """
    return SystemMessage(content=system)


def _log_prompt_cache_usage(response: AIMessage) -> None:
    """Log cache_read / cache_creation from usage_metadata when present."""
    usage = getattr(response, "usage_metadata", None) or {}
    details = usage.get("input_token_details") if isinstance(usage, dict) else None
    if not isinstance(details, dict):
        return
    cache_read = details.get("cache_read") or 0
    cache_creation = details.get("cache_creation") or 0
    if cache_read or cache_creation:
        logger.info(
            "prompt cache usage: cache_read=%s cache_creation=%s",
            cache_read,
            cache_creation,
        )


def trim_messages_by_human_turns(messages: list, max_turns: int) -> list:
    """Keep messages from the last N HumanMessage turns (inclusive)."""
    if max_turns <= 0 or not messages:
        return messages

    human_indices = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
    if len(human_indices) <= max_turns:
        return messages

    return messages[human_indices[-max_turns]:]


async def call_model(state: State, config):
    logger.info(f"###### call_model ######")

    cancel_id = _cancel_id_from_config(config)
    if run_cancel.is_cancelled(cancel_id):
        logger.info(f"call_model skipped: cancelled ({cancel_id})")
        artifacts = state["artifacts"] if "artifacts" in state else []
        return {
            "messages": [AIMessage(content="")],
            "artifacts": artifacts,
        }

    last_message = state['messages'][-1]
    logger.info(f"last message: {last_message}")
    
    artifacts = state['artifacts'] if 'artifacts' in state else []

    cfg = config.get("configurable") or {}
    tools = cfg.get("tools") 
    system = cfg.get("system_prompt") 
    if system is None:
        system = BASE_SYSTEM_PROMPT

    # Capture model id before concurrent requests mutate the shared chat module.
    active_model_id = chat.model_id
    active_model_type = chat.model_type
    chatModel = chat.get_chat()

    model = chatModel.bind_tools(tools) if tools else chatModel
    use_prompt_cache = _supports_prompt_caching(active_model_type)
    if use_prompt_cache:
        # ChatBedrock: marks last message; ChatBedrockConverse: system+tools+last.
        model = model.bind(cache_control=PROMPT_CACHE_CONTROL)

    try:
        raw = state["messages"]
        messages = []
        for msg in sanitize_messages_for_bedrock(raw):
            if isinstance(msg, ToolMessage):
                content = msg.content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            # Remove 'id' field if present, but keep other fields
                            item_clean = {k: v for k, v in item.items() if k != 'id'}
                            if 'text' in item_clean:
                                text_parts.append(item_clean['text'])
                            elif 'content' in item_clean:
                                text_parts.append(str(item_clean['content']))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = '\n'.join(text_parts) if text_parts else str(content)
                elif not isinstance(content, str):
                    content = str(content)
                
                # Create ToolMessage without 'name' field (Bedrock doesn't accept it)
                tool_msg = ToolMessage(
                    content=content,
                    tool_call_id=msg.tool_call_id
                )
                messages.append(tool_msg)
            else:
                messages.append(msg)

        max_turns = (
            config.get("configurable", {}).get("max_turns")
            or config.get("max_turns")
            or MAX_CONTEXT_TURNS
        )
        trimmed = trim_messages_by_human_turns(messages, max_turns)
        if len(trimmed) < len(messages):
            logger.info(
                f"trimmed messages from {len(messages)} to {len(trimmed)} "
                f"(max_turns={max_turns})"
            )
            messages = trimmed

        # Strip thinking/reasoning blocks before Bedrock Claude/Nova (GPT history
        # leaves type='reasoning'; which Bedrock rejects with ValidationException).
        if active_model_type in ("claude", "nova") or chat.uses_adaptive_thinking(
            active_model_id
        ):
            messages = chat.sanitize_adaptive_thinking_messages(messages)

        if use_prompt_cache:
            system_msg = _system_message_with_cache(system)
        else:
            system_msg = SystemMessage(content=system)
        model_messages = [system_msg, *messages]

        # Stream tokens/chunks to the graph via astream (use with stream_mode="messages")
        accumulated: AIMessageChunk | None = None
        async for chunk in model.astream(model_messages):
            if run_cancel.is_cancelled(cancel_id):
                logger.info(f"call_model stream aborted: cancelled ({cancel_id})")
                break
            if accumulated is None:
                accumulated = chunk
            else:
                accumulated = accumulated + chunk

        if run_cancel.is_cancelled(cancel_id) and accumulated is None:
            response = AIMessage(content="")
        elif accumulated is None:
            response = AIMessage(content="답변을 찾지 못하였습니다.")
        else:
            merged = message_chunk_to_message(accumulated)
            response = merged if isinstance(merged, AIMessage) else AIMessage(
                content=getattr(merged, "content", str(merged))
            )
            if run_cancel.is_cancelled(cancel_id):
                # Drop pending tool_calls so the graph can end cleanly.
                response = AIMessage(content=_assistant_text_content(response) or "")
        if active_model_type in ("claude", "nova") or chat.uses_adaptive_thinking(
            active_model_id
        ):
            response = chat.sanitize_adaptive_thinking_messages([response])[0]
        logger.info(f"response of call_model: {response}")
        _log_prompt_cache_usage(response)

    except Exception:
        response = AIMessage(content="답변을 찾지 못하였습니다.")

        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")

    return {"messages": [response], "artifacts": artifacts}

async def should_continue(state: State, config) -> Literal["continue", "end"]:
    logger.info(f"###### should_continue ######")

    cancel_id = _cancel_id_from_config(config)
    if run_cancel.is_cancelled(cancel_id):
        # Route to action so pending tool_calls get cancelled ToolMessages
        # (keeps checkpoint valid for the next user turn on the same thread).
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            logger.info("--- CANCEL → action (heal tool_calls) ---")
            return "continue"
        logger.info("--- CANCEL → end ---")
        return "end"

    messages = state["messages"]    
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_name = last_message.tool_calls[-1]['name']
        logger.info(f"--- CONTINUE: {tool_name} ---")

        tool_args = last_message.tool_calls[-1]['args']

        if last_message.content:
            logger.info(f"last_message: {last_message.content}")

        logger.info(f"tool_name: {tool_name}, tool_args: {tool_args}")

        return "continue"
    else:
        logger.info(f"--- END ---")
        return "end"


def _cancel_id_from_config(config) -> str | None:
    cfg = (config or {}).get("configurable") or {}
    return cfg.get("thread_id")


def _cancelled_tool_messages(state: State) -> list[ToolMessage]:
    messages = state.get("messages") or []
    if not messages:
        return []
    last = messages[-1]
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return []
    out: list[ToolMessage] = []
    for tc in last.tool_calls:
        tid = tc.get("id") if isinstance(tc, dict) else None
        if tid:
            out.append(
                ToolMessage(
                    content="Cancelled by user",
                    tool_call_id=tid,
                )
            )
    return out


async def _action_node(state: State, config, tool_node: ToolNode):
    cancel_id = _cancel_id_from_config(config)
    if run_cancel.is_cancelled(cancel_id):
        cancelled_msgs = _cancelled_tool_messages(state)
        if cancelled_msgs:
            logger.info(
                "action skipped: cancelled — emitting %s ToolMessage(s)",
                len(cancelled_msgs),
            )
            return {"messages": cancelled_msgs}
        logger.info("action skipped: cancelled (no pending tool_calls)")
        return {}

    # Poll cancel while tools run so Stop doesn't wait for execute_code/etc.
    invoke_task = asyncio.create_task(tool_node.ainvoke(state, config))
    try:
        while True:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(invoke_task), timeout=0.25
                )
            except asyncio.TimeoutError:
                if run_cancel.is_cancelled(cancel_id):
                    logger.info(
                        "action aborted mid-tool: cancelled (%s)", cancel_id
                    )
                    invoke_task.cancel()
                    try:
                        await invoke_task
                    except asyncio.CancelledError:
                        pass
                    cancelled_msgs = _cancelled_tool_messages(state)
                    if cancelled_msgs:
                        return {"messages": cancelled_msgs}
                    return {}
    except asyncio.CancelledError:
        invoke_task.cancel()
        raise


def buildChatAgent(tools):
    tool_node = ToolNode(tools, handle_tool_errors=True)

    async def action(state: State, config):
        return await _action_node(state, config, tool_node)

    workflow = StateGraph(State)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", action)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")

    return workflow.compile() 

def buildChatAgentWithHistory(tools):
    tool_node = ToolNode(tools, handle_tool_errors=True)

    async def action(state: State, config):
        return await _action_node(state, config, tool_node)

    workflow = StateGraph(State)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", action)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")

    return workflow.compile(checkpointer=chat.checkpointer)

def load_multiple_mcp_server_parameters(mcp_json: dict):
    """Build per-server configs compatible with langchain.mcp.MCPAdapter / MCPConfig."""
    mcpServers = mcp_json.get("mcpServers")
  
    server_info = {}
    if mcpServers is not None:
        for server_name, config in mcpServers.items():
            if config.get("type") in ("streamable_http", "http", "streamable-http"):
                connection = {
                    "transport": "http",
                    "url": config.get("url"),
                    "headers": config.get("headers", {})
                }
                if config.get("auth_type") == "aws_sigv4":
                    connection["auth"] = agentcore_sigv4_auth.AgentCoreSigV4Auth(
                        region=config.get("auth_region", "us-east-1"),
                        service=config.get("auth_service", "bedrock-agentcore"),
                    )
                server_info[server_name] = connection
            else:
                command = config.get("command", "")
                args = config.get("args", [])
                env = config.get("env", {})
                
                server_info[server_name] = {
                    "transport": "stdio",
                    "command": command,
                    "args": args,
                    "env": env                    
                }
    return server_info

