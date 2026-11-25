"""Drive one Antigravity CLI (agy) session from Python: the loop gemini_agent.py and gemini_audit.py share (T-4252).

agy runs headless with stream-json in and out, so one process keeps the conversation across turns. Two things about
agy 1.5 on this machine shape the loop:
- In print mode a tool call that needs an approval nobody can give (a command no allow rule matches, a file outside
  the workspace) is soft-denied by cancelling the whole turn: the reply stops mid-work and agy still exits 0. That is
  how the T-4248 and T-4250 audits ended after one line (agy log: "soft-denying tool confirmation", then
  CancelCascadeInvocation). A deny rule only fails that one call and the turn goes on. So work() asks for a closing
  marker, and a turn that ends without it gets one more message telling the agent to carry on.
- Windows 10 has no terminal sandbox (agy needs Windows 11 24H2), so --sandbox and auto mode do nothing here. What
  limits an agent is its tool list (a custom agent file, ensure_agent) and the allow and deny rules passed with
  --settings, nothing at the OS level.
agy and anything it starts run in a Windows job object that is killed when the session closes or this process dies.
Stdlib only.
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

AGENTS_DIR = Path.home() / ".gemini" / "config" / "agents"
SAFE = r"[^;&|<>$`(){}\r\n]"   # one argument: no chaining, pipes, redirection, subexpressions or escapes
SEP = r"[/\\]"


def read_rules():
    """Allow rules for commands that only read. agy 1.5 retired its own search and list tools (its log: "Tool
    \"grep_search\" is deprecated and has no effect"), so an agent searches through the shell: git grep and
    Get-ChildItem. It reads files with its file tool, which keeps to the workspace and the read_file rules; Get-Content
    and Select-String would not."""
    return [r"command(regex:git (status|diff|log|show|blame|ls-files|grep)( %s*)?)" % SAFE,
            r"command(regex:(Get-ChildItem|ls|dir)( %s*)?)" % SAFE]


def backstop_rules():
    """Deny rules for what no Gemini run may do. A deny beats any allow, also the global grants in
    ~/.gemini/config/config.json (two aws commands among them), and fails only that call rather than the whole turn."""
    return [
        r"command(regex:git( .*)? (commit|push|pull|fetch|merge|rebase|reset|stash|checkout|switch|restore|clean"
        r"|worktree|cherry-pick|revert|am|apply|gc|prune|update-ref|config|remote|submodule|filter-branch)( .*)?)",
        r"command(regex:.*\b(aws|taskkill|tskill|Stop-Process|wmic)\b.*)",
        r"command(regex:.*(8800|agent-tree|smoke|tools%s(ops|aws)%s).*)" % (SEP, SEP),
        # git diff --output writes a file anywhere; --no-index reads one anywhere; git grep -O starts a program
        r"command(regex:.*( --output| --no-index| -O| --open-files-in-pager| --ext-diff).*)",
        "mcp(*)",
    ]


def find_agy():
    exe = shutil.which("agy")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cand = Path(local) / "agy" / "bin" / "agy.exe"
        if cand.exists():
            return str(cand)
    home_cand = Path.home() / ".local" / "bin" / "agy"
    if home_cand.exists():
        return str(home_cand)
    raise SystemExit("agy CLI not found; install with: irm https://antigravity.google/cli/install.ps1 | iex")


def ensure_agent(name, description, tools, body):
    """Write the custom agent ~/.gemini/config/agents/<name>.md (selected with --agent). Its tools list is the only
    set of tools the agent gets; a name agy doesn't know can hang the run (agy docs), so use names from the init event."""
    text = "---\nname: %s\ndescription: %s\ntools:\n%s\nmainAgent: true\nsubagent: false\n---\n\n%s\n" % (
        name, description, "\n".join("  - " + t for t in tools), body.strip())
    path = AGENTS_DIR / (name + ".md")
    try:
        if path.read_text(encoding="utf-8") == text:
            return path
    except OSError:
        pass
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


# --- a job object, so agy's children (a Vite, a cargo build) die with the session ---------------------------------
def _job_for(proc):
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # 9: extended limit information
    if not k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
        k32.CloseHandle(job)
        return None
    return (k32, job)


def _end_job(job):
    if job:
        k32, h = job
        k32.TerminateJobObject(h, 1)
        k32.CloseHandle(h)


class Session:
    """One agy process: turn(text) sends a user message and returns that turn's result event (a dict) or None."""

    def __init__(self, cwd, log_path, model=None, agent=None, settings=None, effort=None, conversation=None,
                 mode=None, cli_log=None, env=None):
        self.cmd = [find_agy(), "--input-format", "stream-json", "--output-format", "stream-json"]
        for flag, val in (("--model", model), ("--agent", agent), ("--settings", settings), ("--effort", effort),
                          ("--conversation", conversation), ("--mode", mode), ("--log-file", cli_log)):
            if val:
                self.cmd += [flag, str(val)]
        self.cwd, self.env = str(cwd), env
        self.log = open(log_path, "a", encoding="utf-8", newline="\n")
        self.lines = queue.Queue()
        self.proc = self.job = None
        self.conversation = conversation
        self.steps = []          # every finished tool call: {name, args, state, error}
        self.turns = 0
        self.partial = ""        # the current turn's reply so far, for a run cut off by its timeout
        self.ended = None        # why the session stopped answering: "exited", "timeout"

    def note(self, text):
        self.log.write("## %s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), text))
        self.log.flush()

    def start(self):
        self.note("start: " + subprocess.list2cmdline(self.cmd) + " (cwd " + self.cwd + ")")
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                                     errors="replace", bufsize=1, **kw)
        self.job = _job_for(self.proc)
        threading.Thread(target=self._pump, args=(self.proc.stdout, True), daemon=True).start()
        threading.Thread(target=self._pump, args=(self.proc.stderr, False), daemon=True).start()
        return self.proc.pid

    def _pump(self, stream, is_out):
        for line in stream:
            if is_out:
                self.lines.put(line)
            else:
                self.log.write("stderr: " + line)
                self.log.flush()
        if is_out:
            self.lines.put(None)

    def turn(self, text, deadline):
        """Send one user message; return its result event, or None if agy exited or the deadline passed first."""
        if self.ended:
            return None
        self.note("user message (%d chars)" % len(text))
        self.partial = ""
        try:
            self.proc.stdin.write(json.dumps({"event": "user", "message": {"content": text}}) + "\n")
            self.proc.stdin.flush()
        except OSError:
            self.ended = "exited"
            return None
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, min(30, deadline - time.time())))
            except queue.Empty:
                if time.time() >= deadline:
                    self.ended = "timeout"
                    return None
                continue
            if line is None:
                self.ended = "exited"
                return None
            self.log.write(line)
            self.log.flush()
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind = ev.get("event")
            if kind == "init":
                self.conversation = ev.get("conversation_id") or self.conversation
            elif kind == "step_update":
                s = ev.get("step_update") or {}
                if s.get("step_type") == "agent_response":
                    self.partial += s.get("text_delta") or ""
                self._step(s)
            elif kind == "result":
                self.turns += 1
                res = ev.get("result") or {}
                self.conversation = res.get("conversation_id") or self.conversation
                return res

    def _step(self, s):
        if s.get("step_type") != "tool" or s.get("state") not in ("DONE", "ERROR"):
            return
        info = s.get("tool_info") or {}
        args = info.get("parameters") or {}
        arg = args.get("CommandLine") or args.get("Path") or args.get("AbsolutePath") or json.dumps(args)[:200]
        err = (info.get("error") or {}).get("message", "")
        self.steps.append({"name": s.get("tool_name") or info.get("name"), "args": arg, "state": s["state"],
                           "error": err})

    def refused(self):
        """Tool calls the permission rules turned down, in order."""
        return [s for s in self.steps if s["state"] == "ERROR" and "permission" in s["error"].lower()]

    def close(self, grace=20):
        """End the session: close stdin, give agy a moment to save the conversation, then end everything it started."""
        if self.proc:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
            _end_job(self.job)
            self.job = None
            if self.proc.poll() is None:
                self.proc.kill()
        self.note("closed (%s)" % (self.ended or "done"))
        self.log.close()


CARRY_ON = ("Your turn ended before your closing line. If the system refused a tool call, that call is not allowed in "
            "this run: don't retry it or work around it. Carry on with the task, and finish as the instructions say.")
WRAP_UP = "Stop working now and give your final reply as the instructions say, with what you have."


def work(session, prompt, marker, deadline, max_nudges=12):
    """Send prompt and keep the agent going until a reply contains marker. Returns (last result or None, nudges)."""
    res = session.turn(prompt, deadline)
    nudges = 0
    while res is not None and marker not in (res.get("response") or ""):
        if nudges >= max_nudges:
            break
        nudges += 1
        res = session.turn(WRAP_UP if nudges == max_nudges else CARRY_ON, deadline)
    return res, nudges


def reply_text(res, marker):
    """The reply without its closing marker line."""
    text = (res or {}).get("response") or ""
    i = text.rfind(marker)
    return (text[:i] if i >= 0 else text).rstrip()
