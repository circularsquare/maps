"""Run a Gemini code agent (Antigravity CLI, agy) on a brief in its own worktree, the way a Claude subagent works (T-4252).

    venv\Scripts\python.exe trainworld/tools/gemini_agent.py start <name> --brief <file> [--task T-1234] [--tree <tree>]
        [--new-tree] [--cores 2] [--timeout 90] [--model gemini-4-neon-medium] [--trial]
        [--game <city|file.save>]
    venv\Scripts\python.exe trainworld/tools/gemini_agent.py list
    venv\Scripts\python.exe trainworld/tools/gemini_agent.py stop <name>

start returns at once and the run goes on detached (one agy process, up to --timeout minutes). Its tree is
maps/.tw-trees/<tree> (default: the name), made with agent-tree.mjs when --new-tree is given. With --task it marks
the task on the board (tasks.py start). It claims --cores through cores.py for the run (TW_CORES and CARGO_BUILD_JOBS
tell checks and cargo inside), reading the granted count from stdout only and releasing at the end, failure included.
The report, the agent's final answer (the last agent_response step in agy.log, without interim lines) plus one footer line
(files changed, refused calls, log), lands in the agent report inbox (reports.py, kind "gemini"), so the board watch
prints "agent report ... (gemini, ...)"; a run that fails or times out reports that too. Everything agy printed is in
artifacts/trainworld/gemini/agents/<name>-<time>/agy.log (agy's own log beside it; steps.txt lists each tool call).
For the first runs (notes/gemini-agent-rounds.json, her 11-25) the same conversation is then asked which permissions
it lacked or didn't need, and the answer goes in the report under "Permissions"; --trial runs ask but don't count.

A game (T-4261, --game): agy and everything a tool call starts sit in a job object that is killed when the call ends, so an
agent can't keep a Vite or a play.mjs game alive between calls. With --game the runner (outside that job) starts the tree's
Vite (checks/dev-nowatch.mjs, a free port in 6100-6199) and `play.mjs start <game> --port P --cores 1 --idle <timeout>` before
the session, tells the agent in its prompt (port; use play.mjs do|sim|edits|reload only), and in its finally (a failure or
timeout too; `stop` does it after killing the runner) runs `play.mjs stop --port P` and `taskkill /F /T /PID <vite pid>`, the
pids and port recorded in state.json, nothing broader. If the game can't be opened the agent is told and works without.
The runner dying hard leaves the game to its --idle stop. Quoted arguments pass the command rules (agy_session.ARG: "..."
without $ or backtick, '...'): git grep -e "a" -e "b", git grep -E "a|b", play.mjs do '<js>'; an unquoted | ; & ( ) { } is
still refused. TW_MAPS_ROOT (env) points trees, runs, venv and cores.py at the main maps root, to try a copy of this file
from a worktree.

What the agent can do (agy 1.5 on Windows 10, which has no terminal sandbox):
- Its tools are only those of the custom agent trainworld-coder (written to ~/.gemini/config/agents/ each run): read,
  write and edit files, run a command. No subagents, browser, web search, MCP or scheduling.
- Commands: only those matching an allow rule in settings.json of its run (git status/diff/log/show/blame/ls-files/
  grep, Get-ChildItem, the typecheck, agent_cargo.py test|wasm, changelog.py add|list|reword, node
  trainworld/checks/<x>.mjs), one per call, no chaining or redirection. Anything else is refused by agy, and deny
  rules (agy_session.backstop_rules: git commit, push, stash, reset, checkout and the like, aws, taskkill, tools/ops,
  tools/aws, agent-tree, smoke, 8800) also beat the global grants in ~/.gemini/config/config.json. Refused in the git
  reads too (T-4404, also as abbreviations and with quotes inside): -O / --open-files-in-pager (runs a program),
  --output (writes a file), --ext-diff and --textconv (run a configured driver), --no-index (reads anywhere), and any
  option before the subcommand (-c, --config-env, -C, --exec-path, --git-dir, --work-tree, -p).
- File writes: inside its tree only (outside, including through the tree's links to the main tree's data, node_modules,
  venv and artifacts, agy asks and print mode refuses), and not to TASKS, DECISIONS, ARCHIVE, LINES, TODO, the
  handoff, asks.json, the CLAUDE.md and GEMINI.md files, .git, .agents or .claude, nor the temp folder.
Not enforced: an allowed command runs whatever the tree holds, so an agent that edits a check, package.json or build.rs
can run anything as Anita's user. The rules in GEMINI.md and trainworld/CLAUDE.md are asked, not enforced.
"""
import argparse
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agy_session  # noqa: E402
import reports  # noqa: E402

TW = HERE.parent
MAPS = Path(os.environ.get("TW_MAPS_ROOT") or TW.parent)   # the maps root (trees, runs, venv, cores.py); set only to try a copy of this file from a worktree (T-4261)
TREES = MAPS / ".tw-trees"
RUNS = MAPS / "artifacts" / "trainworld" / "gemini" / "agents"
ROUNDS = TW / "notes" / "gemini-agent-rounds.json"
PY = MAPS / "venv" / "Scripts" / "python.exe"
ASK_ROUNDS = 3                     # her 11-25: ask the first three runs about their permissions
MARK = "END OF REPORT"
ANS_MARK = "END OF ANSWER"
AGENT = "trainworld-coder"
TOOLS = ["view_file", "write_to_file", "replace_file_content", "run_command", "command_status"]
AGENT_BODY = """You are a code agent working unattended in a git worktree of Anita's maps repository, for the trainworld
orchestrator. Your rules are GEMINI.md and trainworld/CLAUDE.md in your working directory; the user message gives the
task and what this run allows. Work carefully, change only what the task needs, and finish with the report asked for."""
PROTECTED = ["trainworld/TASKS.md", "trainworld/DECISIONS.md", "trainworld/ARCHIVE.md",
             "trainworld/ARCHIVE-to-2026-10-19.md", "trainworld/LINES.md", "trainworld/TODO.md",
             "trainworld/notes/handoff.md", "trainworld/notes/asks.json", "trainworld/CLAUDE.md", "CLAUDE.md",
             "GEMINI.md", ".git", ".agents", ".claude"]


def temp_dirs():
    """Every Claude scratchpad sits in the temp folder, which agy lets any agent write to unless denied."""
    dirs = {Path(tempfile.gettempdir()).resolve().as_posix()}
    if os.environ.get("LOCALAPPDATA"):
        dirs.add((Path(os.environ["LOCALAPPDATA"]) / "Temp").as_posix())
    return dirs


def settings():
    SAFE, SEP, ARG = agy_session.SAFE, agy_session.SEP, agy_session.ARG
    py = r"(python|py|venv%sScripts%spython(\.exe)?)" % (SEP, SEP)
    return {
        "agentMode": "accept-edits",
        "toolPermission": "request-review",
        "permissions": {
            "allow": agy_session.read_rules() + [
                r"command(regex:npm run check --prefix trainworld%sapp)" % SEP,
                r"command(regex:%s trainworld%stools%s(agent_cargo\.py (test|wasm)|changelog\.py (add|list|reword))( %s*)?)"
                % (py, SEP, SEP, SAFE),
                r"command(regex:node trainworld%schecks%s[\w.-]+\.mjs( %s*)?)" % (SEP, SEP, SAFE),
                # play.mjs calls are JS in quotes: do 'act.camera([1,2], 3)' (T-4261)
                r"command(regex:node trainworld%schecks%splay\.mjs( %s*)?)" % (SEP, SEP, ARG),
                "read_file(%s)" % MAPS.as_posix(),
            ],
            "deny": agy_session.backstop_rules() + ["write_file(%s)" % p for p in sorted(temp_dirs()) + PROTECTED],
        },
    }


PROMPT = """You are Gemini agent "{name}", one of the trainworld orchestrator's code agents, running unattended.
Read GEMINI.md and then trainworld/CLAUDE.md in full first: they are your rules (where they say "agent", that is you).

This run:
- Your working directory is your own git worktree, {tree} (branch tw/{tree_name}). Edit files only inside it and
  leave your changes uncommitted: the orchestrator lands them.
- These commands run, one per call (no chaining with ; && |, no redirection), from the working directory:
    git status | diff | log | show | blame | ls-files | grep, with arguments (git grep is your search)
    Get-ChildItem (or ls, dir) to list folders; read files with your file tool
    npm run check --prefix trainworld/app        (the typecheck)
    python trainworld/tools/agent_cargo.py test | wasm
    python trainworld/tools/changelog.py add | list | reword ...
    node trainworld/checks/<check>.mjs ...       (a Vite of your own on a port in 6100-6199, never 8800)
  Search for alternatives with several -e patterns or one pattern in double quotes: git grep -n -e "a" -e "b" dir/,
  git grep -n -E "a|b" dir/ (single quotes work too; a | outside quotes is a pipe and is refused).
  Anything else is refused, and a refused command ends your turn; you will be told to carry on. Never commit, push,
  stash, reset or check out anything.
{game}
- {cores} cores are claimed for you (TW_CORES is set). Don't run cores.py, tasks.py or handoff.py.
- The file tools can't write to TASKS.md, DECISIONS.md, ARCHIVE.md, LINES.md, TODO.md, notes/handoff.md,
  notes/asks.json or the CLAUDE.md and GEMINI.md files: say in your report what should change there. Detail goes
  in trainworld/notes/{note} in your tree.
- When you're done, reply with your report: at most 15 lines (trainworld/CLAUDE.md "Talking to Anita"): what you
  changed and where, check results, every in-game string you added or changed (old -> new), anything for Anita.
  End the reply with the line {mark}.

The brief:

{brief}
"""

QUESTION = """One more question before you finish, from Anita, for the first few Gemini agent runs: are there permissions
you should have had that you didn't, or permissions you had that you shouldn't? Answer from what happened in this run,
in a few lines, without running anything. End the reply with the line {mark}."""


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def hidden():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def sh(cmd, cwd=MAPS, timeout=600):
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, **hidden())
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def claim(name, n, say):
    r = subprocess.run([str(PY), str(MAPS / "cores.py"), "claim", str(n), "gemini-" + name, "--pid", str(os.getpid()),
                        "--timeout", "1800", "Gemini agent " + name],
                       cwd=str(MAPS), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=1900, **hidden())
    out_lines = [l.strip() for l in (r.stdout or "").splitlines() if l.strip()]
    err_lines = [l.strip() for l in (r.stderr or "").splitlines() if l.strip()]
    say("cores.py claim: exit %d, out=%s%s" % (
        r.returncode, out_lines[-1] if out_lines else "",
        (", err=" + err_lines[-1]) if err_lines else ""))
    if r.returncode != 0 or not out_lines or not out_lines[-1].isdigit():
        if r.returncode == 0:
            release(name)
        return 0
    return int(out_lines[-1])


def release(name):
    sh([str(PY), str(MAPS / "cores.py"), "release", "gemini-" + name])


def final_reply(log_path, res, marker):
    """The last agent_response step carrying marker in agy.log, so interim tool-call lines stay out of the report."""
    steps = []
    by_idx = {}
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("## ") and " user message " in line:
                    by_idx = {}
                    continue
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("event") == "step_update":
                    s = ev.get("step_update") or {}
                    if s.get("step_type") == "agent_response" and s.get("text_delta"):
                        idx = s.get("step_index")
                        if idx not in by_idx:
                            by_idx[idx] = len(steps)
                            steps.append([])
                        steps[by_idx[idx]].append(s["text_delta"])
                elif ev.get("event") == "result":
                    by_idx = {}
    except OSError:
        pass
    for chunks in reversed(steps):
        body = "".join(chunks)
        i = body.rfind(marker)
        if i >= 0:
            return body[:i].rstrip()
    return agy_session.reply_text(res, marker)


def summary(sess, tree, minutes):
    refused = sess.refused() if sess else []
    code, out = sh(["git", "-C", str(tree), "status", "--short"])
    changed = len([l for l in out.splitlines() if l.strip()]) if code == 0 else "?"
    parts = ["%s files changed in its tree" % changed, "%d min" % minutes, "%d turns" % (sess.turns if sess else 0)]
    if refused:
        parts.append("%d refused (%s)" % (len(refused), "; ".join(s["args"][:60] for s in refused[:4])))
    return ", ".join(parts)


GAME_TEXT = """- A game is open for you: {what}, paused, on :{port}, in a Vite and a headless Chrome that this run started outside your\n  tool calls and stops when you finish. Use only play.mjs do | sim | edits | reload | status, with calls in quotes (one game\n  in your tree, so --port is optional). Don't run play.mjs start, stop or serve, or your own Vite for it: a process you\n  start in a tool call ends with the call. For another city or save, use play.mjs reload <city|file.save>."""
GAME_NONE = "- No game is open for you. A process you start in a tool call ends with the call, so play.mjs can't keep a game here."
GAME_FAILED = "- A game could not be opened for you (" + "{err}" + "). Work without one and say so in your report."


def free_port():
    """A port in 6100-6199 that nothing listens on and no play.mjs game has a state file for."""
    base = random.randrange(100)
    for i in range(100):
        port = 6100 + (base + i) % 100
        if (Path(tempfile.gettempdir()) / "tw-play" / ("%d.json" % port)).exists():
            continue
        busy = False
        for host, fam in (("127.0.0.1", socket.AF_INET), ("::1", socket.AF_INET6)):
            try:
                with socket.socket(fam, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    busy = busy or s.connect_ex((host, port)) == 0
            except OSError:
                pass
        if not busy:
            return port
    raise RuntimeError("no free port in 6100-6199")


def open_game(tree, what, timeout_min, run_dir, say):
    """T-4261: agy's job object ends everything a tool call starts, so the harness (outside that job) starts the agent's Vite
    and its play.mjs game, and close_game stops them. Returns {port, vite_pid}."""
    port = free_port()
    log = open(run_dir / "vite.log", "w", encoding="utf-8")
    vite = subprocess.Popen(["node", str(tree / "trainworld" / "checks" / "dev-nowatch.mjs"), str(port)], cwd=str(tree),
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                            creationflags=0x08000000 | 0x200)   # CREATE_NO_WINDOW, CREATE_NEW_PROCESS_GROUP
    game = {"port": port, "vite_pid": vite.pid}
    say("vite pid %d on :%d" % (vite.pid, port))
    try:
        t0 = time.time()
        while True:
            if vite.poll() is not None:
                raise RuntimeError("Vite exited (%s); see vite.log" % vite.returncode)
            try:
                urllib.request.urlopen("http://localhost:%d/" % port, timeout=5).close()
                break
            except Exception:
                if time.time() - t0 > 120:
                    raise RuntimeError("Vite not answering after 120 s")
                time.sleep(1)
        code, out = sh(["node", "trainworld/checks/play.mjs", "start", what, "--port", str(port), "--cores", "1",
                        "--idle", str(timeout_min), "--timeout", "150"], cwd=tree, timeout=240)
        say("play.mjs start: exit %d %s" % (code, out.strip()[-300:]))
        if code != 0 or not out.strip().startswith("ready:"):
            raise RuntimeError("play.mjs start: " + out.strip()[-300:])
    except Exception:
        close_game(tree, game, say)
        raise
    return game


def close_game(tree, game, say):
    """Stop the game (play.mjs stop ends its Chrome) and the Vite, by the pids and port this run recorded, nothing broader."""
    if not game:
        return
    try:
        code, out = sh(["node", "trainworld/checks/play.mjs", "stop", "--port", str(game["port"])], cwd=tree, timeout=90)
        say("play.mjs stop: exit %d %s" % (code, out.strip()[-200:]))
    except Exception as ex:
        say("play.mjs stop failed: %s" % ex)
    try:
        code, out = sh(["taskkill", "/F", "/T", "/PID", str(game["vite_pid"])], timeout=60)
        say("vite stop: exit %d" % code)
    except Exception as ex:
        say("vite stop failed: %s" % ex)


def run(a):
    run_dir = Path(a.run_dir) if a.run_dir else RUNS / ("%s-%s" % (a.name, time.strftime("%Y%m%d-%H%M")))
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    state = {"name": a.name, "pid": os.getpid(), "status": "running", "started": now(), "tree": a.tree,
             "task": a.task, "model": a.model, "trial": a.trial}
    save_json(state_path, state)
    runner_log = open(run_dir / "runner.txt", "a", encoding="utf-8")

    def say(msg):
        runner_log.write("%s %s\n" % (now(), msg))
        runner_log.flush()

    tree = TREES / a.tree
    t0 = time.time()
    sess, granted, text, status = None, 0, "", "failed"
    try:
        if a.new_tree:
            code, out = sh(["node", str(TW / "checks" / "agent-tree.mjs"), "new", a.tree])
            say("agent-tree new: exit %d\n%s" % (code, out))
            if code != 0:
                raise RuntimeError("agent-tree.mjs new %s failed: %s" % (a.tree, out.strip()[-300:]))
        if not (tree / ".git").exists():
            raise RuntimeError("no tree at %s (make it with agent-tree.mjs new, or pass --new-tree)" % tree)
        if a.task:
            code, out = sh([str(PY), str(TW / "tools" / "tasks.py"), "start", a.task, "--agent", a.name,
                            "--model", a.model, "--tree", a.tree])
            say("tasks.py start: exit %d %s" % (code, out.strip()))
        granted = claim(a.name, a.cores, say)
        state["cores_claimed"] = granted
        if not granted:
            raise RuntimeError("cores.py claim for %d cores failed or timed out" % a.cores)
        brief = Path(a.brief).read_text(encoding="utf-8")
        game_text = GAME_NONE
        if a.game:
            try:
                game = open_game(tree, a.game, a.timeout, run_dir, say)
                state["game"] = {"port": game["port"], "vite_pid": game["vite_pid"]}
                save_json(state_path, state)
                game_text = GAME_TEXT.format(what=a.game, port=game["port"])
            except Exception as ex:   # the run goes on without a game, and the agent is told
                say("game: %s: %s" % (type(ex).__name__, ex))
                game_text = GAME_FAILED.format(err=str(ex)[-400:])
        (run_dir / "brief.md").write_text(brief, encoding="utf-8")
        settings_path = run_dir / "settings.json"
        save_json(settings_path, settings())
        agy_session.ensure_agent(AGENT, "Code agent for trainworld/tools/gemini_agent.py (files, search, commands)",
                                 TOOLS, AGENT_BODY)
        prompt = PROMPT.format(name=a.name, tree=tree, tree_name=a.tree, cores=granted, mark=MARK, brief=brief, game=game_text,
                               note=(a.task + ".md") if a.task else "T-xxxx.md")
        env = dict(os.environ, TW_CORES=str(granted), CARGO_BUILD_JOBS=str(granted))
        sess = agy_session.Session(tree, run_dir / "agy.log", model=a.model, agent=AGENT, settings=settings_path,
                                   cli_log=run_dir / "agy-cli.log", env=env)
        state["agy_pid"] = sess.start()
        save_json(state_path, state)
        say("agy started, pid %s" % state["agy_pid"])
        deadline = t0 + a.timeout * 60
        res, nudges = agy_session.work(sess, prompt, MARK, deadline)
        say("work done: nudges %d, ended %s, conversation %s" % (nudges, sess.ended, sess.conversation))
        if res is not None and MARK in (res.get("response") or ""):
            text, status = final_reply(run_dir / "agy.log", res, MARK), "done"
        elif sess.ended == "timeout":
            text, status = "Timed out after %d min. Its last words: %s" % (a.timeout, sess.partial.strip()[-1500:]), "timeout"
        else:
            last = agy_session.reply_text(res, MARK) if res else sess.partial.strip()
            text = "Stopped without a report (%s, %d nudges). Its last words: %s" % (sess.ended or "no closing line",
                                                                                     nudges, last[-1500:])
        rounds = load_json(ROUNDS, {"asked": 0, "runs": []})
        if status == "done" and rounds.get("asked", 0) < ASK_ROUNDS:
            ans, _ = agy_session.work(sess, QUESTION.format(mark=ANS_MARK), ANS_MARK,
                                      min(deadline + 600, time.time() + 600), max_nudges=2)
            answer = final_reply(run_dir / "agy.log", ans, ANS_MARK) if ans else ""
            text += "\n\nPermissions\n" + (answer or "(no answer)")
            if not a.trial:
                rounds["asked"] = rounds.get("asked", 0) + 1
                rounds.setdefault("runs", []).append({"name": a.name, "task": a.task, "at": now()})
                save_json(ROUNDS, rounds)
    except Exception as ex:      # the orchestrator hears about every failure through the inbox
        say("error: %s: %s" % (type(ex).__name__, ex))
        text = text or "Failed before finishing: %s" % ex
    finally:
        if sess:
            sess.close()
        close_game(tree, state.get("game"), say)
        if granted:
            release(a.name)
        minutes = round((time.time() - t0) / 60)
        foot = "(Gemini agent %s, %s, tree %s: %s. Log: %s)" % (a.name, a.model, a.tree,
                                                                 summary(sess, tree, minutes), run_dir / "agy.log")
        report = text.strip() + "\n\n" + foot
        (run_dir / "report.md").write_text(report + "\n", encoding="utf-8")
        if sess:
            with open(run_dir / "steps.txt", "w", encoding="utf-8") as f:
                for s in sess.steps:
                    f.write("%s %s %s %s\n" % (s["state"], s["name"], s["args"], s["error"]))
        rec = reports.Box(str(TW)).add(a.name, report, task=a.task, kind="gemini", model=a.model,
                                       agent_id="gemini-%s-%s" % (a.name, (sess.conversation if sess else None) or
                                                                  run_dir.name),
                                       description="Gemini agent %s in tree %s%s" % (a.name, a.tree,
                                                                                     " (trial)" if a.trial else ""))
        state.update(status=status, ended=now(), report=rec["id"], conversation=sess.conversation if sess else None)
        save_json(state_path, state)
        say("report %s, status %s" % (rec["id"], status))
        runner_log.close()


def runs():
    out = []
    for p in sorted(RUNS.glob("*/state.json")):
        st = load_json(p, {})
        st["dir"] = str(p.parent)
        out.append(st)
    return out


def alive(pid):
    code, out = sh(["tasklist", "/FI", "PID eq %d" % int(pid), "/NH"], timeout=30)
    return str(pid) in out


def start(a):
    run_dir = RUNS / ("%s-%s" % (a.name, time.strftime("%Y%m%d-%H%M%S")))
    run_dir.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, str(Path(__file__).resolve()), "run", a.name, "--brief", str(Path(a.brief).resolve()),
            "--tree", a.tree, "--cores", str(a.cores), "--timeout", str(a.timeout), "--model", a.model,
            "--run-dir", str(run_dir)]
    if a.task:
        args += ["--task", a.task]
    if a.new_tree:
        args.append("--new-tree")
    if a.trial:
        args.append("--trial")
    if a.game:
        args += ["--game", a.game]
    out = open(run_dir / "runner-console.txt", "w", encoding="utf-8")
    # CREATE_NO_WINDOW, not DETACHED_PROCESS: a detached process has no console, so the venv's python.exe redirector and
    # every console program the run starts would each open a window on her screen; a hidden console is inherited.
    flags = 0x08000000 | 0x200 | 0x01000000  # CREATE_NO_WINDOW, CREATE_NEW_PROCESS_GROUP, CREATE_BREAKAWAY_FROM_JOB
    kw = dict(cwd=str(MAPS), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, close_fds=True)
    try:
        p = subprocess.Popen(args, creationflags=flags, **kw)
    except OSError:                       # a job that forbids breakaway: still detached from this console
        p = subprocess.Popen(args, creationflags=flags & ~0x01000000, **kw)
    print("started Gemini agent %s in tree %s, up to %d min; its report arrives in the inbox (reports.py list). "
          "Run folder: %s" % (a.name, a.tree, a.timeout, run_dir))


def stop(a):
    live = [r for r in runs() if r.get("name") == a.name and r.get("status") == "running"]
    if not live:
        print("no running Gemini agent named %s" % a.name)
        return 1
    for r in live:
        if alive(r["pid"]):
            sh(["taskkill", "/F", "/PID", str(r["pid"])], timeout=60)   # its job object takes agy and its children
        close_game(TREES / r["tree"], r.get("game"), lambda m: print(m))   # the runner died before its own finally
        release(a.name)
        text = "Stopped by the orchestrator before it reported. Log: %s" % (Path(r["dir"]) / "agy.log")
        rec = reports.Box(str(TW)).add(a.name, text, task=r.get("task"), kind="gemini", model=r.get("model"),
                                       agent_id="gemini-%s-%s" % (a.name, Path(r["dir"]).name))
        r.update(status="stopped", ended=now(), report=rec["id"])
        d = r.pop("dir")
        save_json(Path(d) / "state.json", r)
        print("stopped %s (pid %s), report %s" % (a.name, r["pid"], rec["id"]))
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("start", "run"):
        s = sub.add_parser(cmd)
        s.add_argument("name")
        s.add_argument("--brief", required=True)
        s.add_argument("--task")
        s.add_argument("--tree")
        s.add_argument("--new-tree", action="store_true")
        s.add_argument("--cores", type=int, default=2)
        s.add_argument("--timeout", type=int, default=90, help="minutes")
        s.add_argument("--model", default="gemini-4-neon-medium")
        s.add_argument("--trial", action="store_true", help="a test run: not counted as a round")
        s.add_argument("--game", help="a city id (nyc) or .save path: a game is opened for the agent before the run (play.mjs)")
        s.add_argument("--run-dir", help=argparse.SUPPRESS)
    sub.add_parser("list")
    t = sub.add_parser("stop")
    t.add_argument("name")
    a = ap.parse_args()
    if a.cmd in ("start", "run"):
        a.tree = a.tree or a.name
        if not re.fullmatch(r"[\w.-]+", a.name) or not re.fullmatch(r"[\w.-]+", a.tree):
            ap.error("name and tree: letters, digits, . _ - only")
        if not Path(a.brief).is_file():
            ap.error("no brief file at %s" % a.brief)
        return start(a) if a.cmd == "start" else run(a)
    if a.cmd == "stop":
        return stop(a)
    for r in runs():
        print("%-14s %-8s %s  tree %s  %s  %s" % (r.get("name"), r.get("status"), r.get("started", ""),
                                                  r.get("tree"), r.get("task") or "", r.get("report") or ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
