"""Hourly watchdog for the trainworld orchestrator (T-4582; Anita 2026-11-28: "maybe hourly").

    venv\Scripts\pythonw.exe trainworld/tools/ops/orch_watchdog.py [--dry-run] [--now ISO] [--handoff F] [--board F]
                             [--projects DIR ...] [--state F]

Run by the Windows scheduled task "trainworld orchestrator watchdog" (pythonw: no window). Reads the Orchestrator
line of notes/handoff.md, finds that session's transcript, and starts a successor with next_orchestrator.py when
the session is stalled:
  (a) its last assistant entry is an API error naming a usage or spend limit -> successor on the OTHER account;
  (b) neither its transcript nor its subagents' transcripts were written for 60 minutes while notes/board-live.json
      lists agents -> successor on the SAME account.
It skips (stamping why in the state file) when the target account's newest main transcript ends in a limit error
whose reset time has not passed (no parseable time: out for 5 hours), or when it already started one in the last
3 hours. On a stall it writes a "Stalled:" line under the Orchestrator line of handoff.md and in handoff-log.md.
A run that is not a stall writes only the one-line stamp in %TEMP%\claude-orchestrator\watchdog.json.
Switch off: schtasks /Change /TN "trainworld orchestrator watchdog" /DISABLE  (on again: /ENABLE).
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MAPS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAPS))
import claude2  # noqa: E402

QUIET_MIN = 60
MIN_GAP_H = 3
OUT_DEFAULT_H = 5
UTC = dt.timezone.utc
LIMIT_RE = re.compile(r"(usage|spend) limit|hit your[^.]*limit|reached your[^.]*limit", re.I)
RESET_RE = re.compile(r"resets\s+(?:(\w{3,9})\s+(\d{1,2}),?\s*)?(?:at\s*)?(\d{1,2})(?::(\d\d))?\s*(am|pm)"
                      r"(?:\s*\(([^)]+)\))?", re.I)
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}


def say(*a):
    if sys.stdout:
        print(*a)


def parse_ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def last_assistant(path):
    """(timestamp, text, is_error) of the last assistant entry, read from the file's tail."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        for tail in (262144, 4194304, size):
            f.seek(max(0, size - tail))
            lines = f.read().decode("utf-8", "replace").splitlines()
            for ln in reversed(lines):
                if '"type":"assistant"' not in ln:
                    continue
                try:
                    e = json.loads(ln)
                except ValueError:
                    continue  # a cut first line
                if e.get("type") != "assistant" or e.get("isSidechain"):
                    continue
                parts = e.get("message", {}).get("content") or []
                text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
                err = bool(e.get("isApiErrorMessage")) or e.get("message", {}).get("model") == "<synthetic>"
                return parse_ts(e["timestamp"]), text, err
            if tail >= size:
                break
    return None


def limit_error(path):
    """(entry time, text) when the transcript's last assistant entry is a usage or spend limit error."""
    la = last_assistant(path)
    if la and la[2] and LIMIT_RE.search(la[1]):
        return la[0], la[1]
    return None


def reset_time(ts, text):
    m = RESET_RE.search(text)
    if m:
        mon, day, hh, mm, ap, tz = m.groups()
        h = int(hh) % 12 + (12 if ap.lower() == "pm" else 0)
        zone = dt.timezone(dt.timedelta(hours=-5))  # her time, unless the message names a zone we can load
        if tz:
            try:
                import zoneinfo
                zone = zoneinfo.ZoneInfo(tz.strip())
            except Exception:
                pass
        base = ts.astimezone(zone)
        try:
            if mon and mon[:3].lower() in MONTHS:
                t = base.replace(month=MONTHS[mon[:3].lower()], day=int(day), hour=h, minute=int(mm or 0),
                                 second=0, microsecond=0)
                if t < base:
                    t = t.replace(year=t.year + 1)
            else:
                t = base.replace(hour=h, minute=int(mm or 0), second=0, microsecond=0)
                if t < base:
                    t += dt.timedelta(days=1)
            return t.astimezone(UTC)
        except ValueError:
            pass
    return ts + dt.timedelta(hours=OUT_DEFAULT_H)


def find_transcript(sid, proj_dirs):
    """(path, account number 1|2|...) of <sid>.jsonl; the account is the config folder's place in claude2.ACCOUNTS."""
    for pd in proj_dirs:
        hit = glob.glob(os.path.join(pd, "*", sid + ".jsonl"))
        if hit:
            cfg = os.path.dirname(pd)
            accts = [os.path.normcase(os.path.abspath(a)) for a in claude2.ACCOUNTS]
            n = os.path.normcase(os.path.abspath(cfg))
            return hit[0], (accts.index(n) + 1 if n in accts else 1)
    return None, None


def newest_main(pd):
    files = glob.glob(os.path.join(pd, "*", "*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def account_dir_for(n, proj_dirs):
    want = os.path.normcase(os.path.abspath(claude2.ACCOUNTS[n - 1] if n <= len(claude2.ACCOUNTS) else ""))
    for pd in proj_dirs:
        if os.path.normcase(os.path.abspath(os.path.dirname(pd))) == want:
            return pd
    return None


def newest_write(path):
    sid = Path(path).stem
    files = [path] + glob.glob(os.path.join(os.path.dirname(path), sid, "subagents", "*.jsonl"))
    return dt.datetime.fromtimestamp(max(os.path.getmtime(f) for f in files), UTC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", help="ISO time to act as (tests)")
    ap.add_argument("--handoff", default=str(MAPS / "trainworld/notes/handoff.md"))
    ap.add_argument("--board", default=str(MAPS / "trainworld/notes/board-live.json"))
    ap.add_argument("--log", default=str(MAPS / "trainworld/notes/handoff-log.md"))
    ap.add_argument("--projects", action="append", help="projects folders (default: every account's)")
    ap.add_argument("--state", default=os.path.join(tempfile.gettempdir(), "claude-orchestrator", "watchdog.json"))
    a = ap.parse_args()
    now = parse_ts(a.now) if a.now else dt.datetime.now(UTC)
    proj_dirs = a.projects or claude2.project_dirs()
    try:
        state = json.load(open(a.state))
    except (OSError, ValueError):
        state = {}

    def stamp(msg):
        state["last_run"] = f"{now:%Y-%m-%d %H:%M}Z {msg}"
        say(state["last_run"])
        if not a.dry_run:
            os.makedirs(os.path.dirname(a.state), exist_ok=True)
            with open(a.state, "w") as f:
                json.dump(state, f)

    hand = Path(a.handoff).read_text(encoding="utf-8")
    m = re.search(r"^Orchestrator: session (\S+?),(?: name (\S+?),)?(?: account (\d))?", hand, re.M)
    if not m:
        return stamp("ok: no Orchestrator line")
    sid, name = m.group(1), m.group(2) or "?"
    path, acct = find_transcript(sid, proj_dirs)
    if not path:
        return stamp(f"ok: no transcript for {sid}")

    reason = None
    le = limit_error(path)
    if le:
        reason, target = f"hit its usage limit on account {acct}", 3 - acct if acct in (1, 2) else 1
    else:
        try:
            agents = json.load(open(a.board)).get("agents") or []
        except (OSError, ValueError):
            agents = []
        quiet = (now - newest_write(path)).total_seconds() / 60
        if agents and quiet >= QUIET_MIN:
            reason, target = f"quiet for {int(quiet)} minutes with {len(agents)} agents listed", acct
    if not reason:
        return stamp("ok")

    if state.get("started_for") == sid:
        return stamp(f"stalled ({reason}); already started a successor for {sid}")
    if state.get("last_start"):
        since = (now - parse_ts(state["last_start"])).total_seconds() / 3600
        if since < MIN_GAP_H:
            return stamp(f"stalled ({reason}); skipped, last start {since:.1f} h ago")
    tpd = account_dir_for(target, proj_dirs)
    nm = newest_main(tpd) if tpd else None
    tl = limit_error(nm) if nm else None
    if tl:
        until = reset_time(*tl)
        if until > now:
            return stamp(f"stalled ({reason}); skipped, account {target} is out until {until:%Y-%m-%d %H:%M}Z")

    seen = f"{now:%Y-%m-%d %H:%M} UTC"
    line = f"Stalled: session {sid}, name {name}, {reason}, seen {seen} by the watchdog"
    cmd = [str(MAPS / "venv/Scripts/python.exe"), str(MAPS / "trainworld/tools/ops/next_orchestrator.py"),
           "trainworld", "--advisor", "fable", "--account", str(target)] + (["--dry-run"] if a.dry_run else [])
    if a.dry_run:
        say("would write:", line)
        say("would run:", " ".join(cmd))
        return stamp(f"dry run: stalled ({reason})")
    if f"Stalled: session {sid}," not in hand:
        eol = hand.index("\n", m.start())
        Path(a.handoff).write_text(hand[:eol + 1] + line + "\n" + hand[eol + 1:], encoding="utf-8")
        with open(a.log, "a", encoding="utf-8") as f:
            f.write("\n" + line + "\n")
    r = subprocess.run(cmd, cwd=str(MAPS), capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:
        return stamp(f"stalled ({reason}); start failed: {(r.stdout + r.stderr).strip()[-200:]}")
    state["last_start"], state["started_for"] = f"{now:%Y-%m-%dT%H:%M:%S}Z", sid
    stamp(f"started a successor on account {target} for {sid}")


if __name__ == "__main__":
    main()
