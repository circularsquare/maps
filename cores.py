"""A shared ledger of CPU cores, so long jobs from different projects stay under one cap.

    python cores.py list
    python cores.py claim 6 religiondots-build --pid 12345 "python build.py --jobs N"
    python cores.py release religiondots-build
    python cores.py cap          # show the cap
    python cores.py cap 12       # set it (1-16); default 10

The machine has 16 cores.  Anita's cap is about 10 in all, every agent together, while she is
using it, and 12-15 overnight or while she is away.  Set the cap when that changes.

A claim never blocks or fails.  It grants min(n, free cores), at least 1, and prints that number
as its last line, so a script can read it:

    jobs=$(python cores.py claim 8 mybuild --pid $$ | tail -n 1)

Size --jobs or workers to the number printed.  Nothing is enforced on any process; this is a
promise between agents.  A claim whose PID is no longer running is dropped on the next call, so
a job that dies releases its cores without anyone remembering to.  A claim made without --pid
is dropped after 6 hours instead.  Claiming a name that already exists replaces that claim.

The ledger is cores.json beside this file.  Standard library only.
"""
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LEDGER = HERE / "cores.json"
MACHINE = 16
DEFAULT_CAP = 10
NO_PID_HOURS = 6


def alive(pid):
    """True if a process with this id is running."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x1000, False, pid)         # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            # Access denied means the process exists but is not ours; anything else, it is gone.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == 259                    # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Locked:
    """Agents in different sessions claim at once; without this one write can undo another."""
    path = HERE / "cores.json.lock"

    def __enter__(self):
        for _ in range(100):
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL))
                return
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > 30:    # left by a crash
                        self.path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
                time.sleep(0.1)
        sys.exit(f"{self.path} has been held for 10 s; delete it if nothing is claiming")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def load():
    """(cap, claims) from cores.json; defaults if it is missing or half-written."""
    try:
        data = json.loads(LEDGER.read_text(encoding="utf-8"))
        cap = int(data.get("cap", DEFAULT_CAP))
        claims = [c for c in data.get("claims", []) if isinstance(c, dict) and "name" in c]
        return cap, claims
    except (OSError, ValueError, TypeError, AttributeError):
        return DEFAULT_CAP, []


def save(cap, claims):
    tmp = LEDGER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"cap": cap, "claims": claims}, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    for _ in range(20):
        try:
            return os.replace(tmp, LEDGER)
        except PermissionError:         # Windows: another call is reading it this instant
            time.sleep(0.05)
    os.replace(tmp, LEDGER)


def prune(claims):
    """(kept, dropped): claims whose process has died, or that had no PID and are old."""
    kept, dropped = [], []
    now = time.time()
    for c in claims:
        pid = c.get("pid")
        if pid:
            ok = alive(int(pid))
        else:
            ok = now - c.get("t", 0) < NO_PID_HOURS * 3600
        (kept if ok else dropped).append(c)
    return kept, dropped


def claimed(claims):
    return sum(int(c.get("cores", 0)) for c in claims)


def age(c):
    s = max(0, int(time.time() - c.get("t", time.time())))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def report_pruned(dropped):
    for c in dropped:
        print(f"dropped {c['name']} ({c.get('cores')} cores): pid {c.get('pid')} is gone"
              if c.get("pid") else f"dropped {c['name']} ({c.get('cores')} cores): older than "
              f"{NO_PID_HOURS} h with no pid")


def update(change):
    """Lock, load, prune, let change(cap, claims) edit them, save.  Returns what change returns."""
    with Locked():
        cap, claims = load()
        claims, dropped = prune(claims)
        report_pruned(dropped)
        result = change(cap, claims)
        save(cap, claims)
    return result


def cmd_list():
    def change(cap, claims):
        total = claimed(claims)
        for c in claims:
            pid = f"pid {c['pid']}" if c.get("pid") else "no pid"
            print(f"{c['name']:24} {c['cores']:>2} {'core ' if c['cores'] == 1 else 'cores'} {pid:12} {age(c):>7}  {c.get('note', '')}")
        if not claims:
            print("no claims")
        over = f"  (over by {total - cap})" if total > cap else ""
        print(f"claimed {total} of cap {cap} (machine {MACHINE}), free {max(0, cap - total)}{over}")
    update(change)


def cmd_claim(n, name, pid, note):
    def change(cap, claims):
        others = [c for c in claims if c["name"] != name]
        free = max(0, cap - claimed(others))
        granted = max(1, min(n, free))
        claims[:] = others + [{"name": name, "cores": granted, "pid": pid, "note": note,
                               "t": int(time.time())}]
        if granted < n:
            print(f"asked for {n}, cap is {cap} and others hold {claimed(others)}: "
                  f"granted {granted}")
        return granted
    print(update(change))


def cmd_release(name):
    def change(cap, claims):
        kept = [c for c in claims if c["name"] != name]
        if len(kept) == len(claims):
            print(f"{name} has no claim")
        else:
            print(f"released {name}")
        claims[:] = kept
    update(change)


def cmd_cap(n):
    if n is None:
        def change(cap, claims):
            print(cap)
        return update(change)
    if not 1 <= n <= MACHINE:
        sys.exit(f"cap must be 1-{MACHINE}")

    with Locked():
        old, claims = load()
        claims, dropped = prune(claims)
        report_pruned(dropped)
        print(f"cap {old} -> {n}")
        save(n, claims)


def number(text, what):
    try:
        return int(text)
    except ValueError:
        sys.exit(f"{what} must be a whole number, got {text!r}")


def main():
    args = sys.argv[1:]
    pid = None
    if "--pid" in args:               # may sit anywhere, so the note can follow it
        i = args.index("--pid")
        if i + 1 >= len(args):
            sys.exit("--pid needs a process id")
        pid = number(args[i + 1], "--pid")
        del args[i:i + 2]
    if args == ["list"] and pid is None:
        return cmd_list()
    if args[:1] == ["claim"] and len(args) in (3, 4):
        n = number(args[1], "cores")
        if n < 1:
            sys.exit("claim at least 1 core")
        return cmd_claim(n, args[2], pid, args[3] if len(args) == 4 else "")
    if args[:1] == ["release"] and len(args) == 2:
        return cmd_release(args[1])
    if args[:1] == ["cap"] and len(args) in (1, 2):
        return cmd_cap(number(args[1], "cap") if len(args) == 2 else None)
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
