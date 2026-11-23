"""A shared ledger of CPU cores, so long jobs from different projects stay under one cap.

    python cores.py list
    python cores.py claim 6 religiondots-build --pid 12345 "python build.py --jobs N"
    python cores.py release religiondots-build
    python cores.py cap          # show the cap
    python cores.py cap 12       # set it (1-16); default 10
    python cores.py total        # exit 3 when over the cap

The machine has 16 cores.  Anita's cap is about 10 in all, every agent together, while she is
using it, and 12-15 overnight or while she is away.  Set the cap when that changes.

A claim is never granted past the cap.  If fewer than n cores are free it prints one line
("waiting for 8 cores: cap 14, claimed 14") and re-checks every few seconds until they are, then
prints the number granted as its last line (the waiting line goes to stderr), so a script can read it:

    jobs=$(python cores.py claim 8 mybuild --pid $$ | tail -n 1)

    --min m        take fewer than asked: granted once at least m are free (then min(n, free)).
                   Default m = n.  Size --jobs or workers to the number printed.
    --timeout s    give up after s seconds (default 1800): prints why on stderr, exits 2, claims
                   nothing.  Always check the exit code.
    --timeout 0    do not wait: grant or fail at once.

A claim never waits on itself: the claim being replaced (same name) counts as free.  A process
that holds a claim and starts a child that claims again must give the child no claim of its own
(checks/cores.mjs honours TW_CORES), or the child waits for the parent's cores until its timeout.

    python cores.py total [--since EPOCH]   exit 0 if claimed <= cap (and, with --since, the ledger
                   was never seen over the cap since then), else exit 3: a timing run calls it
                   before and after and refuses to record a time when it fails.

Nothing is enforced on any process; this is a promise between agents.  A claim whose PID is no longer running is dropped on the next call, so
a job that dies releases its cores without anyone remembering to.  A claim made without --pid
is dropped after 6 hours instead.  Claiming a name that already exists replaces that claim.

The ledger is cores.json beside this file (CORES_LEDGER=<path> overrides it, for tests).  Standard
library only.
"""
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LEDGER = pathlib.Path(os.environ["CORES_LEDGER"]).resolve() if os.environ.get("CORES_LEDGER") else HERE / "cores.json"
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
    path = LEDGER.with_name(LEDGER.name + ".lock")

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
    global OVER
    try:
        data = json.loads(LEDGER.read_text(encoding="utf-8"))
        cap = int(data.get("cap", DEFAULT_CAP))
        claims = [c for c in data.get("claims", []) if isinstance(c, dict) and "name" in c]
        OVER = float(data.get("over", 0))
        return cap, claims
    except (OSError, ValueError, TypeError, AttributeError):
        return DEFAULT_CAP, []


OVER = 0.0      # when a call last saw the ledger over its cap (epoch seconds); see "total --since"


def save(cap, claims):
    global OVER
    if claimed(claims) > cap:
        OVER = time.time()
    tmp = LEDGER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"cap": cap, "claims": claims, "over": OVER}, indent=2, ensure_ascii=False) + "\n",
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


def cmd_claim(n, name, pid, note, low, timeout):
    def change(cap, claims):
        want = min(n, cap)          # asking for more than the cap could never be granted
        need = min(low, want)
        others = [c for c in claims if c["name"] != name]     # a claim never waits on its own
        held = claimed(others)
        free = max(0, cap - held)
        if free < need:
            return None, f"waiting for {need} cores: cap {cap}, claimed {held}"
        granted = min(want, free)
        claims[:] = others + [{"name": name, "cores": granted, "pid": pid, "note": note,
                               "t": int(time.time())}]
        if granted < n:
            print(f"asked for {n}, cap is {cap} and others hold {held}: granted {granted}")
        return granted, None
    start = last_said = time.time()
    said = None
    while True:
        granted, why = update(change)
        if granted is not None:
            print(granted)
            return
        if why != said or time.time() - last_said > 300:
            print(why, file=sys.stderr, flush=True)       # stdout stays the number alone
            said, last_said = why, time.time()
        if time.time() - start >= timeout:
            print(f"cores.py: {name} gave up after {int(time.time() - start)} s, {why}; nothing claimed",
                  file=sys.stderr)
            sys.exit(2)
        time.sleep(min(3, max(0.1, timeout - (time.time() - start))))


def cmd_total(since):
    def change(cap, claims):
        total = claimed(claims)
        over = total > cap or (since is not None and OVER and OVER >= since)
        print(f"claimed {total} of cap {cap}" + (" (over)" if total > cap else "")
              + (f", ledger was over the cap at {int(OVER)}" if since is not None and OVER and OVER >= since else ""))
        return over
    if update(change):
        sys.exit(3)


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
    opts = {}
    for flag in ("--min", "--timeout", "--since"):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                sys.exit(f"{flag} needs a number")
            opts[flag] = number(args[i + 1], flag)
            del args[i:i + 2]
    if args == ["list"] and pid is None:
        return cmd_list()
    if args == ["total"]:
        return cmd_total(opts.get("--since"))
    if args[:1] == ["claim"] and len(args) in (3, 4):
        n = number(args[1], "cores")
        if n < 1:
            sys.exit("claim at least 1 core")
        low = opts.get("--min", n)
        if low < 1:
            sys.exit("--min is at least 1")
        return cmd_claim(n, args[2], pid, args[3] if len(args) == 4 else "", low, opts.get("--timeout", 1800))
    if args[:1] == ["release"] and len(args) == 2:
        return cmd_release(args[1])
    if args[:1] == ["cap"] and len(args) in (1, 2):
        return cmd_cap(number(args[1], "cap") if len(args) == 2 else None)
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
