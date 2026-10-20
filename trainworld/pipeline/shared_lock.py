"""T-786: cross-process locks on the files several runners and tools share, so two runners never
lose each other's edits (docs/adding-a-city.md, Running the pipeline).

    with shared_lock.held("registry", "register.py"):   read, change and write the file
        ...

One lock file per name in data/work/world_run/locks/ (a scratch tree's own under --root); the OS
lock goes when its process dies, so a crash leaves nothing to clean up. A lock is re-entrant within
a process (register.py holds "registry" and calls market_zones.main, which takes it too). Names in
use, always taken in this order when nested:

    extract-<name>  a Geofabrik extract, for a whole region (one run reads and deletes it)
    packs           a joint solve (it waits for every gate-<batch> to go; no gate starts while held)
    gate-<batch>    a region's gate, snapshot to verdicts (gates of several runs overlap)
    registry        registry.json, market-rows.json, data/work/market_rows.json, the .market.json files
    plan            world_plan.json and the plan extents
    made            data/work/world_run/made.json
    batch-<batch>   held by a run for its life: its claimed rows are live while it is held
"""
import contextlib
import datetime
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "data/work/world_run/locks"
WAIT_SAY_S = 5.0

_guard = threading.Lock()
_mine = {}  # path -> [RLock, depth, fd]


def _dir(root):
    return Path(root) / "data/work/world_run/locks" if root else LOCKS


def _try(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _free(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


def who(name, root=None):
    """What the holder wrote when it took the lock, or None."""
    try:
        return json.loads((_dir(root) / f"{name}.who").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def held(name, what="", root=None, say=print):
    """Hold lock `name` (waiting for it) for the block."""
    path = str(_dir(root) / f"{name}.lock")
    with _guard:
        ent = _mine.setdefault(path, [threading.RLock(), 0, None])
    ent[0].acquire()
    try:
        if ent[1] == 0:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT)  # never truncates (a lock probe must not)
            t0, said = time.time(), False
            while not _try(fd):
                if not said and time.time() - t0 > WAIT_SAY_S:
                    w = who(name, root) or {}
                    say(f"    waiting for the {name} lock (held by pid {w.get('pid')}: {w.get('what')}, since {w.get('since')})")
                    said = True
                time.sleep(0.5)
            if said:
                say(f"    took the {name} lock after {time.time() - t0:.0f} s")
            ent[2] = fd
            try:
                tmp = Path(path).with_suffix(f".who.{os.getpid()}")
                tmp.write_text(json.dumps({"pid": os.getpid(), "what": what,
                                           "since": datetime.datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
                os.replace(tmp, Path(path).with_suffix(".who"))
            except OSError:
                pass  # only a note for whoever waits
        ent[1] += 1
        try:
            yield
        finally:
            ent[1] -= 1
            if ent[1] == 0:
                fd, ent[2] = ent[2], None
                _free(fd)
                os.close(fd)
    finally:
        ent[0].release()


def is_held(name, root=None):
    """Whether another process holds lock `name` now (a run's batch lock: its claims are live)."""
    path = _dir(root) / f"{name}.lock"
    if not path.exists():
        return False
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return False
    try:
        if _try(fd):
            _free(fd)
            return False
        return True
    finally:
        os.close(fd)


def write_atomic(path, text):
    """Write a text file whole or not at all, so a reader never sees half of it. Windows refuses the
    rename while another process has the file open; that is brief, so it is tried again."""
    path = Path(path)
    part = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    part.write_text(text, encoding="utf-8")
    for k in range(80):
        try:
            os.replace(part, path)
            return
        except PermissionError:
            if k == 79:
                raise
            time.sleep(0.25)


def read_json(path):
    """A JSON file another process may be replacing: tried again while Windows has it locked."""
    for k in range(40):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if k == 39:
                raise
            time.sleep(0.25)
