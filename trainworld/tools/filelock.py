"""Shared O_EXCL lockfile helper for trainworld tools (tasks.py, asks.py, changelog.py). Stdlib only."""
import os
import time


class FileLock:
    """Exclusive lockfile with stale-lock expiry so a crashed process cannot block later runs."""

    def __init__(self, path, timeout=10.0, stale=30.0, poll=0.05, err=RuntimeError, message=None):
        self.path = str(path)
        self.timeout = timeout
        self.stale = stale
        self.poll = poll
        self.err = err
        self.message = message

    def __enter__(self):
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        end = time.time() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, f"{os.getpid()}\n".encode("ascii"))
                finally:
                    os.close(fd)
                return self
            except FileExistsError:
                try:
                    if self.stale and time.time() - os.path.getmtime(self.path) > self.stale:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                if time.time() > end:
                    msg = self.message or f"{self.path} is held; if nothing is running, delete it"
                    raise self.err(msg)
                time.sleep(self.poll)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass
