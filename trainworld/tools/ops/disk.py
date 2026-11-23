"""Free space on the drive that holds trainworld, and a warning before it runs out (T-4088).

    python trainworld/tools/ops/disk.py                       one line: free space and its level
    python trainworld/tools/ops/disk.py --notify              also chime and show a Windows notification if low
    python trainworld/tools/ops/disk.py --warn-gb 2000        fake thresholds, to see it fire

C: ran out on 2026-11-23 and builds failed. Levels: "low" under 30 GB free, "urgent" under 10 GB.
board/watch.py checks every few minutes during an orchestrator session and prints a "disk:" line once per
level crossing, chiming for Anita (tools/ping.py) with a notification naming the free space.
What takes the space and what can go: notes/T-4088.md.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # the maps root
WARN_GB, URGENT_GB = 30, 10
GB = 1 << 30
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def free_gb(path=ROOT):
    return shutil.disk_usage(str(path)).free / GB


def level(free, warn=WARN_GB, urgent=URGENT_GB):
    """0 fine, 1 low, 2 urgent."""
    return 2 if free < urgent else 1 if free < warn else 0


def line(free, lvl, warn=WARN_GB, urgent=URGENT_GB):
    drive = ROOT.drive or str(ROOT)
    if lvl == 2:
        return f"disk: URGENT {drive} {free:.1f} GB free (under {urgent:g} GB); builds fail at 0. Delete rebuildable output now (notes/T-4088.md)"
    if lvl == 1:
        return f"disk: low, {drive} {free:.1f} GB free (under {warn:g} GB); see notes/T-4088.md for what can go"
    return f"disk: {drive} {free:.1f} GB free, fine again"


TOAST = r"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$n = $t.GetElementsByTagName('text')
$n.Item(0).AppendChild($t.CreateTextNode($env:TW_TOAST_TITLE)) > $null
$n.Item(1).AppendChild($t.CreateTextNode($env:TW_TOAST_BODY)) > $null
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}WindowsPowerShell1.0powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($t))
"""


def notify(free, lvl):
    """The quiet chime, and a Windows notification with the number. Never raises."""
    drive = ROOT.drive or str(ROOT)
    title = f"{drive} is almost full" if lvl == 2 else f"{drive} is getting full"
    body = f"{free:.1f} GB free. Builds fail at 0."
    env = {**__import__("os").environ, "TW_TOAST_TITLE": title, "TW_TOAST_BODY": body}
    for cmd in ([sys.executable, str(ROOT / "tools" / "ping.py")],
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", TOAST]):
        try:
            subprocess.run(cmd, env=env, capture_output=True, timeout=30, creationflags=NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            pass


class Watch:
    """Says something once per level crossing. Back to fine only 2 GB above the warning, so it doesn't flap."""

    def __init__(self, warn=WARN_GB, urgent=URGENT_GB):
        self.warn, self.urgent, self.last = warn, urgent, 0

    def check(self, free=None):
        free = free_gb() if free is None else free
        lvl = level(free, self.warn, self.urgent)
        if lvl == 0 and self.last and free < self.warn + 2:
            lvl = 1
        if lvl == self.last:
            return None
        rising = lvl > self.last
        self.last = lvl
        return line(free, lvl, self.warn, self.urgent), (lvl if rising else 0), free


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--warn-gb", type=float, default=WARN_GB)
    ap.add_argument("--urgent-gb", type=float, default=URGENT_GB)
    ap.add_argument("--notify", action="store_true", help="chime and notify when low or urgent")
    a = ap.parse_args()
    free = free_gb()
    lvl = level(free, a.warn_gb, a.urgent_gb)
    print(line(free, lvl, a.warn_gb, a.urgent_gb))
    if a.notify and lvl:
        notify(free, lvl)
    return lvl


if __name__ == "__main__":
    sys.exit(main())
