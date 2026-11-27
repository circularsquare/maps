"""Second Claude account: keeps C:\\Users\\anita\\.claude-2 in step with .claude, and owns the list of config folders.

    python claude2.py            copy settings, skills, agents and rules from .claude to .claude-2; print what changed
    python claude2.py --dry-run  print what would change

Never touches credentials, .claude.json, projects/ or history. CLAUDE.md in .claude-2 is created once (it imports
the first account's) and left alone after. Tools that read transcripts import config_dirs() from here.
"""
import filecmp
import os
import shutil
import sys

HOME = os.path.expanduser("~")
ACCOUNTS = [os.path.join(HOME, ".claude"), os.path.join(HOME, ".claude-2")]
FILES = ["settings.json", "keybindings.json"]
# skills/synced holds what the first account pulls from its claude.ai; it is not copied
DIRS = ["skills", "agents", "rules"]
IMPORT_LINE = "@" + os.path.join(ACCOUNTS[0], "CLAUDE.md").replace("\\", "/") + "\n"


def config_dirs():
    """Every config folder that exists, plus the one this session runs from (CLAUDE_CONFIG_DIR) if it is new."""
    dirs = list(ACCOUNTS)
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    if env and env not in dirs:
        dirs.append(env)
    return [d for d in dirs if os.path.isdir(d)]


def project_dirs():
    return [os.path.join(d, "projects") for d in config_dirs() if os.path.isdir(os.path.join(d, "projects"))]


def _copy_file(src, dst, dry, log):
    if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
        return
    log.append(("updated " if os.path.exists(dst) else "added   ") + dst)
    if not dry:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


def _mirror(src, dst, dry, log, skip=()):
    seen = set()
    for root, dirs, names in os.walk(src):
        dirs[:] = [d for d in dirs if d not in skip or root != src]
        for n in names:
            s = os.path.join(root, n)
            rel = os.path.relpath(s, src)
            seen.add(rel)
            _copy_file(s, os.path.join(dst, rel), dry, log)
    if os.path.isdir(dst):
        for root, dirs, names in os.walk(dst):
            dirs[:] = [d for d in dirs if d not in skip or root != dst]
            for n in names:
                d = os.path.join(root, n)
                if os.path.relpath(d, dst) not in seen:
                    log.append("removed " + d)
                    if not dry:
                        os.remove(d)


def sync(dry=False):
    src, dst = ACCOUNTS
    log = []
    if not os.path.isdir(dst):
        log.append("made    " + dst)
        if not dry:
            os.makedirs(dst)
    claude_md = os.path.join(dst, "CLAUDE.md")
    if not os.path.exists(claude_md):
        log.append("added   " + claude_md)
        if not dry:
            with open(claude_md, "w", encoding="utf-8", newline="\n") as f:
                f.write(IMPORT_LINE)
    for name in FILES:
        if os.path.exists(os.path.join(src, name)):
            _copy_file(os.path.join(src, name), os.path.join(dst, name), dry, log)
    for name in DIRS:
        if os.path.isdir(os.path.join(src, name)):
            _mirror(os.path.join(src, name), os.path.join(dst, name), dry, log,
                    {"synced"} if name == "skills" else ())
    return log


if __name__ == "__main__":
    out = sync("--dry-run" in sys.argv)
    for line in out[:40]:
        print(line)
    if len(out) > 40:
        print(f"... and {len(out) - 40} more")
    print(f"{len(out)} changes" if out else "nothing changed")
