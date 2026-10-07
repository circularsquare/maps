"""Questions for Anita. One file per question, and ask/OPEN.md listing the open ones.

    python tools/ask.py                                      list open asks
    python tools/ask.py new <cc> --title "..." --summary "<= 40 words"
    python tools/ask.py close <NNN> --ruling "what she decided"

An ask is NEVER a blocker (AGENT_BRIEF.md §3): the file states the decision you already took and
what reversing it costs, and the work ships either way. ask/OPEN.md is the one file she reads.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASK = ROOT / "ask"
LOCK = ASK / ".lock"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def asks():
    out = []
    for p in sorted(ASK.glob("[0-9][0-9][0-9]-*.md")):
        t = p.read_text(encoding="utf-8")
        st = re.search(r"^status:\s*(\w+)", t, re.M)
        ti = re.search(r"^# (.*)$", t, re.M)
        su = re.search(r"^summary:\s*(.*)$", t, re.M)
        out.append({"n": p.name[:3], "file": p.name, "status": st.group(1) if st else "open",
                    "title": ti.group(1) if ti else p.stem, "summary": su.group(1) if su else ""})
    return out


def rewrite_open():
    rows = [a for a in asks() if a["status"] == "open"]
    lines = ["# languagedots: open questions for Anita", "",
             "Each is a decision already taken; the file says what reversing it costs.", ""]
    lines += [f"- **{a['n']}** {a['title']} ({a['file']}): {a['summary']}" for a in rows] or ["(none)"]
    (ASK / "OPEN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="list", choices=["list", "new", "close"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--title")
    ap.add_argument("--summary", default="")
    ap.add_argument("--ruling", default="")
    a = ap.parse_args()
    ASK.mkdir(exist_ok=True)
    if a.cmd == "list":
        for x in asks():
            if x["status"] == "open":
                print(f"{x['n']} {x['title']}  ({x['file']})")
        return
    for _ in range(100):
        try:
            os.close(os.open(LOCK, os.O_CREAT | os.O_EXCL))
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        if a.cmd == "new":
            if not (a.arg and a.title):
                raise SystemExit("new <cc> --title ... --summary ...")
            if len(a.summary.split()) > 40:
                raise SystemExit("summary over 40 words")
            n = max([int(x["n"]) for x in asks()] + [0]) + 1
            p = ASK / f"{n:03d}-{a.arg.lower()}.md"
            p.write_text(
                f"# {a.title}\n\nstatus: open\ncountry: {a.arg.lower()}\nsummary: {a.summary}\n\n"
                "## The decision I already took\n\n\n## What reversing it would cost\n\n\n"
                "## What I need from Anita\n\n", encoding="utf-8")
            print(f"wrote {p}: fill in the three sections now")
        else:
            hits = list(ASK.glob(f"{int(a.arg):03d}-*.md"))
            if not hits:
                raise SystemExit(f"no ask {a.arg}")
            t = hits[0].read_text(encoding="utf-8").replace("status: open", "status: closed", 1)
            t += f"\n## Ruling ({time.strftime('%Y-%m-%d')})\n\n{a.ruling}\n"
            hits[0].write_text(t, encoding="utf-8")
            print(f"closed {hits[0].name}")
        rewrite_open()
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
