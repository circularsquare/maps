"""Replace task lines in TASKS.md with structured blocks (T-3394).

A block file holds blocks of this shape, one per task:

    - T-3264 [P2] {L:trains-drawn} {waits:Anita} Trains drawn through each other
      - Need from Anita: one sentence
      - Context: ...
      - Goal: ...
      - Other: ...

Blank lines, markdown headings and ``` fences between blocks are ignored. Each block's head keeps the
id, priority and tags of the task's current line in TASKS.md (the title text may change); fields use only
the four labels, in that order, one line each; Context and Goal are required; "Need from Anita" is there
exactly when the head has {waits:Anita}. The task's line, and any indented lines under it, are replaced
in place; every other line of TASKS.md is left as it is, line endings included.

    venv\\Scripts\\python.exe trainworld/tools/restructure_tasks.py --check blocks/*.md
    venv\\Scripts\\python.exe trainworld/tools/restructure_tasks.py --snapshot snap.md --keep-old \\
        trainworld/notes/tasks-before-T-3394.md blocks/*.md

--snapshot names the copy of TASKS.md the blocks were written from: a task whose entry has changed
since is refused, so a block never undoes someone else's edit. --retag T-1,T-2 lets those blocks change
their tags (a stale {waits:Anita}, say); everything else about tags is refused. --keep-old appends the
replaced entries, verbatim, to a file agents can grep. Nothing is written if any block has an error.
"""

import argparse
import os
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

TW = Path(__file__).resolve().parent.parent

LABELS = ["Need from Anita", "Context", "Goal", "Other"]
REQUIRED = {"Context", "Goal"}
WAITS_ANITA = "{waits:Anita}"

HEAD_RE = re.compile(r"^- (T-\d+) \[(P\d)\]((?:\s*\{[^{}]*\})*)\s*(.*)$")
TASK_RE = re.compile(r"^- (T-\d+)\b")
FIELD_RE = re.compile(r"^ {2,}- ([^:]+): (.*)$")
TAG_RE = re.compile(r"\{[^{}]*\}")
TAKING_RE = re.compile(r"\btaking,")
DASHES = ("—", "–")


def norm_tag(tag):
    return re.sub(r"\s+", "", tag)


def parse_head(line):
    m = HEAD_RE.match(line)
    if not m:
        return None
    tid, pri, tagpart, title = m.groups()
    return {"id": tid, "pri": pri, "tags": [norm_tag(t) for t in TAG_RE.findall(tagpart)],
            "title": title.strip()}


def read_text(path):
    """Text with lines split, the line ending used and whether the file ends with one."""
    data = Path(path).read_bytes()
    text = data.decode("utf-8")
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    eol = "\r\n" if crlf > lf else "\n"
    mixed = crlf and lf
    lines = text.splitlines()
    return data, lines, eol, text.endswith(("\n", "\r")), mixed


def parse_blocks(path, errors):
    """Blocks of one file: list of dicts with head, fields, src (file:line)."""
    _, lines, _, _, _ = read_text(path)
    blocks, cur = [], None
    for n, raw in enumerate(lines, 1):
        where = f"{path}:{n}"
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith(("#", "```")):
            cur = None  # a block ends at a blank line, heading or fence
            continue
        if TASK_RE.match(line):
            head = parse_head(line)
            if head is None:
                errors.append((where, None, "head line not '- T-#### [P#] {tags} title'"))
                cur = None
                continue
            cur = {"head": head, "line": line, "fields": [], "src": where}
            blocks.append(cur)
            continue
        fm = FIELD_RE.match(line)
        if fm and cur is not None:
            cur["fields"].append((fm.group(1).strip(), fm.group(2).strip(), where))
            continue
        tid = cur["head"]["id"] if cur else None
        errors.append((where, tid, f"line is neither a task head nor a field: {line[:60]!r}"))
    return blocks


def check_block(b, cur_head, retag, errors, warnings):
    tid, head = b["head"]["id"], b["head"]
    where = b["src"]
    labels = [f[0] for f in b["fields"]]
    for label, text, fwhere in b["fields"]:
        if label not in LABELS:
            errors.append((fwhere, tid, f"unknown field label {label!r} (use {', '.join(LABELS)})"))
        if not text:
            errors.append((fwhere, tid, f"{label} is empty"))
    known = [lab for lab in labels if lab in LABELS]
    if len(set(known)) != len(known):
        errors.append((where, tid, "a field appears twice"))
    elif known != sorted(known, key=LABELS.index):
        errors.append((where, tid, "fields out of order (Need from Anita, Context, Goal, Other)"))
    for lab in REQUIRED:
        if lab not in labels:
            errors.append((where, tid, f"{lab} missing"))
    waits = WAITS_ANITA in head["tags"]
    has_need = "Need from Anita" in labels
    if waits and not has_need:
        errors.append((where, tid, "{waits:Anita} but no 'Need from Anita'"))
    if has_need and not waits:
        errors.append((where, tid, "'Need from Anita' without {waits:Anita}"))
    if cur_head is not None:
        if head["pri"] != cur_head["pri"]:
            errors.append((where, tid, f"priority {head['pri']} differs from TASKS.md's {cur_head['pri']}"))
        if head["tags"] != cur_head["tags"]:
            msg = f"tags {' '.join(head['tags'])} differ from TASKS.md's {' '.join(cur_head['tags'])}"
            if tid in retag:
                warnings.append((where, tid, "retagged: " + msg))
            else:
                errors.append((where, tid, msg + " (pass --retag to allow)"))
        if TAKING_RE.search(cur_head["raw"]) and not TAKING_RE.search(b["line"]):
            errors.append((where, tid, "TASKS.md's line has a 'taking,' mark; keep it in the head line"))
    title = head["title"]
    if not title:
        errors.append((where, tid, "title missing"))
    else:
        words = len(TAKING_RE.split(title)[0].split())
        if not 2 <= words <= 10:
            warnings.append((where, tid, f"title has {words} words (2-10)"))
        if title.endswith("."):
            warnings.append((where, tid, "title ends with a full stop"))
    for label, text, fwhere in b["fields"]:
        if any(d in text for d in DASHES):
            warnings.append((fwhere, tid, f"{label} has an em or en dash"))
        if label == "Need from Anita" and re.search(r"[.?!] +[A-Z]", text):
            warnings.append((fwhere, tid, "Need from Anita looks like more than one sentence"))


def entries(lines):
    """{id: [(start, end)]}: each task head line and the indented lines under it."""
    out, i = {}, 0
    while i < len(lines):
        m = TASK_RE.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and lines[j][:1] in (" ", "\t") and lines[j].strip():
            j += 1
        out.setdefault(m.group(1), []).append((i, j))
        i = j
    return out


def render(b):
    out = [b["line"].rstrip()]
    for label, text, _ in b["fields"]:
        out.append(f"  - {label}: {text}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("blocks", nargs="+", help="block files")
    ap.add_argument("--tasks", default=str(TW / "TASKS.md"))
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--snapshot", help="the copy of TASKS.md the blocks were written from")
    ap.add_argument("--retag", default="", help="comma-separated ids whose tags may change")
    ap.add_argument("--keep-old", help="append replaced entries verbatim to this file")
    args = ap.parse_args(argv)
    retag = {t.strip() for t in args.retag.split(",") if t.strip()}

    errors, warnings = [], []
    blocks = []
    for path in args.blocks:
        blocks += parse_blocks(path, errors)

    data, lines, eol, final_eol, mixed = read_text(args.tasks)
    if mixed:
        warnings.append((args.tasks, None, f"mixed line endings; writing all as {eol!r}"))
    where_is = entries(lines)
    snap = None
    if args.snapshot:
        _, slines, _, _, _ = read_text(args.snapshot)
        snap = (slines, entries(slines))

    seen = {}
    plan = {}
    for b in blocks:
        tid = b["head"]["id"]
        if tid in seen:
            errors.append((b["src"], tid, f"id also in {seen[tid]}"))
            continue
        seen[tid] = b["src"]
        spans = where_is.get(tid, [])
        if len(spans) != 1:
            errors.append((b["src"], tid, "not in TASKS.md" if not spans else "in TASKS.md more than once"))
            check_block(b, None, retag, errors, warnings)
            continue
        s, e = spans[0]
        cur_head = parse_head(lines[s])
        if cur_head is None:
            errors.append((b["src"], tid, "TASKS.md's line has no priority: fix it there first"))
            continue
        cur_head["raw"] = lines[s]
        check_block(b, cur_head, retag, errors, warnings)
        if snap is not None:
            sspans = snap[1].get(tid, [])
            old = snap[0][sspans[0][0]:sspans[0][1]] if len(sspans) == 1 else None
            if old != lines[s:e]:
                errors.append((b["src"], tid, "TASKS.md's entry changed since the snapshot: re-read it"))
        plan[tid] = (s, e, render(b))

    changed = {t: p for t, p in plan.items() if lines[p[0]:p[1]] != p[2]}
    for where, tid, msg in errors:
        print(f"ERROR {tid or '-'} {where}: {msg}")
    for where, tid, msg in warnings:
        print(f"warn  {tid or '-'} {where}: {msg}")
    print(f"{len(blocks)} blocks, {len(changed)} to change, {len(plan) - len(changed)} unchanged, "
          f"{len(errors)} errors, {len(warnings)} warnings; TASKS.md line ending {eol!r}")
    if errors:
        if not args.check:
            print("Nothing written.")
        return 1
    if args.check or not changed:
        return 0

    out = []
    i = 0
    starts = {p[0]: p for p in changed.values()}
    while i < len(lines):
        if i in starts:
            s, e, new = starts[i]
            out += new
            i = e
        else:
            out.append(lines[i])
            i += 1
    text = eol.join(out) + (eol if final_eol else "")

    if Path(args.tasks).read_bytes() != data:
        print("TASKS.md changed while this ran; nothing written. Run it again.")
        return 1
    tdir = os.path.dirname(os.path.abspath(args.tasks))
    fd, tmp = tempfile.mkstemp(dir=tdir, prefix=".tasks-", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(text.encode("utf-8"))
    os.replace(tmp, args.tasks)
    print(f"Wrote {args.tasks}: {len(changed)} tasks replaced.")

    if args.keep_old:
        kp = Path(args.keep_old)
        head = "" if kp.exists() else ("# Task lines before T-3394's restructure\n\nEach entry as it stood in "
                                         "TASKS.md just before its block replaced it. Grep, don't read.\n")
        with kp.open("a", encoding="utf-8", newline="") as f:
            f.write(head.replace("\n", eol))
            f.write(f"{eol}## {date.today().isoformat()}{eol}{eol}")
            for tid in sorted(changed, key=lambda t: changed[t][0]):
                s, e, _ = changed[tid]
                f.write(eol.join(lines[s:e]) + eol)
        print(f"Kept the old entries in {kp}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
