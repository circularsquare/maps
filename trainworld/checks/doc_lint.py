"""Doc-state lint: one fact kept in two files, stale task lines, oversized entries, doc drift, dead notes links.

Read-only, no browser, a few seconds. One line per finding, `path:line: [kind] what`, exit 1 if any.
Run from anywhere: `..\\venv\\Scripts\\python.exe trainworld/checks/doc_lint.py` (maps root).

Kinds:
  both          a task id with a line in TASKS.md and in ARCHIVE.md (an ARCHIVE line saying part or stage is allowed)
  stages-done   a TASKS line whose named stages (stages T-x to T-y, built as ...) are all archived: archive the parent
  unblocked     a TASKS line whose waits on / after / once ids are all archived
  for-anita     a line of notes/for-anita.md that opens with an archived id, outside a heading or a section of finished work
  long-task     a TASKS line over 400 characters (only with --long or --kind)
  long-entry    an ARCHIVE entry over 60 words or a DECISIONS line over 80 words, dated on or after --since (only with --long or --kind)
  empty-heading a heading in TASKS.md with nothing under it
  update        an "Update:" paragraph in SPEC.md or docs/
  long-section  a leaf section of SPEC.md or docs/ over 150 lines
  dead-name     a backticked identifier or code path in SPEC.md or docs/ that no code defines
  key           a key letter in docs that toolKeys.ts and diagram/view.ts do not bind to that action
  dead-link     a notes/ link in TASKS, handoff, for-anita, an open task's note or (with --all-links) SPEC, CLAUDE,
                docs and code comments that resolves neither in notes/ nor in notes/archive/<band>/
"""
import argparse
import collections
import os
import re
import sys

TW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(TW, "notes")
# the size checks on ARCHIVE and DECISIONS cover entries dated on or after this day
SIZE_CHECK_SINCE = "2026-11-05"
LONG_TASK_CHARS = 400
LONG_ARCHIVE_WORDS = 60
LONG_DECISION_WORDS = 80
LONG_SECTION_LINES = 150

findings = []  # (path, line, kind, text)


def rel(p):
    return os.path.relpath(p, TW).replace("\\", "/")


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def lines_of(p):
    return read(p).splitlines()


def add(path, line, kind, text):
    findings.append((rel(path) if os.path.isabs(path) else path, line, kind, text))


IDS = r"T-\d+(?:(?:\s*(?:,|and|to|/|\+|&)\s*)T-\d+)*"


def id_list(s):
    return [int(x) for x in re.findall(r"T-(\d+)", s)]


# ---------- tasks and archive ----------
tasks_path = os.path.join(TW, "TASKS.md")
task_lines = lines_of(tasks_path)
tasks = {}  # id -> line number
task_text = {}
for i, line in enumerate(task_lines, 1):
    m = re.match(r"^- T-(\d+)\b", line)
    if m:
        tid = int(m.group(1))
        tasks.setdefault(tid, i)
        task_text[i] = (tid, line)

ARCH_START = re.compile(r"^\s*-\s*\**((?:T-\d+(?:\s*(?:,|/|and|\+|to|&)\s*)?)+)")
archived = {}  # id -> (file, line, text)
archive_files = [os.path.join(TW, "ARCHIVE.md"), os.path.join(TW, "ARCHIVE-to-2026-10-19.md")]
archive_lines = {}
for ap in archive_files:
    if not os.path.exists(ap):
        continue
    archive_lines[ap] = lines_of(ap)
    for i, line in enumerate(archive_lines[ap], 1):
        m = ARCH_START.match(line)
        if m:
            for tid in id_list(m.group(1)):
                archived.setdefault(tid, (ap, i, line))

# 1. both
for tid, ln in sorted(tasks.items()):
    if tid in archived:
        ap, al, text = archived[tid]
        if re.search(r"\b(part|stage|stages|parts|rule \d|L\d(?:\+L\d)?)\b", text, re.I):
            continue
        add(tasks_path, ln, "both", "T-%d is also in %s:%d" % (tid, os.path.basename(ap), al))

# 2 and 3. stages and waits
STAGE_RE = re.compile(r"(?:\bstages?|built as|split into|filed as|\bparts?)[ :]+(?:\w+ ){0,2}(%s)" % IDS)
WAIT_RE = re.compile(r"(?:waits? on|waiting on|\bafter|\bonce|\buntil|blocked by)\s+(?:the\s+)?(%s)" % IDS)
for ln, (tid, line) in sorted(task_text.items()):
    own = {tid}
    sg = set()
    for m in STAGE_RE.finditer(line):
        sg |= set(id_list(m.group(1)))
    sg -= own
    if sg and all(x in archived and x not in tasks for x in sg):
        add(tasks_path, ln, "stages-done", "T-%d: stages %s are all archived, so archive the parent"
            % (tid, ", ".join("T-%d" % x for x in sorted(sg))))
    wt = set()
    for m in WAIT_RE.finditer(line):
        wt |= set(id_list(m.group(1)))
    wt -= own
    if wt and all(x in archived and x not in tasks for x in wt):
        add(tasks_path, ln, "unblocked", "T-%d: waits on %s, all archived"
            % (tid, ", ".join("T-%d" % x for x in sorted(wt))))

# 4. for-anita
fa_path = os.path.join(NOTES, "for-anita.md")
if os.path.exists(fa_path):
    heading = ""
    for i, line in enumerate(lines_of(fa_path), 1):
        if line.startswith("#"):
            heading = line
            continue
        if re.search(r"\b(done|built|overnight|rewording|answered|shipped)\b", heading, re.I):
            continue
        # an id that opens a line or bullet is its subject; one in passing is where a change came from
        m = re.match(r"^\s*(?:[-*]\s+)?\**(T-\d+)\b", line)
        if m:
            tid = int(m.group(1)[2:])
            if tid in archived and tid not in tasks:
                add(fa_path, i, "for-anita", "T-%d is archived" % tid)

# 5. sizes
for i, line in sorted(task_text.items()):
    if len(line[1]) > LONG_TASK_CHARS:
        add(tasks_path, i, "long-task", "T-%d is %d characters; the rest belongs in its notes file" % (line[0], len(line[1])))


def since_ok(date, since):
    return bool(date) and date >= since


def run_size_checks(since):
    ap = archive_files[0]
    for i, line in enumerate(archive_lines.get(ap, []), 1):
        if not ARCH_START.match(line):
            continue
        m = re.search(r"\b(20\d\d-\d\d-\d\d)\b", line)
        if m and since_ok(m.group(1), since) and len(line.split()) > LONG_ARCHIVE_WORDS:
            add(ap, i, "long-entry", "%d words (ARCHIVE entries are about 40, at most 2 lines)" % len(line.split()))
    dp = os.path.join(TW, "DECISIONS.md")
    for i, line in enumerate(lines_of(dp), 1):
        m = re.match(r"^- (20\d\d-\d\d-\d\d)\b", line)
        if m and since_ok(m.group(1), since) and len(line.split()) > LONG_DECISION_WORDS:
            add(dp, i, "long-entry", "%d words (a DECISIONS line is one or two sentences)" % len(line.split()))


# 6. empty headings in TASKS.md
for i, line in enumerate(task_lines, 1):
    m = re.match(r"^(#+) ", line)
    if not m:
        continue
    level = len(m.group(1))
    j = i  # index of the next line
    while j < len(task_lines) and not task_lines[j].strip():
        j += 1
    if j >= len(task_lines):
        add(tasks_path, i, "empty-heading", line.strip("# ").strip())
        continue
    n = re.match(r"^(#+) ", task_lines[j])
    if n and len(n.group(1)) <= level:
        add(tasks_path, i, "empty-heading", line.strip("# ").strip())

# ---------- SPEC and docs ----------
doc_paths = [os.path.join(TW, "SPEC.md")] + sorted(
    os.path.join(TW, "docs", f) for f in os.listdir(os.path.join(TW, "docs")) if f.endswith(".md"))

# 7a. Update paragraphs, 7b. long leaf sections
for p in doc_paths:
    ls = lines_of(p)
    infence = False
    heads = []  # (line number, level)
    for i, line in enumerate(ls, 1):
        if line.startswith("```"):
            infence = not infence
        if infence:
            continue
        if re.match(r"^\**Update\b\s*[:(]", line) or re.search(r"(^|[.!?]\s)\**Update:\**", line):
            add(p, i, "update", "an Update paragraph; rewrite the section to state the new design")
        if re.match(r"^#+ ", line):
            heads.append((i, len(re.match(r"^(#+)", line).group(1))))
    for k, (ln, lv) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(ls) + 1
        is_leaf = k + 1 >= len(heads) or heads[k + 1][1] <= lv
        if is_leaf and end - ln > LONG_SECTION_LINES:
            add(p, ln, "long-section", "%d lines under \"%s\"" % (end - ln, ls[ln - 1].strip("# ").strip()[:60]))
    # a file with no headings under its title
    if len(heads) <= 1 and len(ls) > LONG_SECTION_LINES and not any(f[0] == rel(p) and f[2] == "long-section" for f in findings):
        add(p, 1, "long-section", "%d lines with no subheadings" % len(ls))

# 7c. dead names (a backticked identifier or code path no code defines)
CODE_DIRS = ["app", "sim", "pipeline", "tools", "checks"]
SKIP_DIRS = {"node_modules", "target", "dist", ".git", "pkg", "__pycache__"}
CODE_EXT = (".ts", ".tsx", ".js", ".mjs", ".rs", ".py", ".toml", ".cmd", ".ps1", ".html", ".css")
code_blobs = []
basenames = collections.defaultdict(list)
for d in CODE_DIRS:
    for root, dirs, files in os.walk(os.path.join(TW, d)):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        # an `old/` folder's files exist (docs may name them) but define no live identifier (T-2504)
        retired = "old" in os.path.relpath(root, TW).split(os.sep)
        for fn in files:
            p = os.path.join(root, fn)
            basenames[fn].append(rel(p))
            if not retired and fn.endswith(CODE_EXT) and os.path.getsize(p) < 2_000_000:
                code_blobs.append(read(p))
for fn in os.listdir(os.path.dirname(TW)):  # the maps root's own scripts (cores.py, serve.py)
    if fn.endswith(".py"):
        basenames[fn].append(fn)
for d in ("docs", "notes", "research"):
    for root, dirs, files in os.walk(os.path.join(TW, d)):
        for fn in files:
            basenames[fn].append(rel(os.path.join(root, fn)))
code_blob = "\n".join(code_blobs)
code_idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", code_blob))
# code paths only: a data file (a pack's json) is made by a run, not kept in the tree
PATH_RE = re.compile(r"^([\w./\-*{}]+\.(?:ts|tsx|rs|py|mjs|js|cmd|toml|css|html))(?:[: ].*)?$")
IDENT_RE = re.compile(r"(?<![\w.?=-])([a-z]+[A-Z]\w*|[a-z]+_[a-z_0-9]+|[A-Z][a-z]+[A-Z]\w*|[A-Z]+_[A-Z_0-9]*[A-Z0-9])(?![\w.])")
# a name ending in an underscore is a prefix (NEAR_BAR_, HS_RUN_COST_)
DEAD_NAME_OK_PREFIX = ("NEAR_BAR_",)
for p in doc_paths:
    infence = False
    for i, line in enumerate(lines_of(p), 1):
        if line.startswith("```"):
            infence = not infence
        if infence:
            continue
        for tok in re.findall(r"`([^`\n]{2,120})`", line):
            tok = tok.strip()
            m = PATH_RE.match(tok)
            if m:
                path = m.group(1)
                if "*" in path or "{" in path or path.startswith("data/") or "/packs/" in path:
                    continue
                cands = basenames.get(os.path.basename(path), [])
                if not (cands and (("/" not in path) or any(c.endswith(path.lstrip("./")) for c in cands))):
                    add(p, i, "dead-name", "`%s` is no file" % path)
                continue
            for ident in IDENT_RE.findall(tok):
                if len(ident) < 5 or ident.startswith(DEAD_NAME_OK_PREFIX):
                    continue
                if ident not in code_idents:
                    add(p, i, "dead-name", "`%s` is in no code" % ident)

# 7d. keys
tk = read(os.path.join(TW, "app", "src", "ui", "toolKeys.ts"))
dv = read(os.path.join(TW, "app", "src", "diagram", "view.ts"))


def const_str(src, name):
    m = re.search(r"const %s\s*=\s*\"(\w+)\"" % name, src)
    return m.group(1).lower() if m else None


code_keys = {}  # action word -> key
for name, action in (("BUS_KEY", "bus"), ("FERRY_KEY", "ferry"), ("DEMAND_KEY", "demand"), ("TRIP_KEY", "trip")):
    k = const_str(tk, name)
    if k:
        code_keys[action] = k
k = const_str(dv, "DIAGRAM_KEY")
if k:
    code_keys["diagram"] = k
for m in re.finditer(r"\[\"(\w+)\", \"([^\"]+)\", Icon\.\w+, \"(\d)\"", tk):
    code_keys[{"track": "track tool", "station": "station tool", "delete": "delete tool", "blueprint": "select tool"}.get(m.group(1), m.group(1))] = m.group(3)
am = re.search(r"ACTION_KEYS[^=]*=\s*\{([^}]*)\}", tk)
if am:
    for kk, v in re.findall(r"(\w):\s*\"(\w+)\"", am.group(1)):
        code_keys[{"newLine": "new line", "editStops": "edit stops", "construct": "construct", "addPassed": "add passed"}.get(v, v)] = kk
tm = re.search(r"TAB_KEYS[^=]*=\s*\{([^}]*)\}", tk)
tab_keys = dict(re.findall(r"(\w):\s*\"(\w+)\"", tm.group(1))) if tm else {}
for kk, v in tab_keys.items():
    code_keys[v + " tab"] = kk

# the one table of keys: ui-panels.md "Keys and focus"
panels = os.path.join(TW, "docs", "ui-panels.md")
if os.path.exists(panels):
    pl = lines_of(panels)
    in_keys = False
    table = set()
    table_first = 0
    for i, line in enumerate(pl, 1):
        if line.startswith("## "):
            in_keys = line.startswith("## Keys and focus")
            continue
        if in_keys and line.startswith("|") and not line.startswith("|---") and not line.startswith("| Key"):
            if not table_first:
                table_first = i
            first = line.split("|")[1]
            for tok in re.split(r",\s*", first.strip()):
                tok = tok.strip().strip("`")
                if re.fullmatch(r"[A-Za-z0-9]", tok):
                    table.add(tok.lower())
    if table:
        want = set(v for v in code_keys.values() if len(v) == 1)
        # the table lists the tool numbers by range (1, 2, 3, 4), so single keys compare directly
        for kk in sorted(want - table):
            action = [a for a, v in code_keys.items() if v == kk][0]
            add(panels, table_first, "key", "key %s (%s) is bound in code but not in the key table" % (kk.upper(), action))
        for kk in sorted(table - want - {"d", "e"}):
            add(panels, table_first, "key", "key %s is in the key table but bound to nothing in toolKeys.ts" % kk.upper())

CLAIMS = [
    (r"Select tool", "select tool"),
    (r"Edit stops", "edit stops"),
    (r"[Ff]erry (?:tool|line)", "ferry"),
    (r"[Dd]iagram", "diagram"),
    (r"Construct", "construct"),
    (r"[Bb]us (?:tool|line)", "bus"),
    (r"trip (?:tool|explorer)", "trip"),
]
for p in doc_paths:
    if os.path.basename(p) == "ui-panels.md":
        continue
    for i, line in enumerate(lines_of(p), 1):
        for pat, action in CLAIMS:
            for m in re.finditer(r"(?:%s)[^.\n;()]{0,25}\((?:hover and )?key \**([A-Za-z0-9])\**\b" % pat, line):
                got = m.group(1).lower()
                if action in code_keys and got != code_keys[action]:
                    add(p, i, "key", "says %s is key %s; toolKeys.ts binds %s" % (action, got.upper(), code_keys[action].upper()))

# ---------- 8. notes links ----------
archive_dirs = []
adir = os.path.join(NOTES, "archive")
if os.path.isdir(adir):
    archive_dirs = [os.path.join(adir, d) for d in os.listdir(adir) if os.path.isdir(os.path.join(adir, d))]
    archive_dirs.append(adir)
LINK_RE = re.compile(r"(?<![\w/.-])notes/([\w][\w.\-]*(?:/[\w.\-]+)*)")


def link_resolves(tok):
    tok = tok.rstrip(".,;:)'\"`")
    # placeholders (T-###.md, todo-*.md, T-236x.md, notes/T-) and the folder name itself
    if not tok or tok.endswith("-") or re.search(r"[0-9]x\b", tok) or tok in ("archive", "T-012.md"):
        return True
    first = tok.split("/")[0]
    if os.path.exists(os.path.join(NOTES, tok)):
        return True
    if tok.startswith("archive/"):
        return os.path.exists(os.path.join(NOTES, tok))
    for d in archive_dirs:
        if os.path.exists(os.path.join(d, tok)):
            return True
    return False


def check_links(path):
    infence = False
    for i, line in enumerate(lines_of(path), 1):
        for m in LINK_RE.finditer(line):
            tok = m.group(1)
            # "notes/T-###.md" style placeholders, and prose such as "notes/ is the agents'"
            if not link_resolves(tok):
                add(path, i, "dead-link", "notes/%s resolves nowhere" % tok.rstrip(".,;:)'\"`"))


def run_link_checks(all_links):
    paths = [tasks_path, os.path.join(NOTES, "handoff.md"), fa_path]
    for tid in sorted(tasks):
        for cand in (os.path.join(NOTES, "T-%d.md" % tid),):
            if os.path.exists(cand):
                paths.append(cand)
    if all_links:
        paths += [os.path.join(TW, "CLAUDE.md")] + doc_paths
        for d in ("app/src", "sim/src", "pipeline", "tools", "checks"):
            for root, dirs, files in os.walk(os.path.join(TW, d)):
                dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
                for fn in files:
                    if fn.endswith((".ts", ".tsx", ".mjs", ".js", ".rs", ".py")) and fn != "doc_lint.py":
                        paths.append(os.path.join(root, fn))
    seen = set()
    for p in paths:
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        check_links(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", default=SIZE_CHECK_SINCE, help="size checks on ARCHIVE and DECISIONS cover entries dated on or after this (default %(default)s)")
    ap.add_argument("--all-links", action="store_true", help="also check notes/ links in SPEC, CLAUDE.md, docs and code comments")
    ap.add_argument("--kind", action="append", help="only these kinds")
    ap.add_argument("--long", action="store_true", help="also report long-task and long-entry (off by default: the caps are not followed and cost nothing when files are grepped, T-2831)")
    args = ap.parse_args()
    run_size_checks(args.since)
    run_link_checks(args.all_links)
    out = findings
    if not args.long and not args.kind:
        out = [f for f in out if f[2] not in ("long-task", "long-entry")]
    if args.kind:
        out = [f for f in out if f[2] in args.kind]
    out = sorted(out, key=lambda f: (f[0], f[1], f[2]))
    for path, line, kind, text in out:
        print("%s:%d: [%s] %s" % (path, line, kind, text))
    counts = collections.Counter(f[2] for f in out)
    print("%d findings %s" % (len(out), dict(sorted(counts.items()))), file=sys.stderr)
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main())
