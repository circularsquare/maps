"""Claim a country before working on it, so two agents do not build the same one.

Anita runs two or three agents on this directory at once (spec §12). That is by design and
mostly fine — shared files change under you and surgical edits cope. What does NOT cope is
two sessions independently deciding to build the same country: on 2026-09-08 one session
spent an hour on Peru while another had already finished it, and overwrote its `sources/pe.py`
with a `Write` to a path it had not checked. Nothing was lost (Claude Code's file history had
it), but the hour was.

    python tools/claim.py                       # what is claimed, parked, free, and held
    python tools/claim.py take pe --id <sid>    # claim Peru
    python tools/claim.py drop pe --id <sid>    # release it
    python tools/claim.py park pe --id <sid>    # release it and leave a handoff for the next one
    python tools/claim.py done pe --id <sid>    # release it; if registered, mark it drawn in queue.csv
    python tools/claim.py mine --id <sid>       # what you are holding

`<sid>` is your session id. You know it: it is the last path component of the scratchpad
directory named in your system prompt, e.g. `6d04f949-0a26-4dae-9794-a3e536a1e493`. Any
stable string works; the point is that it is yours and not somebody else's.

**WHY ONE FILE PER COUNTRY RATHER THAN ONE SHARED LIST.** Two agents editing a single
`queue.md` to tick a box is the exact collision this is meant to prevent — it is
`index.html` again. Each claim is its own file created with `O_CREAT|O_EXCL`, which is atomic
on Windows and POSIX alike, so a race has exactly one winner and the loser is told who won.

Claims live under `data/claims/`, which is already gitignored — they are session state, not
project history, and they must never reach a commit.

**A CLAIM IS ADVISORY AND CANNOT STOP ANYONE.** It is a note on the door, not a mutex. The
thing that actually prevents the Peru accident is the habit in spec §12: **check before you
write.** This just makes checking one command instead of an inference from `git status`.

**THE QUEUE IS `queue.csv`; `queue.md` KEEPS THE REASONING** (WORKFLOW_PLAN.md item 10,
2026-09-14). The candidate list used to be read off `queue.md` table rows by regex, which
listed closed, held and drawn countries as free: a table row carries no status, and a ruling
or a closure written as a bullet is invisible to a regex. `queue.csv` has one row per country
code, with the columns

    cc, name, status, grain, source, blocker, held_for_anita, detail

and `status` is one of

    free      worth a build or a scout now
    held      waiting on Anita; `held_for_anita` names the ask, e.g. `ask 017`
    deferred  Anita put it off (South Sudan); not for an unprompted session
    blocked   a route is known and something outside the project stops it: an account, a
              network wall, a release not out yet; `blocker` says which
    closed    a negative is recorded; `detail` says where (and nothing is truly dead, spec §12)
    drawn     registered as a country; `held_for_anita` may still name an open ask about it

`grain`, `source` and `blocker` are a few words each. `detail` points at the prose: the
`queue.md` section heading, then the `sources.md` section where one is cited. Row order is the
queue order, so the free list prints best first; put a new row where it ranks, and drawn rows
sit at the end in code order. Edit it by hand when a status changes in `queue.md`; `done` sets
`drawn` itself, writing a temp file and `os.replace`-ing it so a reader never sees half a file.

The listing warns, one line each, where the files disagree: a registered country whose row is
not `drawn`; a code in a `queue.md` table row with no `queue.csv` row, so a new queue row cannot
be missed silently; and a `queue.csv` code that `queue.md` never mentions by code or name.
Drawn rows are exempt from that last one, since their prose is the country entry and
`sources/<cc>.md`.

An open upgrade to a drawn country is a `drawn` row whose `blocker` starts `upgrade free`. The free
list skips registered countries, so the listing prints those apart, under UPGRADES FREE.
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLAIMS = os.path.join(ROOT, "data", "claims")
QUEUE_MD = os.path.join(ROOT, "queue.md")
QUEUE_CSV = os.path.join(ROOT, "queue.csv")
COLUMNS = ["cc", "name", "status", "grain", "source", "blocker", "held_for_anita", "detail"]
STATUSES = ("free", "held", "deferred", "blocked", "closed", "drawn")

# Handoffs are NOT under data/, which is gitignored — a parked country is the one piece of
# state here that rots, and it has to survive a clean checkout and show up in `git status`.
HANDOFF = os.path.join(ROOT, "handoff")

HANDOFF_TEMPLATE = """# {cc} — parked {date}, session `{sid}`

*Written because I was running out of context, not because anything is wrong.*
*`AGENT_BRIEF.md` §4: take a parked country before a fresh one, it is cheaper.*

## Last COMMANDS.txt step completed

<!-- The NUMBER, e.g. "step 2, geo fetched" or "checkpoint B: normalized CSV reconciles".
     This is the single most useful line in the file. -->

## What is on disk

<!-- Paths, and whether each one is trustworthy. data/raw/... , data/normalized/{cc}.csv ,
     data/geo/{cc}/ , sources/{cc}.py , taxonomy/{cc}YYYY.py .  Say which are stubs. -->

## What I was about to do

<!-- The next concrete action, not the goal. -->

## The one thing that will bite you

<!-- The join that nearly went wrong, the column that lies, the URL that needs a UA. -->

## Everything else

<!-- Anything not yet written into sources/{cc}.md or sources.md. If it IS written there,
     say so and point at the section instead of repeating it. -->
"""

# A claim older than this is reported as probably abandoned. It is not auto-released: a long
# country legitimately takes hours, and silently stealing a live claim would be worse than
# the collision it is meant to prevent.
STALE_HOURS = 6


def _now():
    return time.time()


def _age(ts):
    h = (_now() - ts) / 3600.0
    if h < 1:
        return f"{h * 60:.0f}m"
    if h < 48:
        return f"{h:.1f}h"
    return f"{h / 24:.1f}d"


def drawn():
    """Country codes currently registered. Derived, never typed — spec §12.

    The `countries/<cc>.py` files once WORKFLOW_PLAN.md item 9 has split the registry and that
    directory has any; until then the dict keys in `countries.py`. `tools/negatives.py`
    imports this, so it stays a set of codes."""
    d = os.path.join(ROOT, "countries")
    if os.path.isdir(d):
        codes = {m.group(1) for m in (re.fullmatch(r"([a-z]{2})\.py", f) for f in os.listdir(d))
                 if m}
        if codes:
            return codes
    path = os.path.join(ROOT, "countries.py")
    if not os.path.exists(path):
        return set()
    src = open(path, encoding="utf-8").read()
    return set(re.findall(r'^    "([a-z]{2})": dict\(', src, re.M))


def half_registered():
    """{cc: which half is missing} where countries/<cc>.py and ORDER in countries.py disagree.

    countries.py skips such a country with a warning instead of stopping (a builder writes the
    two back to back and every other session imports in between), so nothing downstream sees it
    until both halves exist. Read from the text, as drawn() is, so this file never imports the
    registry."""
    try:
        src = open(os.path.join(ROOT, "countries.py"), encoding="utf-8").read()
    except OSError:
        return {}
    m = re.search(r"^ORDER = \[(.*?)^\]", src, re.M | re.S)
    if not m:
        return {}
    order = set(re.findall(r'^\s*"([a-z]{2})",?\s*$', m.group(1), re.M))
    files = drawn()
    out = {cc: f"in ORDER but countries/{cc}.py does not exist" for cc in order - files}
    out.update({cc: f"countries/{cc}.py exists but {cc} is not in ORDER" for cc in files - order})
    return out


def _registered_name(cc):
    """The `name=` of a registered country, for a queue.csv row `done` has to add."""
    # Anchored on the entry's own `"<cc>": dict(` line: a bare `name="` also matches a helper's
    # `sheet_name="..."` above the entry (countries/us.py).
    header = rf'^    "{cc}": dict\(\s*\n\s*name="([^"]*)"'
    for path, pat in ((os.path.join(ROOT, "countries", f"{cc}.py"), header),
                      (os.path.join(ROOT, "countries.py"), header)):
        if os.path.exists(path):
            m = re.search(pat, open(path, encoding="utf-8").read(), re.M)
            if m:
                return m.group(1)
    return cc


def built():
    """Country codes with dots actually on disk, which is a stronger claim than registered."""
    d = os.path.join(ROOT, "data", "processed")
    if not os.path.isdir(d):
        return set()
    return {m.group(1) for m in
            (re.fullmatch(r"dots_([a-z]{2})\.geojson", f) for f in os.listdir(d)) if m}


def waiting_for_build():
    """Countries whose dots are newer than the start of the last build tail, so not on the map.

    Under a supervisor, agents stop after the scatter and the supervisor runs
    `tools/build_tail.py` for all of them at once (WORKFLOW_PLAN.md item 5). A finished tail
    writes `data/build_last.json`. Returns (countries, started), or None before the first run."""
    try:
        with open(os.path.join(ROOT, "data", "build_last.json"), encoding="utf-8") as fh:
            since = json.load(fh)["started"]
    except (OSError, ValueError, KeyError):
        return None
    d = os.path.join(ROOT, "data", "processed")
    out = []
    for cc in sorted(built()):
        paths = [os.path.join(d, f"dots_{cc}.geojson"), os.path.join(d, f"dots_{cc}_10k.geojson")]
        newest = max((os.path.getmtime(p) for p in paths if os.path.exists(p)), default=0)
        if newest > since:
            out.append(cc)
    return out, since


def load_claims():
    if not os.path.isdir(CLAIMS):
        return {}
    out = {}
    for f in sorted(os.listdir(CLAIMS)):
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(CLAIMS, f), encoding="utf-8") as fh:
                out[f[:-5]] = json.load(fh)
        except (OSError, ValueError) as e:
            out[f[:-5]] = {"id": "?", "started": 0, "note": f"unreadable claim file: {e}"}
    return out


def parked():
    """Countries with a handoff note waiting. Resuming one is cheaper than a fresh start."""
    if not os.path.isdir(HANDOFF):
        return {}
    out = {}
    for f in sorted(os.listdir(HANDOFF)):
        m = re.fullmatch(r"([a-z]{2})\.md", f)
        if m:
            p = os.path.join(HANDOFF, f)
            out[m.group(1)] = {"path": p, "mtime": os.path.getmtime(p)}
    return out


def read_queue_csv():
    """(rows, problems): the rows of queue.csv as dicts in file order, and one line per fault.

    `utf-8-sig` so a copy saved from a spreadsheet, which adds a BOM, still reads."""
    if not os.path.exists(QUEUE_CSV):
        return [], ["queue.csv is missing, so nothing is listed as free"]
    rows, problems, seen = [], [], set()
    with open(QUEUE_CSV, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            problems.append(f"queue.csv columns are {reader.fieldnames}, expected {COLUMNS}")
        for r in reader:
            r = {k: (r.get(k) or "").strip() for k in COLUMNS}
            cc, at = r["cc"], f"queue.csv line {reader.line_num}"
            if not re.fullmatch(r"[a-z]{2}", cc):
                problems.append(f"{at}: cc {cc!r} is not two lowercase letters")
                continue
            if cc in seen:
                problems.append(f"{at}: {cc} has a second row, which is ignored")
                continue
            if r["status"] not in STATUSES:
                problems.append(f"{at}: {cc} has status {r['status']!r}, not one of "
                                f"{', '.join(STATUSES)}")
            seen.add(cc)
            rows.append(r)
    return rows, problems


def _write_queue_csv(rows):
    """Temp file and `os.replace`, so a concurrent reader sees the old file or the new one.

    Windows refuses to replace a file another process has open this instant, so retry a few
    times before giving up; a reader holds it for milliseconds."""
    tmp = f"{QUEUE_CSV}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for attempt in range(20):
        try:
            os.replace(tmp, QUEUE_CSV)
            return
        except PermissionError:
            if attempt == 19:
                os.remove(tmp)
                raise
            time.sleep(0.25)


def mark_drawn(cc):
    """Set cc's queue.csv row to `drawn`, moving it into the drawn block, or add a row.

    Returns the old status, None when there was no row, or an error string starting `!!`."""
    rows, problems = read_queue_csv()
    if not os.path.exists(QUEUE_CSV) or any(p.startswith("queue.csv columns") for p in problems):
        return f"!! queue.csv is missing or its columns are wrong, so {cc} was not marked drawn"
    row = next((r for r in rows if r["cc"] == cc), None)
    if row is not None and row["status"] == "drawn":
        return "drawn"
    old = row["status"] if row else None
    if row is None:
        row = dict.fromkeys(COLUMNS, "")
        row.update(cc=cc, name=_registered_name(cc),
                   detail=f"registered; row added by claim.py done {datetime.date.today()}")
    else:
        rows.remove(row)
    row["status"] = "drawn"
    at = next((i for i, r in enumerate(rows) if r["status"] == "drawn" and r["cc"] > cc),
              len(rows))
    rows.insert(at, row)
    _write_queue_csv(rows)
    return old


def read_queue_md():
    """({cc: line} for codes in queue.md table rows, the file's text). For the warnings only.

    A row followed by a `|---|` line is a table header, not a country: `| cc | drawn from |`
    was once listed as a free country called `cc`. Struck-through rows count."""
    if not os.path.exists(QUEUE_MD):
        return {}, ""
    text = open(QUEUE_MD, encoding="utf-8").read()
    lines = text.splitlines()
    out = {}
    for i, line in enumerate(lines):
        m = re.match(r"^\|\s*(?:~~)?`?([a-z]{2})`?(?:~~)?\s*\|", line)
        if not m or (i + 1 < len(lines) and re.match(r"^\|\s*:?-{3,}", lines[i + 1])):
            continue
        out.setdefault(m.group(1), i + 1)
    return out, text


def _mentioned(row, flat):
    """Whether lower-cased, whitespace-collapsed queue.md names this row by `cc` or by name.

    A name in brackets is an alias: `United Arab Emirates (UAE)` matches either."""
    if f"`{row['cc']}`" in flat:
        return True
    names = [re.sub(r"\s*\(.*?\)", "", row["name"]).strip()] + re.findall(r"\((.*?)\)", row["name"])
    return any(n and re.search(rf"(?<![\w-]){re.escape(n.lower())}(?![\w-])", flat) for n in names)


def queue_warnings(rows, reg):
    """One line per disagreement between queue.csv, queue.md and the registry."""
    by_cc = {r["cc"]: r for r in rows}
    md_rows, md_text = read_queue_md()
    out = []
    for cc in sorted(reg):
        if cc not in by_cc:
            out.append(f"{cc} is registered and has no queue.csv row "
                       f"(python tools/claim.py done {cc} --id <sid> adds one)")
        elif by_cc[cc]["status"] != "drawn":
            out.append(f"{cc} is registered but queue.csv says {by_cc[cc]['status']}")
    for cc, line in sorted(md_rows.items()):
        if cc not in by_cc:
            out.append(f"{cc} has a queue.md table row (line {line}) and no queue.csv row")
    flat = re.sub(r"\s+", " ", md_text).lower()
    for r in rows:
        if r["status"] != "drawn" and r["cc"] not in md_rows and not _mentioned(r, flat):
            out.append(f"{r['cc']} is in queue.csv as {r['status']} and queue.md never mentions it")
    return out


def cmd_list(args):
    claims, reg, have = load_claims(), drawn(), built()
    rows, problems = read_queue_csv()

    if claims:
        print(f"CLAIMED ({len(claims)}):")
        for cc, c in sorted(claims.items()):
            age = _age(c.get("started", 0))
            stale = (_now() - c.get("started", 0)) / 3600.0 > STALE_HOURS
            flag = "  <-- looks abandoned" if stale else ""
            print(f"  {cc}  {c.get('id', '?')[:18]:18s} {age:>6s} ago  "
                  f"{str(c.get('note', ''))[:44]}{flag}")
    else:
        print("CLAIMED: nothing")

    park = parked()
    if park:
        print(f"\nPARKED ({len(park)}) — TAKE ONE OF THESE FIRST, they are half-built and "
              f"they rot:")
        for cc, p in sorted(park.items(), key=lambda kv: kv[1]["mtime"]):
            state = "drawn now, delete the handoff" if cc in reg else "resumable"
            print(f"  {cc}  {_age(p['mtime']):>6s} ago  {state}   handoff/{cc}.md")

    print(f"\nDRAWN: {len(reg)} registered, {len(have)} with dots on disk")
    missing = sorted(reg - have)
    if missing:
        print(f"  registered but NOT built: {', '.join(missing)}")
    half = half_registered()
    if half:
        print("  HALF-REGISTERED, so countries.py skips it until both halves exist: "
              + "; ".join(f"{cc} ({why})" for cc, why in sorted(half.items())))
    wait = waiting_for_build()
    if wait is None:
        print("  build tail: no data/build_last.json yet, so what is waiting for it is unknown")
    elif wait[0]:
        print(f"  WAITING FOR THE BUILD TAIL (last run started {_age(wait[1])} ago): "
              f"{', '.join(wait[0])}")
    else:
        print(f"  build tail: nothing waiting (last run started {_age(wait[1])} ago)")

    free = [r for r in rows if r["status"] == "free" and r["cc"] not in claims
            and r["cc"] not in reg]
    print(f"\nQUEUE: {len(rows)} rows in queue.csv, {len(free)} free, unclaimed and undrawn")
    for r in free:
        cells = " | ".join(x for x in (r["grain"], r["source"], r["blocker"]) if x)
        print(f"  {r['cc']}  {r['name'][:26]:26s} {cells[:72]}")
    for status, label in (("held", "HELD for Anita"), ("deferred", "DEFERRED by Anita"),
                          ("blocked", "BLOCKED outside the project")):
        sel = [r for r in rows if r["status"] == status]
        if sel:
            codes = ", ".join(f"{r['cc']} ({r['held_for_anita']})" if r["held_for_anita"]
                              else r["cc"] for r in sel)
            print(f"  {label} ({len(sel)}): {codes}")
    n = {s: sum(r["status"] == s for r in rows) for s in ("closed", "drawn")}
    print(f"  closed {n['closed']}, drawn {n['drawn']}; each row's detail names the queue.md "
          f"section with the reason")

    # Scouts record an open upgrade to a drawn country as a `drawn` row whose blocker starts
    # "upgrade free". The free list above skips registered countries, so it cannot show them.
    upgrades = [r for r in rows if r["status"] == "drawn"
                and r["blocker"].lower().startswith("upgrade free")]
    if upgrades:
        taken = [r["cc"] for r in upgrades if r["cc"] in claims]
        open_ = [r for r in upgrades if r["cc"] not in claims]
        print(f"\nUPGRADES FREE to drawn countries ({len(open_)} unclaimed"
              + (f"; claimed: {', '.join(taken)}" if taken else "") + "):")
        for r in open_:
            verdict = re.sub(r"^upgrade free:?\s*", "", r["blocker"], flags=re.I).split("; ")[0]
            print(f"  {r['cc']}  {r['name'][:26]:26s} {verdict[:110]}")
            refs = [x for x in r["detail"].split("; ") if x.startswith("sources")]
            print(f"      see {'; '.join(refs) or r['detail']}"[:120])

    warn = problems + queue_warnings(rows, reg)
    if warn:
        print(f"\nQUEUE WARNINGS ({len(warn)}): fix queue.csv, or add the missing prose to queue.md")
        for w in warn:
            print(f"  !! {w}")


def cmd_take(args):
    cc = args.cc.lower()
    reg = drawn()
    os.makedirs(CLAIMS, exist_ok=True)
    path = os.path.join(CLAIMS, f"{cc}.json")

    if cc in reg:
        print(f"!! {cc} is ALREADY REGISTERED. If you are re-working it that is fine, but check "
              f"sources/{cc}.md first — somebody built it.")
    row = next((r for r in read_queue_csv()[0] if r["cc"] == cc), None)
    if row and row["status"] not in ("free", "drawn"):
        why = "; ".join(x for x in (row["held_for_anita"], row["blocker"]) if x)
        print(f"!! queue.csv has {cc} as {row['status']}{f' ({why})' if why else ''}. "
              f"Read {row['detail'] or 'its queue.md prose'} before starting.")

    rec = {"id": args.id, "started": _now(), "note": args.note or "",
           "pid": os.getpid(), "host": os.environ.get("COMPUTERNAME", "?")}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        with open(path, encoding="utf-8") as fh:
            cur = json.load(fh)
        if cur.get("id") == args.id:
            # idempotent: re-taking your own claim refreshes it rather than failing
            cur["started"] = _now()
            if args.note:
                cur["note"] = args.note
            _write(path, cur)
            print(f"ok — {cc} was already yours; refreshed ({_age(cur['started'])} ago)")
            return 0
        age = _age(cur.get("started", 0))
        stale = (_now() - cur.get("started", 0)) / 3600.0 > STALE_HOURS
        print(f"REFUSED — {cc} is claimed by {cur.get('id', '?')}, {age} ago"
              f"{'  (which looks abandoned)' if stale else ''}")
        if cur.get("note"):
            print(f"  their note: {cur['note']}")
        print(f"  pick another country, or if you are sure it is dead:\n"
              f"      python tools/claim.py drop {cc} --id {args.id} --force")
        return 1
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(f"claimed {cc} for {args.id}")
    print(f"  release it with:  python tools/claim.py done {cc} --id {args.id}")
    return 0


def _write(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)


def cmd_drop(args):
    cc = args.cc.lower()
    path = os.path.join(CLAIMS, f"{cc}.json")
    if not os.path.exists(path):
        print(f"{cc} is not claimed by anyone")
        return 0
    with open(path, encoding="utf-8") as fh:
        cur = json.load(fh)
    if cur.get("id") != args.id and not args.force:
        print(f"REFUSED — {cc} belongs to {cur.get('id', '?')} "
              f"({_age(cur.get('started', 0))} ago), not to you.\n"
              f"  Use --force only if you are sure that session is gone.")
        return 1
    os.remove(path)
    who = "yours" if cur.get("id") == args.id else f"{cur.get('id', '?')}'s (forced)"
    print(f"released {cc} ({who})")
    return 0


def cmd_park(args):
    """Stop cleanly mid-country: leave a handoff, then drop the claim.

    AGENT_BRIEF.md §4. This is the move when you are about half through your context — not a
    failure, and the designed outcome at checkpoint B. What is actually expensive to redo is
    the fetch and the reconciliation; what is cheap is the mapping written against a CSV that
    already exists. So parking after the CSV lands costs the next session almost nothing,
    while running to the wall with the findings still in your head costs it everything.
    """
    cc = args.cc.lower()
    os.makedirs(HANDOFF, exist_ok=True)
    path = os.path.join(HANDOFF, f"{cc}.md")

    if os.path.exists(path):
        print(f"handoff/{cc}.md already exists — you are probably resuming a park.")
        print(f"  UPDATE it in place rather than starting a new one; the next session wants "
              f"one file,\n  not a stack of them.")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(HANDOFF_TEMPLATE.format(
                cc=cc, sid=args.id, date=datetime.date.today().isoformat()))
        print(f"wrote handoff/{cc}.md")

    print("\nFILL IT IN NOW, before you stop. The last COMMANDS.txt step you completed is the")
    print("line that matters; a handoff nobody wrote is a country nobody resumes.")
    print("Delete it when the country is finished.\n")
    return cmd_drop(args)


def cmd_done(args):
    half = half_registered().get(args.cc.lower())
    if half:
        # countries.py skips a half-registered country rather than stopping, so nothing else
        # would say so: it is on no map and in no check. Refuse before releasing the claim.
        print(f"REFUSED: {args.cc.lower()} is half-registered: {half}. countries.py skips it, "
              f"so it is on no map and in no check. Add the missing half, then run done again. "
              f"The claim is still yours.")
        return 1
    rc = cmd_drop(args)
    if rc == 0:
        cc = args.cc.lower()
        if cc in drawn():
            old = mark_drawn(cc)
            if old == "drawn":
                print(f"\nqueue.csv: {cc} was already drawn")
            elif old is None:
                print(f"\nqueue.csv: {cc} had no row; added one as drawn")
            elif old.startswith("!!"):
                print(f"\n{old}")
            else:
                print(f"\nqueue.csv: {cc} set to drawn (was {old})")
        else:
            row = next((r for r in read_queue_csv()[0] if r["cc"] == cc), None)
            print(f"\nqueue.csv: {cc} is not registered, so its row stays "
                  f"{row['status'] if row else '(it has none)'}. Run done again once it is "
                  f"registered;\n  if you closed or parked it instead, set its row by hand.")
        print(f"\nbefore you stop, the §12 tail for {cc}:")
        print(f"  - sources/{cc}.md written, and a `## {cc}-YYYY-MM-DD.` section in sources.md "
              f"(no new §9 letters)")
        print("  - under a supervisor the build tail is its job, and this country now shows as "
              "waiting;\n    on your own, python tools/build_tail.py --id <sid>")
        print("  - the country entry with note_public and gap=")
        print("  - python tools/built_countries.py --check   (both editions present)")
        print(f"  - move {cc}'s queue.md row to *Drawn*, or strike it; queue.csv is what "
              f"claim.py reads")
        if os.path.exists(os.path.join(HANDOFF, f"{cc}.md")):
            print(f"  - DELETE handoff/{cc}.md — you resumed a park and it is now a lie")
        print("  - a trap that generalises into its route's playbook (playbooks/), a general "
              f"working rule into spec §12, a fact about this country into sources/{cc}.md")
    return rc


def cmd_mine(args):
    mine = {cc: c for cc, c in load_claims().items() if c.get("id") == args.id}
    if not mine:
        print("you hold no claims")
        return 0
    for cc, c in sorted(mine.items()):
        print(f"  {cc}  {_age(c.get('started', 0)):>6s} ago  {c.get('note', '')}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list", help="what is claimed, what is free, and what is held (default)")

    for name, help_ in (("take", "claim a country"), ("drop", "release a claim"),
                        ("park", "release it and leave a handoff for the next session"),
                        ("done", "release a claim, mark it drawn in queue.csv if registered, "
                                 "and print the finishing checklist")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("cc")
        s.add_argument("--id", required=True, help="your session id")
        s.add_argument("--note", default="", help="what you are doing, for the other agents")
        s.add_argument("--force", action="store_true",
                       help="drop/take even if the claim is somebody else's")

    m = sub.add_parser("mine", help="what you are holding")
    m.add_argument("--id", required=True)

    args = p.parse_args()
    fn = {"take": cmd_take, "drop": cmd_drop, "park": cmd_park, "done": cmd_done,
          "mine": cmd_mine, "list": cmd_list, None: cmd_list}[args.cmd]
    sys.exit(fn(args) or 0)


if __name__ == "__main__":
    main()
