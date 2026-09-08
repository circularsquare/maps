"""Claim a country before working on it, so two agents do not build the same one.

Anita runs two or three agents on this directory at once (spec §12). That is by design and
mostly fine — shared files change under you and surgical edits cope. What does NOT cope is
two sessions independently deciding to build the same country: on 2026-09-08 one session
spent an hour on Peru while another had already finished it, and overwrote its `sources/pe.py`
with a `Write` to a path it had not checked. Nothing was lost (Claude Code's file history had
it), but the hour was.

    python tools/claim.py                       # what is claimed, and what is free
    python tools/claim.py take pe --id <sid>    # claim Peru
    python tools/claim.py drop pe --id <sid>    # release it
    python tools/claim.py done pe --id <sid>    # release it and note it as built
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
"""

import argparse
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
QUEUE = os.path.join(ROOT, "queue.md")

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
    """Country codes countries.py currently registers. Derived, never typed — spec §12."""
    src = open(os.path.join(ROOT, "countries.py"), encoding="utf-8").read()
    return set(re.findall(r'^    "([a-z]{2})": dict\(', src, re.M))


def built():
    """Country codes with dots actually on disk, which is a stronger claim than registered."""
    d = os.path.join(ROOT, "data", "processed")
    if not os.path.isdir(d):
        return set()
    return {m.group(1) for m in
            (re.fullmatch(r"dots_([a-z]{2})\.geojson", f) for f in os.listdir(d)) if m}


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


def read_queue():
    """The candidate list. Rows look like `| pe | Peru | ... |`; anything else is prose."""
    if not os.path.exists(QUEUE):
        return []
    rows = []
    for line in open(QUEUE, encoding="utf-8"):
        m = re.match(r"^\|\s*`?([a-z]{2})`?\s*\|\s*([^|]+?)\s*\|(.*)\|\s*$", line)
        if m:
            rest = [c.strip() for c in m.group(3).split("|")]
            rows.append({"cc": m.group(1), "name": m.group(2), "cells": rest})
    return rows


def cmd_list(args):
    claims, reg, have = load_claims(), drawn(), built()
    q = read_queue()

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

    print(f"\nDRAWN: {len(reg)} registered in countries.py, {len(have)} with dots on disk")
    missing = sorted(reg - have)
    if missing:
        print(f"  registered but NOT built: {', '.join(missing)}")

    if q:
        free = [r for r in q if r["cc"] not in claims and r["cc"] not in reg]
        print(f"\nQUEUE: {len(q)} candidates in queue.md, {len(free)} free and undrawn")
        for r in free:
            print(f"  {r['cc']}  {r['name'][:26]:26s} "
                  f"{' | '.join(r['cells'])[:72]}")
        taken = [r for r in q if r["cc"] in reg]
        if taken:
            print(f"  (already drawn, ignore: {', '.join(r['cc'] for r in taken)})")
    else:
        print("\nQUEUE: queue.md has no candidate rows")


def cmd_take(args):
    cc = args.cc.lower()
    reg = drawn()
    os.makedirs(CLAIMS, exist_ok=True)
    path = os.path.join(CLAIMS, f"{cc}.json")

    if cc in reg:
        print(f"!! {cc} is ALREADY REGISTERED in countries.py. If you are re-working it that "
              f"is fine, but check sources/{cc}.md first — somebody built it.")

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


def cmd_done(args):
    rc = cmd_drop(args)
    if rc == 0:
        cc = args.cc.lower()
        print(f"\nbefore you stop, the §12 tail for {cc}:")
        print("  - sources/<cc>.md written, and a §9-series section in sources.md")
        print("  - countries.py entry with note_public and gap=")
        print("  - python tools/built_countries.py --check   (both editions present)")
        print(f"  - remove {cc} from queue.md, or mark it drawn there")
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

    sub.add_parser("list", help="what is claimed and what is free (default)")

    for name, help_ in (("take", "claim a country"), ("drop", "release a claim"),
                        ("done", "release a claim and print the finishing checklist")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("cc")
        s.add_argument("--id", required=True, help="your session id")
        s.add_argument("--note", default="", help="what you are doing, for the other agents")
        s.add_argument("--force", action="store_true",
                       help="drop/take even if the claim is somebody else's")

    m = sub.add_parser("mine", help="what you are holding")
    m.add_argument("--id", required=True)

    args = p.parse_args()
    fn = {"take": cmd_take, "drop": cmd_drop, "done": cmd_done,
          "mine": cmd_mine, "list": cmd_list, None: cmd_list}[args.cmd]
    sys.exit(fn(args) or 0)


if __name__ == "__main__":
    main()
