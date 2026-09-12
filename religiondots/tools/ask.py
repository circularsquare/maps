"""The inbox for things that are genuinely Anita's call, and nothing else.

    python tools/ask.py                              # what is open (default)
    python tools/ask.py new <cc> --title "..."       # file one, prints the path to fill in
    python tools/ask.py answered <n>                 # move it out of the way once she has ruled

**AN ASK IS NEVER A BLOCK.** The template forces two lines before the question: the decision you
already took, and what reversing it costs. The country ships either way; she is choosing whether
to flip a call, not being asked to unstick a stalled session. An ask that says "should I do A or
B, waiting for your answer" is the failure mode this whole file exists to prevent.

**AIM FOR ZERO TO ONE PER COUNTRY.** Anita, 2026-09-08: *"try to keep the number of things marked
for review relatively low, dont overuse it."* Ten open asks is worse than one wrong call, because
each is a decision she has to load a country's context to make. The bar is in `AGENT_BRIEF.md` §3
and it is high: §14 questions, unclear source terms, a change to an already-drawn country, a new
legend row nobody else uses, anything needing an account or money.

**THIS IS NOT THE `REVIEW` DICT AND DOES NOT COMPETE WITH IT.** An arguable category mapping goes
in the mapping module's `REVIEW` with its reason — that tier is cheap, uncapped, already in 95
files, and is where most "I am not sure about this" belongs. `tools/review_dump.py` collects those
for reading in bulk. This file is for the handful of things `REVIEW` cannot hold because they are
not about one category.

Files are `ask/NNN-<cc>-<slug>.md`, one per ask, created with `O_CREAT|O_EXCL` — same reason as
`tools/claim.py`, that two or three sessions run here at once and a shared list they all append to
is `index.html` again. Answered ones move to `ask/answered/`; nothing is ever deleted by a tool.
"""

import argparse
import datetime
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASK = os.path.join(ROOT, "ask")
ANSWERED = os.path.join(ASK, "answered")

TEMPLATE = """# {n:03d} — {cc}: {title}

*Filed {date} by session `{sid}`. Anita's call; nothing is waiting on it.*

## What I did

<!-- The decision you ALREADY TOOK, in one or two sentences. Not "I could do A or B". -->

## What it costs to reverse

<!-- One line. "Re-run sources/{cc}.py and retile, 20 min" or "a one-word edit in
     taxonomy/{cc}YYYY.py, no rebuild" or "the country would have to come off the map". -->

## Why it is yours rather than mine

<!-- Which bar in AGENT_BRIEF.md §3 this clears. If you cannot name one, this is a REVIEW
     entry in the mapping module instead, and you should delete this file. -->

## The detail

<!-- The figures, the source, the precedent it would or would not follow. Enough that she
     can rule on it without reopening the country. -->
"""


def _slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s[:44].rstrip("-")) or "untitled"


def _entries(d):
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        m = re.match(r"^(\d+)-([a-z]{2})-(.+)\.md$", f)
        if m:
            out.append((int(m.group(1)), m.group(2), f, os.path.join(d, f)))
    return out


def _title_of(path):
    """First `# ` line, minus the number and country the filename already carries."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("# "):
                    return re.sub(r"^\d+\s*[-—]\s*[a-z]{2}:\s*", "", line[2:].strip())
    except OSError:
        pass
    return "(no title line)"


def _age(path):
    try:
        d = (datetime.datetime.now()
             - datetime.datetime.fromtimestamp(os.path.getmtime(path)))
    except OSError:
        return "?"
    h = d.total_seconds() / 3600.0
    return f"{h * 60:.0f}m" if h < 1 else (f"{h:.1f}h" if h < 48 else f"{h / 24:.0f}d")


def cmd_list(args):
    open_ = _entries(ASK)
    if not open_:
        print("no open asks — which is the healthy state")
    else:
        print(f"OPEN ({len(open_)}) — Anita's to rule on; no session is waiting on any of them:")
        for n, cc, fname, path in open_:
            print(f"  {n:3d}  {cc}  {_age(path):>5s}  {_title_of(path)[:66]}")
            print(f"       ask/{fname}")
        if len(open_) > 4:
            print(f"\n  !! {len(open_)} is above the bar in AGENT_BRIEF.md §3. If you are a "
                  f"supervisor,\n     stop spawning and hand back to Anita.")
    if args.all:
        done = _entries(ANSWERED)
        print(f"\nANSWERED ({len(done)}):")
        for n, cc, fname, path in done:
            print(f"  {n:3d}  {cc}  {_title_of(path)[:66]}")
    return 0


def cmd_new(args):
    os.makedirs(ASK, exist_ok=True)
    cc = args.cc.lower()
    used = {n for n, _, _, _ in _entries(ASK) + _entries(ANSWERED)}
    n = max(used, default=0) + 1
    slug = _slug(args.title)

    # O_CREAT|O_EXCL, walking the number up, so two sessions filing at once cannot collide.
    for _ in range(50):
        path = os.path.join(ASK, f"{n:03d}-{cc}-{slug}.md")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            n += 1
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(TEMPLATE.format(n=n, cc=cc, title=args.title, sid=args.id or "?",
                                     date=datetime.date.today().isoformat()))
        print(f"filed ask {n:03d} — {path}")
        print("\nnow FILL IT IN, and note that the first two sections are the point:")
        print("  * the decision you already took, so the country ships")
        print("  * what reversing it costs, so she knows what she is choosing")
        print("\nthen carry on. Do not wait for an answer.")
        return 0
    print("could not allocate a number — look in ask/ by hand")
    return 1


def cmd_answered(args):
    os.makedirs(ANSWERED, exist_ok=True)
    for n, cc, fname, path in _entries(ASK):
        if n == args.n:
            os.replace(path, os.path.join(ANSWERED, fname))
            print(f"moved {fname} to ask/answered/")
            return 0
    print(f"no open ask numbered {args.n}")
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Also on the bare form, so `ask.py --all` works as well as `ask.py list --all`.
    p.add_argument("--all", action="store_true", help="include answered ones")
    sub = p.add_subparsers(dest="cmd")

    l = sub.add_parser("list", help="what is open (default)")
    l.add_argument("--all", action="store_true", help="include answered ones")

    nw = sub.add_parser("new", help="file one")
    nw.add_argument("cc")
    nw.add_argument("--title", required=True, help="one line, specific")
    nw.add_argument("--id", default="", help="your session id")

    an = sub.add_parser("answered", help="move one out of the way once she has ruled")
    an.add_argument("n", type=int)

    args = p.parse_args()
    if not hasattr(args, "all"):
        args.all = False
    fn = {"new": cmd_new, "answered": cmd_answered, "list": cmd_list, None: cmd_list}[args.cmd]
    sys.exit(fn(args) or 0)


if __name__ == "__main__":
    main()
