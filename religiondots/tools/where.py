"""Where one country appears across the tree, so you read the right hundred lines of it.

    python tools/where.py mz
    python tools/where.py mz --full      # matching lines whole, not cut to width

WORKFLOW_PLAN.md item 6. The shared records are too big to read (sources.md ~24,000 lines, spec.md
~12,800, countries.py ~20,800, queue.md ~950), and an agent that does not know where its country
is mentioned either reads too much or misses the ruling that applies. This lists, with line
numbers:

  files         sources/<cc>*, taxonomy/<cc>YYYY.py, normalized CSVs, data/geo/<cc>/, dots,
                handoff, claim
  countries.py  the entry
  sources.md    record sections: `## <cc>-YYYY-MM-DD.` keys, and older headings naming the country
  spec.md       headings naming the country
  queue.md      rows and lines naming it, with the section each sits under
  asks          open and answered asks filed under the code, and any other ask that mentions it
  RULINGS.md    Anita's rulings that name the code
  playbooks     playbooks that list it
  runlog.md     the last few lines about it

It only reads. Names are matched as whole words, so `Guinea` also finds `Papua New Guinea`;
the heading is printed so you can tell.
"""

import argparse
import glob
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIDTH = 104


def _lines(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def _cut(s, full, width=WIDTH):
    s = s.strip()
    return s if full or len(s) <= width else s[:width - 3] + "..."


def _age(ts):
    h = (time.time() - ts) / 3600.0
    if h < 1:
        return f"{h * 60:.0f}m"
    return f"{h:.1f}h" if h < 48 else f"{h / 24:.0f}d"


def _size(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0


def _block(title, hits, full, fmt=None):
    print(f"\n{title} ({len(hits)})")
    if not hits:
        print("  none")
    for h in hits:
        print(fmt(h) if fmt else f"  {h[0]:>6}  {_cut(h[1], full)}")


def name_of(cc):
    """The country's name from its countries.py entry, or its queue.md row if it is not drawn."""
    m = re.search(rf'^    "{cc}": dict\(\s*\n\s*name="([^"]+)"', "\n".join(_lines("countries.py")),
                  re.M)
    if m:
        return m.group(1)
    for line in _lines("queue.md"):
        m = re.match(rf"^\|\s*`?{cc}`?\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            return m.group(1).strip("* ")
    return None


def files(cc):
    pats = [f"sources/{cc}.py", f"sources/{cc}_*.py", f"sources/{cc}.md", f"sources/{cc}_*.md",
            f"taxonomy/{cc}[0-9][0-9][0-9][0-9].py",
            f"data/normalized/{cc}.csv", f"data/normalized/{cc}_*.csv",
            f"data/processed/dots_{cc}.geojson", f"data/processed/dots_{cc}_*.geojson",
            f"handoff/{cc}.md", f"data/claims/{cc}.json"]
    out = []
    for p in pats:
        out += sorted(glob.glob(os.path.join(ROOT, p)))
    if os.path.isdir(os.path.join(ROOT, "data", "geo", cc)):
        out.append(os.path.join(ROOT, "data", "geo", cc))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cc")
    ap.add_argument("--full", action="store_true", help="print matching lines whole")
    args = ap.parse_args()
    cc, full = args.cc.lower(), args.full
    if not re.fullmatch(r"[a-z]{2}", cc):
        sys.exit("give a two-letter country code, e.g. mz")

    name = name_of(cc)
    name_re = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])") if name else None
    code_re = re.compile(rf"`{cc}`|\({cc}\)")

    def names_it(text):
        return bool(code_re.search(text) or (name_re and name_re.search(text)))

    print(f"{cc}  {name or '(in neither countries.py nor queue.md)'}")

    print("\nfiles")
    fs = files(cc)
    if not fs:
        print("  none")
    for p in fs:
        rel = os.path.relpath(p, ROOT)
        if os.path.isdir(p):
            print(f"  {rel}{os.sep}")
        else:
            print(f"  {rel:<46} {_size(os.path.getsize(p)):>7}  {_age(os.path.getmtime(p))} ago")

    entry = [i for i, l in enumerate(_lines("countries.py"), 1) if l.startswith(f'    "{cc}": dict(')]
    print("\ncountries.py")
    print(f"  entry at line {entry[0]}" if entry else "  not registered")

    key_re = re.compile(rf"^## {cc}-\d{{4}}-\d{{2}}-\d{{2}}[a-z]?\.")
    _block("sources.md record sections",
           [(i, l) for i, l in enumerate(_lines("sources.md"), 1)
            if l.startswith("## ") and (key_re.match(l) or names_it(l))], full)

    _block("spec.md headings",
           [(i, l) for i, l in enumerate(_lines("spec.md"), 1)
            if re.match(r"#{2,4} ", l) and names_it(l)], full)

    row_re = re.compile(rf"^\|\s*`?{cc}`?\s*\|")
    hits, section = [], ""
    for i, l in enumerate(_lines("queue.md"), 1):
        if l.startswith("#"):
            section = l.lstrip("# ").strip()
        elif row_re.match(l) or names_it(l):
            hits.append((i, section, l))
    _block("queue.md", hits, full,
           lambda h: f"  {h[0]:>5}  [{_cut(h[1], False, 34)}]  {_cut(h[2], full, 64)}")

    print("\nasks")
    listed = []
    for sub, label in (("", "open"), ("answered", "answered")):
        for p in sorted(glob.glob(os.path.join(ROOT, "ask", sub, f"[0-9][0-9][0-9]-{cc}-*.md"))):
            print(f"  {label:<9} {os.path.relpath(p, ROOT)}")
            listed.append(p)
    others = sorted(glob.glob(os.path.join(ROOT, "ask", "[0-9][0-9][0-9]-*.md"))
                    + glob.glob(os.path.join(ROOT, "ask", "answered", "[0-9][0-9][0-9]-*.md")))
    for p in others:
        if p not in listed:
            with open(p, encoding="utf-8", errors="replace") as fh:
                if names_it(fh.read()):
                    print(f"  mentions  {os.path.relpath(p, ROOT)}")
                    listed.append(p)
    if not listed:
        print("  none")

    rulings = _lines(os.path.join("ask", "RULINGS.md"))
    if not rulings:
        print("\nask/RULINGS.md: not written yet")
    else:
        heads = [(i, l) for i, l in enumerate(rulings, 1) if l.startswith("- ")]
        _block("ask/RULINGS.md", [(i, l) for i, l in heads if f"`{cc}`" in l.split("|")[0]], full)
        n_all = sum(1 for _, l in heads if "`all`" in l.split("|")[0])
        print(f"  plus {n_all} ruling(s) for all countries; read those too")

    pbs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "playbooks", "*.md"))):
        with open(p, encoding="utf-8", errors="replace") as fh:
            if code_re.search(fh.read()):
                pbs.append(os.path.relpath(p, ROOT))
    print("\nplaybooks")
    print("  " + ("\n  ".join(pbs) if pbs else "none"))

    runs = [(i, l) for i, l in enumerate(_lines("runlog.md"), 1)
            if len(l.split("|")) > 2 and cc in re.split(r"[^a-z]+", l.split("|")[1].lower())]
    _block("runlog.md, latest", runs[-6:], full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
