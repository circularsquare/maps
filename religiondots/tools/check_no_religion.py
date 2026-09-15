"""The draft "no religion" procedure (foot of WORKFLOW_PLAN.md), checked on every mapping.

    python tools/check_no_religion.py        exit 1 on a box that breaks the procedure
    python tools/check_no_religion.py -v     also list every no-religion box and its node

THE TRAP (playbooks/census_table.md, "A no-religion box"). A census "no religion" box sometimes
took traditional religion too: Mozambique's form words it `Sem religião (ateu, animista,
agnóstico,...)`, and Laos defined religion as a system with written doctrine. The procedure:
a box offered beside a separate traditional or animist box is `unaffiliated`; a LUMPED box is
`unknown` until a national source that asks the two separately puts one reading at 80% or
more, and then the split source and the share go in the mapping's REVIEW and in
sources/<cc>.md, saying whether the source measures self-description or practice.

WHAT IT ASSERTS, for every taxonomy/<cc><YYYY>.py that taxonomy/registry.py discovers:
  1. A no-religion box is a MAP key whose label is a no-religion answer.
  2. A box is a CANDIDATE when its label or its REVIEW entry mentions animism, traditional,
     ancestral or customary religion. A candidate whose mapping also has a traditional answer
     of its own (a key mapped to `indigenous...`, or labelled traditional or animist) is
     `separate` on that evidence, printed with the sibling it rests on. Any other candidate
     must be classified in CLASSIFIED below as "lumped" or "separate", with the reason; an
     unclassified one fails, because only the form can say which it is. "warn" is for a drawn
     box that breaks the procedure and is not this lint's to change: printed as WARN on every
     run, with the reason, and not counted as a failure.
  3. A lumped box drawn on anything but `unknown` needs a REVIEW entry that names a split
     source, gives a share of 80% or more, and says self-description or practice; and
     sources/<cc>.md must name the same source.
  4. A CLASSIFIED entry whose box is no longer in the mapping fails, so the register cannot
     quietly outlive the mapping.

WHAT IT CANNOT SEE: a lumped box whose mapping never mentions traditional religion. That is
the agent who did not read the form, and no text test finds it.
"""

import argparse
import importlib
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))

NO_RELIGION = re.compile(r"(?i)\bno\s+religio|\bnone\b|sans\s+religion|sem\s+religi|"
                         r"sin\s+religi|\bninguna\b|không theo tôn giáo|no\s+religious")
# A `None` under a religion is a denomination answer, not the box (Zambia's
# `Christianity Denomination: None`).
UNDER_A_RELIGION = re.compile(r"(?i)christian|islam|muslim|buddhis|hindu|denomination|church|"
                              r"catholic|protestant|jew")
TRADITIONAL = re.compile(r"(?i)animis|animist|tradition|tradicional|ancestr|customary|"
                         r"custom belief|ethnic religion|folk religion")
SPLIT_SOURCES = re.compile(r"(?i)afrobarometer|\bDHS\b|\bMICS\b|\bLSIS\b|\bPew\b|"
                           r"world values|\bWVS\b|global flourishing|arab ?barometer|"
                           r"asian barometer|\bLAPOP\b|latinobar")
SHARE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?\s*%(?!\s*(?:bar|threshold|rule))")
MEASURES = re.compile(r"(?i)self-descri|self-identif|call themselves|\bpractice")
BAR = 80.0

# (cc, MAP key) -> ("lumped" | "separate", why). Quote the form where it can be quoted.
CLASSIFIED = {
    ("mz", "Sem religião"): (
        "lumped",
        "the 2017 form words the box `Sem religião (ateu, animista, agnóstico,...)` and offers "
        "no traditional box. Drawn `unaffiliated` by Anita's ruling of 2026-09-14 on "
        "Afrobarometer R4-R9 and Pew 2009 (93-98% no religion, self-description)."),
    ("la", "No religion"): (
        "lumped",
        "the 2015 census defined religion as a system with written doctrine, so animism could "
        "not be answered, and its report calls the cell `no religion or being animist`. "
        "`unknown` by Anita's ruling of 2026-09-14; the move to traditional religion is queued."),
    ("my", "No Religion"): (
        "warn",
        "my2020.py's own REVIEW calls the near-certain reading indigenous traditional practice "
        "recorded as no religion (35.94% in Kecil Lojing, Temiar Orang Asli; the form gives folk "
        "practice no box) and maps it `unaffiliated` as printed. That reads as a lumped box, "
        "which the draft procedure draws `unknown` until a split source reaches 80%. The "
        "mapping predates the procedure; Anita's call, raised 2026-09-14."),
    ("sc", "No Religion"): (
        "separate",
        "the 2010 form (report Annex 2, P13) prints `No Religion: NONE` beside `Others: write "
        "full name`, and NBS post-coded every write-in: 57 labels across the districts, down "
        "to one agnostic, and none is a traditional or ancestral religion. Seychelles had no "
        "indigenous population before settlement, so there is no traditional practice for the "
        "box to be holding."),
}


def boxes(mod):
    """The MAP keys that are a no-religion answer."""
    return [k for k in getattr(mod, "MAP", {})
            if NO_RELIGION.search(str(k)) and not UNDER_A_RELIGION.search(str(k))]


def shares(text):
    out = []
    for m in SHARE.finditer(text):
        lo = float(m.group(1))
        out.append(lo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()

    from registry import discover
    bad = 0

    def say(ok, msg):
        nonlocal bad
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {msg}")

    found = set()
    n_boxes = warned = 0
    for cc, name in sorted(discover(drawn_only=False).items()):
        mod = importlib.import_module(name)
        review = getattr(mod, "REVIEW", {}) or {}
        here = boxes(mod)
        for box in here:
            n_boxes += 1
            node = mod.MAP[box]
            text = str(review.get(box, ""))
            candidate = bool(TRADITIONAL.search(str(box)) or TRADITIONAL.search(text))
            beside = [k for k, v in mod.MAP.items() if k not in here
                      and (str(v).startswith("indigenous") or TRADITIONAL.search(str(k)))]
            call = CLASSIFIED.get((cc, box))
            if call:
                found.add((cc, box))
            elif beside:
                call = ("separate", "the source has a traditional answer of its own: "
                        + ", ".join(f"{k!r} -> {mod.MAP[k]}" for k in beside))
            if args.v:
                print(f"  --  {cc} {name}: {box!r} -> {node}"
                      + (f" [{call[0]}]" if call else " [candidate]" if candidate else ""))
            if not candidate and (cc, box) not in CLASSIFIED:
                continue
            if not call:
                say(False, f"{cc} {name}: {box!r} -> {node} mentions traditional religion and "
                           "is not in CLASSIFIED. Read the form: add it as `lumped` or "
                           "`separate`, with the form's wording")
                continue
            kind, why = call
            if kind == "warn":
                warned += 1
                print(f"  WARN {cc} {box!r} -> {node}: {why}")
                continue
            if kind == "separate" or node == "unknown":
                say(True, f"{cc} {box!r} -> {node}: {kind}; {why}")
                continue
            srcs = sorted({m.group(0) for m in SPLIT_SOURCES.finditer(text)}, key=str.lower)
            top = max(shares(text), default=None)
            md = os.path.join(ROOT, "sources", f"{cc}.md")
            md_text = open(md, encoding="utf-8").read() if os.path.exists(md) else ""
            in_md = [s for s in srcs if s.lower() in md_text.lower()]
            problems = []
            if not text:
                problems.append("no REVIEW entry")
            if not srcs:
                problems.append("REVIEW names no split source (Afrobarometer, DHS, MICS, LSIS, "
                                "Pew, WVS ...)")
            if top is None or top < BAR:
                problems.append(f"REVIEW gives no share of {BAR:.0f}% or more "
                                f"(largest {top}%)")
            if not MEASURES.search(text):
                problems.append("REVIEW does not say whether the source measures "
                                "self-description or practice")
            if srcs and not in_md:
                problems.append(f"sources/{cc}.md does not name {srcs}")
            say(not problems,
                f"{cc} {box!r} -> {node}: lumped, drawn as one reading"
                + (f" on {', '.join(srcs)}, {top:g}%, named in sources/{cc}.md"
                   if not problems else "; " + "; ".join(problems)))

    for key in sorted(set(CLASSIFIED) - found):
        say(False, f"CLASSIFIED {key} is not a no-religion box in any mapping; remove or rename it")

    print(f"\n{'OK' if not bad else 'FAILED'}: {n_boxes} no-religion boxes, "
          f"{len(CLASSIFIED)} classified, {warned} warning(s), {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
