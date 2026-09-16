"""
Per-country wiring for the scatter step: where the counts are, where the polygons are, and
how a source category becomes a religiondots node.

Everything downstream of this file is country-agnostic. Everything above it is per-country by
necessity — the sources do not agree about anything, and spec §3.9, §8.1 and §2.3 are three
different ways of saying so.

Each entry supplies:
  counts()      -> DataFrame [unit, node, count]   unit = the geography the counts are ON
                optionally `tier` (spec §7): `measured` / `derived` / `modelled`, per row.
                Missing means `measured`, which is right for a census read at its own
                geography and wrong for anything that was spread, so an adapter that spreads
                must say so. `derived` and `modelled` draw DESATURATED, and the weakest tier
                on a (unit, node) pair wins — a pair that is part measurement and part
                estimate is not a measurement.
                optionally `roll` (spec §7a-i-1): for a `derived` row, the node its SOURCE
                COLUMN names — the coarser category the source counted at this same unit,
                which the roll-up falls back to when a reader asks what was measured. It is
                NOT an ancestor lookup: Hungary's Baptists came out of a column called
                `Other Christian denomination`, which is nowhere above them in the tree.
                Missing means "walk the tree", which is what every country did before.
  units         path to the polygons for `unit`, and the column holding its id
  place         path to a finer polygon layer used to place dots inside a unit, and the
                column linking it back to `unit`. spec §8.2: these units are designed to a
                population target, so an equal share of dots per unit is already a population
                weighting and no population data is read.

and, for the viewer, which draws one country at a time and needs to say whose data it is:
  name          what the country is called in the picker, which is also what it sorts by
  name_in       optional, the same name in the sentence "Religion in ___" — the article is
                part of the name in English for some countries and not others, and the
                picker wants "United States" where the title wants "the United States"
  source        the agency and instrument, one line, under the title
  basis         which quantity the numbers are (spec §3.1) — never mixed, and now never
                silently mixed either, since two countries are no longer on screen at once
  how           spec §7c: WHAT KIND OF THING these numbers are, in the same words for every
                country, so the reader can rank them against each other without parsing
                sixty-eight agencies in five languages. `source` already carries the
                citation and cannot do this: "Sčítání 2021 (Czech Statistical Office)" and
                "Sreda «Arena» Atlas 2012" look alike and are a census and a survey.
                A phrase, not a sentence, and it never repeats what `grain` says.
  grain         how fine the counts are, in a labelled row under the title. The unit and its
                average population, and nothing else: the row's label says what the number
                is, so the string must not say it again.

  fill          spec §7d: what a `derived` country's filled-in rows were filled in FROM, as a
                phrase completing "41% filled in ___" — "from the 2010 census", "from the
                same census at province level". Only for a country that HAS derived rows; the
                viewer falls back to "from broader counts" without it, which is the wording
                Anita rejected for describing fifteen real published tables as vaguely as
                possible. `allocate.py`'s invocation in COMMANDS.txt names the coarse level.
  gap           who or what the source leaves out, a few words, shown as the `not drawn` row.
                Not a summary of `note_public` — if it needs a second line it belongs there
                instead. Most countries have one, because most censuses print a non-response
                cell this project does not draw; that was not true when the field was written
                and the note here used to say a row every country carries is a row nobody
                reads. It is still worth writing as if it were rare: say the specific thing.
  gap_share     HOW BIG THAT HOLE IS, as a fraction of the source's own universe. It is the
                width of the hatched segment on the legend's composition bar (spec §10.4):
                drawn plus undrawn is the bar, so 0.3005 for Czechia says 30.05% of the
                country did not answer a voluntary question and is not on the map. Optional.

                `gap` HAS TO STATE THE FIGURE ITSELF, and this is asserted: the viewer's
                hatched-segment card prints `gap` and nothing else, so a sentence that does
                not carry the number leaves the reader without it. It used to print the
                percentage too and that read as two separate holes of the same size ("21% of
                Peru · under-twelves, 21.1% of the country, who were not asked"). Say it once,
                in the country's own words. This is also the only way to give a BREAKDOWN:
                Montenegro is "6.6%: 4.7% withheld by MONSTAT's disclosure control and 1.9%
                who declined to declare", which no composed figure could say. Where part of
                the hole is quantified and part is not, a semicolon does it: Angola's "the
                2.3% who did not answer or did not know; and children under 2, who were not
                asked the religion question", where the missing second figure is the point.

                RUN `python tools/gap_share.py` RATHER THAN WORKING IT OUT. Where the hole is
                a column the census printed and this project declined to draw, the tool
                computes it exactly off `data/normalized/<cc>.csv`, two independent ways, and
                writes it with `--write`. `--check` then fails if an authored figure is
                smaller than a residual the data can prove. Fifty-odd countries are that kind
                and none of them needs a judgement.

                WHAT THE TOOL CANNOT SEE, and where the figure stays hand-written: people who
                were never in any table. Peru's under-twelves, the Galápagos, Abkhazia. Those
                come from the source's own age or population table and nowhere else. Leave the
                field out where nobody has quantified them (Turkey's Alevis, Botswana's
                under-twelves) — no number is a fine answer, and a guessed one is drawn.

                AND DO NOT ADD THE TWO KINDS TOGETHER by hand without saying so. They are
                shares of different universes: Peru's non-response would be a share of the
                12-and-overs and not of Peru. Where a country has both, the sentence names
                both and the figure names which one it is.

                The share is of the whole population the source is about, not of the drawn
                dots — the viewer divides to get the bar's total.

                THESE FOUR ARE PLAIN TEXT — no markup and NO EM DASHES, Anita 2026-09-07.
                They are escaped by the viewer, so a `<b>` or a backtick would show the reader
                its own characters, and the dash is most of what made the block read as
                machine-written. Punctuate with a comma, a semicolon or a bracket. Asserted at
                the foot of this file, so a slip fails the import rather than the review.
  note_public   the country's own caveat, shown in the about panel when it is selected. The
                per-country note that used to be a cross-border paragraph belongs here: with
                a single country on screen the interesting comparison is inside it.

                HOW TO WRITE ONE — Anita, 2026-09-07: *"trying to keep it not sounding very
                ai, so people dont get ick."* The about panel renders `**bold**`, `*italic*`
                and `` `code` `` (spec §7d), and `tools/check_md.py` fails on a marker it
                cannot convert. Beyond that:

                  * A `**bold sentence.**` that STARTS a sentence is read as a topic sentence
                    and becomes a PARAGRAPH BREAK, losing its bold. That is the intended way
                    to structure a long note. Bold inside a sentence survives and is for a
                    figure: `Catholicism is **38.0%** and falling`.
                  * So do not reach for bold to make a point loud. The paragraph break is the
                    structure; bolding it too is the listicle voice, which is the specific
                    thing that reads as machine-written.
                  * Prefer a comma, a semicolon or a bracket to an em dash. The notes written
                    before this rule are full of them and are Anita's to clean up; do not
                    convert an existing one as a side effect of editing near it, and do not
                    add new ones.
                  * Say the specific thing. "from the 2010 census" beats "from an earlier
                    source"; a phrasing general enough to fit every country describes none of
                    them and reads as evasion.
                  * [[feedback_label_voice]]'s rules still apply to any short text: no
                    flourish, no loaded verbs, ask whose point of view a verb encodes.
  view          optional [w, s, e, n] to fly to, where the data bbox is the wrong picture —
                the US spans Hawaii to Maine and fitting that shows mostly ocean. Defaults to
                the bbox of the country's own dots, computed in tiles.py.
  territory     optional, default True. False for an entry that is people rather than a place:
                part of no country on the map, with no outline and no wash of its own
                (country_shapes.py leaves it out), and never chosen by the viewer's Auto from
                where the camera is looking. It is seen in the all-countries view, or picked by
                hand from the list. Written into counts.json by tiles.py, and a bool (asserted
                below). Anita, 2026-09-15, for the Israeli settlements beyond the Green Line
                (`xs`), which Israel's entry stops short of and Palestine's census does not
                count: *"they dont actually have any territory so it shouldnt be possible to
                auto-mode onto them."* Northern Cyprus is a later candidate and is not built.
                Such an entry takes its code from ISO 3166's user-assigned range (`x?`, as
                Kosovo's `xk`), so it can never meet a real country's.
"""
import json
from pathlib import Path
import re
import sys

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "taxonomy"))

# LAYOUT, since 2026-09-14 (WORKFLOW_PLAN.md item 9). Each country's entry and the helpers only
# it uses are in countries/<cc>.py, as `ENTRY = {"<cc>": dict(...)}`. Helpers that two or more
# countries use are in countries/_shared.py, which every country file star-imports. This file
# only loads them: COUNTRIES takes its order from ORDER below, never from the directory, and
# every helper stays importable as `countries.<name>`. A new country is a new file plus its
# code appended to ORDER, written back to back. Until both exist the loader SKIPS that country
# with a warning on stderr and loads the rest (HALF_REGISTERED, below). It used to stop instead,
# which broke every other session's import for as long as the gap lasted: on 2026-09-15 a
# tiles.py worker died on it and hung a build tail for 45 minutes, and check_md.py stopped.
ORDER = [
    "us",
    "ca",
    "cy",
    "cz",
    "au",
    "ie",
    "mx",
    "nz",
    "uk",
    "br",
    "pl",
    "ro",
    "ee",
    "hr",
    "in",
    "de",
    "hu",
    "mk",
    "lk",
    "cl",
    "ph",
    "gh",
    "ru",
    "id",
    "ke",
    "mn",
    "mu",
    "mw",
    "ao",
    "bj",
    "ni",
    "gt",
    "cr",
    "pa",
    "kg",
    "tm",
    "uz",
    "sv",
    "hn",
    "pr",
    "do",
    "ec",
    "bo",
    "co",
    "uy",
    "eg",
    "jo",
    "pe",
    "tr",
    "ar",
    "vu",
    "sb",
    "fm",
    "ki",
    "ws",
    "to",
    "bw",
    "fj",
    "zw",
    "kz",
    "np",
    "mm",
    "kh",
    "rs",
    "lt",
    "kr",
    "gy",
    "vn",
    "cn",
    "sg",
    "hk",
    "et",
    "cf",
    "ci",
    "pk",
    "bd",
    "my",
    "es",
    "gr",
    "pt",
    "xk",
    "ba",
    "li",
    "ch",
    "sk",
    "me",
    "jm",
    "vc",
    "ge",
    "il",
    "fr",
    "it",
    "bz",
    "tt",
    "sr",
    "bs",
    "ky",
    "bb",
    "lc",
    "gd",
    "th",
    "bg",
    "pw",
    "ck",
    "tv",
    "nu",
    "ms",
    "bm",
    "ag",
    "dm",
    "mh",
    "py",
    "la",
    "sz",
    "si",
    "at",
    "za",
    "md",
    "am",
    "fi",
    "se",
    "rw",
    "tl",
    "al",
    "cv",
    "st",
    "ht",
    "ng",
    "tz",
    "zm",
    "mz",
    "lr",
    "gn",
    "gw",
    "cg",
    "ug",
    "nl",
    "iq",
    "nr",
    "vg",
    "fk",
    "kn",
    "sx",
    "ai",
    "aw",
    "cw",
    "bq",
    "jp",
    "be",
    "no",
    "dk",
    "lv",
    "ua",
    "td",
    "ir",
    "bf",
    "sc",
    "tw",
    "ml",
    "by",
    "ve",
    "cm",
    "ye",
    "mg",
    "sl",
    "tg",
    "bn",
    "ps",
    "sn",
    "gi",
    "xs",
    "gm",
    "mt",
    "as",
    "ne",
    "dz",
    "fo",
    "qa",
    "is",
    "ma",
    "tn",
    "ly",
    "im",
    "tk",
    "mr",
    "ad",
    "je",
    "sd",
    "sa",
    "cd",
    "af",
    "so",
    "om",
]

COUNTRIES = {}

# {cc: which half is missing} for a country with its code in ORDER or its countries/<cc>.py but
# not both. Filled by _load(), which skips these and loads everything else; empty when ORDER and
# the directory agree. The strict form of the check reads it: `tools/built_countries.py --check`
# fails on any entry, and `tools/claim.py done` will not mark such a country drawn.
HALF_REGISTERED = {}


def _load():
    """Fill COUNTRIES from countries/<cc>.py in ORDER and re-export every helper by name.

    A half-registered country is skipped with one warning line on stderr, not stopped on: a
    builder writes its file and its ORDER line one after the other, and every other session
    imports this file in between. A code named twice in ORDER, or a file not named for a code,
    still stops, because neither is a builder partway through its two writes."""
    import importlib.util
    import types

    folder = HERE / "countries"
    on_disk = {p.stem for p in folder.glob("*.py")} - {"_shared"}
    problems = []
    doubled = sorted({cc for cc in ORDER if ORDER.count(cc) > 1})
    if doubled:
        problems.append("named twice in ORDER: " + ", ".join(doubled))
    odd = sorted(s for s in on_disk if not re.fullmatch(r"[a-z]{2}", s))
    if odd:
        problems.append("not named for a two-letter country code: "
                        + ", ".join(f"countries/{s}.py" for s in odd))
    if problems:
        raise SystemExit("countries.py: ORDER and the files in countries/ disagree\n  "
                         + "\n  ".join(problems))
    for cc in ORDER:
        if cc not in on_disk:
            HALF_REGISTERED[cc] = f"it is in ORDER but countries/{cc}.py does not exist"
    for cc in sorted(on_disk - set(ORDER)):
        HALF_REGISTERED[cc] = f"countries/{cc}.py exists but {cc} is not in ORDER"
    for cc, why in HALF_REGISTERED.items():
        print(f"countries.py: WARNING skipping {cc}: {why}. Every other country loads. If a "
              f"builder is between its two writes this clears in a moment; if it persists, add "
              f"the missing half.", file=sys.stderr, flush=True)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return mod

    g = globals()
    shared = load("countries._shared", folder / "_shared.py")
    for n in shared.__all__:
        g.setdefault(n, getattr(shared, n))
    owner = {}
    for cc in ORDER:
        if cc in HALF_REGISTERED:
            continue
        mod = load(f"countries.{cc}", folder / f"{cc}.py")
        entry = getattr(mod, "ENTRY", None)
        if not (isinstance(entry, dict) and list(entry) == [cc]):
            got = list(entry) if isinstance(entry, dict) else type(entry).__name__
            raise SystemExit(f'countries/{cc}.py: ENTRY must be {{"{cc}": dict(...)}}, got {got}')
        COUNTRIES[cc] = entry[cc]
        for n, v in vars(mod).items():
            if n.startswith("__") or n == "ENTRY" or isinstance(v, types.ModuleType):
                continue
            if n in shared.__all__ and v is getattr(shared, n):
                continue
            if isinstance(v, (types.FunctionType, type)) and v.__module__ != mod.__name__:
                continue
            if n in g and g[n] is not v:
                other = f"countries/{owner[n]}.py" if n in owner else "countries.py or countries/_shared.py"
                raise SystemExit(f"countries/{cc}.py defines `{n}` and so does {other}. Every helper "
                                 f"is importable as countries.{n}, so the names must stay unique.")
            g[n] = v
            owner[n] = cc


_load()
del _load

# spec §7c: the header block is plain text, and the em dash is the thing it must not contain.
# Anita, 2026-09-07 — the first draft of these rows was correct and read as machine-written, and
# the dash doing the work of a comma, a semicolon and a bracket at once was most of the reason.
# Checked here rather than trusted to review: the fields are edited one country at a time, months
# apart, and a rule this easy to forget is a rule worth failing the import over.
for _cc, _m in COUNTRIES.items():
    for _f in ("how", "fill", "grain", "gap"):
        _v = _m.get(_f, "")
        assert "\u2014" not in _v, f"{_cc}.{_f} has an em dash; use a comma, a semicolon or a bracket"
        assert not (set("<`*") & set(_v)), (
            f"{_cc}.{_f} has markup (< ` or *); these fields are escaped, not rendered")
    # `territory` is read as `!== false` by the viewer, so a truthy string would be ignored there.
    assert isinstance(_m.get("territory", True), bool), f"{_cc}.territory must be True or False"
    # spec \u00a710.4: the hatched segment's width. A share with no sentence beside it is a hole
    # the reader cannot ask about, and a share of 0 or 1 is a bar with nothing in it.
    _g = _m.get("gap_share")
    if _g is not None:
        assert isinstance(_g, float) and 0.0 < _g < 1.0, (
            f"{_cc}.gap_share must be a fraction strictly between 0 and 1, got {_g!r}")
        assert _m.get("gap"), f"{_cc}.gap_share has no gap= sentence to explain it"
        # AND THE SENTENCE HAS TO STATE THE SHARE ITSELF, because the viewer's hatched-segment
        # card prints `gap` and nothing else (spec §10.4). It used to print the percentage too
        # and that read as two separate holes of the same size: "21% of Peru · under-twelves,
        # 21.1% of the country, who were not asked". One number, in the country's own words,
        # which is also the only way a BREAKDOWN can be given (Montenegro's 4.7% withheld plus
        # 1.9% declined). A tenth of a point of slack, because a sentence saying 0.2% for a
        # measured 0.2368% is rounding and not disagreement.
        _pcts = [float(_x) for _x in re.findall(r"(\d+(?:\.\d+)?)\s*%", _m["gap"])]
        assert any(abs(_p - _g * 100) <= 0.1 for _p in _pcts), (
            f"{_cc}.gap_share is {_g * 100:.2f}% and gap= states {_pcts or 'no percentage'}; "
            f"the sentence is where the reader gets the figure, so it has to carry it")
