"""The country loader: every countries/<cc>.py is one country, found automatically.

There is no list to append to. Several agents add countries at once, and religiondots' shared
ORDER list was where their edits collided; here a country exists when its file does.

Each countries/<cc>.py defines ENTRY, a dict with:
  name          the country, as the picker shows it
  source        agency and table, one line
  how           what kind of figure, in the same words everywhere: "census, 2021, mother tongue"
  grain         the unit and its average population: "753 local levels, 38,000 people on average"
  gap           optional: who the source leaves out, with the figure if there is one
  note_public   optional: the country's own caveat for the about text (plain voice, see AGENT_BRIEF)
  view          [w, s, e, n] to frame the country
  counts        () -> DataFrame [unit, node, count], optionally `tier` (measured/derived/modelled)
  mappings      the taxonomy modules (taxonomy/<cc><year>.py) counts() resolves through
  place         the placement layer (a gpkg with `unit` and ideally `pop`)
  place_unit    (GeoDataFrame) -> Series of unit ids, matching counts()' `unit`
  place_weight  usually `pop_weight` from countries/_shared.py
  parts         optional (Anita, 2026-10-06), the viewer's Data block: which source draws which
                part of the population. A list of dicts, one per part, in the order a reader
                should meet them (the language question first, the "everyone else" rule last):
                  covers   who this part is, a few words: "Indigenous languages", "People born
                           abroad", "Foreign citizens", "Everyone else"
                  source   where its figures come from, one short line: "2020 census, language
                           spoken, aged 3 and over", "drawn as Spanish", "Afrobarometer 2012-17,
                           about 3,600 adults"
                  and at most one of these, for the part's people (the viewer prints the number
                  and its share of the country's drawn total):
                  people   int, from the build's own figures (counts() summed over the part)
                  nodes    list of language node ids; the people drawn under any of them
                  rest     True: everyone drawn less the other parts (use it on the last part)
                A country with one source is one part with rest=True. Without `parts` the viewer
                shows source / how / grain / gap as before. No em dashes (AGENT_BRIEF, voice).

  drawn_named   optional (2026-10-06): Natural Earth breakaway areas (BRK_NAME) that religiondots
                hatches as never counted and this entry draws: ge's Abkhazia and South Ossetia,
                md's Transnistria, az's Artsakh. not_drawn.py leaves them to its people test.

A file that fails to import, or lacks a field, is SKIPPED with a warning on stderr, so one
half-written country never stops the build for the others. `python tools/check_country.py <cc>`
says why.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIR = HERE / "countries"
REQUIRED = ("name", "source", "how", "grain", "view", "counts", "mappings", "place",
            "place_unit", "place_weight")

sys.path.insert(0, str(DIR))
sys.path.insert(0, str(HERE / "taxonomy"))


def load_one(cc):
    path = DIR / f"{cc}.py"
    spec = importlib.util.spec_from_file_location(f"ld_country_{cc}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    entry = getattr(mod, "ENTRY", None)
    if not isinstance(entry, dict):
        raise ValueError("no ENTRY dict")
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"ENTRY lacks {missing}")
    for k in ("how", "grain", "gap"):
        if "—" in str(entry.get(k, "")):
            raise ValueError(f"em dash in `{k}` (AGENT_BRIEF.md, voice)")
    entry = regrouped(entry)
    if "parts" in entry:
        bad = parts_problem(entry["parts"])
        if bad:
            # a display field: the country still draws, the viewer falls back to how/source
            print(f"  !! countries/{cc}.py `parts` ignored: {bad}", file=sys.stderr)
            entry = {k: v for k, v in entry.items() if k != "parts"}
    return entry


def regrouped(entry):
    """The entry with its node ids moved to where they are drawn (taxonomy/regroup.txt): counts()
    and `parts` nodes. Mappings keep writing the old ids."""
    from regroup import move
    entry = dict(entry)
    counts = entry["counts"]
    written = {}            # drawn id -> written id, for the weighter below

    def moved_counts(*a, **k):
        df = counts(*a, **k)
        if "node" in getattr(df, "columns", ()):
            df = df.copy()
            for n in df["node"].dropna().unique():
                written.setdefault(move(n), n)
            df["node"] = df["node"].map(move)
        return df
    entry["counts"] = moved_counts

    # A per-language weighter (kh, td, id, ...) is keyed by WRITTEN ids, but scatter.py asks it
    # with the drawn ones from counts(); translate back (2026-10-08: kh and id had been failing on
    # regrouped ids since 2026-10-06, hidden because their dots were rewritten in place).
    pw = entry.get("place_weight")
    if callable(pw):
        def place_weight(place):
            w = pw(place)
            return None if w is None else _WrittenIds(w, written)
        entry["place_weight"] = place_weight
    if isinstance(entry.get("parts"), (list, tuple)):
        entry["parts"] = [dict(p, nodes=[move(n) for n in p["nodes"]])
                          if isinstance(p, dict) and isinstance(p.get("nodes"), (list, tuple)) else p
                          for p in entry["parts"]]
    return entry


class _WrittenIds:
    """A weighter whose weights() is asked with drawn ids and answers with its written ones."""

    def __init__(self, inner, written):
        self._inner, self._written = inner, written

    def weights(self, node, *a, **k):
        return self._inner.weights(self._written.get(node, node), *a, **k)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def parts_problem(parts):
    """What is wrong with an ENTRY's `parts` (see the docstring), or None."""
    if not isinstance(parts, (list, tuple)) or not parts:
        return "not a non-empty list"
    for i, p in enumerate(parts):
        if not isinstance(p, dict):
            return f"part {i} is not a dict"
        for k in ("covers", "source"):
            if not isinstance(p.get(k), str) or not p[k].strip():
                return f"part {i} lacks `{k}`"
            if "—" in p[k]:
                return f"em dash in part {i} `{k}`"
        extra = set(p) - {"covers", "source", "people", "nodes", "rest"}
        if extra:
            return f"part {i} has unknown keys {sorted(extra)}"
        if sum(k in p for k in ("people", "nodes", "rest")) > 1:
            return f"part {i} has more than one of people/nodes/rest"
        if "people" in p and not isinstance(p["people"], int):
            return f"part {i} `people` is not an int"
        if "nodes" in p and not (isinstance(p["nodes"], (list, tuple))
                                 and all(isinstance(n, str) for n in p["nodes"])):
            return f"part {i} `nodes` is not a list of node ids"
    if sum(bool(p.get("rest")) for p in parts) > 1:
        return "more than one part has rest=True"
    return None


def load_all(verbose=True):
    out = {}
    for path in sorted(DIR.glob("[a-z][a-z].py")):
        cc = path.stem
        try:
            out[cc] = load_one(cc)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  !! countries/{cc}.py skipped: {e}", file=sys.stderr)
    return out


COUNTRIES = load_all()
