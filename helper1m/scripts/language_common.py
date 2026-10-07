"""Shared builder for language composition pies drawn from maps/languagedots.

languagedots (READ-ONLY here) already holds, for Pakistan, Afghanistan and Iran, the language
counts per unit (`data/normalized/<cc>.csv`), the mapping from each source's labels to its
language tree (`taxonomy/<cc><year>.py`, `resolve(label) -> node id`), and one colour per node
(`taxonomy/languages.json`, built by its taxonomy/build.py). This module takes counts already
resolved to nodes and keyed by helper1m unit codes, groups the nodes, colours them, and writes
`countries/<id>/composition.json` in the format the viewer reads (README, "Composition pies").

A per-country script (scripts/<id>/language.py) does only what differs: read its source, map
labels with languagedots' mapping, and join its units to helper1m codes. Then:

    comp = Composition("pakistan")
    comp.add(3, "PK2-awaran-awaran", "indoeuropean.iranian.balochi", 176418)   # measured level
    ...
    comp.write(label="Mother tongue", year=2023, source="...")

A row goes in at the level its source measured it; it is summed up through helper1m's own
parent codes to every coarser level and never pushed down, so a level the source does not reach
draws nothing.

GROUPS. A node gets its own group when it reaches NAT_SHARE of the country, or SHARE_MIN of some
unit with at least ABS_MIN people there (India's rule, scripts/india/language.py, with the
national bar as a share so it scales with the country). Everything else folds into "Other
languages", whose title names what it holds. A per-country `keep` set forces a group.

IRAN (not built yet; helper1m's Iran is scripts/iran/). languagedots' `ir.csv` is per province,
geo_id = the English province name (31: "Alborz" ... "Zanjan", 1395/2016 census provinces), labels
mapped by `taxonomy/ir2020.py`, 79,926,270 people. A scripts/iran/language.py needs only:
    ir2020 = lc.ld_module("ir2020"); df = pd.read_csv(lc.LD_NORM / "ir.csv")
    code = {<geo_id>: <helper1m adm1 code>}   # by name, every one checked, all 31 used
    for r in df.itertuples():
        node = ir2020.resolve(r.source_category)          # None for "No answer" (not in ir.csv)
        if node: comp.add(1, code[r.geo_id], node, r.count)
    comp.write(label="Language at home (survey)", year="2005-20", pop_year=2016, ...)
It is a World Values Survey proxy (about 4,200 adults, waves 5 and 7 pooled), so say so in the
label or titles as Afghanistan does; province level only.

COLOURS. One palette for every country built here, `scripts/language_colors.csv`, keyed by
languagedots node id, so Pashto is the same colour in Pakistan, Afghanistan and Iran. A row the
file lacks is appended with languagedots' colour for the node (its `color_own`, the desaturated
one, for a group node such as "Iranian", which languagedots draws as "language not named"). An
existing row is never rewritten: the file is hand-editable.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import csv
import importlib
import json
import sys
import time
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1]
MAPS = HELPER.parent
LD = MAPS / "languagedots"
LD_NORM = LD / "data" / "normalized"
LD_LANGS = LD / "taxonomy" / "languages.json"
COLORS = Path(__file__).with_name("language_colors.csv")

NAT_SHARE = 0.001      # national share for a group of its own (India: 1M of 1.2 billion)
SHARE_MIN = 0.30       # ... or this share of some unit
ABS_MIN = 10_000       # ... with at least this many people there
OTHER = "other"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ld_module(name, sub="taxonomy"):
    """Import a languagedots module (a census mapping, a source reader) without writing into
    languagedots: no __pycache__."""
    sys.dont_write_bytecode = True
    p = str(LD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module(name)


_NODES = None


def nodes():
    """languagedots' tree: id -> {label, color, color_own, parent}."""
    global _NODES
    if _NODES is None:
        with LD_LANGS.open(encoding="utf-8") as fh:
            _NODES = {n["id"]: n for n in json.load(fh)["nodes"]}
    return _NODES


def node(nid):
    """The languages.json entry for a mapping's node id. languages.json holds ids after
    taxonomy/regroup.py has moved them, so an id it lacks is moved the same way first."""
    n = nodes()
    if nid in n:
        return n[nid]
    moved = ld_module("regroup").move(nid)
    if moved in n:
        return n[moved]
    sys.exit(f"node {nid!r} (moved: {moved!r}) is not in {LD_LANGS}")


def node_color(nid):
    n = node(nid)
    has_children = any(m.get("parent") == n["id"] for m in nodes().values())
    return n.get("color_own") if has_children and n.get("color_own") else n["color"]


def read_level(country, level):
    """helper1m's own features for one level: code -> properties (with `_pt`, a
    representative point). Reads adm{N}.geojson, or the adm{N}/ folder of a split level."""
    import shapely.geometry
    base = HELPER / "countries" / country
    files = [base / f"adm{level}.geojson"]
    if not files[0].exists():
        files = sorted((base / f"adm{level}").glob("*.geojson"))
    if not files:
        sys.exit(f"{country}: no adm{level} geojson; build the country first")
    out = {}
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for f in json.load(fh)["features"]:
                p = dict(f["properties"])
                pt = shapely.geometry.shape(f["geometry"]).representative_point()
                p["_pt"] = (pt.x, pt.y)
                out[p["code"]] = p
    return out


class Composition:
    def __init__(self, country):
        self.country = country
        meta = json.loads((HELPER / "countries" / country / "meta.json").read_text("utf-8"))
        self.levels = sorted(l["level"] for l in meta["admin_levels"])
        self.feat = {l: read_level(country, l) for l in self.levels}
        self.rows = {}        # (level, code) -> {node: count}
        self.extra = {}       # non-language group key -> {en, title, color}

    def add(self, level, code, key, count):
        if code not in self.feat[level]:
            sys.exit(f"{self.country}: no level-{level} unit {code!r} in helper1m")
        if count:
            d = self.rows.setdefault((level, code), {})
            d[key] = d.get(key, 0) + int(count)

    def add_extra(self, key, en, title, color):
        """A group that is not a language node (Afghanistan's 'Not described')."""
        self.extra[key] = {"en": en, "title": title, "color": color}

    # ------------------------------------------------------------------ groups

    def _groups(self, keep, titles, names):
        nat, best = {}, {}
        for (lvl, code), d in self.rows.items():
            tot = sum(d.values())
            for k, v in d.items():
                nat[k] = nat.get(k, 0) + v
                if v >= ABS_MIN:
                    best[k] = max(best.get(k, 0), v / tot)
        total = sum(nat.values())
        own = [k for k in nat if k in keep or k in self.extra
               or nat[k] / total >= NAT_SHARE or best.get(k, 0) >= SHARE_MIN]
        own = [k for k in own if k != OTHER]
        own.sort(key=lambda k: -nat[k])
        folded = sorted((k for k in nat if k not in own), key=lambda k: -nat[k])
        groups = []
        for k in own:
            if k in self.extra:
                groups.append({"key": k, **self.extra[k]})
                continue
            en = names.get(k) or node(k)["label"]
            groups.append({"key": k, "en": en, "title": titles.get(k, en)})
        fold_names = [names.get(k) or (node(k)["label"] if k != OTHER else None)
                      for k in folded]
        listed = [f"{n} ({nat[k]:,})" for n, k in zip(fold_names, folded) if n]
        t = titles.get(OTHER, "Languages too small or scattered for a colour of their own")
        if listed:
            t += ": " + ", ".join(listed)
        if OTHER in nat:
            t += f"; and the source's own 'other' ({nat[OTHER]:,})"
        if folded:
            groups.append({"key": OTHER, "en": "Other languages", "title": t})
        assign = {k: i for i, k in enumerate(own)}
        for k in folded:
            assign[k] = len(groups) - 1
        return groups, assign, nat, total

    # ------------------------------------------------------------------ colours

    def _palette(self, groups):
        have = {}
        if COLORS.exists():
            with COLORS.open(encoding="utf-8", newline="") as fh:
                have = {r["key"]: r["color"] for r in csv.DictReader(fh)}
        new = []
        for g in groups:
            if g["key"] in have:
                continue
            c = g.get("color") or node_color(g["key"])
            have[g["key"]] = c
            new.append({"key": g["key"], "en": g["en"], "color": c,
                        "from": "helper1m" if g["key"] in self.extra else "languagedots",
                        "first_used": self.country})
        if new:
            exists = COLORS.exists()
            with COLORS.open("a", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["key", "en", "color", "from", "first_used"])
                if not exists:
                    w.writeheader()
                w.writerows(new)
            log(f"added {len(new)} rows to {COLORS.name}")
        return have

    # ------------------------------------------------------------------ write

    def write(self, label, year, source, keep=(), titles=None, names=None, pop_year=None):
        """Group, sum up the levels, check and write composition.json. `names` overrides a
        group's display name, `titles` its hover title. `pop_year` names the helper1m
        population year the totals are compared with (a report only)."""
        titles, names = titles or {}, names or {}
        groups, assign, nat, total = self._groups(set(keep), titles, names)
        n_g = len(groups)
        colors = self._palette(groups)

        # Each row into its own level and every coarser one, through helper1m's parents.
        acc = {l: {} for l in self.levels}   # level -> code -> [counts, sx, sy, sw]
        for (lvl, code), d in self.rows.items():
            v = [0] * n_g
            for k, c in d.items():
                v[assign[k]] += c
            w = sum(v)
            x, y = self.feat[lvl][code]["_pt"]
            l, c = lvl, code
            while True:
                a = acc[l].setdefault(c, [[0] * n_g, 0.0, 0.0, 0])
                a[0] = [p + q for p, q in zip(a[0], v)]
                a[1] += w * x
                a[2] += w * y
                a[3] += w
                parent = self.feat[l][c].get("parent_code")
                if l == self.levels[0]:
                    break
                if parent not in self.feat[l - 1]:
                    sys.exit(f"{self.country}: level-{l} {c} has parent {parent!r}, "
                             f"not a level-{l - 1} unit")
                l, c = l - 1, parent

        out = {}
        for l in self.levels:
            units = {}
            for code, (v, sx, sy, sw) in acc[l].items():
                g = sorted((i for i in range(n_g) if v[i] > 0), key=lambda i: -v[i])
                units[code] = {"t": sum(v), "x": round(sx / sw, 4), "y": round(sy / sw, 4),
                               "g": g, "k": [v[i] for i in g]}
            if units:
                out[str(l)] = units

        # Checks: each unit's groups add to its total, and every level holds every person
        # measured at or below it.
        for l, units in out.items():
            for code, u in units.items():
                assert sum(u["k"]) == u["t"], (l, code)
            below = sum(sum(d.values()) for (lv, _), d in self.rows.items() if lv >= int(l))
            got = sum(u["t"] for u in units.values())
            if got != below:
                sys.exit(f"level {l}: {got:,} people, expected {below:,}")
        if sum(u["t"] for u in out[str(self.levels[0])].values()) != total:
            sys.exit("top level does not hold every row")
        for l in self.levels:
            have = out.get(str(l), {})
            missing = sorted(set(self.feat[l]) - set(have))
            log(f"level {l}: {len(have):,} of {len(self.feat[l]):,} units have a pie, "
                f"{sum(u['t'] for u in have.values()):,} people"
                + (f"; none for {len(missing)}: {', '.join(missing[:12])}"
                   + (" ..." if len(missing) > 12 else "") if missing else ""))
            if pop_year:
                ratios = []
                for code, u in have.items():
                    p = (self.feat[l][code].get("populations") or {}).get(str(pop_year))
                    if p:
                        ratios.append((u["t"] / p, code, self.feat[l][code].get("name")))
                if ratios:
                    ratios.sort()
                    lo, hi = ratios[0], ratios[-1]
                    log(f"  composition total / helper1m {pop_year} population: "
                        f"{lo[0]:.3f} ({lo[2]}) to {hi[0]:.3f} ({hi[2]})")

        doc = {
            "label": label,
            "year": year,
            "source": source,
            "groups": [{"key": g["key"], "en": g["en"], "title": g["title"],
                        "color": colors[g["key"]]} for g in groups],
            "levels": out,
        }
        path = HELPER / "countries" / self.country / "composition.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
        log(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB), {n_g} groups")
        top = [0] * n_g
        for u in out[str(self.levels[0])].values():
            for i, k in zip(u["g"], u["k"]):
                top[i] += k
        for i in sorted(range(n_g), key=lambda i: -top[i]):
            print(f"    {groups[i]['en']:<34} {top[i]:>12,}  {top[i] / total:6.2%}  "
                  f"{colors[groups[i]['key']]}")
        return doc
