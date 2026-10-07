"""Languages that stop at a border: the same or a close language drawn on different nodes on the
two sides of a frontier, or drawn on one side only.

    python tools/audit_crossborder.py                    every pair of neighbours, ranked
    python tools/audit_crossborder.py --cc mz,zw,mw      pairs touching any of these
    python tools/audit_crossborder.py --africa           pairs with an African country
    python tools/audit_crossborder.py --min 20000        only rows with this many people near the border

Anita, 2026-10-06: "in Africa many languages stop sharply at national borders" (Shona in Zimbabwe
but Ndau/Manyika in Mozambique; Chewa in Malawi but Nyanja in Mozambique...).

How:
  * Neighbours from country_shapes.geojson (shapes within ~5 km of each other).
  * Near the border: the dots (data/processed/dots_<cc>.geojson, one dot = counts.json's dot_value
    people) of country A lying within ~100 km of country B's shape (B buffered by --km, in degrees
    at the equator; good enough for a scan). Tallied per drawn node on each side.
  * Each drawn node is linked to Glottolog by its label (the whole label, the parts inside and
    outside brackets, then with a Bantu class prefix stripped: Ci-, Chi-, Xi-, Ki-, Shi-...),
    preferring a match whose Glottolog countries include a country that draws the node, and a
    language over a dialect. ALIAS below fixes names Glottolog spells differently. Unlinked nodes
    still take part in the asymmetry rows.
  * PAIRS: node x near the border on A's side, node y != x on B's side, both with people, and
    Glottolog puts them in the same language (one is a dialect of the other, or both of one
    language: "same"), or their languages share a subgroup at most --depth steps up ("close").
    Ranked by the smaller side's near-border people.
  * ASYMMETRY: node x is at least --share of A's near-border people, but under 0.5% of B's, with
    B's related nodes (same or close) listed beside it. Ranked by x's near-border people.
  * Where both nodes already share a parent group in the drawn tree, the row says "grouped" with
    the group; the scan does not hide them, since the colours may still differ sharply.

Reads only; the per-country near-border tallies are cached in --cache (default: the system temp
folder), keyed on the dots files' mtimes.
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GLOTTO = ROOT / "data" / "raw" / "glottolog"
AFRICA = set("dz ao bj bw bf bi cm cv cf td km cd cg ci dj eg gq er sz et ga gm gh gn gw ke ls lr ly "
             "mg mw ml mr mu ma mz na ne ng rw st sn sc sl so za ss sd tz tg tn ug eh zm zw xs".split())

# label word -> Glottolog NAME (looked up in languages.csv, so no glottocode is written from memory),
# where Glottolog spells or splits a language differently from what censuses print. A family is
# allowed where Glottolog splits what censuses count as one language (Fula, Kanuri, Kongo, Gbe).
ALIAS = {
    "kongo": "Kikongo Language Cluster", "kikongo": "Kikongo Language Cluster",
    "kituba": "Koongo-Kituba", "monokutuba": "Kituba (Congo)",
    "chewa": "Nyanja", "chichewa": "Nyanja", "cinyanja": "Nyanja",
    "twi": "Akan", "fante": "Akan", "fanti": "Akan", "asante": "Akan", "akuapem": "Akan",
    "brong": "Abron", "bono": "Abron", "agni": "Anyin", "anyi": "Anyin",
    "fulfulde": "Fula", "fulani": "Fula", "peul": "Fula", "pulaar": "Pular", "fula": "Fula",
    "malinke": "Mandinka", "dioula": "Dyula", "jula": "Dyula", "kanuri": "Kanuric",
    "tamajaq": "Tamasheq", "songhai": "Zarma-Kaado-Dendi", "zarma": "Zarma-Kaado",
    "djerma": "Zarma-Kaado", "sesotho": "Southern Sotho", "sotho": "Southern Sotho",
    "setswana": "Tswana", "siswati": "Swati", "swazi": "Swati", "xitsonga": "Tsonga",
    "changana": "Tsonga", "isizulu": "Zulu", "isixhosa": "Xhosa", "ndebele": "Zimbabwean Ndebele",
    "isindebele": "Zimbabwean Ndebele", "kiswahili": "Swahili", "oshiwambo": "Ndonga (R.20)",
    "otjiherero": "Herero", "silozi": "Lozi", "kirundi": "Rundi", "luganda": "Ganda",
    "lingala": "Kinshasa Lingala", "tshiluba": "Luba-Lulua", "ciluba": "Luba-Lulua",
    "kiluba": "Luba-Katanga", "oromo": "Nuclear Oromo", "tigrigna": "Tigrinya",
    "wolof": "Wolof", "moore": "Mossi", "zaghawa": "Beria", "tamazight": "Central Moroccan Berber",
    "masalit": "Masalit", "luo": "Luo (Kenya and Tanzania)", "dholuo": "Luo (Kenya and Tanzania)",
    "kalenjin": "Kalenjin", "maasai": "Masai", "ateso": "Teso", "karamojong": "Karamojong",
    "acholi": "Acoli", "lango": "Lango (Uganda)", "chiyao": "Yao", "emakhuwa": "Makhuwa",
    "lomwe": "Mozambique Lomwe", "elomwe": "Mozambique Lomwe", "chitumbuka": "Tumbuka",
    "sena": "Sena", "cisena": "Sena", "kimbundu": "Kimbundu", "kwanyama": "Kuanyama",
    "gbe": "Gbe",
}
PREFIXES = ("tshi", "chi", "ci", "xi", "ki", "shi", "si", "isi", "se", "otji", "oshi", "lu", "ru",
            "e", "kin", "kiny", "mo", "lo", "ichi", "i")


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def load_glottolog():
    langs = {}
    by_name = defaultdict(list)
    with open(GLOTTO / "languages.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g = r["Glottocode"]
            langs[g] = r
            by_name[norm(r["Name"])].append(g)
            # "Aja (Benin)" also answers to "Aja"
            base = re.sub(r"\s*\(.*\)\s*", "", r["Name"])
            if base != r["Name"]:
                by_name[norm(base)].append(g)
    path = {}
    with open(GLOTTO / "values.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Parameter_ID"] == "classification":
                path[r["Language_ID"]] = r["Value"].split("/")
    return langs, by_name, path


class Glotto:
    def __init__(self):
        self.langs, self.by_name, self.cls = load_glottolog()
        self.exact = {}
        for g, r in self.langs.items():
            if r["Level"] in ("language", "family"):
                self.exact.setdefault(r["Name"], g)
        bad = [v for v in ALIAS.values() if v not in self.exact]
        if bad:
            raise SystemExit(f"ALIAS names not in Glottolog: {bad}")

    def alias(self, n):
        return self.exact[ALIAS[n]]

    def lang_of(self, g):
        """The Glottolog language a code belongs to (itself, or a dialect's language). A family
        (from ALIAS) stands for itself."""
        r = self.langs.get(g)
        if r is None:
            return None
        if r["Level"] == "dialect":
            return r["Language_ID"] or None
        return g

    def lineage(self, g):
        return self.cls.get(g, []) + [g]

    def name(self, g):
        r = self.langs.get(g)
        return r["Name"] if r else g

    def pick(self, cands, ccs):
        rank = {"language": 0, "dialect": 1, "family": 2}

        cands = [g for g in dict.fromkeys(cands) if g in self.langs]
        return min(cands, key=lambda g: (not self.here(g, ccs), rank.get(self.langs[g]["Level"], 3))) \
            if cands else None

    def here(self, g, ccs):
        """Glottolog places g (or a dialect's language) in one of these countries."""
        r = self.langs[g]
        if r["Level"] == "dialect" and r["Language_ID"] in self.langs:
            r = self.langs[r["Language_ID"]]
        return bool(set(r["Countries"].lower().split(";")) & ccs) if r["Countries"] else False

    def link(self, label, nid, ccs):
        """A match placed in a country that draws the node wins over an earlier word's match
        elsewhere ("Isan" is also a Yopno dialect in New Guinea)."""
        words = []
        inside = re.findall(r"\(([^)]*)\)", label)
        outside = re.sub(r"\([^)]*\)", " ", label)
        for s in [label, outside] + inside + [nid.rsplit(".", 1)[-1].replace("_", " ")]:
            for part in re.split(r"[/,;]| or ", s):
                if part.strip():
                    words.append(part.strip())
        found = []
        for strip in (False, True):
            for w in words:
                n = norm(w)
                keys = [n] if not strip else [n[len(p):] for p in PREFIXES
                                              if n.startswith(p) and len(n) - len(p) >= 3]
                for m in keys:
                    if m in ALIAS:
                        return self.alias(m)
                    if m in self.by_name:
                        g = self.pick(self.by_name[m], ccs)
                        if g and self.langs[g]["Level"] in ("language", "dialect"):
                            if self.here(g, ccs):
                                return g
                            found.append(g)
        return found[0] if found else None

    def relation(self, a, b):
        """('same', lang) / ('close', lca, steps) / None for two linked codes."""
        la, lb = self.lang_of(a), self.lang_of(b)
        if not la or not lb:
            return None
        if la == lb:
            return ("same", la, 0)
        pa, pb = self.lineage(la), self.lineage(lb)
        if la in pb:          # a family (Fula, Kongo) holding the other's language
            return ("same", la, 0)
        if lb in pa:
            return ("same", lb, 0)
        k = 0
        while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
            k += 1
        if k == 0:
            return None
        steps = max(len(pa) - k, len(pb) - k)
        return ("close", pa[k - 1], steps)


def neighbours(shapes, tol):
    import shapely
    ccs = list(shapes)
    geoms = [shapes[c] for c in ccs]
    tree = shapely.STRtree(geoms)
    out = set()
    for i, c in enumerate(ccs):
        for j in tree.query(geoms[i].buffer(tol)):
            d = ccs[j]
            if d != c and geoms[i].distance(geoms[j]) <= tol:
                out.add(tuple(sorted((c, d))))
    return sorted(out)


def load_dots(cc):
    p = ROOT / "data" / "processed" / f"dots_{cc}.geojson"
    if not p.exists():
        return None
    import numpy as np
    xs, ys, ns = [], [], []
    pat = re.compile(r'"coordinates":\s*\[\s*([-0-9.eE]+),\s*([-0-9.eE]+)\s*\]\s*\},\s*"properties":\s*\{"n":\s*"([^"]+)"')
    with open(p, encoding="utf-8") as f:
        text = f.read()
    for m in pat.finditer(text):
        xs.append(float(m.group(1)))
        ys.append(float(m.group(2)))
        ns.append(m.group(3))
    return np.array(xs), np.array(ys), np.array(ns, dtype=object)


def near_tallies(pairs, shapes, deg, cache_path):
    """{(a, b): {node: dots of a within deg of b}} for both orders of each pair."""
    import shapely
    import numpy as np
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cache = {}
    by_cc = defaultdict(list)
    for a, b in pairs:
        by_cc[a].append(b)
        by_cc[b].append(a)
    out = {}
    for cc in sorted(by_cc):
        p = ROOT / "data" / "processed" / f"dots_{cc}.geojson"
        if not p.exists():
            continue
        stamp = f"{p.stat().st_mtime_ns}:{deg}"
        todo = [b for b in by_cc[cc] if cache.get(f"{cc}>{b}", {}).get("stamp") != stamp]
        if todo:
            d = load_dots(cc)
            if d is not None:
                xs, ys, ns = d
                for b in todo:
                    zone = shapes[b].buffer(deg)
                    shapely.prepare(zone)
                    m = shapely.contains_xy(zone, xs, ys)
                    vals, cnt = np.unique(ns[m], return_counts=True) if m.any() else ([], [])
                    cache[f"{cc}>{b}"] = {"stamp": stamp,
                                          "nodes": {str(v): int(c) for v, c in zip(vals, cnt)}}
        for b in by_cc[cc]:
            if f"{cc}>{b}" in cache:
                out[(cc, b)] = cache[f"{cc}>{b}"]["nodes"]
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default="")
    ap.add_argument("--africa", action="store_true")
    ap.add_argument("--km", type=float, default=100)
    ap.add_argument("--min", type=float, default=5000, help="people near the border, smaller side")
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "ld_crossborder_cache.json"))
    a = ap.parse_args()

    import shapely
    from shapely.geometry import shape
    sh = json.loads((ROOT / "country_shapes.geojson").read_text(encoding="utf-8"))
    shapes = {}
    for f in sh["features"]:
        g = shape(f["geometry"]).buffer(0)
        cc = f["properties"]["cc"]
        shapes[cc] = shapely.union(shapes[cc], g) if cc in shapes else g
    cj = json.loads((ROOT / "data" / "processed" / "counts.json").read_text(encoding="utf-8"))
    dv = cj["dot_value"]
    drawn = set(cj["countries"])
    shapes = {c: g for c, g in shapes.items() if c in drawn}
    pairs = neighbours(shapes, 0.05)
    only = set(a.cc.split(",")) - {""}
    if only:
        pairs = [p for p in pairs if set(p) & only]
    if a.africa:
        pairs = [p for p in pairs if set(p) & AFRICA]
    near = near_tallies(pairs, shapes, a.km / 111.0, Path(a.cache))

    nodes = json.loads((ROOT / "taxonomy" / "languages.json").read_text(encoding="utf-8"))["nodes"]
    label = {n["id"]: n["label"] for n in nodes}
    parent = {n["id"]: n["parent"] for n in nodes}
    where = defaultdict(set)
    for cc, c in cj["countries"].items():
        for k in c.get("dots", {}):
            where[k].add(cc)
    G = Glotto()
    link_memo = {}

    def link(n):
        if n not in link_memo:
            link_memo[n] = G.link(label.get(n, n), n, where.get(n, set()))
        return link_memo[n]

    def tree_group(x, y):
        """The deepest shared parent below the family root, if any."""
        ax, cur = [], parent.get(x)
        while cur:
            ax.append(cur)
            cur = parent.get(cur)
        cur = parent.get(y)
        while cur:
            if cur in ax and cur.count(".") >= 1:
                return cur
            cur = parent.get(cur)
        return None

    pair_rows, asym_rows = [], []
    for A, B in pairs:
        for X, Y in ((A, B), (B, A)):
            nx = {k: v * dv for k, v in near.get((X, Y), {}).items()}
            ny = {k: v * dv for k, v in near.get((Y, X), {}).items()}
            tx, ty = sum(nx.values()) or 1, sum(ny.values()) or 1
            # pairs, once per unordered pair of countries
            if X == A:
                for x, px in nx.items():
                    gx = link(x)
                    if not gx:
                        continue
                    for y, py in ny.items():
                        if y == x or min(px, py) < a.min:
                            continue
                        gy = link(y)
                        if not gy:
                            continue
                        rel = G.relation(gx, gy)
                        if not rel or (rel[0] == "close" and rel[2] > a.depth):
                            continue
                        pair_rows.append((min(px, py), X, x, px, Y, y, py, rel, tree_group(x, y)))
            # asymmetry
            for x, px in nx.items():
                if px < a.min or px / tx < a.share or ny.get(x, 0) / ty >= 0.005:
                    continue
                gx = link(x)
                rel_y = []
                if gx:
                    for y, py in ny.items():
                        gy = link(y)
                        rel = G.relation(gx, gy) if gy else None
                        if rel and (rel[0] == "same" or rel[2] <= a.depth + 1):
                            rel_y.append((py, y, rel))
                rel_y.sort(key=lambda t: -t[0])
                asym_rows.append((px, X, x, px / tx, Y, ny.get(x, 0), ty, rel_y))

    def fmt_rel(rel):
        if rel[0] == "same":
            return f"same language ({G.name(rel[1])})"
        return f"close: {G.name(rel[1])}, {rel[2]} step{'s' if rel[2] > 1 else ''}"

    pair_rows.sort(key=lambda r: -r[0])
    asym_rows.sort(key=lambda r: -r[0])
    if a.top:
        pair_rows, asym_rows = pair_rows[:a.top], asym_rows[:a.top]
    print(f"# {len(pairs)} neighbour pairs, within {a.km:.0f} km; people = dots x {dv}\n")
    print("## PAIRS: different nodes on the two sides that Glottolog relates\n")
    for m, X, x, px, Y, y, py, rel, grp in pair_rows:
        g = f"  [grouped: {label.get(grp, grp)}]" if grp else ""
        print(f"{m:>11,.0f}  {X}:{label.get(x, x)} {px:,.0f}  <>  {Y}:{label.get(y, y)} {py:,.0f}"
              f"  -- {fmt_rel(rel)}{g}")
        print(f"{'':13}{x}  |  {y}")
    print("\n## ASYMMETRY: big near the border on one side, under 0.5% on the other\n")
    for px, X, x, sh_, Y, py, ty, rel_y in asym_rows:
        rels = "; ".join(f"{label.get(y, y)} {p:,.0f} ({'same' if r[0] == 'same' else G.name(r[1])})"
                         for p, y, r in rel_y[:4]) or ("(not linked to Glottolog)" if not link(x)
                                                       else "(nothing related across)")
        print(f"{px:>11,.0f}  {X}>{Y}  {label.get(x, x)} {100 * sh_:.0f}% of {X}'s border zone; "
              f"{py:,.0f} on {Y}'s side.  {Y} has: {rels}")
        print(f"{'':13}{x}")


if __name__ == "__main__":
    main()
