"""Average-day ridership per station, joined onto the built stations without a rebuild.

    python tools/station_riders.py                 # every source
    python tools/station_riders.py s12 orr         # only these sources (their countries'
                                                   # riders.json are rewritten from all of
                                                   # that country's sources, so name any one)

Each source is a module in tools/riders/ (listed in SOURCES) with

    KEY      short source key, written as "src" in riders.json
    CC       the region it fills
    FOLDER   its download folder under data/raw/riders/ (passed to records())
    META     {"name", "url", "licence", "counts", "note", and "per": "weekday" when the
             figure is an average working day rather than an average of all days}
    records(raw_dir) -> list of dicts, one per source station:
        name   the source's station name (alt: a list of more names to try)
        x, y   lon/lat if the source has them
        box    (w, s, e, n): with no x/y, look for the name only inside this box
        sid    a noritetsu station id when the source carries a code the ids are made of
               (SBB's BPUIC is ch's "c<uic>", SNCF's UIC is fr's "fr<uic>"); sids: several
               (S12's group and station codes, jp's "g<code>")
        n      average daily figure, already normalised to the source's stated meaning
        year   the year the figure is for (a fiscal or April-March year by its first year)
        band   text like "20-49" when the source gives a band, not a number (written as "b")
        op     the operator, when one source covers several (see COMBINE)
    MODES      the classes of line the source counts, from lines.json "kind": "rail" (rail,
               train, narrow gauge), "metro" (subway, light rail, monorail), "tram",
               "funicular". Only stations a line of those classes stops at are candidates.
    RADIUS_KM  how far a source station may lie from the matched station (default 1.5)
    COMBINE    "sum" when several records of the source land on one station and each is a
               separate operator's, line's or entrance's count; "max" when they are the same
               place counted twice (default). Records of one operator are never added across
               years: per "op", only its latest year's records count.
    TRUST_CODE a code match needs no name check (the ids are the source's own codes)
    FORCE      {source name: station id, or None to leave it out}, decided by hand
    FILL_ONLY  only fill stations no earlier source of the country has a figure for

Matching (match()): a code first (checked: same normalised name, or within 0.4 km, unless
TRUST_CODE); then the best name within RADIUS_KM (exact normalised name, then one name
containing the other on a word boundary or, for Hangul, anywhere, then a close spelling; a
contained or close name only within 0.6 x RADIUS_KM), nearest first; a record without a
position takes the station only if exactly one station (in its box, or the country) has that
whole name. A record nothing fits gets nothing: no figure is ever spread, guessed or copied.

Several sources for one country: when two land on one station (서울: Korail intercity,
Korail commuter lines, Seoul Metro, AREX), their figures are different operators' counts and
are added, if they mean the same thing ("per"); "src" becomes "a+b", "year" the latest and
"y0" the earliest when they differ.

A complex split over several ids (池袋's Seibu platforms are an N02 group of their own; S12
counts them with the rest; Zürich HB's register id and OSM's "Zürich Hauptbahnhof"): an id
with no figure of its own, carrying the same whole name as a station with a figure within
ALIAS_KM and served by a line class that figure counts, gets {"at": <that id>}. The app shows
the figure it points at; nothing is copied or added.

Outputs: dist/data/<cc>/riders.json ({id: {n, year, src[, y0]}}, {id: {b, year, src}} for a
band, {id: {at}}) and dist/data/riders_sources.json. Match reports, with every match, every
miss and the biggest source stations: data/raw/riders/_reports/<key>.txt.
"""
import difflib
import importlib
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "data"
RAW = ROOT / "data" / "raw" / "riders"
REPORTS = RAW / "_reports"
ALIAS_KM = 0.6

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Order matters only for the report. One module per source.
SOURCES = [
    "s12",        # jp
    "orr",        # gb
    "tfl",        # gb, London Underground / DLR / Overground / Elizabeth line gatelines
    "sncf",       # fr
    "sbb",        # ch
    "ns",         # nl
    "wd_be",      # be
    "tra",        # tw
    "taichung",   # tw, Taichung MRT
    # Taipei Metro not built: its only per-station file is an OD file of 312 MB a month
    "korail",     # kr, intercity
    "korail_gw",  # kr, Korail commuter lines
    "kric",       # kr, capital-area city railways
    "kr_metro",   # kr, Busan, Daegu, Daejeon, Gwangju metros, Busan-Gimhae
    "nsw",        # au, Sydney trains and metro (Opal)
    "nsw_lr",     # au, Sydney, Parramatta and Newcastle light rail
    "vic",        # au, Melbourne metro and V/Line
    "my_rapid",   # my, Klang Valley LRT, MRT, monorail
    "my_ktm",     # my, KTM Komuter, ETS, Intercity
    "th_drt",     # th, Airport Rail Link only
    "ibb",        # tr, Istanbul rail
    "cdmx",       # mx, Metro CDMX (READY only once a file is downloaded by hand)
    "utk",        # pl, UTK (READY only once a file is downloaded by hand)
]


# ---------------------------------------------------------------- names

SUFFIXES = [
    r"\brailway station\b", r"\btrain station\b", r"\bstation\b", r"\bhauptbahnhof\b",
    r"\bbahnhof\b", r"\bbhf\b", r"\bgare de\b", r"\bgare d\b", r"\bgare\b", r"\bstn\b",
    r"\bstazione\b", r"\bestacion\b", r"\bstation de\b",
]
JA_FOLD = str.maketrans({"ヶ": "ケ", "ヵ": "カ", "ｹ": "ケ", "之": "の", "ノ": "の", "〈": "(",
                         "〉": ")", "（": "(", "）": ")", "・": "", "　": " ", "臺": "台"})


def norm(s, keep_paren=False):
    """Lower case, no accents or punctuation, no 'station'-type words, no 駅/역 suffix."""
    if not s:
        return ""
    s = s.translate(JA_FOLD)
    s = unicodedata.normalize("NFKC", s)
    if not keep_paren:
        s = re.sub(r"\s*\((?:[^)]*)\)\s*", " ", s)        # parentheses: "(Kent)", "(RER)"
    s = re.sub(r"[駅站]$", "", s.strip())
    if len(s) > 2:
        s = re.sub(r"역$", "", s)
    # accents off; NFC puts Hangul syllables back together after NFKD split them into jamo
    s = unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)))
    s = s.lower().replace("&", " and ").replace("ß", "ss")
    s = re.sub(r"\bhbf?\b", "hauptbahnhof", s)
    s = re.sub(r"\bst\.?\s", "saint ", s)
    s = re.sub(r"\bste\.?\s", "sainte ", s)
    for suf in SUFFIXES:
        s = re.sub(suf, " ", s)
    s = re.sub(r"[^\w]+", " ", s)
    return " ".join(s.split())


def squash(s):
    return s.replace(" ", "")


def name_variants(*names, whole=False):
    """Normalised forms of each name and of each part of a bilingual 'A / B' or 'A - B'
    (whole=True: the names as they are, no parts)."""
    out = set()
    for nm in names:
        if not nm:
            continue
        parts = [nm] if whole else [nm] + re.split(r"\s+/\s+|\s+-\s+|\s+–\s+|/", nm)
        for p in parts:
            for v in (norm(p), norm(p, keep_paren=True)):
                if v:
                    out.add(v)
    return out


HANGUL = re.compile(r"^[가-힣 ]+$")


def name_score(a_set, b_set):
    """3 equal, 2 one contains the other on a word boundary (for Hangul, anywhere, the
    shorter at least 3 syllables: 김대중컨벤션센터 / 컨벤션센터), 1 close spelling, 0 none."""
    best = 0
    for a in a_set:
        for b in b_set:
            if a == b or squash(a) == squash(b):
                return 3
            if (re.search(r"(^| )" + re.escape(a) + r"( |$)", b)
                    or re.search(r"(^| )" + re.escape(b) + r"( |$)", a)):
                best = max(best, 2)
            elif HANGUL.match(a) and HANGUL.match(b):
                sa, sb = squash(a), squash(b)
                if min(len(sa), len(sb)) >= 3 and (sa in sb or sb in sa):
                    best = max(best, 2)
            elif best < 1 and min(len(a), len(b)) >= 4:
                if difflib.SequenceMatcher(None, squash(a), squash(b)).ratio() >= 0.88:
                    best = 1
    return best


# ---------------------------------------------------------------- geometry

def km(x1, y1, x2, y2):
    return math.hypot((x2 - x1) * math.cos(math.radians((y1 + y2) / 2)) * 111.32,
                      (y2 - y1) * 110.57)


CLASS = {"tram": "tram", "subway": "metro", "light_rail": "metro", "monorail": "metro",
         "funicular": "funicular"}


class Stations:
    def __init__(self, cc, modes=None):
        """modes: the classes of line ("rail", "metro", "tram", "funicular") a source counts;
        a station is a candidate only if a line of one of them stops there."""
        self.cc = cc
        d = json.loads((DIST / cc / "stations.json").read_text(encoding="utf-8"))["stations"]
        kinds = {l["id"]: CLASS.get(l.get("kind"), "rail") for l in
                 json.loads((DIST / cc / "lines.json").read_text(encoding="utf-8"))["lines"]}
        self.cls = {k: {kinds[l] for l in v.get("l", ()) if l in kinds} for k, v in d.items()}
        self.st = {k: v for k, v in d.items() if not v.get("j")
                   and (not modes or not self.cls[k] or self.cls[k] & set(modes))}
        self.names = {k: name_variants(v.get("n"), v.get("e")) for k, v in self.st.items()}
        self.grid = defaultdict(list)
        for k, v in self.st.items():
            self.grid[(int(v["x"] // 0.05), int(v["y"] // 0.05))].append(k)
        self.by_name = defaultdict(list)
        for k, ns in self.names.items():
            for nm in ns:
                self.by_name[squash(nm)].append(k)

    def near(self, x, y, r_km):
        cells = int(r_km / 4) + 1
        gx, gy = int(x // 0.05), int(y // 0.05)
        out = []
        for i in range(gx - cells, gx + cells + 1):
            for j in range(gy - cells, gy + cells + 1):
                for k in self.grid.get((i, j), ()):
                    v = self.st[k]
                    d = km(x, y, v["x"], v["y"])
                    if d <= r_km:
                        out.append((d, k))
        return sorted(out)


def match(rec, S, radius_km, trust_code=False):
    """-> (station id, how) or (None, reason)."""
    rn = name_variants(rec.get("name"), *(rec.get("alt") or []))
    has_xy = rec.get("x") is not None and rec.get("y") is not None
    for sid in ([rec["sid"]] if rec.get("sid") else []) + list(rec.get("sids") or []):
        if sid in S.st:
            v = S.st[sid]
            if trust_code:
                return sid, "code"
            if name_score(rn, S.names[sid]) >= 2:
                return sid, "code"
            if has_xy and km(rec["x"], rec["y"], v["x"], v["y"]) <= 0.4:
                return sid, "code+pos"
    if has_xy:
        best = None
        for d, k in S.near(rec["x"], rec["y"], radius_km):
            sc = name_score(rn, S.names[k])
            if sc == 0:
                continue
            # a contained or close name must also be close by
            if sc < 3 and d > radius_km * 0.6:
                continue
            key = (-sc, d)
            if best is None or key < best[0]:
                best = (key, k, sc)
        if best:
            return best[1], ["", "fuzzy", "contains", "name"][best[2]]
        return None, "no name match in reach"
    # no position: the record's whole name must equal a station's name or one of its parts
    cands = set()
    for nm in name_variants(rec.get("name"), *(rec.get("alt") or []), whole=True):
        cands.update(S.by_name.get(squash(nm), ()))
    if rec.get("box"):
        w, s, e, n = rec["box"]
        cands = {k for k in cands if w <= S.st[k]["x"] <= e and s <= S.st[k]["y"] <= n}
    if len(cands) == 1:
        return cands.pop(), "unique name"
    if len(cands) > 1:
        return None, f"name not unique ({len(cands)})"
    return None, "name not found"


# ---------------------------------------------------------------- run

def run_source(mod):
    recs = mod.records(RAW / mod.FOLDER)
    S = Stations(mod.CC, getattr(mod, "MODES", None))
    radius = getattr(mod, "RADIUS_KM", 1.5)
    combine = getattr(mod, "COMBINE", "max")
    hits = defaultdict(list)
    lines, unmatched = [], []
    hook = getattr(mod, "match_hook", None)
    force = getattr(mod, "FORCE", {})
    for r in recs:
        if r["name"] in force:
            sid = force[r["name"]]
            if sid is None:
                unmatched.append(f"{'left out by hand':28} {r['name']}  n={r.get('n')}")
                continue
            hits[sid].append(r)
            lines.append(f"forced     {r['name']!s:40} -> {sid} {S.st[sid]['n']}")
            continue
        sid, how = (hook(r, S) if hook else (None, None))
        if not sid:
            sid, how = match(r, S, radius, getattr(mod, "TRUST_CODE", False))
        if sid:
            hits[sid].append(r)
            v = S.st[sid]
            lines.append(f"{how:10} {r['name']!s:40} -> {sid} {v['n']}"
                         f"{' / ' + v['e'] if v.get('e') else ''}  n={r.get('n')}")
        else:
            unmatched.append(f"{how:28} {r['name']}  n={r.get('n')}  "
                             f"xy={r.get('x')},{r.get('y')}")
    out = {}
    for sid, rs in hits.items():
        if len(rs) > 1:
            lines.append(f"SEVERAL  {sid} {S.st[sid]['n']}: "
                         + "; ".join(f"{r['name']}={r.get('n')} ({r.get('year')}"
                                     f"{' ' + ','.join(r['ops']) if r.get('ops') else ''})"
                                     for r in rs) + f" ({combine})")
        numeric = [r for r in rs if r.get("n") is not None]
        if numeric:
            # never add one operator's figures of different years: per operator ("op", if
            # the source has several) keep the latest year's records
            latest = defaultdict(int)
            for r in numeric:
                latest[r.get("op", "")] = max(latest[r.get("op", "")], r["year"])
            numeric = [r for r in numeric if r["year"] == latest[r.get("op", "")]]
            if combine == "sum":
                n = sum(r["n"] for r in numeric)
            else:
                n = max(r["n"] for r in numeric)
            year = max(r["year"] for r in numeric)
            out[sid] = {"n": n, "year": year, "src": mod.KEY}
            if min(r["year"] for r in numeric) != year:
                out[sid]["y0"] = min(r["year"] for r in numeric)
        else:
            r = rs[0]
            out[sid] = {"b": r["band"], "year": r["year"], "src": mod.KEY}
    REPORTS.mkdir(parents=True, exist_ok=True)
    big = sorted(recs, key=lambda r: -(r.get("n") or 0))[:15]
    rep = [f"{mod.KEY} ({mod.CC}): {len(recs)} source stations, {sum(len(v) for v in hits.values())}"
           f" matched onto {len(out)} station ids; {len(unmatched)} unmatched; "
           f"{len(S.st)} stations (junctions left out) in {mod.CC}",
           "", "## biggest source stations and where they went"]
    where = {id(r): sid for sid, rs in hits.items() for r in rs}
    for r in big:
        sid = where.get(id(r))
        rep.append(f"  {r.get('n') or 0:>12,.0f}{' ' + r['band'] if r.get('band') else ''}"
                   f"  {r['name']}  -> "
                   + (f"{sid} {S.st[sid]['n']}" if sid else "UNMATCHED"))
    rep += ["", "## unmatched"] + sorted(unmatched) + ["", "## matched"] + lines
    (REPORTS / f"{mod.KEY}.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(rep[0])
    return out


def main(argv):
    here = Path(__file__).resolve().parent / "riders"
    names = [k for k in SOURCES if (here / f"{k}.py").exists()]
    names += sorted(p.stem for p in here.glob("*.py")             # modules not listed yet
                    if p.stem not in SOURCES and not p.stem.startswith("_"))
    mods = []
    for k in names:
        try:
            mods.append(importlib.import_module(f"riders.{k}"))
        except Exception as e:      # noqa: BLE001 - one broken module must not stop the rest
            print(f"  skipping riders/{k}.py: {e!r}")
    mods = [m for m in mods if hasattr(m, "KEY") and getattr(m, "READY", True)]
    want = set(argv) if argv else None
    ccs = sorted({m.CC for m in mods if want is None or m.KEY in want})
    meta_path = DIST / "riders_sources.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    for cc in ccs:
        merged = {}
        for m in mods:
            if m.CC != cc:
                continue
            meta[m.KEY] = m.META
            fill_only = getattr(m, "FILL_ONLY", False)
            for sid, v in run_source(m).items():
                if sid not in merged:
                    merged[sid] = v
                    continue
                if fill_only:       # an earlier source already counts this station
                    continue
                a = merged[sid]
                per_a = {meta[s].get("per", "day") for s in a["src"].split("+")}
                if "n" in a and "n" in v and per_a == {m.META.get("per", "day")}:
                    y0 = min(a.get("y0", a["year"]), v.get("y0", v["year"]))
                    y1 = max(a["year"], v["year"])
                    merged[sid] = {"n": a["n"] + v["n"], "year": y1,
                                   "src": a["src"] + "+" + v["src"]}
                    if y0 != y1:
                        merged[sid]["y0"] = y0
        for v in merged.values():
            if "n" in v:
                v["n"] = int(round(v["n"]))
        S = Stations(cc)
        # One complex split over several ids (池袋's Seibu platforms are an N02 group of their
        # own; S12 counts them with the rest): an id with no figure of its own, carrying the
        # same name as a station with a figure within ALIAS_KM, points at it with {"at": id}.
        # The app shows the figure it points at. Nothing is copied or added.
        alias = {}
        by_key = {m.KEY: m for m in mods}

        def counted(src):    # the line classes the figure's sources count
            out = set()
            for s in src.split("+"):
                out |= set(getattr(by_key[s], "MODES", None)
                           or {"rail", "metro", "tram", "funicular"})
            return out

        def whole(k):        # whole names (no 'A - B' parts), brackets kept
            v = S.st[k]
            return {squash(norm(x, keep_paren=True)) for x in (v.get("n"), v.get("e"))
                    if x} - {""}

        for k, v in S.st.items():
            if k in merged:
                continue
            for d, k2 in S.near(v["x"], v["y"], ALIAS_KM):
                if k2 in merged and "at" not in merged[k2] and whole(k) & whole(k2) \
                        and S.cls[k] & S.cls[k2] & counted(merged[k2]["src"]):
                    alias[k] = {"at": k2}
                    break
        merged.update(alias)
        print(f"  {cc}: {len(alias)} more ids of a split complex point at its figure")
        print(f"  {cc}: {len(merged) - len(alias)} of {len(S.st)} stations with a figure")
        (DIST / cc / "riders.json").write_text(
            json.dumps(dict(sorted(merged.items())), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
    meta_path.write_text(json.dumps(dict(sorted(meta.items())), ensure_ascii=False, indent=1),
                         encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
