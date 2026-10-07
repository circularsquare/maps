"""Tie each COD 2018 atom to the 2026 unit (aiyl aimak, town or city) its
people are now counted in.

The 2024 workbook is still on the old units (its codes are COD's codes), and
the 2025 and 2026 workbooks are on the new ones, so the link runs through the
villages: each 2024 village sits under an old unit, and the same village, found
by name in the 2026 workbook, sits under a new one.

Usage (diagnostics): python crosswalk.py
"""
import difflib
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import nsc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(HELPER, "data", "kyrgyzstan", "raw")

UNIT_KINDS = ["aa", "town", "city", "city_pgt"]


def norm(n):
    n = str(n).lower().replace("ё", "е").replace("c.", "с.").replace("t", "т")
    n = re.sub(r"\(.*?\)", " ", n)
    n = re.sub(r"^\s*(село|с\.|с |пгт\.?|п\.|г\.|уч\.)\s*", "", n)
    n = re.sub(r"\bим\.\s*", "", n)
    n = re.sub(r"айылный аймак|айылный|аймак", "", n)
    return re.sub(r"[^а-яa-z0-9ӊөүң]", "", n)


def stem(n):
    """Loose form: Kyrgyz letters folded, Russian place suffixes dropped."""
    n = n.translate(str.maketrans("өүңӊюёы", "оуннуеи"))
    n = re.sub(r"(ская|ское|ский|ово|ево|ино|ка|а|о|ое)$", "", n)
    return n


TR = dict(zip("абвгдеёжзийклмнңоөпрстуүфхцчшщъыьэюя",
              ["a", "b", "v", "g", "d", "e", "e", "j", "z", "i", "i", "k", "l", "m", "n", "n",
               "o", "o", "p", "r", "s", "t", "u", "u", "f", "h", "ts", "ch", "sh", "sh", "",
               "i", "", "e", "yu", "ya"]))


def latkey(s):
    """A rough Latin key, so a GeoNames place with only Latin names can be found."""
    s = str(s).lower()
    if re.search("[а-яөүң]", s):
        s = "".join(TR.get(c, c) for c in s)
    s = s.replace("dzh", "j").replace("zh", "j").replace("kh", "h").replace("y", "i")
    s = re.sub(r"^(imeni|im)", "", s)
    return re.sub(r"[^a-z]", "", s)


def places(t, year):
    """Every named place row with the unit it is counted in."""
    k = t.kind
    p = t[k.isin(["village", "town_village", "city_village", "town", "city", "city_pgt",
                  "capital_part"])].copy()
    p = p[p.unit.notna()]
    p["nn"] = p.name.map(norm)
    p["st"] = p.nn.map(stem)
    p["lat"] = p.nn.map(latkey)
    p["year"] = year
    return p


def match_places(p24, p26, neighbours):
    """For each 2024 place: the 2026 unit, or None. Returns p24 with unit26, how.

    A name is looked for in the same rayon first, then in the rayons that border
    it (villages did move between rayons in 2024-25), never further: common
    names (Kyzyl-Tuu, Kurulush, Ak-Terek) recur across an oblast."""
    by = {}
    for key in ["nn", "st"]:
        by[key] = p26.groupby(["rayon", key]).unit.agg(lambda s: sorted(set(s)))
    out_unit, out_how, cands = [], [], []
    for r in p24.itertuples():
        found = None
        scopes = [("rayon", [r.rayon]), ("near", sorted(neighbours.get(r.rayon, [])))]
        for key in ["nn", "st"]:
            for scope, rays in scopes:
                u = sorted({x for ry in rays for x in by[key].get((ry, getattr(r, key)), [])})
                if u:
                    found = (u, f"{key}-{scope}")
                    break
            if found:
                break
        if not found:
            pool = p26[p26.rayon == r.rayon]
            best = difflib.get_close_matches(r.nn, pool.nn.unique().tolist(), n=3, cutoff=0.85)
            if best:
                u = sorted(set(pool[pool.nn == best[0]].unit))
                found = (u, f"fuzzy:{best[0]}")
        if not found:
            out_unit.append(None); out_how.append("none"); cands.append([])
        elif len(found[0]) == 1:
            out_unit.append(found[0][0]); out_how.append(found[1]); cands.append(found[0])
        else:
            out_unit.append(None); out_how.append("ambig-" + found[1]); cands.append(found[0])
    p24 = p24.copy()
    p24["unit26"], p24["how"], p24["cands"] = out_unit, out_how, cands
    # a village that matches by name but not by size is a namesake: drop it
    for i, r in p24[p24.unit26.notna()].iterrows():
        same = p26[(p26.unit == r.unit26) & ((p26.nn == r.nn) | (p26.st == r.st))]["pop"]
        a, b = r["pop"], (same.iloc[0] if len(same) else float("nan"))
        if pd.notna(a) and pd.notna(b) and max(a, b) >= 300 and not (0.7 <= (b + 1) / (a + 1) <= 1.4):
            p24.at[i, "unit26"] = None
            p24.at[i, "how"] = "size-mismatch:" + r.how
    # cities and towns can change rank (Kant, Kara-Balta became cities of oblast
    # significance), so a town name is also looked for oblast-wide among towns
    towns26 = p26[p26.kind.isin(["town", "city"])]
    for i, r in p24[p24.unit26.isna() & p24.kind.isin(["town", "city", "city_pgt"])].iterrows():
        pool = towns26[towns26.oblast == r.oblast]
        hit = pool[(pool.nn == r.nn) | (pool.st == r.st)]
        if not len(hit):
            hit = pool[pool.nn.isin(difflib.get_close_matches(r.nn, pool.nn.tolist(), n=1, cutoff=0.85))]
        if hit.unit.nunique() == 1:
            p24.at[i, "unit26"] = hit.unit.iloc[0]
            p24.at[i, "how"] = "town-oblast"
    # ambiguous names: take the candidate the rest of the old unit went to
    vote = p24[p24.unit26.notna()].groupby("unit").unit26.agg(lambda s: Counter(s))
    for i, r in p24[p24.how.str.startswith("ambig")].iterrows():
        v = vote.get(r.unit, Counter())
        best = [c for c in r.cands if v.get(c)]
        if len(best) == 1:
            p24.at[i, "unit26"] = best[0]
            p24.at[i, "how"] = r.how + "+vote"
    return p24


def rayon_neighbours(g1, g2):
    """SOATE rayon id -> set of bordering rayon ids (cities and capitals count)."""
    import geopandas as gpd
    from atoms import soate
    a = g2[["adm2_pcode", "geometry"]].rename(columns={"adm2_pcode": "c"})
    b = g1[g1.adm1_pcode.isin(["KG11000000000", "KG21000000000"])][["adm1_pcode", "geometry"]]
    a = pd.concat([a, b.rename(columns={"adm1_pcode": "c"})])
    a = gpd.GeoDataFrame(a, geometry="geometry", crs=g2.crs)
    a["r"] = a.c.map(lambda c: soate(c)[:8] + "000000")
    a["geometry"] = a.geometry.buffer(0.002)
    j = gpd.sjoin(a, a, predicate="intersects")
    out = defaultdict(set)
    for x, y in zip(j.r_left, j.r_right):
        if x != y:
            out[x].add(y)
    return out


def load_years():
    files = {
        2024: os.path.join(RAW, "op825_wb20250205.xls"),
        2025: os.path.join(RAW, "op825_wb20250525.xls"),
        2026: os.path.join(RAW, "op825_2026.xls"),
    }
    return {y: nsc.parse(f) for y, f in files.items()}


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def geonames_points():
    import zipfile
    import geopandas as gpd
    cols = ["gid", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2",
            "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
    z = zipfile.ZipFile(os.path.join(RAW, "geonames_KG.zip"))
    g = pd.read_csv(z.open("KG.txt"), sep="\t", header=None, names=cols, dtype=str, quoting=3)
    g = g[g.fclass == "P"]
    rows = []
    for r in g.itertuples():
        names = {r.name} | set(str(r.alt).split(",")) if pd.notna(r.alt) else {r.name}
        for n in names:
            if re.search("[а-яА-Я]", n):
                nn = norm(n)
                rows.append((nn, stem(nn), latkey(nn), float(r.lon), float(r.lat)))
            elif n and n != "nan":
                rows.append(("", "", latkey(n), float(r.lon), float(r.lat)))
    p = pd.DataFrame(rows, columns=["nn", "st", "lat", "x", "y"]).drop_duplicates()
    p = p[p.lat.str.len() >= 3]
    return gpd.GeoDataFrame(p, geometry=gpd.points_from_xy(p.x, p.y), crs="EPSG:4326")


# Old units the village names cannot place, decided by hand (see README).
MANUAL_OLD = {}
# 2026 units with no ground found at all, drawn together with a neighbour.
MANUAL_MERGE = {
    # Jany-Alai (split off before 2024; neither of its two villages is in
    # GeoNames or OSM) is drawn with Pamir-Alai, the other Alay-valley unit
    # made from the same old Taldy-Suu aiyl aimak.
    "41706207830000": "41706207835000",
}
# Atoms given to a unit by hand.
MANUAL_CLAIM = {
    # COD's Kara-Kul city polygon is only its Ketmen-Tebe pgt (1 km2); the town
    # itself lies in this unnamed Toktogul-rayon unit (GeoNames Kara-Kul',
    # Kontur ~25,000 people there).
    "KG03225000911": "41703440000010",
}


def build(d, atoms, g1, g2, log=print):
    import geopandas as gpd
    t24, t25, t26 = d[2024], d[2025], d[2026]
    nb = rayon_neighbours(g1, g2)
    p24, p26 = places(t24, 2024), places(t26, 2026)
    m = match_places(p24, p26, nb)
    m["w"] = m["pop"].fillna(0) + 1

    cap = [nsc.BISHKEK, nsc.OSH]
    old = t24[t24.kind.isin(UNIT_KINDS) | t24.code.isin(cap)].copy()
    new = t26[t26.kind.isin(["aa", "town", "city"]) | t26.code.isin(cap)].copy()
    new_kind = new.set_index("code").kind
    new_rayon = new.set_index("code").rayon

    # --- old unit -> new unit -------------------------------------------------
    o2n, how = {}, {}
    for r in old.itertuples():
        if r.code in cap or (r.kind in ("city", "town") and new_kind.get(r.code) in ("city", "town")):
            o2n[r.code], how[r.code] = r.code, "same code"
            continue
        allv = m[m.unit == r.code]
        g = allv[allv.unit26.notna()]
        # a unit whose villages mostly vanished went into a city or town that
        # lists no villages; a stray namesake must not decide it
        if len(g):
            c = g.groupby("unit26").w.sum().sort_values(ascending=False)
            # half the people found, or the unit kept its own code
            if g.w.sum() >= 0.5 * allv.w.sum() or (c.index[0] == r.code and len(c) == 1):
                o2n[r.code] = c.index[0]
                how[r.code] = f"villages {c.iloc[0] / c.sum():.0%} of {g.w.sum() / allv.w.sum():.0%} found"
    for k, v in MANUAL_OLD.items():
        o2n[k], how[k] = v, "manual"

    # --- atom -> old unit -------------------------------------------------------
    alias = {"41703207812000": "41703207600010"}   # COD's Bazar-Korgon aiyl aimak is the 2024 town
    atoms = atoms.copy()
    atoms["old"] = atoms.soate.map(lambda s: alias.get(s, s))
    atoms.loc[~atoms.old.isin(set(old.code)), "old"] = None
    atoms.loc[atoms.kind == "land", "old"] = None
    log("atoms without a 2024 unit (not land):",
        atoms[atoms.old.isna() & (atoms.kind != "land")][["atom", "kind", "name"]].values.tolist())
    log("2024 units with no atom:",
        old[~old.code.isin(set(atoms.old.dropna()))][["code", "name", "pop"]].values.tolist())
    atoms["unit26"] = atoms.old.map(o2n)
    atoms["how"] = atoms.old.map(how)
    a2u = atoms.set_index("atom").unit26.dropna().to_dict()
    names26 = new.set_index("code").name

    def assign(atom, unit, why):
        a2u[atom] = unit
        atoms.loc[atoms.atom == atom, ["unit26", "how"]] = [unit, why]
        o = atoms.loc[atoms.atom == atom, "old"].iloc[0]
        if o and o not in o2n:
            o2n[o], how[o] = unit, why

    # --- 2026 units with no atom yet: find their villages on the ground ---------
    # (GeoNames points). If they sit in an atom nobody claimed, that atom is the
    # unit's; if the atom is already another unit's, the two units are merged,
    # since one polygon cannot be split between them.
    dsu = DSU()
    gn = geonames_points()
    a_ray = atoms.unit26.map(lambda x: new_rayon.get(x) if isinstance(x, str) else None)
    groundless = [u for u in new.code if u not in set(a2u.values())]
    placed = {}
    for u in groundless:
        # only the ground of the unit's own rayon as COD drew it
        region = atoms[atoms.cod_adm2 == "KG" + new_rayon.get(u)[3:14]]
        if not len(region):
            log(f"  (no COD rayon for {u})")
            placed[u] = Counter()
            continue
        pts = gn[gn.within(region.union_all().buffer(0.005))]
        pool = pts.nn.unique().tolist()
        cand = Counter()
        for v in p26[p26.unit == u].itertuples():
            hit = pts[(pts.nn == v.nn) | (pts.st == v.st) | (pts.lat == v.lat)]
            if not len(hit):
                close = difflib.get_close_matches(v.nn, pool, n=1, cutoff=0.85)
                hit = pts[pts.nn.isin(close)]
            if not len(hit):
                continue
            j = gpd.sjoin(hit, region[["atom", "geometry"]], predicate="within")
            if len(j):
                cand[j.atom.value_counts().index[0]] += (0 if pd.isna(v.pop) else v.pop) + 1
        placed[u] = cand
    # Unclaimed atoms first, biggest unit first.
    kind = atoms.set_index("atom").kind
    claimable = lambda a: a not in a2u  # noqa: E731
    for u in sorted(groundless, key=lambda u: -sum(placed[u].values())):
        free = [a for a, _ in placed[u].most_common() if claimable(a)]
        if free:
            assign(free[0], u, "village points (GeoNames)")
            log(f"  ground: {free[0]} ({kind[free[0]]}) -> {u} {names26.get(u)} (its villages' points)")
    for u in groundless:
        if u in set(a2u.values()) or not placed[u]:
            continue
        taken = [a for a, _ in placed[u].most_common() if a in a2u]
        if not taken:
            continue
        top = taken[0]
        dsu.union(a2u[top], u)
        log(f"  merged: {u} {names26.get(u)} has its villages in {top}, already "
            f"{a2u[top]} {names26.get(a2u[top])}; the two are drawn as one")

    for u, v in MANUAL_MERGE.items():
        if u not in set(a2u.values()):
            dsu.union(v, u)
            log(f"  merged by hand: {u} {names26.get(u)} drawn with {v} {names26.get(v)}")

    # --- COD's unnamed 9xx units are not all empty land: some hold villages ---
    # (Osh's village belt, Nookat town). Such a unit goes to the unit whose
    # villages stand in it, looked for in its own rayon and the ones bordering.
    # A match must be a village (not a town's whole figure), in the same oblast,
    # and its unit must already have ground within 10 km: common names
    # (Kara-Bulak, Kyzyl-Tuu, Ak-Suu) recur everywhere.
    for a, u in MANUAL_CLAIM.items():
        assign(a, u, "by hand (see README)")
        log(f"  claimed by hand: {a} -> {u} {names26.get(u)}")
    land = atoms[(atoms.kind == "land") & ~atoms.atom.isin(set(a2u))]
    pl = pd.concat([p26.assign(u=p26.unit), p24.assign(u=p24.unit.map(o2n))])
    pl = pl[pl.u.notna()]
    # a town's or city's own row carries its whole population; count it as a
    # village-sized hit, enough to place the town but not to swamp the vote
    tc = pl.kind.isin(["town", "city"])
    pl.loc[tc, "pop"] = pl.loc[tc, "pop"].clip(upper=3000)
    pl = pl[pl.lat.str.len() >= 5]
    proj = atoms.to_crs("ESRI:54009").set_index("atom").geometry
    ground = pd.Series(a2u).reset_index().rename(columns={"index": "atom", 0: "u"})
    ground["geometry"] = ground.atom.map(proj)
    ground = gpd.GeoDataFrame(ground, geometry="geometry", crs="ESRI:54009").dissolve("u").geometry
    oblast_of = lambda u: {"11": "08", "21": "06"}.get(u[3:5], u[3:5])  # noqa: E731
    cities = set(new[new.kind == "city"].code) | set(cap)
    jp = gpd.sjoin(gn, land[["atom", "cod_adm2", "geometry"]], predicate="within")
    for a, g in jp.groupby("atom"):
        ob = land.set_index("atom").cod_adm2[a][2:4]
        here = proj[a]
        # cities keep villages across oblast lines (Kyzyl-Kiya's exclave in Nookat)
        # rayon land stays in its rayon; only a city may reach across
        ray = "417" + land.set_index("atom").cod_adm2[a][2:7] + "000000"
        cands = pl[(pl.u.map(lambda u: new_rayon.get(u)) == ray) | pl.u.isin(cities)]
        # a town name inside a vast mountain unit is a river or a pass, not the town
        if here.area > 200e6:
            cands = cands[~cands.kind.isin(["town", "city"])]
        c = Counter()
        seen = set()
        for q in g.itertuples():
            h = cands[((cands.nn == q.nn) & (q.nn != "")) | (cands.lat == q.lat)]
            for x in h.itertuples():
                if (x.code, x.year) in seen or x.u not in ground.index:
                    continue
                if ground[x.u].distance(here) > 10_000:
                    continue
                seen.add((x.code, x.year))
                c[x.u] += 0 if pd.isna(x.pop) else x.pop
        top = c.most_common(2)
        if top and top[0][1] >= 300 and (len(top) == 1 or top[0][1] >= 2 * top[1][1]):
            u, w = top[0]
            assign(a, u, "unnamed unit holding its villages (GeoNames)")
            log(f"  land with villages: {a} -> {u} {names26.get(u)} ({w:.0f} people matched; "
                f"{', '.join(g.lat.unique()[:6])})")

    # --- atoms still unclaimed went into a bordering unit that grew by more ---
    # than it should have. Growth is judged on the unraked figures, which come
    # from the same aiyl okmotu registers in both years.
    raw = {y: d[y].set_index("code")["pop"].fillna(0).to_dict() for y in d}
    r24, r25 = dict(raw[2024]), raw[2025]
    for r in old[old.kind == "city_pgt"].itertuples():     # cities include their pgt
        city = r.code[:8] + "000010"
        r24[city] = r24.get(city, 0) - r24.get(r.code, 0)
    recode25 = {"41703215610020": "41703215600020", "41706207809000": "41706207600010",
                "41708213600020": "41708213610020"}
    r25 = {recode25.get(k, k): v for k, v in r25.items()}
    got = defaultdict(float)
    for o, n in o2n.items():
        got[n] += r24.get(o, 0)
    deficit = {u: r25.get(u, 0) - 1.017 * got.get(u, 0) for u in new.code}
    buf = atoms[["atom", "geometry"]].to_crs("ESRI:54009")
    buf["geometry"] = buf.buffer(300)
    nbr = gpd.sjoin(buf, atoms[["atom", "geometry"]].to_crs("ESRI:54009"), predicate="intersects")
    nbr = nbr[nbr.atom_left != nbr.atom_right].groupby("atom_left").atom_right.agg(set)
    # A pgt that no name placed elsewhere stays with its city (the pgt rank was
    # abolished in 2024-25; the place stayed in the city's territory).
    for r in atoms[~atoms.atom.isin(set(a2u)) & (atoms.kind == "town")
                   & atoms.cod_adm2.str[4].eq("4")].itertuples():
        city = "417" + r.cod_adm2[2:]
        p = r24.get(r.old, 0) if isinstance(r.old, str) else 0
        if deficit.get(city, 0) > 0.5 * p:
            deficit[city] -= 1.017 * p
            assign(r.atom, city, "pgt stays with its city")
            log(f"  pgt: {r.atom} {r.name} ({p:.0f}) -> its city {city} {names26.get(city)}")
    todo = atoms[~atoms.atom.isin(set(a2u)) & (atoms.kind != "land")].copy()
    todo["p"] = todo.old.map(lambda o: r24.get(o, 0) if isinstance(o, str) else 0)
    todo = todo.sort_values("p", ascending=False)
    for _ in range(6):
        left = []
        for r in todo.itertuples():
            cands = {dsu.find(a2u[x]) for x in nbr.get(r.atom, set()) if x in a2u}
            if not cands:
                left.append(r)
                continue
            # the bordering unit whose shortfall this atom closes best
            gain = 1.017 * r.p
            best = min(cands, key=lambda u: abs(deficit.get(u, 0) - gain) - abs(deficit.get(u, 0)))
            deficit[best] = deficit.get(best, 0) - gain
            assign(r.atom, best, "absorbed: border and growth")
            log(f"  absorbed: {r.atom} {r.name} ({r.p:.0f} in 2024) -> {best} {names26.get(best)}")
        todo = pd.DataFrame(left)
        if not len(todo):
            break
    # The greedy pass takes atoms biggest first, so an early one can land in a
    # city that a later one fitted better. Move an absorbed atom to another
    # bordering unit while that shrinks the two units' shortfalls together.
    p_of = {r.atom: (r24.get(r.old, 0) if isinstance(r.old, str) else 0) for r in atoms.itertuples()}
    for _ in range(10):
        moved = 0
        for r in atoms[atoms.how == "absorbed: border and growth"].itertuples():
            cur = a2u[r.atom]
            gain = 1.017 * p_of[r.atom]
            for u in {dsu.find(a2u[x]) for x in nbr.get(r.atom, set()) if x in a2u} - {cur}:
                before = abs(deficit.get(cur, 0)) + abs(deficit.get(u, 0))
                after = abs(deficit.get(cur, 0) + gain) + abs(deficit.get(u, 0) - gain)
                if after < before - 500:
                    deficit[cur] = deficit.get(cur, 0) + gain
                    deficit[u] = deficit.get(u, 0) - gain
                    assign(r.atom, u, "absorbed: border and growth")
                    o2n[r.old] = u
                    log(f"  moved: {r.atom} {r.name} {cur} -> {u} {names26.get(u)}")
                    moved += 1
                    break
        if not moved:
            break
    for r in old.itertuples():
        if r.code not in o2n:
            log(f"  2024 unit not carried: {r.code} {r.name} {r.pop}")
    for u in new.code:
        if u not in set(a2u.values()) and dsu.find(u) == u:
            log(f"  NO GROUND: {u} {names26.get(u)}")
    raked = {y: rake(d[y]) for y in d}
    return atoms, o2n, m, raked, deficit, dsu


def rake(t):
    """Unit figures scaled so each rayon's units sum to the NSC rayon total.
    Returns {code: pop} for aa/town/city/capital rows (and city_pgt, unscaled)."""
    out = {}
    for r in t[t.kind.isin(["city", "city_pgt"]) | t.code.isin([nsc.BISHKEK, nsc.OSH])].itertuples():
        out[r.code] = float(r.pop)
    for ry, g in t[t.kind.isin(["aa", "town"])].groupby("rayon"):
        tot = t[(t.code == ry) & (t.kind == "rayon")]["pop"]
        s = g["pop"].fillna(0).sum()
        f = float(tot.iloc[0]) / s if len(tot) and s else 1.0
        for r in g.itertuples():
            out[r.code] = (0 if pd.isna(r.pop) else float(r.pop)) * f
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 3000)
    from atoms import build_atoms, load_cod
    d = load_years()
    g1, g2, g3 = load_cod()
    atoms = build_atoms()
    atoms, o2n, m, raked, deficit, dsu = build(d, atoms, g1, g2)
    print(m.how.str.split(":").str[0].value_counts())
    newnames = d[2026].set_index("code").name
    print("unresolved atoms:", atoms[atoms.unit26.isna() & (atoms.kind != "land")][["atom", "name"]].values.tolist())
    units = d[2026][d[2026].kind.isin(["aa", "town", "city"]) | d[2026].code.isin([nsc.BISHKEK, nsc.OSH])]
    noatom = units[~units.code.isin(set(atoms.unit26.dropna()))]
    print("2026 units with no atom:", noatom[["code", "name", "pop"]].values.tolist())
    big = sorted(deficit.items(), key=lambda kv: -abs(kv[1]))[:40]
    oldnames = d[2024].set_index("code").name
    oldpop = d[2024].set_index("code")["pop"]
    for u, v in big:
        print(f"  deficit {u} {newnames.get(u)} {v:.0f} of {raked[2026].get(u, 0):.0f}")
        if abs(v) > 3000:
            for o, n in o2n.items():
                if n == u:
                    print(f"        <- {o} {oldnames.get(o)} {oldpop.get(o)}")
    m.to_csv(os.path.join(HELPER, "data", "kyrgyzstan", "match_places.csv"), index=False, encoding="utf-8")
    # unit-level summary
    m["w"] = m["pop"].fillna(0) + 1
    rows = []
    for u, g in m.groupby("unit"):
        tot = g.w.sum()
        hit = g[g.unit26.notna()]
        c = hit.groupby("unit26").w.sum().sort_values(ascending=False)
        rows.append(dict(unit=u, n=len(g), matched=len(hit), share_matched=hit.w.sum() / tot,
                         main=c.index[0] if len(c) else None,
                         main_share=(c.iloc[0] / hit.w.sum()) if len(c) else 0,
                         others=dict(c.iloc[1:]) if len(c) > 1 else {}))
    s = pd.DataFrame(rows)
    names = d[2024].set_index("code").name
    s["name"] = s.unit.map(names)
    s.to_csv(os.path.join(HELPER, "data", "kyrgyzstan", "match_units.csv"), index=False, encoding="utf-8")
    print(s[(s.main_share < 0.97) | (s.share_matched < 0.5)].to_string())
