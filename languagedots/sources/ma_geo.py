"""Morocco: religiondots' Kontur hexes re-keyed from provinces to communes -> data/geo/ma/.

    python sources/ma_geo.py [--fetch]

WHY. religiondots draws Morocco on 73 units (69 provinces and 4 Western Sahara units, its
sources/ma_geo.py), with Kontur 2023-11 hexes it has already cut to Natural Earth's B19 (west of
the berm), cleared of Ceuta and Melilla, and given discs where Kontur lost Smara, Tan-Tan and Assa.
The language table is per commune (1,538 leaves). Language varies inside provinces far more than
religion does (Taza runs from Tarifit in the north to Darija in the south), so the hexes are
re-keyed to communes, keeping every hex's population and every hex inside its province.

BOUNDARIES. HCP publishes no commune layer and COD-AB stops at provinces. OSM does: Kontur
Boundaries MA 2023-06-28 (HDX `kontur-boundaries-morocco`, ODbL, OSM admin_level 8 = Kontur
level 9, 1,531 polygons, including the arrondissements of Tanger, Fès and Marrakech). Downloaded
with --fetch to data/raw/ma/.

THE JOIN, by name inside each religiondots unit (never across units: Morocco repeats commune
names, Oulad Hcine twice in neighbouring provinces):
  1. each OSM polygon goes to the religiondots unit holding its representative point (the three
     in the Tarfaya strip, outside every COD polygon, to the nearest, which is Laâyoune and
     Tarfaya, where religiondots puts that strip's hexes)
  2. names folded (Latin part only, accents, "Commune de", spacing); a folded name unique on both
     sides within the unit pairs exactly
  3. the rest by spelling rules (Ouled/Oulad, My/Moulay, Sid/Sidi, Abdellah/Abdallah, ...) and then
     by similarity, best pair first, at difflib ratio 0.72 or more; every such pair is printed
  4. WITNESS (neither name decides it): Kontur's own population per OSM polygon against the
     census population per commune, log correlation against 500 within-unit shuffles, and the
     fuzzy pairs' ratios printed
  5. a city whose arrondissements do not all pair is drawn as its commune
Every hex goes to the smallest paired polygon (within its religiondots unit) holding its centroid.
Hexes in no paired polygon, and HCP communes that paired with nothing, form one remainder unit per
religiondots unit (`<unit>-rest`): the unpaired communes' people on the unpaired ground. A paired
commune that gets no hex (its polygon holds no populated hex centroid), or a remainder with no
hex, is drawn with the paired commune whose hexes lie nearest (printed, with its population).

Writes data/geo/ma/ma_hexes.gpkg (unit, pop) and data/geo/ma/ma_lookup.csv (geo_id -> unit, how).
"""
import argparse
import difflib
import gzip
import math
import os
import random
import re
import shutil
import sys
import unicodedata
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "ma"
KB = RAW / "kontur_boundaries_MA_20230628.gpkg"
KB_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_boundaries_MA_20230628.gpkg.gz")
OUT = HERE / "data" / "geo" / "ma"
NORM = HERE / "data" / "normalized" / "ma.csv"
MAHBASS = "100710501"           # religiondots moves this Assa-Zag commune to Es-Semara (EH02)
N_OSM = 1531
RATIO = 0.72
# OSM polygons in the Tarfaya strip, which no COD polygon covers; religiondots gives the strip's
# hexes to Laâyoune and Tarfaya (EH03), so these polygons go there too
STRIP = {"tarfaya", "akhfennir", "tah"}
# one OSM polygon carries only an Arabic name: Yacoub El Mansour, Rabat
AR_ALIAS = {"يعقوب المنصور": "yacoubelmansour"}
# (religiondots unit, HCP folded name) -> OSM folded name, where the two spell a place differently
# beyond what the spelling rules catch; each checked by Kontur's population (printed)
ALIAS = {("MA002005", "asoukhourassawda"): "desrochesnoires",     # Assoukhour Assawda = Roches Noires
         ("MA010006", "souani"): "charfsouani",                   # Tanger's arrondissements
         ("MA010006", "medina"): "tangermedina"}

SPELL = [(r"\boule?d\b|\boulad\b|\bouled\b", "ould"), (r"\bmy\b", "moulay"), (r"\bsidl\b", "sidi el"),
         (r"\bsid\b", "sidi"), (r"abdellah", "abdallah"), (r"\bbeni\b", "bni"), (r"\b(el|al|l)\b", ""),
         (r"ou?ad\b", "oued"), (r"kh", "k"), (r"gh|rh", "g"), (r"q", "k"), (r"ou", "u"), (r"y", "i"),
         (r"e", "a"), (r"(.)\1", r"\1")]


def fold(s):
    name0 = s
    s = re.sub(r"[^\x00-ɏ' -]", " ", str(s))           # drop Arabic and Tifinagh
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("*", "")
    s = re.sub(r"^(commune|arrondissement|pachalik|pashalik)( de | d'| d | )", "", s.strip())
    s = re.sub(r"[^a-z]", "", s)
    return AR_ALIAS.get(name_ar(name0), s) if not s else s


def name_ar(s):
    return re.sub(r"[^؀-ۿ ]", "", str(s)).strip()


def loose(s):
    s = re.sub(r"[^\x00-ɏ' -]", " ", str(s))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("*", "")
    s = re.sub(r"^(commune|arrondissement|pachalik|pashalik)( de | d'| d | )", "", s.strip())
    s = re.sub(r"[^a-z]+", " ", s).strip()
    for a, b in SPELL:
        s = re.sub(a, b, s)
    return re.sub(r"[^a-z]", "", s)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    gz = KB.with_suffix(".gpkg.gz")
    urllib.request.urlretrieve(KB_URL, gz)
    with gzip.open(gz, "rb") as fi, open(KB, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    print(f"  fetched {KB_URL}")


def leaves():
    df = pd.read_csv(NORM, dtype={"geo_id": str, "province": str})
    lv = df.drop_duplicates("geo_id")[["geo_id", "geo_level", "geo_name", "province", "pop_municipale"]].copy()
    pieces = pd.read_csv(RD_GEO / "ma" / "ma_pieces.csv", dtype={"piece": str})
    pmap = dict(zip(pieces["piece"], pieces["geo_id"]))
    lv["rd"] = lv["province"].map(pmap)
    lv.loc[lv["geo_id"] == MAHBASS, "rd"] = pmap[MAHBASS]
    if lv["rd"].isna().any():
        raise SystemExit(f"leaves with no religiondots unit: {lv.loc[lv['rd'].isna(), 'geo_name'].tolist()}")
    # the city a leaf belongs to: arrondissement codes are <province>01<nn>, the commune <province>010
    lv["city"] = np.where(lv["geo_level"] == "arrondissement", lv["geo_id"].str[:-2] + "0", lv["geo_id"])
    return lv


def pair(h, o, u):
    """h, o: DataFrames of one unit's HCP leaves and OSM polygons -> {geo_id: osm index}, fuzzy list."""
    out, fuzzy = {}, []
    hf, of = h["geo_name"].map(lambda n: ALIAS.get((u, fold(n)), fold(n))), o["name"].map(fold)
    for key in (fold, loose):
        hk = hf if key is fold else h["geo_name"].map(key)
        ok_ = o["name"].map(key)
        free_h = [i for i in h.index if h.at[i, "geo_id"] not in out]
        used = set(out.values())
        free_o = [i for i in o.index if i not in used]
        hc = pd.Series([hk[i] for i in free_h]).value_counts()
        oc = pd.Series([ok_[i] for i in free_o]).value_counts()
        for i in free_h:
            k = hk[i]
            if hc.get(k, 0) == 1 and oc.get(k, 0) == 1:
                j = next(j for j in free_o if ok_[j] == k)
                out[h.at[i, "geo_id"]] = j
                if key is loose:
                    fuzzy.append((h.at[i, "geo_name"], o.at[j, "name"], 1.0, h.at[i, "geo_id"]))
    # similarity, best pair first
    free_h = [i for i in h.index if h.at[i, "geo_id"] not in out]
    used = set(out.values())
    free_o = [i for i in o.index if i not in used]
    cand = []
    for i in free_h:
        for j in free_o:
            r = max(difflib.SequenceMatcher(None, hf[i], of[j]).ratio(),
                    difflib.SequenceMatcher(None, loose(h.at[i, "geo_name"]), loose(o.at[j, "name"])).ratio())
            if r >= RATIO:
                cand.append((r, i, j))
    taken_h, taken_o = set(), set()
    for r, i, j in sorted(cand, reverse=True):
        if i in taken_h or j in taken_o:
            continue
        taken_h.add(i)
        taken_o.add(j)
        out[h.at[i, "geo_id"]] = j
        fuzzy.append((h.at[i, "geo_name"], o.at[j, "name"], r, h.at[i, "geo_id"]))
    return out, fuzzy


def pear(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    if ap.parse_args().fetch or not KB.exists():
        fetch()
    lv = leaves()
    units = gpd.read_file(RD_GEO / "ma" / "ma_units.gpkg")
    kb = gpd.read_file(KB)
    osm = kb[kb["admin_level"] == 9].copy()
    if len(osm) != N_OSM:
        raise SystemExit(f"Kontur boundaries: {len(osm)} level-9 polygons, expected {N_OSM}")
    # six urban municipalities are mapped in OSM as a pachalik (Kontur level 7), not a commune;
    # they are offered only to leaves no commune polygon paired with
    pach = kb[(kb["admin_level"] == 7) & kb["name"].str.contains(r"(?i)^pa(?:c)?hs?alik|^pashalik")]
    osm["pach"] = False
    osm = pd.concat([osm, pach.assign(pach=True)]).reset_index(drop=True)
    osm = gpd.GeoDataFrame(osm, geometry="geometry", crs=kb.crs)
    print(f"  OSM: {N_OSM} communes and {len(pach)} pachaliks ({', '.join(pach['name'].map(fold))})")
    rp = osm[["geometry"]].copy()
    rp["geometry"] = osm.representative_point()
    j = gpd.sjoin(rp, units[["geo_id", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    osm["rd"] = j["geo_id"]
    osm.loc[osm["name"].map(fold).isin(STRIP) & ~osm["pach"], "rd"] = "EH03"
    miss = osm["rd"].isna()
    if miss.any():
        nn = gpd.sjoin_nearest(rp[miss].to_crs(3857), units[["geo_id", "geometry"]].to_crs(3857), how="left")
        nn = nn[~nn.index.duplicated()]
        osm.loc[miss, "rd"] = nn["geo_id"]
        print(f"  {int(miss.sum())} OSM polygons outside every COD unit, to the nearest: "
              + ", ".join(f"{fold(n)} -> {u}" for n, u in zip(osm.loc[miss, 'name'], osm.loc[miss, 'rd'])))

    # ---- the join ----
    paired, fuzzy = {}, []
    for u in sorted(lv["rd"].unique()):
        h = lv[lv["rd"] == u]
        o = osm[(osm["rd"] == u) & ~osm["pach"]]
        p, f = pair(h, o, u)
        paired.update(p)
        fuzzy += [(u,) + x for x in f]
        h2 = h[~h["geo_id"].isin(p)]
        o2 = osm[(osm["rd"] == u) & osm["pach"]]
        if len(h2) and len(o2):
            p, f = pair(h2, o2, u)
            paired.update(p)
            fuzzy += [(u,) + x for x in f]
            for gid, jx in p.items():
                print(f"  pachalik {fold(osm.at[jx, 'name'])} pairs with {gid}")
    # a city drawn by arrondissement only if all of them paired
    cities = lv[lv["geo_level"] == "arrondissement"].groupby("city")["geo_id"].apply(list)
    city_whole = {}
    for city, ids in cities.items():
        if not all(i in paired for i in ids):
            print(f"  city {city}: {sum(i in paired for i in ids)} of {len(ids)} arrondissements "
                  "paired, so it is drawn as one commune")
            city_whole[city] = ids
    print(f"  paired {len(paired):,} of {len(lv):,} leaves to {len(set(paired.values())):,} of "
          f"{len(osm):,} OSM polygons; {len(fuzzy)} by spelling or similarity:")
    for u, a, b, r, gid in sorted(fuzzy, key=lambda x: x[3]):
        print(f"      {u:9s} {r:4.2f}  {fold(a):28s} ~ {fold(b)}")
    unp_h = lv[~lv["geo_id"].isin(paired)]
    used = set(paired.values())
    unp_o = osm[~osm.index.isin(used) & ~osm["pach"]]
    print(f"  unpaired HCP leaves {len(unp_h)} ({int(unp_h['pop_municipale'].sum()):,} people); "
          f"unpaired OSM polygons {len(unp_o)}")
    for u in sorted(set(unp_h["rd"]) | set(unp_o["rd"])):
        a = [fold(x) for x in unp_h.loc[unp_h["rd"] == u, "geo_name"]]
        b = [fold(x) for x in unp_o.loc[unp_o["rd"] == u, "name"]]
        print(f"      {u:9s} HCP {a}  OSM {b}")

    # leaf -> drawn unit (before the hex step)
    unit_of, how = {}, {}
    for _, r in lv.iterrows():
        gid = r["geo_id"]
        if r["city"] in city_whole and r["geo_level"] == "arrondissement":
            unit_of[gid], how[gid] = None, "city"        # filled below
        elif gid in paired:
            unit_of[gid], how[gid] = f"c{gid}", "name"
        else:
            unit_of[gid], how[gid] = f"{r['rd']}-rest", "rest"
    fuzzy_ids = {x[4] for x in fuzzy}
    for gid in fuzzy_ids:
        how[gid] = "spelling"
    # a whole city: its polygon is the OSM polygon that pairs with the commune's own name, else
    # the union of its paired arrondissements' polygons
    poly_of = {f"c{g}": osm.geometry[i] for g, i in paired.items()}
    for city, ids in city_whole.items():
        name = lv.loc[lv["geo_id"] == city, "geo_name"]
        tgt = f"city{city}"
        geoms = [osm.geometry[paired[i]] for i in ids if i in paired]
        cname = fold(lv.loc[lv["geo_id"] == ids[0], "geo_name"].iloc[0])
        rd = lv.loc[lv["geo_id"] == ids[0], "rd"].iloc[0]
        whole = osm[(osm["rd"] == rd) & (osm["name"].map(fold) == fold(name.iloc[0]) if len(name) else False)]
        for i in ids:
            unit_of[i] = tgt
            poly_of.pop(f"c{i}", None)
        poly_of[tgt] = gpd.GeoSeries(geoms).union_all() if geoms else None
        print(f"  {tgt}: {len(ids)} arrondissements, polygon from {len(geoms)} paired ones ({cname})")

    # ---- hexes ----
    hexes = gpd.read_file(RD_GEO / "ma" / "ma_hexes.gpkg")
    n0, p0 = len(hexes), float(hexes["pop"].sum())
    cent = gpd.GeoDataFrame({"rd": hexes["unit"].to_numpy()}, geometry=hexes.to_crs(3857).geometry.centroid, crs=3857)
    polys = gpd.GeoDataFrame({"unit": list(poly_of)}, geometry=list(poly_of.values()), crs=osm.crs).to_crs(3857)
    rd_of_unit = {}
    for gid, u in unit_of.items():
        rd_of_unit[u] = lv.loc[lv["geo_id"] == gid, "rd"].iloc[0]
    polys["rd"] = polys["unit"].map(rd_of_unit)
    polys["area"] = polys.geometry.area
    sj = gpd.sjoin(cent, polys, how="left", predicate="within")
    sj = sj[sj["rd_left"] == sj["rd_right"]].sort_values("area")
    sj = sj[~sj.index.duplicated(keep="first")]
    hexes["cu"] = None
    hexes.loc[sj.index, "cu"] = sj["unit"]
    rest_units = {u for u in unit_of.values() if u.endswith("-rest")}
    nohex = hexes["cu"].isna()
    hexes.loc[nohex, "cu"] = hexes.loc[nohex, "unit"] + "-rest"
    print(f"  hexes: {int((~nohex).sum()):,} in a paired polygon, {int(nohex.sum()):,} "
          f"({hexes.loc[nohex, 'pop'].sum():,.0f} people) in none")

    # remainder hexes in a unit with no unpaired commune: to the paired polygon nearest
    orphan = hexes["cu"].str.endswith("-rest") & ~hexes["cu"].isin(rest_units)
    if orphan.any():
        oc = cent[orphan.to_numpy()]
        for rd, idx in oc.groupby("rd").groups.items():
            pp = polys[polys["rd"] == rd]
            nn = gpd.sjoin_nearest(oc.loc[idx], pp[["unit", "geometry"]], how="left")
            nn = nn[~nn.index.duplicated()]
            hexes.loc[nn.index, "cu"] = nn["unit"]
        print(f"  {int(orphan.sum()):,} hexes ({hexes.loc[orphan, 'pop'].sum():,.0f} people) in no paired "
              "polygon of a fully paired unit, to the nearest paired polygon")

    # drawn units with no hex: draw with the nearest unit in the same religiondots unit
    have = set(hexes["cu"])
    merged = []
    for _ in range(3):
        empty = sorted({u for u in unit_of.values() if u not in have})
        if not empty:
            break
        for u in empty:
            rd = rd_of_unit[u]
            geom = poly_of.get(u)
            cand = hexes[(hexes["unit"] == rd)]
            if geom is None or cand.empty:
                tgt = cand.groupby("cu")["pop"].sum().idxmax()
            else:
                g = gpd.GeoSeries([geom], crs=osm.crs).to_crs(3857).iloc[0]
                d = cent.loc[cand.index].distance(g)
                tgt = hexes.at[d.idxmin(), "cu"]
            ids = [k for k, v in unit_of.items() if v == u]
            for k in ids:
                unit_of[k] = tgt
                how[k] = how[k] + "+merged"
            merged.append((u, tgt, int(lv[lv["geo_id"].isin(ids)]["pop_municipale"].sum())))
    print(f"  {len(merged)} drawn units with no hex, drawn with the nearest: "
          f"{sum(m[2] for m in merged):,} people; the five largest: "
          + ", ".join(f"{a}->{b} {c:,}" for a, b, c in sorted(merged, key=lambda x: -x[2])[:5]))

    out = gpd.GeoDataFrame({"unit": hexes["cu"].to_numpy(), "pop": hexes["pop"].to_numpy()},
                           geometry=hexes.geometry.to_numpy(), crs=hexes.crs)
    assert len(out) == n0 and abs(out["pop"].sum() - p0) < 1
    lv["unit"] = lv["geo_id"].map(unit_of)
    lv["how"] = lv["geo_id"].map(how)
    if set(lv["unit"]) - set(out["unit"]):
        raise SystemExit(f"units with no hex: {sorted(set(lv['unit']) - set(out['unit']))[:10]}")
    if set(out["unit"]) - set(lv["unit"]):
        raise SystemExit(f"hex units with no leaf: {sorted(set(out['unit']) - set(lv['unit']))[:10]}")
    # every hex stayed in its religiondots unit
    chk = out.assign(rd=hexes["unit"].to_numpy())
    if (chk.groupby("unit")["rd"].nunique() > 1).any():
        raise SystemExit("a drawn unit spans two religiondots units")

    # ---- witness: Kontur against the census, per drawn unit ----
    per_k = out.groupby("unit")["pop"].sum()
    per_c = lv.groupby("unit")["pop_municipale"].sum()
    u = per_c.index[(per_c > 0)]
    ratio = per_k[u].sum() / per_c[u].sum()
    rr = (per_k[u] / per_c[u] / ratio)
    lc, lk = np.log(per_c[u].to_numpy()), np.log(per_k[u].to_numpy())
    r = pear(lc, lk)
    rd_u = pd.Series({x: rd_of_unit.get(x, x.split("-")[0]) for x in u})
    rng = random.Random(0)
    best = -1
    for _ in range(500):
        perm = per_k[u].copy()
        for rd, idx in rd_u.groupby(rd_u).groups.items():
            idx = list(idx)
            vals = list(perm[idx])
            rng.shuffle(vals)
            perm[idx] = vals
        best = max(best, pear(lc, np.log(perm.to_numpy())))
    print(f"  {len(u):,} drawn units; Kontur/census {ratio:.3f}; log r = {r:.3f} against a best of "
          f"{best:.3f} over 500 shuffles within religiondots units")
    print(f"  Kontur over census per unit: median {rr.median():.2f}; under 0.33: {int((rr < 1/3).sum())}, "
          f"over 3: {int((rr > 3).sum())}")
    for x in list(rr.sort_values().index[:6]) + list(rr.sort_values().index[-6:]):
        names = ", ".join(fold(n) for n in lv.loc[lv["unit"] == x, "geo_name"][:3])
        print(f"      {x:16s} {rr[x]:6.2f}  census {per_c[x]:>9,}  ({names})")
    fz = [x[4] for x in fuzzy if unit_of[x[4]].startswith("c")]
    fr = rr[[unit_of[g] for g in fz if unit_of[g] in rr.index]]
    print(f"  the {len(fr)} spelling/similarity pairs: median {fr.median():.2f}, under 0.33: "
          f"{int((fr < 1/3).sum())}, over 3: {int((fr > 3).sum())}"
          + "".join(f"\n      {x:16s} {fr[x]:6.2f}  census {per_c[x]:>9,}  "
                    f"({', '.join(fold(n) for n in lv.loc[lv['unit'] == x, 'geo_name'])})"
                    for x in fr.index if not (1 / 3 <= fr[x] <= 3)))
    if r <= best:
        raise SystemExit("the commune join is not carrying information")

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT / "ma_hexes.gpkg", layer="hexes", driver="GPKG")
    lv[["geo_id", "geo_name", "geo_level", "rd", "unit", "how", "pop_municipale"]].to_csv(
        OUT / "ma_lookup.csv", index=False, encoding="utf-8")
    print(f"  wrote {OUT / 'ma_hexes.gpkg'} ({len(out):,} hexes, {out['unit'].nunique():,} units) and ma_lookup.csv")
    print("  how:", lv["how"].value_counts().to_dict())


if __name__ == "__main__":
    main()
