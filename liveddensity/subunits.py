"""Sub-national units for build.py: admin-1 states/provinces and GHSL functional urban areas.

    python -I subunits.py        # rebuild logs/units.pkl and print a summary

Both come out in Mollweide with a parent ISO3 that matches the country keys in hist.json.
build.py loads them with load_units(); NOTES.md has the decisions.
"""
import collections
import json
import os
import pickle
import re
import sys
import unicodedata

import numpy as np
import pyogrio
import shapely
from pyproj import Transformer
from unidecode import unidecode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I leaves the script dir out
import build  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.dirname(HERE)
ADM1 = ("/vsizip/" + os.path.join(MAPS, "languagedots", "data", "raw", "ne_admin1",
        "ne_10m_admin_1_states_provinces.zip").replace("\\", "/")
        + "/ne_10m_admin_1_states_provinces.shp")
FUA = os.path.join(HERE, "raw", "fua", "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
UCDB = os.path.join(HERE, "raw", "ucdb", "GHS_UCDB_GLOBE_R2024A.gpkg")
CACHE = os.path.join(HERE, "logs", "units.pkl")
CRS = "ESRI:54009"

# Countries whose NE admin-1 layer is a second level (departments, provinces, municipalities):
# these are merged up to the first level, by NE's `region` field (or `geonunit` for the UK).
DISSOLVE = {"FRA": "region", "ITA": "region", "SVN": "region", "LVA": "region",
            "MLT": "region", "MKD": "region", "PHL": "region", "UGA": "region",
            "AZE": "region", "GBR": "geonunit"}
REGION_NAMES = {
    "Bretagne": "Brittany", "Normandie": "Normandy", "Corse": "Corsica",
    "Guyane française": "French Guiana", "Hauts-de-France": "Hauts-de-France",
    "Provence-Alpes-Côte-d'Azur": "Provence-Alpes-Côte d'Azur",
    "Piemonte": "Piedmont", "Lombardia": "Lombardy", "Toscana": "Tuscany", "Sicilia": "Sicily",
    "Sardegna": "Sardinia", "Puglia": "Apulia", "Valle d'Aosta": "Aosta Valley",
    "Trentino-Alto Adige": "Trentino-South Tyrol",
}


ADM1_NAME_FIX = {"RUS-2399": "Altai Krai", "RUS-2613": "Jewish Autonomous Oblast"}
# FUA country codes that are not country keys here
FUA_ISO = {"XKO": "XKX", "REU": "FRA", "GUF": "FRA", "GLP": "FRA", "MTQ": "FRA", "MYT": "FRA",
           "XNC": "CYP"}
# FUA names that the automatic romanisation gets wrong (by eFUA_ID)
FUA_NAME_FIX = {4398: "Chibayish", 2469: "Al Badai", 5275: "Izumi", 3262: "Ungsang",
                6352: "Longtai", 6877: "Anju", 8831: "Meicheng", 9345: "Huating",
                9943: "Heqiao", 10224: "Huaxing", 9491: "Bianzhuang", 9665: "Shishang",
                10138: "Yanguan", 226: "Nicosia", 187: "North Nicosia", 3707: "Narsingdi",
                8961: "Dengfeng",
                # R2019A splits the Pearl River Delta into FUAs all called "Guangzhou", and
                # names two Jakarta pieces "Jakarta"; named here by where they lie
                9711: "Guangzhou", 9725: "Shenzhen", 9721: "Shenzhen (Bao'an)", 9718: "Dongguan",
                9715: "Jiangmen", 4894: "Tangerang", 3703: "Gazipur", 6152: "Owerri",
                4818: "Mekarjaya"}


def is_latin(s):
    for ch in s:
        if ch.isalpha() and "LATIN" not in unicodedata.name(ch, ""):
            return False
    return True


def latin_part(s):
    """'Pirganj (পীরগঞ্জ)' -> 'Pirganj'; 'রামু Ramu' -> 'Ramu'; '' when nothing Latin is left."""
    toks = [t for t in re.split(r"[\s()\[\]]+", s) if t and is_latin(t) and any(c.isalpha() for c in t)]
    return " ".join(toks)


def to_moll():
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

    def proj(xy):
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])
    return proj


def polys(g):
    return [p for p in shapely.get_parts(shapely.make_valid(g))
            if p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty]


def admin0_info():
    """ADM0_A3 -> country key, and the lon/lat pieces moved by build.MOVE_PARTS."""
    with open(build.NE, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    a3key, moved = {}, []
    for feat in feats:
        p = feat["properties"]
        a3key[p["ADM0_A3"]] = build.country_key(p)
        k = a3key[p["ADM0_A3"]]
        for src, dst, (lon, lat), _ in build.MOVE_PARTS:
            if k == src:
                for piece in shapely.get_parts(shapely.geometry.shape(feat["geometry"])):
                    if piece.contains(shapely.Point(lon, lat)):
                        moved.append((piece.buffer(0.05), dst))
    return a3key, moved


def load_admin1(keys, names):
    a3key, moved = admin0_info()
    df = pyogrio.read_dataframe(ADM1)
    proj = to_moll()
    groups = collections.OrderedDict()   # unit id -> dict(name, parent, geoms)
    for r in df.itertuples():
        parent = a3key.get(r.adm0_a3, r.adm0_a3)
        if parent is None:
            continue
        g = shapely.make_valid(r.geometry)
        rp = g.representative_point()
        for piece, dst in moved:
            if piece.contains(rp):
                parent = dst
        pieces = []
        for src, dst, box, _ in build.CUTS:
            if parent == src and g.intersects(shapely.box(*box)):
                pieces.append((shapely.intersection(g, shapely.box(*box)), dst, f"{dst}-all",
                               names[dst]))
                g = shapely.difference(g, shapely.box(*box))
        if parent == "ESH":     # the Polisario strip joins the rest of Western Sahara
            pieces.append((g, "ESH", "ESH-all", names["ESH"]))
            g = None
        if g is not None and not g.is_empty:
            name = next((n for n in (r.name_en, r.name, r.gn_name, r.woe_name)
                         if isinstance(n, str) and n), "Outlying islands")
            name = ADM1_NAME_FIX.get(r.adm1_code, name)
            uid = r.adm1_code
            field = DISSOLVE.get(parent)
            if field:
                reg = getattr(r, field)
                if isinstance(reg, str) and reg not in ("-", ""):
                    name = REGION_NAMES.get(reg, reg)
                    uid = f"{parent}-{re.sub(r'[^A-Za-z0-9]+', '_', unidecode(reg)).strip('_')}"
            pieces.append((g, parent, uid, name))
        typ = r.type_en if isinstance(r.type_en, str) else ""
        for geom, par, uid, name in pieces:
            u = groups.setdefault(uid, {"name": name, "parent": par, "geoms": [], "type": typ})
            u["geoms"].append(geom)
    # every country gets at least one unit
    have = {u["parent"] for u in groups.values()}
    missing = [k for k in keys if k not in have]
    units = []
    for uid, u in groups.items():
        g = shapely.segmentize(shapely.union_all(u["geoms"]), 0.05)
        g = shapely.transform(g, proj)
        if not is_latin(u["name"]):
            u["name"] = unidecode(u["name"])
        units.append({"id": "adm1:" + uid, "name": u["name"], "kind": "state",
                      "parent": u["parent"], "type": u["type"],
                      "parts": [shapely.to_wkb(p) for p in polys(g)]})
    return units, missing


def dedupe(adm1, fua):
    """Same name twice in one country: states get their type, metros their state."""
    def dups(units):
        c = collections.Counter((u["parent"], u["name"]) for u in units)
        return [u for u in units if c[(u["parent"], u["name"])] > 1]
    for u in dups(adm1):
        if u.get("type"):
            u["name"] = f"{u['name']} ({u['type'].lower()})"
    a1geo = [shapely.union_all(shapely.from_wkb(u["parts"])) if u["parts"] else shapely.Point()
             for u in adm1]
    tree = shapely.STRtree(a1geo)
    top = {}
    for u in fua:   # the biggest of a name keeps it plain
        k = (u["parent"], u["name"])
        if k not in top or u["pop2015"] > top[k]["pop2015"]:
            top[k] = u
    for u in [u for u in fua if u["name"] == "Unnamed"] + [u for u in dups(fua)
                                                           if u["name"] != "Unnamed"]:
        if u["name"] != "Unnamed" and top[(u["parent"], u["name"])] is u:
            continue
        g = shapely.union_all(shapely.from_wkb(u["parts"]))
        cand = tree.query(g.representative_point(), predicate="intersects")
        st = [adm1[i]["name"] for i in cand if adm1[i]["parent"] == u["parent"]
              and adm1[i]["name"] != u["name"]]
        if st:
            u["name"] = f"{u['name']}, {st[0]}" if u["name"] == "Unnamed" else f"{u['name']} ({st[0]})"
    for units in (adm1, fua):
        seen = collections.Counter()
        for u in sorted(dups(units), key=lambda u: -u.get("pop2015", 0)):
            seen[(u["parent"], u["name"])] += 1
            n = seen[(u["parent"], u["name"])]
            if n > 1:
                u["name"] = f"{u['name']} {n}"


def load_fua(keys):
    df = pyogrio.read_dataframe(FUA)
    uc = pyogrio.read_dataframe(UCDB, layer="UC_centroids") if os.path.exists(UCDB) else None
    if uc is not None:
        meta = pyogrio.read_dataframe(UCDB, columns=["ID_UC_G0", "GC_UCN_MAI_2025",
                                                     "GC_POP_TOT_2025"], read_geometry=False)
        idcol = [c for c in uc.columns if c.upper().startswith("ID_UC")][0]
        uc = uc.merge(meta, left_on=idcol, right_on="ID_UC_G0")
        uc_tree = shapely.STRtree(uc.geometry.values)
    # an FUA name already used by a bigger FUA (in its own country or across a border)
    biggest = {}
    for r in df.itertuples():
        biggest.setdefault(r.eFUA_name, []).append((r.FUA_p_2015, r.eFUA_ID))
    best = {n: max(v)[1] for n, v in biggest.items()}
    units, renamed = [], []
    keyset = set(keys)
    for r in df.itertuples():
        iso = FUA_ISO.get(r.Cntry_ISO, r.Cntry_ISO)
        parent = iso if iso in keyset else None
        name = r.eFUA_name
        why = None
        if name == "UNNAMED":
            why = "unnamed"
        elif not is_latin(name):
            why = "script"
        elif len(biggest[name]) > 1:
            why = "shared"      # renamed after its own biggest urban centre, if that differs
        if why:
            new = None
            if uc is not None:
                idx = uc_tree.query(r.geometry, predicate="contains")
                if len(idx):
                    sub = uc.iloc[idx].sort_values("GC_POP_TOT_2025", ascending=False)
                    for cand in sub["GC_UCN_MAI_2025"]:
                        if isinstance(cand, str) and cand and cand != "N/A":
                            new = cand
                            break
                if why == "shared" and (new is None or new == name or best[name] == r.eFUA_ID
                                        and new.split()[0] in name):
                    new = None
            if new is None and why == "script":
                new = latin_part(name) or unidecode(name).strip()
            if new is None and why == "unnamed":
                new = "Unnamed"
            if new is not None and new != name:
                if not is_latin(new):
                    new = latin_part(new) or unidecode(new).strip()
                renamed.append((r.eFUA_ID, r.Cntry_ISO, name, new, why))
                name = new
        if not is_latin(name):
            name = latin_part(name) or unidecode(name).strip()
        name = FUA_NAME_FIX.get(int(r.eFUA_ID), name)
        units.append({"id": f"fua:{int(r.eFUA_ID)}", "name": name, "kind": "metro",
                      "parent": parent, "iso": r.Cntry_ISO, "pop2015": float(r.FUA_p_2015),
                      "parts": [shapely.to_wkb(p) for p in polys(r.geometry)]})
    return units, renamed


def load_units(keys, names, rebuild=False):
    if os.path.exists(CACHE) and not rebuild:
        with open(CACHE, "rb") as f:
            c = pickle.load(f)
        if c["keys"] == keys:
            return c["adm1"], c["fua"]
    adm1, missing = load_admin1(keys, names)
    for k in missing:     # countries NE gives no admin-1 for: one unit for the whole country
        adm1.append({"id": f"adm1:{k}-all", "name": names[k], "kind": "state", "parent": k,
                     "parts": [], "whole": True})
    fua, renamed = load_fua(keys)
    dedupe(adm1, fua)
    with open(CACHE, "wb") as f:
        pickle.dump({"keys": keys, "adm1": adm1, "fua": fua, "renamed": renamed,
                     "missing": missing}, f)
    return adm1, fua


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    keys, names, parts, cid, _ = build.load_countries(CRS)
    adm1, fua = load_units(keys, names, rebuild=True)
    with open(CACHE, "rb") as f:
        c = pickle.load(f)
    per = collections.Counter(u["parent"] for u in adm1)
    print(f"{len(adm1)} admin-1 units over {len(per)} countries; whole-country units for "
          f"{c['missing']}")
    print("unknown parents:", sorted({u['parent'] for u in adm1} - set(keys)))
    for k in ("GBR", "FRA", "ITA", "PHL", "UKR", "RUS", "MAR", "ESH", "SOM", "CYP", "XKX"):
        print(k, per[k], [u["name"] for u in adm1 if u["parent"] == k][:30])
    print(f"{len(fua)} FUAs; parent missing for", collections.Counter(u["iso"] for u in fua if u["parent"] is None))
    print(f"renamed {len(c['renamed'])}:", collections.Counter(r[4] for r in c["renamed"]))
    print("shared renames:", [(r[1], r[2], r[3]) for r in c["renamed"] if r[4] == "shared"])
    for units in (fua, adm1):
        dup = collections.Counter((u["parent"], u["name"]) for u in units)
        print("dup names left:", [(k, n) for k, n in dup.items() if n > 1][:40])
    print("RUS:", [u["name"] for u in adm1 if u["parent"] == "RUS"])
    print("Dhaka-ish:", [u["name"] for u in fua if "Dhaka" in u["name"]])
    print("examples:", [u["name"] for u in fua if "(" in u["name"] or "," in u["name"]][:40])
    print("UKR:", [u["name"] for u in adm1 if u["parent"] == "UKR"])
    print("BGD fuas:", [u["name"] for u in fua if u["parent"] == "BGD"][:40])
    bad = [u["name"] for u in adm1 + fua if not is_latin(u["name"])]
    print("non-latin left:", bad[:20])


if __name__ == "__main__":
    main()
