"""Papua New Guinea: a language-area model, not a count.

    python sources/pg_build.py
        -> data/normalized/pg.csv            province x node, counts (every row `modelled`)
        -> data/normalized/pg_languages.csv  one row per drawn language: node, point, estimate
        -> taxonomy/tree.d/pg.txt            the nodes, generated from Glottolog's classification

No PNG census has published first languages by area. 2011 and 2024 asked literacy in Tok Pisin,
English, Hiri Motu and "tok ples" only; 1971 asked "language usually spoken at home" but tabulated
only the three lingua francas; 1980's long form asked the language spoken most at home but
Laycock (1985, Pacific Linguistics C-70) reports it unpublished. So the map is drawn from where
each language is spoken (Glottolog's point for it, CC BY) and how many speak it (Joshua Project's
people-group populations for PNG, which follow Ethnologue; summed by ISO 639-3 code):

  * each province's people (2024 census Final Figures, religiondots' pg_lookup.csv) are shared
    among the languages whose Glottolog point falls in that province, in proportion to their
    estimates; a language with no estimate gets the median estimate of the province's others;
  * NCD (Port Moresby) holds no language point (Motu's and Koitabu's are in Central) and is a
    migrant city: its people are shared among all the country's drawn languages in proportion
    to their estimates (no published origin mix);
  * Tok Pisin is NOT drawn as a first language: no cited urban share was found (sources/pg.md).
  * Languages Glottolog marks extinct, nearly extinct or moribund are left out; their province's
    people go to the province's other languages.

Nodes: papuan.<family>[.<subgroup>].<language>, austronesian.oceanic.<subgroup>.<language>,
isolate.<language>. Families as readers know them: Trans-New Guinea as Pawley & Hammarström
(2018) draw it (Glottolog's Nuclear TNG plus twelve families Glottolog keeps apart), Sepik with
Ndu, Ramu with Lower Sepik. Subgroups: a family with more than MAX_GROUP drawn languages is cut at
the Glottolog level that brings every group to MAX_GROUP or fewer. The record is sources/pg.md.
"""
import io
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

GLOT = ROOT / "data" / "raw" / "glottolog"
JP_CSV = ROOT / "data" / "raw" / "pg" / "joshuaproject_pgic.csv"
LOOKUP = RD_GEO / "pg" / "pg_lookup.csv"
PROV = RD_GEO / "pg" / "pg_provinces.gpkg"
OUT = ROOT / "data" / "normalized" / "pg.csv"
OUT_LANG = ROOT / "data" / "normalized" / "pg_languages.csv"
FRAG = ROOT / "taxonomy" / "tree.d" / "pg.txt"

PNG_2024 = 10_185_363       # 2024 census Final Figures, the sum of pg_lookup.csv
N_PROV = 22
NCD = "PG04"
MAX_GROUP = 40              # the colour grid has 45 slots per parent
SNAP_KM = 150               # a point outside every province joins the nearest within this

# Glottolog families left out: not first languages of a place (pidgins: Tok Pisin, Hiri Motu...;
# sign languages; Bookkeeping holds unattested and spurious entries).
SKIP_FAMILY = {"Pidgin", "Sign Language", "Bookkeeping", "Unattested", "Artificial Language",
               "Mixed Language", "Speech Register", "Unclassifiable",
               # Glottolog files Tok Pisin under Indo-European as an English creole; with English
               # it is the only Indo-European entry here, and neither is drawn (no cited share)
               "Indo-European"}     # Glottolog family names
SKIP_AES = {"aes-extinct", "aes-nearly_extinct", "aes-moribund"}

# Trans-New Guinea as Pawley & Hammarström (2018) accept it (Wikipedia, "Trans-New Guinea
# languages", read 2026-10-05): Glottolog's Nuclear Trans New Guinea plus these families, which
# Glottolog keeps as families of their own. Anim, Eleman, Kamula-Elevala and the rest stay apart.
TNG = "nucl1709"
TNG_FOLD = ["Angan", "Dagan", "Mailuan", "Bosavi", "Koiarian", "East Strickland", "Kiwaian",
            "Suki-Gogodala", "Turama-Kikori", "Yareban", "Manubaran", "Kutubuan", "Kunimaipan"]
# Merged as the familiar wider families (Foley; Wikipedia's "Sepik languages" and
# "Ramu-Lower Sepik languages").
MERGE = {"Sepik": ("sepik", "Sepik"), "Ndu": ("sepik", "Sepik"),
         "Ramu": ("ramu_lower_sepik", "Ramu-Lower Sepik"),
         "Lower Sepik": ("ramu_lower_sepik", "Ramu-Lower Sepik")}
# Family colours, OKLCH. Papuan families sit around papuan's 120 (green); Oceanic is uk.txt's
# 225 blue. Hand-picked so the families that meet on the ground differ.
FAMILY_COLOUR = {
    "papuan.tng": "0.72 0.14 135",
    "papuan.sepik": "0.80 0.15 95",
    "papuan.torricelli": "0.68 0.14 70",
    "papuan.ramu_lower_sepik": "0.84 0.12 160",
}
# Groups another fragment already defines: reuse its id and label.
REUSE = {"Kiwaian": ("papuan.kiwai", "Kiwai"),
         "Northwest Solomonic": ("austronesian.oceanic.nw_solomonic", "Northwest Solomonic")}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    if s and s[0].isdigit():
        s = "l" + s
    return s or "x"


def largest_remainder(vals, total):
    f = np.asarray(vals, dtype=float)
    base = np.floor(f)
    k = int(round(total - base.sum()))
    base[np.argsort(-(f - base))[:k]] += 1
    return base.astype(int)


def load_jp():
    txt = JP_CSV.read_text(encoding="utf-8-sig")
    jp = pd.read_csv(io.StringIO(txt[txt.index("ROG3,"):]), dtype=str)
    jp = jp[jp["Ctry"] == "Papua New Guinea"].copy()
    jp["Population"] = jp["Population"].astype(int)
    return jp


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    say(len(lut) == N_PROV and int(lut["pop"].sum()) == PNG_2024,
        f"pg_lookup.csv: {len(lut)} provinces, {int(lut['pop'].sum()):,} people (2024 census)")
    pop = lut.set_index("unit")["pop"].astype(int)

    g = pd.read_csv(GLOT / "languages.csv", dtype=str)
    names = g.set_index("ID")["Name"].to_dict()
    tops = g[g.Level == "family"]
    gid = lambda n: tops.loc[tops.Name == n, "ID"].tolist()   # noqa: E731
    say(gid("Nuclear Trans New Guinea") == [TNG], "Glottolog: nucl1709 is Nuclear Trans New Guinea")
    aust = [i for i in gid("Austronesian") if pd.isna(tops.set_index("ID").at[i, "Family_ID"])]
    ocea = gid("Oceanic")
    say(len(aust) == 1 and len(ocea) == 1, f"Glottolog: Austronesian {aust}, Oceanic {ocea}")
    AUST, OCEA = aust[0], ocea[0]
    v = pd.read_csv(GLOT / "values.csv", dtype=str,
                    usecols=["Language_ID", "Parameter_ID", "Value", "Code_ID"])
    cls = v[v.Parameter_ID == "classification"].set_index("Language_ID")["Value"].to_dict()
    aes = v[v.Parameter_ID == "aes"].set_index("Language_ID")["Code_ID"].to_dict()

    jp = load_jp()
    say(len(jp) > 800, f"Joshua Project: {len(jp)} PNG people groups, {jp.Population.sum():,} people")
    est = jp.groupby("ROL3")["Population"].sum()

    langs = g[g.Level == "language"].copy()
    in_pg = langs.Countries.fillna("").str.split(";").apply(lambda c: "PG" in c)
    jp_iso = langs.ISO639P3code.isin(est.index)
    cand = langs[in_pg | jp_iso].copy()
    cand = cand[cand.Latitude.notna()]
    cand["aes"] = cand.ID.map(aes).fillna("")
    cand["fam"] = cand.Family_ID.map(names).fillna("")
    n0 = len(cand)
    drop_fam = cand.fam.isin(SKIP_FAMILY)
    drop_aes = cand.aes.isin(SKIP_AES)
    print(f"  Glottolog languages listed in PG or carrying a JP-PNG ISO code: {n0}; "
          f"left out as pidgin/sign/unclassifiable {int(drop_fam.sum())}, "
          f"as extinct, nearly extinct or moribund {int((drop_aes & ~drop_fam).sum())}")
    cand = cand[~drop_fam & ~drop_aes].copy()
    cand["est"] = cand.ISO639P3code.map(est)

    # ---- points to provinces
    import geopandas as gpd
    prov = gpd.read_file(PROV)[["unit", "geometry"]]
    pts = gpd.GeoDataFrame(cand, geometry=gpd.points_from_xy(
        cand.Longitude.astype(float), cand.Latitude.astype(float)), crs=4326)
    j = gpd.sjoin(pts, prov, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    cand["unit"] = j["unit"]
    out_pts = cand["unit"].isna()
    p3 = pts.to_crs(32755)
    pr3 = prov.to_crs(32755)
    snapped, dropped = [], []
    for i in cand.index[out_pts]:
        d = pr3.distance(p3.geometry[i]) / 1000
        k = int(d.idxmin())
        listed_pg = "PG" in str(cand.at[i, "Countries"]).split(";")
        # a point in Indonesia counts only for a language Joshua Project has in PNG
        ok = d[k] <= SNAP_KM and (pd.notna(cand.at[i, "est"]) or
                                   (listed_pg and cand.at[i, "Countries"] == "PG"))
        if ok:
            cand.at[i, "unit"] = pr3.at[k, "unit"]
            snapped.append((cand.at[i, "Name"], round(float(d[k]))))
        else:
            dropped.append((cand.at[i, "Name"], cand.at[i, "Countries"], round(float(d[k]))))
    print(f"  {len(snapped)} points outside every province snapped to the nearest "
          f"(<= {SNAP_KM} km): {snapped}")
    print(f"  {len(dropped)} left out (point outside PNG, no PNG estimate): {dropped}")
    cand = cand[cand.unit.notna()].copy()

    # ---- estimates: missing -> the province median of the known ones
    cand["est_source"] = np.where(cand.est.notna(), "joshuaproject", "province_median")
    med = cand.groupby("unit")["est"].median()
    cand["est"] = cand.est.fillna(cand.unit.map(med)).fillna(cand.est.median())
    print(f"  {len(cand)} languages drawn; {int((cand.est_source == 'joshuaproject').sum())} with "
          f"a Joshua Project estimate ({cand.loc[cand.est_source == 'joshuaproject', 'est'].sum():,.0f} "
          f"people), {int((cand.est_source != 'joshuaproject').sum())} on their province's median")
    unused = sorted(set(est.index) - set(cand.ISO639P3code.dropna()))
    un_pop = int(est[unused].sum())
    print(f"  JP ISO codes not drawn: {len(unused)}, {un_pop:,} people, of them "
          f"{[(c, int(est[c])) for c in unused if est[c] > 3000]}")

    # ---- the tree
    def family_of(r):
        path = cls.get(r.ID, "").split("/") if cls.get(r.ID) else []
        if r.get("Is_Isolate") == "True" or not path:
            return ("isolate", "Language isolates", [])
        top = path[0]
        topname = names.get(top, top)
        if top == TNG:
            return ("papuan.tng", "Trans-New Guinea", path[1:])
        if topname in TNG_FOLD:
            return ("papuan.tng", "Trans-New Guinea", path)
        if top == AUST:
            # Austronesian > Malayo-Polynesian > ... > Oceanic: start below Oceanic
            if OCEA in path:
                return ("austronesian.oceanic", "Oceanic", path[path.index(OCEA) + 1:])
            return ("austronesian", "Austronesian", path[1:])
        if topname in MERGE:
            fid, flab = MERGE[topname]
            return (f"papuan.{fid}", flab, path)
        if topname.endswith("Torricelli"):
            return ("papuan.torricelli", "Torricelli", path[1:])
        return (f"papuan.{slug(topname)}", topname, path[1:])

    fams = cand.apply(family_of, axis=1)
    cand["fam_id"] = [f[0] for f in fams]
    cand["fam_label"] = [f[1] for f in fams]
    cand["path"] = [f[2] for f in fams]

    # count drawn languages under each (family, glottolog ancestor)
    cnt = {}
    for fid, path in zip(cand.fam_id, cand.path):
        cnt[fid] = cnt.get(fid, 0) + 1
        for a in path:
            cnt[(fid, a)] = cnt.get((fid, a), 0) + 1
    groups = []
    for fid, path in zip(cand.fam_id, cand.path):
        if fid == "isolate" or cnt[fid] <= MAX_GROUP or not path:
            groups.append(None)
            continue
        i = 0
        while cnt[(fid, path[i])] > MAX_GROUP and i + 1 < len(path):
            i += 1
        # a merged family's member named like the family itself: go one level further down
        while (i + 1 < len(path) and slug(names.get(path[i], "")) == fid.split(".")[-1]):
            i += 1
        if slug(names.get(path[i], "")) == fid.split(".")[-1]:
            groups.append(None)       # straight under the family it is named for
        else:
            groups.append(path[i])
    cand["grp"] = groups

    nodes = {}      # id -> (label, colour or "")

    def add_node(nid, label, colour=""):
        if nid in nodes and nodes[nid][0] != label:
            raise SystemExit(f"node clash {nid}: {nodes[nid][0]!r} vs {label!r}")
        nodes.setdefault(nid, (label, colour))

    add_node("papuan", "Papuan languages")
    add_node("austronesian", "Austronesian")
    add_node("austronesian.oceanic", "Oceanic")
    add_node("isolate", "Language isolates")
    leaf_ids = []
    for r in cand.itertuples():
        add_node(r.fam_id, r.fam_label, FAMILY_COLOUR.get(r.fam_id, ""))
        parent = r.fam_id
        if r.grp:
            # Glottolog's "X linkage" (a chain of dialects, not a tree) reads as plain "X"
            gname = re.sub(r" linkage$", "", names.get(r.grp, r.grp))
            if gname in REUSE:
                parent, glab = REUSE[gname]
            else:
                parent, glab = f"{r.fam_id}.{slug(gname)}", gname
            add_node(parent, glab)
        leaf_ids.append((parent, r.Name, r.ID))
    seen = {}
    ids = []
    for parent, name, gc in leaf_ids:
        nid = f"{parent}.{slug(name)}"
        if nid in nodes or nid in seen:
            nid = f"{nid}_{gc}"
        seen[nid] = 1
        ids.append(nid)
        add_node(nid, name)
    cand["node"] = ids

    # ---- counts
    rows = []
    scale = PNG_2024 / cand.est.sum()
    for u, p in pop.items():
        d = cand[cand.unit == u]
        if u == NCD:
            say(len(d) == 0, f"NCD: {len(d)} language points inside it (Motu and Koitabu's "
                "points are in Central); its people shared across all the country's languages")
            for nid, n in zip(cand.node, cand.est / cand.est.sum() * p):
                rows.append((u, nid, n, "Port Moresby, national mix of speaker estimates"))
        else:
            say(len(d) > 0, f"{u}: {len(d)} languages with points")
            for nid, n in zip(d.node, d.est / d.est.sum() * p):
                rows.append((u, nid, n, "province's people by speaker estimate"))
    df = pd.DataFrame(rows, columns=["unit", "node", "count", "label"])
    df = df.groupby(["unit", "node"], as_index=False).agg(count=("count", "sum"),
                                                           label=("label", "first"))
    parts = []
    for u, d in df.groupby("unit"):
        d = d.copy()
        d["count"] = largest_remainder(d["count"], pop[u])
        parts.append(d)
    df = pd.concat(parts)
    df = df[df["count"] > 0]
    say(int(df["count"].sum()) == PNG_2024, f"drawn total {int(df['count'].sum()):,}")
    bad = [u for u in pop.index if int(df.loc[df.unit == u, "count"].sum()) != pop[u]]
    say(not bad, f"every province sums to its 2024 count {bad}")

    nat = df.groupby("node")["count"].sum().sort_values(ascending=False)
    fam_tot = df.assign(f=df.node.map(dict(zip(cand.node, cand.fam_id)))).groupby("f")["count"].sum()
    print("\n  families, as drawn:")
    for k, v_ in fam_tot.sort_values(ascending=False).head(15).items():
        print(f"    {k:40s} {v_:>10,}  {v_ / PNG_2024:6.2%}")
    print("  largest languages, as drawn (JP estimate):")
    e_by = dict(zip(cand.node, cand.est))
    for k, v_ in nat.head(15).items():
        print(f"    {k:60s} {v_:>9,}  ({e_by[k]:,.0f})")
    r = np.corrcoef(np.log([e_by[k] for k in nat.index]), np.log(nat.values))[0, 1]
    print(f"  log drawn vs log estimate, across {len(nat)} languages: r = {r:.3f}")

    lab = lut.set_index("unit")["name"]
    res = pd.DataFrame({
        "geo_id": df.unit, "geo_level": "province", "geo_name": df.unit.map(lab),
        "source_category": df.node, "source_label": df.label, "count": df["count"],
        "tier": "modelled", "source_id": "pg_language_area_2024", "year": 2024,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["geo_id", "count"], ascending=[True, False]).to_csv(OUT, index=False,
                                                                         encoding="utf-8")
    cand[["node", "ID", "ISO639P3code", "Name", "Latitude", "Longitude", "unit", "est",
          "est_source", "aes", "fam_id"]].rename(columns={"ID": "glottocode", "ISO639P3code": "iso"}
                                                 ).to_csv(OUT_LANG, index=False, encoding="utf-8")
    print(f"wrote {OUT} ({len(res)} rows, {res.source_category.nunique()} nodes), {OUT_LANG}")

    # ---- fragment
    used = set(cand.node)
    for n in list(used):
        while "." in n:
            n = n.rsplit(".", 1)[0]
            used.add(n)
    lines = [
        "# Papua New Guinea (taxonomy/pg2024.py). GENERATED by sources/pg_build.py from Glottolog's",
        "# classification (data/raw/glottolog, CC BY); edit the script, not this file, except to",
        "# hand-pick a colour (then move it into the script's FAMILY_COLOUR so a rebuild keeps it).",
        "# Families as readers know them: Trans-New Guinea as Pawley & Hammarstrom (2018), i.e.",
        "# Glottolog's Nuclear TNG plus " + ", ".join(TNG_FOLD) + ";",
        "# Sepik with Ndu; Ramu with Lower Sepik; isolates on `isolate`. A family with more than",
        f"# {MAX_GROUP} drawn languages is cut into Glottolog subgroups of at most {MAX_GROUP}.",
        "# Colours: papuan families around papuan's green (120), Oceanic blue (225); TNG, Sepik,",
        "# Torricelli and Ramu-Lower Sepik hand-picked so the families that meet differ.",
    ]
    for nid in sorted(used, key=lambda s: (s.count("."), s)):
        label, colour = nodes[nid]
        lines.append(f"{nid} | {label}" + (f" | {colour}" if colour else ""))
    FRAG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {FRAG} ({len(used)} nodes)")


if __name__ == "__main__":
    main()
