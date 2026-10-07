"""Solomon Islands: place the 2019 census's NATIONAL first-language counts on wards
-> data/normalized/sb_model.csv (every row `modelled`).

    python sources/sb_model.py          (after sources/sb_census.py)

WHY A MODEL. The census prints first language for the whole country only (sources/sb_census.py).
Drawn at that grain, Kwara'ae's 50,506 speakers would be spread over all 900 islands by
population, which says how many and nothing about where. But almost every language here belongs
to one stretch of one island, and the census prints three province tables that say how far
people have moved from home. The model keeps every national count exactly and only decides where
inside the country each language's speakers are drawn (AGENT_BRIEF.md 4.4; Anita allowed the
same kind of model for Azerbaijan, ask 006, and Indonesia, ask 009).

STEP 1, PROVINCES. A languages x provinces table fitted by iterative proportional fitting:
  * rows sum to the census's national count of each language (Table 9.6.1-9.6.3);
  * columns sum to each province's population aged 5+ (Table 9.2.2) less its Pidgin speakers;
  * Pidgin is not fitted: Figure 9.6.1 prints each province's share of the 101,588 directly;
  * the starting table (seed) for a local language is where people BORN IN ITS HOME PROVINCE were
    counted (Table P7.2, birthplace x province of enumeration). So Kwara'ae starts out spread like
    the Malaita-born: 80% in Malaita, 12% in Honiara, 6% in Guadalcanal. The fit then scales it,
    because a Honiara-born child of Malaitan parents counts as born in Honiara and still may
    have learnt Kwara'ae first;
  * Kiribati (Gilbertese) starts out like the Micronesian population (Table 8.4.3), since the
    resettled Gilbertese communities are the only Kiribati speakers;
  * the unnamed remainder (90,859: local languages the report does not list, English, other)
    starts out proportional to each province's 5+ population.
STEP 2, WARDS. Inside each province the same fit is done again, languages x wards:
  * columns sum to each ward's 5+ population (P8.3's ward total, religiondots' sb.csv, scaled by
    its province's 5+ share);
  * a language spoken in that province starts out by distance from its Glottolog location
    (data/raw/glottolog, CC BY), exp(-d / KERNEL_KM), or from a list of wards named for it
    (WARDS below) where the census's own ward names say where it is spoken;
  * Pidgin, the remainder, and every language away from its home province start out flat, so
    they fill what the home languages leave, in proportion to population.

WHAT IT CANNOT KNOW. Who in Honiara speaks what (the fit gives Honiara each language in
proportion to its home province's migrants there); where inside a ward; and whether a language's
territory is really a disc around one point. A language with no location is spread over its
province (none at present).
"""
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
NORM = ROOT / "data" / "normalized"
RD = ROOT.parent / "religiondots"
OUT = NORM / "sb_model.csv"
GLOTTO = ROOT / "data" / "raw" / "glottolog" / "languages.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sb_census import PROVINCES, PIDGIN, REMAINDER  # noqa: E402

KERNEL_KM = 10.0
EPS = 1e-4

# census label -> Glottolog id whose location anchors it. Every id was looked up in
# data/raw/glottolog/languages.csv (name, country SB, Oceanic or Central Solomons path), none
# from memory. Where the census name is not Glottolog's: Tolo/Talise is Talise (Tolo is its
# dialect there); Baenggu = Baeggu; Mono = Mono-Alu; Aiwoo = Äiwoo; Tairaha = Bauro (Wikipedia's
# Bauro article gives Tairaha as its other name); Asumnoa = Asumboa (the report: "also referred
# to as Asumbuo"); Laube = Lavukaleve (the report says so); Ghoighoi is the Susubona dialect of
# Blablanga and Mae a dialect of Cheke Holo (the report), anchored on those; Engdewu and Noipa
# are both footnoted in the report to Ethnologue's `ngr`, Glottolog's Nanggu (Nendö); Tauma is
# taken to be Taumako (Glottolog's dialect of Vaeakau-Taumako, Duff Islands), which the report
# says it cannot verify; Ulawa is a dialect of Sa'a in Glottolog but its own island, so it is
# anchored on the Ulawa wards instead.
GLOTTOCODE = {
    "Kwara'ae": "kwar1239", "Are'are": "area1240", "Lau": "lauu1247", "Kwaio": "kwai1243",
    "Lengo": "leng1259", "To'abaita": "toab1237", "Tolo/Talise": "tali1259", "Gela": "gela1263",
    "Ghari": "ghar1239", "Roviana": "rovi1238", "Cheke Holo": "chek1238", "Baelelea": "bael1237",
    "Babatana": "baba1268", "Bilua": "bilu1245", "Sa'a": "saaa1240", "Arosi": "aros1241",
    "Baenggu": "baeg1237", "Marovo": "maro1244", "Birao": "bira1254", "Wala": "wala1266",
    "Aiwoo": "ayiw1239", "Tairaha": "baur1252", "Varisi": "vari1239", "Kahua": "kahu1241",
    "Owa": "owaa1237", "Mono": "mono1273", "RenBell": "renn1242", "Simbo": "simb1256",
    "Lungga": "lung1249", "Duke": "duke1237", "Dori'o": "dori1246", "Touo": "touo1238",
    "Vaghua": "vagh1249", "Nalögo": "nalo1235", "Gula'alaa": "gula1270", "Ughele": "ughe1237",
    "Sikaiana": "sika1261", "Vangunu": "vang1243", "Amba": "amba1266", "Engdewu": "nang1262",
    "Tanibili": "tani1255", "Ghoighoi": "blab1237", "Zazao": "zaza1245", "Anuta": "anut1237",
    "Mae": "chek1238", "Ririo": "riri1237", "Asumnoa": "asum1237", "Laghu": "lagh1246",
    "Laube": "lavu1241", "Tauma": "pile1238", "Noipa": "nang1262", "Guliguli": "guli1244",
    "Lovono": "vano1237", "Tanema": "tane1237", "Dororo": "doro1267", "Kazukuru": "kazu1245",
}
# census ward ids (province digit(s) + ward number) named for the language; these replace the
# Glottolog point as the anchor
WARDS = {
    "Ulawa": ["801", "802", "803"],                 # North, South, West Ulawa
    "Arosi": ["805", "806", "807", "808"],          # Arosi South, West, North, East
    "Tairaha": ["809", "810", "811"],               # Bauro West, Central, East
    "Owa": ["815", "816"],                          # Santa Ana, Santa Catalina (Owa Raha, Owa Riki)
    "Kiribati": ["101"],                            # Wagina, the Gilbertese resettlement island
}
# home province for the labels the report does not file under one (Table 9.6.2), asserted
# against the province of the ward nearest the Glottolog point
HOME = {
    "Wala": "Malaita", "Owa": "Makira-Ulawa", "Ulawa": "Makira-Ulawa", "Mono": "Western",
    "Simbo": "Western", "Lungga": "Western", "Duke": "Western", "Dori'o": "Malaita",
    "Touo": "Western", "Vaghua": "Choiseul", "Nalögo": "Temotu", "Gula'alaa": "Malaita",
    "Ughele": "Western", "Sikaiana": "Malaita", "Vangunu": "Western", "Amba": "Temotu",
    "Anuta": "Temotu",
}


def ipf(seed, rows, cols, iters=500, tol=1e-6):
    x = seed.astype(float).copy()
    for _ in range(iters):
        rs = x.sum(1)
        x *= np.where(rs > 0, rows / np.where(rs > 0, rs, 1), 0)[:, None]
        cs = x.sum(0)
        x *= np.where(cs > 0, cols / np.where(cs > 0, cs, 1), 0)[None, :]
        if np.abs(x.sum(1) - rows).max() < tol * max(rows.max(), 1):
            break
    return x


def hav(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


def main():
    import geopandas as gpd

    nat = pd.read_csv(NORM / "sb.csv")
    nat = dict(zip(nat["source_category"], nat["count"].astype(float)))
    inp = pd.read_csv(NORM / "sb_inputs.csv", keep_default_na=False)
    def tab(name):
        t = inp[inp["table"] == name]
        return dict(zip(t["row"], t["value"].astype(float)))
    pop5, fig, mic = tab("pop5_9.2.2"), tab("pidgin_share_fig9.6.1"), tab("micronesian_8.4.3")
    b = inp[inp["table"] == "birth_P7.2"]
    born = defaultdict(dict)                     # born[h][p] = born in h, counted in p
    for _, r in b.iterrows():
        born[r["col"]][r["row"]] = float(r["value"])
    nat_home = pd.read_csv(NORM / "sb.csv")
    home = {}
    for _, r in nat_home.iterrows():
        m = str(r["note"]).split("home=")
        if len(m) > 1 and m[1]:
            home[r["source_category"]] = m[1]
    home.update(HOME)

    # ---- wards: 5+ population and a population-weighted centre ----
    rdn = pd.read_csv(RD / "data" / "normalized" / "sb.csv", dtype={"geo_id": str})
    rdn = rdn[(rdn["geo_level"] == "ward") & (rdn["source_category"] == "Total")].copy()
    rdn["prov"] = rdn["note"].str.extract(r"province=([^;]+)")[0].str.strip()
    rdn["prov"] = rdn["prov"].replace({"Honiara City Council": "Honiara"})
    assert len(rdn) == 183 and rdn["count"].sum() == 720_956, "religiondots ward totals moved"
    assert set(rdn["prov"]) == set(PROVINCES), set(rdn["prov"])
    lut = pd.read_csv(RD / "data" / "geo" / "sb" / "sb_lookup.csv", dtype=str)
    rdn["unit"] = rdn["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    assert rdn["unit"].notna().all()
    ptot = rdn.groupby("prov")["count"].sum()
    rdn["pop5"] = rdn["count"] * rdn["prov"].map(lambda p: pop5[p] / ptot[p])
    hexes = gpd.read_file(RD / "data" / "geo" / "sb" / "sb_hexes.gpkg")
    c = hexes.geometry.representative_point()
    hexes["x"], hexes["y"] = c.x, c.y
    hexes = hexes[hexes["pop"] > 0]
    hexes = pd.DataFrame({"unit": hexes["unit"], "px": hexes["x"] * hexes["pop"],
                          "py": hexes["y"] * hexes["pop"], "pop": hexes["pop"]})
    s = hexes.groupby("unit")[["px", "py", "pop"]].sum()
    cen = pd.DataFrame({"lon": s["px"] / s["pop"], "lat": s["py"] / s["pop"]})
    rdn = rdn.join(cen, on="unit")
    assert rdn["lon"].notna().all(), "a ward with no populated hex"
    wards = rdn.reset_index(drop=True)

    g = pd.read_csv(GLOTTO, dtype=str, keep_default_na=False).set_index("ID")
    point = {lab: (float(g.at[gc, "Latitude"]), float(g.at[gc, "Longitude"]))
             for lab, gc in GLOTTOCODE.items()}

    # assert each language's home province against the ward nearest its anchor
    for lab, (la, lo) in point.items():
        if lab in WARDS:
            continue
        d = wards.apply(lambda r: hav(la, lo, r["lat"], r["lon"]), axis=1)
        near = wards.loc[d.idxmin(), "prov"]
        if lab not in home:
            raise SystemExit(f"{lab}: no home province")
        if near != home[lab] and lab not in ("Anuta",):
            print(f"  note: {lab} is filed under {home[lab]} but its Glottolog point is nearest "
                  f"a ward in {near} ({d.min():.0f} km)")
    for lab, ids in WARDS.items():
        for i in ids:
            assert i in set(wards["geo_id"]), f"{lab}: ward {i} missing"
    named = [k for k in nat if k not in ("Pidgin", REMAINDER)]
    missing = [k for k in named if k not in point and k not in WARDS]
    if missing:
        raise SystemExit(f"no anchor for {missing}")
    for k in named:
        if k != "Kiribati" and k not in home:
            raise SystemExit(f"{k}: no home province")

    # ---- step 1: provinces ----
    pid = {p: PIDGIN * fig[p] / sum(fig.values()) for p in PROVINCES}
    cats = [k for k in nat if k != "Pidgin"]
    rows = np.array([nat[k] for k in cats])
    cols = np.array([pop5[p] - pid[p] for p in PROVINCES])
    assert abs(rows.sum() - cols.sum()) < 1, (rows.sum(), cols.sum())
    seed = np.zeros((len(cats), len(PROVINCES)))
    for i, k in enumerate(cats):
        for j, p in enumerate(PROVINCES):
            if k == REMAINDER:
                seed[i, j] = pop5[p]
            elif k == "Kiribati":
                seed[i, j] = mic[p]
            else:
                seed[i, j] = born[home[k]][p]
    seed += EPS
    xp = ipf(seed, rows, cols)
    assert np.allclose(xp.sum(1), rows, rtol=1e-4) and np.allclose(xp.sum(0), cols, rtol=1e-4)
    prov = pd.DataFrame(xp, index=cats, columns=PROVINCES)
    prov.loc["Pidgin"] = [pid[p] for p in PROVINCES]

    # ---- step 2: wards inside each province ----
    out = []
    for p in PROVINCES:
        wp = wards[wards["prov"] == p].reset_index(drop=True)
        langs = [k for k in prov.index if prov.at[k, p] > 1e-6]
        r = np.array([prov.at[k, p] for k in langs])
        cw = wp["pop5"].to_numpy()
        s = np.ones((len(langs), len(wp)))
        for i, k in enumerate(langs):
            if k in WARDS and (k != "Kiribati" or p == "Choiseul"):
                ids = set(WARDS[k])
                inside = wp["geo_id"].isin(ids).to_numpy()
                if inside.any():
                    pts = wards[wards["geo_id"].isin(ids)][["lat", "lon"]].to_numpy()
                    d = np.array([min(hav(a, o, la, lo) for a, o in pts)
                                  for la, lo in wp[["lat", "lon"]].to_numpy()])
                    s[i] = np.exp(-d / KERNEL_KM) + EPS
            elif k in point and home.get(k) == p:
                la, lo = point[k]
                d = np.array([hav(la, lo, a, o) for a, o in wp[["lat", "lon"]].to_numpy()])
                s[i] = np.exp(-d / KERNEL_KM) + EPS
        x = ipf(s, r, cw, iters=2000)
        if not (np.allclose(x.sum(1), r, rtol=1e-3) and np.allclose(x.sum(0), cw, rtol=1e-3)):
            print(f"  !! {p}: ward fit off by {np.abs(x.sum(1) - r).max():.1f} people (rows), "
                  f"{np.abs(x.sum(0) - cw).max():.1f} (cols)")
        for i, k in enumerate(langs):
            for j in range(len(wp)):
                if x[i, j] >= 0.005:
                    out.append((wp.at[j, "geo_id"], wp.at[j, "unit"], p, k, round(x[i, j], 2)))

    df = pd.DataFrame(out, columns=["geo_id", "unit", "province", "source_category", "count"])
    tot = df.groupby("source_category")["count"].sum()
    for k, v in nat.items():
        assert abs(tot.get(k, 0) - v) < max(2, 1e-3 * v), (k, tot.get(k, 0), v)
    assert abs(df["count"].sum() - 631_061) < 50, df["count"].sum()
    df["tier"] = "modelled"
    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)

    # ---- report ----
    print(f"wrote {OUT}: {len(df):,} rows, {df['count'].sum():,.0f} people, "
          f"{df['geo_id'].nunique()} wards")
    print("\nprovinces, top languages (share of the province's 5+):")
    for p in PROVINCES:
        col = prov[p].sort_values(ascending=False)
        tops = ", ".join(f"{k if k != REMAINDER else 'unnamed'} {100 * v / pop5[p]:.0f}%"
                         for k, v in col.head(6).items())
        print(f"  {p:16s} {tops}")
    print("\nwhere each big language lands (share of its speakers by province):")
    for k in ["Kwara'ae", "Are'are", "Lau", "Lengo", "Gela", "Roviana", "RenBell", "Kiribati"]:
        row = prov.loc[k] / prov.loc[k].sum()
        print(f"  {k:10s} " + ", ".join(f"{p} {100 * v:.0f}%" for p, v in
                                         row.sort_values(ascending=False).head(4).items()))
    print("\nlargest language by ward, Malaita:")
    m = df[df["province"] == "Malaita"]
    top = m.loc[m.groupby("geo_id")["count"].idxmax()]
    wt = m.groupby("geo_id")["count"].sum()
    names = dict(zip(wards["geo_id"], wards["geo_name"]))
    for _, r in top.iterrows():
        print(f"  {r['geo_id']} {names[r['geo_id']]:24s} {r['source_category']:12s} "
              f"{100 * r['count'] / wt[r['geo_id']]:.0f}%")


if __name__ == "__main__":
    main()
