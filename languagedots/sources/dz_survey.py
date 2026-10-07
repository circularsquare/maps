"""Algeria: Arab Barometer waves VI and VII (2020-2022), ethnic group read as language, by wilaya,
on the RGPH 2008 wilaya populations -> data/normalized/dz.csv. The record is sources/dz.md.

    python sources/dz_survey.py --fetch   extract Algeria's rows from religiondots' Arab Barometer
                                          and Afrobarometer .sav files (read-only) -> data/raw/dz/
    python sources/dz_survey.py           shares, checks, data/normalized/dz.csv
    python sources/dz_survey.py --check1966   compare each wilaya with the 1966 census surface

NO CENSUS SINCE 1966 ASKS. The 1966 census asked mother tongue (19% Berber-speaking, 2,287,997
of 12 million) and published it by wilaya and daira (C.N.R.P. 1970, 15 vols, not online);
Nesson (1994, Travaux de l'Institut de Geographie de Reims 85-86, pp. 93-107, persee.fr) quotes
it daira by daira, and those figures are ANCHORS_1966 below. 1977 onwards ask no language.

TWO KINDS OF SURVEY QUESTION, AND THEY DISAGREE (sources/dz.md section 2):
  * "first language" / "home language": Arab Barometer II (2011), III (2013), IV (2016) and
    Afrobarometer R5 (2013), R6 (2015), about 6,000 adults. Berber 5-11% nationally, and in the
    Kabyle heartland far below the census: Bejaia 29% pooled against >85% in 1966, Khenchela
    1 of 58 against 72%, Oum El Bouaghi 0 of 62. Every Afrobarometer interview was in Arabic, by
    Arabic-speaking interviewers, in Tizi Ouzou too. Used here only for French.
  * "ethnic group" (Arab / Amazigh / Tuareg / other): Arab Barometer VI part 3 (2021) and VII
    (2021-22), 3,366 adults, all 48 wilayas. Amazigh 23% nationally; Bejaia 89%, Tizi Ouzou 87%,
    Khenchela 68%, Algiers 41%: the 1966 census's geography. These are drawn, read as language
    under AGENT_BRIEF section 2's ethnicity ruling.
A wilaya's Amazigh answers go on the Berber language of that wilaya where one is spoken there
alone (KABYLE, CHAOUI, MZAB, TUAREG below; the place-dependent label rule of spec section 3),
and on the Berber group node elsewhere (Algiers, Oran, Setif's neighbours...).
"""
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD, RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "dz"
OUT = HERE / "data" / "normalized" / "dz.csv"
ETH_X = RAW / "arb_dz_ethnicity.csv"
LANG_X = RAW / "dz_first_language.csv"
ARB = RD / "data" / "raw" / "arabbarometer"
AFB = RD / "data" / "raw" / "afrobarometer"
LOOKUP = RD_GEO / "dz" / "dz_lookup.csv"          # religiondots: 48 wilayas, RGPH 2008 `pop`
GEONAMES = RD / "data" / "raw" / "dz" / "geonames_DZ.zip"
RGPH_2008 = 34_080_030
SOURCE_ID = "arab_barometer_vi_vii_algeria"
K = 8.0            # prior weight in respondents (one interviewer's cluster), towards the nation

# (wave, file, ethnic-group column)
ETH_WAVES = [
    ("VI", "Arab_Barometer_Wave_6_Part_3_ENG_RELEASE.sav", "Q1012B"),
    ("VII", "AB7_ENG_Release_Version6.sav", "Q1012B"),
]
# (survey, folder, file, language column, wilaya column, weight column, country code)
LANG_WAVES = [
    ("ArB II 2011", ARB, "ABII_English.sav", "q10191", "q1", "wt"),
    ("ArB III 2013", ARB, "ABIII_English.sav", "q1019_1", "q1", "wt"),
    ("ArB IV 2016", ARB, "ABIV_English_Updated.sav", "q1019a", "q1", "wt"),
    ("AfB R5 2013", AFB, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "Q2", "REGION", "withinwt"),
    ("AfB R6 2015", AFB, "merged_r6_data_2016_36countries2.sav", "Q2", "LOCATION.LEVEL.1",
     "withinwt"),
]
# Afrobarometer R5's 36 REGION labels are the official wilaya names 1-36 in code order, but the
# respondents under them are not: 128 under "Tamanghasset" (176,637 people), 130 under
# "Tiaret", 30 under "Algiers", and the Berber answers under "Bouira" not "Tizi Ouzou". Its
# wilaya is not used; it counts in the national checks only.
NO_WILAYA = {"AfB R5 2013"}

# Amazigh answers: wilaya -> the one Berber language spoken there (sources/dz.md section 4)
KABYLE = {"DZ06", "DZ10", "DZ15", "DZ19", "DZ34", "DZ35"}   # Bejaia Bouira Tizi Ouzou Setif BBA Boumerdes
CHAOUI = {"DZ04", "DZ05", "DZ07", "DZ12", "DZ40"}           # Oum El Bouaghi Batna Biskra Tebessa Khenchela
MZAB = {"DZ47"}                                             # Ghardaia
TUAREG = {"DZ11", "DZ33"}                                   # Tamanrasset Illizi

ALIAS = {   # survey spellings -> religiondots `name` (folded)
    "algeris": "algiers", "alger": "algiers", "bouria": "bouira", "bejia": "bejaia",
    "beskra": "biskra", "tbessa": "tebessa", "stif": "setif", "bashar": "bechar",
    "bachar": "bechar", "musker": "mascara", "alwad": "eloued", "masila": "msila",
    "taref": "eltarf", "eltaref": "eltarf", "oeb": "oumelbouaghi", "oumelboughi": "oumelbouaghi",
    "bba": "bordjbouarreridj", "borjbouarreridj": "bordjbouarreridj",
    "tamenrasset": "tamanrasset", "tamanghasset": "tamanrasset", "echlef": "chlef",
    "jelfa": "djelfa", "tiziouzou": "tiziouzou", "ouergla": "ouargla",
    "sidibelabbes": "sidibelabbes", "aintecmouchent": "aintemouchent", "tipasa": "tipaza",
    "naama": "naama",
}

# 1966 census, Berber mother tongue as % of the daira's population (Nesson 1994, pp. 94-97).
# (place as GeoNames spells the daira seat, %, what the article says). A placement guide only:
# inside a wilaya, Berber dots lean towards the places that were Berber-speaking in 1966.
ANCHORS_1966 = [
    ("Tizi Ouzou", 90, ">85%"), ("L'Arbaa Nait Irathen", 90, ">85%"), ("Azazga", 90, ">85%"),
    ("Draa el Mizan", 90, ">85%"), ("Bouira", 90, ">85%"), ("Bejaia", 90, ">85%"),
    ("Akbou", 90, ">85%"), ("Sidi Aich", 90, ">85%"),
    ("Bougaa", 70.2, ""), ("Kherrata", 38.6, ""), ("Bordj Menaiel", 49.7, ""),
    ("Setif", 4.2, ""), ("Bordj Bou Arreridj", 18, ""), ("Sour el Ghozlane", 2.7, ""),
    ("Lakhdaria", 21.4, ""), ("Dar el Beida", 11.5, ""), ("Blida", 7.6, ""),
    ("Algiers", 30, "Alger-ville"), ("Cheraga", 31, "Alger-Sahel"),
    ("Arris", 92.3, ""), ("Khenchela", 71.7, ""), ("Merouana", 75.1, ""), ("Batna", 52, ""),
    ("Ain Mlila", 33, "close to 33%"), ("Ain Beida", 33, "close to 33%"),
    ("Cherchell", 70.4, ""), ("Djanet", 74.1, ""), ("Tamanrasset", 62.9, ""),
    ("Ghardaia", 50, ""),
    # "null or very weak, a few hundred or a few thousand" (p. 95): 2%
    ("Medea", 2, "weak"), ("Ksar el Boukhari", 2, "weak"), ("Ain Oussera", 2, "weak"),
    ("Theniet el Had", 2, "weak"), ("Tissemsilt", 2, "weak"), ("Tiaret", 2, "weak"),
    ("Frenda", 2, "weak"), ("Chlef", 2, "weak"), ("Sidi Bel Abbes", 2, "weak"),
    ("Maghnia", 2, "weak"), ("Ain Sefra", 2, "weak"),
    # the East's other dairas: Berbers "in the dairas with big centres or ports" only
    # (Constantine 23,008, Djidjelli 2,231, Skikda 3,511, Annaba 3,886; p. 95); the article
    # gives counts, not shares, so these are rough: ports 2%, Constantine 5%
    ("Jijel", 2, "2,231"), ("Skikda", 2, "3,511"), ("Annaba", 2, "3,886"),
    ("Constantine", 5, "23,008, share not given"),
    # 1966 seats the article does not name in its Berber list, read as weak like the rest
    ("El Milia", 2, "not named"), ("Collo", 2, "not named"), ("Guelma", 2, "not named"),
    ("Souk Ahras", 2, "not named"), ("Mila", 2, "not named"), ("El Eulma", 2, "not named"),
    ("M'Sila", 2, "not named"), ("Bou Saada", 2, "not named"), ("Barika", 2, "not named"),
    ("Biskra", 2, "not named"), ("El Oued", 2, "not named"), ("Djelfa", 2, "not named"),
    ("Laghouat", 2, "not named"), ("Tlemcen", 2, "not named"), ("Mostaganem", 2, "not named"),
    ("Mascara", 2, "not named"), ("Relizane", 2, "not named"), ("Saida", 2, "not named"),
    ("El Bayadh", 2, "not named"), ("Adrar", 2, "not named"), ("Tindouf", 2, "not named"),
    ("Ain Temouchent", 2, "not named"), ("Oran", 5, "25,053, share not given"),
    ("Touggourt", 2, "4,948"), ("Bechar", 2, "2,754"),
    # Saharan dairas named with counts only; rough shares for placement
    ("Ouargla", 10, "8,978, share not given"), ("Timimoun", 30, "16,664, share not given"),
    ("Tebessa", 10, "18,452, share not given"), ("El Aouinet", 20, "18,876, share not given"),
]
BACKGROUND = 2.0       # % far from every anchor: the 1966 rest of the country
BACKGROUND_KM = 40.0   # an anchor this far away weighs the same as the background


def say(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        raise SystemExit(msg)


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().casefold()
    s = re.sub(r"^\d+\.\s*", "", s.strip())
    return re.sub(r"[^a-z]", "", s)


def _read(path, cols=None, metadataonly=False):
    import pyreadstat
    kw = dict(metadataonly=True) if metadataonly else dict(usecols=cols)
    try:
        return pyreadstat.read_sav(str(path), **kw)
    except Exception:  # noqa: BLE001  some releases are not valid UTF-8
        return pyreadstat.read_sav(str(path), encoding="LATIN1", **kw)


def _algeria(df, meta, col):
    lab = meta.variable_value_labels.get(col, {})
    return df[df[col].map(lab).astype(str).map(fold).str.endswith("algeria")]


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    out = []
    for wave, name, eth in ETH_WAVES:
        _, meta = _read(ARB / name, metadataonly=True)
        up = {c.upper(): c for c in meta.column_names}
        lab = str(meta.column_names_to_labels.get(up[eth], "")).casefold()
        say("ethnicity" in lab or "consider yourself" in lab, f"{wave} {eth} is the ethnic group ({lab})")
        df, meta = _read(ARB / name, [up["COUNTRY"], up["Q1"], up[eth], up["WT"]])
        s = _algeria(df, meta, up["COUNTRY"])
        vl = meta.variable_value_labels
        w = pd.to_numeric(s[up["WT"]], errors="coerce")
        o = pd.DataFrame({"wave": wave, "wilaya": s[up["Q1"]].map(vl[up["Q1"]]),
                          "eth": s[up[eth]].map(vl[up[eth]]), "w": w / w.mean()})
        say(o[["wilaya", "eth"]].notna().all().all(), f"{wave}: {len(o):,} Algerians, every code labelled")
        out.append(o)
    pd.concat(out).to_csv(ETH_X, index=False)
    print(f"wrote {ETH_X}")

    out = []
    for wave, folder, name, q, wil, wt in LANG_WAVES:
        _, meta = _read(folder / name, metadataonly=True)
        up = {c.upper(): c for c in meta.column_names}
        lab = str(meta.column_names_to_labels.get(up[q.upper()], "")).casefold()
        say("language" in lab and ("first" in lab or "respondent" in lab or "home" in lab),
            f"{wave} {q} is the first/home language ({lab})")
        df, meta = _read(folder / name, [up["COUNTRY"], up[wil.upper()], up[q.upper()], up[wt.upper()]])
        s = _algeria(df, meta, up["COUNTRY"])
        vl = meta.variable_value_labels
        wc = up[wil.upper()]
        w = pd.to_numeric(s[up[wt.upper()]], errors="coerce")
        o = pd.DataFrame({"wave": wave, "wilaya": s[wc].map(vl[wc]) if wc in vl else s[wc],
                          "lang": s[up[q.upper()]].map(vl[up[q.upper()]]), "w": w / w.mean()})
        say(len(o) >= 1199 and o["lang"].notna().all(), f"{wave}: {len(o):,} Algerians, every answer labelled")
        out.append(o)
    pd.concat(out).to_csv(LANG_X, index=False)
    print(f"wrote {LANG_X}")


def _units():
    lut = pd.read_csv(LOOKUP, dtype=str)
    lut["pop"] = lut["pop"].astype(int)
    say(len(lut) == 48 and lut["pop"].sum() == RGPH_2008, f"48 wilayas, RGPH 2008 {RGPH_2008:,}")
    return lut


def _join(labels, lut):
    key = {fold(n): g for n, g in zip(lut["name"], lut["geo_id"])}
    m = {}
    for lab in labels:
        f = fold(lab)
        m[lab] = key.get(ALIAS.get(f, f))
    return m


def _lang_class(s):
    f = s.astype(str).map(fold)
    return np.select([f.str.contains("amazi|berber|tamazi"), f.str.contains("french"),
                      f.str.contains("arabic")], ["berber", "french", "arabic"], "other")


def shares():
    lut = _units()
    e = pd.read_csv(ETH_X)
    jm = _join(e["wilaya"].unique(), lut)
    bad = sorted(k for k, v in jm.items() if v is None and fold(k) != "dontknow")
    say(not bad, f"every ethnicity-wave wilaya label joins ({bad})")
    e["geo_id"] = e["wilaya"].map(jm)
    f = e["eth"].map(fold)
    e["cat"] = np.select([f == "arab", f.str.startswith("amazigh"), f == "tourag", f == "other"],
                         ["Arab", "Amazigh", "Tourag", "Other"], "drop")
    print("  ethnic answers by wave:\n" + e.groupby(["wave", "cat"])["w"].sum().round(0)
          .unstack(fill_value=0).to_string())
    print("  Tourag answers by wilaya:", e[e["cat"] == "Tourag"].groupby("wilaya").size().to_dict())
    e = e[(e["cat"] != "drop") & e["geo_id"].notna()]
    say(e["geo_id"].nunique() == 48, f"ethnicity pool reaches {e['geo_id'].nunique()} of 48 wilayas")

    # wave check: Amazigh share by wilaya, VI against VII
    t = e.assign(am=(e["cat"] == "Amazigh") * e["w"]).groupby(["geo_id", "wave"]).agg(
        am=("am", "sum"), n=("w", "sum")).unstack()
    both = t.dropna()
    both = both[(both[("n", "VI")] >= 15) & (both[("n", "VII")] >= 15)]
    r = np.corrcoef(both[("am", "VI")] / both[("n", "VI")], both[("am", "VII")] / both[("n", "VII")])[0, 1]
    print(f"  wave check: Amazigh share VI vs VII across {len(both)} wilayas with 15+ in each, r = {r:+.3f}")
    for wv in ("VI", "VII"):
        sub = e[e["wave"] == wv]
        print(f"    {wv}: Amazigh {sub.loc[sub['cat'] == 'Amazigh', 'w'].sum() / sub['w'].sum():.1%} "
              f"of {len(sub):,}")

    cats = ["Arab", "Amazigh", "Tourag", "Other"]
    pw = e.pivot_table(index="geo_id", columns="cat", values="w", aggfunc="sum", fill_value=0)
    pw = pw.reindex(columns=cats, fill_value=0)
    nat = pw.sum() / pw.values.sum()
    n = pw.sum(axis=1)
    # prior, worth K respondents: the wilaya's 1966 Berber-speaking share for Amazigh, the rest
    # split as the nation's non-Amazigh answers are. It matters only where the pool is thin
    # (Tindouf 2, Illizi 2, Tamanrasset 3, El Bayadh 4 respondents), and there a wilaya's own
    # 1966 share is a better guess than a national one made mostly of Kabyles.
    p66 = wilaya_1966(lut).reindex(pw.index)
    rest = nat.drop("Amazigh") / nat.drop("Amazigh").sum()
    prior = pd.DataFrame({c: (1 - p66) * rest[c] for c in rest.index}).assign(Amazigh=p66)[cats]
    sh = (pw + K * prior).div(n + K, axis=0)

    # French, from the first/home-language waves that carry a usable wilaya
    l = pd.read_csv(LANG_X)
    l["cls"] = _lang_class(l["lang"])
    l = l[l["cls"] != "other"]
    natl = l.groupby(["wave", "cls"])["w"].sum().unstack(fill_value=0)
    print("  first/home language, national, by wave:\n" + (natl.div(natl.sum(axis=1), axis=0) * 100)
          .round(1).to_string())
    l = l[~l["wave"].isin(NO_WILAYA)]
    jl = _join(l["wilaya"].unique(), lut)
    bad = sorted(k for k, v in jl.items() if v is None)
    say(not bad, f"every language-wave wilaya label joins ({bad})")
    l["geo_id"] = l["wilaya"].map(jl)
    fr = l.assign(fr=(l["cls"] == "french") * l["w"]).groupby("geo_id").agg(fr=("fr", "sum"), n=("w", "sum"))
    fnat = fr["fr"].sum() / fr["n"].sum()
    fr = fr.reindex(lut["geo_id"]).fillna(0)
    fsh = (fr["fr"] + K * fnat) / (fr["n"] + K)
    print(f"  French first language, pooled: {fnat:.2%} ({int((l['cls'] == 'french').sum())} answers)")

    # the language waves' own Berber share, for the record's comparison
    lb = l.assign(b=(l["cls"] == "berber") * l["w"]).groupby("geo_id").agg(b=("b", "sum"), n=("w", "sum"))
    return lut, sh, n, fsh, lb, nat


def build():
    lut, sh, n, fsh, lb, nat = shares()
    rows = []
    for _, u in lut.iterrows():
        g, pop = u["geo_id"], u["pop"]
        s = sh.loc[g] * (1 - fsh[g])
        if g in KABYLE:
            am = "Amazigh (in a Kabyle-speaking wilaya)"
        elif g in CHAOUI:
            am = "Amazigh (in a Chaoui-speaking wilaya)"
        elif g in MZAB:
            am = "Amazigh (in Ghardaia)"
        elif g in TUAREG:
            am = "Amazigh (in a Tuareg wilaya)"
        else:
            am = "Amazigh"
        # wave VII's 22 "Tourag" answers are in Djelfa (8), Laghouat (3) and ten northern
        # wilayas, none in the Tuareg south: drawn with "Other" (sources/dz.md section 3)
        parts = {"Arab": s["Arab"], am: s["Amazigh"],
                 "Other ethnic group (incl. Tourag)": s["Other"] + s["Tourag"],
                 "French (first language)": fsh[g]}
        exact = {k: v * pop for k, v in parts.items()}
        fl = {k: int(np.floor(v)) for k, v in exact.items()}
        short = pop - sum(fl.values())
        for k in sorted(exact, key=lambda k: exact[k] - fl[k], reverse=True)[:short]:
            fl[k] += 1
        for k, c in fl.items():
            rows.append(dict(geo_id=g, geo_level="wilaya", geo_name=u["name"], source_category=k,
                             count=c, tier="modelled", source_id=SOURCE_ID, year="2020-2022",
                             note=f"share {parts[k]:.4f}; {n.get(g, 0):.0f} respondents (ethnic group)"))
    df = pd.DataFrame(rows)
    say(int(df["count"].sum()) == RGPH_2008, f"drawn total {int(df['count'].sum()):,} = RGPH 2008")
    say(df.groupby("geo_id")["count"].sum().eq(lut.set_index("geo_id")["pop"]).all(),
        "every wilaya sums to its RGPH 2008 population")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")
    tot = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print((tot / tot.sum() * 100).round(2).to_string())


# ---- the 1966 surface: placement inside a wilaya, and a check on the counts ----

def anchors():
    import zipfile
    with zipfile.ZipFile(GEONAMES) as z, z.open("DZ.txt") as fh:
        g = pd.read_csv(fh, sep="\t", header=None, quoting=3, low_memory=False,
                        usecols=[1, 2, 3, 4, 5, 6, 7, 14],
                        names=["name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "pop"])
    g = g[g["fclass"] == "P"]
    g["key"] = g["ascii"].map(fold)
    alt = g["alt"].fillna("").str.split(",")
    out = []
    for place, pct, why in ANCHORS_1966:
        k = fold(place)
        hit = g[g["key"] == k]
        if hit.empty:
            hit = g[alt.map(lambda a: k in {fold(x) for x in a})]
        if hit.empty:
            raise SystemExit(f"1966 anchor {place!r} not in GeoNames")
        h = hit.sort_values("pop", ascending=False).iloc[0]
        out.append(dict(place=place, pct=pct, why=why, lat=h["lat"], lon=h["lon"],
                        geonames=h["name"], gn_pop=h["pop"]))
    return pd.DataFrame(out)


def surface(lat, lon, a=None):
    """1966 Berber-speaking % at each point: inverse-square-distance mean of the anchors, with a
    background of BACKGROUND % weighing as an anchor BACKGROUND_KM away."""
    a = anchors() if a is None else a
    lat = np.asarray(lat, dtype=float)[:, None]
    lon = np.asarray(lon, dtype=float)[:, None]
    dy = (lat - a["lat"].to_numpy()[None, :]) * 111.0
    dx = (lon - a["lon"].to_numpy()[None, :]) * 111.0 * np.cos(np.radians(lat))
    w = 1.0 / np.maximum(dx * dx + dy * dy, 4.0)
    w0 = 1.0 / BACKGROUND_KM ** 2
    return (w @ a["pct"].to_numpy() + w0 * BACKGROUND) / (w.sum(axis=1) + w0)


def wilaya_1966(lut):
    """Each wilaya's 1966 Berber-speaking share, as the surface reads on today's Kontur
    population (religiondots' hexes, read-only). A fraction, indexed by geo_id."""
    import geopandas as gpd
    hx = gpd.read_file(RD_GEO / "dz" / "dz_hexes.gpkg")
    c = hx.geometry.to_crs(3857).centroid.to_crs(4326)
    s = surface(c.y.to_numpy(), c.x.to_numpy()) / 100
    w = hx.assign(b=hx["pop"] * s).groupby("unit").agg(b=("b", "sum"), p=("pop", "sum"))
    u2g = dict(zip(lut["unit"], lut["geo_id"]))
    out = (w["b"] / w["p"]).rename(index=u2g)
    if set(out.index) != set(lut["geo_id"]):
        raise SystemExit("dz_hexes.gpkg units do not match dz_lookup.csv")
    return out


def check1966():
    a = anchors()
    print(a[["place", "pct", "geonames", "lat", "lon", "gn_pop"]].to_string())
    lut, sh, n, fsh, lb, nat = shares()
    t = pd.DataFrame({"name": lut.set_index("geo_id")["name"]})
    t["1966_surface"] = wilaya_1966(lut).reindex(t.index) * 100
    t["ethnic_VI_VII"] = (sh["Amazigh"] + sh["Tourag"]) * 100
    t["n_eth"] = n
    t["lang_waves"] = (lb["b"] / lb["n"] * 100).reindex(t.index)
    t["n_lang"] = lb["n"].reindex(t.index)
    pop = lut.set_index("geo_id")["pop"]
    print(t.round(1).to_string())
    for col in ("1966_surface", "ethnic_VI_VII", "lang_waves"):
        v = t[col].fillna(0)
        print(f"  national, on 2008 populations: {col} {(v * pop).sum() / pop.sum():.1f}%")
    for col in ("ethnic_VI_VII", "lang_waves"):
        ok = t[col].notna()
        print(f"  r(1966 surface, {col}) across {ok.sum()} wilayas = "
              f"{np.corrcoef(t.loc[ok, '1966_surface'], t.loc[ok, col])[0, 1]:+.3f}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    if "--check1966" in sys.argv:
        check1966()
    else:
        build()
