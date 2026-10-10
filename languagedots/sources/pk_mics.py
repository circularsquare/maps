"""Pakistan MICS microdata: Gilgit-Baltistan MICS5 2016-17 and Khyber Pakhtunkhwa MICS6 2019
-> data/normalized/pk_mics_gb2016.csv (GB district shares, read by sources/pk_north.py)
-> data/normalized/pk_kp_mics.csv      (KP tehsils: the census's OTHERS split, read by countries/pk.py)

    python sources/pk_mics.py        (after sources/pk_t11.py; before sources/pk_north.py)

SOURCE. UNICEF MICS SPSS files (Anita's account, 2026-10-09) in data/raw/pk/mics_gb2016/ and
data/raw/pk/mics_kp2019/, gitignored, read in place (research use, no redistribution). Persons are
hl.sav members merged to hh.sav on HH1, HH2 and weighted by hhweight; households with hhweight 0
were not interviewed. The record is sources/pk.md §0.

GILGIT-BALTISTAN (MICS5 2016-17, 6,460 households, 6,213 interviewed, 323 clusters, HH7 = the
ten census districts; "Baltistan" is Skardu). Item HC1B, mother tongue of the household head
(Urdu, Shina, Balti, Burushaski, Khowar, Wakhi, other), read as every member's. HC1C (language
usually spoken at home) agrees in 96.6% of households and drifts to Urdu (5 heads, 40 homes),
so HC1B, which is also the census's question. There is no finer "other" item: the files hold no
other language variable, so "other" cannot be split further. CHECK: the unweighted household
counts reproduce, cell for cell, the district table printed in the Pamir Times (2023-12-23) that
sources/pk_north.py used until now (Urdu folded into its "other languages"). Written: weighted
shares of persons and of households, and unweighted households, per district and language.

KHYBER PAKHTUNKHWA (MICS6 2019, 23,740 households, 23,501 interviewed, 1,187 clusters, HH7 = 32
districts of 2019). The census (Table 11, by tehsil) stays: MICS only splits its OTHERS.
- Items. HC1B (head's language) has Pashto, Hindko, Saraiki, Urdu, "Kohistani/Gujri" (one code)
  and other; Chitral's heads are 88% "other". HH16 (respondent's native language) adds Khowar and
  Shina. Within HC1B-"other" households HH16 is 764 Khowar and 189 Shina of 1,046, so HC1B is used
  with its "other" split by HH16 into Khowar / Shina / still other (the Iraq method,
  sources/iq_mics6.py). HH16 alone was not used: 95 Kohistani/Gujari-headed households (Swat,
  Shangla, Upper Dir, Kohistan) answer HH16 Pashto, against 35 the other way.
- Split. Per MICS district d (census districts crosswalked to it, all asserted): census population
  P, OTHERS O, KOHIOSTANI K. MICS persons' shares: Khowar k, Kohistani/Gujari g, still-other o.
  Khowar = k P; the Kohistani/Gujari the census did not already name Kohistani = max(0, g P - K);
  other = o P. If those add to more than O they are scaled down to O (f = O / sum); otherwise
  they are taken whole and the rest of O stays OTHERS. Each tehsil's OTHERS is split in its MICS
  district's proportions, largest remainder.
- The Kohistani/Gujari residual is named by place (spec §3, place-dependent label): in Hazara
  (Abbottabad, Haripur, Mansehra, Batagram, Torghar) it is Gujari, the one language of the code
  spoken there apart from the Kohistani the census names; elsewhere (Swat's Behrain, Dir Kohistan,
  Shangla) it also holds Torwali, Gawri and other "Kohistani" the census filed under OTHERS. No
  data separates them, so (Anita, 2026-10-09: split rather than lump, from knowledge) it is named
  by tehsil (BY_PLACE): Behrain shared evenly between Torwali and Gawri (Joshua Project, asked
  for, has no Pakistan row for either), Sharingal (Dir Kohistan) Gawri, Bisham Indus Kohistani, Upper Chitral left on
  Indo-Aryan, and every other tehsil Gujari (lower Swat, Buner, Dir, Malakand, lower Chitral's
  Drosh and the plains, where the Gujars are the people of that code).
- MICS Shina in Kohistan (17%) is above the census's (6.5%), and its Kohistani/Gujari (71%) below
  the census's Kohistani (88%); the census's named counts are kept, MICS changes only OTHERS.
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
RAW = HERE / "data" / "raw" / "pk"
NORM = HERE / "data" / "normalized"
OUT_GB = NORM / "pk_mics_gb2016.csv"
OUT_KP = NORM / "pk_kp_mics.csv"

# ------------------------------------------------------------------ Gilgit-Baltistan
GB_N = (6460, 6213, 46276)                      # households, interviewed, members
GB_DISTRICT = {"Astore": "Astore", "Baltistan": "Skardu", "Diamir": "Diamer",
               "Ghanche": "Ghanche", "Ghizer": "Ghizer", "Gilgit": "Gilgit", "Hunza": "Hunza",
               "Kharmang": "Kharmang", "Nagar": "Nagar", "Shigar": "Shigar"}
GB_LABEL = {"Urdu": "Urdu", "Sheena": "Shina", "Balti": "Balti", "Brushaski": "Burushaski",
            "Khawar": "Khowar", "Wakhi": "Wakhi", "Other language": "Other"}

# ------------------------------------------------------------------ Khyber Pakhtunkhwa
KP_N = (23740, 23501, 178631)
# census 2023 district (religiondots / pk_t11 id) -> MICS 2019 HH7. Chitral (split 2018) and
# Kohistan (split 2014-17) are one district each in MICS.
KP_DISTRICT = {
    "abbottabad-district": "Abbotabad", "bajaur-district": "Bajor", "bannu-district": "Bannu",
    "batagram-district": "Batagram", "buner-district": "Buner", "charsadda-district": "Charsadda",
    "dera-ismail-khan-district": "Dera Ismail Khan", "hangu-district": "Hangu",
    "haripur-district": "Hari Pur", "karak-district": "Karak", "khyber-district": "Khyber",
    "kohat-district": "Kohat", "kolai-palas-kohistan-district": "Kohistan",
    "lower-kohistan-district": "Kohistan", "upper-kohistan-district": "Kohistan",
    "kurram-district": "Kuram", "lakki-marwat-district": "Laki Marwat",
    "lower-chitral-district": "Chitral", "upper-chitral-district": "Chitral",
    "lower-dir-district": "Lower Dir", "malakand-protected-area": "Malakand",
    "mansehra-district": "Mansehra", "mardan-district": "Mardan", "mohmand-district": "Mohmind",
    "north-waziristan-district": "North Waziristan", "nowshera-district": "Nowshehra",
    "orakzai-district": "Orakzai", "peshawar-district": "Peshawar", "shangla-district": "Shangla",
    "south-waziristan-district": "South Waziristan", "swabi-district": "Swabi",
    "swat-district": "Swat", "tank-district": "Tank", "torghar-district": "Torghar",
    "upper-dir-district": "Upper Dir",
}
# where MICS's Kohistani/Gujari, less the census's Kohistani, is Gujari (see the docstring)
GUJARI_DISTRICTS = {"Abbotabad", "Hari Pur", "Mansehra", "Batagram", "Torghar"}
KP_HEAD = {"PUSHTO": "Pashto", "HINDKO": "Hindko", "SARAIKI": "Saraiki", "URDU": "Urdu",
           "KOHISTANI/GUJRI": "Kohistani/Gujari", "ENGLISH": "Other", "OTHER LANGUAGE": "Other"}
SPLIT_OTHER = {"Khowar(Chitrali)": "Khowar", "Shena": "Shina"}   # HH16 inside HC1B "other"
LAB_KHOWAR, LAB_GUJARI, LAB_KG = "MICS: Khowar", "MICS: Gujari", "MICS: Kohistani or Gujari"
# Outside Hazara, MICS's Kohistani/Gujari residual named by place, from knowledge (Anita,
# 2026-10-09: split rather than lump; decide from knowledge where no data separates them).
# Keyed by tehsil id without the province. Not listed: Gujari, the Gujars of lower Swat, Buner,
# Dir, Malakand and the plains being the one people of that code living there.
PLACE = "MICS Kohistani/Gujari by place: "
BY_PLACE = {
    # Swat Kohistan: Torwali (Bahrain, Chail) and Gawri (Kalam, Utror, Ushu), shared evenly, with
    # Kontur nearness to each Glottolog point as Gawri's floor (behrain_shares)
    "swat-district/behrain-tehsil": "behrain",
    # Dir Kohistan (Kumrat, Thal, Lamuti, Kalkot): Gawri. Kalkoti (kalk1245, one village, a few
    # thousand) is not split out: nearness to its point would hand it most of Sharingal.
    "upper-dir-district/sharingal-sub-division": {PLACE + "Gawri": 1.0},
    # Bisham, on the Indus beside Lower Kohistan (Pattan): Indus Kohistani
    "shangla-district/bisham-tehsil": {PLACE + "Kohistani": 1.0},
    # Upper Chitral (Mastuj): few Gujars and no Kohistani language; left unnamed
    "upper-chitral-district/mastuj-sub-division": {LAB_KG: 1.0},
}
GLOTTOLOG_POINTS = {PLACE + "Torwali": (35.3101, 72.5316),   # torw1241
                    PLACE + "Gawri": (35.5303, 72.5738)}     # kala1373


def behrain_shares():
    """Behrain's Kohistani/Gujari share between Torwali and Gawri: an even split, floored so that
    Gawri never falls below what nearness alone gives the Kalam end of the valley.

    The supervisor asked (2026-10-09) for Joshua Project's speaker estimates (trw, gwc), as
    id_papua.py uses them. The PGIC file (data/raw/pg/joshuaproject_pgic.csv, 2026-10-05, 773
    Pakistan rows) has no Pakistan row for either code: it files Swat and Dir Kohistan under
    "Pashtun Kohistani" (142,000) and "Pashtun Dir Kohistan" (137,000), primary language Northern
    Pashto; its only Gawri row is Afghanistan's "Garwi, Kohistani" (2,200). So the fallback the
    supervisor named earlier, an even split, is used. The floor is nearness: each Kontur hex of
    the placement layer goes to the nearer Glottolog point (cos-latitude scaled), people summed.
    That reading alone gave Torwali 85.8%, since Bahrain and Chail are lower and denser than Kalam
    (sources/pk.md §0.2)."""
    import math
    import geopandas as gpd
    unit = "PK23-khyber-pakhtunkhwa/swat-district/behrain-tehsil"
    g = gpd.read_file(HERE / "data" / "geo" / "pk" / "pk_hexes.gpkg")
    g = g[g["unit"] == unit]
    if g.empty:
        raise SystemExit(f"{unit} is not its own unit in pk_hexes.gpkg")
    c = g.geometry.representative_point()
    k = math.cos(math.radians(35.4))
    near = {lab: ((c.y - la) ** 2 + ((c.x - lo) * k) ** 2) for lab, (la, lo) in
            GLOTTOLOG_POINTS.items()}
    tor = near[PLACE + "Torwali"] <= near[PLACE + "Gawri"]
    s = g["pop"][tor].sum() / g["pop"].sum()
    gawri = max(0.5, 1 - s)
    print(f"  Behrain: {len(g):,} hexes; Kontur people nearer Torwali's point {s * 100:.1f}%, "
          f"Gawri's {(1 - s) * 100:.1f}% (the floor); drawn Torwali {(1 - gawri) * 100:.1f}%, "
          f"Gawri {gawri * 100:.1f}%")
    return {PLACE + "Torwali": 1 - gawri, PLACE + "Gawri": gawri}


def read(tag, hh_cols, n):
    import pyreadstat
    hh, _ = pyreadstat.read_sav(str(RAW / tag / "hh.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "hhweight"] + hh_cols)
    hl, _ = pyreadstat.read_sav(str(RAW / tag / "hl.sav"), usecols=["HH1", "HH2", "HL1"])
    got = (len(hh), int((hh.hhweight > 0).sum()), len(hl))
    if got != n:
        raise SystemExit(f"{tag}: households / interviewed / members {got}, expected {n}")
    p = hl.merge(hh, on=["HH1", "HH2"], how="left", indicator=True)
    if (p["_merge"] != "both").any():
        raise SystemExit(f"{tag}: hl.sav members with no household")
    for c in hh_cols:
        if hh[c].dtype.name == "category":
            hh[c] = hh[c].astype(str)
            p[c] = p[c].astype(str)
    return hh, p[p.hhweight > 0].copy()


def miss(clusters, p):
    """chance that a group confined to a share p of a unit's people is in none of its clusters"""
    return (1 - p) ** clusters


def gilgit_baltistan():
    import pk_north
    hh, p = read("mics_gb2016", ["HH7", "HC1B", "HC1C"], GB_N)
    i = hh[hh.hhweight > 0].copy()
    if i["HC1B"].isin(["nan", "Missing"]).any() or set(i.HH7) != set(GB_DISTRICT):
        raise SystemExit("GB: an interviewed household with no HC1B, or an unknown district")
    clusters = hh.groupby("HH7").HH1.nunique()
    print(f"Gilgit-Baltistan MICS5 2016-17: {len(i):,} interviewed households, "
          f"{len(p):,} members, {hh.HH1.nunique()} clusters ({clusters.min()}-{clusters.max()} "
          f"per district); a group in 2% of a district is missed by all "
          f"{clusters.min()} clusters {miss(clusters.min(), 0.02) * 100:.0f}% of the time, 5% "
          f"{miss(clusters.min(), 0.05) * 100:.0f}%")

    agree = (i.HC1B == i.HC1C).mean()
    print(f"  HC1B (head's mother tongue) = HC1C (home language) in {agree * 100:.1f}% of "
          f"households; Urdu {int((i.HC1B == 'Urdu').sum())} heads, "
          f"{int((i.HC1C == 'Urdu').sum())} homes")
    if agree < 0.95:
        raise SystemExit("GB: HC1B and HC1C disagree in more than 5% of households")

    # check: the unweighted counts are the Pamir Times table (pk_north.MICS17), Urdu in "other"
    i["lab"] = i.HC1B.map(GB_LABEL)
    p["lab"] = p.HC1B.map(GB_LABEL)
    if i.lab.isna().any():
        raise SystemExit(f"GB: HC1B answers with no label {sorted(set(i.HC1B) - set(GB_LABEL))}")
    n = pd.crosstab(i.HH7.map(GB_DISTRICT), i.lab)
    for d, (row, tot) in pk_north.MICS17.items():
        mine = [int(n.loc[d].get(c, 0)) for c in pk_north.MICS17_COLS[:-1]]
        mine.append(int(n.loc[d].get("Other", 0) + n.loc[d].get("Urdu", 0)))
        if mine != row:
            raise SystemExit(f"GB {d}: microdata {mine}, Pamir Times {row}")
    print("  unweighted HC1B households = the Pamir Times district table, all 60 cells")

    wp = p.groupby([p.HH7.map(GB_DISTRICT), "lab"]).hhweight.sum()
    wh = i.groupby([i.HH7.map(GB_DISTRICT), "lab"]).hhweight.sum()
    rows = []
    for (d, lab), v in wp.items():
        rows.append(dict(district=d, label=lab,
                         persons_share=v / wp[d].sum(),
                         households_share=wh.get((d, lab), 0) / wh[d].sum(),
                         households_n=int(n.loc[d].get(lab, 0)),
                         clusters=int(clusters[next(k for k, x in GB_DISTRICT.items() if x == d)])))
    df = pd.DataFrame(rows)
    s = df.pivot(index="district", columns="label", values="persons_share").fillna(0) * 100
    print("  weighted % of persons by district (HC1B):")
    print(s.round(1).to_string())
    u = pd.crosstab(i.HH7.map(GB_DISTRICT), i.lab, normalize="index") * 100
    print("  largest move from the article's unweighted household shares, points: "
          + ", ".join(f"{d} {c} {u.loc[d, c]:.1f} -> {s.loc[d, c]:.1f}"
                      for d, c in (s - u.reindex_like(s).fillna(0)).abs().stack()
                      .sort_values(ascending=False).index[:6]))
    gp = p.groupby("lab").hhweight.sum()
    gh = i.groupby("lab").hhweight.sum()
    print("  GB-wide, weighted % of persons / households: " + ", ".join(
        f"{k} {gp[k] / gp.sum() * 100:.1f} / {gh[k] / gh.sum() * 100:.1f}"
        for k in gp.sort_values(ascending=False).index))
    return df


def kp_shares():
    hh, p = read("mics_kp2019", ["HH7", "division", "HC1B", "HH15", "HH16"], KP_N)
    i = hh[hh.hhweight > 0]
    print(f"\nKhyber Pakhtunkhwa MICS6 2019: {len(i):,} interviewed households, {len(p):,} "
          f"members, {hh.HH1.nunique()} clusters")
    ct = pd.crosstab(i.HC1B, i.HH16)
    o = ct.loc["OTHER LANGUAGE"]
    print(f"  HC1B 'other' households by HH16: Khowar {o['Khowar(Chitrali)']}, Shina "
          f"{o['Shena']}, other {o['OTHER LANGUAGE']}, of {o.sum()}")
    print(f"  Kohistani/Gujari heads answering HH16 Pashto {ct.loc['KOHISTANI/GUJRI', 'PUSHTO']}, "
          f"Pashto heads answering Kohistani/Gujari {ct.loc['PUSHTO', 'KOHISTANI / GUJARI']}")
    for a, b in (("PUSHTO", "PUSHTO"), ("HINDKO", "HINDKO"), ("SARAIKI", "SARAIKI"),
                 ("KOHISTANI/GUJRI", "KOHISTANI / GUJARI")):
        if ct.loc[a, b] / ct.loc[a].sum() < 0.85:
            raise SystemExit(f"KP: HC1B {a} and HH16 agree in under 85% of households")
    ch = i[(i.HH7 == "Chitral") & (i.HC1B == "OTHER LANGUAGE")]
    if (ch.HH16 == "Khowar(Chitrali)").mean() < 0.9:
        raise SystemExit("KP: Chitral's HC1B-other households are not 90% Khowar by HH16")

    p = p[p.HC1B != "NO RESPONSE"].copy()
    lab = p.HC1B.map(KP_HEAD)
    if lab.isna().any():
        raise SystemExit(f"KP: HC1B answers with no label {sorted(set(p.HC1B) - set(KP_HEAD))}")
    oth = p.HC1B == "OTHER LANGUAGE"
    lab[oth] = p.HH16[oth].map(SPLIT_OTHER).fillna("Other")
    p["lab"] = lab
    w = p.groupby(["HH7", "lab"]).hhweight.sum().unstack(fill_value=0)
    w = w.div(w.sum(axis=1), axis=0)
    clusters = hh.groupby("HH7").HH1.nunique()
    division = hh.groupby("HH7")["division"].first()
    return w, clusters, division


def khyber_pakhtunkhwa():
    w, clusters, division = kp_shares()
    c = pd.read_csv(NORM / "pk.csv")
    c = c[c.geo_id.str.startswith("PK23-khyber")].copy()
    c["district"] = c.geo_id.str.split("/").str[1]
    if set(c.district) != set(KP_DISTRICT):
        raise SystemExit(f"KP census districts not in the crosswalk: "
                         f"{sorted(set(c.district) ^ set(KP_DISTRICT))}")
    if set(KP_DISTRICT.values()) != set(w.index):
        raise SystemExit(f"MICS districts not crosswalked: "
                         f"{sorted(set(w.index) ^ set(KP_DISTRICT.values()))}")
    if set(GUJARI_DISTRICTS) - {d for d in w.index if division[d] == "Hazara"}:
        raise SystemExit("GUJARI_DISTRICTS must be in MICS's Hazara division")
    c["mics"] = c.district.map(KP_DISTRICT)
    by = c.pivot_table(index="mics", columns="source_category", values="count", aggfunc="sum",
                       fill_value=0)
    pop = by.sum(axis=1)

    frac, report = {}, []
    for d in w.index:
        P, O, K = pop[d], by.loc[d, "OTHERS"], by.loc[d, "KOHIOSTANI"]
        kh = w.loc[d].get("Khowar", 0) * P
        kg = max(0.0, w.loc[d].get("Kohistani/Gujari", 0) * P - K)
        ot = w.loc[d].get("Other", 0) * P
        e = kh + kg + ot
        f = min(1.0, O / e) if e > 0 else 0.0
        frac[d] = (kh * f / O if O else 0, kg * f / O if O else 0)
        report.append(dict(mics=d, clusters=clusters[d], pop=P, others=O,
                           others_pct=O / P * 100,
                           mics_khowar=w.loc[d].get("Khowar", 0) * 100,
                           mics_kg=w.loc[d].get("Kohistani/Gujari", 0) * 100,
                           census_koh=K / P * 100, mics_other=w.loc[d].get("Other", 0) * 100,
                           khowar=kh * f, kg=kg * f,
                           kg_as="Gujari" if d in GUJARI_DISTRICTS else "Indo-Aryan"))
    r = pd.DataFrame(report).set_index("mics")
    print("  per MICS district: census OTHERS and Kohistani (% of census), MICS % of persons, "
          "and what OTHERS is split into:")
    show = r[r.others_pct >= 0.5].sort_values("others", ascending=False)
    print(show[["clusters", "others", "others_pct", "census_koh", "mics_khowar", "mics_kg",
                "mics_other", "khowar", "kg", "kg_as"]].round(1).to_string())

    behrain = behrain_shares()
    rows = []
    for _, t in c[c.source_category == "OTHERS"].iterrows():
        O = int(t["count"])
        fk, fg = frac[t["mics"]]
        raw = {LAB_KHOWAR: O * fk, "OTHERS": O * (1 - fk - fg)}
        if t["mics"] in GUJARI_DISTRICTS:
            place = {LAB_GUJARI: 1.0}
        else:
            place = BY_PLACE.get(t["geo_id"].split("/", 1)[1], {PLACE + "Gujari": 1.0})
            if place == "behrain":
                place = behrain
        for k, s in place.items():
            raw[k] = raw.get(k, 0) + O * fg * s
        cnt = {k: int(v) for k, v in raw.items()}
        for k in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[:O - sum(cnt.values())]:
            cnt[k] += 1
        assert sum(cnt.values()) == O
        for k, v in cnt.items():
            if v > 0:
                rows.append(dict(geo_id=t["geo_id"], geo_level="tehsil", geo_name=t["geo_name"],
                                 source_category=k, count=v,
                                 tier="measured" if k == "OTHERS" else "modelled",
                                 source_id="pk_kp_mics2019_others_split", year=2023,
                                 note=("census 2023 OTHERS" if k == "OTHERS" else
                                       f"census 2023 OTHERS x MICS 2019 {t['mics']} share"
                                       + (", named by place" if k.startswith(PLACE) else ""))))
    df = pd.DataFrame(rows)
    split = df.groupby("geo_id")["count"].sum()
    orig = c[c.source_category == "OTHERS"].set_index("geo_id")["count"]
    if not split.reindex(orig.index).fillna(0).astype(int).equals(orig.astype(int)):
        raise SystemExit("KP: split rows do not add back to each tehsil's OTHERS")
    tot = df.groupby("source_category")["count"].sum()
    print("  KP OTHERS {:,} -> ".format(int(orig.sum())) + ", ".join(
        f"{k} {v:,}" for k, v in tot.sort_values(ascending=False).items()))
    chit = c[c.mics == "Chitral"]
    kh = df[df.geo_id.isin(chit.geo_id) & (df.source_category == LAB_KHOWAR)]["count"].sum()
    print(f"  Chitral: Khowar {kh:,} of {int(chit['count'].sum()):,} people "
          f"({kh / chit['count'].sum() * 100:.1f}%)")
    if kh < 0.8 * chit[chit.source_category == "OTHERS"]["count"].sum():
        raise SystemExit("KP: Chitral's Khowar is under 80% of its census OTHERS")
    return df


def main():
    gb = gilgit_baltistan()
    kp = khyber_pakhtunkhwa()
    for df, out in ((gb, OUT_GB), (kp, OUT_KP)):
        tmp = out.with_suffix(".part")
        df.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
        tmp.replace(out)
        print(f"wrote {out.relative_to(HERE)} ({len(df)} rows)")


if __name__ == "__main__":
    main()
