"""Sweden: data/normalized/se.csv from SCB's population by birth country plus cited estimates.

    python sources/se_build.py          (after python sources/se_scb.py --fetch)

Sweden keeps no language statistics (sources/se.md). Every row is `derived`. The method is
Mikael Parkvall's, the only Swedish source that counts first-language speakers language by
language (Sveriges språk – vem talar vad och var?, Stockholm University 2009, data ~2006;
Sveriges språk i siffror, Språkrådet/Morfem 2016, data 2012):

  1. FIRST GENERATION. Foreign-born residents, 31 Dec 2025, per kommun and birth country
     (SCB FolkmRegFlandKCKM). SCB suppresses small kommun cells into "övriga födelseländer";
     `unsuppress` spreads each kommun's övriga back over the countries it hides (IPF). Each
     country goes on its main language (fr_build.COUNTRY_LANG, SE_OVERRIDES), a few split
     (`splits`, from Parkvall's per-language counts and origins). Parkvall credits the
     foreign-born with their origin language; exceptions in KEEP_FIRST (Finland-Swedes,
     adoptees from South Korea and Ethiopia).
  2. SECOND GENERATION. Sweden-born with two foreign-born parents keep the parents' language
     at 78%, with one foreign-born parent at 14% (Parkvall 2009 pp. 83-84, from Boyd 1985 and
     Nekby & Özcan 2006). Per kommun from UtlSvBakgFinCKM; their parents' countries nationally
     from FolkmForUrspCKMv2, spread over kommuner by IPF seeded on the kommun's foreign-born.
  3. NATIONAL MINORITY LANGUAGES (Meänkieli, Sami, Romani, Yiddish) from cited totals, placed
     on the kommuner their sources name, carved out of Swedish (and Romani partly out of the
     immigrant rows it came with). Finnish needs no layer: steps 1-2 make it.
  4. Swedish: everyone else.
"""
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "data", "normalized", "se.csv")

from fr_build import COUNTRY_LANG  # noqa: E402
import se_scb  # noqa: E402

SOURCE_ID = "se_scb2025_x_parkvall"
YEAR = 2025

# ---------------------------------------------------------------------------------------------
# birth country -> language (SCB writes ISO alpha-2: GB, GR; own codes for dissolved states)
# ---------------------------------------------------------------------------------------------
SE_OVERRIDES = {
    "GB": "English", "GR": "Greek", "CH": "German", "BE": "Dutch", "CA": "English",
    "YU": "Serbo-Croatian",   # born in Yugoslavia: Bosnians, Serbs, Croats alike (Parkvall: BKS)
    "CS": "Serbian",          # Serbia and Montenegro, 1992-2006
    "SU": "Russian", "QT": "Czech",
    "HK": "Cantonese",        # Parkvall 2009: Cantonese 14,000, Mandarin 4,500
    "BY": "Russian", "FR": "French", "BM": "English", "AI": "English", "GI": "English",
    "VG": "English",
}

# Parkvall 2009, p. 160 (speakers ~2006) and each language's share born in Sweden (profiles)
PV = {"Kurdish": (66000, .22), "Aramaic": (36000, .23), "Turkish": (34000, .32),
      "Persian": (59000, .15), "Azerbaijani": (4800, .15), "Turkmen": (2100, .23),
      "Korean": (1900, 0.0), "Amharic": (3900, .20), "Oromo": (1500, .20)}
# born-in-Sweden share not printed for Azerbaijani (Persian's used), Turkmen (Aramaic's,
# the other Iraqi minority), Oromo (Amharic's); Korean taken as all foreign-born (upper bound)


def splits(st06):
    """Language shares for IQ, SY (2006 cohort), TR, IR from Parkvall's 2006 foreign-born
    speakers over SCB's 2006 birth-country stocks. Printed. Assumptions, sources/se.md §2:
      * Turkish: 90% of foreign-born speakers from Turkey (Parkvall).
      * Persian includes Dari: the Afghanistan-born (2006) are taken off before Iran.
      * Azerbaijani: all from Iran ("mainly from Iran").
      * Aramaic: half from Iraq (Parkvall); the other half Turkey:Syria 3:2 ("then Turkey,
        then Syria"; the ratio is this build's).
      * Turkmen: all from Iraq ("mostly from Iraq").
      * Kurdish: Iran and Turkey take their remainders; what is left of Parkvall's total is
        shared between Iraq and Syria's 2006 cohort in proportion to their people not already
        placed (Parkvall: "first and foremost from Iraq").
      * Arabic: the rest of Iraq and Syria."""
    fb = {k: n * (1 - b) for k, (n, b) in PV.items()}
    IQ, SY, TR, IR, AF = (st06[c] for c in ("IQ", "SY", "TR", "IR", "AF"))
    tr_turk = fb["Turkish"] * .9
    ir_pers = fb["Persian"] - AF
    aram_iq, aram_tr, aram_sy = fb["Aramaic"] / 2, fb["Aramaic"] / 2 * .6, fb["Aramaic"] / 2 * .4
    ir_azer = fb["Azerbaijani"]
    ir_kurd = IR - ir_pers - ir_azer
    tr_kurd = TR - tr_turk - aram_tr
    kurd_left = fb["Kurdish"] - ir_kurd - tr_kurd
    iq_free = IQ - aram_iq - fb["Turkmen"]
    sy_free = SY - aram_sy
    k = kurd_left / (iq_free + sy_free)
    out = {
        "IQ": {"Arabic": iq_free * (1 - k), "Kurdish": iq_free * k,
               "Assyrian Neo-Aramaic": aram_iq, "Iraqi Turkmen": fb["Turkmen"]},
        "SY": {"Arabic": sy_free * (1 - k), "Kurdish": sy_free * k, "Turoyo": aram_sy},
        "TR": {"Turkish": tr_turk, "Kurdish": tr_kurd, "Turoyo": aram_tr},
        "IR": {"Persian": ir_pers, "Kurdish": ir_kurd, "Azerbaijani": ir_azer},
    }
    for c, d in out.items():
        s = sum(d.values())
        out[c] = {lab: v / s for lab, v in d.items()}
        assert all(v > 0 for v in d.values()), (c, d)
    print("  splits from Parkvall 2006 over SCB 2006 stocks:")
    for c, d in out.items():
        print(f"    {c} ({st06[c]:,.0f} born there in 2006): "
              + ", ".join(f"{lab} {v:.1%}" for lab, v in d.items()))
    return out


# first generation: share keeping the origin language, where it is not ~all (Parkvall)
FINLAND_SWEDISH = 0.25     # Parkvall 2009: Swedish-speaking Finns "more than a quarter"
# Ethiopia: Amharic + Oromo speakers (Parkvall 2006, foreign-born part) over the 2006 stock;
# the rest are mostly adoptees. Korea: Korean speakers over the 2006 stock (adoptees).
SECOND_TWO, SECOND_ONE = 0.78, 0.14   # Parkvall 2009 pp. 83-84 (Boyd 1985; Nekby & Özcan 2006)
SYRIA_COHORT_YEAR = 2006   # Syria-born above the 2006 stock are drawn Arabic (sources/se.md §2)

# ---------------------------------------------------------------------------------------------
# national minority languages
# ---------------------------------------------------------------------------------------------
FIVE = ["2583", "2518", "2521", "2584", "2523"]   # Haparanda Övertorneå Pajala Kiruna Gällivare
# Meänkieli: Parkvall 2009 p. 48, 15,000-45,000 grew up as active users; ~15,000 native
# speakers in the old Finnish-majority villages of the Five Kommuner, ~15,000 elsewhere. The
# elsewhere half follows his phone-book surnames: 19 parts rest of Norrbotten, 42 rest of
# Sweden (of 61 outside the five).
MEANKIELI_FIVE, MEANKIELI_ELSE = 15000, 15000
# Sami: about 6,000 first-language speakers (ISOF Vanliga frågor 2024; Parkvall 2016 for
# 2012), three quarters North, ~15% Lule, ~a tenth South (Parkvall 2009 p. 53).
SAMI = {"North Sami": 4500, "Lule Sami": 900, "South Sami": 600}
# Parkvall 2009 pp. 54-55 (Sameutredningen, early 1970s): Sami speakers in the ten largest
# kommuner, and by län; the län remainders are spread over the län's Sami administrative-area
# kommuner (minoritet.se, 2025) by population. Each place belongs to one language area.
SAMI_KOMMUN = {"2584": 1449, "2523": 1034, "2510": 576, "2505": 387, "2506": 376,
               "2421": 275, "2580": 219, "0180": 217, "2417": 195, "2422": 177}
SAMI_LAN = {"25": 4482, "24": 1288, "23": 751, "01": 592, "22": 245 - 31, "20": 116,
            "03": 92}
SAMI_LAN["22"] = 214
SAMI_OTHER = 808
SAMI_ADMIN = {"25": ["2581", "2514"], "24": ["2481", "2418", "2462", "2425", "2463", "2404",
                                             "2480", "2482"],
              "23": ["2326", "2361", "2309", "2313", "2321", "2380"], "22": ["2281", "2284"],
              "20": ["2039"]}
SAMI_AREA = {"North Sami": {"2584", "2523", "25rest"}, "Lule Sami": {"2510", "2505", "2506"},
             "South Sami": {"2421", "2417", "2422", "24rest", "23rest", "22rest", "20rest"}}
DISPERSED = {"2580", "0180", "01rest", "03rest", "other"}
# Romani: 11,000 first-language speakers (Parkvall 2016, data 2012), 16% born in Sweden
# (Parkvall 2009); the foreign-born part comes out of these countries' rows.
ROMANI, ROMANI_SWEDEN_BORN = 11000, 0.16
ROMANI_FROM = ["FI", "YU", "BA", "RS", "CS", "XK", "MK", "RO", "BG", "PL", "HU", "SK"]
# Yiddish: about 1,000 (Parkvall 2009; ISOF 2024: 750-1,500), on the three cities with Jewish
# communities, by population
YIDDISH = 1000
YIDDISH_AT = ["0180", "1480", "1280"]


ROMANI_NODE = "indoeuropean.indoaryan.romani.romani"


def lang_of(iso, sp):
    """{label or node: share}. Parkvall's splits (`sp`: Iraq, Syria, Turkey, Iran, Ethiopia)
    and Finland (Finnish, the Finland-Swedes handled by FINLAND_SWEDISH) are Sweden's own
    overrides; everything else is the shared origin table (sources/origin_mix.py, 2026-10-05),
    Swedish as a label. ROMANI_FROM countries' home mixes lose their Romani share, which
    Parkvall's Romani total already counts. SE_OVERRIDES above is kept for the record: its
    dissolved states are origin_mix's PSEUDO, the rest went back to the home mixes."""
    if iso in sp:
        return sp[iso]
    if iso == "FI":
        return {"Finnish": 1.0}
    from origin_mix import mix
    m = {("Swedish" if n == "indoeuropean.germanic.north.swedish" else n): s
         for n, s in mix(iso, "se").items()}
    if iso in ROMANI_FROM and ROMANI_NODE in m and len(m) > 1:
        m.pop(ROMANI_NODE)
        t = sum(m.values())
        m = {k: v / t for k, v in m.items()}
    return m


# ---------------------------------------------------------------------------------------------
def ipf(seed, rows, cols, tol=0.5):
    x = seed.copy()
    for it in range(1000):
        x = x.mul(rows / x.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        x = x.mul(cols / x.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
        if (x.sum(axis=1) - rows).abs().max() < tol:
            break
    return x, it + 1, (x.sum(axis=1) - rows).abs().max(), (x.sum(axis=0) - cols).abs().max()


def unsuppress():
    """kommun x birth country (foreign-born), each kommun's "övriga födelseländer" spread back
    over the countries SCB hid in it."""
    kom, riket, _, _ = se_scb.regional()
    named = [c for c in riket if c not in ("TOTfod", "OVFOD", "ÖOF", "SE")]
    m = pd.DataFrame({k: {c: v.get(c, 0) for c in named} for k, v in kom.items()}).T
    ovr = pd.Series({k: v.get("OVFOD", 0) for k, v in kom.items()})
    deficit = pd.Series({c: riket[c] - m[c].sum() for c in named})
    true_other = riket["OVFOD"]
    print(f"  suppressed into övriga: {ovr.sum() - true_other:,} of {ovr.sum():,} (the rest, "
          f"{true_other:,}, is countries SCB names nowhere); named countries' national "
          f"deficit {deficit.sum():,} (the gap is SCB's cell perturbation)")
    assert (deficit >= 0).all(), deficit[deficit < 0]
    rows = (ovr - true_other * ovr / ovr.sum()).clip(lower=0)
    cols = deficit * rows.sum() / deficit.sum()
    fb = pd.Series({k: v["TOTfod"] - v.get("SE", 0) for k, v in kom.items()})
    seed = (m == 0).astype(float).mul(fb, axis=0).loc[:, cols > 0]
    x, n, er, ec = ipf(seed, rows, cols[cols > 0])
    print(f"  IPF övriga: {n} sweeps, worst kommun off {er:.2f}, worst country off {ec:.2f}")
    full = m.astype(float).add(x, fill_value=0)
    full["OTHER"] = true_other * ovr / ovr.sum()
    tot = pd.Series({k: v["TOTfod"] for k, v in kom.items()})
    sweden = pd.Series({k: v.get("SE", 0) for k, v in kom.items()})
    unknown = pd.Series({k: v.get("ÖOF", 0) for k, v in kom.items()})
    off = (full.sum(axis=1) + sweden + unknown - tot).abs().max()
    print(f"  every kommun rebuilt to within {off:.1f} of its total (SCB's own table is "
          f"internally off by up to 52)")
    return full, tot, sweden, unknown


def second_generation(first, bg):
    """kommun x parents' country for Sweden-born with two (two) and one (one) foreign-born
    parent(s). Mixed-country couples count half under each parent's country."""
    par = se_scb.parents()
    nat_two = pd.Series(par["Inr2uS"], dtype=float) + 0.5 * (
        pd.Series(par["Inr2uOF"], dtype=float) + pd.Series(par["Inr2uOM"], dtype=float))
    nat_one = pd.Series(par["Inr1uF"], dtype=float) + pd.Series(par["Inr1uM"], dtype=float)
    out = {}
    for name, nat, code in (("two", nat_two, "4"), ("one", nat_one, "5")):
        rows = pd.Series({k: bg[k][code] for k in first.index}, dtype=float)
        nat = nat[nat > 0]
        missing = sorted(set(nat.index) - set(first.columns))
        cols = nat.reindex([c for c in nat.index if c in first.columns]).fillna(0)
        print(f"  second generation ({name} foreign-born parents): kommuner {rows.sum():,.0f}, "
              f"by parents' country {nat.sum():,.0f}; countries with no foreign-born column "
              f"({len(missing)}, {nat[missing].sum():,.0f} people) folded into OTHER")
        cols["OTHER"] = nat[missing].sum()
        cols = cols * rows.sum() / cols.sum()
        seed = first[cols.index].clip(lower=0) + 1e-6
        x, n, er, ec = ipf(seed, rows, cols)
        print(f"    IPF: {n} sweeps, worst kommun off {er:.2f}, worst country off {ec:.2f}")
        out[name] = x
    return out


# ---------------------------------------------------------------------------------------------
def main():
    first, tot, sweden, unknown = unsuppress()
    bg = se_scb.background()
    st06 = se_scb.stock2006()
    sp = splits(st06)
    # Syria: the 2006 cohort takes the split, everyone above it Arabic
    sy_old = st06["SY"] / first["SY"].sum()
    sp_sy_old = sp.pop("SY")
    keep = {c: 1.0 for c in first.columns}
    keep["FI"] = 1 - FINLAND_SWEDISH
    keep["KR"] = PV["Korean"][0] / st06["KR"]
    eth = sum(PV[k][0] * (1 - PV[k][1]) for k in ("Amharic", "Oromo"))
    keep["ET"] = eth / st06["ET"]
    sp["ET"] = {"Amharic": PV["Amharic"][0] / 5400, "Oromo": PV["Oromo"][0] / 5400}
    print(f"  first generation keeping the origin language: Finland {keep['FI']:.0%}, "
          f"South Korea {keep['KR']:.1%}, Ethiopia {keep['ET']:.1%}; everyone else 100%")
    gen2 = second_generation(first, bg)

    recs = []

    def add(unit, label, n, part):
        if n > 0:
            recs.append((unit, label, float(n), part))

    other_mix = None
    for unit in first.index:
        for gen, frame, rate in (("born abroad", first, 1.0),
                                 ("born in Sweden, two foreign-born parents", gen2["two"],
                                  SECOND_TWO),
                                 ("born in Sweden, one foreign-born parent", gen2["one"],
                                  SECOND_ONE)):
            for iso in frame.columns:
                n = frame.at[unit, iso]
                if n <= 0:
                    continue
                if iso == "OTHER":
                    add(unit, "Other", n * rate, gen)
                    add(unit, "Swedish", n * (1 - rate), gen)
                    continue
                k = keep.get(iso, 1.0) * rate
                if iso == "SY":
                    old = sy_old   # the second generation follows the first's mix
                    for lab, s in sp_sy_old.items():
                        add(unit, lab, n * old * k * s, gen)
                    add(unit, "Arabic", n * (1 - old) * k, gen)
                else:
                    lang = lang_of(iso, sp)
                    if isinstance(lang, dict):
                        for lab, s in lang.items():
                            add(unit, lab, n * k * s, gen)
                    else:
                        add(unit, lang, n * k, gen)
                if iso == "FI":
                    add(unit, "Swedish", n * rate * FINLAND_SWEDISH, gen + ", Finland-Swedish")
                    add(unit, "Swedish", n * (1 - rate), gen)
                else:
                    add(unit, "Swedish", n * (1 - k), gen)
    df = pd.DataFrame(recs, columns=["unit", "label", "count", "part"])
    # third generation and beyond, and whatever the second-generation tables do not cover
    bg_df = pd.DataFrame(bg).T.astype(float)
    native = bg_df["6"]
    resid = tot - unknown - df.groupby("unit")["count"].sum() - native
    print(f"  residual after generations 1-2 and Swedish-born of Swedish-born parents: "
          f"{resid.sum():,.0f} (range {resid.min():,.0f} to {resid.max():,.0f}); added to Swedish")
    for unit in first.index:
        add(unit, "Swedish", native[unit] + resid[unit], "born in Sweden")
    df = pd.DataFrame(recs, columns=["unit", "label", "count", "part"])

    # ---- national minority languages, carved out of Swedish (and Romani's countries) ----
    pop = tot.astype(float)
    carve = []

    def take(unit, label, n, part):
        carve.append((unit, label, n, part))
        carve.append((unit, "Swedish", -n, part))

    def spread(units, n, label, part):
        p = pop[units]
        for u, v in (n * p / p.sum()).items():
            take(u, label, v, part)

    lan = pop.index.str[:2]
    norrb_rest = pop.index[(lan == "25") & ~pop.index.isin(FIVE)]
    spread(FIVE, MEANKIELI_FIVE, "Meänkieli", "Meänkieli, Five Kommuner")
    spread(norrb_rest, MEANKIELI_ELSE * 19 / 61, "Meänkieli", "Meänkieli, rest of Norrbotten")
    spread(pop.index[lan != "25"], MEANKIELI_ELSE * 42 / 61, "Meänkieli",
           "Meänkieli, rest of Sweden")

    # Sami: place -> weight
    w = dict(SAMI_KOMMUN)
    for l, n in SAMI_LAN.items():
        listed = sum(v for k, v in SAMI_KOMMUN.items() if k[:2] == l)
        w[f"{l}rest"] = n - listed
    w["other"] = SAMI_OTHER
    disp = sum(w[k] for k in DISPERSED)
    total_w = sum(w.values())
    assert total_w == 8343 or abs(total_w - 8343) < 50, total_w
    place_units = {k: [k] for k in SAMI_KOMMUN}
    for l in SAMI_LAN:
        place_units[f"{l}rest"] = SAMI_ADMIN.get(l) or list(
            pop.index[(lan == l) & ~pop.index.isin(list(SAMI_KOMMUN))])
    place_units["25rest"] = list(pop.index[(lan == "25") & ~pop.index.isin(list(SAMI_KOMMUN))])
    place_units["other"] = list(pop.index[~lan.isin(list(SAMI_LAN))])
    for lang, total in SAMI.items():
        d_share = disp / total_w
        area = SAMI_AREA[lang]
        aw = sum(w[k] for k in area)
        for k in area:
            spread(place_units[k], total * (1 - d_share) * w[k] / aw, lang, "Sami area")
        for k in DISPERSED:
            spread(place_units[k], total * d_share * w[k] / disp, lang, "Sami, dispersed")
    print(f"  Sami: {disp / total_w:.1%} dispersed outside the Sami area (Stockholm, Luleå, "
          f"Uppsala, the south), by Parkvall's 1970s counts")

    # Romani
    sb = ROMANI * ROMANI_SWEDEN_BORN
    spread(list(pop.index), sb, "Romani", "Romani, born in Sweden")
    src = df[(df["part"] == "born abroad")].copy()
    src["iso_lang"] = src["label"]
    base = first[ROMANI_FROM].sum(axis=1)
    fb_take = ROMANI - sb
    langs = {lab for c in ROMANI_FROM for lab in lang_of(c, sp) if lab != "Swedish"}
    for u, b in (fb_take * base / base.sum()).items():
        if b <= 0:
            continue
        rows_u = df[(df["unit"] == u) & (df["part"] == "born abroad") & df["label"].isin(langs)]
        s = rows_u["count"].sum()
        carve.append((u, "Romani", b, "Romani, born abroad"))
        for lab, c in rows_u.groupby("label")["count"].sum().items():
            carve.append((u, lab, -b * c / s, "Romani, born abroad"))
    # Yiddish
    spread(YIDDISH_AT, YIDDISH, "Yiddish", "Yiddish")

    df = pd.concat([df, pd.DataFrame(carve, columns=["unit", "label", "count", "part"])])
    res = df.groupby(["unit", "label"], as_index=False)["count"].sum()
    neg = res[res["count"] < -0.5]
    if len(neg):
        sys.exit(f"!! negative rows after carving:\n{neg}")
    # rows under half a person (the origin mixes' long tails) go onto Swedish, so totals hold
    small = res["count"] < 0.5
    tiny = res[small].groupby("unit")["count"].sum()
    res = res[~small].copy()
    is_sv = res["label"] == "Swedish"
    res.loc[is_sv, "count"] += res.loc[is_sv, "unit"].map(tiny).fillna(0.0)
    chk =res.groupby("unit")["count"].sum() - (tot - unknown)
    print(f"  kommuner sum to SCB's population less unknown birth country within "
          f"{chk.abs().max():.1f}; Sweden {res['count'].sum():,.0f} "
          f"(SCB {tot.sum():,}, unknown birth country {unknown.sum():,} not drawn)")
    nat = res.groupby("label")["count"].sum().sort_values(ascending=False)
    print("  national, top 30:")
    for lab, n in nat.head(30).items():
        print(f"    {lab:24s} {n:>12,.0f}  {n / nat.sum():6.2%}")
    for lab in ("Finnish", "Meänkieli", "North Sami", "Lule Sami", "South Sami", "Romani",
                "Yiddish", "Turoyo", "Assyrian Neo-Aramaic"):
        print(f"    {lab:24s} {nat.get(lab, 0):>12,.0f}")
    res = res.rename(columns={"unit": "geo_id", "label": "source_category"})
    res["geo_level"] = "kommun"
    res["tier"] = "derived"
    res["year"] = YEAR
    res["source_id"] = SOURCE_ID
    res = res[["geo_id", "geo_level", "source_category", "count", "tier", "year", "source_id"]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_csv(OUT, index=False, encoding="utf-8")
    print(f"  wrote {OUT}: {len(res):,} rows, {res['geo_id'].nunique()} kommuner, "
          f"{res['source_category'].nunique()} labels")
    return res


if __name__ == "__main__":
    main()
