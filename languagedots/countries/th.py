# Thailand. 2010 Population and Housing Census, language usually spoken in the household
# (sources/th_census.py), 76 changwat, on religiondots' Kontur hexes for the same 76 (its unit
# codes, read-only). The census's Thai is split into four Thai varieties by World Values Survey
# shares (sources/th_wvs.py).
from _shared import *  # noqa: F401,F403


def _counts():
    """Each household answered Thai only, Thai and another language, or another language only,
    and named that other language. Spec §3.6's exact split for the two-language households:

      Thai           = Thai only + half of Thai and other
      language l     = row_l * (other only + half of Thai and other) / (Thai and other + other only)

    Thai only is measured. The scale on the language rows is exact in total (it is the other-
    language half of the bilingual households plus the other-only households) but shared over
    the rows in proportion, because the table does not say which languages the bilingual
    households named; those rows are `derived`, as is the bilingual half of Thai. People whose
    other language has no row (the table's rows fall short of the two counts by 81,307 summed
    over the provinces) take their share of the scale with them and are not drawn."""
    import th2010
    df = pd.read_csv(NORM / "th.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "province"]
    wide = df.pivot(index="geo_id", columns="source_category", values="count")
    langs = [c for c in th2010.NAMES if c != "Thai"]
    to, oo = wide["Thai and other languages"], wide["Only other languages"]
    scale = (oo + to / 2) / (to + oo)
    out = [
        pd.DataFrame({"unit": wide.index, "node": "kradai.thai",
                      "count": wide["Only Thai language"], "tier": "measured"}),
        pd.DataFrame({"unit": wide.index, "node": "kradai.thai",
                      "count": to / 2, "tier": "derived"}),
    ]
    for c in langs:
        out.append(pd.DataFrame({"unit": wide.index, "node": th2010.resolve(c),
                                 "count": wide[c] * scale, "tier": "derived"}))
    out = pd.concat(out, ignore_index=True)
    out = _split_thai(out, th2010)
    out = out[out["count"] > 0]
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _split_thai(out, th2010):
    """Split every Thai row into Central Thai, Isan, Northern Thai and Southern Thai by the
    changwat's World Values Survey shares (data/normalized/th_wvs.csv, sources/th_wvs.py: the
    share of each variety among respondents who named a Thai variety as their language at home,
    wave 7 by changwat shrunk towards its region). The census's Thai total in each changwat is
    unchanged; the split rows are `modelled`."""
    sh = pd.read_csv(NORM / "th_wvs.csv", dtype={"unit": str}).set_index("unit")
    if set(sh.index) != set(out["unit"]):
        raise SystemExit(f"th_wvs.csv units {len(sh)} do not match the census's "
                         f"{out['unit'].nunique()}")
    if ((sh[list(th2010.VARIETY_NODES)].sum(axis=1) - 1).abs() > 1e-4).any():
        raise SystemExit("th_wvs.csv shares do not sum to 1")
    vs = list(th2010.VARIETY_NODES)
    sh[vs] = sh[vs].div(sh[vs].sum(axis=1), axis=0)     # printed to 5 places; exact again
    thai = out["node"] == th2010.NAMES["Thai"]
    keep, parts = out[~thai], []
    for v, node in th2010.VARIETY_NODES.items():
        p = out[thai].copy()
        p["count"] = p["count"] * p["unit"].map(sh[v])
        p["node"], p["tier"] = node, "modelled"
        parts.append(p)
    new = pd.concat([keep] + parts, ignore_index=True)
    assert abs(new["count"].sum() - out["count"].sum()) < 1, "the split changed the total"
    return new


ENTRY = dict(
    name="Thailand",
    source="2010 Population and Housing Census, provincial reports, table \"Population by usual "
           "languages spoken at home\" (National Statistical Office)",
    how="census, 2010, language usually spoken in the household; a household speaking Thai and "
        "another language is shared half and half between them; the census's Thai split into "
        "Central Thai, Isan, Northern Thai and Southern Thai by World Values Survey shares "
        "(2018, language at home, 1,500 adults, 49 provinces, each province's figure drawn "
        "towards its region's; Bangkok adds the 2013 wave)",
    parts=[
        dict(covers="Thai, split into four",
             source="2010 census, Thai at home; split into Central Thai, Isan, Northern and "
                    "Southern Thai by World Values Survey shares (2018, 1,500 adults)",
             nodes=["kradai.thai", "kradai.isan", "kradai.northern_thai",
                    "kradai.southern_thai"]),
        dict(covers="Other languages",
             source="2010 census, language usually spoken at home; a household speaking Thai "
                    "and another language shared half and half",
             rest=True),
    ],
    grain="76 provinces, 870,000 people on average",
    gap="95,150 people whom Rayong's and Ratchaburi's language tables leave out (their own "
        "population tables count them), and about 50,000 in households that named a language "
        "other than Thai without the table saying which",
    view=[97.3, 5.6, 105.7, 20.5],
    counts=_counts,
    mappings=["th2010"],
    place=RD_GEO / "th" / "th_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked each household which language it usually speaks: Thai only, Thai and "
        "another language, or another language only, and which. Everyone in the household is "
        "counted under its answer, and a household with two languages is drawn half in each. "
        "The census has no box for Isan, Kham Mueang (Northern Thai) or Pak Tai (Southern "
        "Thai). Its Thai is split among them and Central Thai using the World Values Survey "
        "of 2018, which asked 1,500 adults in 49 provinces their language at home and offered "
        "all four; each province takes its own respondents' shares, drawn towards its "
        "region's, and the 28 provinces with no respondents take the region's. So the split "
        "is an estimate from a small survey, not a count. In Bangkok the survey found almost "
        "no Isan spoken at home, though many people there come from the Northeast. Its \"local "
        "language\" answer is drawn as a language not named: 863,000 of its 958,000 are in "
        "Surin, Buri Ram and Si Sa Ket, where Northern Khmer and Kuy are spoken, but the census "
        "does not say which. \"Tai Khün, Thai Loei or Lao Loei\" is a single census answer. "
        "The figures are estimates from the census's sample of households."),
)
