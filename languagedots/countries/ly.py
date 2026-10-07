# Libya. World Values Survey waves 6 and 7 and Arab Barometer III language answers pooled by
# district, with cited estimates for Nafusi, Tuareg and Tebu (sources/ly_surveys.py), on the BSC's
# 2020 estimate of Libyans; religiondots' 400m Kontur hexes (read-only). The record is sources/ly.md.
from _shared import *  # noqa: F401,F403

NAFUSI = "afroasiatic.berber.nafusi"
# (unit, node) -> boxes (lon0, lat0, lon1, lat1, weight): where inside the district the Berber
# speakers live. Moves people only inside the district (AGENT_BRIEF 4.4).
BOXES = {
    ("LY0215", NAFUSI): [(11.95, 32.80, 12.25, 33.00, 1)],    # Zuwara town
    ("LY0216", NAFUSI): [(12.30, 31.95, 12.95, 32.15, 1)],    # Yafran, Kikla, al-Qalaa
}


class _BoxWeighter(PopWeighter):
    def __init__(self, place):
        super().__init__(place)
        c = place.geometry.representative_point()
        self.x, self.y = c.x.to_numpy(), c.y.to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        self.n_box = 0

    def weights(self, node, idx, count, plain=False):
        units = set(self.unit[idx])
        key = (next(iter(units)), node) if len(units) == 1 else None
        if key in BOXES:
            p = self.pop[idx]
            x, y = self.x[idx], self.y[idx]
            w = p * 0.0
            for x0, y0, x1, y1, share in BOXES[key]:
                inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                if p[inside].sum() <= 0:
                    raise SystemExit(f"ly: box {(x0, y0, x1, y1)} for {key} holds no people")
                w = w + inside * p * (share / p[inside].sum())
            self.n_box += 1
            return w
        return super().weights(node, idx, count, plain)

    def summary(self):
        return (f"{self.n_box} (district, language) rows placed inside their named towns; "
                + super().summary())


def _place_weight(place):
    if "pop" not in place.columns:
        print("  !! ly_hexes.gpkg has no `pop` column; equal shares")
        return None
    return _BoxWeighter(place)


def _counts():
    import ly2020
    df = pd.read_csv(NORM / "ly.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 22 or df["count"].sum() != 6_872_674:
        raise SystemExit("ly.csv: expected 22 districts, 6,872,674; re-run sources/ly_surveys.py")
    df["node"] = df["source_category"].map(ly2020.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"ly.csv categories that resolve to nothing: {missing}")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Libya",
    source=("World Values Survey waves 6 (2014) and 7 (2022), language at home; Arab Barometer "
            "III (2014), first language; Ethnologue's Nafusi figure and Wikipedia's Tuareg and "
            "Toubou estimates; on the Bureau of Statistics and Census's 2020 estimate of Libyans "
            "by district"),
    how=("survey, three rounds 2014 to 2022 pooled, home or first language; Nafusi, Tuareg and "
         "Tebu raised to published estimates"),
    parts=[
        dict(covers="Nafusi (Berber)",
             source="survey answers topped up to Ethnologue's 300,000 speakers",
             nodes=["afroasiatic.berber.nafusi"]),
        dict(covers="Tuareg and Tebu",
             source="low ends of published estimates, split between southern districts by this map",
             nodes=["afroasiatic.berber.tamahaq", "nilosaharan.tubu"]),
        dict(covers="Everyone else",
             source="WVS 2014 and 2022, Arab Barometer 2014, home or first language, pooled by "
                    "district",
             rest=True),
    ],
    grain="22 districts, 312,000 Libyans on average",
    gap="non-Libyans, about 827,000 in 2020 (UN migrant stock), who are not in the estimate",
    view=[9.3, 19.5, 25.2, 33.3],
    counts=_counts,
    mappings=["ly2020"],
    place=RD_GEO / "ly" / "ly_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_place_weight,
    note_public=(
        "Libya's census has never published language, and none has been held since 2006. "
        "This map pools three surveys that asked the language people speak at home or learned "
        "first (4,574 answers) and applies each district's shares to its Libyans in the 2020 "
        "population estimate. Every interview was in Arabic, which pulls answers toward "
        "Arabic. The surveys found almost no Berber in the Nafusa mountain towns and no Tebu, "
        "so Berber is raised to Ethnologue's 300,000 Nafusi speakers, the extra placed in "
        "Yafran, Kikla and al-Qalaa, and Tuareg (100,000) and Tebu (50,000) are drawn at the "
        "low ends of published estimates. Ghadames and Awjila, two small Berber languages, are "
        "drawn inside Nafusi. Foreign residents, about one in nine people in Libya, are not on "
        "the map."),
)
