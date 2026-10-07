# Syria. No census or survey asks: cited minority estimates on the CBS end-2011 governorate
# populations (sources/sy_build.py), on religiondots' 400m Kontur hexes (read-only). Inside a
# governorate, a minority whose estimate names its places is drawn only there (boxes below); every
# other row is spread on hex population. The record is sources/sy.md.
from _shared import *  # noqa: F401,F403

# (unit, node) -> [(lon0, lat0, lon1, lat1, people), ...]: where inside the governorate the
# estimate puts them, people shared over the boxes as the estimate splits them. Moves people only
# inside the governorate that counts them (AGENT_BRIEF 4.4); counts are untouched.
KURD = "indoeuropean.iranian.kurdish"
_AFRIN = (36.55, 36.33, 37.00, 36.85)        # Afrin district (Kurd Dagh)
_KOBANI = (37.95, 36.40, 38.75, 36.95)       # Ayn al-Arab district
_ALEPPO_CITY = (37.05, 36.12, 37.28, 36.29)
BOXES = {
    # sources/sy_build.py: Afrin 172,095 x 100%, Ayn al-Arab 192,513 x 55%, city 2,181,061 x 22.5%
    ("SY02", KURD): [(*_AFRIN, 172_095), (*_KOBANI, 192_513 * 0.55),
                     (*_ALEPPO_CITY, 2_181_061 * 0.225)],
    ("SY02", "indoeuropean.armenian.armenian"): [(*_ALEPPO_CITY, 1)],
    # Turkmen villages of the Azaz, al-Rai and Jarabulus countryside (Balanche 2018, figure 29)
    ("SY02", "turkic.syrian_turkmen"): [(36.95, 36.45, 38.10, 36.85, 1)],
    # Jabal al-Turkmen: Rabia and Qastal Ma'af subdistricts
    ("SY06", "turkic.syrian_turkmen"): [(35.75, 35.70, 36.10, 35.97, 1)],
    # Hasakah's Kurds and Assyrians live in the north: the Turkish border towns, Hasakah city and
    # the Khabur villages; the Arab south (al-Shaddadi) is left out
    ("SY08", KURD): [(39.50, 36.45, 42.40, 37.40, 1)],
    ("SY08", "afroasiatic.aramaic"): [(39.50, 36.45, 42.40, 37.40, 1)],
    # Maaloula and Jubb'adin
    ("SY03", "afroasiatic.western_neo_aramaic"): [(36.45, 33.75, 36.65, 33.90, 1)],
}


class _BoxWeighter(PopWeighter):
    def __init__(self, place):
        super().__init__(place)
        c = place.geometry.representative_point()
        self.x, self.y = c.x.to_numpy(), c.y.to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        self.n_box = 0

    def weights(self, node, idx, count, plain=False):
        p = self.pop[idx]
        units = set(self.unit[idx])
        key = (next(iter(units)), node) if len(units) == 1 else None
        if key in BOXES:
            w = p * 0.0
            x, y = self.x[idx], self.y[idx]
            for x0, y0, x1, y1, people in BOXES[key]:
                inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                s = p[inside].sum()
                if s <= 0:
                    raise SystemExit(f"sy: box {(x0, y0, x1, y1)} for {key} holds no people")
                w = w + inside * p * (people / s)
            self.n_box += 1
            return w
        return super().weights(node, idx, count, plain)

    def summary(self):
        return (f"{self.n_box} (governorate, language) rows placed inside their named places; "
                + super().summary())


def _place_weight(place):
    if "pop" not in place.columns:
        print("  !! sy_hexes.gpkg has no `pop` column; equal shares")
        return None
    return _BoxWeighter(place)


def _counts():
    import sy2011
    df = pd.read_csv(NORM / "sy.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 14 or df["count"].sum() != 21_377_000:
        raise SystemExit("sy.csv: expected 14 governorates, 21,377,000; re-run sources/sy_build.py")
    df["node"] = df["source_category"].map(sy2011.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"sy.csv categories that resolve to nothing: {missing}")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Syria",
    source=("Published estimates: F. Balanche, Sectarianism in Syria's Civil War (Washington "
            "Institute, 2018); Wikipedia (Al-Hasakah Governorate, Western Neo-Aramaic, district "
            "pages for 2004 census figures); on the Central Bureau of Statistics' governorate "
            "populations for the end of 2011 (via OCHA)"),
    how=("no census or survey asks; published estimates of each minority placed on its "
         "governorates, the rest drawn as the governorate's Arabic; pre-war population, 2011"),
    parts=[
        dict(covers="Kurdish, Turkmen, Aramaic and Armenian",
             source="published estimates (Balanche 2018; 2004 census figures and others on "
                    "Wikipedia), placed on their governorates",
             nodes=["indoeuropean.iranian.kurdish", "turkic.syrian_turkmen",
                    "afroasiatic.aramaic", "indoeuropean.armenian.armenian",
                    "afroasiatic.western_neo_aramaic"]),
        dict(covers="Everyone else",
             source="2011 governorate populations, drawn as Levantine or Mesopotamian Arabic",
             rest=True),
    ],
    grain="14 governorates, 1.5 million people on average",
    gap="Circassian, Domari and other small languages, which no estimate places",
    view=[35.5, 32.2, 42.5, 37.4],
    counts=_counts,
    mappings=["sy2011"],
    place=RD_GEO / "sy" / "sy_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_place_weight,
    note_public=(
        "Syria has had no census since 2004, none has published language, and no survey "
        "with a language question has reached the country. Every figure here is an estimate "
        "from a named source, and every dot is inferred. "
        "The dots stand where people lived at the end of 2011, the last population count "
        "before the war; millions have since left the country or moved inside it, and the "
        "Kurdish, Armenian and Christian figures are pre-war too. Kurds follow Fabrice "
        "Balanche's estimates. His figure for Damascus counts Kurds by origin, and many "
        "Damascus Kurds have spoken Arabic for generations, so the Kurdish share drawn there "
        "is an upper bound. The Aramaic drawn in Hasakah is the governorate's Christians, some "
        "of whom speak Arabic or Armenian, so it too is an upper bound. Arabic is drawn as "
        "Levantine in the west and as Mesopotamian along the Euphrates and in the Jazira. "
        "Circassians, and Armenians outside Aleppo, are not drawn: no estimate places them."),
)
