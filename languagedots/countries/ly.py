# Libya. World Values Survey waves 6 and 7 and Arab Barometer III language answers pooled by
# district, with cited estimates for Nafusi, Tuareg and Tebu (sources/ly_surveys.py), on the BSC's
# 2020 estimate of Libyans; religiondots' 400m Kontur hexes (read-only). The record is sources/ly.md.
# Coverage gap (2026-10-19): the 584,509 migrants IOM's DTM counted in July-August 2020, by
# neighbourhood and nationality, are added as `modelled` rows on their nationality's home languages
# (sources/ly_migrants.py, sources/origin_mix.py) and placed around the DTM's neighbourhood points.
from _shared import *  # noqa: F401,F403
import numpy as np

NAFUSI = "afroasiatic.berber.nafusi"
ZUWARA = "afroasiatic.berber.zuwara"
GHADAMES = "afroasiatic.berber.ghadames"
AWJILA = "afroasiatic.berber.awjila"
LIBYAN_ARABIC = "afroasiatic.libyan_arabic"
# 2026-10-19: Berber split by place (sources/ly.md section 6). Zuwara district's Berber answers are
# Zuwara's own speech. Ghadames and Awjila are cited speaker counts (Ethnologue 27th ed., 2020, via
# Wikipedia: 17,000 and 2,700), carved out of Nalut's Nafusi and Ajdabiya's Libyan Arabic.
GHADAMES_PEOPLE = 17_000
AWJILA_PEOPLE = 2_700
# (unit, node) -> boxes (lon0, lat0, lon1, lat1, weight): where inside the district the Berber
# speakers live. Moves people only inside the district (AGENT_BRIEF 4.4).
BOXES = {
    ("LY0215", ZUWARA): [(11.95, 32.80, 12.25, 33.00, 1)],    # Zuwara town
    ("LY0216", NAFUSI): [(12.30, 31.95, 12.95, 32.15, 1)],    # Yafran, Kikla, al-Qalaa
    ("LY0209", NAFUSI): [(10.30, 31.40, 11.80, 32.30, 1)],    # Nalut, Kabaw, Wazin: not the oasis
    ("LY0209", GHADAMES): [(9.40, 30.05, 9.65, 30.25, 1)],    # Ghadames oasis
    ("LY0105", AWJILA): [(21.10, 29.05, 21.40, 29.25, 1)],    # Awjila oasis
}


SIGMA_KM = 2.0       # reach of one DTM neighbourhood point over the hexes around it
FLOOR = 0.02         # share of the strongest hex weight every hex keeps (rural parts of a district)


def _migrants():
    import sys
    sys.path.insert(0, str(ROOT / "sources"))
    import ly_migrants
    return ly_migrants.counts(), pd.read_csv(NORM / "ly_migrant_sites.csv")


class _BoxWeighter(PopWeighter):
    def __init__(self, place):
        super().__init__(place)
        c = place.geometry.representative_point()
        self.x, self.y = c.x.to_numpy(), c.y.to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        self.n_box = 0
        # migrants: (unit, node) -> migrant people, and the Libyan people of the same row (a merged
        # row is told by its count), with the DTM's neighbourhood points per district
        mig, self.sites = _migrants()
        self.mig = {(u, n): float(c) for u, n, c in zip(mig["unit"], mig["node"], mig["count"])}
        lib = _libyans().groupby(["unit", "node"])["count"].sum()
        self.lib = {k: float(v) for k, v in lib.items()}
        self._mw = {}
        self.n_mig = 0

    def _migrant_weights(self, u, idx):
        """Hex weights of one district for migrants: population around the DTM's neighbourhood
        points, a Gaussian of SIGMA_KM, with a floor; neighbourhoods without coordinates share their
        weight out on population."""
        if u in self._mw:
            return self._mw[u]
        p = self.pop[idx]
        s = self.sites[self.sites["unit"] == u]
        loc = s[s["lat"].notna()]
        if p.sum() <= 0:
            self._mw[u] = None
            return None
        w = p / p.sum()
        if len(loc) and loc["people"].sum() > 0:
            x, y = self.x[idx], self.y[idx]
            k = np.zeros(len(idx))
            for r in loc.itertuples():
                dx = (x - r.lon) * 111.0 * np.cos(np.radians(r.lat))
                dy = (y - r.lat) * 111.0
                k += r.people * np.exp(-(dx * dx + dy * dy) / (2 * SIGMA_KM ** 2))
            k = p * (k + FLOOR * k.max()) if k.max() > 0 else p
            f = float(s.loc[s["lat"].isna(), "people"].sum() / s["people"].sum())
            w = (1 - f) * k / k.sum() + f * w
        self._mw[u] = w
        return w

    def weights(self, node, idx, count, plain=False):
        u = self.unit[idx[0]]
        mig = self.mig.get((u, node), 0.0)
        if mig > 0:
            lib = self.lib.get((u, node), 0.0)
            if abs(count - (lib + mig)) < 0.5:
                wm = self._migrant_weights(u, idx)
                if wm is not None:
                    self.n_mig += 1
                    if lib <= 0:
                        return wm
                    wl = self._libyan_weights(node, idx, count, plain)
                    wl = np.asarray(wl if wl is not None else self.pop[idx], dtype=float)
                    return wl / wl.sum() * (lib / (lib + mig)) + wm * (mig / (lib + mig))
        return self._libyan_weights(node, idx, count, plain)

    def _libyan_weights(self, node, idx, count, plain=False):
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
        return (f"{self.n_box} (district, language) rows placed inside their named towns, "
                f"{self.n_mig} rows with migrants placed around the DTM's neighbourhoods; "
                + super().summary())


def _place_weight(place):
    if "pop" not in place.columns:
        print("  !! ly_hexes.gpkg has no `pop` column; equal shares")
        return None
    return _BoxWeighter(place)


def _libyan_counts():
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
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return _split_berber(out)


def _libyans():
    return _libyan_counts()


def _counts():
    lib = _libyan_counts()
    mig, _ = _migrants()
    return pd.concat([lib, mig.assign(tier="modelled")], ignore_index=True).groupby(
        ["unit", "node", "tier"], as_index=False)["count"].sum()


def _move(df, unit, src, dst, n):
    """Take n people from the (unit, src) row to a (unit, dst) row of the same tier."""
    i = df.index[(df["unit"] == unit) & (df["node"] == src)]
    if len(i) != 1 or df.loc[i[0], "count"] < n:
        raise SystemExit(f"ly: {unit} has no {src} row with {n:,} people to move")
    df.loc[i[0], "count"] -= n
    row = df.loc[i[0]].copy()
    row["node"], row["count"] = dst, n
    return pd.concat([df, row.to_frame().T], ignore_index=True)


def _split_berber(df):
    """Zuwara district's Berber to Zuwara; Ghadames carved from Nalut's Nafusi; Awjila from
    Ajdabiya's Libyan Arabic. Totals per district do not change."""
    df = df.copy()
    z = df.index[(df["unit"] == "LY0215") & (df["node"] == NAFUSI)]
    if len(z) != 1:
        raise SystemExit("ly: expected one Nafusi row in Zuwara (LY0215)")
    df.loc[z, "node"] = ZUWARA
    df = _move(df, "LY0209", NAFUSI, GHADAMES, GHADAMES_PEOPLE)
    df = _move(df, "LY0105", LIBYAN_ARABIC, AWJILA, AWJILA_PEOPLE)
    df["count"] = df["count"].astype(int)
    return df[df["count"] > 0].reset_index(drop=True)


ENTRY = dict(
    name="Libya",
    source=("World Values Survey waves 6 (2014) and 7 (2022), language at home; Arab Barometer "
            "III (2014), first language; Ethnologue's Nafusi figure and Wikipedia's Tuareg and "
            "Toubou estimates; on the Bureau of Statistics and Census's 2020 estimate of Libyans "
            "by district; migrants from the International Organization for Migration's Displacement "
            "Tracking Matrix, round 32 (July to August 2020), with home languages from the "
            "migrants' countries on this map"),
    how=("survey, three rounds 2014 to 2022 pooled, home or first language; Nafusi, Tuareg and "
         "Tebu raised to published estimates; 584,509 migrants counted by IOM drawn on their "
         "nationalities' home languages"),
    parts=[
        dict(covers="Nafusi, Zuwara, Ghadames and Awjila (Berber)",
             source="survey answers topped up to Ethnologue's 300,000 Nafusi speakers, split by "
             "place; Ghadames and Awjila at Ethnologue's counts",
             nodes=["afroasiatic.berber.nafusi", "afroasiatic.berber.zuwara",
                    "afroasiatic.berber.ghadames", "afroasiatic.berber.awjila"]),
        dict(covers="Tuareg and Tebu",
             source="low ends of published estimates, split between southern districts by this map",
             nodes=["afroasiatic.berber.tamahaq", "nilosaharan.tubu"]),
        dict(covers="Migrants",
             source="IOM Displacement Tracking Matrix round 32 (2020), migrants by neighbourhood "
                    "and nationality, drawn on each nationality's home languages",
             people=584_509),
        dict(covers="Everyone else",
             source="WVS 2014 and 2022, Arab Barometer 2014, home or first language, pooled by "
                    "district",
             rest=True),
    ],
    grain="22 districts, 312,000 Libyans on average",
    gap="non-Libyans beyond the 584,509 migrants IOM counted in 2020, such as refugees and the "
        "rest of the UN's estimate of 827,000",
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
        "low ends of published estimates. Zuwara, Ghadames and Awjila, which are not Nafusi, "
        "are drawn as their own languages, the last two at Ethnologue's counts. The "
        "survey and the population estimate cover Libyans only. The 584,509 migrants whom the "
        "International Organization for Migration counted in 2020, neighbourhood by "
        "neighbourhood, are added, the largest groups from Niger, Egypt, Chad and Sudan. The "
        "count has no languages, so each person is drawn on the languages of their country "
        "(Hausa and Zarma for Niger, Egyptian and Sa'idi Arabic for Egypt), and dots sit "
        "around the neighbourhoods IOM visited. The UN's higher estimate of 827,000 is not "
        "used."),
)
