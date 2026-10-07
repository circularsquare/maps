# Egypt. No census or survey counts its languages: cited speaker estimates on their home
# governorates, the rest of each governorate Egyptian or Sa'idi Arabic, refugees from UNHCR's
# governorate counts by nationality (sources/eg_build.py), on CAPMAS's 2026 governorate
# populations and religiondots' Kontur hexes for the 27 governorates. Record: sources/eg.md.
from _shared import *  # noqa: F401,F403
import numpy as np

CAPMAS_2026 = 108_528_518
UNITS = 27

# Inside a governorate, a few languages are kept to the part of it that is their home (a
# placement only; every governorate's counts are sources/eg_build.py's either way):
#   Siwi in Matrouh: the Siwa and Qara oases box; Libyan (Awlad Ali) Arabic: Matrouh outside it.
#   Nobiin, Kenzi and Beja in Aswan: south of 24.6N (Kom Ombo, Nasr al-Nuba, Daraw, Aswan city),
#     not Edfu's Sa'idi north.
#   Beja in the Red Sea: south of 24N (Shalateen, Abu Ramad, Halaib).
SIWA = (25.0, 28.8, 27.0, 29.8)          # w, s, e, n
ZONES = {
    ("EG33", "afroasiatic.berber.siwi"): "siwa",
    ("EG33", "afroasiatic.libyan_arabic"): "not_siwa",
    ("EG28", "nilosaharan.nobiin"): "aswan_south",
    ("EG28", "nilosaharan.kenzi"): "aswan_south",
    ("EG28", "afroasiatic.cushitic.beja"): "aswan_south",
    ("EG31", "afroasiatic.cushitic.beja"): "redsea_south",
}


def _counts():
    import eg2026
    df = pd.read_csv(NORM / "eg.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"eg.csv: {df['geo_id'].nunique()} governorates, expected {UNITS} -- "
                         "re-run sources/eg_build.py")
    if int(df["count"].sum()) != CAPMAS_2026:
        raise SystemExit(f"eg.csv sums to {int(df['count'].sum()):,}, not {CAPMAS_2026:,}")
    lut = pd.read_csv(RD_GEO / "eg" / "eg_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit("eg.csv governorates missing from religiondots' eg_lookup.csv")
    df["node"] = df["source_category"].map(eg2026.resolve)
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return out[["unit", "node", "count", "tier"]]


class _EgWeighter:
    """Population weights, cut to a zone for the (governorate, language) pairs in ZONES."""

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        c = place.geometry.to_crs(3857).centroid.to_crs(4326)
        x, y = c.x.to_numpy(), c.y.to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        w, s, e, n = SIWA
        siwa = (x >= w) & (x <= e) & (y >= s) & (y <= n)
        self.mask = {"siwa": siwa, "not_siwa": ~siwa, "aswan_south": y < 24.6,
                     "redsea_south": y < 24.0}
        self.n = {"zone": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() <= 0:
            self.n["none"] += 1
            return None
        z = ZONES.get((self.unit[idx[0]], node))
        if z:
            w = pop * self.mask[z][idx]
            if w.sum() <= 0:
                raise SystemExit(f"eg: zone {z} holds no population for {node}")
            self.n["zone"] += 1
            return w
        self.n["pop"] += 1
        return pop

    def summary(self):
        return (f"{self.n['zone']} (governorate, language) rows kept to their home zone, "
                f"{self.n['pop']:,} on population, {self.n['none']} on equal shares")


def _weight(place):
    if "pop" not in place.columns:
        raise SystemExit("eg_hexes.gpkg has no `pop` column")
    return _EgWeighter(place)


ENTRY = dict(
    name="Egypt",
    source=("Ethnologue's speaker figures as Wikipedia carries them (Sa'idi Arabic, Nobiin, Kenzi, "
            "Siwi, Beja; Domari from the 2016 edition via Joshua Project); Bedouin shares from "
            "Huesken (International Affairs, 2017) for Matrouh, Aziz (Brookings, 2017) for North "
            "Sinai and Senri Ethnological Studies 55 (2001) for South Sinai; refugees registered "
            "with UNHCR Egypt by governorate, 31 May 2026; each governorate's population from "
            "CAPMAS's own estimate for 1 January 2026"),
    how=("no census or survey counts languages; speaker estimates placed on their home "
         "governorates, refugees by nationality, everyone else Egyptian or Sa'idi Arabic by "
         "governorate"),
    parts=[
        dict(covers="Nubian, Beja, Siwi and Domari",
             source="Ethnologue speaker estimates, placed on their home governorates",
             nodes=["nilosaharan.nobiin", "nilosaharan.kenzi", "afroasiatic.cushitic.beja",
                    "afroasiatic.berber.siwi", "indoeuropean.indoaryan.domari"]),
        dict(covers="Bedouin Arabic in Matrouh and Sinai",
             source="published Bedouin shares of each governorate (2001, 2017)",
             nodes=["afroasiatic.libyan_arabic", "afroasiatic.bedawi_arabic"]),
        dict(covers="Refugees", source="UNHCR registered refugees by governorate and "
             "nationality, May 2026", people=1_101_744),
        dict(covers="Upper Egypt, Minya to Aswan and the New Valley",
             source="drawn as Sa'idi Arabic", nodes=["afroasiatic.saidi_arabic"]),
        dict(covers="Everyone else", source="drawn as Egyptian Arabic", rest=True),
    ],
    grain="27 governorates, 4.0 million people on average",
    gap=("Bedouin outside Matrouh and Sinai, Upper Egyptians elsewhere and unregistered "
         "migrants, all drawn on the governorate's main Arabic"),
    view=[24.6, 21.9, 37.0, 31.8],
    counts=_counts,
    mappings=["eg2026"],
    place=RD_GEO / "eg" / "eg_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "No Egyptian census asks about language, and the surveys that do (Afrobarometer and the "
        "Arab Barometer) interviewed everyone in Arabic, so nothing on this map of Egypt is a "
        "count of a language. Arabic is split by governorate: Sa'idi from Minya south to Aswan "
        "and in the New Valley oases, Egyptian Arabic elsewhere. Nubian is drawn around Aswan "
        "and Kom Ombo, with the rest of its 537,000 speakers in Cairo, Giza and Alexandria. "
        "Bedouin Arabic is drawn for 85% of Matrouh (the Awlad Ali, whose Arabic is close to "
        "Libya's) and for the Bedouin of North and South Sinai; Bedouin living elsewhere are "
        "not separated. Refugees are the 1.1 million registered with UNHCR, most of them "
        "Sudanese in Giza and Cairo."),
)
