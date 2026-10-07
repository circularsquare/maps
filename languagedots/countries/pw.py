# Palau. 2015 Census, Table 16 by state of usual residence (sources/pw_census.py), with the
# census's own "language used most" item applied: people who speak another language at home but
# use Palauan more are drawn as Palauan, taken off English (sources/pw.md says why English).
# Placement: religiondots' Kontur hexes re-keyed to the 16 states (sources/pw_geo.py).
from _shared import *  # noqa: F401,F403

LESS = "No, less frequently than Palauan"


def _counts():
    import pw2015
    df = pd.read_csv(NORM / "pw.csv")
    df = df[df["geo_level"] == "state"]
    less = df[df["source_category"] == LESS].set_index("geo_id")["count"]
    df = df[df["source_category"].isin(pw2015.NAMES)].copy()
    df["node"] = df["source_category"].map(pw2015.resolve)
    df["unit"] = df["geo_id"]
    df["tier"] = "measured"
    out = []
    for u, g in df.groupby("unit"):
        g = g.copy()
        n = int(less.get(u, 0))
        eng = g["source_category"] == "English"
        pal = g["source_category"] == "Yes, Palauan only"
        if n > int(g.loc[eng, "count"].sum()):
            raise SystemExit(f"pw: {u} has {n} using Palauan more but fewer English speakers")
        if n:
            g.loc[eng, "count"] -= n
            g.loc[eng | pal, "tier"] = "derived"
            g.loc[pal, "count"] += n
        out.append(g)
    df = pd.concat(out)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Palau",
    source="2015 Census of Population, Housing and Agriculture, Table 16 (Office of Planning and "
           "Statistics, Republic of Palau)",
    how="census, 2015, language used most at home",
    parts=[dict(covers="Everyone",
                source="2015 census, language spoken at home and which is used most",
                rest=True)],
    grain="16 states, 1,080 people on average (Koror 11,444)",
    gap="310 people counted with a usual residence outside Palau or unknown (1.8%)",
    view=[131.0, 2.8, 134.8, 8.2],
    counts=_counts,
    mappings=["pw2015"],
    place=GEO / "pw" / "pw_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked everyone whether they speak Palauan at home, which other language they "
        "speak there, and whether they use it more often than Palauan. People who use Palauan "
        "more are shown as Palauan, taken off English, which the 2005 census found was their "
        "other language 95% of the time. People who use both equally are shown under the other "
        "language. The languages of the Philippines are counted together, so a ninth of the "
        "country is drawn as a Philippine language without saying which. Figures are from 2015 "
        "because the 2020 tables name no language for people who do not speak Palauan."),
)
