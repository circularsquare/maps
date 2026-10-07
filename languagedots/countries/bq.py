# Caribbean Netherlands (Bonaire, Sint Eustatius, Saba). CBS Omnibus survey 2021, StatLine
# 82867NED, the language each person aged 15+ speaks most, as shares per island, applied to each
# island's population on 1 January 2022 (sources/bq_survey.py). Placed inside each island on
# WorldPop's constrained 2020 grid (sources/bq_geo.py); Kontur has no cells here. Record:
# sources/bq.md.
from _shared import *  # noqa: F401,F403

POP_2022 = 27_726


def _counts():
    import bq2021
    df = pd.read_csv(NORM / "bq.csv", dtype={"geo_id": str})
    if sorted(df["geo_id"].unique()) != ["GM9001", "GM9002", "GM9003"]:
        raise SystemExit("bq.csv: expected the three islands")
    if abs(df["count"].sum() - POP_2022) > 30:
        raise SystemExit(f"bq.csv sums to {df['count'].sum():,}, expected {POP_2022:,}")
    df["node"] = df["source_category"].map(bq2021.resolve)
    df = df[df["node"].notna()]
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Caribbean Netherlands",
    source=("Omnibus survey 2021, table 82867NED (Statistics Netherlands), on its 1 January "
            "2022 island populations (83774NED); WorldPop 2020 constrained grid for placement"),
    how=("a survey, 2021, the language people aged 15 and over speak most; shares per island "
         "applied to each island's whole population"),
    parts=[dict(covers="Everyone",
                source="Omnibus survey 2021, language spoken most, aged 15 and over, island "
                       "shares", rest=True)],
    grain="3 islands, 9,200 people on average",
    gap=("4 people on Saba, where Statistics Netherlands withheld the Papiamentu share as too "
         "uncertain to publish"),
    view=[-68.45, 12.00, -62.90, 17.70],
    counts=_counts,
    mappings=["bq2021"],
    place=GEO / "bq" / "bq_cells.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "These are survey shares, not a census count. Statistics Netherlands asked people aged "
        "15 and over in 2021 which language they speak most; each island's shares are applied "
        "to its whole 2022 population, children included. On Bonaire 62% speak Papiamentu "
        "most, 15% Spanish and 15% Dutch; on Sint Eustatius 81% and on Saba 83% speak English "
        "most, and Spanish is the next language on both. The survey offered five answers, so "
        "every other language is drawn as other. Within each island the dots follow "
        "WorldPop's 2020 population grid, not the survey."),
)
