# Norway. No language statistics: Norwegian, North Sami from a cited estimate, and immigrant
# languages by country background (immigrants and Norwegian-born to immigrant parents), per
# kommune, SSB 1 January 2023 (sources/no_ssb.py). Placed on Kontur hexes keyed to the 356
# kommuner of 2020-2023 (religiondots' GISCO LAU 2021 polygons, read-only). Svalbard (in no
# kommune) is three more units by citizenship, SSB 1 January 2026 (sources/no_svalbard.py), on
# religiondots' Svalbard hexes. Record: sources/no.md.
from _shared import *  # noqa: F401,F403

SSB_2023 = 5_488_984
SVALBARD_2026 = 2_914      # SSB 07430: Longyearbyen and Ny-Alesund, Barentsburg and Pyramiden, Hornsund


def _counts():
    import no2023
    df = pd.read_csv(NORM / "no.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 356 or int(df["count"].sum()) != SSB_2023:
        raise SystemExit(f"no.csv: {df['geo_id'].nunique()} kommuner, {df['count'].sum():,} "
                         "people -- run sources/no_ssb.py")
    sv = pd.read_csv(NORM / "no_svalbard.csv", dtype={"geo_id": str})
    if sv["geo_id"].nunique() != 3 or int(sv["count"].sum()) != SVALBARD_2026:
        raise SystemExit("no_svalbard.csv: run sources/no_svalbard.py")
    df = pd.concat([df, sv], ignore_index=True)
    df["node"] = df["source_category"].map(no2023.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Norway",
    source=("Statistics Norway, tables 07459 (population) and 09817 (immigrants and "
            "Norwegian-born to immigrant parents by country background), per kommune, 1 January "
            "2023; for Svalbard, tables 07430 and 12622 and SSB's citizenship figures of 1 January "
            "2026; Samisk språkundersøkelse 2012 (Solstad, ed.) for North Sami; home languages of "
            "other countries from their sources on this map"),
    how=("no language question: immigrants and their Norwegian-born children drawn on their "
         "country of background's languages, North Sami from a published estimate, everyone "
         "else Norwegian; Svalbard by citizenship"),
    parts=[
        dict(covers="Immigrants and their Norwegian-born children",
             source="Statistics Norway 2023, country of background, drawn on that country's "
                    "languages; 22% of the children drawn as Norwegian",
             people=1_043_966),
        dict(covers="North Sami speakers",
             source="Sami language survey 2012, about 10,000 speakers",
             nodes=["uralic.saami_north"]),
        dict(covers="Svalbard", source="Statistics Norway 2026, by settlement and citizenship",
             people=SVALBARD_2026),
        dict(covers="Everyone else", source="Statistics Norway 2023, drawn as Norwegian",
             rest=True),
    ],
    grain="356 kommuner, 15,000 people on average, and 3 Svalbard settlements",
    gap="36 people of unknown or tiny-territory background drawn as other",
    view=[4.5, 57.9, 31.2, 71.2],
    counts=_counts,
    mappings=["no2023"],
    place=ROOT / "data" / "geo" / "no" / "no_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Norway does not ask about language in any census or register. Statistics Norway "
        "counts immigrants and the Norwegian-born children of two immigrant parents in every "
        "kommune, by country of background, and they are drawn on that country's languages, "
        "except 22% of the children, drawn as Norwegian (a Swedish study's share, as no "
        "Norwegian figure exists). Everyone else is drawn as Norwegian, which covers both "
        "written standards, Bokmål and Nynorsk. About 10,000 people speak North Sami (the Sami "
        "language survey of 2012), drawn half in Kautokeino and Karasjok and half in the rest "
        "of the Sami language area. Lule and South Sami, a few hundred speakers each, are not "
        "drawn. Svalbard is drawn by citizenship at its settlements (Statistics Norway, 2026): "
        "Norwegians as Norwegian, the rest on their country's languages. Barentsburg's "
        "Ukrainians, mostly from the Donbas, are drawn at Donetsk oblast's 76% Russian."),
)
