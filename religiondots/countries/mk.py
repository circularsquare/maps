# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _mk_counts():
    """SSO Popis 2021 at municipality: 13 categories on 80 units.

    ONE level. mk.csv carries `country` as well, which is the same 1.84M people again.

    NO ALLOCATION, and none is possible: SSO publishes these categories at this geography
    and nothing finer or coarser, so every row is `measured` and may ring. Czechia's shape,
    for a much shallower table.

    THE DRAWN POPULATION IS 92.5% OF THE COUNTRY. Four categories resolve to nothing —
    the universe total, the 1,964 who declined, the 894 unknown, and the 132,260 people
    whose data came from administrative registers and who were never asked. That last one
    is 7.2% and is a coverage residual rather than a refusal; taxonomy/mk2021.py says why
    it is not irreligion.
    """
    from mk2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mk.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "mk" / "mk_lookup.csv",
                      dtype={"geo_id": str, "kod": str})
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    missing = df["unit"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} mk.csv rows have no LAU code -- re-run "
                         "sources/mk_geo.py, the lookup is stale")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "mk": dict(
        name="North Macedonia",
        source="Попис 2021 (State Statistical Office)",
        basis="self-identification",
        view=[20.4, 40.8, 23.1, 42.4],
        gap=("7.4%, nearly all of them taken from administrative registers that carry no "
             "religion question"),
        gap_share=0.07357,
        note_public=(
            "Two communities and a long thin tail. Orthodox Christians and people who "
            "answered simply 'Christian' are together 59% of the country and Muslims are "
            "32%, and both follow the ethnic map almost exactly — Orthodox where the "
            "population is Macedonian, Serb or Vlach, Muslim where it is Albanian, "
            "Turkish, Roma, Bosniak or Torbeš. **Read 'Orthodox' and 'Christian' "
            "together.** The census offered both and the choice between them turns out to "
            "be regional rather than doctrinal: in the eastern municipalities half the "
            "population wrote 'Christian' — 76% of Rosoman, 70% of Makedonska Kamenica — "
            "where in the west and in Skopje almost everyone wrote 'Orthodox'. Taken "
            "apart they draw a divide in eastern Macedonia that is about how people "
            "answered, not what they believe. That correlation is the thing to hold in "
            "mind while reading this one: at 80 municipalities it is close to being an "
            "ethnic map with religious labels, and the census asks for a religion rather "
            "than a church, so 847,000 Orthodox arrive with no jurisdiction attached and "
            "the Sunni and Bektashi of the west are not told apart. One category in nine "
            "is not a religion at all: 132,260 people, 7.2%, were taken from "
            "administrative registers rather than enumerated in person and carry no "
            "answer, so this map draws 92.5% of the country. Irreligion is 0.5%, among "
            "the lowest anywhere here."),
        how="census, 2021",
        grain="municipalities, 21,000 people on average",
        counts=_mk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mk" / "mk_opstini.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="80 municipalities is the source's ceiling for religion, not a choice: the "
             "same census publishes ethnicity by settlement and religion only by "
             "municipality. Refining religion inside a municipality from that ethnicity "
             "table is what spec §14.4 forbids, so the coarse grain stands "
             "(sources/mk.md §2).",
    ),
}
