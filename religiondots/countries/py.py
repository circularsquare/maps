# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _py_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Paraguay needs this more than a compact country would, because half of it is empty.
    The Chaco is 60% of the national territory and held about 2% of the people in 2002, and
    Boqueron was a SINGLE district covering 91,000 km2 with 30,896 people aged 10 and over.
    An equal share per polygon smears those over an area the size of Portugal and puts the
    Mennonite colonies, which are the reason anyone looks at Boqueron here, 200 km from
    where they are (sources/py_grid.py).
    """
    return _kontur_place_weight(place, "py_hexes.gpkg", "sources/py_grid.py")


def _py_counts():
    """DGEEC 2002 variable P17 at distrito: 53 drawn categories on 224 units.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.

    THE UNIVERSE IS PEOPLE AGED 10 AND OVER, which is the census's own restriction on P17
    and not a choice made here. sources/py.py drops `No Aplica` (1,270,595 under-10s) before
    this function sees it, and asserts that what is left is 3,892,603 -- the UNSD
    Demographic Yearbook's figure for Paraguay 2002, to the person.

    ASUNCION IS SIX CENSUS DISTRICTS AND ONE POLYGON. The capital is tabulated as La
    Encarnacion, Catedral, San Roque, Lambare, Recoleta and Santisima Trinidad; every
    boundary source treats it as one unit. Their counts are summed onto `0000`, which is a
    real loss of resolution over 512,000 people and is why `grain` says so.

    `No especificado` IS THE ONLY THING NOT DRAWN: 37,206 people, 0.96%, carried as a §3.5
    residual. Paraguay is 99.04% drawn.
    """
    from py2002 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "py.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "distrito"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "py" / "py_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"py.csv districts with no polygon: {missing} -- re-run "
                         "sources/py_geo.py, the lookup is stale")
    if df["unit"].nunique() != 224:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 224")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "py": dict(
        name="Paraguay",
        source="Censo Nacional de Población y Viviendas 2002, variable P17 "
               "(DGEEC, now INE Paraguay)",
        basis="self-identification, population aged 10 and over",
        view=[-62.7, -27.7, -54.2, -19.2],
        gap="children under 10, whom the question was not asked",
        note_public=(
            "**Paraguay asked one religion question in 2002 and coded the answers into "
            "fifty-four categories, the longest list on this map.** It has not asked since. "
            "The 2022 census has no religion question at all, so this is both the best and "
            "the last picture of the country. "
            "**The headline is that Paraguay is more Catholic than almost anywhere here**, "
            "at **89.6%**, and in the central departments it is higher still: Paraguari "
            "96.2%, Cordillera 95.7%. Four districts are above 99%. Against that, the "
            "second-largest answer is a residual rather than a church, `Otras Evangelica` "
            "at **4.8%**, which sits at the end of a run of nineteen named Protestant "
            "bodies rather than in place of them. "
            "**The whole of the interesting variation is in the Chaco, and Boqueron is "
            "unlike any other unit on this map.** It was a single district in 2002, "
            "91,000 square kilometres with 30,896 people aged 10 and over, and it is "
            "**12.2% Mennonite**, **39.5% other evangelical** and **10.3% no religion**, in "
            "a country that is none of those things. That is the Fernheim and Menno "
            "colonies, settled from Russia and Canada from 1927, counted in a census that "
            "gave them their own line. "
            "**Indigenous religion is 0.6% nationally and 7.3% of Amambay**, on the "
            "Brazilian border, rising to 19.7% in Itanara and 13.2% in Ypehu. The census "
            "also counts five kinds of both at once, `Indigena + catolica`, `+ anglicana`, "
            "`+ evangelica`, `+ mennonita` and `+ otras`, which is an answer no other "
            "source drawn here offers. Those 1,478 people are drawn as indigenous, because "
            "a map that puts each person in one place cannot show them twice. "
            "**And the smallest categories find the Japanese colonies exactly.** Reyukai, a "
            "Japanese lay Buddhist movement, has 72 members in the whole country; Shinto "
            "has 30. The four districts with the highest Buddhist shares in Paraguay are La "
            "Paz (4.6%), Pirapo (2.9%), Yguazu (2.0%) and La Colmena (1.5%), which are the "
            "four agricultural colonies planted from Japan between 1936 and the 1950s. "
            "**One person in a hundred is not drawn**, the 37,206 who gave no answer."),
        how="census, 2002, people aged 10 and over",
        grain="districts, 17,400 people on average",
        fill="from the district the census counted them in; Asuncion is six census "
             "districts drawn as one",
        counts=_py_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "py" / "py_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_py_place_weight,
        note="FOUR EARLIER PASSES CLOSED PARAGUAY ON THE PUBLISHED OUTPUT AND THE "
             "MICRODATA TABULATOR WAS OPEN THE WHOLE TIME. §9k left it *unresolved*, §11t "
             "resolved the oracle row (2002, 12 categories) and never opened the office, "
             "and queue.md carried that forward. All of it was true of what DGEEC PRINTED: "
             "the 2002 library serves 52 national tables and 20 district ones, religion is "
             "in exactly one of them (CUADRO P11), and that one is national with an "
             "urban/rural split and four named categories. REDATAM has the variable itself "
             "at 229 districts with 54 categories. "
             "THE THING THAT HID IT WAS ONE CHARACTER. prod.redatam.org/binpry serves a "
             "4,611-byte portal page that looks empty, and the frame-walker written for the "
             "older R+SP servers finds nothing in it, because those emit <frame> and this "
             "emits <iframe>. A browser was launched to prove the deployment was dead "
             "before the regex was re-read. sources/py.py has the working POST. "
             "THE PARTITION IS EXACT THREE WAYS: every district's categories sum to its own "
             "printed total (229 of 229), the national figure is the UNSD oracle's "
             "3,892,603 to the person, and adding the under-10s gives the 2002 census "
             "population of 5,163,198 to the person. "
             "THE VINTAGE GAP TO THE PLACEMENT GRID IS TWENTY-ONE YEARS, the widest here, "
             "and Kontur still correlates r=0.959 with the census's own district sizes "
             "against a best of 0.488 over 500 shuffles. "
             "26 DISTRICTS CREATED AFTER 2002 are dissolved back into a 2002 parent by "
             "longest shared boundary (sources/py_geo.py); eleven of those calls are close "
             "and none of them moves a count, only where a dot lands inside a department. "
             "Pto. Pinasco is the one to look at: it counted 2,702 people and its "
             "reconstructed unit holds 40,056 in Kontur, 14.8x against a national 1.80.",
    ),
}
