# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _si_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Slovenia is 58% forest and its municipalities do not know it. Kočevje is 556 km2 with
    the Kočevski Rog uninhabited inside it, Bovec and Bohinj are the Julian Alps, and the
    Kras units are limestone with the villages on the poljes. An equal share per polygon
    would put a serious part of the country on forest and rock; it also removes Lake Bohinj
    and Lake Cerknica, which sit inside their municipalities rather than between them.
    """
    return _kontur_place_weight(place, "si_hexes.gpkg", "sources/si_grid.py")


def _si_counts():
    """SURS Popis 2002 at občina: 7 nodes on 192 units.

    ONE level, no allocation, nothing modelled. Every drawn row is `measured` and may ring.
    si.csv also carries a country row and the fourteen-category national table, which are
    the same people twice more and are check levels rather than tiers.

    THE UNITS ARE THE 192 MUNICIPALITIES OF 2002 AND THE POLYGONS ARE GISCO'S COMMUNES
    2001, which is the enumeration's own boundary set: Slovenia stood at 192 from 1998 to
    2006 and is at 212 now, so a current file would put twenty holes in the map. No
    crosswalk is written because none is needed.

    THE DRAWN POPULATION IS 77.1% OF THE COUNTRY. 307,973 people (15.68%) did not want to
    answer a voluntary question and 139,097 (7.08%) were never asked one, because questions
    29 and 30 of the P-3 form had to be answered by the person themselves and no household
    member could answer them by proxy. A further 2,317 are in cells SURS withheld. None of
    the three is filled in (spec §3.5) and taxonomy/si2002.py records which way the hole
    leans.
    """
    from si2002 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "si.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 192:
        raise SystemExit(f"{df['geo_id'].nunique()} Slovenian units, expected 192")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "si": dict(
        name="Slovenia",
        source="Popis prebivalstva 2002, table 05W1006S (Statistical Office of the "
               "Republic of Slovenia)",
        basis="self-identification, whole resident population",
        view=[13.3, 45.4, 16.7, 46.95],
        note_public=(
            "**This is the 2002 census, and Slovenia has not asked since.** The 2021 "
            "census was register-based and carried no religion question, so there is no "
            "figure for Slovenia after 2002 at any geography, the national one included. "
            "Read every share here as a quarter of a century old. How fast it was moving "
            "when the counting stopped is at least on the record, because SURS published "
            "1991 beside 2002 on identical categories: Catholic identification fell from "
            "**71.6%** to **57.8%** in those eleven years, the atheist answer went from "
            "4.4% to **10.2%**, and Islam from 1.5% to **2.4%**. "
            "**More than a fifth of the country is not drawn, and that is the first thing "
            "to know about this map.** 307,973 people, **15.68%**, did not want to answer, "
            "and another 139,097, **7.08%**, were never established. Both are the "
            "question's doing rather than the country's. Declaring a religion is voluntary "
            "under article 41 of the constitution and the form said so; and questions 29 "
            "and 30 had to be answered by the person themselves, aged 14 or over, with no "
            "household member permitted to answer for them, so an enumerator who found "
            "nobody in left a prepaid envelope and 7.08% of Slovenia never sent it back. "
            "Between 1991 and 2002 the refusal nearly quadrupled, from 4.25% to 15.68%, "
            "while the never-established half halved, from 14.56% to 7.08%. "
            "**The hole is not spread evenly.** It runs from **3.1%** of Hodoš to "
            "**38.3%** of Žetale, and across the 192 municipalities it tracks the share "
            "reporting belief without a church (**+0.55**) and the atheist share "
            "(**+0.38**) while barely touching the Catholic share (+0.04). So the map "
            "understates those two answers slightly and overstates Protestantism, which is "
            "the one strong negative at −0.47. Nothing here corrects for that; the "
            "correction would be a number nobody published. "
            "**Protestantism in Slovenia is one corner of the country and very little "
            "else.** 16,135 people, **0.82%**, of whom the Evangelical Lutheran Church "
            "holds 14,736. On the map they are Prekmurje, along the Hungarian border: "
            "Hodoš is **84.9%** Protestant of what is drawn there, Gornji Petrovci "
            "**67.3%** and Puconci **59.8%**, and eight municipalities hold 11,432 of the "
            "15,855 the map can place. Prekmurje was Hungarian until 1919 and the "
            "Counter-Reformation never reached it the way it reached the rest of Slovenia. "
            "The same corner answered the census most fully, at 3.1% and 3.2% non-response "
            "in Hodoš and Gornji Petrovci. "
            "**The highest Muslim share and the highest Orthodox share in the country are "
            "the same municipality, and it is a steel town in the Alps.** Jesenice reads "
            "**22.7%** Muslim and **10.1%** Orthodox of what is drawn there, against 3.1% "
            "and 3.0% nationally, and 47.3% Catholic against 75.0%. Velenje's coal basin "
            "and the Zasavje towns are the same shape at 10.9%, 8.7% and 7.5% Muslim. At "
            "the other end of the range Piran is **26.0%** not-a-believer and Ljubljana "
            "**25.8%**, against 13.2% nationally, while Osilnica, 332 people on the Kolpa, "
            "is the one municipality where everyone the map draws is Catholic. "
            "**Two of the five religion boxes are catch-alls, and what is inside them was "
            "published for the country and for nowhere in it.** The Protestant cell is "
            "14,736 Evangelical Lutherans plus 1,399 others. `Other religions` is 3,831 "
            "people: 1,877 other Christians, 1,026 in a single box called Oriental "
            "religions, 558 other, 271 agnostics and **99** Jews, against 199 Jews at the "
            "1991 census. Both cells are drawn whole rather than split, because nothing "
            "published says which municipality any of those people were in."),
        how="census, 2002",
        grain="municipalities as they stood in 2002, 10,200 people on average",
        gap_share=0.2288,
        gap="447,070 people, 22.8%, who declined the religion question or whose answer was "
            "never established; and 2,222 more in cells the office withheld to protect "
            "small numbers",
        counts=_si_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "si" / "si_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_si_place_weight,
        note="THIS COUNTRY WAS CLOSED TWICE, CORRECTLY, ON THE WRONG TIER. sources.md §11c "
             "and §11k both struck Slovenia off because the 2021 census is register-based "
             "and does not ask religion, which is true and is why there is no recent "
             "figure. §11k downloaded the SURS PxWeb catalogue, recorded that it is 908 KB "
             "and live, and concluded there was nothing to draw; that catalogue contains "
             "05W1006S.px, religion by občina from Popis 2002. Finland was the same shape "
             "the same day. §9cg records it, and the check that would have caught both is "
             "free: tools/oracle.py's row for Slovenia already said 2002. "
             "THE MUNICIPALITY CODES IN THE RELIGION TABLE ARE NOT SLOVENIA'S MUNICIPALITY "
             "CODES. Its OBČINA dimension is an alphabetical sequence 001-193 in which 001 "
             "is SLOVENIJA, so joining on it shifts every unit by one place and every "
             "total still reconciles. sources/si.py recovers the real codes from a second "
             "table of the same census, 05W0405S.px, whose settlement labels carry them, "
             "and checks the name join on population: the two tables agree to the person "
             "on all 192 municipalities. "
             "THE FOURTEEN-CATEGORY TABLE IS NATIONAL AND DOES NOT EXIST AT OBČINA, so it "
             "is used as a check instead, and a strong one: every shared cell of "
             "05W1606S.px matches the drawn table exactly and its extra rows decompose "
             "both catch-alls to the person. "
             "AND THE HOLE IS ABOUT THE RELIGION QUESTION AND NOT ABOUT THE CENSUS. The "
             "same form asked ethnicity under the same voluntary, answer-for-yourself "
             "rule, and 05W1002S.px puts that non-response at 174,913 people, 8.9%, "
             "against 22.76% for religion. That comparison is not in si.csv and so is "
             "kept out of note_public.",
    ),
}
