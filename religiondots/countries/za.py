# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _za_place_weight(place):
    """countries.py hook. `place` is the 400 m Kontur hex layer scatter.py has read.

    It mattered more at nine provinces than it does at 213 municipalities, and that is the
    right way round. The units are now a median 123,419 people and 2,861 km², so the grid is
    doing ordinary within-unit work rather than carrying the whole geography. It is still
    needed, because South African municipalities are fine in people and wild in area: Dawid
    Kruiper is 44,231 km² with 107,161 people, Mandeni is 545 km² with 147,808. An equal
    share over the outlines would still draw the Karoo and the Kalahari as populous as
    the coast.

    ONE THING FOR WHOEVER READS sources/za_grid.py's OUTPUT. Kontur's own population per
    municipality tracks the drawn population closely (log-log r=0.9815 against a best of
    0.2253 over 500 shuffles), but two municipalities are badly out: Matatiele reads 8.8x and
    Mtubatuba 3.6x, on hexes that are neither duplicated nor mis-joined. That is Kontur being
    wrong there, and it changes nothing, because this weighter normalises inside each unit —
    a municipality's dot count comes from the survey and the grid only decides where in it
    they land. It is worth knowing before anyone reads that layer as a population estimate.
    """
    return _kontur_place_weight(place, "za_hexes.gpkg", "sources/za_grid.py")


def _za_counts():
    """Stats SA Community Survey 2016 person microdata: 25 categories on 213 municipalities.

    ONE level, no allocation, nothing spread — every row is `measured` and may ring. Each
    person drawn is one CS 2016 respondent's own answer, weighted by Stats SA's own person
    weight, in the municipality that respondent was enumerated in.

    THIS WAS NINE PROVINCES UNTIL 2026-09-09 AND THE UNIT CHANGE IS THE WHOLE POINT. The
    published provincial profiles are the finest OPEN tabulation of this variable and 6.1M
    people per unit was the coarsest counting geography on the map. Anita registered with
    DataFirst and downloaded catalogue 611 (ask/answered/002-za), so the same survey is now
    read per person: 3,328,867 records, ~261,000 people per municipality, the same 24
    published categories. The nine profiles did not go to waste — sources/za_profiles.py
    still parses them and sources/za.py refuses to write unless every published province
    cell reproduces, which 215 of 216 do to within a person.

    THE SOURCE IS THE SURVEY AND NOT THE 2022 CENSUS, DELIBERATELY. The census publishes
    religion for the nine provinces over eleven categories, openly, with `Christianity` as
    one undivided cell holding 83.6% of the country, and at no finer unit. CS 2016 splits
    Christianity fourteen ways, which on the country that is the historic home of the
    African Independent Churches is the whole reason to draw it. The trade is six years and
    a survey against a census, §3.1 forbids mixing them, and this map has taken categories
    over vintage every time it has been asked (Benin, Trinidad, Paraguay, Türkiye).

    A SURVEY AT THIS TIER NEEDS ITS ADEQUACY STATED AND CS 2016 STATES IT. Report 03-01-07
    §1.2.2: *"At enumeration area (EA) level, all in-scope EAs were included in the sample
    and a sample of dwelling units was taken within each EA (i.e. there was no subsampling
    of EAs)."* Every EA in the country is in the sample, so no municipality is represented
    by a neighbour's households; §1.2 calls the survey *"one of the few available data
    sources providing data at municipal level"*. The smallest municipal sample is 529
    records (Prince Albert) against a median of 7,890, and 0.107% of the people drawn are in
    a (municipality, category) cell resting on fewer than ten records. Every row of
    data/normalized/za.csv carries its own `cell_n`.

    TWO CATEGORIES BOTH CALLED `Other` mean different things — one a religion, one a
    Christian denomination — so sources/za.py emits every row prefixed and this function
    would resolve them wrongly against a bare label. The prefix is asserted below.
    """
    from za2016 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "za.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "municipality"].copy()
    cats = set(df["source_category"])
    if not {"Religion: Other", "Christian: Other"} <= cats:
        raise SystemExit("za.csv has lost its `Religion: `/`Christian: ` prefixes -- the "
                         "two `Other` rows are about to collapse into one node and 1.5M "
                         "people of other faiths would be drawn as Christians")

    lut = pd.read_csv(HERE / "data" / "geo" / "za" / "za_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"za.csv municipalities with no polygon: {missing} -- re-run "
                         "sources/za_geo.py, the lookup is stale")
    if df["unit"].nunique() != 213:
        raise SystemExit(f"{df['unit'].nunique()} municipalities, expected 213")

    df["node"] = df["source_category"].map(resolve)
    # `Religion: Do not know` and `Religion: Unspecified` resolve to None on purpose --
    # za2016.EXCLUDED, 707,295 people, the `gap` below. They are in the normalised file so
    # tools/gap_share.py can compute the share instead of it being authored, and they are
    # dropped here.
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"])
                        - {"Religion: Do not know", "Religion: Unspecified"})
    if unresolved:
        raise SystemExit(f"za.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "za": dict(
        name="South Africa",
        source="Community Survey 2016 person microdata (Statistics South Africa, "
               "DataFirst catalogue 611)",
        basis="self-identification, whole survey population",
        view=[16.2, -35.0, 33.1, -22.0],
        gap=("707,295 people, 1.3%, who answered 'do not know' to the religion question "
             "or gave no answer at all"),
        gap_share=0.0127,
        note_public=(
            "**One answer in four in South Africa is an African Independent Church, and "
            "nowhere else on this map comes close.** Stats SA counted **14,158,461** "
            "people in a single box covering the Zion Christian Church, the Apostolic "
            "churches and Shembe's Nazaretha; that is 25.8% of everyone who answered and "
            "32.6% of the country's Christians. Drawn municipality by municipality it is "
            "not a tendency, it is a majority: **61.98%** of every answer given in Big "
            "Five Hlabisa and **61.62%** in Mfolozi, both in northern KwaZulu-Natal, and "
            "75.6% of Mthonjaneni's Christians against 4.6% of Kai !Garib's in the "
            "Northern Cape. One cell holds thousands of separate churches, so none of them "
            "can be told apart here. "
            "**The mission churches still sit where the missions were, and a province was "
            "too coarse to show it.** Methodists are **33.8%** of Ntabankulu's Christians "
            "and 33.4% of Umzimvubu's, both in the old Transkei, against 0.2% in Limpopo's "
            "Greater Giyani; the Eastern Cape as a whole is 15.2%, which is that Methodist "
            "Transkei averaged with a coast that is Anglican and Reformed. The Dutch "
            "Reformed family is **41.9%** of Bergrivier's Christians and 39.1% of "
            "Matzikama's, the Swartland and the Olifants River valley, against 0.2% in "
            "Mfolozi. The Scottish Presbyterian mission shows up where it was planted: "
            "11.1% of Raymond Mhlaba, which contains Alice and Lovedale. "
            "**These are 2016 figures and the 2022 census tells a different story, which "
            "is why the two are not mixed.** The census leaves Christianity undivided and "
            "publishes religion for the nine provinces only, so it cannot show any of the "
            "above, but it also disagrees about the country: this survey found "
            "**5,964,889** people with no religious affiliation, 10.9% of the answers "
            "given, where the 2022 census counted 1,760,784 out of the 62.0 million people "
            "it enumerated. Six years does not do that. The two questions were put "
            "differently, and only one of them can be on screen at a time. "
            "**No religious affiliation is not the urban answer a reader expects.** It is "
            "**32.7%** of Okhahlamba in KwaZulu-Natal and 27.7% of Ephraim Mogale in "
            "Limpopo, both rural and both in the provinces where the African Independent "
            "Churches are strongest, against 14.7% in Johannesburg and 7.8% in Cape Town. "
            "Where church membership is the dominant idiom, someone outside a particular "
            "church may answer 'none' rather than name an ancestral practice; nothing in "
            "the survey settles that. "
            "**KwaDukuza and Durban hold nearly all of South Africa's Hindus.** Hinduism "
            "is **9.2%** of KwaDukuza's answers and 9.0% of eThekwini's and close to zero "
            "in most of the country, which is the population descended from the indentured "
            "labourers brought to the Natal sugar estates from 1860. Islam runs the other "
            "way, **8.4%** of Cape Town against nothing at all in much of the Eastern "
            "Cape, and the two ends are different communities: the Cape figure is largely "
            "Cape Malay, descended from people exiled and enslaved from the Dutch East "
            "Indies. "
            "**707,295 people answered no religion question at all, and they are not "
            "drawn.** That is **1.3%** of the survey, recorded as 'do not know' or left "
            "blank, and Stats SA leaves them out of every religion total it publishes. "
            "They are not spread evenly: non-response runs from nothing at all in five "
            "municipalities to **8.4%** in eDumbe, and across the 213 it tracks the "
            "no-religion share (r = +0.31, permutation p under 0.0001). So the people left "
            "out come disproportionately from the least religious places, every share here "
            "is slightly more religious than South Africa is, and the correction is not "
            "made. "
            "**African traditional religion is 4.5%, and read that as a floor.** The box "
            "is exclusive of the Christian ones, and consulting a sangoma or honouring the "
            "ancestors commonly accompanies church membership here rather than replacing "
            "it. The floor is much higher in the KwaZulu-Natal midlands, **31.4%** of "
            "Maphumulo and 30.8% of Mkhambathini, than nine provinces could show. "
            "**Two municipalities return something a province tabulation hid, and neither "
            "is corrected.** uPhongolo, rural KwaZulu-Natal, reports **4.8%** atheists "
            "where the country reports 0.1%, which puts an eighth of every atheist counted "
            "in South Africa in one municipality; Swellendam reports **57.3%** of its "
            "answers as 'just a Christian' where its province reports 7.1%. Both rest on "
            "hundreds of interviews rather than one household, and in both the other "
            "categories fall short by about as much as those rise. That is the shape of an "
            "enumeration team's habit rather than a town, and it is drawn as returned."),
        how="household survey, 2016, read per person",
        grain="local municipalities, 258,000 people on average",
        counts=_za_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "za" / "za_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_za_place_weight,
        note="REDRAWN 2026-09-09 FROM 9 PROVINCES TO 213 LOCAL MUNICIPALITIES. The source is "
             "the CS 2016 person microdata, DataFirst catalogue 611, 3,328,867 records; the "
             "nine published provincial profiles that used to BE this country are now the "
             "check, and sources/za.py will not write unless all 216 published province x "
             "category cells reproduce. 215 do, to within a person. THE 216th IS THE ONE "
             "GOOD REASON THIS TIER WAS WORTH IT BEYOND RESOLUTION: North West's Report "
             "03-01-11 prints an `Other` denomination cell of 21 873 and its fourteen rows "
             "fall 336,482 short of its own total, and the microdata puts that cell at "
             "358,355, which is 21,873 + 336,482 exactly. The row was mis-set, the province "
             "build's second hypothesis (a `Not applicable` universe) is wrong, and those "
             "336,482 people are now drawn as Christians of another denomination instead of "
             "sitting on bare `christianity`. sources/za.py asserts that defect is still in "
             "the PDF, because the reconciliation expects it. "
             "SAMPLING ADEQUACY AT THIS TIER IS THE SURVEY'S OWN CLAIM, not an assumption "
             "made here: Report 03-01-07 §1.2.2 says every in-scope enumeration area was in "
             "the sample with no subsampling of EAs, and §1.2 calls CS 2016 'one of the few "
             "available data sources providing data at municipal level'. Smallest municipal "
             "sample 529 records, median 7,890; 0.107% of the people drawn sit in a cell "
             "resting on fewer than ten records, and every row of data/normalized/za.csv "
             "carries its own cell_n so a reviewer can check any figure quoted anywhere. "
             "THE JOIN IS A CODE JOIN, not a name join: the microdata and COD-AB ADM3 carry "
             "the same MDB municipality codes and match 213/213 both ways, checked "
             "independently against district (52), province (9) and name (210 of 211 "
             "testable). Two names come from COD because Stats SA's own labels are unusable "
             "-- LIM345 is labelled the literal word `New` and is Collins Chabane, and "
             "NC067's label is corrupt in the .dta itself. "
             "THE VINTAGE IS THE 2016 DEMARCATION AND THE FILE CARRIES BOTH: MN_CODE_2016 "
             "has 213 municipalities and MN_CODE_2011 has 234, Stata truncates the label-set "
             "names so neither is called after its variable, and taking the wrong one "
             "resolves silently to the wrong municipalities with every total still "
             "reconciling. sources/za.py picks by size and then confirms on the label text.",
    ),
}
