# United Kingdom. Three censuses (sources/uk_census.py): England and Wales 2021 and Northern
# Ireland 2021 (ONS, NISRA), Scotland 2022 (NRS). Drawn on religiondots' own unit polygons
# (Output Areas in England, Wales and Scotland, Data Zones in Northern Ireland), which are the
# finest units published and the units every table here is on, so there is no placement layer
# and no population weight: one polygon per unit, as religiondots draws the UK.
from _shared import *  # noqa: F401,F403


def _counts():
    import uk2021
    df = pd.read_csv(NORM / "uk_units.csv")
    df["node"] = [uk2021.resolve(c, u) for c, u in zip(df["source_category"], df["unit"])]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="United Kingdom",
    source=("Census 2021 TS024 and main language (26 categories) by output area (ONS); "
            "Scotland's Census 2022 UV212 (NRS); Census 2021 main language by data zone and "
            "MS-B13 (NISRA); in Wales, Census 2021 Welsh speaking ability by output area (ONS); "
            "in Scotland, Census 2022 UV204 and UV204b country of birth by ward and output "
            "area, and Census 2011 AT_002 and AT_003, language used at home and country of "
            "birth (NRS), with England and Wales's Census 2021 country of birth (ONS); for "
            "Welsh, Gaelic, Scots and Irish, Annual Population Survey frequency of speaking "
            "Welsh by local authority (Welsh Government), Census 2022 UV208 and UV209 and "
            "Census 2011 AT_002 and Gaelic Report (NRS), and Census 2021 frequency of "
            "speaking Irish by data zone, and by full-time education by district electoral "
            "area, super data zone and data zone (NISRA)"),
    how=("census, 2021 and 2022, main language; Welsh, Gaelic, Scots and Irish in Northern "
         "Ireland drawn at daily or home use instead, placed on each area's speakers; "
         "Scotland's 'other language' split by country of birth"),
    grain="239,023 output areas and data zones, 270 people on average",
    gap="children under 3, who were not asked, 2.1 million",
    view=[-8.7, 49.8, 2.0, 61.0],
    counts=_counts,
    mappings=["uk2021"],
    parts=[
        dict(covers="Welsh in Wales",
             source="2021 census speakers who speak it daily, by the Annual Population "
                    "Survey's share in each local authority", people=306_466),
        dict(covers="Gaelic in Scotland",
             source="2011 census, used at home, placed on 2022 speakers", people=24_974),
        dict(covers="Scots in Scotland",
             source="2011 census, used at home, placed on 2022 speakers", people=55_817),
        dict(covers="Irish in Northern Ireland",
             source="2021 census, speaks Irish daily, less the full-time students and "
                    "pupils among them by area", people=28_817),
        dict(covers="Wales, everyone else answering English or Welsh",
             source="2021 census, drawn as English", people=2_611_215),
        dict(covers="Scotland, English and sign",
             source="2022 census, main language, aged 3 and over", people=4_940_943),
        dict(covers="Scotland, other languages",
             source="2022 census, split by country of birth and the 2011 and England and "
                    "Wales censuses' languages", people=272_820),
        dict(covers="Northern Ireland, everyone else",
             source="2021 census, main language, aged 3 and over", people=1_807_792),
        dict(covers="England and Wales, other answers",
             source="2021 census, main language, aged 3 and over", rest=True),
    ],
    # uk_units.gpkg cut by Kontur hexes, so an area's dots follow where its people live
    # (sources/kontur_cut.py; 2026-10-08)
    place=GEO / "uk" / "uk_konturcut.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Three censuses asked the same question, what is your main language, of everyone "
        "aged 3 and over: England, Wales and Northern Ireland in 2021, Scotland in 2022.\n\n"
        "Welsh, Scottish Gaelic, Scots and Irish are drawn for the people who speak them "
        "daily or at home, which is how Irish is drawn in Ireland. Almost every speaker of "
        "these languages also speaks English, and few name them as their main language, so "
        "main language alone would show very few.\n\n"
        "Welsh: in Wales the form had one box for English or Welsh. The census counts 538,000 "
        "people in Wales who can speak Welsh. In each local authority the Annual Population "
        "Survey (2019 to 2022) gives the share of Welsh speakers who speak it daily, from 87% "
        "in Gwynedd to 32% in Torfaen, and that share of each output area's speakers is drawn "
        "as Welsh: 306,000 people. Everyone else who ticked the box is drawn as English.\n\n"
        "Scottish Gaelic: 25,000 people, the number who said in the 2011 census that they use "
        "Gaelic at home. They are placed on the 2022 census's 70,000 Gaelic speakers, using "
        "the 2011 share of speakers who used it at home: 74% in the Western Isles, 42% in "
        "Highland, 33% in Argyll and Bute and 24% elsewhere. Only 3,500 named Gaelic as their "
        "main language in 2022.\n\n"
        "Scots: 56,000 people, the 2011 census's count of people using Scots at home, placed "
        "on the 2022 census's 1.5 million Scots speakers. 13,500 named it as their main "
        "language in 2022.\n\n"
        "Irish in Northern Ireland: 29,000 people. The 2021 census counted 44,000 who speak "
        "Irish daily, but the question did not leave out school, and 15,000 of them are pupils "
        "or students. Those 15,000 are taken out in the areas where they live and drawn as "
        "English, to match Ireland's count of daily speakers outside education. Pupils at "
        "Irish-medium schools who also speak it at home are taken out with them. 6,000 named "
        "Irish as their main language.\n\n"
        "England and Wales name 26 languages or groups for each output area and 95 languages "
        "for each local authority. Inside an output area a group such as other South Asian "
        "languages or African languages is shared among its languages in the local "
        "authority's proportions, so Somali, Tigrinya, Romanian and Lithuanian are placed by "
        "district rather than by output area.\n\n"
        "Scotland's census names English, Scots, Gaelic and sign language and puts the other "
        "273,000 people in one group, other language. They are shared out here by country of "
        "birth, which the census does publish by ward. Each birthplace is given its home "
        "country's languages, then two older or neighbouring censuses say how many such "
        "people, with their children, report each language: Scotland's 2011 census, which "
        "named 180 languages used at home, and England and Wales's 2021 census, which asked "
        "the same main language question. That predicts 260,000 people with another main "
        "language in Scotland, close to the 273,000 counted. Polish is the largest, then "
        "Spanish, Punjabi, Arabic and Urdu. Every one of these dots is an estimate; the "
        "number in each output area is the census's own. Scotland's 2011 census also showed "
        "that most people who write Chinese do not name Mandarin or Cantonese, so many "
        "Chinese dots stay unnamed.\n\n"
        "Cornish, Manx and Ulster Scots are drawn as main language only, under 600 "
        "people each; no census asks how often they are used.\n\n"
        "Northern Ireland names 19 "
        "languages for each data zone; its other 14,000 people are shared out in the "
        "province's proportions.\n\n"
        "The census counts Sylheti, the home language of most British Bangladeshis, as "
        "Bengali."),
)
