"""Afghanistan: the village-majority language of each province, from the Ministry of Rural
Rehabilitation and Development's provincial profiles (c. 2006-07), on NSIA's 1404 (2025-26)
settled population -> data/normalized/af.csv.

    python sources/af_mrrd.py [--fetch]

WHY THIS SOURCE. Nothing open asks Afghans their first language by province (sources/af.md says
what was searched, twice). Anita allowed this proxy on 2026-10-05 (ask/017-af.md), knowing it is
rough, as a picture of what is spoken where.

THE TEXT. The MRRD / NABDP provincial profiles (written for the Provincial Development Plans,
c. 2006-07; the copies on nps.edu read word for word the same) were reprinted in the US Army
Center for Army Lessons Learned Handbook 11-16, *Afghanistan Provincial Reconstruction Team
Handbook*, Annex A (2011), which globalsecurity.org carries as one HTML page. Each province has a
sentence or two like "Dari is spoken by 77 percent of the population and 80 percent of the
villages. The second most frequent language is Uzbeki, spoken by the majorities in villages
representing 12 percent of the population." The figures count every villager under the language
of the village majority.

THE TRANSCRIPTION (PROFILES below) quotes each province's sentence(s) verbatim; --fetch, and every
run, asserts each quote occurs in the downloaded page, so a transcription slip fails loudly.
Figures come in four kinds, each turned into a share of the province:
  pct       a percentage of the population, as printed
  people    a head count, over the profile's own 2008 population for the province (POP2008)
  villages  a count of villages, over the total the profile gives (Kunar 771, Badghis 964); the
            proxy's own premise is that a village stands for its people
  ratio     Parwan's "outnumber Pashto speakers by a ratio of 5-to-2"
Where a province's shares sum past 100 (Laghman 100.3, Daykundi 104, Nimroz 108) they are scaled
to 100. Where they sum short of 100, the remainder is a `Not described` row: not drawn, counted
in the gap. A figure the profile gives for two or three languages at once is one row
("Dari and Pashtu, one figure") and taxonomy/af2007.py puts it on the narrowest node holding all
of them. Languages named with no figure, or "less than 1 percent", are left out (listed in NOTES).

KABUL. The profile gives Pashtu 60%, Dari 40% "of the population", but its figures are the
villages', and 87% of Kabul province (NSIA 1404) is Kabul city. The 60/40 split is applied to
NSIA's rural population only; the urban population is a `Kabul urban population` row, since
nothing here describes the city.

DARI/PASHTO SPLIT (2026-10-06, session 5d7dac7e-af). Kabul city and Herat's "Dari and Pashtu,
one figure" (7.7 million people) used to be drawn on the Iranian group node, which the viewer
washes out as "language not named": about 90% of the dots in Afghan cities. They are now split
into Dari and Pashto at one ratio, the residual that makes the drawn map's Dari:Pashto total
match the only open single-answer first-language figure for Afghanistan: the Asia Foundation's
*Afghanistan in 2006* survey, Q-45 "in which language did you learn to speak, first?" (single
response, 6,226 adults, all 34 provinces): Dari 49%, Pashto 40% (data/raw/af/taf_survey2006.pdf,
checked every run). Solving (D + d*U) : (P + (1-d)*U) = 49 : 40 over the drawn named rows gives
d = 0.93. It is a national figure spread over two unsplit places, not a measure of either; the
record (sources/af.md §3a) gives its sensitivity. Kandahar's and Helmand's "Balochi and Dari"
and Herat's "Turkmeni and Uzbeki" stay on their group nodes (0.2 million): the survey's Balochi
is 0% after rounding and gives no ratio worth using.

MINORITY LANGUAGES (2026-10-06, session 5d7dac7e-pam). The Pamiri languages of Badakhshan
(Shughni, Wakhi, Munji, Sanglechi, Ishkashimi), the Wakhan's Kyrgyz, Parachi (Kapisa),
Gawar-Bati (Kunar) and Brahui (the south), none named in the profiles, are carved out of their
provinces from cited speaker estimates (MINORITIES, BRAHUI below; sources/af.md §3b).

THE BASE. NSIA's Estimated Population 1404 settled population by province, rural and urban, as
religiondots joined it to COD-AB's 34 provinces (religiondots/data/geo/af/af_lookup.csv, read
only). 1.5 million Kuchis have no province there and are the gap, as in religiondots.
"""
import csv
import re
import sys
import html as htmlmod
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "af"
OUT = HERE / "data" / "normalized" / "af.csv"
LOOKUP = HERE.parent / "religiondots" / "data" / "geo" / "af" / "af_lookup.csv"
URL = "https://www.globalsecurity.org/military/library/report/call/call_11-16_appa.htm"
PAGE = RAW / "call_11-16_appa.htm"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
SOURCE_ID = "af_mrrd_profiles_c2007"
SETTLED_1404 = 34_935_197

# The Asia Foundation, Afghanistan in 2006: A Survey of the Afghan People, Appendix 3, Q-45
# (single response, base 6,226). Wayback copy of http://www.asiafoundation.org/pdf/AG-survey06.pdf
TAF_PDF = RAW / "taf_survey2006.pdf"
TAF_URL = "https://web.archive.org/web/2008id_/http://www.asiafoundation.org/pdf/AG-survey06.pdf"
TAF_FIRST = {"Dari": 49, "Pashtu": 40}
UNSPLIT = ("Kabul urban population", "Dari and Pashtu, one figure")
SPLIT_LABEL = {"Dari": "Dari, by the national first-language residual",
               "Pashtu": "Pashtu, by the national first-language residual"}

# The profile's own 2008 population, used only to turn head counts into shares.
POP2008 = {"AF02": 392_900, "AF10": 398_000, "AF11": 1_092_600, "AF12": 287_300,
           "AF13": 490_900, "AF14": 511_600, "AF23": 614_900, "AF25": 311_900}

# geo_id: (quote(s) that must occur in the page, [(label, value, kind)])
PROFILES = {
    "AF01": (["Pashtu is spoken by around 60 percent of the population, and Dari is spoken by "
              "around 40 percent."],
             [("Pashtu", 60, "pct"), ("Dari", 40, "pct")]),          # rural only, see KABUL
    "AF02": (["Dari is spoken by about 176,000 people and 304 villages",
              "The second language is Pashtu, spoken by 107,000 people",
              "A third language spoken by a sizeable portion of the population (17 percent) is "
              "Pashaie."],
             # the head counts, not the "30 percent" printed beside Dari: 107,000 is the 27%
             # printed for Pashtu, so the counts are the consistent figures
             [("Dari", 176_000, "people"), ("Pashtu", 107_000, "people"),
              ("Pashaie", 17, "pct")]),
    "AF03": (["Dari speakers outnumber Pashto speakers by a ratio of 5-to-2."],
             [("Dari", 5, "ratio"), ("Pashtu", 2, "ratio")]),
    "AF04": (["Pashtu, which is spoken by 70 percent of the population and Dari, which is "
              "spoken by 27 percent."],
             [("Pashtu", 70, "pct"), ("Dari", 27, "pct")]),
    "AF05": (["About two-thirds of villages and 60 percent of the population speak Pashto, and "
              "one-third of villages and 40 percent of the population speak Dari."],
             [("Pashtu", 60, "pct"), ("Dari", 40, "pct")]),
    "AF06": (["Pashtu is spoken by 92.1 percent of the villages.",
              "The remaining 8 percent speak Pashaie (60 villages), Dari (36 villages), and some "
              "other unspecified languages."],
             # 92.1% of villages; the other 7.9% split 60:36 by the village counts printed
             [("Pashtu", 92.1, "pct"), ("Pashaie", 7.9 * 60 / 96, "pct"),
              ("Dari", 7.9 * 36 / 96, "pct")]),
    "AF07": (["Pashto is spoken by 345 villages out of 620 and around 58 percent of the "
              "population.",
              "The second most frequent language is Pashaie spoken in 210 villages by a third of "
              "the population.",
              "Dari is spoken in 57 villages, representing just over 9 percent of the "
              "population."],
             [("Pashtu", 58, "pct"), ("Pashaie", 100 / 3, "pct"), ("Dari", 9, "pct")]),
    "AF08": (["The major ethnic group living in Panjshir are the Tajiks, along with a very small "
              "population of Pashtun Kuchis."],
             # no language sentence: the profile's ethnic group, Tajiks, drawn as Dari
             [("Dari (Tajiks, the profile's ethnic group)", 100, "pct")]),
    "AF09": (["Dari is spoken by 70 percent of the population and 73 percent of the villages.",
              "The second most frequent language is Pashtu, spoken by the majorities in 528 "
              "villages representing 22 percent of the population."],
             [("Dari", 70, "pct"), ("Pashtu", 22, "pct")]),
    "AF10": (["Dari is spoken by 96 percent of the population and 98 percent of the villages.",
              "In another 24 villages with a population of approximately 5,000, the main "
              "language spoken is Pashtu."],
             [("Dari", 96, "pct"), ("Pashtu", 5_000, "people")]),
    "AF11": (["Pashtu, which is spoken by about half of the population, and Dari, which is "
              "spoken by 47 percent of the population.",
              "Uzbeki is spoken by about 1,000 residents (0.1 percent), and about 23,000 people "
              "in 53 villages speak some other language."],
             [("Pashtu", 50, "pct"), ("Dari", 47, "pct"), ("Uzbeki", 1_000, "people"),
              ("Other language", 23_000, "people")]),
    "AF12": (["Pashtu is spoken by more than 96 percent of the population.",
              "Five villages with a total population of about 15,000 speak Uzbeki, and another "
              "four villages with a total population of about 5,000 people speak other "
              "languages."],
             [("Pashtu", 96, "pct"), ("Uzbeki", 15_000, "people"),
              ("Other language", 5_000, "people")]),
    "AF13": (["Pashtu is spoken by 97 percent of the population, Dari is spoken by 21,000 "
              "individuals, and around 1,000 individuals speak other languages."],
             [("Pashtu", 97, "pct"), ("Dari", 21_000, "people"),
              ("Other language", 1_000, "people")]),
    "AF14": (["Pashtu is spoken by 99 percent of the villages.",
              "Dari is spoken in two villages of approximatly 1,000 residents."],
             [("Pashtu", 99, "pct"), ("Dari", 1_000, "people")]),
    "AF15": (["Pashtu is spoken by 705 villages out of 771 and more than 90 percent of the "
              "population.",
              "Dari and Uzbeki are spoken in two villages each, Pashaie is spoken in 15 villages, "
              "and Nooristani in 35 villages."],
             [("Pashtu", 705 / 771 * 100, "pct"), ("Nuristani", 35 / 771 * 100, "pct"),
              ("Pashaie", 15 / 771 * 100, "pct"), ("Dari", 2 / 771 * 100, "pct"),
              ("Uzbeki", 2 / 771 * 100, "pct")]),
    "AF16": (["Nuristani is spoken by 78 percent of the population and 84 percent of the "
              "villages.",
              "The second most common language is Pashayi, spoken by the majorities in 39 "
              "villages representing 15 percent of the population."],
             [("Nuristani", 78, "pct"), ("Pashaie", 15, "pct")]),
    "AF17": (["Dari is spoken by 77 percent of the population and 80 percent of the villages.",
              "The second most frequent language is Uzbeki, spoken by the majorities in villages "
              "representing 12 percent of the population."],
             [("Dari", 77, "pct"), ("Uzbeki", 12, "pct")]),
    "AF18": (["The major ethnic groups living in Takhar province are Uzbek and Tajiks, followed "
              "by Pashtuns and Hazaras."],
             []),                                   # no language figure: not drawn
    "AF19": (["Pashtu, Dari, and Uzbeki are spoken by 90 percent of the population.",
              "A fourth language, Turkmeni, is spoken by 8 percent of the population."],
             []),                                   # one figure for three families: not drawn
    "AF20": (["Dari is spoken by more than 72.5 percent of the population.",
              "The second most frequent language is Uzbeki, spoken by 22.1 percent of the "
              "population."],
             [("Dari", 72.5, "pct"), ("Uzbeki", 22.1, "pct")]),
    "AF21": (["Dari is spoken by 50 percent of the population and 58 percent of the villages.",
              "The second most frequent language is Pashtu, spoken by the majorities in 266 "
              "villages representing 27 percent of the population, followed by Turkmani (11.9 "
              "percent) and Uzbeki (10.7 percent)."],
             [("Dari", 50, "pct"), ("Pashtu", 27, "pct"), ("Turkmeni", 11.9, "pct"),
              ("Uzbeki", 10.7, "pct")]),
    "AF22": (["It is spoken by 56 percent of the population.",
              "The second most frequent language is Uzbeki, spoken by 19 percent of the "
              "population."],
             [("Dari", 56, "pct"), ("Uzbeki", 19, "pct")]),
    "AF23": (["Dari is spoken by 97 percent of the population.",
              "The second most frequent language is Pashtu, spoken by about 15,000."],
             [("Dari", 97, "pct"), ("Pashtu", 15_000, "people")]),
    "AF24": (["Dari is spoken by 91 percent of the population and 85 percent of the villages.",
              "The second most frequent language is Pashtu, spoken by the majorities in 151 "
              "villages representing 13 percent of the population."],
             [("Dari", 91, "pct"), ("Pashtu", 13, "pct")]),
    "AF25": (["Pashtu is spoken by 90 percent of the population and 90 percent of the villages.",
              "The second most frequent language is Dari, spoken by the majorities in 46 villages "
              "and approximately 19,000 people."],
             [("Pashtu", 90, "pct"), ("Dari", 19_000, "people")]),
    "AF26": (["Pashto is spoken by four persons out of five."],
             [("Pashtu", 80, "pct")]),
    "AF27": (["Pashtu is spoken by more than 98 percent of the population.",
              "Balochi and Dari are spoken by a small portion of the population."],
             [("Pashtu", 98, "pct"), ("Balochi and Dari, one figure", 2, "pct")]),
    "AF28": (["Uzbek is spoken by the largest proportion of population (39.5 percent).",
              "Turkmen is second with 28.7 percent of the population.",
              "Pashtu and Dari are spoken respectively by 17.2 percent and 12.1 percent of the "
              "total population."],
             [("Uzbeki", 39.5, "pct"), ("Turkmeni", 28.7, "pct"), ("Pashtu", 17.2, "pct"),
              ("Dari", 12.1, "pct")]),
    "AF29": (["Uzbeki is spoken by over half (53.5 percent) of the population and 49 percent of "
              "the villages.",
              "The second most frequent language is Dari, spoken by the majorities in 311 "
              "villages representing 27 percent of the population.",
              "Pashtu is spoken by 17 percent of the villages and 13 percent of the population."],
             [("Uzbeki", 53.5, "pct"), ("Dari", 27, "pct"), ("Pashtu", 13, "pct")]),
    "AF30": (["Pashtu is spoken by 92 percent of the population.",
              "The second most frequent language is Dari, followed by Balochi."],
             [("Pashtu", 92, "pct"), ("Balochi and Dari, one figure", 8, "pct")]),
    "AF31": (["Dari, spoken by 56 percent of the population, and Pashto, spoken by 40 percent of "
              "the population, followed by Uzbeki, spoken by five out of 964 villages, Turkmani "
              "by four villages, and Balochi spoken by only one village."],
             [("Dari", 56, "pct"), ("Pashtu", 40, "pct"), ("Uzbeki", 5 / 964 * 100, "pct"),
              ("Turkmeni", 4 / 964 * 100, "pct"), ("Balochi", 1 / 964 * 100, "pct")]),
    "AF32": (["Dari and Pashtu are spoken by 98 percent of the population, with Turkmeni and "
              "Uzbeki spoken by the rest."],
             [("Dari and Pashtu, one figure", 98, "pct"),
              ("Turkmeni and Uzbeki, one figure", 2, "pct")]),
    "AF33": (["Dari is spoken by 50 percent of the population and 544 of the 1,125 total villages "
              "in the province.",
              "The second most frequent language is Pashtu, spoken by 48 percent of the "
              "population and 566 villages."],
             [("Dari", 50, "pct"), ("Pashtu", 48, "pct")]),
    "AF34": (["Baluchi is spoken by 61 percent of the population.",
              "The second most frequent language is Pashtu, spoken by 27 percent of the "
              "population, followed by Dari and Uzbeki each spoken by 10 percent of the "
              "population."],
             [("Balochi", 61, "pct"), ("Pashtu", 27, "pct"), ("Dari", 10, "pct"),
              ("Uzbeki", 10, "pct")]),
}

# Named in the profiles with no usable figure, so not drawn (sources/af.md):
NOTES = {
    "AF01": "Pashaie, 'a small number of people located in five villages'",
    "AF06": "'some other unspecified languages' inside Nangarhar's 8%",
    "AF15": "the 12 of 771 villages the sentence does not account for",
    "AF17": "Pashtu, Turkmeni and Nuristani, 'less than 1 percent each'",
    "AF19": "Turkmeni 8%: drawing it alone would show Kunduz as Turkmen",
    "AF24": "Turkmani (two villages) and Baluchi (one village)",
    "AF26": "Dari, second, no figure",
}


# MINORITY LANGUAGES THE PROFILES NEVER NAME (2026-10-06, session 5d7dac7e-pam; Anita: "the map
# shows essentially no Pamiri languages in Afghanistan"). Drawn on the ask-019 route (cited
# speaker estimates, rows `modelled`, sources/af.md §3b): each figure is carved out of rows of its
# own province, in the order listed, so every province still sums to NSIA's figure. The profiles
# count each village under its majority language, so a Pamiri, Parachi or Gawar-Bati village is
# in none of the named figures: those come out of the province's `Not described` remainder.
# countries/af.py keeps each one to its homeland inside the province (ZONES there).
#   label: (geo_id, speakers, [(row taken from, at most this share of that row)], source)
MINORITIES = {
    "Shughni (speaker estimate)": (
        "AF17", 20_000, [("Not described", 1)],
        "Endangered Language Alliance, Shughni: 'approximately 20,000' in Afghan Badakhshan; "
        "Rushani, a Shughni dialect (Glottolog rush1239), included"),
    "Wakhi (speaker estimate)": (
        "AF17", 17_500, [("Not described", 1)],
        "Wikipedia, Wakhi people, infobox: Afghanistan 17,500 (2018)"),
    "Munji (speaker estimate)": (
        "AF17", 5_300, [("Not described", 1)],
        "Ethnologue 18th ed. via Wikipedia, Munji language: 5,300 (2008)"),
    "Sanglechi (speaker estimate)": (
        "AF17", 2_200, [("Not described", 1)],
        "Ethnologue 25th ed. via Wikipedia, Sanglechi language: 2,200 (2009)"),
    "Ishkashimi (speaker estimate)": (
        "AF17", 1_500, [("Not described", 1)],
        "Wikipedia, Ishkashimi language: 1,500 in Ishkashim and Wakhan districts"),
    "Kyrgyz (speaker estimate)": (
        "AF17", 1_500, [("Not described", 1)],
        "Callahan, The Kyrgyz of the Afghan Pamir Ride On (2007): 1,500, 600 Big Pamir, "
        "900 Little Pamir"),
    "Parachi (speaker estimate)": (
        "AF02", 3_500, [("Not described", 1)],
        "Ethnologue via Wikipedia, Parachi language: 3,500 (2009), upper Nijrab"),
    "Gawar-Bati (speaker estimate)": (
        "AF15", 7_500, [("Not described", 1)],
        "Ethnologue 25th ed. via Wikipedia, Gawar-Bati language: Afghanistan 7,500"),
}
# Brahui: 200,000 in Afghanistan (Ethnologue, as quoted by E. Bashir, SALRC workshop notes on
# Brahui, 2003); "Helmand and Kandahar provinces: Chakhansoor to Shorawak" (Ethnologue 2016, via
# Joshua Project 10959). Split between the three provinces by the population of their southern
# belt as religiondots' Kontur hexes hold it (countries/af.py ZONES: Nimroz south of 31.0N
# outside Zaranj, Helmand south of 31.3N, Kandahar south of 31.0N west of 66.3E), figures
# computed 2026-10-06 and pinned here. Brahui are Baloch by identity and mostly bilingual in
# Balochi, so each province's share comes first out of its Balochi-type row (at most half of it),
# the rest out of Pashto.
BRAHUI = 200_000
BRAHUI_SOURCE = ("Ethnologue via Bashir (2003): 200,000 in Afghanistan; split by the "
                 "population of the southern belt")
BRAHUI_ZONE_POP = {"AF34": 41_001, "AF30": 243_832, "AF27": 26_076}
BRAHUI_TAKE = {"AF34": [("Balochi", 0.5), ("Pashtu", 1)],
               "AF30": [("Balochi and Dari, one figure", 0.5), ("Pashtu", 1)],
               "AF27": [("Balochi and Dari, one figure", 0.5), ("Pashtu", 1)]}


def carve(rows, geo_id, label, n, take, note):
    """Move n people of geo_id into a new `label` row, out of the rows named in `take`."""
    left = n
    for src, cap in take:
        hit = [r for r in rows if r["geo_id"] == geo_id and r["source_category"] == src]
        if not hit or left == 0:
            continue
        r = hit[0]
        k = min(left, int(r["count"] * cap))
        r["count"] -= k
        left -= k
    if left:
        raise SystemExit(f"{geo_id} {label}: {left:,} of {n:,} found no row to come out of")
    base = next(r for r in rows if r["geo_id"] == geo_id)
    rows.append(dict(base, source_category=label, count=n, basis="speaker estimate",
                     source_id="af_speaker_estimates", note=note))


def add_minorities(rows):
    for label, (g, n, take, src) in MINORITIES.items():
        carve(rows, g, label, n, take, src)
    zt = sum(BRAHUI_ZONE_POP.values())
    split = {g: round(BRAHUI * p / zt) for g, p in BRAHUI_ZONE_POP.items()}
    split["AF30"] += BRAHUI - sum(split.values())
    for g, n in split.items():
        carve(rows, g, "Brahui (speaker estimate)", n, BRAHUI_TAKE[g], BRAHUI_SOURCE)
    print("minorities: " + ", ".join(f"{lab.split(' (')[0]} {v[1]:,}" for lab, v in
                                     MINORITIES.items())
          + "; Brahui " + ", ".join(f"{g} {n:,}" for g, n in split.items()))
    return [r for r in rows if r["count"] > 0]


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        PAGE.write_bytes(r.read())
    print(f"fetched {PAGE} ({PAGE.stat().st_size:,} bytes)")


def check_taf():
    """Q-45's Dari and Pashto shares, read from the downloaded report, must be TAF_FIRST's."""
    if not TAF_PDF.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(TAF_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            TAF_PDF.write_bytes(r.read())
    import fitz
    text = " ".join(p.get_text() for p in fitz.open(TAF_PDF))
    text = re.sub(r"\s+", " ", text)
    m = re.search(r"in which language did you learn to speak, first\? \(Single response\) "
                  r"Pashto (\d+)% Dari (\d+)%", text)
    if not m:
        raise SystemExit("taf_survey2006.pdf: Q-45 not found")
    got = {"Pashtu": int(m.group(1)), "Dari": int(m.group(2))}
    if got != TAF_FIRST:
        raise SystemExit(f"taf_survey2006.pdf Q-45 reads {got}, expected {TAF_FIRST}")


def split_unsplit(rows):
    """Kabul city and Herat's joint Dari+Pashtu figure -> Dari and Pashtu at one ratio d, the one
    that brings the drawn map's Dari:Pashto total to Q-45's 49:40 (module docstring)."""
    dari = sum(r["count"] for r in rows if r["source_category"].startswith("Dari")
               and r["source_category"] not in UNSPLIT)
    pashto = sum(r["count"] for r in rows if r["source_category"] == "Pashtu")
    u = sum(r["count"] for r in rows if r["source_category"] in UNSPLIT)
    a, b = TAF_FIRST["Dari"], TAF_FIRST["Pashtu"]
    d = (a * (pashto + u) - b * dari) / ((a + b) * u)
    if not 0.5 < d < 1:
        raise SystemExit(f"Dari share of the unsplit rows {d:.3f}: outside 0.5-1, look again")
    out = []
    for r in rows:
        if r["source_category"] not in UNSPLIT:
            out.append(r)
            continue
        nd = round(r["count"] * d)
        for lab, c in (("Dari", nd), ("Pashtu", r["count"] - nd)):
            out.append(dict(r, source_category=SPLIT_LABEL[lab], count=c,
                            note=(r["note"] + "; " if r["note"] else "")
                            + f"{r['source_category']} split {d:.3f} Dari"))
    after_d = dari + sum(r["count"] for r in out if r["source_category"] == SPLIT_LABEL["Dari"])
    after_p = pashto + sum(r["count"] for r in out if r["source_category"] == SPLIT_LABEL["Pashtu"])
    print(f"split {u:,} unsplit Dari+Pashto people at Dari {d:.3f}: drawn Dari:Pashto "
          f"{after_d:,}:{after_p:,} = {after_d / after_p:.4f} (Q-45 {a}:{b} = {a / b:.4f})")
    assert abs(after_d / after_p - a / b) < 1e-3
    return out


def page_text():
    t = PAGE.read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = htmlmod.unescape(t)
    return re.sub(r"\s+", " ", t)


def shares(geo_id, items):
    out = []
    for label, value, kind in items:
        if kind == "pct":
            s = value
        elif kind == "people":
            s = value / POP2008[geo_id] * 100
        elif kind == "ratio":
            s = value / sum(v for _, v, k in items if k == "ratio") * 100
        else:
            raise ValueError(kind)
        out.append((label, s))
    total = sum(s for _, s in out)
    if total > 100:
        out = [(lab, s * 100 / total) for lab, s in out]
        total = 100.0
    return out, total


def main():
    if "--fetch" in sys.argv or not PAGE.exists():
        fetch()
    text = page_text()
    missing = [(g, q) for g, (qs, _) in PROFILES.items() for q in qs if q not in text]
    if missing:
        raise SystemExit("quotes not found in the page:\n" + "\n".join(f"  {g}: {q}" for g, q in missing))
    if len(PROFILES) != 34:
        raise SystemExit(f"{len(PROFILES)} provinces transcribed, expected 34")

    with open(LOOKUP, encoding="utf-8", newline="") as fh:
        lut = {r["geo_id"]: r for r in csv.DictReader(fh)}
    if set(lut) != set(PROFILES):
        raise SystemExit("religiondots' af_lookup.csv provinces differ from PROFILES")
    tot = sum(int(r["pop"]) for r in lut.values())
    if tot != SETTLED_1404:
        raise SystemExit(f"NSIA 1404 settled population {tot:,}, expected {SETTLED_1404:,}")

    rows, drawn = [], 0
    for g in sorted(PROFILES):
        r = lut[g]
        pop, rural, urban = int(r["pop"]), int(r["rural"]), int(r["urban"])
        assert rural + urban == pop, g
        items = PROFILES[g][1]
        base = rural if g == "AF01" else pop
        sh, total = shares(g, items)
        counts = [(lab, round(base * s / 100)) for lab, s in sh]
        if g == "AF01":
            counts.append(("Kabul urban population", urban))
        rest = pop - sum(c for _, c in counts)
        if rest > 0:
            counts.append(("Not described", rest))
        elif rest < 0:   # rounding only
            lab, c = counts[0]
            counts[0] = (lab, c + rest)
        for lab, c in counts:
            if c <= 0:
                continue
            rows.append(dict(geo_id=g, geo_level="province", geo_name=r["name"],
                             source_category=lab, count=c, basis="village majority",
                             year=2007, source_id=SOURCE_ID,
                             note=NOTES.get(g, "")))
            if lab != "Not described":
                drawn += c
        print(f"{g} {r['name']:<14} {pop:>9,}  described {min(total, 100):5.1f}%  "
              + ", ".join(f"{lab} {s:.1f}" for lab, s in sh))

    rows = add_minorities(rows)
    drawn += sum(v[1] for v in MINORITIES.values())    # out of `Not described`
    check_taf()
    rows = split_unsplit(rows)
    s = sum(x["count"] for x in rows)
    assert s == SETTLED_1404, s
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    nd = SETTLED_1404 - drawn
    print(f"wrote {OUT}: {len(rows)} rows, {SETTLED_1404:,} settled people, "
          f"{drawn:,} described, {nd:,} not described ({nd / SETTLED_1404:.1%})")


if __name__ == "__main__":
    main()
