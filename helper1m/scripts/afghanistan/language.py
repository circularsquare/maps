"""Language for Afghanistan's provinces -> countries/afghanistan/composition.json. A PROXY.

Afghanistan has had no census since 1979, and no open survey gives language below the national
level. This is maps/languagedots' figure (READ-ONLY here; its record is sources/af.md), which
Anita allowed there as "a vague picture of what is where":

  * the Ministry of Rural Rehabilitation and Development's provincial profiles (c. 2006-07),
    which give the share of each province's people living in villages where most people speak
    each language: everyone in a village counts under its majority language, so minorities in
    mixed villages vanish, and Hazaragi is inside Dari because the profiles never name it;
  * applied to NSIA's settled population for 1404 (2025-26), the same figures helper1m's 2025
    column holds;
  * Kabul city and Herat, which have no usable figure of their own, split 93/7 Dari/Pashto by
    the Asia Foundation's 2006 national first-language shares;
  * Pamiri languages, Kyrgyz, Parachi, Gawar-Bati and Brahui from published speaker estimates.

Province level only: the profiles are per province and nothing here splits them by district.
The part of each province the profile leaves undescribed is kept as its own grey group,
"Not described", so a pie's percentages are of the whole settled population. Takhar and Kunduz
are all "Not described" (their profiles give no usable figure).

languagedots' province ids AF01-AF34 come from religiondots' join of NSIA to COD-AB, the same
pcodes helper1m's COD-AB v03 provinces carry; main() checks every code against the name.

    C:\\Python39\\python.exe helper1m\\scripts\\afghanistan\\language.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import language_common as lc  # noqa: E402

import pandas as pd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SETTLED = 34_935_197          # NSIA 1404 settled population, languagedots' countries/af.py
NOT_DESCRIBED = "not_described"

# languagedots' spelling -> COD-AB v03's, where they differ.
SPELLING = {"Herat": "Hirat", "Helmand": "Hilmand", "Paktia": "Paktya", "Panjshir": "Panjsher",
            "Sar-e Pol": "Sar-e-Pul", "Jowzjan": "Jawzjan"}

PROXY = ("Village-majority language (MRRD provincial profiles, c. 2006-07) on NSIA's 1404 "
         "population: a proxy, not a count")
IR = "indoeuropean.iranian"
NAMES = {
    IR: "Balochi or Dari (one figure)",
    "turkic": "Turkmen or Uzbek (one figure)",
}
TITLES = {
    f"{IR}.dari": f"Dari, Hazaragi included (the profiles never name it). {PROXY}. Kabul city "
                  "and Herat are drawn 93% Dari by the Asia Foundation's 2006 national shares",
    f"{IR}.pashto": f"Pashto. {PROXY}. Kabul city and Herat are drawn 7% Pashto by the Asia "
                    "Foundation's 2006 national shares",
    IR: "Kandahar's and Helmand's profiles give one figure for Balochi and Dari together",
    "turkic": "Herat's profile gives one figure for Turkmen and Uzbek together",
    "dravidian.northern.brahui": "Brahui: a speaker estimate (Ethnologue, 200,000), not from the "
                                 "profiles, split over Nimroz, Helmand and Kandahar",
    "other": "Languages too small for a colour of their own, mostly speaker estimates placed in "
             "their home province",
}


def fold(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    af2007 = lc.ld_module("af2007")
    df = pd.read_csv(lc.LD_NORM / "af.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 34 or int(df["count"].sum()) != SETTLED:
        sys.exit(f"af.csv: {df['geo_id'].nunique()} provinces, {df['count'].sum():,} people; "
                 f"expected 34 and {SETTLED:,}")

    comp = lc.Composition("afghanistan")
    # Same pcode, same province: check every one by name.
    names = df.drop_duplicates("geo_id").set_index("geo_id")["geo_name"]
    for code, name in names.items():
        ours = comp.feat[1].get(code, {}).get("name")
        if ours is None or fold(SPELLING.get(name, name)) != fold(ours):
            sys.exit(f"{code}: languagedots {name!r}, helper1m {ours!r}")
    lc.log(f"all 34 province codes agree with helper1m's COD-AB v03 names")

    comp.add_extra(NOT_DESCRIBED, "Not described",
                   "The part of the province the profile's figures leave out (all of Takhar "
                   "and Kunduz, whose profiles give no usable figure)", "#d9d9d9")
    for r in df.itertuples():
        node = af2007.resolve(r.source_category)
        comp.add(1, r.geo_id, node if node else NOT_DESCRIBED, r.count)

    comp.write(
        label="Language (village-majority proxy)",
        year="c. 2007",
        names=NAMES,
        titles=TITLES,
        keep={NOT_DESCRIBED},
        pop_year=2025,
        source=(
            "PROXY. maps/languagedots' Afghanistan figure: the MRRD provincial profiles' "
            "village-majority language (c. 2006-07, reprinted in CALL Handbook 11-16 Annex A) as "
            "shares of NSIA's 1404 settled population by province; Kabul city and Herat split "
            "93/7 Dari/Pashto by the Asia Foundation's 2006 national first-language shares; "
            "Pamiri languages, Kyrgyz, Parachi, Gawar-Bati and Brahui from published speaker "
            "estimates. Province level only. 'Not described' is the part each profile leaves "
            "out, all of Takhar and Kunduz. Kuchi nomads (1.5 million) are not in it."),
    )


if __name__ == "__main__":
    main()
