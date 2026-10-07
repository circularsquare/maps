"""Language for Iran's provinces -> countries/iran/composition.json. A SURVEY, not a count.

Iran's census asks no language question. This is maps/languagedots' figure (READ-ONLY here; its
record is sources/ir.md): World Values Survey waves 5 (2005) and 7 (2020), language spoken at
home by about 4,200 adults, pooled per province and applied to each province's whole 1395 (2016)
census population, children included. Many provinces had 10 to 50 interviews, so a province's
mix can be well off. Province level only.

languagedots' geo_id is the English province name, which equals helper1m's COD-AB level-1 name
for all 31 provinces (both are the 1395 census provinces, Tabas in South Khorasan).

    C:\\Python39\\python.exe helper1m\\scripts\\iran\\language.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import language_common as lc  # noqa: E402

import pandas as pd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

POP_2016 = 79_926_270          # languagedots' countries/ir.py

SURVEY = ("World Values Survey 2005 and 2020 pooled, language at home, about 4,200 adults: a "
          "survey share per province, not a count")
TITLES = {
    "indoeuropean.iranian.persian": f"Persian. {SURVEY}. Undercounted in Mazandaran",
    "turkic.azerbaijani": f"Azerbaijani, with Iran's other Turkic languages such as Qashqai. {SURVEY}",
    "indoeuropean.iranian.kurdish": f"Kurdish, one answer on the survey card. {SURVEY}",
    "other": "Languages too small for a colour of their own, and the survey's bare 'other' "
             "(half of Golestan, where Turkmen was not on the 2020 card)",
}


def main():
    ir2020 = lc.ld_module("ir2020")
    df = pd.read_csv(lc.LD_NORM / "ir.csv")
    if df["geo_id"].nunique() != 31 or int(df["count"].sum()) != POP_2016:
        sys.exit(f"ir.csv: {df['geo_id'].nunique()} provinces, {df['count'].sum():,} people; "
                 f"expected 31 and {POP_2016:,}")

    comp = lc.Composition("iran")
    code = {p["name"]: c for c, p in comp.feat[1].items()}
    missing = sorted(set(df["geo_id"]) - set(code))
    unused = sorted(set(code) - set(df["geo_id"]))
    if missing or unused:
        sys.exit(f"province names: not in helper1m {missing}, no data for {unused}")
    lc.log("all 31 languagedots provinces match helper1m's level-1 names")

    for r in df.itertuples():
        node = ir2020.resolve(r.source_category)
        if node:
            comp.add(1, code[r.geo_id], node, r.count)

    comp.write(
        label="Language at home (survey)",
        year="2005-20",
        titles=TITLES,
        pop_year=2016,
        source=(
            "SURVEY. maps/languagedots' Iran figure: World Values Survey Iran waves 5 (2005) and "
            "7 (2020), language spoken at home, pooled per province (about 4,200 adults; many "
            "provinces 10-50 interviews), applied to each province's 1395 (2016) census "
            "population. Province level only."),
    )


if __name__ == "__main__":
    main()
