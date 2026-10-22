"""Afghanistan: the Kuchis (nomads), whom NSIA's settled population and MICS's frame leave out.

    python sources/af_kuchi.py        check the transcription against the downloaded page and write
                                      data/normalized/af_kuchi.csv

NSIA's 2025-26 estimate counts 1.5 million nomadic Kuchis outside the provinces' settled
population (countries/af.py KUCHI). Where they are: the Ministry of Rural Rehabilitation and
Development's Provincial Development Plan profiles (c. 2008) give, for each province, the Kuchi
population that winters there, from the 2003-04 National Multi-sectoral Assessment on Kuchi
(NMAK; the profiles say so in their Kuchi paragraphs). The profiles are reprinted in the US Army
CALL Handbook 11-16, Annex A (data/raw/af/call_11-16_appa.htm, the page af_mrrd.py already
downloads), and every figure below must occur verbatim there.

The winter figures sum to about 2.44 million, the NMAK's own total (2.4 million, "the overall
Kuchi population" the profiles quote shares of: Nangarhar 558,627 = 23%). NSIA counts 1.5
million as nomadic now, so the 1.5 million is spread by the provincial winter shares.

Language. Kuchis are Pashtun nomads: Pashto, drawn for all but one stated group, the Helmand
Kuchis "all belonging to the Baochi tribes" (the profile's spelling), 17% of Helmand's winter
Kuchis, who go on Balochi. That is the only language or ethnic statement about Kuchis by province
in the profiles; Baloch nomads also winter in Nimroz and Farah, not stated, so not drawn.
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "af" / "call_11-16_appa.htm"
OUT = ROOT / "data" / "normalized" / "af_kuchi.csv"

KUCHI = 1_500_000

# province id -> (winter Kuchi, a phrase that must occur in the profile text, province name)
WINTER = {
    "AF01": (49_754, "49,754 individuals", "Kabul"),
    "AF02": (29_286, "29,286 Kuchi stay in Kapisa", "Kapisa"),
    "AF03": (30_290, "30,290 Kuchis live in Parwan", "Parwan"),
    "AF04": (10_670, "10,670 individuals", "Maidan Wardak"),
    "AF05": (96_280, "96,280 individuals", "Logar"),
    "AF06": (558_627, "558,627 people", "Nangarhar"),
    "AF07": (94_020, "94,020 individuals", "Laghman"),
    "AF08": (0, "very small population of Pashtun Kuchis", "Panjshir"),
    "AF09": (97_500, "97,500 individuals", "Baghlan"),
    "AF10": (0, "no Kuchi stay there during winter", "Bamyan"),
    "AF11": (31_230, "just 31,230 individuals", "Ghazni"),
    "AF12": (51_074, "51,074 individuals", "Paktika"),
    "AF13": (9_588, "9,588 individuals", "Paktia"),
    "AF14": (104_965, "104,965 Kuchis living in Khost", "Khost"),
    "AF15": (13_200, "13,200 individuals", "Kunar"),
    "AF16": (3_160, "3,160 individuals", "Nuristan"),
    "AF17": (9_417, "9,417 individuals", "Badakhshan"),
    "AF18": (172_530, "172,530 individuals", "Takhar"),
    "AF19": (88_208, "88,208 individuals", "Kunduz"),
    "AF20": (14_150, "14,150 individuals", "Samangan"),
    "AF21": (52_929, "52,929 individuals", "Balkh"),
    "AF22": (59_843, "59,843 individuals", "Sar-e Pol"),
    "AF23": (0, "no Kuchi stay there during winter", "Ghor"),
    "AF24": (0, None, "Daykundi"),
    "AF25": (37_115, "37,115 individuals", "Uruzgan"),
    "AF26": (53_030, "53,030 Kuchi", "Zabul"),
    "AF27": (79_949, "79,949 individuals", "Kandahar"),
    "AF28": (76_850, "76,850 individuals", "Jowzjan"),
    "AF29": (98_220, "98,220 individuals", "Faryab"),
    "AF30": (95_325, "95,325 individuals", "Helmand"),
    "AF31": (115_100, "115,100 individuals", "Badghis"),
    "AF32": (98_506, "98,506 individuals", "Herat"),
    "AF33": (166_070, "166,070 individuals", "Farah"),
    "AF34": (29_500, "29,500 Kuchi stay in Nimroz", "Nimroz"),
}
BALOCH_HELMAND = 0.17      # "Seventeen percent are short-range, partially migratory (all belonging
                           # to the Baochi tribes)"


def page_text():
    import html
    t = RAW.read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return html.unescape(re.sub(r"\s+", " ", t))


def table():
    t = page_text()
    for pid, (n, phrase, name) in WINTER.items():
        if phrase and phrase not in t:
            raise SystemExit(f"af_kuchi: {pid} {name}: {phrase!r} not in {RAW.name}")
    if "Seventeen percent are short-range, partially migratory (all belonging to the Baochi tribes)" not in t:
        raise SystemExit("af_kuchi: Helmand's Baloch sentence not found")
    d = pd.DataFrame([(k, v[2], v[0]) for k, v in WINTER.items()], columns=["geo_id", "geo_name", "winter"])
    tot = d["winter"].sum()
    d["share"] = d["winter"] / tot
    d["kuchi"] = d["share"] * KUCHI
    return d, tot


def rows():
    """[(geo_id, label, people)]: Kuchis per province, Pashto except Helmand's Baloch 17%."""
    d, _ = table()
    out = []
    for r in d.itertuples():
        if r.kuchi <= 0:
            continue
        b = BALOCH_HELMAND if r.geo_id == "AF30" else 0.0
        out.append((r.geo_id, "Pashto", r.kuchi * (1 - b)))
        if b:
            out.append((r.geo_id, "Balochi", r.kuchi * b))
    return out


if __name__ == "__main__":
    d, tot = table()
    print(f"winter Kuchi, 34 provinces: {tot:,} (NMAK total 2.4 million); drawing {KUCHI:,}")
    print(d.sort_values("winter", ascending=False).head(8).to_string(index=False))
    r = pd.DataFrame(rows(), columns=["geo_id", "label", "people"])
    assert abs(r["people"].sum() - KUCHI) < 1e-6
    r.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(r)} rows, {r['people'].sum():,.0f} people")
