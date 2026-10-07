"""Taiwan 2020 Population and Housing Census, language learned earliest in childhood -> node.

Keyed by the census's own labels (DGBAS county reports, table 7, 兒時最早學會語言; the same five
labels head table 6, language used now, which is recorded but not drawn).

  * 國語 is Standard Mandarin, drawn on `mandarin` as everywhere else on the map.
  * 閩南語 (Taiwanese, Southern Min) goes on `min_nan`, whose label already reads "Taiwanese,
    Hokkien".
  * 客語 is Hakka, on `hakka`. The census does not split Sixian, Hailu and the other Hakka
    varieties.
  * 原住民族語 (indigenous languages) is every Formosan language and Yami together, named
    singly nowhere in the census: on the areal group `austronesian.taiwan`, then shared per
    township across the registered peoples (PEOPLES below; ask 012), the undeclared share
    staying on the group (tree.d/tw.txt says why it is a group).
  * 其他語言 is, in the table's own note, Taiwan Sign Language, "dialects of other places" (各地
    方言), foreign languages and other countries' sign languages: Sinitic, sign and foreign
    together, so the narrowest node holding it is the root `other`.
  * Not drawn: 不知或無 (does not know, or none), 60,672 people, in `gap`.
"""
ST = "sinotibetan.sinitic"

NAMES = {
    "國語": f"{ST}.mandarin",
    "閩南語": f"{ST}.min_nan",
    "客語": f"{ST}.hakka",
    "原住民族語": "austronesian.taiwan",
    "其他語言": "other",
}
EXCLUDED = {"不知或無"}

# The household register's peoples (族別; sources/tw_cip.py), each read as its own language, for
# sharing a township's 原住民族語 across the peoples registered there (ask 012, Anita 2026-10-05;
# countries/tw.py does the sharing, tier derived). 尚未申報 (people not yet declared) stays on the
# group node. 太魯閣 Truku and 賽德克族 Seediq are counted apart in the register and kept apart
# here, though Glottolog has Truku as a dialect of Seediq (tree.d/tw.txt).
TW = "austronesian.taiwan"
PEOPLES = {
    "阿美": f"{TW}.amis",
    "泰雅": f"{TW}.atayal",
    "排灣": f"{TW}.paiwan",
    "布農": f"{TW}.bunun",
    "魯凱": f"{TW}.rukai",
    "卑南": f"{TW}.puyuma",
    "鄒": f"{TW}.tsou",
    "賽夏": f"{TW}.saisiyat",
    "雅美": f"{TW}.yami",
    "邵": f"{TW}.thao",
    "噶瑪蘭": f"{TW}.kavalan",
    "太魯閣": f"{TW}.truku",
    "撒奇萊雅": f"{TW}.sakizaya",
    "賽德克族": f"{TW}.seediq",
    "拉阿魯哇族": f"{TW}.saaroa",
    "卡那卡那富族": f"{TW}.kanakanavu",
    "尚未申報": TW,
}
EXTRA_NODES = sorted(set(PEOPLES.values()))


def resolve(label):
    if label in EXCLUDED:
        return None
    return NAMES[label]
