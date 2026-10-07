"""Syria: the labels sources/sy_build.py writes (cited estimates, no census or survey asks) ->
language nodes. Every row is `modelled`. Record: sources/sy.md.

- Kurdish: Syria's Kurds speak Kurmanji; drawn on the shared `kurdish` node, as Iraq and Iran
  are (no source names a variety).
- Turkmen: Syria's Turkmen speak Anatolian Turkish dialects (the Gaziantep-Urfa type); a leaf of
  its own as Iraq's Turkmen are, since the estimate names the people, not Turkish.
- Aramaic: Hasakah's Assyrians and Syriacs, Turoyo and Assyrian Neo-Aramaic speakers together;
  the estimate does not split them, so it sits on Canada's leaf `afroasiatic.aramaic`.
- Western Neo-Aramaic (Glottolog west2763): Maaloula and Jubb'adin.
- Mesopotamian Arabic: the Euphrates and Jazira Arabic of Deir-ez-Zor, Raqqa and Hasakah
  (Glottolog nort3142 North Mesopotamian, whose Euphrates qeltu is Deir-ez-Zor's town speech, and
  meso1252's Euphrates cluster, the Bedouin-type rural dialects); one leaf, not Iraq's "Iraqi
  Arabic" node, whose name would read oddly in Syria.
"""
NAMES = {
    "Levantine Arabic": "afroasiatic.levantine_arabic",
    "Mesopotamian Arabic": "afroasiatic.mesopotamian_arabic",
    "Kurdish": "indoeuropean.iranian.kurdish",
    "Turkmen": "turkic.syrian_turkmen",
    "Aramaic": "afroasiatic.aramaic",
    "Western Neo-Aramaic": "afroasiatic.western_neo_aramaic",
    "Armenian": "indoeuropean.armenian.armenian",
}

EXTRA_NODES = []


def resolve(label):
    return NAMES.get(label)
