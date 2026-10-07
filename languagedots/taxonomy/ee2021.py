"""Estonia, Rahvaloendus 2021, mother tongue (emakeel) -> node.
sources/ee_census.py; keyed by Statistics Estonia's English labels in RL21434, exactly as
data/normalized/ee.csv carries them.

Seventeen categories at the drawn level. CALLS (sources/ee.md says more):
  The 15 named languages: one leaf each, on the nodes every other country uses. No new nodes.
  Estonian includes Võro and Seto: the census files them as Estonian dialects and asks about
    them only in a separate dialect-knowledge table, which is ability, not mother tongue.
  "Other mother tongue" (15,848, 1.19%): `other`. RL21431's national list shows what it holds:
    229 languages, from Italian (1,048) and Swedish (818) to Estonian Sign Language (444),
    Romani (457), Ingrian (5) and Votic (3). Indigenous and foreign answers are mixed and the
    drawn table cannot tell them apart, so the bucket sits on `other`, not on a family.
  "Mother tongue unknown" (8,176, 0.61%): not drawn, in `gap`.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic"

NAMES = {
    "Estonian": "uralic.estonian",
    "Finnish": "uralic.finnish",
    "Russian": f"{SL}.east.russian",
    "Ukrainian": f"{SL}.east.ukrainian",
    "Belarusian": f"{SL}.east.belarusian",
    "English": f"{IE}.germanic.english",
    "German": f"{IE}.germanic.continental.german",
    "Latvian": f"{IE}.baltic.latvian",
    "Lithuanian": f"{IE}.baltic.lithuanian",
    "Spanish": f"{IE}.romance.spanish",
    "French": f"{IE}.romance.french",
    "Armenian": f"{IE}.armenian.armenian",
    "Azerbaijani": "turkic.azerbaijani",
    "Tatar": "turkic.tatar",
    "Other mother tongue": "other",
}
NOT_DRAWN = {"Mother tongue total", "Mother tongue unknown"}


def resolve(label):
    if label in NOT_DRAWN:
        return None
    if label not in NAMES:
        raise KeyError(f"ee2021: unmapped label {label!r}")
    return NAMES[label]
