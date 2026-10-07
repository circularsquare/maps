"""Hungary, Népszámlálás 2022, mother tongue (anyanyelv) -> node. sources/hu_census.py.

Keyed by KSH's own Hungarian labels from the WBS003 codelist (CL_TEL_SZ_ADAT), exactly as
data/normalized/hu.csv carries them. At settlement level KSH prints Hungarian, the languages of
the 13 recognised minorities (Romani and Boyash apart, so 14 labels), "other" and "no answer".

CALLS (sources/hu.md says more):
  "Cigány (beás) anyanyelvű" (Boyash, 5,558 drawn): a new leaf `indoeuropean.romance.boyash`
    beside Romanian. Boyash (Beás) is an archaic Romanian dialect spoken by Roma in Baranya,
    Somogy and Zala, not a Romani variety; Glottolog has no entry of its own for it (its
    Romanian dialects, under roma1327, include Banat Romanian but no Boyash). A sibling, not a
    child, of Romanian: a child would turn Romanian into a group, drawn washed out (cz.txt's
    Moravian, pl.txt's Lemko). The census prints it apart from Romanian and so does this map.
  "Cigány (romani) anyanyelvű" (Romani, 11,315): the leaf `romani.romani`, variety not stated
    (Romungro and Vlax Romani are both spoken in Hungary; the census does not say which).
  "Ruszin anyanyelvű" (Ruthenian, 1,651): Rusyn, as ro2021 and cz2021. Ukrainian is printed
    apart (13,046) and stays apart.
  "Horvát anyanyelvű" (Croatian, 6,838): Croatian. Hungary's Bunjevac and Šokac communities
    (Baja, Mohács) answer Croatian in the census; no label of their own.
  "Más anyanyelvű" (other mother tongue, 61,963): `other`. It holds every language that is not
    Hungarian or a recognised minority's: Russian, Chinese, Vietnamese, English, Arabic and the
    rest, mostly migrants. KSH does not split it at any level of this database, so it is not split.
  NOT_STATED: "Nem válaszolt az anyanyelv kérdésre" (1,175,656, 12.2%): not drawn, the gap.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic"

NAMES = {
    "Magyar anyanyelvű": "uralic.hungarian",
    "Német anyanyelvű": f"{IE}.germanic.continental.german",
    "Ukrán anyanyelvű": f"{SL}.east.ukrainian",
    "Ruszin anyanyelvű": f"{SL}.east.rusyn",
    "Szlovák anyanyelvű": f"{SL}.west.slovak",
    "Lengyel anyanyelvű": f"{SL}.west.polish",
    "Horvát anyanyelvű": f"{SL}.south.croatian",
    "Szerb anyanyelvű": f"{SL}.south.serbian",
    "Szlovén anyanyelvű": f"{SL}.south.slovenian",
    "Bolgár anyanyelvű": f"{SL}.south.bulgarian",
    "Román anyanyelvű": f"{IE}.romance.romanian",
    "Cigány (beás) anyanyelvű": f"{IE}.romance.boyash",
    "Cigány (romani) anyanyelvű": f"{IE}.indoaryan.romani.romani",
    "Görög anyanyelvű": f"{IE}.hellenic.greek",
    "Örmény anyanyelvű": f"{IE}.armenian.armenian",
    "Más anyanyelvű": "other",
}
NOT_STATED = {"Nem válaszolt az anyanyelv kérdésre"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"hu2022: unmapped label {label!r}")
    return NAMES[label]
