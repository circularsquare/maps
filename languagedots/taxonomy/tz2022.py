"""Tanzania, Afrobarometer R4 and R6-R9 (2008-2022), home language -> node.

Keyed by the answers sources/tz_afro.py writes to data/normalized/tz.csv: the card's labels
(CODED there, "Ki-" dropped and spellings merged) and the languages its "Other (specify)" free
text names (VERBATIM there). The tree, and the Glottolog check of every branch, is
taxonomy/tree.d/tz.txt.

  * Chaga is one answer on the card for Glottolog's Chaga subgroup (Machame, Vunjo, Rombo...);
    a leaf, as Kenya's Luhya.
  * Digo (free text, Tanga) is on Kenya's Mijikenda leaf: Digo is a Mijikenda language and
    Kenya counts its Digo answers there.
  * Meru here is Rwa (rwaa1238), the Bantu language of Mount Meru in Arusha, not Kenya's Meru.
  * Ngoni here is Tanzanian Ngoni (tanz1241), not the Nyanja-Sena Ngoni of Zambia and Malawi.
  * Arusha is the Maa of the Arusha people; the card names it beside Maasai, so a sibling.
  * Shirazi (the card's Kishirazi) is Zanzibar Swahili under the speakers' own name; a sibling
    of Swahili, as Bajuni is.

Remainders:
  * "Other African language": a language named in free text by one respondent, or a word not
    identifiable as a language (sources/tz_afro.py lists them). `africa_other`.
  * "Other language": "Kihindi" (Indian), one respondent. `other`.
"""
BA = "nigercongo.bantu"
MN = f"{BA}.mambwe_nyiha"
NI = "nilosaharan.nilotic"
SC = "afroasiatic.cushitic.south"

NAMES = {
    "Swahili": f"{BA}.swahili",
    "Shirazi": f"{BA}.shirazi",
    "Bajuni": f"{BA}.bajuni",
    "Digo": f"{BA}.mijikenda",
    "Segeju": f"{BA}.segeju",
    # Lake and west
    "Sukuma": f"{BA}.sukuma", "Nyamwezi": f"{BA}.nyamwezi", "Sumbwa": f"{BA}.sumbwa",
    "Kimbu": f"{BA}.kimbu", "Konongo": f"{BA}.konongo",
    "Ha": f"{BA}.ha", "Hangaza": f"{BA}.hangaza", "Shubi": f"{BA}.shubi",
    "Haya": f"{BA}.haya", "Nyambo": f"{BA}.nyambo", "Zinza": f"{BA}.zinza",
    "Kerewe": f"{BA}.kerewe", "Tongwe": f"{BA}.tongwe", "Bembe": f"{BA}.bembe",
    "Manyema": f"{BA}.manyema",
    # Mara
    "Jita": f"{BA}.jita", "Kwaya": f"{BA}.kwaya", "Kara": f"{BA}.kara",
    "Zanaki": f"{BA}.zanaki", "Ikizu": f"{BA}.ikizu", "Ikoma": f"{BA}.ikoma",
    "Kabwa": f"{BA}.kabwa", "Suba-Simbiti": f"{BA}.suba_simbiti", "Kuria": f"{BA}.kuria",
    # centre
    "Nyaturu": f"{BA}.nyaturu", "Nyiramba": f"{BA}.nilamba", "Rangi": f"{BA}.rangi",
    "Mbugwe": f"{BA}.mbugwe", "Gogo": f"{BA}.gogo",
    # east and north-east
    "Kaguru": f"{BA}.kaguru", "Luguru": f"{BA}.luguru", "Zaramo": f"{BA}.zaramo",
    "Kwere": f"{BA}.kwere", "Kutu": f"{BA}.kutu", "Sagala": f"{BA}.sagala",
    "Zigua": f"{BA}.zigua", "Nguu": f"{BA}.nguu", "Shambala": f"{BA}.shambala",
    "Bondei": f"{BA}.bondei", "Pare": f"{BA}.pare", "Chaga": f"{BA}.chaga",
    "Meru (Rwa)": f"{BA}.rwa", "Sonjo": f"{BA}.sonjo",
    # south-west
    "Hehe": f"{BA}.hehe", "Bena": f"{BA}.bena", "Kinga": f"{BA}.kinga",
    "Magoma": f"{BA}.magoma", "Pangwa": f"{BA}.pangwa", "Sangu": f"{BA}.sangu",
    "Wanji": f"{BA}.wanji", "Nyakyusa": f"{BA}.nyakyusa", "Ndali": f"{BA}.ndali",
    "Fipa": f"{MN}.fipa", "Pimbwe": f"{MN}.pimbwe", "Rungwa": f"{MN}.rungwa",
    "Bungu": f"{MN}.bungu", "Safwa": f"{MN}.safwa", "Malila": f"{MN}.malila",
    "Mambwe": f"{MN}.mambwe", "Nyamwanga": f"{MN}.namwanga", "Nyiha": f"{MN}.nyiha",
    # south
    "Pogoro": f"{BA}.pogoro", "Ndamba": f"{BA}.ndamba", "Matengo": f"{BA}.matengo",
    "Ndengereko": f"{BA}.ndengereko", "Matumbi": f"{BA}.matumbi",
    "Ndendeule": f"{BA}.ndendeule", "Ngindo": f"{BA}.ngindo", "Mwera": f"{BA}.mwera",
    # R7 mother-tongue free text (2026-10-05)
    "Mpoto": f"{BA}.mpoto", "Isanzu": f"{BA}.isanzu", "Kisi": f"{BA}.kisi",
    "Lambya": f"{MN}.lambya",
    "Ngoni": f"{BA}.ngoni_tz", "Yao": f"{BA}.yao", "Makonde": f"{BA}.makonde",
    "Makhuwa": f"{BA}.makhuwa.emakhuwa", "Nyasa": f"{BA}.nyanja_sena.nyasa",
    # Nilotic
    "Maasai": f"{NI}.maasai", "Arusha": f"{NI}.arusha", "Luo": f"{NI}.luo",
    "Datooga": f"{NI}.datooga",
    # Cushitic, isolate
    "Iraqw": f"{SC}.iraqw", "Gorowa": f"{SC}.gorowa", "Alagwa": f"{SC}.alagwa",
    "Sandawe": "isolate.sandawe",
    # others
    "English": "indoeuropean.germanic.english",
    "Arabic": "afroasiatic.arabic",
    "Other African language": "africa_other",
    "Other language": "other",
}

EXTRA_NODES = []


def resolve(name):
    return NAMES.get(name)
