"""Pakistan Digital Census 2023, Table 11 mother tongue -> node; and the labels of
sources/pk_north.py for Gilgit-Baltistan and Azad Kashmir, which Table 11 leaves out.

Fourteen named languages and OTHERS. OTHERS (3.34M) covers Khowar, Burushaski, Wakhi,
Gujari, Persian and Dari, Hazaragi, Marwari and more, and nothing in the table separates them;
it is drawn on `other`. PBS spells Kohistani "KOHIOSTANI" and Brahui "BRAHVI".

Gilgit-Baltistan and Azad Kashmir (sources/pk_north.py, sources/pk.md) use mixed-case labels, so
they cannot collide with Table 11's capitals.
- GB's census categories (Shina, Balti, Pashto, Kohistani, Urdu) and the three MICS names its
  "Others" is divided into (Burushaski, Khowar, Wakhi). GB's remaining "Other" (Domaaki, Gojri,
  Kashmiri, Punjabi and whatever else MICS filed there, unnamed) goes on `other`.
- Burushaski is an isolate (Glottolog buru1296 is a family of its own). Khowar sits in Dardic as
  conventionally grouped; Glottolog (khow1242) puts it straight under Indo-Aryan.
- AJK Table 15.31's "Pahari" column names a local variety in seven districts (Dhundi-Khairali,
  Chibali, Punchi, Pahari Pothwari, Mirpuri). Glottolog files Chibhali, Punchhi, Pothwari, Mirpur
  Panjabi and Pahari as dialects of Pahari Potwari (paha1251), the same answer as India's Pahari
  in Jammu and Kashmir (in2011.py), so all of them are one node across the Line of Control. The
  table is an estimate in whole percents, not a census answer list; splitting it into five
  dialect nodes would claim more than it measures.
- Kundal Shahi (kund1257) is Shinaic in Glottolog: Dardic. AJK "Others" goes on `other`.
"""
IA = "indoeuropean.indoaryan"

NAMES = {
    "PUNJABI": f"{IA}.northwestern.punjabi",
    "PUSHTO": "indoeuropean.iranian.pashto",
    "SINDHI": f"{IA}.northwestern.sindhi",
    "SARAIKI": f"{IA}.northwestern.saraiki",
    "URDU": f"{IA}.central.urdu",
    "BALOCHI": "indoeuropean.iranian.balochi",
    "HINDKO": f"{IA}.northwestern.hindko",
    "OTHERS": "other",
    "BRAHVI": "dravidian.northern.brahui",
    "MEWATI": f"{IA}.rajasthani.mewati",
    "KOHIOSTANI": f"{IA}.dardic.kohistani",
    "KASHMIRI": f"{IA}.dardic.kashmiri",
    "SHINA": f"{IA}.dardic.shina",
    "BALTI": "sinotibetan.tibetic.balti",
    "KALASHA": f"{IA}.dardic.kalasha",
    # Gilgit-Baltistan (sources/pk_north.py)
    "Shina": f"{IA}.dardic.shina",
    "Balti": "sinotibetan.tibetic.balti",
    "Pashto": "indoeuropean.iranian.pashto",
    "Kohistani": f"{IA}.dardic.kohistani",
    "Urdu": f"{IA}.central.urdu",
    "Burushaski": "isolate.burushaski",
    "Khowar": f"{IA}.dardic.khowar",
    "Wakhi": "indoeuropean.iranian.wakhi",
    "Other": "other",
    # Azad Kashmir (sources/pk_north.py), AJK Statistical Year Book 2025 Table 15.31
    "Kashmiri": f"{IA}.dardic.kashmiri",
    "Gojri": f"{IA}.rajasthani.gujari",
    "Pahari": f"{IA}.northwestern.pahari_pothwari",
    "Pahari (Dhundi-Khairali)": f"{IA}.northwestern.pahari_pothwari",
    "Pahari (Chibali)": f"{IA}.northwestern.pahari_pothwari",
    "Pahari (Punchi)": f"{IA}.northwestern.pahari_pothwari",
    "Pahari (Pahari Pothwari)": f"{IA}.northwestern.pahari_pothwari",
    "Pahari (Mirpuri)": f"{IA}.northwestern.pahari_pothwari",
    "Kundal Shahi": f"{IA}.dardic.kundal_shahi",
    "Dogri": f"{IA}.northwestern.dogri",
    "Punjabi": f"{IA}.northwestern.punjabi",
    "Others": "other",
}


def resolve(name):
    return NAMES[name]
