"""Republic of the Congo: the labels sources/cg_afro.py writes (Afrobarometer R9 home language,
lingua francas read through the respondent's ethnic group) -> language nodes. Record: sources/cg.md.

Glottolog (data/raw/glottolog): Mbosi mbos1242, Mbere-Mbamba mber1257 (the Mbede / Mbete),
Kota kota1274, Laari laar1238, Teke-Laali teke1277, Kituba (Congo) kitu1245 (Munukutuba), Likuba
liku1242, Akwa akwa1248 (spoken round Makoua), Beembe beem1239 (Congo's, not DR Congo's Bembe),
Suundi suun1239, Koyo koyo1242, Bomitaba bomi1238.
- The ethnic group "Kongo" is drawn on the `kongo` leaf: in Congo it spans Laari, Vili, Yombe,
  Beembe, Suundi, Kunyi and more, and the survey does not say which. Answers that name a variety
  (Lari, Beembe, Suundi) keep their own leaf.
- "Autochtones" (the forest peoples) and the verbatim BaYaka on `aka`, the BaAka's language,
  the largest of them in Likouala and Sangha.
- Teke-Laali, a sibling of `teke` (which other countries draw as a leaf), not a child.
"""
B = "nigercongo.bantu"
NAMES = {
    "Kongo": f"{B}.kongo",
    "Teke": f"{B}.teke",
    "Teke-Laali": f"{B}.teke_laali",
    "Mbosi": f"{B}.mbosi",
    "Mbere (Mbede)": f"{B}.mbere",
    "Laari": f"{B}.laari",
    "Kituba": f"{B}.kituba",
    "Lingala": f"{B}.lingala",
    "Makaa": f"{B}.makaa",
    "Kota": f"{B}.kota",
    "Sira": f"{B}.sira",
    "Aka": f"{B}.aka",
    "Fang": f"{B}.fang",
    "Likuba": f"{B}.likuba",
    "Akwa (Makoua)": f"{B}.akwa",
    "Bomitaba": f"{B}.bomitaba",
    "Koyo": f"{B}.koyo",
    "Beembe": f"{B}.beembe",
    "Suundi": f"{B}.suundi",
    "French": "indoeuropean.romance.french",
}

EXTRA_NODES = []


def resolve(label):
    return NAMES.get(label)
