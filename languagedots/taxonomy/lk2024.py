"""Sri Lanka CPH 2024 ETHNIC GROUP, read as a language, -> node. A proxy: the census asked no
home-language question (Anita's 2026-10-05 ruling for ethnicity-only countries; every row
`derived`).

sources/lk_census.py has already split each GN division's ethnic counts into the languages they
are drawn as, using CPH 2012's ability-to-speak tables by district, so every label here is
"<ethnic group> > <language drawn>" and only the part after ">" decides the node. sources/lk.md
has the reasoning:

  * Sinhalese -> Sinhala; Sri Lanka Tamil, Indian Tamil, Sri Lanka Moor -> Tamil; Burgher ->
    English; except members who in 2012 could not speak that language, who go to the one of
    Sinhala, Tamil or English they could speak.
  * Malay -> Sri Lanka Malay (no source measures retention; APiCS puts speakers at 30,000-40,000
    against about 40,000 ethnic Malays in 2012).
  * Sri Lanka Chetty, Bharatha -> Tamil; Veddahs -> the GN division's majority of Sinhala/Tamil.
  * Other (which also holds every group's counts under 10 in a GN division, the census's
    disclosure rule) -> `other`.
"""
LANGS = {
    "Sinhala": "indoeuropean.indoaryan.sinhala",
    "Tamil": "dravidian.southern.tamil",
    "English": "indoeuropean.germanic.english",
    "Sri Lanka Malay": "austronesian.malayic.sri_lanka_malay",
    "Other": "other",
}

GROUPS = ["Sinhalese", "Sri Lanka Tamil", "Indian Tamil/ Malaiyaga Thamilar",
          "Sri Lanka Moor/Muslim", "Burgher", "Malay", "Sri Lanka Chetty", "Bharatha",
          "Veddahs", "Other"]

# Every label sources/lk_census.py can write, so build.py checks them all.
NAMES = {f"{g} > {lang}": node for g in GROUPS for lang, node in LANGS.items()}

EXCLUDED = set()


def resolve(label):
    return NAMES.get(label)
