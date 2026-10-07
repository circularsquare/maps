"""São Tomé and Príncipe, IV RGPH 2012, Quadro 10 "língua falada" (people of 1 and over, several
allowed): census label -> node. Labels exactly as sources/st_rgph.py writes them into
data/normalized/st.csv.

  Português            Portuguese
  Fôrro                Forro, Sãotomense (saot1239), the creole of São Tomé island
  Angolar              Angolar (ango1258), the creole of the south coast, Caué
  Lunguié              Lung'ie, Principense (prin1242), the creole of Príncipe
  Cabo verdiano        Kabuverdianu (kabu1256): the Cape Verdean community of the roças, 31% of
                       Príncipe; a real first language here, so it is shared in, not folded
  Francês, Inglês      French and English, the shared nodes. countries/st.py draws their shares
                       as Portuguese: both are school languages here (AGENT_BRIEF §2, "learned
                       second languages are not home languages")
  Outra(s) língua(s)   "(inclusive sinais)": any other language, sign languages included, unnamed.
                       `other` is the narrowest node holding a sign language and an unnamed
                       spoken one (spec §3.2). Not folded: in Caué it is 5% of children aged 1-4,
                       which is no school language.

`População 1+` is the denominator (the table's Total), not a language: resolve() gives None.
"""

NAMES = {
    "Português": "indoeuropean.romance.portuguese",
    "Fôrro": "creole.portuguese_based.forro",
    "Angolar": "creole.portuguese_based.angolar",
    "Lunguié": "creole.portuguese_based.lungie",
    "Cabo verdiano": "creole.portuguese_based.kabuverdianu",
    "Francês": "indoeuropean.romance.french",
    "Inglês": "indoeuropean.germanic.english",
    "Outra(s) língua(s)": "other",
    "População 1+": None,
}


def resolve(label):
    return NAMES[label]
