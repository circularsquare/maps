"""Bolivia, Censo de Poblacion y Vivienda 2024, PERSONA.IDIOMAT, question 34.1 "Primer idioma o
lengua en el que aprendio a hablar en su ninez" (mother tongue; INE's universe, aged 4+ and
usually resident): census label -> node. Labels exactly as sources/bo_censo.py writes them into
data/normalized/bo.csv, after its repair of the double-encoded ones.

Every label the census prints as a language gets a node (spec 3.1). Families from Glottolog
(data/raw/glottolog/languages.csv), checked per language; see taxonomy/tree.d/bo.txt.

Calls worth knowing:
  Guarani -> tupian.tupiguarani.guarani.chiriguano. Bolivia's Guarani is Eastern Bolivian Guarani
      (Glottolog east2555; Ava, Isoseno and Simba varieties), the node ar.txt made for Ava
      Guarani (Chiriguano). Not the Guarani group: that would draw it as "not named".
  Besiro -> chiquitano.chiquitano: Besiro is the Chiquitano name for their language (br.txt's node).
  Zamuco -> zamucoan.ayoreo: the 2009 constitution's "zamuco" is the living Ayoreo language
      (Glottolog ayor1240); the 18th-century Zamuco of the Jesuit grammars is extinct.
  Machineri -> arawakan.manchineri (br.txt), the same people's language across the Acre border.
  Mojeno Ignaciano, Mojeno Trinitario -> two languages under a Mojeno group (Glottolog moxo1234).
  Joaquiniano -> arawakan.joaquiniano, beside Baure: Glottolog lists it as a dialect of Baure,
      and Baure stays a leaf so that its own speakers are not drawn as "language not named".
  Maropa -> tacanan.reyesano (Glottolog reye1240, Reyesano; Maropa is the people's own name).
  More -> chapacuran.itene (Glottolog iten1243, Itene; More is the people's name).
  Guarasu'we -> tupian.tupiguarani.pauserna (Pauserna, the Guarasu'we people's language).
  Tsimane', Moseten -> two nodes under a Mosetenan root: Glottolog makes Moseten-Chimane one
      isolate with dialects, but the census and most readers name two languages.
  Macha'juyay Kallawaya -> quechuan.kallawaya: Glottolog files Callawalla as a speech register,
      Quechua grammar with a largely Puquina vocabulary; set under Quechuan by its grammar.
  Uru-Chipaya -> uruchipayan.uru_chipaya (the census's one label for the family's languages).
  Afroboliviano -> indoeuropean.romance.afrobolivian, Afro-Bolivian Spanish of the Yungas, a
      variety beside Spanish rather than a creole (it is contested; Romance is the safer home).
  Valenciano -> indoeuropean.romance.valencian, beside Catalan (not under it: Catalan is a leaf).
  Taiwanes -> Min Nan. Chino -> sinotibetan.sinitic, the Chinese group, as pl2021 and ca2021 do:
      "Chinese" names no variety.
  Suizo -> indoeuropean: "Swiss" names no language (German, French, Italian or Romansh), as
      au2021 does with "Swiss, so described".
  Aleman -> German, as the census names it. Most of its 75,852 are Mennonites of the Santa Cruz
      colonies, whose home language is Plautdietsch; the census does not say so, so the dots
      stay on German and the note says it.
  Lenguaje de senas -> signlanguage: the census does not name which sign language.
Remainders:
  "Otras declaraciones", "Otro idioma extranjero" -> other. Neither is filed as indigenous, and
  INE's labels do not say what "other declarations" holds.
Not drawn (gap): "Sin especificar".
"""

IE = "indoeuropean"
RO = "indoeuropean.romance"
GC = "indoeuropean.germanic.continental"
NG = "indoeuropean.germanic.north"
TG = "tupian.tupiguarani"

NAMES = {
    "Castellano": f"{RO}.spanish",
    "Afroboliviano": f"{RO}.afrobolivian",
    # Quechuan, Aymaran, Uru-Chipaya, Puquina
    "Quechua": "quechuan.quechua",
    "Macha´juyay Kallawaya": "quechuan.kallawaya",
    "Aymara": "aymaran.aymara",
    "Uru-Chipaya": "uruchipayan.uru_chipaya",
    "Puquina": "isolate.puquina",
    # Tupi-Guarani
    "Guaraní": f"{TG}.guarani.chiriguano",
    "Tapiete": f"{TG}.guarani.tapiete",
    "Gwarayu": f"{TG}.guarayu",
    "Sirionó": f"{TG}.siriono",
    "Yuqui": f"{TG}.yuqui",
    "Guarasu´we": f"{TG}.pauserna",
    # Arawakan
    "Mojeño Trinitario": "arawakan.mojeno.trinitario",
    "Mojeño Ignaciano": "arawakan.mojeno.ignaciano",
    "Baure": "arawakan.baure",
    "Joaquiniano": "arawakan.joaquiniano",
    "Machineri": "arawakan.manchineri",
    # Tacanan, Panoan
    "Tacana": "tacanan.tacana",
    "Ese Ejja": "tacanan.ese_eja",
    "Kabineña": "tacanan.cavinena",
    "Araona": "tacanan.araona",
    "Maropa": "tacanan.reyesano",
    "Chácobo": "panoan.chacobo",
    "Pacahuara": "panoan.pacahuara",
    "Yaminawa": "panoan.yaminawa",
    # Mosetenan, Chapacuran, Matacoan, Zamucoan, Chiquitano
    "Tsimane´": "mosetenan.tsimane",
    "Mosetén": "mosetenan.moseten",
    "Moré": "chapacuran.itene",
    "Weenhayek": "matacoan.weenhayek",
    "Zamuco": "zamucoan.ayoreo",
    "Bésiro": "chiquitano.chiquitano",
    # isolates
    "Movima": "isolate.movima",
    "Itonama": "isolate.itonama",
    "Cayubaba": "isolate.cayubaba",
    "Canichana": "isolate.canichana",
    "Leco": "isolate.leco",
    "Yurakaré": "isolate.yurakare",
    # foreign
    "Alemán": f"{GC}.german",
    "Holandés": f"{GC}.dutch",
    "Suizo": IE,
    "Inglés": "indoeuropean.germanic.english",
    "Danés": f"{NG}.danish",
    "Noruego": f"{NG}.norwegian",
    "Sueco": f"{NG}.swedish",
    "Portugués": f"{RO}.portuguese",
    "Francés": f"{RO}.french",
    "Italiano": f"{RO}.italian",
    "Catalán": f"{RO}.catalan",
    "Valenciano": f"{RO}.valencian",
    "Gallego": f"{RO}.galician",
    "Rumano": f"{RO}.romanian",
    "Latin": f"{RO}.latin",
    "Ruso": "indoeuropean.slavic.east.russian",
    "Ucraniano": "indoeuropean.slavic.east.ukrainian",
    "Polaco": "indoeuropean.slavic.west.polish",
    "Checo": "indoeuropean.slavic.west.czech",
    "Croata": "indoeuropean.slavic.south.croatian",
    "Serbio": "indoeuropean.slavic.south.serbian",
    "Búlgaro": "indoeuropean.slavic.south.bulgarian",
    "Griego": "indoeuropean.hellenic.greek",
    "Albanés": "indoeuropean.albanian.albanian",
    "Persa": "indoeuropean.iranian.persian",
    "Vasco": "isolate.basque",
    "Húngaro": "uralic.hungarian",
    "Finlandés": "uralic.finnish",
    "Turco": "turkic.turkish",
    "Árabe": "afroasiatic.arabic",
    "Hebreo": "afroasiatic.hebrew",
    "Chino": "sinotibetan.sinitic",
    "Taiwanés": "sinotibetan.sinitic.min_nan",
    "Japonés": "japonic.japanese",
    "Coreano": "koreanic.korean",
    "Tailandés": "kradai.thai",
    "Vietnamés": "austroasiatic.vietnamese",
    # sign language, remainders
    "Lenguaje de señas": "signlanguage",
    "Otras declaraciones": "other",
    "Otro idioma extranjero": "other",
    # not drawn
    "Sin especificar": None,
}


def resolve(label):
    return NAMES[label]
