"""Finland, population register, 31 Dec 2025, language (mother tongue as registered) -> node.
sources/fi_register.py.

Keys are data/normalized/fi.csv's labels: Statistics Finland's code, a space, its English text
("sv Swedish"). The codes are ISO 639-1 plus `98` (other language) and `X` (unknown); CODES
below is keyed on the code, so a reworded English text does not break the join.

CALLS (sources/fi.md says more):
  se "Sami" (2,076): the leaf `uralic.saami`. One register code covers all three Sami languages
    spoken in Finland, North, Inari and Skolt Sami, though ISO's `se` is North Sami alone; the
    register cannot tell them apart and neither can the map. The tree's Sami is one leaf
    already (pl.txt), so nothing is guessed.
  zh "Chinese" (20,320): `sinotibetan.sinitic`, which build.py draws unwashed as "Chinese", as
    for the US, Canada and Kyrgyzstan. The register does not say Mandarin or Cantonese.
  tw "Twi" (942) and ak "Akan" (705): the register prints both. Twi is the largest variety of
    Akan; each label gets its own node, so Twi is a new sibling `nigercongo.kwa.twi` beside
    us.txt's Akan, not a child (a child would turn Akan into a group, drawn washed out).
  ee "Ewe" (170): a new sibling `nigercongo.kwa.ewe` beside us.txt's "Gbe (Ewe, Fon)" leaf,
    for the same reason.
  mo "Moldavian" (365): cz.txt's `indoeuropean.romance.moldovan`, apart from Romanian (7,310)
    as the register prints it.
  sh "Serbo-Croatian" (1,145) beside bs, hr and sr: four labels, four nodes, as printed.
  ny "Chichewa, Chewa, Nyanja" (204): pl.txt's Chewa leaf, Statistics Finland's first name.
  gn "Guarani" (2): Paraguayan Guarani, the language the bare name means; ar.txt's `guarani`
    is a group and a named label may not sit on one.
  rm "Romansh" (2): ch.txt's `rhaetoromance`, which is Switzerland's Romansh.
  bh "Bihari languages" (2) and cr "Cree" (5) name groups, not languages, and sit on the group.
  io "Ido" (25): a new leaf `other.ido` beside `other.esperanto`, a constructed language.
  98 "Other language" (13,258): `other`. The register's list is ISO 639-1's, so a language with
    no two-letter code (Karelian has none) can only be here or under a listed language; the
    table does not split it.
  X "Unknown" (2,209): not drawn, the gap.
"""
IE = "indoeuropean"
IA = f"{IE}.indoaryan"
GE = f"{IE}.germanic"
RO = f"{IE}.romance"
SL = f"{IE}.slavic"
NC = "nigercongo"
BT = f"{NC}.bantu"
AA = "afroasiatic"
AN = "austronesian"
ST = "sinotibetan"

CODES = {
    # national languages
    "fi": "uralic.finnish",
    "sv": f"{GE}.north.swedish",
    "se": "uralic.saami",
    # Uralic, Europe
    "et": "uralic.estonian",
    "hu": "uralic.hungarian",
    "kv": "uralic.komi",
    # Indo-European, Europe
    "ru": f"{SL}.east.russian",
    "uk": f"{SL}.east.ukrainian",
    "be": f"{SL}.east.belarusian",
    "pl": f"{SL}.west.polish",
    "cs": f"{SL}.west.czech",
    "sk": f"{SL}.west.slovak",
    "bg": f"{SL}.south.bulgarian",
    "mk": f"{SL}.south.macedonian",
    "sl": f"{SL}.south.slovenian",
    "bs": f"{SL}.south.bosnian",
    "hr": f"{SL}.south.croatian",
    "sr": f"{SL}.south.serbian",
    "sh": f"{SL}.south.serbocroatian",
    "lv": f"{IE}.baltic.latvian",
    "lt": f"{IE}.baltic.lithuanian",
    "en": f"{GE}.english",
    "de": f"{GE}.continental.german",
    "nl": f"{GE}.continental.dutch",
    "af": f"{GE}.continental.afrikaans",
    "lb": f"{GE}.continental.luxembourgish",
    "li": f"{GE}.continental.limburgish",
    "yi": f"{GE}.continental.yiddish",
    "fy": f"{GE}.frisian",
    "da": f"{GE}.north.danish",
    "no": f"{GE}.north.norwegian",
    "is": f"{GE}.north.icelandic",
    "fo": f"{GE}.north.faroese",
    "ga": f"{IE}.celtic.irish",
    "gd": f"{IE}.celtic.scottishgaelic",
    "cy": f"{IE}.celtic.welsh",
    "fr": f"{RO}.french",
    "es": f"{RO}.spanish",
    "pt": f"{RO}.portuguese",
    "it": f"{RO}.italian",
    "ro": f"{RO}.romanian",
    "mo": f"{RO}.moldovan",
    "ca": f"{RO}.catalan",
    "gl": f"{RO}.galician",
    "an": f"{RO}.aragonese",
    "rm": f"{RO}.rhaetoromance",
    "el": f"{IE}.hellenic.greek",
    "sq": f"{IE}.albanian.albanian",
    "hy": f"{IE}.armenian.armenian",
    # Iranian
    "fa": f"{IE}.iranian.persian",
    "ku": f"{IE}.iranian.kurdish",
    "ps": f"{IE}.iranian.pashto",
    "tg": f"{IE}.iranian.tajik",
    "os": f"{IE}.iranian.ossetian",
    # Indo-Aryan
    "hi": f"{IA}.central.hindi",
    "ur": f"{IA}.central.urdu",
    "bn": f"{IA}.eastern.bengali",
    "as": f"{IA}.eastern.assamese",
    "or": f"{IA}.eastern.odia",
    "ne": f"{IA}.pahari.eastern.nepali",
    "pa": f"{IA}.northwestern.punjabi",
    "sd": f"{IA}.northwestern.sindhi",
    "gu": f"{IA}.gujarati.gujarati",
    "mr": f"{IA}.southern.marathi",
    "si": f"{IA}.sinhala",
    "dv": f"{IA}.dhivehi",
    "ks": f"{IA}.dardic.kashmiri",
    "bh": f"{IA}.bihari",                       # "Bihari languages": a group
    # Dravidian
    "ta": "dravidian.southern.tamil",
    "ml": "dravidian.southern.malayalam",
    "kn": "dravidian.southern.kannada",
    "te": "dravidian.southcentral.telugu",
    # Turkic, Mongolic, Caucasus
    "tr": "turkic.turkish",
    "az": "turkic.azerbaijani",
    "uz": "turkic.uzbek",
    "kk": "turkic.kazakh",
    "ky": "turkic.kyrgyz",
    "tk": "turkic.turkmen",
    "tt": "turkic.tatar",
    "ba": "turkic.bashkir",
    "cv": "turkic.chuvash",
    "ug": "turkic.uyghur",
    "mn": "mongolic.mongolian",
    "ka": "kartvelian.georgian",
    "ce": "nakhdaghestanian.nakh.chechen",
    "av": "nakhdaghestanian.avarandic.avar",
    "ab": "abkhazadyghe.abkhaz",
    "eu": "isolate.basque",
    # Afroasiatic
    "ar": f"{AA}.arabic",
    "he": f"{AA}.hebrew",
    "mt": f"{AA}.maltese",
    "am": f"{AA}.ethiosemitic.amharic",
    "ti": f"{AA}.ethiosemitic.tigrinya",
    "so": f"{AA}.cushitic.lowland.somali",
    "om": f"{AA}.cushitic.lowland.oromo",
    "aa": f"{AA}.cushitic.lowland.afar",
    "ha": f"{AA}.chadic.hausa",
    # Niger-Congo and other African
    "yo": f"{NC}.voltaniger.yoruba",
    "ig": f"{NC}.voltaniger.igbo",
    "ak": f"{NC}.kwa.akan",
    "tw": f"{NC}.kwa.twi",
    "ee": f"{NC}.kwa.ewe",
    "wo": f"{NC}.atlantic.wolof",
    "ff": f"{NC}.atlantic.fulah",
    "bm": f"{NC}.mande.bambara",
    "sg": f"{NC}.sango",
    "sw": f"{BT}.swahili",
    "rw": f"{BT}.kinyarwanda",
    "rn": f"{BT}.kirundi",
    "lg": f"{BT}.ganda",
    "ki": f"{BT}.gikuyu",
    "ln": f"{BT}.lingala",
    "kg": f"{BT}.kongo",
    "lu": f"{BT}.luba_katanga",
    "sn": f"{BT}.shona",
    "ny": f"{BT}.nyanja_sena.chewa",
    "ng": f"{BT}.ndonga",
    "kj": f"{BT}.kwanyama",
    "hz": f"{BT}.herero",
    "zu": f"{BT}.nguni.zulu",
    "xh": f"{BT}.nguni.xhosa",
    "ss": f"{BT}.nguni.siswati",
    "nd": f"{BT}.nguni.ndebele_zw",             # "Northern Ndebele": Zimbabwe's
    "nr": f"{BT}.nguni.ndebele_za",             # "Southern ndebele": South Africa's
    "tn": f"{BT}.sotho_tswana.setswana",
    "st": f"{BT}.sotho_tswana.sesotho",
    "ve": f"{BT}.venda",
    "ts": f"{BT}.tswa_ronga.tsonga",
    "kr": "nilosaharan.kanuri",
    # East and Southeast Asia, Pacific
    "zh": f"{ST}.sinitic",                      # "Chinese": build.py draws it unwashed
    "my": f"{ST}.burmish.burmese",
    "bo": f"{ST}.tibetic.tibetan",
    "dz": f"{ST}.tibetic.dzongkha",
    "ja": "japonic.japanese",
    "ko": "koreanic.korean",
    "vi": "austroasiatic.vietnamese",
    "km": "austroasiatic.khmer",
    "th": "kradai.thai",
    "lo": "kradai.lao",
    "za": "kradai.zhuang",
    "tl": f"{AN}.philippine.tagalog",
    "id": f"{AN}.malayic.indonesian",
    "ms": f"{AN}.malayic.malay",
    "jv": f"{AN}.javanese",
    "su": f"{AN}.sundanese",
    "mg": f"{AN}.malagasy",
    "ch": f"{AN}.chamorro",
    "fj": f"{AN}.oceanic.fijian",
    "sm": f"{AN}.oceanic.samoan",
    "to": f"{AN}.oceanic.tongan",
    "ty": f"{AN}.oceanic.tahitian",
    "na": f"{AN}.oceanic.nauruan",
    "bi": "creole.english_based.bislama",
    # Americas, Arctic
    "kl": "eskimoaleut.greenlandic",
    "cr": "algic.creeinnu.cree",                # "Cree": a group of Cree languages
    "qu": "quechuan.quechua",
    "ay": "aymaran.aymara",
    "gn": "tupian.tupiguarani.guarani.paraguayan",
    # constructed, and the remainder
    "eo": "other.esperanto",
    "io": "other.ido",
    "98": "other",
}
NOT_STATED = {"X Unknown"}


def resolve(label):
    if label in NOT_STATED:
        return None
    code = label.split(" ", 1)[0]
    if code not in CODES:
        raise KeyError(f"fi2025: unmapped label {label!r}")
    return CODES[code]
