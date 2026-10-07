"""Indonesia, Sensus Penduduk 2010, language used daily at home, persons aged 5+ -> node.

Keyed by the labels sources/id_sp2010.py writes: BPS's eight largest language groups by province
(Table L4.5), L4.2's foreign-language box, and the share-out of the "other regional languages"
remainder (sources/id_shareout.py), one label per language, all `modelled`.

MEASURED (L4.5, L4.2):
  * Jawa, Sunda, Madura, Bugis: the languages, on their own leaves.
  * Indonesia: Indonesian, on `malayic.indonesian`. The question puts Indonesian first: a
    household that talks Indonesian at home is coded Indonesian whatever else it speaks.
  * Melayu: BPS's "Melayu" group, on `malayic.malay`. BPS prints Melayu Perdagangan (the trade
    Malays), Melayu Tengah, Musi/Palembang and Betawi apart from it, inside the remainder.
  * Minangkabau, Banjar: Malayic, on their own leaves (Glottolog mina1268, banj1239).
  * "Bahasa asing" (L4.2's foreign-language box): on `other`. Mostly Chinese varieties by its
    geography, but BPS never says which and L4.1 does not print 312,062 of its 756,035.
  * "Tidak terjawab" (not answered, 561,711): not drawn; countries/id.py's gap.

MODELLED, "Bahasa daerah lainnya: <language>" (sources/id_shareout.py; sources/id.md §5).
Placement in the tree follows Glottolog's classification (data/raw/glottolog/values.csv), with
Indonesia's languages kept flat under `austronesian` as the earlier fragments had them, except:
  * Batak becomes a group: Toba, Mandailing (with Angkola, which Katadata's split does not print
    apart), Karo, Simalungun, Pakpak Dairi, and two that BPS files under Aceh: Alas-Kluet
    (bata1292) and Singkil (Glottolog sing1240, a dialect of Karo; a leaf of its own because
    BPS's code list names it).
  * Malayic: Musi/Palembang (musi1241, "Music"), Col (Lembak, coll1240, Music), Central or South
    Barisan Malay (cent2053; BPS's "Melayu Tengah", with Besemah, Semendo, Ogan, Enim), Kerinci
    (keri1250), Pekal (peka1242, Minangkabauic), Kaur (kaur1269, South Sumatra Malay), Kutai
    (teng1267, Riau-Johoric), Tamiang and Aneuk Jamee (dialects of Malay and Minangkabau that
    BPS codes under Aceh), and the trade Malays as `malayic.trade_malay`, "Eastern Indonesian
    trade Malay" (Glottolog's Eastern Indonesia Trade Malay): BPS's "Melayu Perdagangan", named
    by place as Manado (mala1481), Ambonese (ambo1250), North Moluccan (nort2828), Papuan
    (papu1250) and Kupang (kupa1239) Malay; where the province names no variety the row stays on
    the group, which is BPS's own label.
  * Timoric (tl.txt's grouping of the Austronesian languages of Timor): Uab Meto (uabm1237),
    Rote (Rote-Meto with it), and Tetun, on tl.txt's Tetun Terik, the Tetun of Belu.
  * North Halmahera (Papuan, nort2923): Ternate, Tidore, Tobelo, Galela, Loloda, Tabaru.
  * Indonesian New Guinea (since 2026-10-06): "Papua: <name> [<glottocode>]", 234 languages
    from sources/id_papua.py, mapped by the CSV it writes (`_papua()` below). A language PNG
    also draws is on pg.txt's node; the rest under Glottolog's families as pg_build.py draws
    them (Trans-New Guinea's Dani, Paniai Lakes, Mek and pg's Asmat-Awyu-Ok groups; the other
    Papuan families under `papuan`; isolates on `isolate`; Austronesian under South
    Halmahera-West New Guinea or Oceanic's Sarmi-Jayapura Bay). The 22 clusters this mapping
    had before (Dani, Arfak, Yapen, Bird's Head ...) are gone.
  * Alor-Pantar: the Alor ethnic group, under Timor-Alor-Pantar (most of its languages are; the
    Austronesian Alorese is in it too).
  * Clusters named by an ethnic group that speaks several languages, each one leaf as Dayak
    already was: Sumba (Kambera, Weyewa and five more), Buton (Wolio, Cia-Cia, Tukang Besi and
    others), Kaili (Ledo, Da'a, Unde), Seram, Tanimbar, Aru, Babar, Rote.
  * Bajau: on ph.txt's `sama_bajaw.bajau`. Sign language ("Bahasa Isyarat", 40,373, filed by BPS in
    the regional box): `signlanguage`.
  * "tidak disebut": what the share-out cannot name (groups NC does not split, "Flores",
    the Makian), on `indonesia_other`, which crosses Austronesian and Papuan.
"""
SHARE = "Bahasa daerah lainnya: "
A, M, P = "austronesian", "austronesian.malayic", "papuan"
NAMES = {
    "Jawa": f"{A}.javanese",
    "Indonesia": f"{M}.indonesian",
    "Sunda": f"{A}.sundanese",
    "Melayu": f"{M}.malay",
    "Madura": f"{A}.madurese",
    "Minangkabau": f"{M}.minangkabau",
    "Banjar": f"{M}.banjar",
    "Bugis": f"{A}.buginese",
    "Bahasa asing": "other",
}
_SHARED = {
    "Aceh": f"{A}.acehnese", "Gayo": f"{A}.gayo", "Alas-Kluet": f"{A}.batak.alas_kluet",
    "Singkil": f"{A}.batak.singkil", "Simeulue": f"{A}.simeulue",
    "Aneuk Jamee": f"{M}.aneuk_jamee", "Tamiang": f"{M}.tamiang",
    "Batak Toba": f"{A}.batak.toba", "Batak Mandailing dan Angkola": f"{A}.batak.mandailing",
    "Batak Karo": f"{A}.batak.karo", "Batak Simalungun": f"{A}.batak.simalungun",
    "Batak Pakpak Dairi": f"{A}.batak.pakpak", "Nias": f"{A}.nias",
    "Kerinci": f"{M}.kerinci", "Musi/Palembang": f"{M}.musi",
    "Melayu Tengah (Besemah, Semendo, Ogan, Enim)": f"{M}.central_malay",
    "Lembak (Col)": f"{M}.col", "Komering": f"{A}.komering", "Lampung": f"{A}.lampung",
    "Rejang": f"{A}.rejang", "Mentawai": f"{A}.mentawai", "Pekal": f"{M}.pekal",
    "Kaur": f"{M}.kaur", "Betawi": f"{M}.betawi", "Cirebon": f"{A}.cirebonese",
    "Bali": f"{A}.balinese", "Sasak": f"{A}.sasak", "Bima": f"{A}.bima",
    "Sumbawa": f"{A}.sumbawa", "Uab Meto (Dawan)": f"{A}.timoric.uab_meto",
    "Manggarai": f"{A}.manggarai", "Sumba": f"{A}.sumba", "Lamaholot": f"{A}.lamaholot",
    "Ngada": f"{A}.ngada", "Tetun": f"{A}.timoric.tetun_terik", "Rote": f"{A}.timoric.rote",
    "Alor-Pantar": f"{P}.timor_alor_pantar.alor_pantar", "Lio": f"{A}.lio",
    "Hawu (Sabu)": f"{A}.hawu", "Dayak": f"{A}.dayak", "Kutai": f"{M}.kutai",
    "Paser": f"{A}.paser", "Makassar": f"{A}.makassarese", "Minahasa": f"{A}.minahasan",
    "Gorontalo": f"{A}.gorontalo", "Toraja": f"{A}.toraja", "Mandar": f"{A}.mandar",
    "Tae' (Luwu)": f"{A}.tae", "Duri": f"{A}.duri", "Mamasa": f"{A}.mamasa",
    "Selayar": f"{A}.selayar", "Mamuju": f"{A}.mamuju",
    "Buton (Wolio, Cia-Cia, Tukang Besi)": f"{A}.buton", "Tolaki": f"{A}.tolaki",
    "Muna": f"{A}.muna", "Moronene": f"{A}.moronene", "Banggai": f"{A}.banggai",
    "Saluan": f"{A}.saluan", "Sangir": f"{A}.sangir", "Mongondow": f"{A}.mongondow",
    "Talaud": f"{A}.talaud", "Kaili": f"{A}.kaili", "Pamona": f"{A}.pamona",
    "Buol": f"{A}.buol", "Tomini": f"{A}.tomini", "Lauje": f"{A}.lauje",
    "Bajau": f"{A}.sama_bajaw.bajau", "Kei": f"{A}.kei", "Seram": f"{A}.seram",
    "Tanimbar (Yamdena, Fordata, Selaru)": f"{A}.tanimbar", "Sula": f"{A}.sula",
    "Buru": f"{A}.buru", "Geser-Gorom": f"{A}.geser_gorom", "Aru": f"{A}.aru",
    "Kisar": f"{A}.kisar", "Babar": f"{A}.babar",
    "Ternate": f"{P}.north_halmahera.ternate", "Tidore": f"{P}.north_halmahera.tidore",
    "Tobelo": f"{P}.north_halmahera.tobelo", "Galela": f"{P}.north_halmahera.galela",
    "Loloda": f"{P}.north_halmahera.loloda", "Tabaru": f"{P}.north_halmahera.tabaru",
    "Melayu Perdagangan: Melayu Manado": f"{M}.trade_malay.manado",
    "Melayu Perdagangan: Melayu Ambon": f"{M}.trade_malay.ambonese",
    "Melayu Perdagangan: Melayu Maluku Utara": f"{M}.trade_malay.north_moluccan",
    "Melayu Perdagangan: Melayu Papua": f"{M}.trade_malay.papuan",
    "Melayu Perdagangan: Melayu Kupang": f"{M}.trade_malay.kupang",
    "Melayu Perdagangan: ragam tidak disebut": f"{M}.trade_malay",
    "Bahasa Isyarat": "signlanguage",
    "tidak disebut": "indonesia_other",
}
NAMES.update({SHARE + k: v for k, v in _SHARED.items()})


def _papua():
    """Indonesian New Guinea's languages, "Papua: <Glottolog name> [<glottocode>]" -> node, as
    sources/id_papua.py wrote them (its nodes are pg's for a language PNG also draws, else
    Glottolog's family and Trans-New Guinea group; taxonomy/tree.d/id.txt's generated block)."""
    import csv
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "data" / "normalized" / "id_papua_languages.csv"
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                NAMES[SHARE + r["label"]] = r["node"]


_papua()
EXCLUDED = {"Tidak terjawab"}


def resolve(label):
    if label in EXCLUDED:
        return None
    return NAMES[label]
