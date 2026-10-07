# Regrouping the tree

Anita, 2026-10-06: "we should probably group all the arabics under a family", "are there bantu
subgroups? it fans out quite a bit at once", "austronesian also fans out a lot at once", "we should
make dialects grouped under the main language". Done by session 5d7dac7e-tree.

## How it works

`taxonomy/regroup.txt` lists every move; `taxonomy/regroup.py` (docstring) applies it. Ids stay
WRITTEN as before in fragments, mappings and normalized CSVs (`{BANTU}.swahili`,
`afroasiatic.arabic`); they are DRAWN at their new place (`nigercongo.bantu.zone_g.swahili.swahili`,
`afroasiatic.arabic.arabic`). The move is applied in `taxonomy/build.py` (the tree, and the mapping
check), `countries.py` (every counts() and `parts`), and `tools/audit_groups.py`. languages.json,
the dots, the tiles and the viewer see only drawn ids. No fragment, mapping or CSV was edited.

Colours: the tree is coloured as written and then moved, so no existing node changed colour (diffed:
4,922 nodes, 0 changed). A language that became a group keeps its colour on both the group and its
own leaf. New middle groups take a generated colour from the slot grid around their parent.

**For agents adding languages:** keep writing ids as before. A new Bantu language written
`nigercongo.bantu.x` lands directly under Bantu beside the zones until it is added to its zone in
regroup.txt; do that (one line). A new dialect of a grouped language can be written straight under
the group (`afroasiatic.arabic.juba_arabic`, `indoeuropean.romance.catalan.balearic`).

## What was grouped

**Arabic.** `afroasiatic.arabic` is now a group "Arabic" holding every variety (Gulf, Darija,
Egyptian, Hassaniya, Levantine, Saudi, Sudanese, Yemeni, Tunisian, Algerian, Libyan,
Mesopotamian, Sa'idi, Baharna, Iraqi, Omani, Bedawi, Judeo-Arabic, and the existing Shuwa and
Nubi). Plain census "Arabic" is the leaf `afroasiatic.arabic.arabic`, "Arabic (variety not given)",
in Arabic's old colour, so it no longer draws washed out as "Arabic, not named" (10.85M people in
68 countries). Maltese stays outside. Afroasiatic: 35 children -> 17.

**Bantu: Guthrie zones** (Maho 2009, J for the Great Lakes), the letters the tree's own subgroups
already cite. Bantu: 272 children -> 16 zones (A B C D E F G H J K L M N P R S). Inside the big
ones, Glottolog's subgroups or Guthrie's decades: A into A.10-30, A.40-60, Beti-Fang A.70,
Makaa-Kako A.80-90; C into Rivers Bantu, Mongo and Kasai, Kele-Poke (Sakata direct); G into Ruvu,
Seuta, Southern Highlands, Swahili, Comorian; J into West Nyanza, East Nyanza, Western Lakes, Luyia.
Largest set now: 19 (Zone B, Rivers Bantu). Uncertain placements, by Guthrie's code where known and
by area otherwise: Tere (B, Mai-Ndombe), Kundu (C, as Lonkundo), Zela (L), Benye Nonda and
Ngengele (D), Manyema and Sheng (G), Bodo (C), Mbala, Pende and Kwese in K as Glottolog's Holu
(K.10). `mbuun` "Mbunda (Mbuun)" is B (DRC's Mbuun, B.87).

**Austronesian:** 87 children -> 29. New: Gorontalo-Mongondow, Sangiric and Minahasan,
Bali-Sasak-Sumbawa, Lampungic, Northwest Sumatran (Batak, Nias, Gayo, Simeulue, Mentawai), South
Sulawesi, Celebic (with Tomini and Lauje), Central Malayo-Polynesian (Blust's grouping; Glottolog
keeps its lineages apart) holding Bima-Lembata, Timor-Babar, Tanimbar-Bomberai, Central Maluku
and Aru. Acehnese and Tsat moved into Chamic, Tetun into Timoric, Merina and Kibushi under a
Malagasy group. Oceanic: 50 -> 25, with new Polynesian (15) and Micronesian (11). Left as they
were: Philippine (60), North Borneo (35), SHWNG (30), Timoric (30), the Oceanic subgroups.

**Languages over their dialects** (Glottolog's dialect level; a Glottolog *language* is never moved
under another just for being related). The language becomes a group; its own people go to a leaf
of the same name: Catalan (Valencian), Spanish (Afro-Bolivian Spanish), French (Gallo), German
(Swiss German, Alemannic, Alsatian: Swiss German at Anita's request; Glottolog files these under a
separate Alemannic language), Bulgarian (Pomak), Rusyn (Lemko), Shughni (Bartangi, Bajui), Bundeli
(Pawari), Chhattisgarhi (Surgujia), Maithili (Bajjika), Bhadrawahi (Padari), Nepali (Doteli,
Bajhangi), Mundari (Bhumij), Brao (Kavet), Oy (Cheng), Chut (May), Lowa (Baragaunle), Bouyei
(Giay), Lao (Lao Khrang), Georgian (Ingilo), Nogai (Yurt Tatar), Akan (Twi, Asante), Nyabwa
(Niedeboua), Mahou (Baralaka, Finanga), Koyaga (Nigbi), Wojenaka (Odienneka), Farefare (Talni),
Banda-Banda (Ka, Ndi), Ngemba (Mankon), Meta' (Moghamo), Kambaata (Timbaro, Qebena), Me'en (Bodi),
Bari (Nyepu), Ndyuka (Pamaka), Baure (Joaquiniano), Piraha (Mura), Wai Wai (Tunayana), Cuiba
(Chiricoa), Martu Wangka (Wangkajunga), Gurindji (Malngin), Jaru (Wanyjirra), Gupapuyngu
(Madarrpa), Djinba (Ganalbingu), Burarra (Gun-nartpa), Malagasy (Merina, Kibushi), Pohnpeian
(Sapwuahfik), Cebuano (Boholano), Kalinga Bangad (Sumadel), Minangkabau (Aneuk Jamee), Brunei Malay
(Kedayan; both are Glottolog's Brunei language, not Malay), Sa'a (Ulawa), and in Bantu Swahili
(Bajuni, Zanzibar Swahili, Congo Swahili, Chimwiini), Comorian (the four islands; plain "Comorian"
is "Comorian (island not given)"), Kinyarwanda (Rufumbira), Zinza (Shubi), Nyoro (Tagwenda), Ndali
(Sukwa), Mbati (Bonzio), Nyanja (Chewa, Mang'anja, Nyasa), Nsenga (Kunda), Lamba (Lima), Lenje
(Twa), Ila (Lundwe), Simaa (Imilangu, Mwenyi, Liyuwa, Mulonga), Lunda (Ndembu), Nkoya (Lukolwe,
Lushangi, Mashasha).

Found by matching every node's label against Glottolog's dialects
(`data/raw/glottolog/languages.csv`, Level=dialect) and keeping the pairs drawn as siblings. Name
clashes dropped (Ngombe is not Bushoong, Kisanga is not Mwani, San is not Bambara, Tae' is a
language). Not grouped though asked about: Hazaragi and Dari (Glottolog: separate languages, as are
Dari and Persian); Westphalian (already under Low German; a language in Glottolog).

## Cross-border groups (2026-10-06, session 5d7dac7e-xb)

Anita: "in Africa many languages stop sharply at national borders". `tools/audit_crossborder.py`
finds them; the table of every case, grouped or left, is in `followups.md` (2026-10-06). Here a
Glottolog *language* may join another's group, because neighbouring censuses name one language or
cluster differently (Anita approved grouping "like we do for Arabic"). Same mechanics as above;
the dots were rewritten in place (old drawn id -> new).

Languages over their varieties (`=`): **Shona** (Ndau, Manyika, Tewe), **Kongo** ("Kongo (variety
not given)", the DRC's nine varieties, Fiote, Laari, Suundi; cd's `kongo_dialects` node is left
empty), **Oshiwambo** (Kwanyama, Ndonga), **Nyakyusa** (Nkhonde), **Lomwe** (Malawi Lomwe),
**Sena** (Malawi Sena), **Kalenjin** (Pokot, Sabaot, Kupsabiny), **Somali** (Benaadir, Maay),
**Lugbara** (Aringa), **Uab Meto** (Baikenu), Abron into **Akan**, Isan into **Lao**.
Clusters of separate languages: **Gbe** (`=`, 17 languages, with us.txt's "Gbe (language not
given)" leaf), **Manding** (`=`, 13, with us.txt's "Manding (language not given)"; it must stay
above the Mahou, Koyaka and Wojenaka rules in regroup.txt), and new `+` groups **Rwanda-Rundi**,
**Konzo-Nande**, **Ateker (Teso-Turkana)**, **Tuareg**, **Yoruba and Ede**.
