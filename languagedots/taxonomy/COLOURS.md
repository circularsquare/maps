# Colour changes

One line per change: node, old → new (OKLCH `L C h`, hex), why. Newest pass first. Distances are
OKLab; the important ones were judged by eye on the map too.

## 2026-10-09, DR Congo's Bantu in subgroup bands, Swahili lemon-lime, Konkomba blue (session 32a047f0-col)

Anita, after DR Congo's rebuild from MICS 2017 (sources/cd.md §0; Swahili about 20% of the country):
"the drc is kinda like colors all over the place so we should maybe slightly narrow them into
tighter subfamily based color groups to make a little bit of room for swahili. currently at a
glance its quite hard to see swahili." Congo Swahili (generated #ffa86d) sat 0.048 from Luba-Kasai
(generated #ffa08d) over 135 shared cells, and DR Congo's Bantu languages ran round the whole wheel.
Also asked: Kabiyè and Konkomba apart in Togo (0.033; Konkomba is 12.5% of Kara).

**How.** Each Guthrie zone or Glottolog subgroup the tree has (taxonomy/GROUPING.md) got a narrow
hue band, and its members differ mostly by lightness and chroma: Kongo green, Rivers Bantu cyan
round Lingala, Mongo orange, Kele-Poke coral, Luba red and rose, Zone K tan and ochre, Zone B violet,
Zone D blue, Western Lakes magenta and orchid. That leaves hue 108-128 (lemon-lime) to Swahili
alone among the Bantu languages of the region. Lightness and chroma inside each band were picked by
a small search (scratch scripts, not kept) over every pair that meets on the ground (dots in 0.5°
cells, both at 5% of a cell, same or neighbouring cell, over cd ao zm tz ug rw bi cf cg ss ke mw mz),
weighted by how many people meet, plus every pair drawn in one country at 0.05% or more, starting
from a hand plan and kept near it. Distances below are OKLab on the drawn hex, with the old distance
to the same neighbour in brackets; "~" marks an old colour read back from its hex (generated ones
had no fragment colour). Rendered on a trial archive of the current dots (cd, tg rescattered
2026-10-09) at http://localhost:8800, not yet judged by Anita.

**Swahili now.** Congo Swahili is at least 0.156 from every language it meets inside DR Congo
(nearest Pende and Ndembu, pale sands; Luba-Kasai, the old clash, 0.358). Swahili (Tanzania, Kenya,
Burundi) is the same lemon-lime a shade brighter, 0.023 from Congo Swahili, and at least 0.11 from
what it meets on the coast and inland (Nyamwezi 0.109, Somali 0.113, Pare 0.118, Rangi 0.121,
Kamba 0.137, Emakhuwa 0.140); it was orange, 0.054-0.062 from Gikuyu and Luhya. The yellows and
yellow-greens that sat near that hue in DR Congo moved out of it (Kete, Sanga,
Yansi, Lega, Bembe, Tabwa, Ngbaka Minagende, Manyanga, Kituba).

**Wider moves, so a language keeps one colour on both sides of a border:** Kinyarwanda (Rwanda)
olive to orchid, Kirundi (Burundi) salmon to rose, Teke and Mbosi (Congo), Chokwe and Lunda
(Angola, Zambia), Kaonde (Zambia) blue to light coral with the Luba, Kongo's Fiote (Cabinda) and
Suundi (Congo) into the Kongo greens, Aka (Congo, CAR) into Rivers cyan. Not moved: Lingala,
Tetela, Shi, Kongo, Yombe, Bemba (anchors); Kimbundu (Zone H but Angola-only, pink beside Kongo's
green and Umbundu's red); Luvale, Ngangela, Mbunda (Zone K, Angola and Zambia only, left green);
the `chokwe_lunda` group colour (zm.txt, green); Konzo, Ha, Hangaza, Rufumbira; Northern Ngbandi,
Mbandja and the other Ubangian, Central Sudanic and Nilotic languages except Logo.

**Group colours (regroup.txt `+` lines),** so the legend's group rows match their members (they
were generated slot colours, Zone L an olive above red members, Rivers Bantu an orange above
cyan ones): `zone_l` #b39f18 -> 0.72 0.14 15 #ef7d88; `zone_c.mongo` #ff9caf -> 0.76 0.15 55
#f99549; `zone_c.rivers` #f7945a -> 0.76 0.11 210 #49c4d8; `zone_c.kele_poke` #ce6489 -> 0.74 0.10
32 #e39383; `zone_d` #b46200 -> 0.72 0.11 258 #79a6e9; `zone_b` #9a7400 -> 0.74 0.11 298 #b39be7;
`zone_k` #c17a00 -> 0.74 0.10 78 #cea35f; `zone_h` #e67d58 -> 0.72 0.13 155 #57bc80;
`zone_j.western_lakes` #ffa9bc -> 0.72 0.15 335 #dd7eca. Its two generated subgroups follow it:
`rwanda_rundi` #ffbdf2 -> #efabff, `konzo_nande` #f68675 -> #c95463. Zones E, F and S, which these
moved, are pinned at their old slots in regroup.txt (0.64 0.14 40, 0.58 0.14 40, 0.70 0.14 20).
Zone C (Rivers and Mongo together) and the zones outside DR Congo keep generated colours.

**Generated colours.** Fourteen nodes got their first fragment colour: Congo Swahili, Luba-Kasai, Mongo,
Bembe, Tabwa (added to cd.txt, under a colour-pass comment), Kinyarwanda (rw.txt), Kirundi (bi.txt),
Ndembu (zm.txt), Aka and Ngbaka Minagende (cf.txt), Suundi (cg.txt), Besi Ngombe, Mboma and Kongo of
the south-east bank (cd.txt). Freeing and taking slots under `nigercongo.bantu` moved 53 generated
Bantu siblings; each is pinned at its exact old slot, so its hex is unchanged: Duala, Ewondo, Fang
(cm.txt); Basaa, Kele, Lumbu, Ndowe, Sangu, Vili, Vumbu (ga.txt); Bobangi, Bodo, Bomitaba, Kako,
Kari, Mbati, Bonzio, Pomo (cf.txt); Akwa, Beembe, Koyo, Likuba (cg.txt); Kwasio (gq.txt);
Ganda, Nyankole, Nyoro, Runyakitara (ug.txt); Lushangi, Mashasha (zm.txt); Ndonga (na.txt);
Shimaore (km.txt); Mushungulu, Chimwiini (so.txt); Ikoma, Isanzu, Kabwa, Kimbu, Kisi, Konongo,
Kutu, Kwaya, Magoma, Manyema, Matumbi, Mbugwe, Mpoto, Ndengereko, Pangwa, Sagala, Segeju, Sonjo,
Tongwe, Wanji (tz.txt). Diffed after: every Bantu node whose drawn colour changed is listed below.
Nine Austronesian group colours (`austronesian.cmp` and its four branches, Lampungic, Northwest
Sumatran, South Sulawesi, Yarikman) changed in the same build, from another session's edits made
meanwhile (bare `austronesian.yapese` and `nigercongo.bantu.kota_gabon` lines in ar, br, co, cr,
pa.txt, nodes the 2026-10-08 merge had folded away), not from this pass.

**Leftovers on the ground, under 0.08:** Kituba | Ndibu 0.058 and Fiote | Ndibu 0.072 (one Kongo
cluster), Tabwa | Bisa 0.062 and Tongwe | Tabwa 0.066 (Tabwa beside other oranges), Bushoong |
Songe 0.068, Kirundi | Nyambo 0.069 (Kagera), Shirazi | Kamba 0.077, Kongo | Mbata 0.077,
Lega | Songola 0.080. In one country's legend, not meeting: Teke | Mono 0.031, Luba-Katanga |
Ntomba 0.031 (cd), Lunda | Kabende 0.028 (zm).

**Swahili (G): lemon-lime, hue 108-128, the one Bantu colour there.**
- `zone_g.swahili.swahili_congo` Congo Swahili: ~0.81 0.13 54 #ffa86d -> 0.9 0.2 120 #cef034. Nearest on the ground: Nyamwezi 0.087 (was 0.140); Lala (Zambia) 0.101 (was 0.111).
- `zone_g.swahili.swahili` Swahili: ~0.70 0.15 55 #e48233 -> 0.92 0.21 122 #cef934. Nearest on the ground: Somali 0.113 (was 0.205); Pare (Asu) 0.118 (was 0.229).
- `zone_g.swahili.shirazi` Shirazi (Zanzibar Swahili): ~0.80 0.11 65 #efaf6f -> 0.8 0.16 128 #a0d157. Nearest on the ground: Kamba (Kikamba) 0.077 (was 0.083); Swahili 0.130 (was 0.110).
- `zone_g.swahili.bajuni` Bajuni: ~0.66 0.08 95 #a19258 -> 0.74 0.14 108 #b4b136. Nearest on the ground: Somali 0.104 (was 0.192); Gikuyu 0.115 (was 0.133).

**Western Lakes (J), with Rwanda-Rundi and Konzo-Nande: magenta, pink and orchid, hue 312-6.**
- `zone_j.western_lakes.rwanda_rundi.kinyarwanda.kinyarwanda` Kinyarwanda: ~0.64 0.13 100 #a08d00 -> 0.76 0.19 315 #de87ff. Nearest on the ground: Nyambo 0.109 (was 0.283); Shi 0.110 (was 0.303).
- `zone_j.western_lakes.rwanda_rundi.kirundi` Kirundi (Rundi): ~0.76 0.14 30 #fd8c7b -> 0.78 0.13 6 #fd93ab. Nearest on the ground: Nyambo 0.069 (was 0.127); Vira 0.112 (was 0.188).
- `zone_j.western_lakes.havu` Havu: ~0.84 0.08 95 #dbcb8e -> 0.66 0.19 6 #ec5480. Nearest on the ground: Fuliiru 0.120 (was 0.094); Shi 0.122 (was 0.313).
- `zone_j.western_lakes.hunde` Hunde: ~0.64 0.08 250 #6690bb -> 0.86 0.08 330 #f0bfea. Nearest on the ground: Nande 0.124 (was 0.268); Nyanga 0.131 (was 0.172).
- `zone_j.western_lakes.konzo_nande.nande` Nande: ~0.70 0.19 36 #ff6a43 -> 0.78 0.16 354 #ff88be. Nearest on the ground: Kinyarwanda 0.112 (was 0.189); Nyankole 0.120 (was 0.053).
- `zone_j.western_lakes.fuliiru` Fuliiru: ~0.84 0.17 83 #ffbd00 -> 0.58 0.1 3 #ab6074. Nearest on the ground: Havu 0.120 (was 0.094); Shi 0.151 (was 0.357).
- `zone_j.western_lakes.vira` Vira: ~0.58 0.08 20 #a56767 -> 0.86 0.13 330 #ffb1fa. Nearest on the ground: Kirundi (Rundi) 0.112 (was 0.188); Kinyarwanda 0.119 (was 0.155).

**Luba (L), with Kaonde: red, rose and coral, hue 0-36.**
- `zone_l.luba_kasai` Luba-Kasai: ~0.80 0.12 32 #ffa08d -> 0.74 0.17 0 #fe77a6. Nearest on the ground: Ha (Kiha) 0.081 (was 0.167); Kaonde 0.108 (was 0.296).
- `zone_l.luba_katanga` Luba-Katanga: ~0.64 0.14 40 #d26b46 -> 0.62 0.17 36 #d85734. Nearest on the ground: Kete 0.105 (was 0.220); Bemba 0.109 (was 0.088).
- `zone_l.songe` Songe (Songye): ~0.74 0.08 279 #a0a6dd -> 0.84 0.09 5 #feb2c3. Nearest on the ground: Bushoong (Kuba) 0.068 (was 0.161); Hemba 0.105 (was 0.105).
- `zone_l.kete` Kete: ~0.80 0.17 97 #ddbe00 -> 0.62 0.17 0 #d35182. Nearest on the ground: Luba-Katanga 0.105 (was 0.220); Luntu 0.116 (was 0.246).
- `zone_l.kanyok` Kanyok: ~0.62 0.20 350 #d84497 -> 0.7 0.07 36 #c68f80. Nearest on the ground: Lunda 0.120 (was 0.237); Other African languages 0.121 (was 0.172).
- `zone_l.luntu` Luntu: ~0.66 0.08 200 #4fa1a5 -> 0.68 0.07 0 #bd8696. Nearest on the ground: Nkutu (Kutshu) 0.106 (was 0.121); Other African languages 0.110 (was 0.160).
- `zone_l.sanga` Sanga (Kisanga): ~0.66 0.14 108 #9c9900 -> 0.58 0.17 0 #c54476. Nearest on the ground: Luba-Katanga 0.112 (was 0.161); Other African languages 0.129 (was 0.158).
- `zone_l.hemba` Hemba: ~0.76 0.08 200 #6fc1c5 -> 0.74 0.07 24 #d49a95. Nearest on the ground: Songe (Songye) 0.105 (was 0.105); Luba-Kasai 0.110 (was 0.200).
- `zone_l.zela` Zela: ~0.84 0.08 200 #89dbdf -> 0.72 0.12 0 #e183a1. Nearest on the ground: Bwile 0.118 (was 0.303); Bemba 0.130 (was 0.271).
- `zone_l.bangubangu` Bangubangu: ~0.84 0.08 40 #f9b9a3 -> 0.86 0.07 30 #fcc1b6. Meets nothing else at 5% of a cell.
- `zone_l.luban.kaonde` Kaonde: ~0.60 0.13 265 #5a7dce -> 0.8 0.12 30 #ffa08f. Nearest on the ground: Bemba 0.106 (was 0.310); Luba-Kasai 0.108 (was 0.296).

**Mongo and Kasai (C.60-80) and Sakata: orange, apricot and amber, hue 40-73.**
- `zone_c.mongo.mongo` Mongo: ~0.70 0.14 50 #e28247 -> 0.84 0.13 70 #ffbb65. Nearest on the ground: Kundu 0.088 (was 0.096); Northern Ngbandi 0.108 (was 0.067).
- `zone_c.mongo.ekonda` Ekonda: ~0.77 0.17 64 #ff9a11 -> 0.74 0.07 73 #c6a47a. Nearest on the ground: Sakata 0.099 (was 0.187); Sengele 0.102 (was 0.128).
- `zone_c.mongo.ntomba` Ntomba: ~0.62 0.20 25 #e64343 -> 0.64 0.19 40 #e65719. Nearest on the ground: Sakata 0.110 (was 0.094); Sengele 0.120 (was 0.272).
- `zone_c.mongo.sengele` Sengele: ~0.86 0.08 50 #fdc2a2 -> 0.76 0.16 40 #ff8a5e. Nearest on the ground: Ekonda 0.102 (was 0.128); Mongo 0.115 (was 0.170).
- `zone_c.sakata` Sakata: ~0.70 0.20 10 #ff5c82 -> 0.66 0.13 73 #c28421. Nearest on the ground: Ekonda 0.099 (was 0.187); Ntomba 0.110 (was 0.094).
- `zone_c.mongo.kusu` Kusu: ~0.77 0.17 64 #ff9800 -> 0.8 0.16 73 #fbab29. Nearest on the ground: Tetela 0.132 (was 0.091); Songe (Songye) 0.158 (was 0.245).
- `zone_c.mongo.bushoong` Bushoong (Kuba): ~0.82 0.08 160 #96d5b2 -> 0.84 0.1 46 #ffb693. Nearest on the ground: Songe (Songye) 0.068 (was 0.161); Pende 0.076 (was 0.309).
- `zone_c.mongo.lele` Lele (DR Congo): ~0.68 0.20 0 #f45693 -> 0.68 0.16 55 #e1791b. Nearest on the ground: Kwese 0.111 (was 0.322); Chokwe 0.112 (was 0.302).
- `zone_c.mongo.dengese` Dengese: ~0.86 0.08 25 #ffbdb7 -> 0.58 0.19 40 #d14300. Nearest on the ground: Lele (DR Congo) 0.116 (was 0.221); Tetela 0.123 (was 0.201).
- `zone_c.mongo.ngando` Ngando: ~0.80 0.08 300 #c5b3ea -> 0.62 0.16 52 #ce6400. Nearest on the ground: Other African languages 0.109 (was 0.246); Nkutu (Kutshu) 0.109 (was 0.210).
- `zone_c.mongo.nkutu` Nkutu (Kutshu): ~0.66 0.08 101 #9d9459 -> 0.7 0.1 73 #c49454. Nearest on the ground: Luntu 0.106 (was 0.121); Ngando 0.109 (was 0.210).
- `zone_c.mongo.kundu` Kundu: ~0.66 0.08 15 #be7e82 -> 0.88 0.07 40 #ffc9b5. Nearest on the ground: Mongo 0.088 (was 0.096); Yansi 0.111 (was 0.204).

**Kele-Poke (C.50), round Kisangani: coral, hue 24-41.**
- `zone_c.kele_poke.lokele` Lokele: ~0.80 0.08 251 #97c2f0 -> 0.74 0.12 26 #ed8c83. Meets nothing else at 5% of a cell.
- `zone_c.kele_poke.poke` Poke (Topoke): ~0.84 0.08 120 #c3d398 -> 0.7 0.12 41 #dd8362. Nearest on the ground: Mbandja 0.108 (was 0.250); Mbesa 0.114 (was 0.181).
- `zone_c.kele_poke.lombo` Lombo (Turumbu): ~0.62 0.08 74 #a37f4d -> 0.86 0.08 38 #ffbfab. Meets nothing else at 5% of a cell.
- `zone_c.kele_poke.mbesa` Mbesa: ~0.70 0.08 210 #5dacba -> 0.8 0.07 29 #e7ada4. Nearest on the ground: Poke (Topoke) 0.114 (was 0.181); Mbandja 0.157 (was 0.183).

**Rivers Bantu (C.10-40), round Lingala (not moved, 0.78 0.12 210 #3ecce2): cyan and azure, hue 190-237.**
- `zone_c.rivers.ngombe` Ngombe: ~0.75 0.19 347 #ff73c3 -> 0.62 0.13 237 #1690ca. Nearest on the ground: Mbosi (Mbochi) 0.104 (was 0.153); Mpama 0.105 (was 0.158).
- `zone_c.rivers.budza` Budza: ~0.66 0.08 229 #5b9cba -> 0.88 0.13 192 #55f2ed. Nearest on the ground: Lingala 0.107 (was 0.130); Ngbaka Minagende 0.179 (was 0.290).
- `zone_c.rivers.bwa` Bwa (Boa): ~0.66 0.14 86 #b98c00 -> 0.66 0.08 200 #4fa1a5. Nearest on the ground: Lingala 0.129 (was 0.255); Makere (Mangbetu) 0.147 (was 0.276).
- `zone_c.rivers.benza` Benza: ~0.78 0.08 360 #e3a2b5 -> 0.7 0.07 216 #69aaba. Meets nothing else at 5% of a cell.
- `zone_c.rivers.libinza` Libinza: ~0.86 0.08 70 #f3c898 -> 0.68 0.11 192 #23adaa. Nearest on the ground: Lingala 0.108 (was 0.204); Ngombe 0.110 (was 0.226).
- `zone_c.rivers.mpama` Mpama: ~0.64 0.08 330 #a87aa3 -> 0.68 0.07 201 #61a6aa. Nearest on the ground: Ngombe 0.105 (was 0.158); Tere 0.111 (was 0.205).
- `zone_c.rivers.nunu` Nunu: ~0.76 0.08 220 #74bdd4 -> 0.88 0.07 228 #a7e1fc. Nearest on the ground: Lingala 0.115 (was 0.049); Kundu 0.137 (was 0.185).
- `zone_c.rivers.mbosi` Mbosi (Mbochi): ~0.66 0.20 25 #f4514f -> 0.56 0.1 195 #008687. Nearest on the ground: Ngombe 0.104 (was 0.153); Other or unclassified 0.112 (was 0.204).
- `zone_c.rivers.aka` Aka: ~0.81 0.13 54 #ffa86d -> 0.88 0.07 192 #a1e7e3. Nearest on the ground: Lingala 0.116 (was 0.244); Ngbaka Minagende 0.170 (was 0.124).

**Zone D (Maniema, Kivu forest, Ituri): blue, hue 242-272.**
- `zone_d.lega` Lega: ~0.74 0.17 115 #a9b700 -> 0.58 0.13 272 #6073c7. Nearest on the ground: Songola 0.080 (was 0.262); Bembe 0.117 (was 0.067).
- `zone_d.lega_mwenga` Rega (Lega-Mwenga): ~0.62 0.20 15 #e4405e -> 0.82 0.11 242 #81cdff. Nearest on the ground: Bembe 0.150 (was 0.246); Vira 0.171 (was 0.126).
- `zone_d.bembe` Bembe: ~0.70 0.14 100 #b39f18 -> 0.68 0.07 272 #8996c4. Nearest on the ground: Songola 0.106 (was 0.220); Nyanga 0.108 (was 0.164).
- `zone_d.nyanga` Nyanga: ~0.78 0.08 174 #7ec9b4 -> 0.78 0.07 242 #90bee1. Nearest on the ground: Bembe 0.108 (was 0.164); Hunde 0.131 (was 0.172).
- `zone_d.komo` Komo (Kumu): ~0.70 0.08 220 #61aac1 -> 0.62 0.11 260 #5e86c8. Nearest on the ground: Other African languages 0.166 (was 0.185); Nyanga 0.169 (was 0.103).
- `zone_d.budu` Budu: ~0.75 0.16 32 #ff836c -> 0.78 0.11 272 #9fb3fe. Nearest on the ground: Mamvu 0.114 (was 0.227); Lingala 0.119 (was 0.278).
- `zone_d.nyali` Nyali: ~0.72 0.08 150 #80b38a -> 0.72 0.11 266 #84a3ea. Nearest on the ground: Ma'di 0.121 (was 0.209); Lingala 0.123 (was 0.122).
- `zone_d.lengola` Lengola: ~0.66 0.20 300 #a76ef8 -> 0.76 0.13 266 #8aaeff. Nearest on the ground: Lingala 0.115 (was 0.262); Other African languages 0.247 (was 0.233).
- `zone_d.songola` Songola: ~0.66 0.08 301 #9a87bc -> 0.58 0.07 242 #5480a0. Nearest on the ground: Lega 0.080 (was 0.262); Bembe 0.106 (was 0.220).
- `zone_d.benye_nonda` Benye Nonda: ~0.62 0.08 130 #76905c -> 0.82 0.07 269 #b1c3f3. Nearest on the ground: Songe (Songye) 0.122 (was 0.196); Bembe 0.140 (was 0.114).
- `zone_d.ngengele` Ngengele: ~0.76 0.08 331 #cf9fc9 -> 0.84 0.08 265 #b1caff. Meets nothing else at 5% of a cell.

**Zone B (Kwilu, Mai-Ndombe, Teke): violet and lilac, hue 284-314.**
- `zone_b.mbuun` Mbunda (Mbuun): ~0.70 0.08 280 #9499d0 -> 0.62 0.13 314 #a26cbc. Nearest on the ground: Ngongo 0.114 (was 0.180); Kete 0.122 (was 0.268).
- `zone_b.yansi` Yansi: ~0.76 0.16 103 #c6b500 -> 0.88 0.07 284 #d1d2ff. Nearest on the ground: Teke 0.101 (was 0.222); Kundu 0.111 (was 0.204).
- `zone_b.ngongo` Ngongo: ~0.60 0.08 60 #a4754e -> 0.72 0.13 290 #a495f0. Nearest on the ground: Mbunda (Mbuun) 0.114 (was 0.180); Teke 0.114 (was 0.255).
- `zone_b.teke` Teke: ~0.80 0.08 230 #87c8e8 -> 0.82 0.13 314 #e3aaff. Nearest on the ground: Yansi 0.101 (was 0.222); Tere 0.112 (was 0.260).
- `zone_b.boma` Boma: ~0.64 0.08 160 #5f9b7b -> 0.62 0.11 290 #857ac4. Nearest on the ground: Tere 0.126 (was 0.237); Mpama 0.144 (was 0.159).
- `zone_b.tere` Tere: ~0.74 0.17 50 #ff8534 -> 0.74 0.07 284 #a5a5d6. Nearest on the ground: Mpama 0.111 (was 0.205); Teke 0.112 (was 0.260).

**Zone K (Chokwe, Lunda, Pende, Mbala, Kwese): tan, ochre and sand, hue 60-92, low chroma.**
- `zone_k.chokwe_lunda.chokwe` Chokwe: ~0.88 0.08 100 #e3d99c -> 0.76 0.12 84 #d5aa4f. Nearest on the ground: Nyaneka 0.084 (was 0.074); Kwese 0.099 (was 0.064).
- `zone_k.chokwe_lunda.lunda.lunda` Lunda: ~0.80 0.10 300 #c7aff5 -> 0.62 0.12 84 #a87f19. Nearest on the ground: Kwese 0.085 (was 0.185); Other African languages 0.097 (was 0.253).
- `zone_k.chokwe_lunda.lunda.ndembu` Ndembu: ~0.82 0.12 184 #56ddcc -> 0.88 0.1 69 #ffcc8e. Nearest on the ground: Kaonde 0.110 (was 0.272); Chokwe 0.121 (was 0.149).
- `zone_k.pende` Pende: ~0.64 0.20 40 #ea5202 -> 0.88 0.06 84 #ead5ab. Nearest on the ground: Bushoong (Kuba) 0.076 (was 0.309); Yansi 0.120 (was 0.228).
- `zone_k.mbala` Mbala: ~0.84 0.08 20 #fbb6b5 -> 0.78 0.12 60 #efa464. Nearest on the ground: Kundu 0.116 (was 0.180); Kwese 0.120 (was 0.143).
- `zone_k.kwese` Kwese: ~0.86 0.08 145 #b1dfb1 -> 0.68 0.06 84 #aa966e. Nearest on the ground: Lunda 0.085 (was 0.185); Chokwe 0.099 (was 0.064).

**Kongo (H), round Kongo (not moved, 0.68 0.15 150 #45b164): green, hue 138-186.**
- `zone_h.kongo.ntandu` Ntandu: ~0.66 0.12 179 #00ab95 -> 0.56 0.08 190 #2f837f. Nearest on the ground: Yombe (Kikongo) 0.102 (was 0.055); Lemfu 0.121 (was 0.136).
- `zone_h.kongo.ndibu` Ndibu: ~0.74 0.16 135 #7dc052 -> 0.84 0.15 138 #98e17f. Nearest on the ground: Kituba (Munukutuba) 0.058 (was 0.147); Fiote 0.072 (was 0.136).
- `zone_h.kongo.manyanga` Manyanga: ~0.82 0.13 120 #b9d06b -> 0.84 0.15 180 #17eace. Nearest on the ground: Besi Ngombe 0.090 (was 0.155); Lingala 0.097 (was 0.181).
- `zone_h.kongo.mbata` Mbata: ~0.86 0.08 140 #b6dead -> 0.74 0.15 168 #00c899. Nearest on the ground: Kongo (variety not given) 0.077 (was 0.195); Fiote 0.087 (was 0.059).
- `zone_h.kongo.lemfu` Lemfu: ~0.58 0.08 120 #74814a -> 0.58 0.15 138 #488e2c. Nearest on the ground: Yombe (Kikongo) 0.078 (was 0.106); Kongo (variety not given) 0.104 (was 0.135).
- `zone_h.kongo.besingombe` Besi Ngombe: ~0.82 0.14 190 #12e0d8 -> 0.88 0.07 177 #a6e7d7. Nearest on the ground: Manyanga 0.090 (was 0.155); Ndibu 0.113 (was 0.161).
- `zone_h.kongo.mboma` Mboma: ~0.58 0.14 120 #708500 -> 0.76 0.07 138 #9cbc91. Nearest on the ground: Mbata 0.098 (was 0.288); Ndibu 0.112 (was 0.166).
- `zone_h.kongo.kongo_se` Kongo of the south-east bank: ~0.82 0.14 140 #92da82 -> 0.66 0.09 168 #55a488. Meets nothing else at 5% of a cell.
- `zone_h.kituba` Kituba (Munukutuba): ~0.86 0.08 120 #cad99e -> 0.78 0.15 138 #85ce6c. Nearest on the ground: Ndibu 0.058 (was 0.147); Mbata 0.088 (was 0.028).
- `zone_h.yaka` Yaka: ~0.72 0.20 340 #f269cb -> 0.88 0.07 138 #c1e3b6. Nearest on the ground: Ndibu 0.089 (was 0.351); Manyanga 0.115 (was 0.326).
- `zone_h.suku` Suku: ~0.80 0.17 72 #ffa800 -> 0.58 0.09 141 #5c8855. Nearest on the ground: Yombe (Kikongo) 0.077 (was 0.278); Lunda 0.111 (was 0.250).
- `zone_h.pelende` Pelende: ~0.62 0.08 340 #a77395 -> 0.7 0.07 138 #89a97f. Nearest on the ground: Kongo (variety not given) 0.085 (was 0.236); Mbata 0.104 (was 0.286).
- `zone_h.kongo.fiote` Fiote: ~0.86 0.12 115 #cedb7c -> 0.78 0.11 138 #94c883. Nearest on the ground: Ndibu 0.072 (was 0.136); Mbata 0.087 (was 0.059).
- `zone_h.kongo.suundi` Suundi: ~0.76 0.14 80 #dfa635 -> 0.76 0.12 145 #7fc581. Meets nothing else at 5% of a cell.

**Bemba (M): Tabwa out of the yellows, beside Bemba's orange.**
- `zone_m.bemba.tabwa` Tabwa: ~0.80 0.14 85 #e7b643 -> 0.72 0.14 76 #d69727. Nearest on the ground: Bisa 0.062 (was 0.141); Tongwe 0.066 (was 0.130).

**Outside Bantu.**
- `nigercongo.gbaya.ngbaka_minagende` Ngbaka Minagende: ~0.86 0.15 100 #e8d34f -> 0.74 0.12 138 #84bd71. Nearest on the ground: Ngbaka Ma'bo 0.113 (was 0.240); Libinza 0.122 (was 0.091).
- `nilosaharan.centralsudanic.logo` Logo: ~0.68 0.14 165 #00b381 -> 0.56 0.12 145 #418646. Nearest on the ground: Lendu 0.087 (was 0.118); Lugbara 0.130 (was 0.019).
**Togo.** `nigercongo.gur.konkomba` (gh.txt): 0.66 0.14 330 #c170bb -> 0.66 0.16 250 #3496ef, an
azure, out of Kabiyè's purple. Kabiyè 0.178 (was 0.033, over 1,355 dots' worth of contact in Kara and
across into Ghana), Ditammari (Benin) over 0.16, Dendi 0.143, Gonja 0.150, Ikposo 0.150, Dagbani
and the other northern Ghana languages further. A periwinkle (0.62 0.15 268) was tried first: it
sat 0.023 from Zarma in Ghana's list; now 0.068. Kabiyè (tg.txt) not moved.

## 2026-10-06, Berber more saturated (session 5d7dac7e-ber)

Anita: "up the saturation of the Berber group because it's mostly a minority group... stand out a
bit against Arabic". Every Berber node now set in build.py HAND/GROUP (the fragments' own colours
in ca, ly, pt, ma, ml, ne changed to match). Chroma 0.12-0.15 → 0.14-0.19; hue kept to 42-125
(yellow, gold, orange, lime), so Siwi, Tamazight and Tunisian Berber leave the greens they had been
generated into, next to the Arabic block. Distances OKLab on the drawn hex, on the ground (dots in
0.5° cells, meeting in a cell or the next) over ma dz tn ly eg ml ne bf mr td; not yet judged on
the map. Diffed: only these 14 nodes changed.
- `berber` (GROUP): 0.78 0.12 100 #c9b957 → 0.80 0.16 95 #debb19 (unnamed, washed: #d6cba2).
  Algerian Arabic 0.100 (was 0.079).
- `tachelhit`: 0.87 0.15 95 #f3d350 → 0.87 0.18 97 #f6d300. Tamazight 0.115, Darija 0.197 (0.176).
- `tamazight`: 0.78 0.12 122 #abc369 (generated) → 0.80 0.19 125 #a2d128, a lime. Darija 0.161
  (0.125), Hassaniya 0.211.
- `tarifit`: 0.66 0.12 68 #c28336 (generated) → 0.64 0.16 42 #d96533, a red-orange. Darija and
  the other two Moroccan Berbers 0.23+.
- `kabyle`: 0.72 0.12 78 #cd9b43 (generated) → 0.74 0.17 62 #f58e02. Tumzabt 0.085, Chaouia 0.180.
- `chaouia`: 0.90 0.12 89 #feda7c (generated) → 0.86 0.18 108 #ddd81f.
- `tumzabt`: 0.66 0.12 89 #b08e2a (generated) → 0.70 0.14 88 #c39810.
- `tamahaq`: 0.84 0.12 89 #eac769 (generated) → 0.90 0.14 96 #fadd68.
- `tamasheq` (ml): 0.76 0.14 70 #e9a03e → 0.77 0.165 72 #f4a000; `tamajaq` (ne): 0.72 0.14 62
  #e28e3a → 0.72 0.165 62 #ec8904. Kept a shade apart so the Tuareg still read as one belt
  (0.057, was 0.046). Tamasheq: Moore 0.086, Senufo 0.094, Winye 0.059 (was 0.043). Tamajaq:
  Fulah 0.135, Hausa further. A lighter or redder Tuareg pair was tried and came within 0.06 of
  Moore (Burkina) or 0.075 of Fulah (Niger).
- `nafusi` (ly): 0.66 0.14 115 #909b1f → 0.72 0.16 115 #a2af10. Libyan Arabic 0.181 (0.134).
- `tunisian_berber`: 0.72 0.12 122 #98b056 (generated) → 0.78 0.17 110 #bfc002. Tunisian Arabic
  0.184 (0.110); Nafusi 0.064 across the border (0.065).
- `siwi`: 0.78 0.12 143 #88cb85 (generated, a green 0.041 from Gulf Arabic) → 0.82 0.17 100
  #dec504. Egypt's Arabics 0.136+ (0.060).
- `zenaga`: 0.90 0.12 68 #ffcf84 (generated) → 0.76 0.15 70 #ed9e2f. Poland only.
- Leftovers, all in France's immigrant palette (hundreds of languages): Kabyle 0.018 from Hindi,
  Tamasheq 0.042 from Kabyle, Tachelhit 0.032 from Bengali; old ones of the same size.

## 2026-10-06, regrouping (session 5d7dac7e-tree)

No colour changed. `taxonomy/regroup.txt` moves nodes after they are coloured, so all 4,922
existing nodes kept their hex (diffed against the as-written tree). A language turned into a group
over its dialects shares its colour with its own new leaf (Arabic's 0.82 0.08 160 is now on both
`afroasiatic.arabic` and `afroasiatic.arabic.arabic`). The ~70 new middle groups (Bantu zones,
Austronesian branches, Polynesian, Micronesian) have generated colours around their parent's.
See `taxonomy/GROUPING.md`.

**Nepali further from Hindi** (Anita, 2026-10-06), build.py HAND, shared by the Nepali group and
its leaf: 0.80 0.16 45 #ff9960 → 0.78 0.17 30 #ff8874, a redder orange. Hindi 0.086 (was 0.062),
Garhwali 0.052 (was 0.033), Bajjika 0.064, Awadhi 0.074, Doteli 0.105; Maithili, Bhojpuri, Tharu,
Tamang, Newar all over 0.1. No other node moved (diffed).

## 2026-10-06, Chinese varieties closer, Southern Thai, saturated South American languages (session 5d7dac7e-col3)

Anita's requests of 2026-10-06 (second round). Distances are OKLab on the drawn hex, checked per
country and on the ground (dots binned to 0.5° cells, each pair that meets in a cell or the
neighbouring ones); not yet judged on the map.

**Chinese varieties closer together.** Jin toward Mandarin; Wu, Teochew, Taishanese and Puxian
out of the yellows and ambers into their branch's reds.
- `sinitic.jin` (cn.txt): 0.56 0.11 75 #9a6a17 → 0.60 0.15 60 #bf6600, a burnt orange, a darker
  Mandarin. Mandarin 0.094 (was 0.155), Mongolian 0.163, Leizhou 0.055 (was 0.060), Daur 0.055
  (small, Hulunbuir).
- `sinitic.wu` (ca.txt, cn.txt): 0.62 0.12 60 #ba7331 → 0.66 0.18 5 #e85983, a raspberry rose with
  Hakka, Xiang and Hainanese. On the ground: Mandarin 0.110 (was 0.091), Hui 0.135, Gan 0.168, Min
  Nan 0.181, Min Dong 0.183, Min Bei 0.078 (was 0.146; they meet in Zhejiang's south-west).
- `sinitic.teochew` (cn, hk, sg): 0.76 0.13 62 #ec9d53 → 0.62 0.11 28 #bf6b60, a lighter step of Min
  Nan's brick (0.120). Hakka 0.072 in eastern Guangdong (was 0.222), Cantonese 0.182, Mandarin
  0.091, Chinese 0.085. Not drawn in Thailand or Cambodia (their censuses say "Chinese").
- `sinitic.siyi` (cn, hk): 0.88 0.07 60 #fbcdaa → 0.70 0.12 18 #df7e82, a deeper coral beside
  Cantonese (0.100 in the Pearl River Delta, was 0.111). Hakka 0.126, Mandarin 0.079. Not drawn in
  the US or Canada lists. Leftover: Min Bei 0.028 (Fujian; they do not meet).
- `sinitic.puxian` (cn): 0.88 0.13 100 #ebd96e → 0.62 0.09 350 #af6f8e, a dusty plum between Min
  Dong (0.120) and Min Nan (0.146). Hakka 0.076, Xiang 0.085, Teochew 0.069 (they do not meet).
- Not moved: Hui, Pinghua and Shao-Jiang (violets, 290-320) still sit apart from the red block.

**Southern Thai nearer Central Thai.**
- `kradai.southern_thai` (th.txt): 0.72 0.15 95 #c2a200 → 0.76 0.17 108 #bcb800, a yellow-olive, a
  stronger Thai. Thai 0.064 (was 0.110), Lao 0.085, Lao Khrang 0.127, Mon 0.115, Malay 0.127,
  Pattani Malay 0.2+, Isan 0.138, Northern Thai 0.142.

**South American indigenous languages more saturated.** Every node with a colour of its own (fragment
or group) under a family whose dots are mostly in South America (for Arawakan, Chibchan and the
isolates, only the nodes whose own dots are mostly there) moved by one rule: C → max(C, min(0.16,
C + 0.05)); L above 0.78 halved toward 0.78 (0.90 → 0.84), since the palest tiers cannot hold more
chroma. Hue kept, so every family stays in its band. Generated members that the new group colours
would have moved were set explicitly at the rule applied to their old colour (so they did not
reshuffle). `unclassified` (deliberately quiet), the Mayan, Uto-Aztecan and Oto-Manguean languages
were left out. Hand picks for the big ones:
- `quechuan.quechua` (br.txt): 0.66 0.17 20 #e75f66 → 0.62 0.20 22 #e6424c. Aymara 0.283 (was 0.24), Kichwa 0.139, Spanish
  0.2+.
- `aymaran.aymara` (br.txt): 0.74 0.12 280 #9ca2f5 → 0.68 0.17 285 #9085fb. Awajún 0.047 in Peru
  (not neighbours).
- `guarani.paraguayan` (ar.txt): 0.78 0.15 45 #ff9661 → 0.74 0.18 45 #ff7f3b. Spanish 0.142,
  Chiriguano 0.167. Leftover: Mandarin 0.055 (small in Paraguay).
- `guarani.nhandeva` (br.txt): 0.74 0.15 45 #f78955 → 0.66 0.15 60 #d37812; the rule left it 0.017
  from the new Paraguayan Guarani. Now 0.087, Kaiowá 0.083.
- `guarani.chiriguano` (ar.txt): 0.70 0.15 36 #ec7859 → 0.58 0.13 65 #ae6700; it had come 0.046
  from Paraguayan Guarani in the Chaco.
- `quechuan.kichwa` (rule): 0.76 0.15 30 #ff8977 → 0.76 0.16 30 #ff8572. Its leftover with Gujarati
  (0.022) is old.
- Ground check over ar, bo, br, cl, co, ec, gy, pe, py, sr, uy, ve: no pair that meets fell under
  0.06 apart from Mandarin | Paraguayan Guarani.
- Pinned at their exact old generated colours, because the Bolivian isolates (Canichana, Cayubaba,
  Itonama, Leco, Movima, now set at the rule) moved their generated siblings: `isolate.` abun, ainu,
  anem, asabano, basque, bogaya, burmeso, duna, elseng, fasu, kaki_ae, kapori, kibiri, kimki,
  kol_papua_new_guinea, ktunaxa, kuot, lenca_salvador, marori, mawes, molof, momuna, muno, odiai,
  pawaia, pele_ata, purari, pyu, sandawe, seri, siamou (each in the fragment that first lists it).
  Diffed after: every drawn change is one of the nodes in this section.

**Gujarati, Kutchi and Rajasthani: wanted, not done** (all three are in build.py HAND). Simulated
with no knock-ons:
- `indoaryan.northwestern.kachchhi` (0.78, 0.12, 15) #fa969f → (0.74, 0.13, 5) #ef86a0. Gujarati
  0.065 (was 0.042), Rajasthani 0.114, Sindhi 0.179, Marathi 0.082 (Mumbai).
- `indoaryan.rajasthani.rajasthani` (0.80, 0.12, 30) #ffa08f → (0.83, 0.12, 42) #ffad89. Gujarati
  0.068 (was 0.038), Hindi 0.091 (same), Mewari 0.061, Marwari 0.165, Harauti 0.037 (was 0.044).
  Also: Rajasthani and Saraiki are both (0.80, 0.12, 30) in HAND, identical, and meet at the
  Rajasthan-Punjab border; this change parts them by 0.035.

Every South American node whose drawn colour changed (old L C h, hex → new):

- **araucanian**: `araucanian` 0.66 0.12 160 #42a878 → 0.66 0.16 160 #00af6f; `huilliche` 0.84 0.1 150 #9bdda8 → 0.81 0.15 150 #72dc8c; `mapuche` 0.74 0.12 160 #5dc291 → 0.74 0.16 160 #23c987; `mapuche_tehuelche` 0.8 0.09 175 #7bd1ba → 0.79 0.14 175 #2fd7b5; `ranquel` 0.64 0.12 170 #1ea381 → 0.64 0.16 170 #00aa7d
- **arawakan**: `arawakan` 0.78 0.15 145 #75d079 → 0.78 0.16 145 #6ed274; `achagua` 0.84 0.12 135 #a8dc8c → 0.81 0.16 135 #92d769; `apurina` 0.8 0.15 175 #09dcb8 → 0.79 0.16 175 #00dbb5; `ashaninka` 0.68 0.15 140 #62ae50 → 0.68 0.16 140 #5daf4a; `baniva` 0.64 0.14 205 #00a3b4 → 0.64 0.16 205 #00a5ba; `baniwa` 0.86 0.15 163 #60efb5 → 0.82 0.16 163 #40e4a6; `cabiyari` 0.64 0.14 146 #4ba255 → 0.64 0.16 146 #3ca54b; `chane` 0.8 0.13 135 #99d079 → 0.79 0.16 135 #8cd162; `enawene_nawe` 0.74 0.15 142 #6fc267 → 0.74 0.16 142 #6ac361; `joaquiniano` generated 0.9 0.15 115 #dbea6d → 0.84 0.16 115 #c8d64c; `kaixana` 0.88 0.15 175 #47f7d2 → 0.83 0.16 175 #00e8c1; `kakinte` 0.88 0.1 140 #b5e8a9 → 0.83 0.15 140 #91df7f; `kinikinau` 0.66 0.15 190 #00aea6 → 0.66 0.16 190 #00afa8; `kuripako` 0.86 0.15 146 #8ceb94 → 0.82 0.16 146 #79df83; `lokono` 0.84 0.12 160 #7fe3b0 → 0.81 0.16 160 #48e09d; `manchineri` 0.58 0.15 145 #33903c → 0.58 0.16 145 #299236; `matapi` 0.8 0.1 172 #74d3b6 → 0.79 0.15 172 #19d9af; `matsigenka` 0.82 0.14 160 #65e0a5 → 0.8 0.16 160 #43dd9a; `mawayana` 0.9 0.15 134 #b4f48b → 0.84 0.16 134 #9ee170; `mehinaku` 0.8 0.15 137 #8ed470 → 0.79 0.16 137 #87d266; `mojeno` 0.72 0.14 145 #67bb6b → 0.72 0.16 145 #5bbe62; `mojeno.ignaciano` 0.64 0.14 147 #49a257 → 0.64 0.16 147 #38a54e; `mojeno.trinitario` 0.76 0.15 150 #61cb7c → 0.76 0.16 150 #58cd78; `nomatsigenga` 0.62 0.14 148 #3f9c53 → 0.62 0.16 148 #2b9f4a; `palikur` 0.9 0.15 145 #9bf89f → 0.84 0.16 145 #82e687; `paresi` 0.62 0.15 159 #00a064 → 0.62 0.16 159 #00a261; `piapoco` 0.7 0.15 150 #4cb86a → 0.7 0.16 150 #43b966; `tariana` 0.68 0.15 153 #3ab26a → 0.68 0.16 153 #2cb467; `terena` 0.74 0.15 145 #68c36d → 0.74 0.16 145 #61c568; `wapixana` 0.62 0.15 145 #409d48 → 0.62 0.16 145 #399e43; `warekena` 0.78 0.15 134 #8ecc64 → 0.78 0.16 134 #8bcd5d; `waura` 0.74 0.15 171 #00c89e → 0.74 0.16 171 #00ca9d; `wayuu` 0.6 0.15 140 #499537 → 0.6 0.16 140 #449630; `yanesha` 0.86 0.12 142 #a3e59c → 0.82 0.16 142 #84de7a; `yawalapiti` 0.58 0.15 190 #00948e → 0.58 0.16 190 #00968f; `yine` 0.74 0.15 170 #00c89c → 0.74 0.16 170 #00ca9b; `yukuna` 0.76 0.15 158 #48cd8c → 0.76 0.16 158 #3acf89
- **arawan**: `arawan` 0.82 0.11 5 #ffa6ba → 0.8 0.16 5 #ff8eb0; `arawa` 0.9 0.11 5 #ffbfd4 → 0.84 0.16 5 #ff9bbc; `banawa` 0.62 0.11 19 #bf696c → 0.62 0.16 19 #d5565f; `deni` 0.86 0.11 23 #ffb4af → 0.82 0.16 23 #ff9793; `jamamadi` 0.62 0.11 347 #b66a93 → 0.62 0.16 347 #c85899; `jarawara` 0.68 0.11 335 #c27eb3 → 0.68 0.16 335 #d36ebf; `kulina_madija` 0.74 0.11 5 #e58da1 → 0.74 0.16 5 #fc7b9d; `paumari` 0.86 0.11 351 #ffb3da → 0.82 0.16 351 #ff96d1; `suruwaha` 0.8 0.11 35 #fca48d → 0.79 0.16 35 #ff9171
- **aymaran**: `aymaran` 0.76 0.12 280 #a2a8fc → 0.76 0.16 280 #9ea3ff; `aymara` 0.74 0.12 280 #9ca2f5 → 0.68 0.17 285 #9085fb; `cauqui` generated 0.82 0.12 280 #b5bbff → 0.8 0.16 280 #abb0ff; `jaqaru` generated 0.7 0.12 269.17 #809ae9 → 0.7 0.16 269.17 #7697ff
- **barbacoan**: `barbacoan` 0.76 0.12 175 #4acaad → 0.76 0.16 175 #00d1ab; `ambalo` generated 0.76 0.12 153.33 #70c78d → 0.76 0.16 153.33 #4dce7f; `awa_pit` 0.7 0.14 160 #37b880 → 0.7 0.16 160 #00bc7b; `chachi` 0.8 0.12 200 #43d5dc → 0.79 0.16 200 #00d8e3; `coconuco` 0.8 0.09 150 #94cf9f → 0.79 0.14 150 #72d48a; `namtrik` 0.86 0.13 145 #99e89b → 0.82 0.16 145 #7cdf81; `polindara` generated 0.76 0.12 218.33 #3fc3e4 → 0.76 0.16 218.33 #00c7f4; `quizgo` generated 0.64 0.12 164.17 #30a379 → 0.64 0.16 164.17 #00a972; `totoro` 0.64 0.12 143 #5d9f5a → 0.64 0.16 143 #47a444; `tsafiki` 0.64 0.12 185 #00a396 → 0.64 0.16 185 #00a999
- **boran**: `boran` 0.84 0.08 240 #9bd2fa → 0.81 0.13 240 #6accff; `bora` generated 0.74 0.08 272.5 #99a8de → 0.74 0.13 272.5 #8fa5fd; `miranha` 0.74 0.08 240 #7bb2d9 → 0.74 0.13 240 #52b5f4; `muinane` generated 0.86 0.08 191.25 #91e2de → 0.82 0.13 191.25 #39ded9
- **cahuapanan**: `cahuapanan` 0.82 0.14 150 #7cdd93 → 0.8 0.16 150 #66da85; `shawi` 0.82 0.15 152 #70e093 → 0.8 0.16 152 #60db89; `shiwilu` 0.68 0.12 140 #6faa62 → 0.68 0.16 140 #5daf4a
- **cariban**: `cariban` 0.86 0.15 325 #ffadff → 0.82 0.16 325 #f89efd; `akawaio` 0.82 0.15 319 #eea3ff → 0.8 0.16 319 #e99aff; `akuriyo` 0.82 0.15 336 #ff9de9 → 0.8 0.16 336 #fd93e5; `apalai` 0.74 0.15 318 #d28ae8 → 0.74 0.16 318 #d488ec; `arara_do_para` 0.6 0.15 318 #a55fb9 → 0.6 0.16 318 #a75cbd; `arekuna` 0.66 0.15 331 #c56dbc → 0.66 0.16 331 #c869bf; `bakairi` 0.9 0.15 325 #ffbaff → 0.84 0.16 325 #fea4ff; `carijona` generated 0.8 0.15 295 #c5a9ff → 0.79 0.16 295 #c3a4ff; `enepa` 0.58 0.15 331 #ab54a3 → 0.58 0.16 331 #ae51a5; `faruk_woto` 0.74 0.15 314 #cd8cec → 0.74 0.16 314 #cf8af1; `galibi_kali_na` 0.7 0.15 322 #c97cd5 → 0.7 0.16 322 #cc79d9; `hixkaryana` 0.68 0.15 318 #be78d4 → 0.68 0.16 318 #c075d7; `ikpeng` 0.88 0.15 332 #ffb1ff → 0.83 0.16 332 #ff9ef5; `ingariko` 0.8 0.15 332 #f598e8 → 0.79 0.16 332 #f591e8; `japreria` 0.66 0.15 332 #c66cbb → 0.66 0.16 332 #c969bd; `kalapalo` 0.74 0.15 332 #e185d5 → 0.74 0.16 332 #e482d7; `kamarakoto` 0.9 0.15 314 #ffbfff → 0.84 0.16 314 #f0a9ff; `katwena` 0.74 0.15 336 #e584cf → 0.74 0.16 336 #e880d1; `katxuyana` 0.68 0.15 327 #c874c8 → 0.68 0.16 327 #cb71cb; `kuikuro` 0.8 0.15 323 #ec9bf5 → 0.79 0.16 323 #eb95f6; `makuxi` 0.74 0.15 325 #da87de → 0.74 0.16 325 #dd84e2; `matipu` 0.66 0.14 315 #b276cd → 0.66 0.16 315 #b670d5; `nahukua` 0.58 0.15 336 #af539c → 0.58 0.16 336 #b24f9e; `patamona` 0.82 0.15 314 #e7a5ff → 0.8 0.16 314 #e39cff; `pemon` 0.72 0.13 3 #e7809d → 0.72 0.16 3 #f4759b; `taurepang` 0.86 0.15 322 #ffafff → 0.82 0.16 322 #f49fff; `tiriyo` 0.62 0.15 328 #b561b3 → 0.62 0.16 328 #b85eb6; `tunayana` 0.7 0.15 334 #d578c5 → 0.7 0.16 334 #d875c7; `wai_wai` 0.86 0.15 330 #ffacff → 0.82 0.16 330 #fd9cf5; `waimiri_atroari` 0.62 0.15 320 #ad65be → 0.62 0.16 320 #af62c1; `wayana` 0.66 0.15 336 #ca6bb5 → 0.66 0.16 336 #cd67b7; `xerewyana` 0.62 0.15 333 #ba60ac → 0.62 0.16 333 #bd5caf; `ye_kwana` 0.58 0.15 325 #a557aa → 0.58 0.16 325 #a853ad; `yukpa` 0.78 0.15 325 #e794ec → 0.78 0.16 325 #ea91ef
- **chapacuran**: `chapacuran` 0.68 0.14 144 #5dae5e → 0.68 0.16 144 #51b153; `itene` generated 0.8 0.14 174 #38dbb7 → 0.79 0.16 174 #00dbb3; `kujubim` 0.58 0.14 144 #3d8f40 → 0.58 0.16 144 #2e9134; `oro_at` 0.62 0.14 137 #5b993e → 0.62 0.16 137 #539b2e; `oro_eo` 0.86 0.14 138 #a2e78b → 0.82 0.16 138 #8edc72; `oro_mon` 0.9 0.14 144 #a3f6a2 → 0.84 0.16 144 #85e585; `oro_nao` 0.74 0.14 144 #70c170 → 0.74 0.16 144 #64c466; `oro_waram` 0.86 0.14 143 #98e993 → 0.82 0.16 143 #81de7c; `oro_waram_xijein` 0.62 0.14 139 #569a42 → 0.62 0.16 139 #4d9c34; `oro_win` 0.68 0.14 132 #77aa48 → 0.68 0.16 132 #72ac36; `pakaa_nova` 0.8 0.14 155 #6ad895 → 0.79 0.16 155 #52d88c; `tora` 0.74 0.14 134 #85be5f → 0.74 0.16 134 #7fc050
- **charruan**: `charruan` 0.74 0.1 225 #5cb8da → 0.74 0.15 225 #00bdf1; `chana` 0.86 0.07 235 #a4d9f9 → 0.82 0.12 235 #6ed0ff; `charrua` 0.74 0.1 225 #5cb8da → 0.74 0.15 225 #00bdf1; `minuan` 0.62 0.08 215 #4592a4 → 0.62 0.13 215 #0098b7
- **chibchan**: `bari` 0.78 0.14 255 #77baff → 0.78 0.16 255 #6cbaff; `ette_taara` 0.86 0.1 270 #b7ceff → 0.82 0.15 270 #9ebeff; `kankuamo` 0.8 0.09 280 #b2b8f7 → 0.79 0.14 280 #a9afff; `kogui` 0.84 0.11 322 #edb3f5 → 0.81 0.16 322 #f19cfe; `muisca` 0.7 0.09 250 #73a3d5 → 0.7 0.14 250 #53a3f2; `uwa` 0.66 0.15 245 #2b99e7 → 0.66 0.16 245 #1899ec
- **chicham**: `chicham` 0.7 0.15 300 #ad87ed → 0.7 0.16 300 #ae84f2; `achuar` 0.62 0.15 322 #af64bb → 0.62 0.16 322 #b261be; `shiwiar` 0.86 0.09 340 #fabae4 → 0.82 0.14 340 #ff9fe1; `shuar` 0.76 0.15 268 #88acff → 0.76 0.16 268 #86abff; `wampis` 0.84 0.1 285 #c3c2ff → 0.81 0.15 285 #b8b2ff
- **chiquitano**: `chiquitano` 0.62 0.1 350 #b36c8f → 0.62 0.15 350 #c65b93; `chiquitano` 0.74 0.12 355 #e68aaf → 0.74 0.16 355 #f77cb0
- **chocoan**: `chocoan` 0.76 0.15 30 #ff8977 → 0.76 0.16 30 #ff8572; `embera` 0.76 0.15 30 #ff8977 → 0.76 0.16 30 #ff8572; `embera.dobida` 0.8 0.11 12 #fca0ab → 0.79 0.16 12 #ff8b9f; `embera.eperara` 0.7 0.14 39 #e77d5a → 0.7 0.16 39 #ef764d; `embera.katio` 0.84 0.14 45 #ffac7b → 0.81 0.16 45 #ff9c64; `wounaan` 0.88 0.1 25 #ffbeb6 → 0.83 0.15 25 #ff9e95
- **chonan**: `chonan` 0.72 0.12 310 #bc8fdd → 0.72 0.16 310 #c385ef; `haush` 0.62 0.1 295 #8a7abc → 0.62 0.15 295 #8e71d6; `selknam` 0.84 0.1 325 #ecb4ef → 0.81 0.15 325 #f19df6; `tehuelche` 0.72 0.12 310 #bc8fdd → 0.72 0.16 310 #c385ef
- **enlhet_enenlhet**: `enlhet_enenlhet` 0.76 0.15 150 #61cb7c → 0.76 0.16 150 #58cd78; `angaite` 0.8 0.09 200 #71d0d5 → 0.79 0.14 200 #00d5de; `enxet_sur` 0.84 0.15 120 #bed95f → 0.81 0.16 120 #b4cf4a; `guana` 0.84 0.07 135 #b7d5a8 → 0.81 0.12 135 #9fd283; `sanapana` 0.54 0.12 145 #3b8040 → 0.54 0.16 145 #178529; `toba_maskoy` 0.66 0.13 115 #909b2f → 0.66 0.16 115 #909c00
- **guahiboan**: `guahiboan` 0.82 0.15 30 #ff9c89 → 0.8 0.16 30 #ff927e; `amorua` 0.62 0.13 34 #c8664f → 0.62 0.16 34 #d55b40; `chiricoa` 0.84 0.12 40 #ffaf8e → 0.81 0.16 40 #ff9a6e; `cuiba` 0.66 0.15 22 #df6768 → 0.66 0.16 22 #e36364; `hitnu` 0.88 0.1 24 #ffbeb7 → 0.83 0.15 24 #ff9e97; `jiw` 0.72 0.14 39 #ed8360 → 0.72 0.16 39 #f67c53; `macaguane` generated 0.92 0.15 20 #ffbabb → 0.85 0.16 20 #ffa0a2; `mapayerri` 0.88 0.1 20 #ffbdbc → 0.83 0.15 20 #ff9d9e; `masiguare` generated 0.8 0.15 0 #ff92ba → 0.79 0.16 0 #ff8bb6; `sikuani` 0.82 0.16 30 #ff9985 → 0.8 0.16 30 #ff927e; `tsiripu` generated 0.86 0.15 0 #ffa5cd → 0.82 0.16 0 #ff95c0; `wipiwi` generated 0.8 0.15 60 #ffa44e → 0.79 0.16 60 #ff9f3f; `yamalero` generated 0.74 0.15 0 #f57fa7 → 0.74 0.16 0 #fa7ba7
- **guaicuruan**: `guaicuruan` 0.66 0.13 260 #6292e1 → 0.66 0.16 260 #5590f3; `abipon` 0.86 0.07 260 #b7d2ff → 0.82 0.12 260 #96c5ff; `kadiweu` 0.74 0.13 260 #7aabfc → 0.74 0.16 260 #6daaff; `mocovi` 0.78 0.13 295 #bda6ff → 0.78 0.16 295 #bfa0ff; `pilaga` 0.82 0.12 245 #7eccff → 0.8 0.16 245 #55c6ff
- **harakmbut**: `harakmbut` 0.74 0.13 222 #25bce5 → 0.74 0.16 222 #00bff2; `harakbut` generated 0.62 0.13 242 #2b8ecd → 0.62 0.16 242 #008edd
- **huarpean**: `huarpean` 0.78 0.11 20 #f69a9a → 0.78 0.16 20 #ff898c; `huarpe` 0.78 0.11 20 #f69a9a → 0.78 0.16 20 #ff898c
- **isolate**: `andaqui` 0.7 0.08 320 #b58ebe → 0.7 0.13 320 #c282d0; `andoque` 0.86 0.1 345 #ffb7e0 → 0.82 0.15 345 #ff9ada; `betoi` 0.8 0.08 340 #e2aace → 0.79 0.13 340 #f298d5; `cayubaba` generated 0.82 0.17 340 #ff95e6 → 0.8 0.17 340 #ff8fe0; `cofan` 0.66 0.15 318 #b872cd → 0.66 0.16 318 #ba6fd1; `gununa_kune` 0.62 0.13 335 #b566a5 → 0.62 0.16 335 #be5bac; `jodi` 0.86 0.12 0 #ffafce → 0.82 0.16 0 #ff95c0; `kamentsa` 0.8 0.14 305 #d3a6ff → 0.79 0.16 305 #d39eff; `kandozi` 0.8 0.15 0 #ff92ba → 0.79 0.16 0 #ff8bb6; `kanoe` 0.8 0.17 322 #f096fe → 0.79 0.17 322 #ec93fb; `kunza` 0.62 0.15 320 #ad65be → 0.62 0.16 320 #af62c1; `lule` 0.66 0.12 320 #b379c0 → 0.66 0.16 320 #bc6ece; `menku` 0.8 0.17 0 #ff8ab9 → 0.79 0.17 0 #ff87b6; `pume` 0.74 0.15 0 #f57fa7 → 0.74 0.16 0 #fa7ba7; `tinigua` 0.62 0.12 300 #9274c3 → 0.62 0.16 300 #966cd7; `trumai` 0.8 0.12 290 #bdb0ff → 0.79 0.16 290 #baa7ff; `urarina` 0.66 0.15 340 #cd6aaf → 0.66 0.16 340 #d066b1; `vilela` 0.86 0.09 345 #fdbadf → 0.82 0.14 345 #ff9ed9; `warao` 0.86 0.17 348 #ffa0e5 → 0.82 0.17 348 #ff93d7; `yagan` 0.84 0.12 320 #edb1fb → 0.81 0.16 320 #ee9dff
- **jabutian**: `jabutian` 0.86 0.09 20 #ffbab9 → 0.82 0.14 20 #ff9d9e; `arikapu` 0.86 0.09 38 #ffbda6 → 0.82 0.14 38 #ffa280; `djeoromitxi_jaboti` 0.74 0.09 20 #de9493 → 0.74 0.14 20 #f68486
- **katukinan**: `katukinan` 0.7 0.09 50 #cc8e6b → 0.7 0.14 50 #e28247; `kanamari` 0.74 0.09 50 #d99a77 → 0.74 0.14 50 #f08e54; `katawixi` 0.62 0.09 41 #b4735b → 0.62 0.14 41 #ca653e; `katukina_do_rio_bia` 0.86 0.09 59 #fec396 → 0.82 0.14 59 #ffac61
- **kawesqar**: `kawesqar` 0.72 0.12 245 #5eaceb → 0.72 0.16 245 #36acff; `kawesqar` 0.72 0.12 245 #5eaceb → 0.72 0.16 245 #36acff
- **macroje**: `bororo.bororo` 0.8 0.17 9 #ff8aa7 → 0.79 0.17 9 #ff87a4; `brobo` 0.86 0.17 353 #ff9fdb → 0.82 0.17 353 #ff92ce; `je.kaingang` 0.86 0.17 6 #ff9ec0 → 0.82 0.17 6 #ff91b3; `je.kayapo` 0.86 0.17 359 #ff9ece → 0.82 0.17 359 #ff91c2; `je.kisedje` 0.8 0.17 0 #ff8ab9 → 0.79 0.17 0 #ff87b6; `je.menkrangnoti` 0.88 0.17 9 #ffa4c0 → 0.83 0.17 9 #ff94b0; `je.tapayuna` 0.9 0.17 2 #ffabd5 → 0.84 0.17 2 #ff97c2; `je.timbira.gaviao_pykopje` 0.8 0.17 9 #ff8aa7 → 0.79 0.17 9 #ff87a4; `je.timbira.kanela` 0.88 0.17 9 #ffa4c0 → 0.83 0.17 9 #ff94b0; `je.timbira.krikati` 0.9 0.17 2 #ffabd5 → 0.84 0.17 2 #ff97c2; `je.xacriaba` 0.9 0.17 12 #ffabc0 → 0.84 0.17 12 #ff97ad; `kamaka` 0.86 0.17 6 #ff9ec0 → 0.82 0.17 6 #ff91b3; `karaja` 0.9 0.17 2 #ffabd5 → 0.84 0.17 2 #ff97c2; `karaja.karaja_javae` 0.86 0.17 359 #ff9ece → 0.82 0.17 359 #ff91c2; `kariri.kipea` 0.86 0.17 6 #ff9ec0 → 0.82 0.17 6 #ff91b3; `kariri.xoco` 0.9 0.17 2 #ffabd5 → 0.84 0.17 2 #ff97c2; `krenak.krenak` 0.8 0.17 9 #ff8aa7 → 0.79 0.17 9 #ff87a4; `maxakali` 0.8 0.17 9 #ff8aa7 → 0.79 0.17 9 #ff87a4; `maxakali.pataxo_ha_ha_hae` 0.9 0.17 352 #ffacea → 0.84 0.17 352 #ff98d6; `ofaye` 0.86 0.17 359 #ff9ece → 0.82 0.17 359 #ff91c2; `rikbaktsa` 0.88 0.17 9 #ffa4c0 → 0.83 0.17 9 #ff94b0; `yate` 0.8 0.17 0 #ff8ab9 → 0.79 0.17 0 #ff87b6
- **matacoan**: `matacoan` 0.76 0.13 195 #00cacb → 0.76 0.16 195 #00cfd0; `chorote` 0.86 0.1 170 #8ae7c7 → 0.82 0.15 170 #38e3b5; `maka` 0.62 0.1 180 #2c9a88 → 0.62 0.15 180 #00a28a; `nivacle` 0.56 0.11 190 #008883 → 0.56 0.16 190 #009089; `weenhayek` 0.88 0.08 205 #96e7f1 → 0.83 0.13 205 #3bdfef; `wichi` 0.74 0.14 195 #00c5c6 → 0.74 0.16 195 #00c8ca
- **mosetenan**: `mosetenan` 0.7 0.15 215 #00b4d8 → 0.7 0.16 215 #00b5dc; `moseten` 0.84 0.1 228 #80d8fe → 0.81 0.15 228 #2bd3ff; `tsimane` 0.7 0.15 215 #00b4d8 → 0.7 0.16 215 #00b5dc
- **muran**: `muran` 0.62 0.1 15 #ba6c73 → 0.62 0.15 15 #d05a69; `mura` 0.86 0.1 24 #ffb7b1 → 0.82 0.15 24 #ff9b94; `piraha` 0.74 0.1 15 #e39096 → 0.74 0.15 15 #fa7f8c; `warapakai` 0.62 0.1 6 #b96c7d → 0.62 0.15 6 #ce5978
- **nadahup**: `nadahup` 0.8 0.12 17 #ff9da2 → 0.79 0.16 17 #ff8c95; `daw` 0.86 0.12 9 #ffafc0 → 0.82 0.16 9 #ff95ae; `hupd_ah` 0.74 0.12 17 #ed8a8f → 0.74 0.16 17 #ff7c86; `judpa` generated 0.92 0.12 344.5 #ffc4f8 → 0.85 0.16 344.5 #ffa1e6; `kakua` generated 0.68 0.12 333.67 #c57bb7 → 0.68 0.16 333.67 #d16ec1; `nadeb` 0.62 0.12 6 #c1657b → 0.62 0.16 6 #d25577; `nukak` generated 0.74 0.12 38.67 #eb8f71 → 0.74 0.16 38.67 #fe835a; `puinave` 0.8 0.12 38 #ffa285 → 0.79 0.16 38 #ff926b; `yuhupdeh` 0.86 0.12 28 #ffb2a5 → 0.82 0.16 28 #ff9889
- **nambikwaran**: `nambikwaran` 0.8 0.09 22 #f2a7a4 → 0.79 0.14 22 #ff9492; `alantesu` 0.8 0.09 37 #f1aa95 → 0.79 0.14 37 #ff9878; `hahaintesu` 0.62 0.09 29 #b67167 → 0.62 0.14 29 #cd6153; `halotesu` 0.74 0.09 9 #dd939f → 0.74 0.14 9 #f48398; `kithaulu` 0.8 0.09 18 #f2a6a8 → 0.79 0.14 18 #ff9398; `latunde` 0.68 0.09 26 #ca827b → 0.68 0.14 26 #e2726a; `mamainde` 0.86 0.09 31 #ffbbad → 0.82 0.14 31 #ffa08c; `manduka` 0.74 0.09 35 #dd9684 → 0.74 0.14 35 #f5886c; `nambikwara` 0.74 0.09 22 #de9491 → 0.74 0.14 22 #f68482; `negarote` 0.62 0.09 13 #b66f77 → 0.62 0.14 13 #cc5e6e; `sabane` 0.68 0.09 7 #c8818f → 0.68 0.14 7 #df7089; `tawande` 0.9 0.09 22 #ffc7c3 → 0.84 0.14 22 #ffa4a1; `waikisu` 0.58 0.09 22 #a96462 → 0.58 0.14 22 #bf5354; `wasusu` 0.86 0.09 15 #ffb9be → 0.82 0.14 15 #ff9ca7
- **panoan**: `panoan` 0.7 0.12 175 #30b69a → 0.7 0.16 175 #00bd99; `amahuaca` generated 0.7 0.12 207.5 #01b3c5 → 0.7 0.16 207.5 #00b8d1; `arara_do_acre` 0.8 0.12 167 #66d6ad → 0.79 0.16 167 #11daa5; `capanahua` generated 0.64 0.12 131.67 #719b4a → 0.64 0.16 131.67 #679f25; `chacobo` generated 0.64 0.12 207.5 #00a0b1 → 0.64 0.16 207.5 #00a5be; `isconahua` generated 0.76 0.12 131.67 #95c16f → 0.76 0.16 131.67 #8bc551; `kakataibo` 0.84 0.1 170 #83e0c1 → 0.81 0.15 170 #33dfb2; `kaxarari` 0.74 0.12 149 #71c080 → 0.74 0.16 149 #55c670; `kaxinawa` 0.74 0.12 175 #42c3a6 → 0.74 0.16 175 #00caa5; `korubo` 0.68 0.12 183 #09b09f → 0.68 0.16 183 #00b7a2; `kulina_pano` 0.6 0.12 145 #4d9351 → 0.6 0.16 145 #31983d; `marubo` 0.86 0.12 193 #5eeae6 → 0.82 0.16 193 #00e3e0; `mastanawa` 0.9 0.12 150 #a3f5b4 → 0.84 0.16 150 #74e791; `matis` 0.68 0.12 145 #66ac69 → 0.68 0.16 145 #4db155; `matses` 0.86 0.12 161 #84eab7 → 0.82 0.16 161 #48e3a2; `mayoruna` 0.62 0.12 189 #009c95 → 0.62 0.16 189 #00a39a; `nahua` generated 0.58 0.12 207.5 #008d9f → 0.58 0.16 207.5 #0092aa; `noke_vana` 0.8 0.12 205 #44d4e2 → 0.79 0.16 205 #00d6eb; `nukini` 0.88 0.12 205 #65eefd → 0.83 0.16 205 #00e4f8; `pacahuara` generated 0.82 0.12 142.5 #96d890 → 0.8 0.16 142.5 #7cd775; `poyanawa` 0.74 0.12 201 #24c1ca → 0.74 0.16 201 #00c7d4; `shanenawa` 0.58 0.12 175 #009176 → 0.58 0.16 175 #009775; `sharanahua` generated 0.82 0.12 185.83 #53ddcf → 0.8 0.16 185.83 #00decd; `shipibo` 0.72 0.13 192 #00beb9 → 0.72 0.16 192 #00c2be; `yaminawa` 0.62 0.12 157 #3a9b68 → 0.62 0.16 157 #00a15d; `yawanawa` 0.9 0.12 175 #7df8d9 → 0.84 0.16 175 #00ecc5
- **pebayaguan**: `pebayaguan` 0.8 0.12 10 #ff9cac → 0.79 0.16 10 #ff8ba3; `yagua` 0.8 0.12 10 #ff9cac → 0.79 0.16 10 #ff8ba3
- **quechuan**: `diaguita_quechua` 0.86 0.09 5 #ffb9c9 → 0.82 0.14 5 #ff9cb8; `inga` 0.8 0.15 14 #ff92a0 → 0.79 0.16 14 #ff8b9b; `kichwa` 0.76 0.15 30 #ff8977 → 0.76 0.16 30 #ff8572; `kolla` 0.78 0.12 355 #f496bb → 0.78 0.16 355 #ff88bc; `kolla_quechua` 0.66 0.13 0 #d16e8f → 0.66 0.16 0 #dd628e; `quechua` 0.66 0.17 20 #e75f66 → 0.62 0.2 22 #e6424c; `quichua` 0.86 0.16 16 #ffa2ad → 0.82 0.16 16 #ff95a0
- **saliban**: `saliban` 0.74 0.12 225 #41bae4 → 0.74 0.16 225 #00bef6; `mako` 0.62 0.13 195 #009d9e → 0.62 0.16 195 #00a2a4; `piaroa` 0.74 0.12 225 #41bae4 → 0.74 0.16 225 #00bef6; `saliba` 0.86 0.09 210 #86e2f2 → 0.82 0.14 210 #13dcf6
- **tacanan**: `tacanan` 0.78 0.12 350 #f197c2 → 0.78 0.16 350 #ff89c6; `araona` generated 0.78 0.12 22.5 #fb9894 → 0.78 0.16 22.5 #ff8a87; `cavinena` generated 0.72 0.12 317.5 #c38cd6 → 0.72 0.16 317.5 #cd82e5; `ese_eja` generated 0.66 0.12 317.5 #b07ac2 → 0.66 0.16 317.5 #b96fd1; `reyesano` generated 0.9 0.12 328.33 #ffc1ff → 0.84 0.16 328.33 #ffa3fe; `tacana` generated 0.84 0.12 350 #ffaad5 → 0.81 0.16 350 #ff93cf
- **tucanoan**: `tucanoan` 0.8 0.11 200 #56d3da → 0.79 0.16 200 #00d8e3; `arapaso` 0.68 0.11 208 #23abbc → 0.68 0.16 208 #00b1cb; `bara` 0.9 0.11 200 #7af4fb → 0.84 0.16 200 #00e9f3; `barasana` 0.58 0.11 200 #008d94 → 0.58 0.16 200 #00949f; `desana` 0.62 0.11 214 #0096ae → 0.62 0.16 214 #009cc0; `dujos` generated 0.68 0.11 247.27 #5d9ed9 → 0.68 0.16 247.27 #309ef5; `jeeruriwa` generated 0.92 0.11 164.55 #9bfcd2 → 0.85 0.16 164.55 #48eeb3; `karapana` 0.74 0.11 226 #51b9e0 → 0.74 0.16 226 #00bdf7; `koreguaje` 0.76 0.12 205 #31c7d5 → 0.76 0.16 205 #00cce1; `kubeo` 0.86 0.11 186 #71e9dc → 0.82 0.16 186 #00e4d4; `letuama` generated 0.92 0.11 247.27 #a8ecff → 0.85 0.16 247.27 #6ed6ff; `maijuna` generated 0.86 0.11 164.55 #87e8be → 0.82 0.16 164.55 #39e4a9; `makaguaje` generated 0.86 0.11 247.27 #95d8ff → 0.82 0.16 247.27 #63ccff; `makuna` 0.68 0.11 170 #42ae8e → 0.68 0.16 170 #00b789; `mirititapuia` 0.58 0.11 245 #3a80b7 → 0.58 0.16 245 #0080d1; `piratapuia` 0.8 0.11 230 #6bcbf7 → 0.79 0.16 230 #00ccff; `pisamira` generated 0.74 0.11 247.27 #6fb1ed → 0.74 0.16 247.27 #47b1ff; `secoya` generated 0.8 0.11 152.73 #84d29c → 0.79 0.16 152.73 #5ad787; `siona` 0.66 0.12 190 #00a9a2 → 0.66 0.16 190 #00afa8; `siriano` 0.74 0.11 174 #52c1a5 → 0.74 0.16 174 #00caa3; `taiwano` generated 0.74 0.11 152.73 #72bf89 → 0.74 0.16 152.73 #47c778; `tanimuka` 0.8 0.11 192 #58d4d0 → 0.79 0.16 192 #00dad5; `tatuyo` generated 0.92 0.11 223.64 #8cf5ff → 0.85 0.16 223.64 #00e3ff; `tukano` 0.74 0.11 200 #3ebfc6 → 0.74 0.16 200 #00c7d2; `tuyuca` 0.62 0.11 182 #079b8b → 0.62 0.16 182 #00a38d; `wanano` 0.86 0.11 218 #73e3ff → 0.82 0.16 218 #00dbff; `yauna` generated 0.68 0.11 223.64 #38a7ca → 0.68 0.16 223.64 #00abe0; `yuruti` generated 0.8 0.11 247.27 #82c4ff → 0.79 0.16 247.27 #59c2ff
- **tupian**: `tupian` 0.8 0.15 38 #ff9974 → 0.79 0.16 38 #ff926b; `arikem` 0.58 0.15 25 #c34f4b → 0.58 0.16 25 #c74b47; `arikem.karitiana` 0.9 0.15 38 #ffb994 → 0.84 0.16 38 #ffa27b; `juruna` 0.86 0.15 43 #ffae7e → 0.82 0.16 43 #ff9e6b; `juruna.xipaya` 0.8 0.15 36 #ff9878 → 0.79 0.16 36 #ff916f; `juruna.yudja` 0.68 0.15 40 #e4744b → 0.68 0.16 40 #e87045; `mawe` 0.62 0.15 42 #ce6234 → 0.62 0.16 42 #d35e2c; `mawe.satere_mawe` 0.8 0.15 46 #ff9c66 → 0.79 0.16 46 #ff965b; `monde` 0.8 0.15 36 #ff9878 → 0.79 0.16 36 #ff916f; `monde.arua` 0.62 0.15 33 #d15e47 → 0.62 0.16 33 #d55a42; `monde.gaviao_de_rondonia` 0.74 0.15 31 #fb836f → 0.74 0.16 31 #ff7f6a; `monde.paiter` 0.9 0.15 38 #ffb994 → 0.84 0.16 38 #ffa27b; `monde.salamay` 0.86 0.15 34 #ffab8e → 0.82 0.16 34 #ff9a7d; `monde.zoro` 0.58 0.15 38 #c25430 → 0.58 0.16 38 #c65029; `munduruku` 0.68 0.15 30 #e6705f → 0.68 0.16 30 #ea6c5a; `munduruku.kuruaya` 0.74 0.15 38 #f98662 → 0.74 0.16 38 #fe825c; `munduruku.munduruku` 0.58 0.15 38 #c25430 → 0.58 0.16 38 #c65029; `purubora` 0.9 0.15 25 #ffb5ab → 0.84 0.16 25 #ff9e95; `purubora.purobora` 0.88 0.15 46 #ffb67f → 0.83 0.16 46 #ffa368; `ramarama` 0.9 0.15 47 #ffbd84 → 0.84 0.16 47 #ffa769; `ramarama.arara_de_rondonia` 0.8 0.15 46 #ff9c66 → 0.79 0.16 46 #ff965b; `tupari` 0.68 0.15 40 #e4744b → 0.68 0.16 40 #e87045; `tupari.ajuru` 0.74 0.15 31 #fb836f → 0.74 0.16 31 #ff7f6a; `tupari.akuntsu` 0.86 0.15 43 #ffae7e → 0.82 0.16 43 #ff9e6b; `tupari.makurap` 0.8 0.15 46 #ff9c66 → 0.79 0.16 46 #ff965b; `tupari.sakurabiat` 0.58 0.15 38 #c25430 → 0.58 0.16 38 #c65029; `tupari.tupari` 0.86 0.15 34 #ffab8e → 0.82 0.16 34 #ff9a7d; `tupiguarani` 0.74 0.15 38 #f98662 → 0.74 0.16 38 #fe825c; `tupiguarani.ache` 0.6 0.14 300 #8e6ac7 → 0.6 0.16 300 #9065d0; `tupiguarani.amanaye` 0.7 0.15 35 #ec785b → 0.7 0.16 35 #f07456; `tupiguarani.anambe` 0.62 0.15 48 #cc6526 → 0.62 0.16 48 #d06217; `tupiguarani.apiaka` 0.78 0.15 41 #ff9469 → 0.78 0.16 41 #ff9062; `tupiguarani.arawete` 0.9 0.15 25 #ffb5ab → 0.84 0.16 25 #ff9e95; `tupiguarani.asurini_do_tocantins` 0.78 0.15 25 #ff8e86 → 0.78 0.16 25 #ff8a82; `tupiguarani.asurini_do_xingu` 0.74 0.15 31 #fb836f → 0.74 0.16 31 #ff7f6a; `tupiguarani.ava_canoeiro` 0.58 0.15 25 #c34f4b → 0.58 0.16 25 #c74b47; `tupiguarani.awa_guaja` 0.8 0.15 36 #ff9878 → 0.79 0.16 36 #ff916f; `tupiguarani.aweti` 0.74 0.15 45 #f78955 → 0.74 0.16 45 #fb864d; `tupiguarani.guajajara` 0.86 0.15 34 #ffab8e → 0.82 0.16 34 #ff9a7d; `tupiguarani.guarani` 0.86 0.15 43 #ffae7e → 0.82 0.16 43 #ff9e6b; `tupiguarani.guarani.ava_guarani` 0.9 0.1 85 #fdd990 → 0.84 0.15 85 #f7c243; `tupiguarani.guarani.chiriguano` 0.7 0.15 36 #ec7859 → 0.58 0.13 65 #ae6700; `tupiguarani.guarani.kaiowa` 0.62 0.15 33 #d15e47 → 0.62 0.16 33 #d55a42; `tupiguarani.guarani.mbya` 0.7 0.15 355 #e573a3 → 0.7 0.16 355 #e96fa3; `tupiguarani.guarani.nhandeva` 0.74 0.15 45 #f78955 → 0.66 0.15 60 #d37812; `tupiguarani.guarani.paraguayan` 0.78 0.15 45 #ff9661 → 0.74 0.18 45 #ff7f3b; `tupiguarani.guarani.tapiete` 0.9 0.1 38 #ffc7ae → 0.84 0.15 38 #ffa681; `tupiguarani.guarani.tupi_guarani` 0.78 0.11 40 #f49f81 → 0.78 0.16 40 #ff9064; `tupiguarani.guarayu` 0.66 0.15 32 #df6a55 → 0.66 0.16 32 #e36650; `tupiguarani.ka_apor` 0.68 0.15 40 #e4744b → 0.68 0.16 40 #e87045; `tupiguarani.kaiabi` 0.84 0.12 35 #ffae95 → 0.81 0.16 35 #ff9778; `tupiguarani.kamayura` 0.58 0.15 46 #bf571b → 0.58 0.16 46 #c35408; `tupiguarani.kambeba` 0.88 0.15 46 #ffb67f → 0.83 0.16 46 #ffa368; `tupiguarani.kawahiva` 0.8 0.15 46 #ff9c66 → 0.79 0.16 46 #ff965b; `tupiguarani.kawahiva.jiahui` 0.62 0.15 42 #ce6234 → 0.62 0.16 42 #d35e2c; `tupiguarani.kawahiva.juma` 0.62 0.15 33 #d15e47 → 0.62 0.16 33 #d55a42; `tupiguarani.kawahiva.kawahiba_dos_amondawa` 0.6 0.15 30 #ca5747 → 0.6 0.16 30 #ce5342; `tupiguarani.kawahiva.kawahiba_dos_karipuna` 0.74 0.15 38 #f98662 → 0.74 0.16 38 #fe825c; `tupiguarani.kawahiva.parintintim` 0.86 0.15 34 #ffab8e → 0.82 0.16 34 #ff9a7d; `tupiguarani.kawahiva.tenharim` 0.74 0.15 45 #f78955 → 0.74 0.16 45 #fb864d; `tupiguarani.kokama` 0.6 0.15 30 #ca5747 → 0.6 0.16 30 #ce5342; `tupiguarani.nheengatu` 0.9 0.15 38 #ffb994 → 0.84 0.16 38 #ffa27b; `tupiguarani.omagua` generated 0.68 0.15 78 #ca8a00 → 0.68 0.16 78 #cd8900; `tupiguarani.parakana` 0.74 0.15 38 #f98662 → 0.74 0.16 38 #fe825c; `tupiguarani.pauserna` generated 0.8 0.15 8 #ff92ab → 0.79 0.16 8 #ff8ba7; `tupiguarani.siriono` generated 0.8 0.15 68 #fca942 → 0.79 0.16 68 #fda42e; `tupiguarani.surui_do_para` 0.74 0.15 51 #f48c49 → 0.74 0.16 51 #f88940; `tupiguarani.tapirape` 0.68 0.15 30 #e6705f → 0.68 0.16 30 #ea6c5a; `tupiguarani.tembe` 0.7 0.15 25 #ed756e → 0.7 0.16 25 #f2716a; `tupiguarani.tenetehara` 0.66 0.15 51 #d8732d → 0.66 0.16 51 #dc7020; `tupiguarani.tupi_potiguara` 0.62 0.15 42 #ce6234 → 0.62 0.16 42 #d35e2c; `tupiguarani.wajapi` 0.62 0.15 33 #d15e47 → 0.62 0.16 33 #d55a42; `tupiguarani.xeta` 0.9 0.15 51 #ffc07e → 0.84 0.16 51 #ffa962; `tupiguarani.yuqui` generated 0.74 0.15 358 #f47faa → 0.74 0.16 358 #f87baa; `tupiguarani.zo_e` 0.58 0.15 38 #c25430 → 0.58 0.16 38 #c65029
- **uruchipayan**: `uruchipayan` 0.8 0.12 165 #69d6aa → 0.79 0.16 165 #23daa1; `uru_chipaya` 0.8 0.12 165 #69d6aa → 0.79 0.16 165 #23daa1
- **witotoan**: `witotoan` 0.7 0.1 250 #6da3da → 0.7 0.15 250 #4ba3f7; `nonuya` generated 0.64 0.1 224 #3999b9 → 0.64 0.15 224 #009dcf; `ocaina` generated 0.76 0.1 211 #59c2d5 → 0.76 0.15 211 #00c9e7; `witoto` 0.74 0.1 250 #79b0e8 → 0.74 0.15 250 #58b0ff
- **yanomaman**: `yanomaman` 0.7 0.15 295 #a689f1 → 0.7 0.16 295 #a787f6; `ninam` 0.62 0.15 277 #717adf → 0.62 0.16 277 #7079e4; `sanuma` 0.86 0.15 313 #f3b3ff → 0.82 0.16 313 #e8a3ff; `xiriana` 0.86 0.15 281 #c0c4ff → 0.82 0.16 281 #b3b6ff; `xirixana` 0.8 0.15 325 #ee9af3 → 0.79 0.16 325 #ed94f3; `yanomami` 0.74 0.15 295 #b296ff → 0.74 0.16 295 #b394ff; `yanoman` 0.62 0.15 309 #a06aca → 0.62 0.16 309 #a267ce
- **zamucoan**: `zamucoan` 0.68 0.15 320 #c077d1 → 0.68 0.16 320 #c374d5; `ayoreo` 0.68 0.15 320 #c077d1 → 0.68 0.16 320 #c374d5; `chamacoco` 0.76 0.13 355 #f18db5 → 0.76 0.16 355 #fe82b6; `chamacoco.tomaraho` 0.62 0.14 345 #c06099 → 0.62 0.16 345 #c6589c; `chamacoco.ybytoso` 0.8 0.13 5 #ff99b2 → 0.79 0.16 5 #ff8bad
- **zaparoan**: `zaparoan` 0.78 0.1 35 #f0a08c → 0.78 0.15 35 #ff9173; `andoa` generated 0.84 0.1 87 #e7c77c → 0.81 0.15 87 #eab936; `arabela` generated 0.9 0.1 22 #ffc4c0 → 0.84 0.15 22 #ffa19e; `zaparo` generated 0.84 0.1 35 #ffb39e → 0.81 0.15 35 #ff9b7d

## 2026-10-06, Isan, Iberian and Sicilian, Polynesian, Mon, Leizhou, Dutch, red Uralic (session 5d7dac7e-col)

Anita's seven requests of 2026-10-06. Distances are OKLab on the drawn hex, checked per country
and on the ground (dots binned to 0.5-0.75° cells, each node against what falls in its own and
the neighbouring cells, cross-border included); not yet judged on the map. "Generated" = no
fragment colour before.

**1. Isan near Central Thai.** Isan had been made orange, far from the other Thai varieties.
- `kradai.isan` (th.txt): 0.76 0.16 60 #f99532 → 0.89 0.13 120 #d0e882, a pale yellow-green: a
  lighter Thai. Thai 0.093 (related, still apart), Northern Thai 0.239, Southern Thai 0.183, Lao
  0.213, Tai Khün/Loei 0.100. Across the Mekong: Phu Thai 0.073, Tai Dam 0.054 (Laos).
- `kradai.lao_khrang` (th.txt): 0.88 0.13 100 #ebd96e → 0.64 0.13 100 #9f8d0e, a dark olive near
  Lao (0.048), of which it is a variety. It would have sat 0.025 from the new Isan.
- Thai (yellow-green), Northern Thai (green), Southern Thai (gold) not moved: the four now share
  the yellow-green band, Isan the light end of it.

**2. Catalan, Valencian, Galician and Sicilian out of the Arabic teal.** All four were generated or
set cyans (hue 195-215) that read as the Arabic block.
- `romance.catalan` (now set in es.txt; generated #7bf3ff) → 0.58 0.19 5 #cf386b, crimson. Spain:
  Spanish 0.293, Valencian 0.123, Arabic 0.35; Roussillon: French 0.085. Barcelona leftovers: Urdu
  0.055, Fula 0.088 (small).
- `romance.valencian` (now set in es.txt; generated #65dfee) → 0.66 0.10 355 #c27895, a dusty rose
  beside Catalan. Moved with Catalan, though not asked: it would have stayed cyan. Romanian (big in
  Valencia and Castellón) 0.122, Spanish 0.194, French 0.086. Lighter roses sat 0.05 from Romanian.
- `romance.galician` (now set in es.txt; generated #87d4ff) → 0.70 0.16 310 #bd7fe8, a lilac.
  Portuguese 0.129 across the Miño, Spanish 0.262, Arabic 0.244, Romanian 0.128.
- `romance.sicilian` (it.txt): 0.80 0.10 200 #64d1d7 → 0.68 0.19 20 #f75c66, a coral red. Italian
  0.187, Neapolitan far, Tunisian Arabic 0.286, Darija 0.327, Romanian 0.161. Catalan 0.112 and
  Galician 0.203 where all three are drawn (Venezuela, Uruguay, Argentina, Chile lists).

**3. Polynesian languages pink, away from Oceanic blue and English.** There is no Polynesian node;
the Polynesian languages sit directly under Oceanic and were generated or set blues and greens,
Māori a yellow-green. They now share hot pinks and magentas (hue 320-20), apart by lightness.
- `oceanic.maori` (au.txt): 0.80 0.14 130 #a0d06b → 0.62 0.20 350 #d84497. English 0.354 (was
  0.195), Samoan 0.146, Tongan 0.261, Punjabi 0.098, Mandarin 0.172 (Auckland).
- `oceanic.samoan` (pt.txt, us.txt): 0.62 0.13 240 #248fcc → 0.74 0.17 325 #df81e5. Tongan 0.141.
  Leftover: Turkish 0.044 in the US and Australia.
- `oceanic.tongan` (now set in nz.txt; generated #77ccff) → 0.86 0.10 335 #fbb8eb. English 0.119.
  A coral was tried first; it sat 0.035 from Cantonese in Auckland and Sydney. Leftover: Khmer 0.041.
- `oceanic.hawaiian` (now set in us.txt; generated #0089b4) → 0.66 0.19 340 #da5ab6. Leftover:
  Marathi 0.042 in the US list.
- `oceanic.cook_islands_maori` (au.txt): 0.72 0.12 150 #69ba7c → 0.78 0.13 350 #f594c3.
- `oceanic.tuvaluan` (au.txt): generated #64a1ee → 0.70 0.17 340 #e16fbf.
- `oceanic.niuean` (au.txt): generated #1ec5e4 → 0.80 0.14 320 #e59ff6.
- `oceanic.tokelauan` (au.txt): generated #1997cf → 0.84 0.10 350 #fdb0d4.
- `oceanic.rapanui` (cl.txt): generated #009bc7 → 0.66 0.18 345 #db5dab.
- `oceanic.tahitian` (fi.txt): generated #0085bc → 0.74 0.15 20 #fb8083.
- Solomons outliers (sb.txt; generated teals and blues): `sikaiana` → 0.72 0.16 340 #e579c4,
  `anuta` → 0.80 0.12 0 #fd9cba, `rennell_bellona` → 0.64 0.18 355 #db5392, `tauma` → 0.84 0.10
  320 #e8b6f3.
- Not changed: French Polynesia draws its census's Polynesian languages on the `austronesian.oceanic`
  group (washed blue), so the change does not reach it. The PNG outliers (Nukumanu, Nukuria, Takuu)
  and Fagauvea (nc) are 1-2 dots and kept their colours.

**4. Mon away from Burmese.** Both were pink-mauve (0.208 apart numerically, alike to the eye).
- `austroasiatic.mon` (now set in mm.txt; generated #ffa2e9) → 0.78 0.14 65 #f4a34b, an orange.
  Burmese 0.248, Rakhine 0.193, Karen 0.184; Southern Thai 0.095 across the border.

**5. Leizhou Min with the Min varieties.**
- `sinitic.leizhou` (cn.txt): 0.62 0.13 305 #9970c4 → 0.56 0.12 45 #ad5b33, a rust between Min
  Nan's brick (0.070) and the other reds. Vietnamese 0.247 (was 0.072); on the ground Hakka 0.094,
  Hainanese 0.222, Cantonese 0.246, Mandarin 0.129.

**6. Dutch and Luxembourgish further from German.**
- `continental.dutch` (cz.txt, pt.txt, us.txt): 0.78 0.12 225 #51c7f1 → 0.80 0.14 225 #35d0ff,
  brighter. German 0.146 (was 0.125), English 0.136, Darija 0.103, Javanese 0.150.
- `continental.luxembourgish` (lu.txt, pt.txt): 0.58 0.10 200 #008c92 → 0.54 0.10 170 #198166, a
  darker green-teal. German 0.179 (was 0.114), Portuguese 0.184, Limburgish 0.075, Bulgarian 0.073
  (was 0.040). Brighter greens scored higher but looked neon.

**7. Uralic red, and Chuvash out of its way.** Uralic was all cyans and teals (hue 130-285). It now
holds the crimson-to-rose band (hue 330-38), split by lightness. The Volga is where it meets Turkic:
Tatar (red-orange, 28) and Bashkir (yellow-orange, 70) stay; Chuvash, which was pink at 345 beside
Mari El, goes violet. Every fragment that carried a colour was changed (cz, fi, pt, ro, ru, se,
sk, th, uk, us).
- `uralic` (group): 0.74 0.12 190 #2ac3bb → 0.64 0.17 15 #e0576a
- `uralic.finnish`: 0.80 0.10 190 #67d2cc → 0.66 0.19 355 #e65598. Swedish 0.249, Russian 0.342,
  Estonian 0.141, Hungarian 0.066 (not neighbours). At 0.64 0.19 25 it was 0.010 from Kyrgyz.
- `uralic.estonian`: 0.66 0.11 220 #29a1c1 → 0.78 0.13 10 #fe93a4. Russian 0.280, Tatar 0.086. At
  0.70 0.16 5 it was Punjabi's colour.
- `uralic.hungarian`: 0.82 0.11 225 #6cd3fa → 0.62 0.20 10 #e3406a. Romanian 0.202 (Transylvania),
  Romani 0.097 (Slovakia, Romania), Slovak 0.32, Serbian 0.37. Leftover: Catalan 0.045, Urdu 0.040
  in diaspora lists.
- `uralic.karelian`: 0.78 0.11 255 → 0.78 0.13 30 #ff9685
- `uralic.meankieli`: 0.66 0.12 215 → 0.74 0.15 38 #f98662
- `uralic.saami`: 0.62 0.12 165 → 0.76 0.13 330 #df92d8; `saami_north` 0.60 0.13 150 → 0.62 0.16
  350 #ca5794; `saami_lule` 0.74 0.12 130 → 0.80 0.12 345 #f59ecf; `saami_south` 0.52 0.10 175 →
  0.56 0.13 335 #a25492
- `uralic.mari`: 0.80 0.11 200 → 0.80 0.20 330 #ff88fa; `meadow_mari` 0.88 0.07 190 → 0.78 0.12 350
  #f197c2; `hill_mari` 0.62 0.12 215 → 0.70 0.12 330 #c882c2
- `uralic.udmurt`: 0.66 0.14 250 → 0.58 0.18 25 #cf4040
- `uralic.mordvin`: 0.76 0.12 175 → 0.88 0.10 330 #fec0f7; `erzya` 0.62 0.12 185 → 0.66 0.15 345
  #d168a7; `moksha` 0.86 0.08 205 → 0.80 0.10 25 #f8a49d
- `uralic.komi`: 0.84 0.09 230 → 0.80 0.12 15 #ff9da5; `komi_permyak` 0.64 0.12 200 → 0.64 0.14 340
  #c367a7
- `uralic.khanty`: 0.68 0.13 175 → 0.64 0.17 5 #dd577e; `mansi` 0.84 0.09 210 → 0.80 0.11 0
  #f89fbb; `nenets` 0.68 0.14 285 → 0.62 0.15 325 #b263b7; `selkup` 0.74 0.12 220 → 0.72 0.13 345
  #de82b7
- Generated members follow the new group: `livonian` #ff9355, `veps` #c0401f, `votic` #ff8a82,
  `izhorian` #e27300, `enets` #d55434, `nganasan` #fc7499.
- `turkic.chuvash` (ru.txt): 0.68 0.17 345 #df67b0 → 0.60 0.16 305 #9763cc, violet.
- Volga, every pair that meets: Tatar | Mari 0.198, | Udmurt 0.142, | Mordvin 0.216, | Komi-Permyak
  0.152, | Komi 0.100, | Moksha 0.106; Chuvash | Mari 0.212, | Hill Mari 0.122, | Erzya 0.122;
  Bashkir | every Uralic 0.106+; Mari | Udmurt 0.280, | Komi-Permyak 0.168. Russian is 0.27+ from
  all. The first try (Mordvin 0.74 0.14 0, Mari 0.60 0.18 355) put Mordvin 0.039 from Uzbek and
  0.032 from Mari; the Uzbek migrant dots everywhere in Russia (pink, 345) are the main squeeze.
- Leftovers, all small on the ground: Chuvash | Uyghur 0.029 (a ring); Udmurt | Romani 0.051; Mari
  | Uzbek 0.088; Mordvin | Kazakh 0.083; Erzya | Uzbek 0.080.

**Generated knock-ons, pinned at their exact old colours.** Moving generated siblings (Mon, the
Polynesian languages, Catalan, Valencian, Galician, Tongan and the others) freed or took slots, and
the generator moved about 150 other nodes, mostly in Papua New Guinea. 61 nodes pinned at their
old generated L C h, in the fragment that first lists each (ar, au, be, cn, co, es, fm, fr, gu,
it, kh, la, mm, pg, pl, pw, vn): Oceanic subgroups (admiralty_islands, bali_vitu, central_pacific,
huon_gulf, ngero, north_papuan_mainland_d_entrecasteaux, peripheral_papuan_tip, schouten,
st_matthias, suauic, vitiaz, willaumez, new_ireland_northwest_solomonic) and Oceanic languages
(carolinian, chuukese, gilbertese, marshallese, mokilese, motu, nauruan, nukuoro_kapingamarangi,
pingelapese, rotuman, sapwuahfik, sonsorol_tobi, yapese); Austroasiatic (bit, blang, chut,
ka_chrook, kanh_chok, khang, kri, mang, may, nguon, palaung, phong_vietic, ro_ong, samtao, thmon,
tho, tum, wa, yinchia); Kra-Dai (giay, mulam, qabiao, saek, sui, tai_lue, tai_nua); Romance
(corsican, ladino, latin, ligurian, occitan, sardinian, walloon); Continental Germanic (alsatian,
wymysorys). Diffed after: only the nodes above changed. Giay (0.92 0.14 125, Laos and Vietnam) is
now 0.033 from the new Isan; not neighbours, and small.

## 2026-10-06, Dari, Tajik and Persian apart (session 5d7dac7e-pid)

Anita: Dari, Tajik, Balochi and Farsi look too similar, Dari and Tajik most of all. All four sat
at hue 65-80, L 0.66-0.78: Persian | Dari 0.048, Dari | Tajik 0.054, Tajik | Balochi 0.050. Persian
(Iran's backdrop amber) and Balochi (build.py HAND) kept; Dari moved to a clear orange, Tajik to a
pale gold, so the four now span orange, amber, gold and brown, still the warm Iranian part of the
wheel. Distances are OKLab on the drawn hex, not yet judged on the map.
- `indoeuropean.iranian.dari` (af.txt, pt.txt): 0.74 0.13 65 #e39849 → 0.72 0.16 48 #f3813f.
  Afghanistan: Balochi 0.107 (was 0.091), Pashto 0.168 (0.117), Uzbek 0.163 (0.180), Turkmen 0.189.
  Persian 0.097 (0.048), Tajik 0.180 (0.054).
- `indoeuropean.iranian.tajik` (cz.txt): 0.70 0.12 80 #c5953b → 0.86 0.12 92 #edcf6f. Tajikistan:
  Uzbek 0.250, Russian 0.206, Kyrgyz 0.291, Kongrat 0.180; Uzbekistan: Kazakh 0.135. Persian 0.088
  (0.082), Balochi 0.205 (0.050).
- `indoeuropean.iranian.hazaragi` (au.txt; Australia only): 0.82 0.11 60 #f9b379 → 0.84 0.11 45
  #ffb48e, kept a lighter step of Dari (0.127), of which it is a variety. Persian 0.076.
- Not moved: Persian (Iran: Mazandarani 0.085, Luri 0.095, Balochi 0.124, Kurdish 0.137), Balochi
  (Pakistan: Sindhi 0.157, Saraiki 0.160, Brahui 0.165, Pashto 0.167; Iran: Kurdish 0.067), Pashto.
- Generated knock-ons pinned at their old colours in by.txt so they did not move: `iranian.oroshori`
  (0.76 0.09 61.11 #dba475) and `iranian.yazghulami` (0.82 0.09 32.22 #f9af9f). Diffed: only Dari,
  Tajik and Hazaragi changed.
- Known leftovers, all where the Iranian language is small: Dari 0.041 from Hindi in the Gulf
  states, 0.046 from Mandarin in Australia and Canada; Tajik 0.055 from Bashkir in Russia's list;
  Balochi | Gilaki 0.054 in Iran (not neighbours).

## 2026-10-05, one Arabic block, warm Iranian, Yiddish, Uyghur, Romani, Wu (colour pass, edd42a8c-col2)

Anita: Yiddish and Levantine Arabic too alike (they meet in Israel); the Arabic varieties, mostly
nationalities, should look more alike, and their neighbours (Kurdish, Farsi, maybe all of Iranian)
further away; Uyghur too close to Mandarin; label Wu "Wu". Also from earlier today: Turkish and
Romani alike in Greek Thrace; Luba-Katanga near Tabwa and Luba-Kasai (sources/cd.md §6).
Distances are OKLab on the drawn hex; not yet judged on the map.

**Arabic, one block.** Every Arabic variety now sits at hue 140-182 and chroma 0.06-0.10 (it ran
112-192 before, with Egyptian and Sa'idi in olive and Levantine and Hassaniya in cyan), told apart by
lightness (0.60-0.86) and a little by hue: Gulf, Nile and Iraqi varieties lean green (140-166), the
Maghreb leans teal (172-182). Picked by a small search: every pair of varieties that meet on the
ground (Levantine | Iraqi, Saudi | Gulf, Egyptian | Sudanese, Darija | Hassaniya ...) at least 0.072
apart, every pair drawn in the same country (the Gulf states draw nearly all of them) at least
0.038, and each at least 0.05 from the non-Arabic languages of its own country. Plain `arabic`
(build.py GROUP) not moved. Big varieties kept to L 0.66-0.84 so none goes near-white or dark.
- `afroasiatic.egyptian_arabic` (eg, sa): 0.86 0.09 128 #c0dd9d → 0.66 0.06 166 #6f9e8a
- `afroasiatic.saidi_arabic` (eg): 0.72 0.11 112 #a6ab56 → 0.72 0.10 146 #7bb67f
- `afroasiatic.bedawi_arabic` (eg): 0.64 0.11 136 #6d9b56 → 0.86 0.06 146 #b9dcba
- `afroasiatic.libyan_arabic` (eg, ly): 0.76 0.09 172 #71c4aa → 0.60 0.10 172 #2f9379
- `afroasiatic.sudanese_arabic` (sd): 0.84 0.07 158 #a5d9b9 → 0.78 0.06 140 #a4c19d
- `afroasiatic.levantine_arabic` (sa): 0.74 0.09 192 #5ebdb9 → 0.74 0.06 169 #86b7a4
- `afroasiatic.mesopotamian_arabic` (sy): 0.68 0.09 150 #6fa87b → 0.86 0.10 162 #93e6bd
- `afroasiatic.iraqi_arabic` (iq): 0.80 0.08 152 #97cda5 → 0.66 0.10 150 #63a471
- `afroasiatic.gulf_arabic` (ae, bh, kw, qa): 0.64 0.08 150 #689a72 → 0.76 0.10 160 #75c59b
- `afroasiatic.saudi_arabic` (sa): 0.77 0.07 142 #9bc097 → 0.84 0.10 140 #a8db9d
- `afroasiatic.omani_arabic` (om): 0.72 0.10 118 #9fad63 → 0.60 0.10 140 #5f8f54
- `afroasiatic.yemeni_arabic` (sa): 0.70 0.11 173 #47b497 → 0.70 0.06 140 #8ba885
- `afroasiatic.baharna_arabic` (bh): 0.82 0.08 186 #86d5cb → 0.86 0.06 150 #b6ddbd
- `afroasiatic.darija` (ma, pt): 0.82 0.08 175 #8bd6c1 → 0.84 0.10 180 #7ae0cd
- `afroasiatic.algerian_arabic` (dz): 0.80 0.08 168 #8acfb4 → 0.78 0.06 181 #8dc4b9
- `afroasiatic.tunisian_arabic` (tn): 0.84 0.08 158 #9edbb7 → 0.70 0.10 180 #4bb3a1
- `afroasiatic.hassaniya` (ml, pt): 0.66 0.10 190 #33a6a0 → 0.66 0.10 182 #3aa697
- `afroasiatic.arabic.shuwa` (now set in td.txt; was generated): 0.68 0.08 144 #7aa579 → 0.60 0.06
  164 #5e8c77. Was 0.059 from Ngas and 0.060 from Bilala; now 0.057 or more from everything in
  Chad, Nigeria, Cameroon and Niger (nearest Kalabari, in Nigeria's south, not a neighbour).
- `afroasiatic.judeo_arabic` (now set in pl.txt; was generated, a yellow): 0.86 0.08 95 #e1d195 →
  0.70 0.07 158 #79ac8e, into the block. Poland only.
- `afroasiatic.arabic.nubi` (ug, 0.50 0.10 150) already in the block; not moved.

**Iranian, out of the greens into sand, amber and rust.** Persian was a generated mint 0.064 from
Arabic, Kurdish an olive 0.063 from Iraqi Arabic, Dari a green.
- `indoeuropean.iranian.persian` (now set in cz.txt): 0.76 0.09 148 #8ac191 → 0.78 0.12 75
  #e4ac59, amber. Arabic 0.144 (was 0.064), Azerbaijani 0.287.
- `indoeuropean.iranian.dari` (af, pt): 0.74 0.15 140 #74c163 → 0.74 0.13 65 #e39849, a darker
  amber 0.048 from Persian (one language across the border), 0.117 from Pashto (was 0.122).
- `indoeuropean.iranian.tajik` (now set in cz.txt): 0.70 0.09 119 #98a765 → 0.70 0.12 80 #c5953b.
- `indoeuropean.iranian.kurdish` (now set in iq.txt): 0.76 0.09 119 #abba77 → 0.66 0.13 45
  #d2764a, rust. Iraqi Arabic 0.183 (was 0.063), Levantine 0.187, Turkish 0.183, Balochi (ir) 0.067.
- `indoeuropean.iranian.zazaki` (now set in ae.txt, its first definer): 0.76 0.09 61 #dba475 →
  0.84 0.10 70 #f5bf81. Kurdish 0.19, Romani (tr) 0.103.
- `indoeuropean.iranian.luri` (ir.txt): 0.82 0.09 33 #f9af9f → 0.86 0.08 95 #e1d195.
- `indoeuropean.iranian.gilaki` (ir.txt): 0.76 0.09 32 #e49c8d → 0.64 0.10 105 #958f42.
- `indoeuropean.iranian.mazandarani` (ir.txt): 0.64 0.09 119 #869453 → 0.85 0.10 58 #ffc091.
  In Iran now: Luri | Mazandarani 0.058, Balochi | Gilaki 0.054 (not neighbours), everything else
  0.067+.
- `indoeuropean.iranian.shabaki` (now set in iq.txt; generated mint): 0.64 0.09 148 #659b6d → 0.58
  0.10 70 #a06f30. Kurdish 0.098, Iraqi Arabic 0.152.
- `indoeuropean.iranian.zoroastrian_dari` (now set in ir.txt): 0.88 0.09 105 #e0db95 → 0.72 0.08
  115 #a3ab70. Generated, the Iranian moves had pushed it into the mint (0.76 0.09 148); pinned
  away from the new Luri.
- `indoeuropean.iranian.hazaragi` (au): 0.64 0.13 45 #cc7044 → 0.82 0.11 60 #f9b379. Was the new
  Kurdish's exact colour; now beside Dari, of which it is a variety.
- Not moved: Pashto and Balochi (build.py HAND), the `indoeuropean.iranian` group, Talysh, Tat,
  Yezidi.
- Generated knock-ons, Belarus's and Canada's lists only: `iranian.shughni` #9dd5a3 → #bc7769,
  `oroshori` #b49c5a → #dba475, `yazghulami` #e6bc81 → #f9af9f (by); `parsi` #bc7769 → #abba77
  (ca). Pinned at their old colours so they did not move: `iranian.ossetian` (ge.txt, 0.88 0.09
  76), `afroasiatic.western_neo_aramaic` (sy.txt, 0.86 0.08 209), `germanic.continental.mocheno`
  (it.txt, 0.56 0.13 245), `germanic.continental.cimbrian` (it.txt, 0.57 0.11 230; #0082ad →
  #1082ab, the slot rounded).
- Turkic left alone: Iraqi and Syrian Turkmen, Azerbaijani and Turkish were already 0.12+ from
  both blocks where they meet.

**Yiddish** (us.txt): 0.82 0.10 195 #6cd9d8 → 0.57 0.11 228 #0082a9, a deep blue. In Israel:
Levantine 0.197 (was 0.082), Arabic 0.277 (was 0.058), Hebrew 0.149 (was 0.170). A darker teal
(0.62 0.11 215) was tried first; it sat 0.010 from Afrikaans in the US, UK and Canada. Now Afrikaans
0.056, Czech 0.074, German 0.098. Leftovers where Yiddish is small: Hawaiian 0.024, Acholi 0.024.

**Uyghur** (cn, kg, kz): 0.76 0.14 55 #f49752 → 0.58 0.16 312 #9a59be, a purple. Mandarin 0.253
(was 0.093), Kazakh 0.297; in Kazakhstan and Kyrgyzstan Dungan 0.080, Azerbaijani 0.082. Known
leftover: 0.037-0.044 from Lisu, Jino and Naxi, all in Yunnan.

**Romani** (cz, lt, sk, uk; `romani.romani` now set in cz.txt, was generated): group 0.70 0.12 330
#c882c2 and Romani 0.58 0.12 341 #a95c90 → both 0.58 0.13 30 #bb584a, a dark brick. Turkish 0.190
(group was 0.04 from it). Gold and coral were tried: gold sat 0.023 from Bosnian-Serbian-Croatian
and 0.043 from Montenegrin, coral 0.019 from Serbia's Vlach and 0.061 from Romanian. Now Albanian
0.092, Crimean Tatar 0.088, Kurdish (tr) 0.087, Romanian 0.221. Generated members follow: `vlax`
(co) #b570af → #d87183, `angloromani` (uk) #766ebd → #a84e7c. Known leftovers: Greece's tiny
Shona, Comorian and Tigrinya lists 0.02-0.035.

**Luba-Katanga** (`nigercongo.bantu.luba_katanga`, now pinned in cd.txt at its generated colour
0.64 0.14 40 #d26b46, no change): the stable generator already moved it to 0.192 from Tabwa and
0.159 from Luba-Kasai; pinned so it stays there. No other node moved.

**Wu**: label "Wu (Shanghainese)" → "Wu" in every fragment that defines it (be, bm, bt, ca, cn,
dk, es, fr, ga, gl, gr, hk, is, it, jp, kn, kr, nl, no, nr, pt, se, tt, uy).

**Known leftovers, not moved:**
- Armenian (am.txt, 0.80 0.12 170) is 0.047 from plain Arabic in Iran and Türkiye, 0.065 from
  Mesopotamian Arabic and 0.087 from Levantine in Syria. Moving it to blue put it 0.03 from Tibetan
  and Filipino in North America; the Arabic it meets in Iran and Türkiye is far from Armenian areas.
- Persian is 0.022 from Mongolian and 0.043 from Spanish in diaspora lists (Czechia, Cyprus).

## 2026-10-05, Bosnian, Slovak, Bambara (Anita's request 2026-10-05)

Anita: Bosnian and Croatian a little further apart; Slovak too unlike Czech and Polish; Bambara and
Mooré too alike, and they are close on the ground. Distances are OKLab, not yet judged on the map.

- `indoeuropean.slavic.south.bosnian` (now set in ba.txt; was generated): 0.78 0.13 110 #bdbe53 →
  0.76 0.16 120 #a5bf36, a stronger, greener olive. Croatian 0.122 → 0.152; Serbian 0.132 →
  0.124; Montenegrin 0.075 → 0.102; Macedonian 0.060 → 0.089; Ukrainian 0.065 → 0.082; Rusyn
  0.083 → 0.073; Kurdish (de) 0.048 → 0.070.
- `indoeuropean.slavic.south.montenegrin` (pinned in ba.txt at its old colour, 0.84 0.13 90
  #ebc75e): no visible change. Left generated, it took the slot Bosnian freed, 0.044 from the new
  Bosnian.
- `indoeuropean.slavic.west.slovak` (sk.txt): 0.78 0.13 70 #eca851 → 0.56 0.15 138 #428824, a deep
  green beside Polish's and Russian's. Polish 0.098, Czech 0.128, Bulgarian 0.118, Belarusian
  0.110, Rusyn 0.164, Ukrainian 0.151 (was 0.129), Hungarian over 0.24, Serbian (Vojvodina) 0.223,
  Bunjevac 0.100. Every lighter green tried sat within 0.09 of Polish, Russian or Serbian; at L
  0.70 hue 142 it was 0.013 from Russian.
- `nigercongo.mande.bambara` (ml.txt): 0.84 0.15 95 #e9c944 → 0.76 0.16 110 #b8b91c, an olive
  yellow, darker and greener (towards Mande's greens). Mooré 0.020 → 0.100; in Mali Soninke 0.090
  (was 0.122), Samogo 0.091, Tamasheq 0.104, Bomu and Mamara further, Spanish 0.071 (was 0.078).
  Mooré (bf.txt) not moved: Burkina's neighbours were placed around it.
- Generated knock-ons, all small and only in foreign-language lists: `nigercongo.mande.manding`
  (pl, us) #bdbe53 → #ebc75e, now 0.011 from Mooré in Poland's list; `nigercongo.mande.wojenaka`
  (ca) #d8b349 → #adc35e; `indoeuropean.slavic.west.upper_sorbian` (pl) #558113 → #008892.

## 2026-10-05, stable generated colours; Greek; Dutch

**build.py: a generated colour now depends on the node's own id** (Anita: "yes let's make stable").
It was the node's place among its uncoloured siblings, so any sibling added anywhere moved the rest
(Slovak's hand-pick moved Poland's dialects a step; Russia froze four nodes against it). Now the
id, hashed, picks a slot in a fixed grid of 45 around the parent's colour (5 lightnesses × 9 hues,
±40°), and a node moves off it only when an ancestor or an already-placed sibling sits within 0.04;
siblings are placed in id order. The rule is in build.py's docstring.
- A one-off move: 850 of the 851 generated colours changed. No hand-picked or fragment colour
  changed (HAND, GROUP, every `L C h`), and no group's washed colour either.
- Stability, tested by adding a dummy sibling under each of the 180 parents with generated
  children: sorted last in id order it moves nothing anywhere; sorted first, nothing moves under
  123 of them, and 156 nodes move in all, each one whose slot the dummy took, or a knock-on. The
  old generator moved 464 for the same test. West Slavic: un-picking Slovak moved 7 Polish dialects
  before and moves none now.
- Sibling clashes: exact duplicate groups 14 before, 14 after (dup_colours.py); pairs with a
  generated member that are identical 37 → 20, under 0.02 apart 160 → 32, under 0.04 apart 655 →
  182. The identical ones left are all greys under `signlanguage` and `other`, which have no chroma,
  so only lightness can separate them; the old generator duplicated them too.
- South Asia: 219 small India and Nepal languages moved (none of the hand-picked big ones). Looked
  at on the map in the Nepal hills, the Bhil belt and the Northeast: the families still read as
  their regions of the wheel, and nothing new clashes.

**Greek** (cz, th, us; `indoeuropean.hellenic` too, so the group's wash follows): 0.70 0.13 250
#5aa3ec → 0.56 0.18 295 #7f57d1, a deep violet. Was 0.046 from German (#359bd9), drawn together in
Germany and US cities. Now 0.179 from German and at least 0.128 from everything mapped in Germany
and the US: nearest Vietnamese, Portuguese and Burmese 0.128, French 0.136, Turkish 0.174; Polish,
Italian, Albanian and Spanish much further. Known leftovers where Greek is small: Basque 0.027
(Australia), Dungan 0.024 (Russia).

**Dutch** (cz, us): 0.72 0.12 225 #39b4dd → 0.78 0.12 225 #51c7f1, the same blue made lighter.
Javanese (id.txt) was 0.081 from it in Commewijne and Wanica (Suriname's leftover, below); moving
Dutch was cheaper, since Javanese is Indonesia's largest language and id.txt placed its neighbours
around it. Now 0.131 from Javanese, 0.141 from English, 0.142 from Lokono, 0.162 from Portuguese.
Known leftover: 0.031 from Pennsylvania German in the US (was 0.048), where Dutch is small.

## 2026-10-05, Suriname (sr agent; distances computed, not yet judged on the map)

- `creole.english_based.sranan` (pl.txt, generated) → 0.78 0.14 128 #9dc861, set in sr.txt.
  Suriname's lingua franca, beside Dutch everywhere; kept 0.096 off Lokono's pale green.
- New in sr.txt: Sarnami 0.74 0.16 50, Saramaccan 0.68 0.15 95, Ndyuka 0.62 0.12 170, Pamaka
  0.86 0.08 85 (Maroon creoles stepped apart along the rivers they share).
- Known leftover, not touched (both other countries' colours): Dutch ~ Javanese 0.081, and they
  meet in Commewijne and Wanica. Fixed the same day by moving Dutch (section above).

## 2026-10-05, Paraguay (py agent; distances computed, not yet judged on the map)

- `tupian.tupiguarani.guarani.paraguayan` (ar.txt): 0.86 0.15 44 #ffaf7c → 0.78 0.15 45 #ff9661.
  Paraguay's majority, beside Spanish everywhere: it was 0.088 from the new yellow Spanish, now
  0.108. In Buenos Aires 0.137 from Quechua (was 0.187).
- `tupian.tupiguarani.guarani.ava_guarani` (br.txt): 0.88 0.15 46 #ffb67f → 0.90 0.10 85 #fdd990.
  Was 0.016 from Paraguayan Guarani, all around it in Canindeyu and Alto Parana; now a pale gold.
  In Brazil further from Nhandeva (0.186, was 0.104) and Mbya.
- `tupian.tupiguarani.guarani.mbya` (br.txt): 0.74 0.15 31 #fb836f → 0.70 0.15 355 #e573a3. Would
  have been 0.053 from the new Paraguayan Guarani; a pink now. In Brazil 0.133 from Nhandeva (was
  0.037), 0.110 from Kaingang; known leftover 0.049 from Xokleng in Santa Catarina.
- `matacoan.nivacle` (ar.txt): 0.64 0.12 215 #009eb9 → 0.56 0.11 190 #008883. Was 0.056 from
  German, and both are large in Boqueron (Filadelfia and the Mennonite colonies); now 0.137. In
  Argentina 0.183 from Wichi (was 0.111).
- Known leftover, not touched (shared globally): Portuguese and German are 0.059 apart, and both
  are drawn in the Alto Parana and Itapua colonies.

## 2026-10-05, yellow Spanish, red Latin American languages, red Chinese

Anita, 2026-10-05: Spanish, Portuguese and English had gone "a bit too washed out"; colour should
not read as prevalence; the US and South America felt drab; families should be identifiable at a
glance where it helps (Dravidian is the model). Then: "make Spanish a slightly dim yellow and change
all the yellowish minority languages in Latin America to be more red ... yellow acts better as a
sort of backdrop colour. Then maybe we could even move Chinese to reddish." And: Mandarin should sit
close to plain "Chinese", since undifferentiated Chinese is mostly Mandarin.

**The backdrops**
- `indoeuropean.romance.spanish` (us, cz): 0.64 0.10 32 #c17565 → 0.78 0.10 95 #cbb76a. A dim
  straw yellow: clearly a colour, quiet enough to sit under everything. Most of the work below is
  clearing the yellow range around it.
- `indoeuropean.romance.portuguese` (br): 0.62 0.07 215 #5191a1 → 0.64 0.11 265 #6b8acf. A calm
  periwinkle blue. It sits opposite the yellow along Brazil's borders and the Amazon's languages,
  now mostly reds, stand out on it. Teal was tried and is crowded: Tukano, the Panoan languages,
  Filipino and Low German sat on it. Violet was tried and clashed with the Yanomaman languages.
  Closest on the ground: Ninam 0.053 (Roraima) and German 0.059 (Switzerland).
- `indoeuropean.germanic.english` (build.py HAND) and the `indoeuropean.germanic` GROUP: 0.92 0.02
  250 #dbe6f2 → 0.90 0.05 245 #c3e2fe. A pale but real blue, the Germanic blue made light. The
  near-white made rural US look empty and the cities grey.

**Chinese, now red** (us, uk, ca, hk, sg, mu, zm, cz, kg fragments)
- `sinotibetan.sinitic` 267 blue → 0.62 0.19 38 #e04f1a, a vermilion. `.mandarin` → 0.68 0.17 41
  #ec6d3a, the same hue a step lighter (0.064 apart), so "Chinese" and Mandarin read as nearly the
  same thing. The red is placed between Urdu (0.065) and Punjabi, Korean, French (raspberry, 0.2+
  away) and Vietnamese. Tagalog is far away.
- The other varieties sit around it, each distinct from both: Cantonese a light coral (Hong Kong's
  ground), Hakka a dark rose, Min Nan a dark brick, Teochew amber, Sze Yap pale peach, Wu dark
  amber, Min Dong dark plum. Hakka, Wu and Min Dong were generated; they are pinned now, because
  generated, they landed on the new reds. Dungan (kg) is pinned at its old violet: in Kyrgyzstan
  it sits next to Kyrgyz, which is red.
- `afroasiatic.ethiosemitic.tigrinya` (et, us) 0.64 0.17 35 → 0.58 0.14 15: it sat 0.03 from the
  new Chinese in the US, UK and Canada.
- French stays: the new Chinese is 0.2+ from it. Hakka is 0.067 from French; both are small where
  they meet.

**Off the yellow**
- `creole.french_based.haitian` (us) gold 0.80 0.17 75 → lilac 0.78 0.12 305 #c9a3f5. The gold was
  next to the new Spanish in NYC and Miami. An orchid (0.74 0.13 320) was tried first and read as a
  paler French in Montréal-Nord.
- `nigercongo.mande` (us, ca) 0.78 0.13 95 → 0.72 0.13 130: it was 0.03 from Spanish (the Bronx).
  Its generated members follow it into greens.
- Left near Spanish, all small where they meet it: Pashto 0.045 (South Asia, untouched), Mongolian
  0.055, Sinhala 0.064, Chaldean 0.065 (Detroit), Kabuverdianu 0.069; the Iranian, Baltic and Kwa
  group washes.

**Latin American families, from yellows to reds.** Each family now holds one band of the red end,
and its members are told apart by lightness and chroma, as Dravidian is. Every explicitly coloured
member moved by the same rule: new hue = band centre + k × (old hue − old family centre), keeping L
and C. The k compresses each family's old spread into its band. Generated members follow their
group.
- Tupian, Tupi-Guarani and Guarani: centre 70 → 38, k 0.28 (vermilion to orange, about 25-51).
- Macro-Jê: 15 → 2, k 0.22 (rose-red, about 352-17).
- Cariban: 110 → 325, k 0.25 (orchid-magenta, about 314-336). Applied in two rounds: first at
  340, k 0.30, then pulled to 325 because the Xingu's Cariban and Jê peoples (Kuikuro, Kalapalo,
  Ikpeng | Kisêdjê, Tapayuna, Mentuktire) came out 0.03 apart. Now 0.057 or more.
- Quechuan: 45 → 15, k 0.5. `quechuan.quechua` hand-set to 0.66 0.17 20 #e75f66, a strong red
  (apricot before). Quechua | Aymara is now 0.24; Quechua | Spanish 0.21.
- Chocoan (Embera) 50 → 30; Uto-Aztecan 100 → 22, k 0.45, with `utoaztecan.nahuatl` hand-set to
  0.68 0.17 25; Yuman and Tequistlatecan 60 → 35; Guahiboan (Sikuani was Spanish's own hue) 95 →
  30, k 0.4; Nambikwaran 35 → 22, k 0.5, with chroma capped at 0.09, so it reads as dusty rose
  beside the Tupian oranges; Naduhup 30 → 20; Mura 55 → 15; Katukinan 85 → 50; Zaparoan 85 → 35;
  "no established classification" 60 → 30, k 0.4 (still low-chroma, now dusty pinks and tans).
- Chiquitano (bo, br) olive → rose (0.74 0.12 355). Huarpe (ar) yellow-green → 0.78 0.11 20.
- `tupian.tupiguarani.guarayu` → 0.66 0.15 32: it landed 0.016 from Guarani in Santa Cruz.
- Xingu hand fixes after the transform: Kaiabi → 0.84 0.12 35 (it was 0.014 from Kamayurá),
  Matipu → 0.66 0.14 315 (0.022 from Kuikuro), Mentuktire → 0.64 0.17 10 (0.021 from Kisêdjê).
- Not moved: the Mayan languages (blues, violets and cyans, already clear of yellow), Aymara,
  Mapuche, the Tucanoan, Panoan and Arawakan cool families, the Isolates.

**Yellow-green members pulled back into their green families**: Arawakan, Oto-Manguean and
Chapacuran members under hue 135 → 140 + 0.4 × (h − 115); also Totoró, Uru-Chipaya, Shawi
(Cahuapanan), Mastanawa, San Andrés Creole, Garifuna, Kickapoo and the Zapotecan and Popolocan
nodes (Ixcateco and Chocholteco were generated yellow-greens). Wayuu, the largest, is now 0.60 0.15
140.

**Not done, for whoever owns them:**
- `cariban.pemon` and `cariban.japreria` in ve.txt (another agent was writing it) keep 190 and 140.
  They are clear of Spanish, but they are outside the Cariban orchid band. The same rule would put
  them at 0.72 0.13 3 and 0.66 0.15 332.
- Small siblings left close because they are similar and small: Mbya | Nhandeva 0.037, Tenetehara |
  Ka'apor 0.036, Guajajara | Awá-Guajá 0.044.
- Stale comments that still name old colours: ar.txt header ("Guarani's yellows", "Huarpean green-yellow"), co.md
  and pe.md (Spanish "red-orange"), us.md.

Every node whose drawn colour changed in this pass (old → new; "generated" = no fragment colour):

### indoeuropean

- `indoeuropean.germanic` (build.py GROUP): 0.92 0.02 250 #dbe6f2 → 0.9 0.05 245 #c3e2fe
- `indoeuropean.germanic.english` (build.py HAND): 0.92 0.02 250 #dbe6f2 → 0.9 0.05 245 #c3e2fe
- `indoeuropean.romance.portuguese`: 0.62 0.07 215 #5191a1 → 0.64 0.11 265 #6b8acf
- `indoeuropean.romance.spanish`: 0.64 0.1 32 #c17565 → 0.78 0.1 95 #cbb76a

### sinotibetan

- `sinotibetan.sinitic`: 0.62 0.2 267 #507afd → 0.62 0.19 38 #e04f1a
- `sinotibetan.sinitic.mandarin`: 0.56 0.17 298 #8258c9 → 0.68 0.17 41 #ec6d3a
- `sinotibetan.sinitic.min_nan`: 0.52 0.12 250 #286cab → 0.5 0.12 28 #9c443b
- `sinotibetan.sinitic.min_dong`: generated #8d91ff → 0.5 0.1 350 #8d4a6b
- `sinotibetan.sinitic.wu`: generated #006de2 → 0.62 0.12 60 #ba7331
- `sinotibetan.sinitic.cantonese`: 0.8 0.12 245 #78c6ff → 0.8 0.11 15 #fca0a7
- `sinotibetan.sinitic.hakka`: generated #00a0ff → 0.58 0.14 5 #bc516f
- `sinotibetan.sinitic.teochew`: 0.66 0.11 222 #2da1c2 → 0.76 0.13 62 #ec9d53
- `sinotibetan.sinitic.siyi`: 0.84 0.09 300 #d3befd → 0.88 0.07 60 #fbcdaa

### creole

- `creole.english_based.san_andres`: 0.84 0.12 120 #c0d67a → 0.84 0.12 150 #8fe1a1
- `creole.french_based.haitian`: 0.8 0.17 75 #fcab00 → 0.78 0.12 305 #c9a3f5

### afroasiatic

- `afroasiatic.ethiosemitic.tigrinya`: 0.64 0.17 35 #e05d3d → 0.58 0.14 15 #be525f

### nigercongo

- `nigercongo.mande.mandinka`: generated #ffc081 → generated #e5c057
- `nigercongo.mande.dan`: generated #d08436 → generated #9b8600
- `nigercongo.mande.kpelle`: generated #fcb861 → generated #bfb94c
- `nigercongo.mande.loma`: generated #ffcd6d → generated #c3cd65
- `nigercongo.mande.mano`: generated #bb9220 → generated #79922b
- `nigercongo.mande`: 0.78 0.13 95 #d1b64a → 0.72 0.13 130 #89b559
- `nigercongo.mande.bambara`: generated #e2c65b → generated #99c569
- `nigercongo.mande.wojenaka`: generated #e7db70 → generated #9ad884
- `nigercongo.mande.soninke`: generated #9c9f31 → generated #4b9b54
- `nigercongo.mande.dyula`: generated #bed36d → generated #6acc8e
- `nigercongo.mande.susu`: generated #c0e887 → generated #6bdeaa
- `nigercongo.mande.manding`: generated #74aa54 → generated #009f7a

### algic

- `algic.kickapoo`: generated #d9bd44 → 0.72 0.13 140 #77b868

### tupian

- `tupian`: 0.8 0.15 70 #faab3f → 0.8 0.15 38 #ff9974
- `tupian.tupiguarani`: 0.74 0.15 70 #e69825 → 0.74 0.15 38 #f98662
- `tupian.tupiguarani.guarani`: 0.86 0.15 88 #faca4b → 0.86 0.15 43 #ffae7e
- `tupian.tupiguarani.guarani.mbya`: 0.74 0.15 44 #f78857 → 0.74 0.15 31 #fb836f
- `tupian.tupiguarani.guarani.paraguayan`: 0.86 0.15 92 #f4cd4b → 0.86 0.15 44 #ffaf7c
- `tupian.tupiguarani.guarani.chiriguano`: 0.7 0.15 62 #df8623 → 0.7 0.15 36 #ec7859
- `tupian.tupiguarani.guarani.tupi_guarani`: 0.78 0.11 78 #deae62 → 0.78 0.11 40 #f49f81
- `tupian.tupiguarani.guarani.tapiete`: 0.9 0.1 70 #ffd394 → 0.9 0.1 38 #ffc7ae
- `tupian.tupiguarani.guarayu`: 0.86 0.15 108 #dbd856 → 0.66 0.15 32 #df6a55
- `tupian.tupiguarani.siriono`: generated #fbc044 → generated #ffab6c
- `tupian.tupiguarani.yuqui`: generated #ce710c → generated #d8625a
- `tupian.tupiguarani.pauserna`: generated #ff9c66 → generated #ff92a0
- `tupian.tupiguarani.guarani.kaiowa`: 0.62 0.15 52 #ca6719 → 0.62 0.15 33 #d15e47
- `tupian.tupiguarani.guajajara`: 0.86 0.15 56 #ffb569 → 0.86 0.15 34 #ffab8e
- `tupian.mawe`: 0.62 0.15 84 #b07c00 → 0.62 0.15 42 #ce6234
- `tupian.mawe.satere_mawe`: 0.8 0.15 100 #d5bf36 → 0.8 0.15 46 #ff9c66
- `tupian.tupiguarani.nheengatu`: 0.9 0.15 70 #ffcb64 → 0.9 0.15 38 #ffb994
- `tupian.munduruku`: 0.68 0.15 40 #e4744b → 0.68 0.15 30 #e6705f
- `tupian.munduruku.munduruku`: 0.58 0.15 70 #b16600 → 0.58 0.15 38 #c25430
- `tupian.tupiguarani.guarani.nhandeva`: 0.74 0.15 96 #c7a90e → 0.74 0.15 45 #f78955
- `tupian.tupiguarani.ka_apor`: 0.68 0.15 78 #ca8a00 → 0.68 0.15 40 #e4744b
- `tupian.tupiguarani.guarani.ava_guarani`: 0.88 0.15 100 #efd956 → 0.88 0.15 46 #ffb67f
- `tupian.monde`: 0.8 0.15 62 #ffa54b → 0.8 0.15 36 #ff9878
- `tupian.tupiguarani.kokama`: 0.6 0.15 40 #c85b32 → 0.6 0.15 30 #ca5747
- `tupian.tupiguarani.parakana`: 0.74 0.15 70 #e69825 → 0.74 0.15 38 #f98662
- `tupian.tupiguarani.kaiabi`: 0.58 0.15 115 #798300 → 0.84 0.12 35 #ffae95
- `tupian.tupiguarani.wajapi`: 0.62 0.15 52 #ca6719 → 0.62 0.15 33 #d15e47
- `tupian.tupiguarani.tupi_potiguara`: 0.62 0.15 84 #b07c00 → 0.62 0.15 42 #ce6234
- `tupian.tupiguarani.tapirape`: 0.68 0.15 40 #e4744b → 0.68 0.15 30 #e6705f
- `tupian.monde.paiter`: 0.9 0.15 70 #ffcb64 → 0.9 0.15 38 #ffb994
- `tupian.monde.zoro`: 0.58 0.15 70 #b16600 → 0.58 0.15 38 #c25430
- `tupian.monde.gaviao_de_rondonia`: 0.74 0.15 44 #f78857 → 0.74 0.15 31 #fb836f
- `tupian.tupiguarani.kawahiva`: 0.8 0.15 100 #d5bf36 → 0.8 0.15 46 #ff9c66
- `tupian.tupiguarani.kawahiva.tenharim`: 0.74 0.15 96 #c7a90e → 0.74 0.15 45 #f78955
- `tupian.tupiguarani.awa_guaja`: 0.8 0.15 62 #ffa54b → 0.8 0.15 36 #ff9878
- `tupian.tupiguarani.kambeba`: 0.88 0.15 100 #efd956 → 0.88 0.15 46 #ffb67f
- `tupian.tupiguarani.surui_do_para`: 0.74 0.15 115 #a8b532 → 0.74 0.15 51 #f48c49
- `tupian.tupiguarani.tenetehara`: 0.66 0.15 115 #909c00 → 0.66 0.15 51 #d8732d
- `tupian.tupari`: 0.68 0.15 78 #ca8a00 → 0.68 0.15 40 #e4744b
- `tupian.tupari.tupari`: 0.86 0.15 56 #ffb569 → 0.86 0.15 34 #ffab8e
- `tupian.ramarama`: 0.9 0.15 103 #f0e260 → 0.9 0.15 47 #ffbd84
- `tupian.ramarama.arara_de_rondonia`: 0.8 0.15 100 #d5bf36 → 0.8 0.15 46 #ff9c66
- `tupian.arikem.karitiana`: 0.9 0.15 70 #ffcb64 → 0.9 0.15 38 #ffb994
- `tupian.tupiguarani.zo_e`: 0.58 0.15 70 #b16600 → 0.58 0.15 38 #c25430
- `tupian.tupiguarani.asurini_do_xingu`: 0.74 0.15 44 #f78857 → 0.74 0.15 31 #fb836f
- `tupian.tupiguarani.aweti`: 0.74 0.15 96 #c7a90e → 0.74 0.15 45 #f78955
- `tupian.juruna`: 0.86 0.15 88 #faca4b → 0.86 0.15 43 #ffae7e
- `tupian.juruna.xipaya`: 0.8 0.15 62 #ffa54b → 0.8 0.15 36 #ff9878
- `tupian.juruna.yudja`: 0.68 0.15 78 #ca8a00 → 0.68 0.15 40 #e4744b
- `tupian.tupiguarani.kawahiva.kawahiba_dos_amondawa`: 0.6 0.15 40 #c85b32 → 0.6 0.15 30 #ca5747
- `tupian.munduruku.kuruaya`: 0.74 0.15 70 #e69825 → 0.74 0.15 38 #f98662
- `tupian.tupiguarani.kamayura`: 0.58 0.15 97 #947800 → 0.58 0.15 46 #bf571b
- `tupian.monde.arua`: 0.62 0.15 52 #ca6719 → 0.62 0.15 33 #d15e47
- `tupian.tupiguarani.kawahiva.parintintim`: 0.86 0.15 56 #ffb569 → 0.86 0.15 34 #ffab8e
- `tupian.tupiguarani.kawahiva.jiahui`: 0.62 0.15 84 #b07c00 → 0.62 0.15 42 #ce6234
- `tupian.tupari.makurap`: 0.8 0.15 100 #d5bf36 → 0.8 0.15 46 #ff9c66
- `tupian.tupiguarani.amanaye`: 0.7 0.15 58 #e2832d → 0.7 0.15 35 #ec785b
- `tupian.tupiguarani.apiaka`: 0.78 0.15 82 #e7ac2a → 0.78 0.15 41 #ff9469
- `tupian.tupari.sakurabiat`: 0.58 0.15 70 #b16600 → 0.58 0.15 38 #c25430
- `tupian.tupari.ajuru`: 0.74 0.15 44 #f78857 → 0.74 0.15 31 #fb836f
- `tupian.tupiguarani.anambe`: 0.62 0.15 106 #938a00 → 0.62 0.15 48 #cc6526
- `tupian.purubora.purobora`: 0.88 0.15 100 #efd956 → 0.88 0.15 46 #ffb67f
- `tupian.tupiguarani.xeta`: 0.9 0.15 115 #dbea6d → 0.9 0.15 51 #ffc07e
- `tupian.tupiguarani.kawahiva.kawahiba_dos_karipuna`: 0.74 0.15 70 #e69825 → 0.74 0.15 38 #f98662
- `tupian.tupari.akuntsu`: 0.86 0.15 88 #faca4b → 0.86 0.15 43 #ffae7e
- `tupian.tupiguarani.kawahiva.juma`: 0.62 0.15 52 #ca6719 → 0.62 0.15 33 #d15e47
- `tupian.monde.salamay`: 0.86 0.15 56 #ffb569 → 0.86 0.15 34 #ffab8e
- `tupian.tupiguarani.omagua`: generated #b79500 → generated #d88018

### macroje

- `macroje`: 0.7 0.17 15 #f56b7c → 0.7 0.17 2 #f06a96
- `macroje.je`: 0.74 0.17 15 #ff7788 → 0.74 0.17 2 #ff77a2
- `macroje.je.kaingang`: 0.86 0.17 33 #ffa386 → 0.86 0.17 6 #ff9ec0
- `macroje.je.xavante`: 0.62 0.17 357 #d25188 → 0.62 0.17 358 #d25186
- `macroje.je.kayapo`: 0.86 0.17 1 #ff9eca → 0.86 0.17 359 #ff9ece
- `macroje.maxakali`: 0.8 0.17 45 #ff9659 → 0.8 0.17 9 #ff8aa7
- `macroje.maxakali.pataxo`: 0.68 0.17 345 #df67b0 → 0.68 0.17 355 #e6659e
- `macroje.je.xerente`: 0.58 0.17 15 #ca4459 → 0.58 0.17 2 #c64472
- `macroje.je.timbira`: 0.62 0.17 29 #d95446 → 0.62 0.17 5 #d55078
- `macroje.je.timbira.kraho`: 0.74 0.17 349 #f779bc → 0.74 0.17 356 #fb78ae
- `macroje.karaja`: 0.9 0.17 15 #ffabb9 → 0.9 0.17 2 #ffabd5
- `macroje.karaja.karaja`: 0.74 0.17 41 #ff804e → 0.74 0.17 8 #ff7796
- `macroje.yate`: 0.8 0.17 7 #ff8aab → 0.8 0.17 0 #ff8ab9
- `macroje.yate.yathe`: 0.68 0.17 23 #ef6665 → 0.68 0.17 4 #ea648c
- `macroje.je.apinaye`: 0.6 0.17 345 #c34e97 → 0.6 0.17 355 #ca4b85
- `macroje.maxakali.maxakali`: 0.74 0.17 15 #ff7788 → 0.74 0.17 2 #ff77a2
- `macroje.je.xikrin`: 0.7 0.17 60 #e88000 → 0.7 0.17 12 #f46a82
- `macroje.je.timbira.kanela`: 0.88 0.17 45 #ffb073 → 0.88 0.17 9 #ffa4c0
- `macroje.je.timbira.kanela.rankokamekra`: 0.62 0.17 357 #d25188 → 0.62 0.17 358 #d25186
- `macroje.karaja.karaja_javae`: 0.86 0.17 1 #ff9eca → 0.86 0.17 359 #ff9ece
- `macroje.bororo`: 0.62 0.17 29 #d95446 → 0.62 0.17 5 #d55078
- `macroje.bororo.bororo`: 0.8 0.17 45 #ff9659 → 0.8 0.17 9 #ff8aa7
- `macroje.je.mebengokre_kayapo`: 0.68 0.17 345 #df67b0 → 0.68 0.17 355 #e6659e
- `macroje.je.timbira.krikati`: 0.9 0.17 15 #ffabb9 → 0.9 0.17 2 #ffabd5
- `macroje.je.xokleng`: 0.74 0.17 15 #ff7788 → 0.74 0.17 2 #ff77a2
- `macroje.je.timbira.kanela.apanyekra`: 0.74 0.17 41 #ff804e → 0.74 0.17 8 #ff7796
- `macroje.je.kisedje`: 0.8 0.17 7 #ff8aab → 0.8 0.17 0 #ff8ab9
- `macroje.je.panara`: 0.68 0.17 23 #ef6665 → 0.68 0.17 4 #ea648c
- `macroje.rikbaktsa`: 0.88 0.17 45 #ffb073 → 0.88 0.17 9 #ffa4c0
- `macroje.rikbaktsa.rikbaktsa`: 0.6 0.17 345 #c34e97 → 0.6 0.17 355 #ca4b85
- `macroje.maxakali.pataxo_ha_ha_hae`: 0.9 0.17 330 #ffb2ff → 0.9 0.17 352 #ffacea
- `macroje.kariri`: 0.74 0.17 349 #f779bc → 0.74 0.17 356 #fb78ae
- `macroje.kariri.kipea`: 0.86 0.17 33 #ffa386 → 0.86 0.17 6 #ff9ec0
- `macroje.kariri.dzubokua`: 0.62 0.17 357 #d25188 → 0.62 0.17 358 #d25186
- `macroje.je.xacriaba`: 0.9 0.17 60 #ffc05b → 0.9 0.17 12 #ffabc0
- `macroje.krenak`: 0.58 0.17 334 #b34ca3 → 0.58 0.17 353 #c24583
- `macroje.krenak.krenak`: 0.8 0.17 45 #ff9659 → 0.8 0.17 9 #ff8aa7
- `macroje.je.timbira.gaviao_parkateje`: 0.68 0.17 345 #df67b0 → 0.68 0.17 355 #e6659e
- `macroje.je.tapayuna`: 0.9 0.17 15 #ffabb9 → 0.9 0.17 2 #ffabd5
- `macroje.bororo.umutina`: 0.58 0.17 15 #ca4459 → 0.58 0.17 2 #c64472
- `macroje.karaja.xambioa`: 0.74 0.17 349 #f779bc → 0.74 0.17 356 #fb78ae
- `macroje.je.timbira.gaviao_kykateje`: 0.74 0.17 41 #ff804e → 0.74 0.17 8 #ff7796
- `macroje.brobo`: 0.86 0.17 334 #ffa4ff → 0.86 0.17 353 #ff9fdb
- `macroje.brobo.brobo`: 0.68 0.17 23 #ef6665 → 0.68 0.17 4 #ea648c
- `macroje.je.menkrangnoti`: 0.88 0.17 45 #ffb073 → 0.88 0.17 9 #ffa4c0
- `macroje.kariri.kariri`: 0.6 0.17 345 #c34e97 → 0.6 0.17 355 #ca4b85
- `macroje.je.timbira.krenye`: 0.74 0.17 15 #ff7788 → 0.74 0.17 2 #ff77a2
- `macroje.kamaka`: 0.86 0.17 33 #ffa386 → 0.86 0.17 6 #ff9ec0
- `macroje.kamaka.kamaka`: 0.62 0.17 357 #d25188 → 0.62 0.17 358 #d25186
- `macroje.ofaye`: 0.86 0.17 1 #ff9eca → 0.86 0.17 359 #ff9ece
- `macroje.ofaye.ofaye`: 0.62 0.17 29 #d95446 → 0.62 0.17 5 #d55078
- `macroje.je.timbira.gaviao_pykopje`: 0.8 0.17 45 #ff9659 → 0.8 0.17 9 #ff8aa7
- `macroje.je.mentuktire`: 0.78 0.17 330 #f38ceb → 0.64 0.17 10 #de5774
- `macroje.kariri.xoco`: 0.9 0.17 15 #ffabb9 → 0.9 0.17 2 #ffabd5
- `macroje.je.timbira.kreepyn_kateje`: 0.58 0.17 15 #ca4459 → 0.58 0.17 2 #c64472

### cariban

- `cariban`: 0.86 0.15 110 #d7d958 → 0.86 0.15 325 #ffadff
- `cariban.makuxi`: 0.74 0.15 110 #b1b228 → 0.74 0.15 325 #da87de
- `cariban.wai_wai`: 0.86 0.15 128 #b4e373 → 0.86 0.15 330 #ffacff
- `cariban.waimiri_atroari`: 0.62 0.15 92 #a78100 → 0.62 0.15 320 #ad65be
- `cariban.taurepang`: 0.86 0.15 96 #eed04c → 0.86 0.15 322 #ffafff
- `cariban.tiriyo`: 0.62 0.15 124 #74940e → 0.62 0.15 328 #b561b3
- `cariban.ingariko`: 0.8 0.15 140 #87d576 → 0.8 0.15 332 #f598e8
- `cariban.hixkaryana`: 0.68 0.15 80 #c88b00 → 0.68 0.15 318 #be78d4
- `cariban.bakairi`: 0.9 0.15 110 #e4e667 → 0.9 0.15 325 #ffbaff
- `cariban.ye_kwana`: 0.58 0.15 110 #818000 → 0.58 0.15 325 #a557aa
- `cariban.apalai`: 0.74 0.15 84 #d7a10c → 0.74 0.15 318 #d28ae8
- `cariban.kalapalo`: 0.74 0.15 136 #7ec05b → 0.74 0.15 332 #e185d5
- `cariban.kuikuro`: 0.8 0.15 102 #d1c038 → 0.8 0.15 323 #ec9bf5
- `cariban.katxuyana`: 0.68 0.15 118 #91a41d → 0.68 0.15 327 #c874c8
- `cariban.ikpeng`: 0.88 0.15 140 #a0f08f → 0.88 0.15 332 #ffb1ff
- `cariban.arara_do_para`: 0.6 0.15 80 #ae7300 → 0.6 0.15 318 #a55fb9
- `cariban.wayana`: 0.66 0.15 155 #29ac68 → 0.66 0.15 336 #ca6bb5
- `cariban.patamona`: 0.82 0.15 65 #ffae4e → 0.82 0.15 314 #e7a5ff
- `cariban.nahukua`: 0.58 0.15 152 #0a924b → 0.58 0.15 336 #af539c
- `cariban.galibi_kali_na`: 0.7 0.15 98 #b89e00 → 0.7 0.15 322 #c97cd5
- `cariban.katwena`: 0.74 0.15 155 #4bc680 → 0.74 0.15 336 #e584cf
- `cariban.faruk_woto`: 0.74 0.15 65 #ea942f → 0.74 0.15 314 #cd8cec
- `cariban.enepa`: 0.58 0.15 134 #538c20 → 0.58 0.15 331 #ab54a3
- `cariban.arekuna`: 0.66 0.15 134 #6aa53d → 0.66 0.15 331 #c56dbc
- `cariban.kamarakoto`: 0.9 0.15 65 #ffc86a → 0.9 0.15 314 #ffbfff
- `cariban.akawaio`: 0.82 0.15 86 #efbc3a → 0.82 0.15 319 #eea3ff
- `cariban.xerewyana`: 0.62 0.15 143 #479c44 → 0.62 0.15 333 #ba60ac
- `cariban.tunayana`: 0.7 0.15 146 #58b762 → 0.7 0.15 334 #d578c5
- `cariban.matipu`: 0.82 0.15 119 #bad157 → 0.66 0.14 315 #b276cd
- `cariban.akuriyo`: 0.82 0.15 155 #68e099 → 0.82 0.15 336 #ff9de9
- `cariban.yukpa`: 0.78 0.15 110 #bebf3a → 0.78 0.15 325 #e794ec
- `cariban.carijona`: generated #d4f47d → generated #ffbdff

### quechuan

- `quechuan`: 0.66 0.16 30 #e36654 → 0.66 0.16 8 #e1627f
- `quechuan.quechua`: 0.84 0.15 62 #ffb259 → 0.66 0.17 20 #e75f66
- `quechuan.kolla`: 0.78 0.12 5 #f896ad → 0.78 0.12 355 #f496bb
- `quechuan.kolla_quechua`: 0.66 0.13 15 #d56e78 → 0.66 0.13 0 #d16e8f
- `quechuan.diaguita_quechua`: 0.86 0.09 25 #ffbab3 → 0.86 0.09 5 #ffb9c9
- `quechuan.kallawaya`: generated #ff8b5a → generated #ff8386
- `quechuan.quichua`: 0.86 0.16 48 #ffae6e → 0.86 0.16 16 #ffa2ad
- `quechuan.inga`: 0.8 0.15 42 #ff9b6d → 0.8 0.15 14 #ff92a0
- `quechuan.otavaleno`: generated #c04350 → generated #b94377
- `quechuan.kichwa`: 0.76 0.15 75 #e8a127 → 0.76 0.15 30 #ff8977

### chocoan

- `chocoan`: 0.76 0.15 50 #fc9252 → 0.76 0.15 30 #ff8977
- `chocoan.embera`: 0.76 0.15 50 #fc9252 → 0.76 0.15 30 #ff8977
- `chocoan.embera.embera`: 0.74 0.16 50 #f98942 → 0.74 0.16 30 #ff7f6c
- `chocoan.embera.katio`: 0.84 0.14 80 #fac053 → 0.84 0.14 45 #ffac7b
- `chocoan.embera.chami`: 0.64 0.17 32 #e05c45 → 0.64 0.17 21 #e0585e
- `chocoan.embera.dobida`: 0.8 0.11 15 #fca0a7 → 0.8 0.11 12 #fca0ab
- `chocoan.embera.eperara`: 0.7 0.14 68 #d78c29 → 0.7 0.14 39 #e77d5a
- `chocoan.wounaan`: 0.88 0.1 40 #ffc1a6 → 0.88 0.1 25 #ffbeb6

### utoaztecan

- `utoaztecan`: 0.8 0.15 105 #ccc23b → 0.8 0.15 24 #ff948e
- `utoaztecan.nahuatl`: 0.84 0.17 80 #ffbc19 → 0.68 0.17 25 #ef6661
- `utoaztecan.corachol`: generated #d7eb6f → generated #ffb897
- `utoaztecan.corachol.cora`: 0.64 0.15 125 #779b1e → 0.64 0.15 33 #d7644d
- `utoaztecan.corachol.huichol`: 0.88 0.12 100 #eada78 → 0.88 0.12 22 #ffb7b4
- `utoaztecan.taracahitan`: generated #bf9b00 → generated #ec7385
- `utoaztecan.taracahitan.tarahumara`: 0.78 0.15 115 #b5c242 → 0.78 0.15 29 #ff8f7f
- `utoaztecan.taracahitan.guarijio`: generated #aecd55 → generated #ff9a6f
- `utoaztecan.taracahitan.mayo`: 0.86 0.13 95 #ecd065 → 0.86 0.13 20 #ffadae
- `utoaztecan.taracahitan.yaqui`: 0.7 0.14 80 #cb9317 → 0.7 0.14 13 #e77685
- `utoaztecan.tepiman`: generated #ffc64d → generated #ffa5cd
- `utoaztecan.tepiman.tepehuan`: generated #ffe262 → generated #ffb9ca
- `utoaztecan.tepiman.tepehuan.north`: generated #f4ea68 → generated #ffbbb3
- `utoaztecan.tepiman.tepehuan.south`: 0.76 0.15 70 #ed9e2f → 0.76 0.15 8 #ff859f
- `utoaztecan.tepiman.pima`: generated #ee9d30 → generated #f587c2
- `utoaztecan.tepiman.papago`: generated #ffca7b → generated #ffbdff

### yuman

- `yuman`: 0.8 0.12 60 #f6ab6b → 0.8 0.12 35 #ffa189
- `yuman.paipai`: generated #ffd181 → generated #ffc598
- `yuman.kiliwa`: generated #da8659 → generated #df7f7b
- `yuman.kumiai`: generated #ffb49a → generated #ffb0bd
- `yuman.cucapa`: generated #cea448 → generated #e39759

### tequistlatecan

- `tequistlatecan`: 0.82 0.1 50 #f9b189 → 0.82 0.1 30 #feac9e
- `tequistlatecan.chontal_oaxaca`: generated #ffd69f → generated #ffcfb0

### guahiboan

- `guahiboan`: 0.82 0.15 95 #e3c23b → 0.82 0.15 30 #ff9c89
- `guahiboan.sikuani`: 0.82 0.16 95 #e5c226 → 0.82 0.16 30 #ff9985
- `guahiboan.cuiba`: 0.66 0.15 75 #c78200 → 0.66 0.15 22 #df6768
- `guahiboan.jiw`: 0.72 0.14 118 #9db03d → 0.72 0.14 39 #ed8360
- `guahiboan.hitnu`: 0.88 0.1 80 #fad18a → 0.88 0.1 24 #ffbeb7
- `guahiboan.amorua`: 0.62 0.13 105 #928a07 → 0.62 0.13 34 #c8664f
- `guahiboan.macaguane`: generated #ffc87e → generated #ffc293
- `guahiboan.masiguare`: generated #e1901e → generated #f47a81
- `guahiboan.wipiwi`: generated #ffc950 → generated #ffacc8
- `guahiboan.yamalero`: generated #ffe362 → generated #f9944b
- `guahiboan.tsiripu`: generated #aeab19 → generated #ffcf6f
- `guahiboan.chiricoa`: generated #c4e36c → 0.84 0.12 40 #ffaf8e
- `guahiboan.mapayerri`: generated #b9fb93 → 0.88 0.1 20 #ffbdbc

### nambikwaran

- `nambikwaran`: 0.8 0.12 35 #ffa189 → 0.8 0.09 22 #f2a7a4
- `nambikwaran.nambikwara`: 0.74 0.12 35 #ec8e76 → 0.74 0.09 22 #de9491
- `nambikwaran.mamainde`: 0.86 0.12 53 #ffbb85 → 0.86 0.09 31 #ffbbad
- `nambikwaran.negarote`: 0.62 0.12 17 #c4666c → 0.62 0.09 13 #b66f77
- `nambikwaran.wasusu`: 0.86 0.12 21 #ffb1af → 0.86 0.09 15 #ffb9be
- `nambikwaran.hahaintesu`: 0.62 0.12 49 #bf6e40 → 0.62 0.09 29 #b67167
- `nambikwaran.alantesu`: 0.8 0.12 65 #f3ad66 → 0.8 0.09 37 #f1aa95
- `nambikwaran.sabane`: 0.68 0.12 5 #d5778e → 0.68 0.09 7 #c8818f
- `nambikwaran.tawande`: 0.9 0.12 35 #ffc1a8 → 0.9 0.09 22 #ffc7c3
- `nambikwaran.waikisu`: 0.58 0.12 35 #b65d47 → 0.58 0.09 22 #a96462
- `nambikwaran.halotesu`: 0.74 0.12 9 #eb8a9b → 0.74 0.09 9 #dd939f
- `nambikwaran.manduka`: 0.74 0.12 61 #e19857 → 0.74 0.09 35 #dd9684
- `nambikwaran.kithaulu`: 0.8 0.12 27 #ff9f94 → 0.8 0.09 18 #f2a6a8
- `nambikwaran.latunde`: 0.68 0.12 43 #d67e5a → 0.68 0.09 26 #ca827b

### nadahup

- `nadahup`: 0.8 0.12 25 #ff9e96 → 0.8 0.12 17 #ff9da2
- `nadahup.hupd_ah`: 0.74 0.12 25 #ed8c84 → 0.74 0.12 17 #ed8a8f
- `nadahup.yuhupdeh`: 0.86 0.12 43 #ffb791 → 0.86 0.12 28 #ffb2a5
- `nadahup.nadeb`: 0.62 0.12 7 #c2657a → 0.62 0.12 6 #c1657b
- `nadahup.daw`: 0.86 0.12 11 #ffb0bd → 0.86 0.12 9 #ffafc0
- `nadahup.puinave`: 0.8 0.12 60 #f6ab6b → 0.8 0.12 38 #ffa285
- `nadahup.nukak`: generated #ffc2a5 → generated #ffbfb0
- `nadahup.kakua`: generated #de7d89 → generated #dc7d94
- `nadahup.judpa`: generated #ffafcc → generated #ffb0d8

### muran

- `muran`: 0.62 0.1 55 #b47548 → 0.62 0.1 15 #ba6c73
- `muran.piraha`: 0.74 0.1 55 #dc9a6c → 0.74 0.1 15 #e39096
- `muran.mura`: 0.86 0.1 73 #f9c786 → 0.86 0.1 24 #ffb7b1
- `muran.warapakai`: 0.62 0.1 37 #ba705a → 0.62 0.1 6 #b96c7d

### katukinan

- `katukinan`: 0.7 0.09 85 #b89a5a → 0.7 0.09 50 #cc8e6b
- `katukinan.kanamari`: 0.74 0.09 85 #c5a767 → 0.74 0.09 50 #d99a77
- `katukinan.katukina_do_rio_bia`: 0.86 0.09 103 #dbd48e → 0.86 0.09 59 #fec396
- `katukinan.katawixi`: 0.62 0.09 67 #ab7b48 → 0.62 0.09 41 #b4735b

### unclassified

- `unclassified`: 0.74 0.05 60 #c3a48c → 0.74 0.05 30 #c9a098
- `unclassified.diaguita`: 0.8 0.07 45 #e5b099 → 0.8 0.07 24 #e8ada8
- `unclassified.comechingon`: 0.74 0.07 115 #a9b17d → 0.74 0.07 52 #cf9f82
- `unclassified.sanaviron`: 0.86 0.05 80 #e3cead → 0.86 0.05 38 #efc6ba
- `unclassified.tonokote`: 0.64 0.07 40 #b27e6c → 0.64 0.07 22 #b37b79
- `unclassified.querandi`: 0.74 0.05 60 #c3a48c → 0.74 0.05 30 #c9a098
- `unclassified.lule_vilela`: 0.8 0.05 90 #cabd9a → 0.8 0.05 42 #dbb4a5
- `unclassified.omaguaca`: 0.7 0.07 75 #b8996d → 0.7 0.07 36 #c68f80
- `unclassified.ocloya`: 0.86 0.06 55 #f1c7ac → 0.86 0.06 28 #f6c3bb
- `unclassified.tilian`: 0.62 0.06 60 #a27e62 → 0.62 0.06 30 #a77971
- `unclassified.tastil`: 0.82 0.06 95 #d0c498 → 0.82 0.06 44 #e7b8a5
- `unclassified.toara`: 0.66 0.05 30 #af8780 → 0.66 0.05 18 #af8687
- `unclassified.fiscara`: 0.76 0.05 85 #c0af8d → 0.76 0.05 40 #cea79a
- `unclassified.chicha`: 0.88 0.05 40 #f6cdbf → 0.88 0.05 22 #f7cbc9
- `unclassified.churumata`: 0.6 0.05 50 #997866 → 0.6 0.05 26 #9c7571
- `unclassified.jujuies`: 0.74 0.05 70 #c0a689 → 0.74 0.05 34 #c8a096
- `unclassified.corundi`: 0.8 0.05 65 #d5b89d → 0.8 0.05 32 #dcb3aa
- `unclassified.iogys`: 0.7 0.05 90 #aa9e7b → 0.7 0.05 42 #ba9586
- `unclassified.wayteca`: 0.84 0.05 45 #e8c1b0 → 0.84 0.05 24 #eabfbb
- `unclassified.kolla_atacameno`: 0.66 0.06 20 #b48483 → 0.66 0.06 14 #b48387
- `unclassified.tupinamba`: 0.74 0.05 60 #c3a48c → 0.74 0.05 30 #c9a098
- `unclassified.pankararu`: 0.86 0.05 78 #e3cead → 0.86 0.05 37 #f0c6ba
- `unclassified.atikum`: 0.62 0.05 42 #a17d6f → 0.62 0.05 23 #a27b78
- `unclassified.kariri_xoco`: 0.86 0.05 46 #eec8b6 → 0.86 0.05 24 #f0c5c1
- `unclassified.tapeba`: 0.62 0.05 74 #998265 → 0.62 0.05 36 #a27c71
- `unclassified.tupiniquim`: 0.8 0.05 90 #cabd9a → 0.8 0.05 42 #dbb4a5
- `unclassified.truka`: 0.68 0.05 30 #b58d86 → 0.68 0.05 18 #b58c8d
- `unclassified.maragua`: 0.9 0.05 60 #f8d7be → 0.9 0.05 30 #fed2cb
- `unclassified.kwaytikindo`: 0.58 0.05 60 #91745d → 0.58 0.05 30 #967069
- `unclassified.tapuia`: 0.74 0.05 34 #c8a096 → 0.74 0.05 20 #c99f9e
- `unclassified.acona`: 0.74 0.05 86 #b9a987 → 0.74 0.05 40 #c8a193
- `unclassified.pitaguari`: 0.8 0.05 52 #d9b5a1 → 0.8 0.05 27 #dcb2ad
- `unclassified.ypy`: 0.68 0.05 68 #ad9377 → 0.68 0.05 33 #b58e85
- `unclassified.tabajara`: 0.88 0.05 90 #e4d7b3 → 0.88 0.05 42 #f5cdbe
- `unclassified.tremembe`: 0.6 0.05 30 #9c756f → 0.6 0.05 18 #9c7575
- `unclassified.nawa`: 0.58 0.08 105 #817d42 → 0.58 0.08 48 #a16c50
- `unclassified.tuxa`: 0.66 0.08 105 #99955a → 0.66 0.08 48 #bb8367
- `unclassified.pankara`: 0.62 0.08 99 #92874c → 0.62 0.08 46 #af775d
- `unclassified.bremem_taipwenya`: 0.9 0.08 105 #e5e2a4 → 0.9 0.08 48 #ffcfb0
- `unclassified.pankarare`: 0.78 0.08 15 #e6a3a7 → 0.78 0.08 12 #e6a3aa
- `unclassified.ybutrite`: 0.7 0.08 15 #cb8a8e → 0.7 0.08 12 #cb8a91
- `unclassified.kapinawa`: 0.58 0.08 15 #a4666b → 0.58 0.08 12 #a4666e
- `unclassified.paiaku`: 0.62 0.08 15 #b17277 → 0.62 0.08 12 #b17279
- `unclassified.kiriri`: 0.58 0.08 42 #a36a54 → 0.58 0.08 23 #a56764
- `unclassified.kambiwa`: 0.66 0.08 75 #af8b59 → 0.66 0.08 36 #be8170
- `unclassified.sabuja`: 0.9 0.08 78 #fcd8a2 → 0.9 0.08 37 #ffccb9
- `unclassified.anace`: 0.82 0.08 15 #f3afb3 → 0.82 0.08 12 #f3afb6
- `unclassified.tapajos`: 0.78 0.08 105 #bfbb7e → 0.78 0.08 48 #e3a88b
- `unclassified.tupinambarana`: 0.58 0.08 78 #947441 → 0.58 0.08 37 #a46958
- `unclassified.borari`: 0.86 0.08 105 #d8d597 → 0.86 0.08 48 #fec2a4
- `unclassified.karapoto`: 0.7 0.08 105 #a6a266 → 0.7 0.08 48 #c89073
- `unclassified.wassu`: 0.86 0.08 30 #ffbeb2 → 0.86 0.08 18 #ffbcbd
- `unclassified.arapium`: 0.74 0.08 48 #d59c7f → 0.74 0.08 25 #d99791
- `unclassified.maytapu`: 0.7 0.08 51 #c79071 → 0.7 0.08 26 #cc8b84
- `unclassified.kumaruara`: 0.58 0.05 99 #817b59 → 0.58 0.05 46 #947162
- `unclassified.botocudo`: 0.74 0.08 15 #d8969b → 0.74 0.08 12 #d8969d
- `unclassified.kaxixo`: 0.74 0.08 78 #c7a570 → 0.74 0.08 37 #d89987
- `unclassified.tuxi`: 0.78 0.08 81 #d2b37c → 0.78 0.08 38 #e5a692
- `unclassified.kaninde`: 0.86 0.08 84 #eacd94 → 0.86 0.08 40 #ffc0aa
- `unclassified.kaimbe`: 0.82 0.08 63 #eab98e → 0.82 0.08 31 #f4b1a5
- `unclassified.kambiwa_pipipa`: 0.9 0.05 15 #fed1d3 → 0.9 0.05 12 #fdd1d5
- `unclassified.zenu`: 0.8 0.06 70 #d7b894 → 0.8 0.06 34 #e2b0a5
- `unclassified.pasto`: 0.68 0.06 40 #b98c7d → 0.68 0.06 22 #bb8a88
- `unclassified.pijao`: 0.86 0.05 95 #dbd1ad → 0.86 0.05 44 #eec7b7
- `unclassified.yanacona`: 0.62 0.06 60 #a27e62 → 0.62 0.06 30 #a77971
- `unclassified.mokana`: 0.74 0.05 25 #c99f9b → 0.74 0.05 16 #c99fa0
- `unclassified.quillacinga`: generated #eabebd → generated #e5becf
- `unclassified.canamomo`: generated #a9817b → generated #a6808b
- `unclassified.guane`: generated #d8b0a4 → generated #d8aeb4
- `unclassified.nutabe`: generated #e7c1af → generated #eabebf
- `unclassified.guanaco`: generated #a5856f → generated #a9817d
- `unclassified.chitarero`: generated #d2b49a → generated #d9b0a5
- `unclassified.quimbaya`: generated #dfc6a7 → generated #e8c1b1
- `unclassified.calima`: generated #9b8a6a → generated #a58470
- `unclassified.panche`: generated #c6ba97 → generated #d3b49b
- `unclassified.yari`: generated #d2cca7 → generated #e0c6a8

### zaparoan

- `zaparoan`: 0.78 0.1 85 #d5b36a → 0.78 0.1 35 #f0a08c
- `zaparoan.arabela`: generated #ead88a → generated #ffafba

### chiquitano

- `chiquitano`: 0.62 0.1 100 #948739 → 0.62 0.1 350 #b36c8f
- `chiquitano.chiquitano`: 0.74 0.1 100 #b9ac5f → 0.74 0.12 355 #e68aaf

### huarpean

- `huarpean`: 0.8 0.1 120 #b5c87d → 0.78 0.11 20 #f69a9a
- `huarpean.huarpe`: 0.8 0.1 120 #b5c87d → 0.78 0.11 20 #f69a9a

### isolate

- `isolate.trumai`: 0.86 0.17 316 #fcacff → 0.8 0.12 290 #bdb0ff

### arawakan

- `arawakan.mojeno.ignaciano`: 0.64 0.14 132 #6b9d3b → 0.64 0.14 147 #49a257
- `arawakan.wapixana`: 0.62 0.15 127 #6e961b → 0.62 0.15 145 #409d48
- `arawakan.kuripako`: 0.86 0.15 131 #aee578 → 0.86 0.15 146 #8ceb94
- `arawakan.ashaninka`: 0.68 0.15 115 #96a212 → 0.68 0.15 140 #62ae50
- `arawakan.enawene_nawe`: 0.74 0.15 119 #a1b73a → 0.74 0.15 142 #6fc267
- `arawakan.wayuu`: 0.6 0.15 115 #7e8900 → 0.6 0.15 140 #499537
- `arawakan.mawayana`: 0.9 0.15 100 #f5e05d → 0.9 0.15 134 #b4f48b
- `arawakan.warekena`: 0.78 0.15 100 #ceb92d → 0.78 0.15 134 #8ecc64
- `arawakan.cabiyari`: 0.64 0.14 130 #6f9c37 → 0.64 0.14 146 #4ba255
- `arawakan.garifuna`: generated #bcd961 → 0.8 0.14 150 #75d78d
- `arawakan.yanesha`: 0.86 0.12 120 #c7dd81 → 0.86 0.12 142 #a3e59c

### otomanguean

- `otomanguean.otopamean.otomi`: 0.74 0.16 128 #8dbd42 → 0.74 0.16 145 #61c568
- `otomanguean.popolocan.mazateco`: 0.66 0.14 105 #a09600 → 0.66 0.14 136 #69a549
- `otomanguean.popolocan.chocholteco`: generated #608208 → 0.56 0.14 145 #34893c
- `otomanguean.popolocan.ixcateco`: generated #a4ae37 → 0.8 0.12 160 #71d6a3
- `otomanguean.zapotecan`: generated #aed46b → 0.8 0.13 155 #72d699
- `otomanguean.mixtecan`: generated #00ba98 → generated #aed46b
- `otomanguean.mixtecan.mixteco`: 0.9 0.13 120 #d3eb85 → 0.9 0.13 142 #acf4a3
- `otomanguean.mixtecan.triqui`: generated #04dacb → generated #b6fb9e
- `otomanguean.mixtecan.cuicateco`: generated #009966 → generated #a4ae37
- `otomanguean.amuzgo`: 0.62 0.14 115 #848f02 → 0.62 0.14 140 #549a44

### chapacuran

- `chapacuran`: 0.68 0.14 125 #85a73b → 0.68 0.14 144 #5dae5e
- `chapacuran.itene`: generated #8bcc70 → generated #60d291
- `chapacuran.oro_nao`: 0.74 0.14 125 #97ba4f → 0.74 0.14 144 #70c170
- `chapacuran.oro_at`: 0.62 0.14 107 #918b00 → 0.62 0.14 137 #5b993e
- `chapacuran.oro_eo`: 0.86 0.14 111 #d5d965 → 0.86 0.14 138 #a2e78b
- `chapacuran.oro_win`: 0.68 0.14 95 #b49600 → 0.68 0.14 132 #77aa48
- `chapacuran.oro_mon`: 0.9 0.14 125 #c9ee84 → 0.9 0.14 144 #a3f6a2
- `chapacuran.kujubim`: 0.58 0.14 125 #678811 → 0.58 0.14 144 #3d8f40
- `chapacuran.tora`: 0.74 0.14 99 #c1ab2d → 0.74 0.14 134 #85be5f

### barbacoan

- `barbacoan.totoro`: 0.64 0.12 128 #779a45 → 0.64 0.12 143 #5d9f5a

### uruchipayan

- `uruchipayan`: 0.82 0.15 120 #b8d259 → 0.8 0.12 165 #69d6aa
- `uruchipayan.uru_chipaya`: 0.82 0.15 120 #b8d259 → 0.8 0.12 165 #69d6aa

### cahuapanan

- `cahuapanan`: 0.82 0.14 122 #b5d265 → 0.82 0.14 150 #7cdd93
- `cahuapanan.shawi`: 0.82 0.15 125 #aed561 → 0.82 0.15 152 #70e093

### panoan

- `panoan.mastanawa`: 0.9 0.12 130 #c4ee99 → 0.9 0.12 150 #a3f5b4

## 2026-10-04, French (Anita: "more different from Chinese; also confusing in Canada / the NYC area")

- `indoeuropean.romance.french` (us, cz, mu, zm): 0.76 0.12 300 #bd9ff2 → 0.60 0.15 340 #b9579c. A
  muted raspberry. The lavender sat among the Sinitic blues and violets (0.09 from Min Dong and Sze
  Yap) and the Germanic blues (0.11 from Swiss German). Now 0.21 from Chinese, 0.12 from Mandarin,
  0.28 from Cantonese, 0.22 from German, 0.26 from Dutch, 0.27 from Swiss German, 0.13 from
  Spanish, 0.28 from Italian, 0.31 from Haitian Creole, 0.32 from Arabic, further from English.
  Closest neighbours that share ground: Romansh 0.09 (Graubünden, where French is scarce), Wolof
  0.09 (Senegal, not drawn yet), Turkish 0.10, Innu 0.11 (Côte-Nord), Vietnamese 0.12, Punjabi 0.12. Kept at L 0.60 C 0.15 rather than a
  bright magenta (0.62 0.21 342 was tried first): French is the ground across Quebec and will be
  across France, and the bright one shouted over everything, as Spanish and Romanian did. Known
  leftover: 0.06 from Nyanja in Zambia, where French is a handful of dots.
- `indoeuropean.romance.cajun_french` (us): 0.64 0.15 315 #ae6dca → 0.48 0.15 345 #943171. A dark
  plum, French's hue made darker, so it still reads as kin (0.12 apart). It was 0.05 from
  Vietnamese, and both are drawn around New Orleans; now 0.22.
- Stale comments, left for whoever owns the fragment: ca.txt line 21 ("French violet"), ch.txt line
  7 ("French's lavender"), mu.txt line 7 ("French (violet)").

## 2026-10-04, colour pass (Anita's notes on US cities, Brazil, and clashes found by country agents)

**build.py**
- `UNWASHED = {"sinotibetan.sinitic"}`: Chinese with no variety named is drawn in Sinitic's full
  colour, not the washed `color_own`. The census "Chinese" (US 2.1M, Canada's n.o.s., UK, Australia)
  reads as one language, and washed it came out pale blue-grey #9babcd beside English's near-white.
  No re-scatter needed: the viewer falls back to `color` when a group has no `color_own`.

**Chinese** (us, uk, ca, cz, hk, zm fragments)
- `sinotibetan.sinitic`: 0.70 0.15 265 (#709afb, drawn washed #9babcd) → 0.62 0.20 267 #507afd. A
  clear saturated blue, now what most "Chinese" dots draw as; 0.35 from English.
- `sinotibetan.sinitic.mandarin`: 0.64 0.17 272 #6a81f2 → 0.56 0.17 298 #8258c9. Was 0.07 from the
  new Sinitic; now a violet, 0.12 from Sinitic and Vietnamese.
- `sinotibetan.sinitic.cantonese`: unchanged, 0.80 0.12 245 #78c6ff (light sky blue, 0.21 from Sinitic).
- `sinotibetan.sinitic.min_nan`: 0.76 0.13 292 #b3a1fd → 0.52 0.12 250 #286cab. Sat on French's
  lavender; now a dark navy, quiet for a small language.
- `sinotibetan.sinitic.siyi` (hk.txt, was generated): → 0.84 0.09 300 #d3befd. Generated from the
  new Sinitic it landed 0.04 from Mandarin.

**Substrate languages, quieter**
- `indoeuropean.romance.spanish` (us, cz): 0.68 0.20 35 #fa5d36 → 0.64 0.10 32 #c17565. A muted brick,
  same hue: still Spanish, no longer shouting under everything it shares a place with. Quechua
  (light peach), Aymara (lavender), the Mayan blues and Nahuatl (yellow) all stay clear of it.
- `indoeuropean.romance.portuguese` (br.txt, was generated #61dde3): → 0.62 0.07 215 #5191a1. A muted
  slate teal, so Brazil reads as a quiet ground with the indigenous languages on top; 0.17 from
  Spanish along the borders, far from English.

- `quechuan.quechua` (br.txt): 0.86 0.11 45 #ffba94 → 0.84 0.15 62 #ffb259. On the map the light
  peach still read as a paler Spanish once Spanish went brick; apricot-orange now, still in the
  Quechuan oranges. 0.07 from Paraguayan Guarani (both in Greater Buenos Aires), 0.065 from Kichwa.

**Namibia**
- `africa_other` (cf, na, zm): 0.74 0.06 70 #c4a582 → 0.58 0.06 40 #9a6e5f. Was a near twin of
  Bantu's washed "language not named" (#c3a48c), side by side in Kavango (79% and 13%). Bantu is
  left alone because every generated Bantu language steps from it; the remainder goes darker, 0.16
  from Bantu's wash and 0.10 from `other`.

**London reds** (Turkish, Albanian, Romanian, Punjabi side by side; Punjabi is South Asia, kept)
- `turkic.turkish` (us, cz): 0.68 0.16 15 #ea6878 → 0.70 0.16 330 #d476cd. Orchid, still in the
  Turkic pinks; 0.10 from Punjabi, 0.12 from Azerbaijani. The `turkic` root keeps 0.68 0.16 15, so
  generated Turkic members do not move.
- `indoeuropean.albanian` and `.albanian.albanian` (us, cz): 0.66 0.15 5 #db6686 → 0.60 0.13 70
  #b17000. Was identical to Punjabi; a dark ochre now.
- `indoeuropean.romance.romanian` (us, cz): 0.72 0.14 20 #ef7e80 → 0.78 0.09 345 #e2a0c5. A dusty
  pink: fixes London (0.12 from Punjabi, 0.10 from Gujarati, 0.11 from Turkish) and Dobruja together,
  and as Romania's majority it is kept low in chroma, like Spanish (a brighter 0.84 0.13 340 was
  tried first and was loud across the whole country).
- Known leftover: Turkish is 0.06 from Uzbek (kg.txt 0.74 0.15 345); in Kyrgyzstan that is 11
  Turkish dots beside 883 Uzbek. Every orchid/magenta that cleared it landed on Romani, Vietnamese
  or Azerbaijani instead.
- `turkic.crimean_tatar`: unchanged (0.64 0.18 15). Once Romanian and Turkish moved, the three in
  Dobruja are 0.14-0.17 apart, so the cheapest move turned out to be none.

**Bolivia / Colombia**
- `tupian.tupiguarani.guarayu` (ar.txt): 0.74 0.12 100 #bdac4a → 0.86 0.15 108 #dbd856. Was the same
  olive as Chiquitano next door; lighter Tupi-Guarani yellow now.
- `nadahup.puinave` (br.txt): 0.62 0.12 39 #c26a4d → 0.80 0.12 60 #f6ab6b. Was 0.03 from the new
  Spanish, which surrounds it in Guainía.

**Brazil, br.txt hand picks that only duplicated a sibling** (69 lines). The fragment's agent coloured
every node to dodge a generator bug since fixed, and many siblings came out identical. Walking
br.txt in its own order (largest first), the first holder of a colour kept it; each later duplicate
moved to the lightness and hue farthest in OKLab from its siblings, same chroma, hue within the
family's span ±15°. "No established classification" members could also go up to chroma 0.08 from
0.05, as at 0.05 there was no room for ~60 Northeast peoples. A member equal to its parent was left
alone: the parent is drawn washed out, so it is no clash.
- `tupian.ramarama`: 0.62 0.15 84 → 0.90 0.15 103
- `tupian.arikem`: 0.68 0.15 40 → 0.58 0.15 25
- `tupian.purubora`: 0.68 0.15 78 → 0.90 0.15 25
- `tupian.tupiguarani.kaiabi`: 0.86 0.15 88 → 0.58 0.15 115
- `tupian.tupiguarani.tembe`: 0.86 0.15 56 → 0.70 0.15 25
- `tupian.tupiguarani.arawete`: 0.60 0.15 40 → 0.90 0.15 25
- `tupian.tupiguarani.surui_do_para`: 0.74 0.15 70 → 0.74 0.15 115
- `tupian.tupiguarani.tenetehara`: 0.62 0.15 52 → 0.66 0.15 115
- `tupian.tupiguarani.asurini_do_tocantins`: 0.88 0.15 100 → 0.78 0.15 25
- `tupian.tupiguarani.kamayura`: 0.86 0.15 88 → 0.58 0.15 97
- `tupian.tupiguarani.amanaye`: 0.68 0.15 40 → 0.70 0.15 58
- `tupian.tupiguarani.apiaka`: 0.90 0.15 70 → 0.78 0.15 82
- `tupian.tupiguarani.anambe`: 0.74 0.15 96 → 0.62 0.15 106
- `tupian.tupiguarani.ava_canoeiro`: 0.80 0.15 62 → 0.58 0.15 25
- `tupian.tupiguarani.xeta`: 0.60 0.15 40 → 0.90 0.15 115
- `arawakan.kinikinau`: 0.74 0.15 145 → 0.66 0.15 190
- `arawakan.yawalapiti`: 0.86 0.15 163 → 0.58 0.15 190
- `arawakan.mawayana`: 0.62 0.15 127 → 0.90 0.15 100
- `arawakan.warekena`: 0.86 0.15 131 → 0.78 0.15 100
- `isolate.ruaynyn_reue`: 0.90 0.17 330 → 0.66 0.17 15
- `unclassified.nawa`: 0.74 0.05 60 → 0.58 0.08 105
- `unclassified.tuxa`: 0.86 0.05 78 → 0.66 0.08 105
- `unclassified.pankara`: 0.62 0.05 42 → 0.62 0.08 99
- `unclassified.bremem_taipwenya`: 0.86 0.05 46 → 0.90 0.08 105
- `unclassified.pankarare`: 0.62 0.05 74 → 0.78 0.08 15
- `unclassified.ybutrite`: 0.80 0.05 90 → 0.70 0.08 15
- `unclassified.kapinawa`: 0.68 0.05 30 → 0.58 0.08 15
- `unclassified.paiaku`: 0.90 0.05 60 → 0.62 0.08 15
- `unclassified.kiriri`: 0.58 0.05 60 → 0.58 0.08 42
- `unclassified.kambiwa`: 0.74 0.05 34 → 0.66 0.08 75
- `unclassified.sabuja`: 0.74 0.05 86 → 0.90 0.08 78
- `unclassified.anace`: 0.80 0.05 52 → 0.82 0.08 15
- `unclassified.tapajos`: 0.68 0.05 68 → 0.78 0.08 105
- `unclassified.tupinambarana`: 0.88 0.05 90 → 0.58 0.08 78
- `unclassified.borari`: 0.60 0.05 30 → 0.86 0.08 105
- `unclassified.karapoto`: 0.74 0.05 60 → 0.70 0.08 105
- `unclassified.wassu`: 0.86 0.05 78 → 0.86 0.08 30
- `unclassified.arapium`: 0.62 0.05 42 → 0.74 0.08 48
- `unclassified.maytapu`: 0.86 0.05 46 → 0.70 0.08 51
- `unclassified.kumaruara`: 0.62 0.05 74 → 0.58 0.05 99
- `unclassified.botocudo`: 0.80 0.05 90 → 0.74 0.08 15
- `unclassified.kaxixo`: 0.68 0.05 30 → 0.74 0.08 78
- `unclassified.tuxi`: 0.90 0.05 60 → 0.78 0.08 81
- `unclassified.kaninde`: 0.58 0.05 60 → 0.86 0.08 84
- `unclassified.kaimbe`: 0.74 0.05 34 → 0.82 0.08 63
- `unclassified.kambiwa_pipipa`: 0.74 0.05 86 → 0.90 0.05 15
- `panoan.mastanawa`: 0.74 0.12 175 → 0.90 0.12 130
- `macroje.krenak`: 0.62 0.17 29 → 0.58 0.17 334
- `macroje.brobo`: 0.80 0.17 7 → 0.86 0.17 334
- `macroje.je.xikrin`: 0.86 0.17 33 → 0.70 0.17 60
- `macroje.je.xokleng`: 0.58 0.17 15 → 0.74 0.17 15
- `macroje.je.xacriaba`: 0.86 0.17 1 → 0.90 0.17 60
- `macroje.je.mentuktire`: 0.68 0.17 345 → 0.78 0.17 330
- `tucanoan.mirititapuia`: 0.88 0.11 230 → 0.58 0.11 245
- `cariban.wayana`: 0.74 0.15 110 → 0.66 0.15 155
- `cariban.patamona`: 0.86 0.15 128 → 0.82 0.15 65
- `cariban.nahukua`: 0.62 0.15 92 → 0.58 0.15 152
- `cariban.galibi_kali_na`: 0.86 0.15 96 → 0.70 0.15 98
- `cariban.katwena`: 0.62 0.15 124 → 0.74 0.15 155
- `cariban.faruk_woto`: 0.80 0.15 140 → 0.74 0.15 65
- `cariban.enepa`: 0.68 0.15 80 → 0.58 0.15 134
- `cariban.arekuna`: 0.90 0.15 110 → 0.66 0.15 134
- `cariban.kamarakoto`: 0.58 0.15 110 → 0.90 0.15 65
- `cariban.akawaio`: 0.74 0.15 84 → 0.82 0.15 86
- `cariban.xerewyana`: 0.74 0.15 136 → 0.62 0.15 143
- `cariban.tunayana`: 0.80 0.15 102 → 0.70 0.15 146
- `cariban.matipu`: 0.68 0.15 118 → 0.82 0.15 119
- `cariban.akuriyo`: 0.88 0.15 140 → 0.82 0.15 155
- `macroje.maxakali.pataxo_ha_ha_hae`: 0.74 0.17 15 → 0.90 0.17 330

## 2026-10-05, Cote d'Ivoire share-out (ask 011, session edd42a8c-ci)

New leaves in ci.txt hand-picked where a generated colour sat under 0.05 OKLab from a neighbour at 2%+ of a shared region: `kwa.{abidji, adjoukrou, avikam}`, `kru.{aizi, dida, godie, kroumen, nyabwa, wobe}`, `mande.{gban, karandjan, koro, koyaka, mahou, mwan, toura, wan}`, `gur.{birifor, djimini, tagbana}`. One existing node: `nigercongo.kwa.nzema` (pl.txt, bare, generated) 0.72 0.12 20, beside Attie in Sud-Comoe. Remaining closest pairs 0.052-0.059.

## 2026-10-05, China's dialect groups (session edd42a8c-cnd)

19 new Sinitic leaves in cn.txt, all hand-picked (Language Atlas groups; the record is sources/cn.md §6). Checked over every pair of counties that share a border and carry different groups (OKLab distance, borders counted), and against the minority languages drawn in the same areas: Zhongyuan 0.66 0.13 65 (was 0.72 0.17 50, then 0.74 0.16 58: 0.03 from Uyghur in southern Xinjiang, 0.05 from Mongolian); Lan-Yin 0.72 0.14 0 (was 0.84 0.11 25: 0.03 from Kazakh in Ili and Altay); Shao-Jiang 0.60 0.11 290 (was 0.84 0.08 345: 0.05 from Gan and Min Zhong). Mandarin's eight groups oranges, yellows and corals (SW 0.68 0.17 18, Jilu 0.80 0.14 88, NE 0.86 0.09 70, Jiang-Huai 0.88 0.13 100, Jiao-Liao 0.66 0.15 355, Beijing 0.56 0.17 25); Jin 0.56 0.11 75; Gan 0.80 0.11 340, Xiang 0.56 0.15 345, Hui 0.70 0.14 320, Pinghua 0.72 0.13 300; Min Bei 0.72 0.13 10, Min Zhong 0.86 0.07 20, Pu-Xian 0.66 0.14 45, Leizhou 0.62 0.13 305, Hainanese 0.74 0.14 345. Existing Wu, Hakka, Cantonese, Sze Yap, Min Nan, Min Dong, Teochew unchanged. Closest remaining neighbours: Min Dong / Min Nan 0.073 (4 borders; both other countries' colours, left), Ji-Lu / Northeastern 0.086, Cantonese / Hainanese 0.092; every other pair 0.097+.

**Retired the same day** (session edd42a8c-cnm; Anita: "we probably shouldn't split all the different Mandarins"): the eight Mandarin leaves above (SW, Jilu, NE, Jiang-Huai, Jiao-Liao, Beijing, Zhongyuan, Lan-Yin) are gone from cn.txt; China draws them all on `sinitic.mandarin` 0.68 0.17 41, unchanged. Laos's `mandarin_southwestern` (la.txt, bare) stays and now takes a generated colour. Pu-Xian 0.66 0.14 45 sat 0.038 from Mandarin, so it takes the freed Jiang-Huai yellow 0.88 0.13 100 (0.089 from Sze Yap, 0.10 from Kra-Dai; no Mandarin county near Putian). Mandarin's nearest drawn neighbours in China: Tatar 0.056, Kyrgyz 0.072 (Xinjiang), Wu 0.091, Uyghur 0.093, Min Bei 0.098.

## Frisian (supervisor, 2026-10-05)

- indoeuropean.germanic.frisian: 0.80 0.10 200 -> 0.84 0.14 135 (light green), in ca.txt. The old teal sat next to Dutch's light blue on the Fryslân border (nl agent); neighbours there are Dutch (#51c7f1), Gronings (#7565bb), Westphalian (#196ea9).
