# languagedots — follow-ups

Work queued for later, not tied to one country's build. A supervisor spawns these when there is
room; whoever does one moves it to Done with a line on the outcome. (Anita's own notes stay in her
own files; this list is the agents'.)

## Queued

- **Scout: non-native speakers in Brazil and Mexico** (Anita, 2026-10-05: "put it on a queue and
  scout, no need to do it right away"). Both censuses ask language only of indigenous people, so
  everyone else snaps to Portuguese / Spanish (derived). Find whether IBGE (2022) and INEGI (2020)
  publish birthplace or nationality by municipio, and whether it would make a clean, marked proxy
  for immigrant languages (Venezuelans in Roraima, Bolivians and Paraguayans in São Paulo,
  Haitians; US-born in Mexico are mostly Spanish-speaking children of Mexican families, so likely
  a poor proxy there). Scout only: write findings into `sources/br.md` / `sources/mx.md` and here,
  build nothing; a proxy is Anita's to allow.

- **Done 2026-10-06: Isan in Thailand** (Anita, 2026-10-06: "i think we should maybe also try to
  do isan somehow"). The census's Thai is now split into Central Thai, Isan, Northern Thai and
  Southern Thai by World Values Survey shares (2018 by changwat, shrunk towards the region;
  `sources/th.md`, "The Thai varieties"). Open: Bangkok draws almost no Isan because the survey
  found almost none spoken at home there; a better source (Mahidol's Ethnolinguistic Maps by
  province, WVS microdata) would firm up the 28 unsampled changwat.

- **Proposal: a shared "neighbour pull" placement weight** (2026-10-06, session 5d7dac7e-edge,
  after Anita's note that unit borders read as sharp language edges). `countries/ph.py` now has
  one as a per-country weighter (`_PhWeighter`; `sources/ph.md` §7), and `countries/id.py` has
  the Glottolog-point half of it. The general form, for `countries/_shared.py` or `scatter.py`
  (Anita's):
  - per unit, rake a placement-polygon x language table (IPF) to the polygons' population and the
    unit's per-language counts, so every count stays the census's and every polygon's dots still
    add up to its people;
  - seed: a language under 40% of the unit gets `(1 - λ) + λ·exp(-d/K)`, d the distance to the
    nearest polygon of a neighbouring unit where it holds 40% or more, applied only when that
    homeland is within ~25 km of the unit (adjacent or across a strait); else to its Glottolog
    points, within ~150 km; everything else an even seed, so the unit's main language fills what
    the minorities leave. ph uses K = 12 km, λ = 0.998;
  - cost: one rake per unit at scatter time (Philippines, 117 units and 42k barangays: 7 s).
  Where it helps: any country whose units are big and whose languages change inside them, e.g.
  Nigeria's states, Kenya's counties, India's districts at state borders, Indonesia's provinces
  (its rake already exists), Peru and Bolivia's Quechua/Aymara edges. It needs no data beyond the
  counts and the placement layer, so a country could opt in with `place_weight=neighbour_pull`.
  What it cannot do: tell which side of a unit a minority lives on when the homeland is not next
  door (Sipalay's Cebuano in Negros Occidental's far south-west stays thin), or split a unit's
  count; it only moves people inside the unit they were counted in.
  **Anita, 2026-10-06: Philippines only, and weakened.** "the softening is kinda sus tbh ...
  maybe slightly weaken it ... especially for small languages in super mountainous areas we
  dont wanna dilute much." So it is not to become a shared weight: China uses atlas polygons
  (Crissman) instead, and no other country gets it. In ph it now pulls only the twelve regional
  languages of 1M+, only across a land line (15 km, not 25), with a seed floor of 0.2 (was
  0.002), a cap (the share of the language's speakers in the 15 km band nearest its homeland is
  at most twice the share of the unit's people living there), a fade between 300 and 700 m
  elevation so lowland languages are not pulled into the Cordillera, the Mindanao highlands, or
  Mindoro's and Palawan's interiors, and no pull at all in units led by a smaller language
  (Benguet, Mountain Province, Ifugao...). Smaller languages use their Glottolog points only. Numbers in `sources/ph.md` §7 (session
  5d7dac7e-php).

- **The grey "other languages" wedge** (Anita, 2026-10-06: "in some cases the gray 'other
  languages' segment is really quite big, like Qatar and Munich and London. is there anything
  we can do to show more?"; session 5d7dac7e-oth).
  - **Mostly the viewer, not the data.** A pie keeps its 7 largest languages and folds the rest
    into one grey wedge (`index.html`, PIE_K = 8, `pieWedges`). Share of the pie that fold
    takes, from the dots: Qatar 43% (72 languages, almost all named: Sinhala, Tamil, Pashto,
    Tagalog...), Munich 17% (46), London 13% (62), Newham 16%. Data-side grey (dots on `other`
    or an unnamed group) in the same places: Qatar 1.9%, Munich 2.9% (now 0.7% `other`),
    London 0.3% (+1.2% Arabic and Chinese, named but on group nodes, washed not grey).
  - **Options for Anita (index.html is hers):** (a) fold the tail by family instead of into one
    grey: keep 5 languages and give the rest up to 3 family wedges in the family's washed colour
    ("9 other Indo-Aryan languages"), the leftover grey. Simulated grey with families at the
    root: Qatar 15%, Munich 2%, London 4%, Newham 4% (at the branch level, Indo-Aryan /
    Dravidian / Semitic: 20 / 7 / 9 / 12%). (b) raise PIE_K: 12 wedges gives 28 / 11 / 9 / 12%,
    16 gives 20 / 7 / 7 / 9%; costs shader stride and legibility. (c) both.
  - **Done for the data side:** Germany's three continental Mikrozensus remainders (Europe,
    Asia, Africa, 0.98M) split by Zensus 2022 citizenship (`sources/de_rest.py`, `sources/de.md`).
    Qatar's `other` is DESA's unnamed `Others` (32,601) plus 1% tails of home mixes, nothing
    finer exists; London's `other` is ONS's "any other X language" residuals, TS024 is the
    finest table. Both left as they are (`sources/qa.md`, `sources/uk.md`).
  - **Done 2026-10-06: Scotland's "Other language"** (272,820): split by 2022 country of birth
    by ward x `origin_mix`, fitted to a national estimate (2011 AT_002 labels, England and
    Wales 2021 main-language ratios; predicts 259,920 before scaling), placed in OAs by birth
    region (`sources/uk_scot_other.py`, `sources/uk.md` §9).
  - **Biggest data-side grey shares, all countries** (`counts.json`: dots on `other`, its
    children, or a group node, as a share of the country; 2026-10-06, before the de change):
    1. pf 29.5%: census "Langue polynésienne" on Oceanic (no Polynesian node; see Done below).
    2. gu 28.2%: ACS "Philippine languages" on the Philippine group (PCT25 does not split it).
    3. my 27.3%: Chinese on Sinitic (census names no variety); Indians' 20% and "Others" on `other`.
    4. af 25.1%: Kabul city, Herat, Kandahar/Helmand on Iranian, Dari and Pashto not split.
    5. na 16.5%: Kavango and Caprivi languages on Bantu, San on Khoisan (census groups).
    6. sb 14.8%: the census's remainder (90,859), English and Pijin inside it, on `other`.
    7. tc 14.3%: the census's "Other" (probably mostly Jamaicans), on `other`.
    8. pw 13.3%: "Philippine languages" on the Philippine group (as Guam).
    9. mt 13.2%: "Other" 57,818 on `other`, mostly non-Maltese residents; Arabic on its group.
    10. vi 12.7%: "French, Haitian, or Cajun" on French-based creoles; "Other" on `other`.
    11. mm 12.2%: Karen, Chin, Kachin, Chinese as national races on their group nodes.
    12. cf 11.1%: "Autres" (54% in Bangui, probably mostly French) on `other`; Arabic group.
    13. ad 11.0%: "Other European" and "Other" on `other` (survey, one unit).
    14. dz 10.1%: Berber on its group (the 1966 census and surveys name no variety).
    15. mo 7.0%: "Other Chinese dialects" on Sinitic; "Others" on `other`.
    Then gq 6.7% (foreigners, no nationality), gw 6.5% ("Sem dialecto"), cm 5.2% (Bamileke
    group), nc, il, fj, mz, tr (Arabic group 3.8%), es. **Possibly fixable**: mt, cf and tc by
    citizenship or birthplace x `origin_mix`, if the census publishes one (not checked); my by DOSM's Chinese
    dialect table if found. pf, gu, pw, na, dz, mm, af are the census's own grain: a hard limit
    unless a finer source turns up. Small states (tc, ad, vg, ky, mc) are a few dots each.

## 2026-10-06: named labels on group nodes, audit (5d7dac7e-aud)

`python tools/audit_groups.py --min 50000` lists the people drawn on group (non-leaf) nodes per
country, with the mapping labels that resolve there and each label's people from
`data/normalized/`. It reads `counts.json`, so countries waiting for the build tail show their
previous scatter (id's Dani row below is already gone).

**Main finding: plain Arabic is a group node by accident.** `afroasiatic.arabic` got two children,
`afroasiatic.arabic.shuwa` (td/ne/ng/cm, repeated in 16 fragments) and `afroasiatic.arabic.nubi`
(ug), while every other variety sits beside it (`afroasiatic.egyptian_arabic`, `levantine_arabic`,
`saudi_arabic`...) and dj2024's docstring calls it "the generic `arabic` leaf". So every census
"Arabic" draws washed out as unnamed: 10.85M people in 68 countries (tr 3.39M, ir 1.60M, de 1.45M,
us 1.42M, es 0.83M, ca 0.55M, au 0.37M, se 0.32M, uk 0.21M, sa 0.16M...). **Fixed 2026-10-06 (5d7dac7e-tree) another way:** `afroasiatic.arabic` is now a group holding
every Arabic variety, and plain Arabic draws on the leaf `afroasiatic.arabic.arabic` "Arabic (variety
not given)", through `taxonomy/regroup.txt` (no ids renamed where written; see
`taxonomy/GROUPING.md`, which also covers the Bantu zones, Austronesian branches and dialect groups
done at the same time). The rename below was not needed. Old plan, kept for the record:
rename `afroasiatic.arabic.shuwa` -> `afroasiatic.shuwa_arabic` and `afroasiatic.arabic.nubi` ->
`afroasiatic.nubi` in `taxonomy/tree.d/*.txt` (ar be br co cr dk fr ga it ne ng nl no pa pt se td
ug), `taxonomy/{cm2022,ne2001,ng2022,td2009,ug2024}.py`, `data/normalized/{be,be_communes,dk,fr,ga,
it,nl,no,pt,se}.csv`, `data/geo/nl/nl_weights.csv`, `sources/td.md`, `taxonomy/COLOURS.md`; then
`taxonomy/build.py` and re-scatter the 16 countries that draw Shuwa or Nubi (ar be br cm dk fr ga it
ne ng nl no pt se td ug). The 68 countries drawing plain Arabic need no re-scatter (their node id
is unchanged; only languages.json changes). Shuwa and Nubi keep their colours (td.txt, ug.txt).

**Fixed:** cn's Jingpo (160k) from the Sino-Tibetan root to mm.txt's Kachin group, the narrowest
node holding Jingpho, Zaiwa, Lashi and Lhaovo (`sources/cn.md`). Re-scattered.

Top 40 by people (counts.json, 2026-10-06). Verdicts: **remainder** = an unnamed "other", correct;
**group** = the source's label covers several languages, correct, listed; **tree** = the Arabic
accident above; **place?** = a group label a place split could resolve (spec §3, Pahari rule).

| # | cc | node | people | share | label(s) | verdict |
|---|----|------|-------:|------:|----------|---------|
| 1 | in | Indo-Aryan | 16.71M | 1.4% | Others under HINDI | remainder |
| 2 | cn | Loloish | 8.98M | 0.6% | Yi nationality | group |
| 3 | cn | Hmongic | 8.14M | 0.6% | Miao nationality | group |
| 4 | af | Iranian | 7.85M | 24.8% | Balochi and Dari one figure (158k); Dari/Pashto not split in Kabul, Herat, Kandahar | group (af agent's) |
| 5 | my | Chinese | 6.89M | 23.2% | Chinese (ethnic group) | group; DOSM dialect table would fix |
| 6 | cn | Tibetic | 5.43M | 0.4% | Tibetan nationality | group; place? (U-Tsang/Amdo/Kham by prefecture) |
| 7 | dz | Berber | 3.45M | 10.1% | Amazigh | group; place? (Kabylie, Aures, M'zab) |
| 8 | pk | Other | 3.41M | 1.4% | OTHERS | remainder; 2026-10-09 KP's part named by MICS 2019 (Khowar 422k, Gujari 365k, Torwali 70k, Gawri 91k, the last three partly named by place), now 2.45M (`sources/pk.md` §0.2) |
| 9 | tr | Arabic | 3.39M | 3.8% | Arapca, incl. Syrians under protection | tree |
| 10 | mm | Karen | 3.20M | 6.6% | Karen (national race) | group |
| 11 | cn | Hmong-Mien | 2.85M | 0.2% | Yao nationality (Bunu split out) | group |
| 12 | us | Chinese | 2.12M | 0.7% | Chinese (ACS, Mandarin and Cantonese together) | group |
| 13 | in | Other | 1.88M | 0.2% | Others under OTHERS | remainder |
| 14 | ir | Arabic | 1.60M | 2.0% | Arabic | tree |
| 15 | de | Arabic | 1.45M | 1.7% | Arabisch | tree |
| 16 | cm | Bamileke | 1.43M | 4.9% | Bamileke (language not named) | remainder |
| 17 | us | Arabic | 1.42M | 0.4% | Arabic | tree |
| 18 | ir | Other | 1.36M | 1.7% | Other | remainder |
| 19 | cn | Kra-Dai | 1.26M | 0.1% | Dai nationality | group; place? (Xishuangbanna Tai Lue, Dehong Tai Nua) |
| 20 | es | Other | 1.13M | 2.5% | Otra | remainder |
| 21 | mm | Kuki-Chin | 1.00M | 2.1% | Chin (national race) | group |
| 22 | mz | Bantu | 0.95M | 4.4% | Outras linguas mocambicanas | remainder |
| 23 | cn | Other | 0.86M | 0.1% | undetermined nationality, naturalised | remainder |
| 24 | es | Arabic | 0.83M | 1.8% | Arabe | tree |
| 25 | za | Other | 0.83M | 1.6% | Other | remainder |
| 26 | id | Other | 0.76M | 0.4% | Bahasa asing (foreign) | remainder |
| 27 | mm | Kachin | 0.74M | 1.5% | Kachin (national race) | group |
| 28 | ci | Other | 0.73M | 3.4% | Aucune langue nationale parlee | remainder |
| 29 | mm | Other | 0.62M | 1.3% | Other, Foreign | remainder |
| 30 | my | Other | 0.62M | 2.1% | Others | remainder |
| 31 | ca | Arabic | 0.55M | 1.5% | Arabic | tree |
| 32 | my | Austronesian | 0.50M | 1.7% | Other Sabah / Sarawak Bumiputera | remainder |
| 33 | de | Other | 0.39M | 0.5% | another Asian / European language | remainder |
| 34 | id | Dani | 0.38M | 0.2% | (stale: id's Papua languages replaced the cluster) | already gone |
| 35 | uk | Other | 0.37M | 0.6% | Scotland "Other language", ONS residuals | remainder (queued above) |
| 36 | au | Arabic | 0.37M | 1.5% | Arabic | tree |
| 37 | il | Other | 0.34M | 4.1% | AnotherLanguage | remainder |
| 38 | se | Arabic | 0.32M | 3.1% | Arabic | tree |
| 39 | ph | Manobo | 0.32M | 0.3% | Manobo (ethnicity) | group (ph not touched) |
| 40 | in | Eastern | 0.32M | 0.0% | Others under BENGALI, ODIA | remainder |

Below 40 but worth a line: **group**, large shares: pf 28.4% "Langue polynesienne" (place? by
archipelago), gu 25.2% and pw 12.3% "Philippine languages", na 14.9% Kavango and Caprivi
languages, vn Hoa 315k, th Karen 297k, kh/th Chinese, tr "Turki Diller" 241k and "Balkan" 198k, my
Proto-Malay 94k, ng "Gwoza" 164k (an LGA, several Chadic languages). za "Sign language" 231k on
`signlanguage`: South Africa's only sign language is SASL, so a leaf would be defensible, but the
census does not name it; left. vi "French, Haitian, or Cajun" (7k, 8.9%) sits on French-based
creoles though French is in it; small, left. **remainder**: sb 14.7%, mt 11.5%, cf 9.1%, gq 6.7%,
la, nz, iq, tw, et, gh's unnamed Guan, cm's Grassfields, id's trade Malay, India's other "Others
under X". Inherited: ae's Indo-Aryan and sa/dj's Arabic come from origin mixes.

## 2026-10-06: Archive size (5d7dac7e-dep)

Measured only; tiles.py unchanged. The archive is 880 MB, 884,866 tiles, z0-12, four layers
(`dotsm1`, `dots`, `dots1`, `dots2`, one per Aggregation step: high, medium, low, very low),
every mark carrying n, c, p, t, z; point coordinates on a 4096 extent; each tile gzip level 6.
Per zoom (stored MB): z8 93, z9 116, z10 134, z11 155, z12 197; z0-7 together 186.
religiondots is smaller mainly because its archive stops at z10 and its unmerged dots come
from data/buffers/ instead.

The four layers share most of their marks: a cell holding one language gives the same mark at
every merge grid. In sampled tiles 54% of marks at z9 and 72% at z12 are exact duplicates
(same position, n, c, p, t) of a mark in another layer. Also, layer `dots2` at zoom z uses the
same grid as `dots1` at z+1, `dots` at z+2 and `dotsm1` at z+3, so most grids are stored up to
four times over.

Savings scaled from trial archives of np, bd and fr (24.7 MB built by a scratch copy of
tiles.py, same code, one change each), which matched the full-archive sample within a point:

| option | saves | what is lost |
|---|---|---|
| (e) one shared layer, each mark stored once with a bitmask `l` of the steps that draw it | ~300 MB (34%) | nothing on screen; the viewer filters on `l` instead of switching source-layer (its decoder and rebuildPies) |
| (a) drop `dots2` | ~230 MB (26%) | the "very low" Aggregation step |
| (b) max zoom 11, overzoom past it | ~200 MB (23%) | at z12+ every step is one grid coarser than now and the finest grid goes; ARCHIVE_MAXZ to 11 |
| (d) brotli 11 instead of gzip 6 | ~180 MB (20%) | pmtiles.js only inflates gzip; needs a JS brotli decoder passed as the PMTiles decompress function, and a slower build |
| (a) drop `dotsm1` | ~135 MB (15%) | the "high" step |
| (c) extent 1024 instead of 4096 | ~40 MB (5%) | positions snap to half a pixel at the tile's own zoom; invisible |
| (c) drop `z` (tile zoom) | ~23 MB (3%) | rebuildPies and the dot layer must learn the tile zoom some other way |
| (d) gzip 9 | ~14 MB (2%) | nothing |
| dropping `t` | ~13% in samples | not an option: the per-cell ink cap needs it |

They do not add up. Trial combinations: (e) + extent 1024 + no `z` + max zoom 11 comes to
48% of today, about 420 MB, with gzip; adding brotli takes it to 41%, about 360 MB. The bigger
step after that is storing each grid once (one layer per zoom, the viewer asking for zoom z+k
tiles for step k through its own pmtiles protocol handler): per-layer sizes say about a third
of today's bytes, but it is real viewer work.

Scripts: tools/archive_size_measure.py `<copy of the archive> 0.01` (per layer and zoom, plus
re-encode variants on a 1% sample; ~40 min; point it at a COPY, an open archive blocks the
build's rename) and tools/archive_size_trial.py `np,bd,fr <variant>` (variants: base, z11, noz,
ext1024, two, no_m1, no_d2, gz9, br, shared, lean, maxgz, max; ~30 s each, brotli ~3 min).
Both only read data/.

**Done 2026-10-06 (5d7dac7e-shrink), Anita "let's try shrinking archive":** (e) + extent 1024 +
gzip 9 in tiles.py; one layer `marks` with `l` (1 dotsm1, 2 dots, 4 dots1, 8 dots2), still
carrying z. 881 MB -> 572 MB. The viewer filters on `l` (decodeMarks, rebuildPies, STEP_BIT);
the Aggregation slider no longer re-adds the circle layer. Checked: every tile and step decodes
to the same marks as the old layers (trial np/bd/fr/uk all tiles; full archive 1 in 40, only uk
and no differed, whose dots were rewritten between the builds); with draw order made
content-based in a scratch copy, old vs new screenshots were pixel-identical at extent 1024.
What does change: ties in draw order between equal-size overlapping marks (a new shuffle) and
sub-pixel positions. Not done: brotli, dropping a step, max zoom 11, dropping `z`.

## 2026-10-06: languages that stop at a border (5d7dac7e-xb)

Anita: "in Africa many languages stop sharply at national borders" (Shona, Chewa, Kongo, Ewe,
Akan). `python tools/audit_crossborder.py` (its docstring says how) compares the dots within
~100 km of each frontier, links each node to Glottolog by its label, and lists (a) different nodes
on the two sides that Glottolog puts in one language or a close subgroup, (b) a language big on
one side and under 0.5% on the other. People in the table are near-border dots; totals are the
whole group. Africa first, then the world.

**What was done.** No mapping was wrong: each census label sat on the node its words say. The
break was that neighbouring censuses name one language or cluster differently (Mozambique's
Ndau, Manyika and Tewe against Zimbabwe's Shona) and the tree kept the names apart. Every fix is
a group in `taxonomy/regroup.txt` (section "Cross-border groups"; GROUPING.md), Anita's "like we do
for Arabic" (2026-10-06): each member keeps its own leaf and colour, a plain census name gets its
own leaf ("Kongo (variety not given)"), nothing named sits on a group. The dots, rings and
counts.json were rewritten in place (old drawn id -> new, 97 ids, ~219k dots; no re-scatter, no
count changed); `taxonomy/build.py` changed no existing id's colour; `tools/audit_groups.py --min
1000` shows nobody on the new groups; `check_country` ok for every country touched. **Colours are
untouched**, so a group reads as one entry in the folded legend and the tooltip, but members can
still differ sharply in colour across a border (Kinyarwanda olive against Kirundi salmon).
Pulling member colours towards one hue is the next step if the map still shows a hard edge.

| case | border, near-border people | verdict | what changed |
|---|---|---|---|
| Shona / Ndau, Manyika, Tewe | zw-mz: Shona 2.98M vs Ndau 338k, Tewe 178k, Manyika 121k | grouped | `= shona`: Shona, Ndau, Manyika, Tewe (Glottolog's Core Shona), 12.9M. Kalanga and Nambya (Western Shona) left beside it |
| Chewa / Nyanja | mw-mz: Chewa 8.37M vs Nyanja 930k; zm prints both | already grouped | One language (Glottolog nyan1308; Chewa a dialect). Mozambique prints "Cinyanja", Malawi and Zambia print both names, so each label keeps its leaf in the "Nyanja" group the earlier regroup made. Nothing changed |
| Kongo / DRC varieties | cd-cg 2.26M, cd-ao 745k | grouped | `= kongo`: "Kongo (variety not given)" (ao, cg) plus cd's nine varieties (Yombe, Ndibu, Manyanga, Ntandu, Mbata, Lemfu, Besi Ngombe, Kongo of the south-east bank, Mboma), Angola's Fiote, Congo's Laari and Suundi, 14.4M. Beembe, Vili (H.11, H.12) and Kituba (a creole) left. cd's old `kongo_dialects` node is now empty |
| Gbe: Ewe / Aja, Fon, Gen, Waci... | tg-bj: Ewe 2.6M vs Fon 1.86M, Aja 840k; gh-tg Ewe vs Gen 444k | grouped (cluster) | `= gbe`: 17 Gbe languages (Ewe, Gen, Waci, Aja, Fon, Gun, Saxwe, Ayizo, Kotafon, Maxi, Tofin, Xwela, Defi, Toli, Weme, Ci, Agouna), us.txt's "Gbe (Ewe, Fon)" on a "Gbe (language not given)" leaf, 13.0M. Separate languages: a group, not a merge. Benin's small Ewe count is right; its Gbe speakers name other Gbe languages |
| Akan / Abron | gh-ci: Akan 2.94M vs Abron 132k | grouped | Abron (Bono) into the Akan group, 269k; Ghana already files Bono under Akan (gh2021.py) |
| Akan / Baoulé, Anyi | gh-ci | left | Baoulé and Anyi are Central Tano languages of their own (Glottolog: both Bia, beside Akanic), not Akan varieties. Côte d'Ivoire's census is right |
| Kwanyama / Oshiwambo | ao-na: 808k vs 751k | grouped | `= oshiwambo`: Namibia's cluster label on its own leaf, Angola's Kwanyama and Ndonga beside it (Glottolog's Ndonga (R.20)), 2.0M |
| Lomwe | mw-mz: 2.21M vs 515k | grouped | `= elomwe` "Lomwe": Malawi Lomwe and Mozambique Lomwe (two Glottolog languages, one name), 3.8M |
| Sena | mw-mz: 470k vs 409k | grouped | `= sena`: Malawi Sena with Mozambique's Cisena, 2.0M |
| Nyakyusa / Nkhonde | tz-mw: 1.07M vs 125k | grouped | `= nyakyusa`: Nkhonde is Glottolog's Ngonde dialect of Nyakyusa-Ngonde, 1.6M |
| Manding: Bambara / Maninka / Dyula / Mandinka | ml-gn 1.12M, ml-bf, ci-bf, gn-sl | grouped (cluster) | `= manding` (ISO's Mandingo macrolanguage, Glottolog's Manding): Bambara, Maninka, Mandinka, Dyula, Jahanka, Khassonké, Konyanka, Marka, Mahou, Koyaka, Wojenaka, Worodougou, Koro; us.txt's "Manding" on its own leaf, 21.9M. Kuranko (Mokole) and ci's small varieties Glottolog does not place (Djamala, Gandjé, Komara...) left under Mande |
| Rwanda-Rundi | rw-bi: 12.2M vs 9.7M; bi-tz Kirundi vs Ha 2.26M | grouped (cluster) | `+ rwanda_rundi`: Kinyarwanda (with Rufumbira), Kirundi, Ha, Hangaza (Glottolog's West Highlands Kivu), 32.5M |
| Konzo / Nande | cd-ug: Nande 4.87M vs Konzo 976k | grouped (cluster) | `+ konzo_nande` (Glottolog's Rwenzori), 6.3M |
| Kalenjin / Pokot, Kupsabiny | ke-ug: 1.56M vs 320k, 173k | grouped | `= kalenjin` (ISO macrolanguage): Kenya's "Kalenjin" on "Kalenjin (variety not given)", Pokot, Sabaot, Kupsabiny, 6.5M |
| Ateker: Teso / Turkana / Karamojong / Toposa | ke-ug 590k, ss-ug 184k | grouped (cluster) | `+ ateker` (Glottolog's Teso-Turkana), with Nyangatom, 5.9M |
| Somali / Benaadir, Maay | so-et, so-ke | grouped | `= somali`: Benaadir (a Glottolog dialect) and Maay (its own language in Glottolog, commonly called a Somali variety), 27.9M |
| Tuareg | ml-ne-bf-dz-ly | grouped (cluster) | `+ tuareg`: Tamasheq, Tamajaq, Tamahaq, 2.7M |
| Lugbara / Aringa | cd-ug: 846k vs 666k | grouped | `= lugbara`: Aringa ("Low Lugbara", Glottolog's Lugbaric), 2.9M |
| Yoruba / Ede (Nago, Idaasha, Ifè...) | ng-bj | grouped (cluster) | `+ edekiri` "Yoruba and Ede": Yoruba, Ede Nago, Ede Idaca, Ede Ije, Ede Cabe, Ifè, Manigri-Kambolé, 41.4M |
| Uab Meto / Baikenu | id-tl: 555k vs 69k | grouped | `= uab_meto`: Baikenu is Glottolog's Oecusse dialect, 0.7M. Same colour already |
| Lao / Isan | la-th: Lao 3.07M vs Isan | grouped | Isan (18.9M) into the Lao group (Glottolog's Northeastern Thai, a sister of Lao in Lao-Thai). The call most open to reversal: Thailand counts Isan as Thai |
| Kirundi, Kinyarwanda, Ha colours | rw-bi-tz | left (colour) | grouped above; their colours stay far apart |
| Shi, Havu, Hunde vs Kinyarwanda | cd-rw | left | separate languages (Glottolog's Kivu) |
| Swati | sz/za-mz: 894k vs 0 | data gap | Mozambique 2017 prints no Swati; its speakers are inside "Outras línguas moçambicanas" (on Bantu, 947k). Not invented |
| Kanuri / Kanembu | ng-td: 785k vs 202k | left | Kanembu is its own Glottolog language; whether Chad's census counts its Kanuri under Kanembu was not checked. 2026-10-09: RGPH2 Annexe 3 p208 lists "Kanembou/Kanouro/Bornou" in the Kanembou row but also Kanouri in "Autres"; MICS 2019 has Kanembou only, and Kanembou-group heads answering "other" (Lac, N'Djaména) are drawn unnamed |
| Luo / Adhola, Acholi, Alur | ke-ug, cd-ug | left | separate Southern Lwoo languages |
| Gusii / Kuria, Mòoré / Farefare, Maba / Masalit, Tigre / Tigrinya, Saho / Afar, Yom / Nawdm, Lukpa / Kabiyè, Chokwe / Luvale, Kaonde / Sanga, Moba / Bimoba, Konkomba / Gangam | various | left | separate languages in Glottolog and in common use |
| Zarma / Songhay, Dendi; Tumbuka / Senga; Lunda / Ndembu; Kinyarwanda / Rufumbira; Arabic varieties; Swati / Zulu | various | already grouped | the earlier regroup or an existing group |
| Kongo vs Yaka, Suku, Pelende, Punu | cd-ao-cg | left | the scan's Kongo alias is Glottolog's broad Kikongo cluster, which holds these; they are separate H.30 and B.40 languages |
| Ngoni (mw, zm) vs Tumbuka, Chewa | mw-zm-tz | left | "Ngoni" is a dialect name in both Glottolog languages; Malawi's Ngoni label is the people, who speak Chewa or Tumbuka; Tanzania's Ngoni is its own language |
| Thai / Tai Dam, Bhojpuri / Tharu, Bambara / San (Samo) | | false matches | name clashes in the scan |
| Cantonese / Taishanese | cn-hk-mo | left | both Yue, already inside the Chinese group |

People moved between nodes: none. The groups gather about 218M people; the largest are
Yoruba-Ede 41.4M, Rwanda-Rundi 32.5M, Somali 27.9M, Manding 21.9M, Isan 18.9M. Outside Africa the
scan finds almost only real language borders (German, Dutch, Polish, Czech...).

## 2026-10-07: speed and size catalog (session da1b1b09, measured, nothing changed)

Scripts in that session's scratchpad `perf/`. Biggest first:
- **Tiles fetched twice at desktop world zoom**: repeated world copies refetch every z1 tile
  (3.4 of 6.9 MB). `renderWorldCopies: false`, or a min zoom, or a byte cache in the protocol.
- **Shuffle -> position order inside tiles** (tiles.py `bucket()`): big low-zoom tiles ~50% smaller,
  archive ~15%; equal-size ties then break on a position hash in sortMarks/sortPies.
- **Low-zoom tiles carry all four Aggregation steps**: the default uses 27-29% of z0's 191k marks.
  One archive per step (world view 3.4 -> 0.9 MB; storage x1.5) or one grid per zoom (archive
  x0.45-0.6). z0 alone is 2.2 MB, 2/2/1 is 2.75 MB; 53 tiles over 500 kB.
- **Pies rebuild 330-450 ms per rebuild, ~1 s per jump** (rebuildPies via querySourceFeatures, then
  pieWedges for every pie; the `sourcedata` handler re-runs it as tiles land). Read raw tile bytes as
  decodeMarks does, cache per tile, rebuild only changed tiles: ~3-5x less.
- Smaller: the invisible `dots` circle layer still lays out every mark (filter to nothing); a 9k-node
  composition bar on every buildPanel; `gl.readPixels` every frame while something is selected;
  DOT_CACHE ~100 MB (cap lower on phones); mousemove not throttled to a frame.
- Side files: counts.json drop `note_public` (unused, 256 -> 195 kB gz) and index node ids
  (-> 127); languages.json drop `parent`; country_shapes simplify (603 -> 257 kB, recheck Auto);
  brotli instead of gzip on all (~35%); `cache: 'no-store'` -> `'no-cache'` and set Cache-Control on
  upload (R2 sends none); r2.dev is not CDN-cached, a custom domain is.
- Brotli inside tiles ~29% more (needs a JS decoder); a column tile format ~50% (large).

**Done 2026-10-07 (da1b1b09):** world copies off while the whole world fits on screen (on again
when zoomed in, so the date line still pans); pies read the dot layer's decoded-marks cache instead
of querySourceFeatures (India z5 ~540 -> 142 ms; same cells, people and parts in four views checked);
tiles.py stores marks in position order (np/fr/jp trial 14.3 MB vs ~16), ties between equal dots
and pies broken by markHash (position and language) so no direction is favoured. The rest above is
still open.

## 2026-10-07: weakest places, queued (session da1b1b09 review; Anita: "note all the weakest
places as things we can follow up in the future")

Weighted by people. **Tried 2026-10-07, all drawn and in the archive; cd only partly closed (no
open province table; four cities shifted)** (fix agents, Anita's pick of the worst
plus the placement ones): ci foreign residents (6.46M not drawn), ss Jonglei/Unity/Upper Nile
(5M hatched), bd Sylheti/Chittagonian (~24M drawn as Bengali) and the Rohingya camps (~1M, in no
census), cd Swahili/Lingala (ethnicity read as language, far under MICS; closed 2026-10-09 by
MICS-Palu 2017-18 microdata, `sources/cd.md` §0), ru placement by
tochno.st settlements, pk at tehsil grain, om Al Mazyunah (~492k in a near-empty wilayat), sn/ml
placement by CLEAR Global département/cercle shares. Outcomes go in runlog.md and sources/<cc>.md.

**Queued, not started:**
- [ ] ng (217M): language from Afrobarometer, ~280 respondents per state, state grain, ~119 of
      ~500 languages named. Hard limit unless a finer survey turns up.
      **Partly closed 2026-10-09 by MICS6 2021 microdata** (`sources/ng.md` §0): the nine
      languages MICS names (Hausa, Yoruba, Igbo, Fulfulde, Kanuri, Tiv, Ibibio, Ijaw, Edo; 74%
      of people) now rest on ~1,000 households a state; Benue Tiv 36 -> 63%, Yobe Kanuri 9 ->
      22%, Fulfulde 3.7 -> 7.1% nationally (an upper reading, HH16 gives 5.4%). Still open:
      MICS's "other language" (25%) is split by the same Afrobarometer, still state grain, and
      3.8% is now drawn unnamed where the Afrobarometer met too few speakers (Ekiti, Oyo).
- [ ] cn: nationality -> language with one retention share; Han on the county's main dialect
      (Hakka 27M vs the usual 40-50M); 15 provinces on 2000 mixes; Guangxi off its gazetteer
      (Hakka, Pinghua, Yue); Hainan Putonghua 10% vs WVS 52%; Luhe filed as Yue. Tibetan (Ü-Tsang /
      Amdo / Kham by prefecture) and Dai (Xishuangbanna / Dehong) could be split by place.
      Viewers, 2026-10-09: Xiang, Gan and Wu still look too strong in the big cities (Changsha,
      Nanchang, the Wu cities). Anita: a lot of time has gone into this already, but it is still
      doubtful. What the map rests on (`sources/cn.md` §10): CLDS 2016 settled locals' "main
      language after work" is only 7.6% Putonghua in Hunan's non-Mandarin cities, 3.4% in
      Jiangxi's, 5.1% in Jiangsu's Wu cities, and the share is one per city across every age,
      though CLDS's own age split is far steeper (Zhejiang 32 / 19 / 5% at 15-30 / 31-45 /
      46-64). WVS 2018 disagrees in both directions (Jiangxi 26% against CLDS's 3.3; Hunan 1.8%).
      Open: no source settles it. Leads are an age-weighted share, or a city-core vs county split.
- [ ] id: 2010, 8 languages measured, the rest from ethnicity; Javanese, Sundanese, Malay, Banjar
      seeded evenly across provinces (Madurese of the Tapal Kuda missing). Regency homelands for
      the even-seeded languages would help.
- [ ] id, "too Sundanese" in places (viewers' complaints, Anita 2026-10-08: "the totals are
      reasonable but its probably just like allocation / shifting"). Likely the even seed above:
      West Java's and Banten's Sundanese spread by population alone, so it also lands where other
      languages hold (Bekasi, Depok and the Jakarta fringe: Betawi/Indonesian; Cirebon and
      Indramayu: Cirebonese/Javanese; northern Banten, Serang: Banten Javanese). Placement only:
      keep province counts, move Sundanese towards its regencies (a regency-level ethnicity or
      language source, e.g. the 2010 census by kabupaten, or Glottolog points as id.py does for
      smaller languages). Ask which spots people named, if she has them.
      Viewers, 2026-10-09: a sharp Betawi / Indonesian edge at the DKI Jakarta border. Same cause:
      DKI is measured 89% Indonesian (7.93M), West Java 19% (7.28M) and Banten 39%, and Indonesian
      is seeded evenly, so Bekasi, Depok and Tangerang draw their province's rural mix. Betawi is
      seeded towards its Glottolog point, so West Java's 1.42M and Banten's 0.44M modelled Betawi
      crowd the fringe, against DKI's own 0.29M. **Found:** the SP2020 long form (2022 fieldwork)
      has, open and keyless, "uses a regional language in the family" Ya / Tidak by all 514
      regencies: `https://sensus.bps.go.id/topik/tabular/sp2022/201/<area>/<fmt>` (area 1 the
      nation, 2-35 the provinces in code order, DKI 12, Jawa Barat 13, Banten 17; fmt 3 JSON).
      Also 198 (first language: Indonesian / regional / foreign / sign) and 204 (with neighbours).
      Tidak (Indonesian or foreign) runs Kota Bekasi 95%, Depok 92%, Tangerang Selatan 95%, Kota
      Tangerang 93%, Kab. Bekasi 81%, Kab. Tangerang 54%, Kab. Bogor 42%, Karawang 20%, against
      DKI 95-97% and 1-4% in Garut, Tasikmalaya, Cianjur. Names no regional language below the
      nation. **Done 2026-10-09** (Anita: "lets do that for indonesia"): placement only, 2010
      counts kept, Indonesian seeded by regency Tidak share; `sources/id.md` §11 has the before
      and after (Kota Bekasi Indonesian 13 -> 63%, Garut 15 -> 0%). BPS's *Profil Suku* (2024)
      also says 98.69% of ethnic Betawi use Indonesian or a foreign language in the family, so
      the 2.24M Betawi speakers of 2010's L4.1 may be mostly older people or a coding artefact.
- [ ] iq, Iraqi Turkmen too few (viewer, 2026-10-09): drawn 1.6% (728k), Kirkuk 11%, Salah
      al-Din and Diyala 0, Baghdad 0.03%. Arab Barometer's 2020-22 ethnicity question is lower
      still (28 of 3,476, 0.8%; Kirkuk 7-13%), so every survey runs low the same way. Turkmen
      dots are also spread over the whole governorate (Tal Afar's land in Mosul). The open MICS6
      2018 report (washdata.org/report/iraq-mics-2018-sfr, 592 pp.) tabulates no language,
      ethnicity or religion anywhere, though HC1B asks the head's mother tongue (Arabic, Kurdish,
      Turkman, Assyrian, Other) and HH16 / WM14 / FS14 the respondent's (Sorani and Badini apart).
      Only the microdata has it (UNICEF registration). **Done 2026-10-09:** Anita's UNICEF
      account; Iraq now drawn from MICS6 HC1B, Kurdish split by HH16 (`sources/iq_mics6.py`,
      `sources/iq.md` §0): Kirkuk 38 Arabic / 30 Kurdish / 31 Turkmen, Nineveh Turkmen 11 -> 4%,
      Salah al-Din 0 -> 3.3%, national Turkmen 2.0%; Sorani and Badini split. Open: placement
      inside governorates; Baghdad's Turkmen (MICS 0 of 2,153 households, but 180 clusters can
      miss an enclave of 1% one time in six; Arab Barometer ethnicity 2 of 767).
- MICS rebuilds, Anita's answers 2026-10-09 ("use your own discretion for what aligns best with
  language at home ... bias toward splitting"): kept as built: la RETENTION on (children's home
  language), cd Nord-Kivu 71% Swahili and Kinshasa on HC1B (CLEAR Global's North Kivu map, from
  CAID, also puts Swahili at roughly 80-100% spoken by territory), ng Fulfulde on HC1B and MICS
  Hausa replacing the ask 018 mother-tongue step, af Brahui 44,592 (not taken from measured
  Pashto), tg foreign languages on `africa_other`. Changed: tg French back to Afrobarometer's
  home-language share; pk Swat/Dir/Shangla Indo-Aryan split by place; td Sara regrouped over
  Ngambay and Sar; Bantu colour pass so Swahili reads in DR Congo; Kurdish Sorani back to the
  old orange.
- [ ] MICS microdata, now that Anita has a UNICEF account (2026-10-09): countries whose notes
      name MICS as the blocker or best improvement: cd (2010 / 2017-18 HC1B by province; **done
      2026-10-09**, 2017-18 HC1B drawn for the national languages, `sources/cd.md` §0; open:
      Nord-Kivu reads 71% Swahili, so Nande falls 5.0M -> 1.5M, worth a second source), pk
      Gilgit-Baltistan (2016-17 / 2024-25 HC1B by district; **done 2026-10-09**, 2016-17
      weighted microdata seeds GB, which reproduces the Pamir Times table exactly, and KP MICS6
      2019 splits KP's census OTHERS, `sources/pk.md` §0; open: "other" in GB has no finer
      item, and MICS's one Kohistani/Gujari code is named by place outside Hazara, from
      knowledge: Behrain's Torwali/Gawri split evenly, since Joshua Project has no Pakistan row for
      trw or gwc; a published speaker estimate would replace it), la (done 2026-10-09 with LSIS III
      2023: HH16 follows the interview language, so retention comes from FL7, children's home
      language, `sources/la.md` §0; open: children overstate the shift for adults), sd (2014), af (done 2026-10-09 with MICS6 2022-23, `sources/af.md` §0), ne,
      tg (done 2026-10-09 with MICS6 2017: HC1B's eleven groups by region, split by the
      Afrobarometer pool, `sources/tg.md` §0; open: MICS's 4.7% foreign languages unnamed,
      still six units); gy (done 2026-10-09 with MICS6 2019-20: HC1B retention per region for
      the census Amerindians, five interviewers who recorded no indigenous heads dropped,
      `sources/gy.md` §0; open: languages still unnamed, North Rupununi possibly a team effect,
      MICS5 2014 not tried); td done 2026-10-09
      with MICS6 2019 (HC1B by région, "other" split by the head's ethnic group, `sources/td.md`
      §0; open: nomads and refugee camps probably thin in the frame). dz's
      MICS6 has no language item; tn's codes only French / Arabic / other.
- [ ] ir, tr: small surveys by region, minorities spread evenly, Kurdish varieties not split.
      ir: Mazandaran 0% Persian; Golestan's ~0.95M `other` probably Turkmen; WVS microdata form.
- [ ] No measured source at all: sd (ad hoc 80% non-Arabic cap; war displacement not shown;
      Eritreans/Ethiopians drawn Arabic), eg, sy, pg. (af closed 2026-10-09: MICS6 2022-23 replaced
      the c.2006 village majorities.)
- [ ] Zoomed right in, every dot is 1,000 people at one point (viewers noticed; Anita 2026-10-08:
      "not sure if we have a good reason to keep it like that, maybe we could more evenly
      distribute. it does kinda look good as is though"). Each dot sits at one random spot in its
      placement polygon (a Kontur hex, ~0.74 km², or a settlement), so a village of 1,000 is a
      single blob and the space around it looks empty. Options: (a) in the viewer, past ~z11,
      split each dot into 10 dots of 100 scattered within a hex-sized radius (~500 m) around it,
      no data change, positions no less invented than today's single point; (b) scatter at
      1 dot = 100 people for the top zooms only (an archive several times bigger at z11-12, a
      re-scatter of everything); (c) leave it. (a) is the cheap try; judge it on screen.
- [ ] Old censuses draw fewer people than today (et 56%, td 54% since 2026-10-09 (was 40%, the
      6+ universe), bf 53%, mz 62%, ne 63%):
      a density step at borders. Rescaling to today's population is against spec §1 (Anita's call).
- [ ] Coverage: af 1.5M Kuchis (Takhar/Kunduz drawn from MICS since 2026-10-09); my non-citizens 2.69M (23.7% of
      Sabah); sg non-residents 1.64M; mm Wa/Mongla 433k; ml 941k not enumerated; ly 827k
      non-Libyans; Tindouf camps 174k; no entry for Isle of Man, Channel Islands, American Samoa,
      Northern Marianas, Cook Islands, Falklands (~0.4M).
- [ ] Lumped: mm Karen/Chin/Kachin on groups; my Chinese 6.9M not by dialect (DOSM table?); dz
      Berber 3.45M (place split possible: Kabylie, Aurès, M'zab); tj Pamiri as Tajik; pg Tok
      Pisin not drawn; th Bangkok ~1% Isan. (af's Dari/Pashto split in Kabul, Herat, Kandahar
      closed 2026-10-09 by MICS6; Hazaragi still drawn as Dari, MICS does not name it.)
- [ ] Placement, coarse: ir, tr, et zones, iq governorates, td régions (measured by MICS since
      2026-10-09; was national), mz provinces, ne, tz/cm/zw survey
      districts, ke counties, uz (Tajik in Samarkand/Bukhara evened), mm Kontur cap blocks, India
      layer across the LoC near Poonch, az Karabakh resettlers counted twice.
- [ ] Unanswered shares likely not random: ru 16.6M no native language, ro 13%, hu 12%, bg 10%.
- [ ] Logins only Anita can use: za 2022 by municipality (DataFirst), ss World Bank phone survey.
- [ ] Record conflict: the Shenzhen item under "Not ours" vs sources/cn.md §9, which says county
      totals were rescaled to 2020 on 2026-10-06.

## Not ours, noted for religiondots

- **Angola religion totals** (found by the ao agent, 2026-10-05). In Cacuso, Ngola Luiji, Luena and
  Lucusse, religiondots' religion-table totals differ from the RGPH 2024 provincial volumes, whose
  language and ethnicity tables agree with each other. Anita has not decided what to do; this
  session does not touch religiondots. Details in `sources/ao.md`.
- **Latvia placement populations swapped** (found by the lv agent, 2026-10-05). GISCO's LAU 2021
  workbook swaps Jelgava city (0090000) and Jēkabpils city (0110000) populations; religiondots'
  `lv_lau.gpkg` `pop` column and `_lv_place_weight` use it, so Zemgale's dots lean to Jēkabpils.
  Counts unaffected. Details in `sources/lv.md`.
- **Senegal 2023 regional reports** (found by the sn agent, 2026-10-05). ANSD published 14 RGPH-5
  regional reports in April 2026 (https://www.ansd.sn/rapports/rgph-5-2023), with tables by région;
  religiondots' Senegal still uses 1988 data. Details in `sources/sn.md`.
- **Oman Kontur false desert blocks** (found by fix-place, 2026-10-07). religiondots' `om_hexes.gpkg`
  places Thumrayt's people 29% on the Fasad oil field (one hex 44,302/km2) and Hayma's 14% on one
  empty hex; counts per wilaya are unaffected. languagedots lowers hexes above 8,000/km2 away from
  a wilaya seat (`sources/om_place.py`); the same rule would suit religiondots. Details in `sources/om.md`.

## Done

- **Ethiopia unasked-children check** (2026-10-05, supervisor). The CAR trap (USCB putting the
  unasked under-3s into one language column) is not in `et`: no language holds a floor share
  across the 93 zones (every language's minimum zone share is ~0%; CAR's Fulfulde never fell
  below 9%), and the total, 73,750,932, is the full 2007 population, so the question was asked of
  every age.
- **DONE 2026-10-05. Pinned tooltip to legend** (Anita, 2026-10-05). Click a pie or dot to freeze its tooltip; click a language in the frozen tooltip to scroll to it in the legend, opening its parent groups; a click anywhere else unfreezes it.
- **Polynesian group node** (2026-10-05, pf agent). Tahitian, Maori, Samoan etc. sit flat under `austronesian.oceanic` (fi.txt, au.txt, cl.txt), so French Polynesia's 'Langue polynésienne' (28%) draws washed out on Oceanic. A `polynesian` group would let it sit one level closer; it means re-parenting leaves in other countries' fragments.
- **religiondots bug, Cyprus (2026-10-05, found by languagedots cy agent; religiondots not edited).** GISCO's south community polygons run 170 km² past the ceasefire line, so 147 of religiondots' south hexes sit in the north (25k Kontur people, 19k of them north Nicosia inside Lefkosia 1000) and religiondots places south dots there. languagedots' own cut is `data/geo/cy/cy_hexes.gpkg` (OSM line).
- **Kontur cap rows can't be overridden from languagedots** (2026-10-05, mm agent). Eight Myanmar blocks (Ayeyarwady delta, rural Yangon; up to 60% of a township, Mawlamyinegyun) are `unreviewed` in religiondots' registry; a `capped` row in languagedots/kontur_cap.csv makes the scatter stop on 'rows that disagree'. Needs rdlink.py (Anita's) to let a languagedots row win, or the religiondots rows promoted. Hit again by `tr` (a block near İzmir it wanted `real`), so it recurs.
- **Myanmar Rohingya** (2026-10-05, mm agent). GAD's religion table counts 1.2M more people than its ethnicity table, 584k in Rakhine (Islam there 588k). Undrawn; per-township gap in data/normalized/mm_unrecorded.csv. Drawing them as Rohingya would be a religion-to-language proxy, a kind the 2026-10-05 rulings don't cover.
- **Belgium Moroccan Berber** (2026-10-05, nl agent). be used Morocco's own census mix (Tarifit 3%); NIDI Demos 2023 gives Dutch Moroccans Arabic 51 / Berber 43, and Belgian Moroccans are likewise largely Riffian. A one-line change in be's Morocco mapping plus a re-scatter. **Done 2026-10-05 (edd42a8c-imm)**: be's Moroccans are 40% Tarifit (Reniers 1999), an override in `sources/origin_mix.py`.
- **Immigrant origin mapping is inconsistent across countries** (2026-10-05, supervisor). India is Hindi in fr/es, Punjabi in it/pt; Mozambique is Emakhuwa in fr, Portuguese in pt; Morocco's Berber share varies (fr 22%, es 20%, be 3%, nl 43%). sa's 'home mix' (`sources/sa_census.py`: each nationality takes its home country's own languagedots counts, India by emigrant state) would make them consistent; a sweep over fr/es/it/be/nl/pt/gr/se would apply it. **Done 2026-10-05 (edd42a8c-imm)**: all of fr es it be nl pt gr se kr jp now use `sources/origin_mix.py`; overrides and before/after in `sources/origin_mix.md`. Re-scattered; the build tail has not run.
- **Turkish and Romani share a colour** (2026-10-05, gr agent): both L 0.70, hue 330, overlapping in Rodopi and Xanthi. Both are defined in other countries' fragments and drawn across Europe, so left for Anita's colour pass.
- **South Africans abroad drawn as Zulu** (2026-10-05, fix agent): the home mix gives South Africa's whole map; in the Gulf (and anywhere without an override) they should be mostly English/Afrikaans. An uncited override in origin_mix (English 60 / Afrikaans 30 / rest home mix, or similar) under Anita's ruling.
- **Concise pass on the Data text** (Anita, 2026-10-06): the per-country Source/What/Grain text is too long (France's pushes the legend off screen). Later, an agent sweep to shorten `how`/`grain`/`note_public` for every country, keeping sources named. The panel now caps the section's height meanwhile.
- **India's placement layer crosses the LoC near Poonch and Rajouri** (2026-10-06, from the pk GB/AJK agent): 105 of Pakistan's AJK dots fall inside India's polygons there. An `in` geography fix.
- **nl: are the Belgium-born drawn as Dutch, or on Belgium's own Dutch/French shares?** (2026-10-06, sweep batch D) The old note_public said the latter, sources/nl.md the former; the note line was cut, so check origin_mix's NATIVE/override handling for BE->nl and fix whichever record is wrong.
- **si.csv: Bosnian and Serbian both sum to exactly 31,294 at municipality level** (national rows 31,499 and 31,329). Looks like a copied column. (2026-10-06, s-z sweep)
- **cn: county totals in chinaethnicity's 15 estimated provinces use a 2000 pattern** (2026-10-06, cnmig agent): Shenzhen draws 10.5M against the 2020 census 17.6M; the same for any fast-growing city there. Fix belongs in chinaethnicity's fallback.py (Dong and Wang's 2020 county panel in helper1m has the right totals).
- **cn: Luhe (Shanwei) filed as Yue in the county dialect table; it is Hakka-speaking** (2026-10-06, mcp agent). Fix in sources/cn_dialect.py.
- **om: religiondots' om_hexes put ~492k people in Al Mazyunah wilayat** (register 11,117): a false Kontur block or keying error, in religiondots' layer (2026-10-06, Gulf agent). **Done 2026-10-07 (fix-place)**: Kontur's own blocks; dots always followed the register per wilaya (Mazyunah ~11 dots), only placement inside Thumrayt, Hayma and Mazyunah was off. Own layer `sources/om_place.py`; `sources/om.md`.
- **bs: the answer "GHANA" is mapped to Twi** (2026-10-08, cross-border survey): a guess at a nationality answer; Akan is likelier than Twi alone but still a guess. `taxonomy/bs2010.py`.
- **Maring and Sam still read the same as another node** (2026-10-08): one twin in tree.txt, the other in pg.txt, which sources/pg_build.py regenerates; relabel at the source.
