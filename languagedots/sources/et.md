# Ethiopia: 2007 census, mother tongue by zone

Drawn 2026-10-04 (session d9e44929-et). 73,750,932 people, 93 zones, 89 nodes from 91 census
categories. 73,702 dots at 1:1000, 5 rings.

## Source

- **Table.** CSA, 2007 Population and Housing Census, Table 3.2 "Population by Urban-Rural
  Residence, Sex, and Mother Tongue", from the eleven regional reports. Read as transcribed by
  the U.S. Census Bureau: "Ethiopia Subnational Population and Housing Data Tables with
  Administrative Boundaries" (HDX, CC BY, release `uscb_202308`), sheet `Language`. Same file
  religiondots uses for religion; fetched separately into `data/raw/et/`.
- **Question.** Mother tongue, "the language used during childhood when speaking with family
  members" (USCB's metadata, quoting CSA). One answer per person.
- **Vintage.** 2007 is the latest census. The 2017 round was postponed and never held.
- **Grain.** Zone (ADM2): 93 zones with data, 793,000 people on average (35,000 to 2.95M).
  The queue said woreda, but that was the religion table. USCB's Language and Ethnicity sheets
  stop at ADM2 because CSA published mother tongue by zone in the regional reports. I found no
  woreda-level table and did not search the CSA PDFs further: USCB transcribed those reports
  and went to woreda only where the report did.
- **Geography.** The 2021 units in USCB's file, with the 2007 figures recut onto them by USCB
  (Sidama split from SNNPR; Segen Peoples zone formed in 2011). Finfinne Zuria special zone is
  a row with no data and no polygon.

## Checks (`python sources/et_uscb.py`)

All pass:

1. 91 language columns. Each has exactly one CSA field name in the Data Dictionary sheet, and
   the names are distinct.
2. No negative cells. The religion sheet's -999 sentinel does not appear in this one.
3. The categories sum to 73,750,932 nationally, the census population.
4. Each of the 93 zones sums to its own Age-Sex total. 0 differ.
5. Zones, and separately regions, sum to the national figure on all 91 categories.
6. **The independent check:** each zone's total equals the sum of its woredas in the Religion
   sheet (CSA Table 3.4, a different table). 0 differ. The same check proves the id nesting
   that `countries/et.py` uses to key woreda hexes to zones: every woreda `ETH_aa_bb_cc`
   lies in zone `ETH_aa_bb`, and its ADM2_NAME agrees.

There is no "not stated" cell, so nothing goes in `gap` for non-response. The `gap` is the
four woredas with no published figure (three in Afar, Beltu in Oromia). They are missing from
the national total too.

## Labels

**USCB renamed every column to an ISO 639-3 name, and several renamings are wrong.** Its
Data Dictionary keeps CSA's original field name for every column, so `source_category` is
CSA's spelling and `taxonomy/et2007.py` maps from that. USCB's name is kept in the
`uscb_name` column for reference. Wrong or misleading USCB names:

| CSA field | USCB name | Drawn as | Why |
|---|---|---|---|
| Shitagna (880,818) | "Shetagna" | Silt'e | 83% in Silti zone, 4% next door in Gurage; Silt'e has no other line |
| Mossigna (9,552) | Mossi (Burkina Faso) | Mosiye | a name for Bussa (Ethnologue); 74% in Segen zone |
| UPO (1,751) | Ignaciano (Bolivia) | Opo | 54% in Etang woreda, Gambela, where Opo (Opuo) is spoken |
| Shegna (492) | She (China) | unidentified | all in Addis Ababa; not the She variety of Bench, which is in Sheka |
| Debosgna (70,419) | "Debo" | Debase (Gawwada) | 88% in Segen zone; Segen's other languages all have their own lines |
| Felashigna (946) | Kemant | Felashigna, on Agaw | "language of the Falasha"; scattered, not around Gondar |
| Koregna / Koyrigna | Koregna / Koorete | Koore / Koyra | Koregna is 98% in Segen (Amaro), so it is Koore |

Calls that rest on where a label's speakers live rather than on a reference:

- **Silt'e** (Shitagna), **Opo** (UPO), **Gawwada** (Debosgna), **Koore** (Koregna): see the table.
- **Banna** (Benagna, 96% South Omo), **Donga** and **Timbaro** (94% and 96% in Kembata
  Tembaro, both Kambaata varieties), **Demegna** (89% South Omo, where Dime is spoken; kept apart
  from the census's separate, scattered Dimegna).
- **Kusume** (Kusumegna, 96% Segen): not found in Glottolog or Ethnologue. It is placed under
  Lowland East Cushitic because every Segen language except Koore is in that group.
- **Six labels are not identified with any language** and each sits on its own node under
  `other`: Merigna (8,159; 47% in Sidama zone), Brayligna (3,401, spread over Amhara's zones,
  most likely Braille: blindness from trachoma is commonest in Amhara), Guagugna (3,282,
  Addis Ababa and around), Wergigna (2,037), Gebatogna (1,421), Shegna (492). Searched
  Glottolog's names for each; nothing matched.

Labels that name one language twice are kept as two nodes, because the census asked them
separately: Gidole and Dirasha (Glottolog: one language); Mosiye and Mashile (both Bussa); Bacha
and Koygo (both Kwegu); Majang and Mesengo (Mejengerigna is zero everywhere in 2007); Hamer and
Banna (Glottolog: Hamer-Banna). "Guragiegna" is the census's single Gurage, covering Sebat Bet,
Soddo and Mesqan alike, so it is one node, Gurage.

`Other Ethiopian Language` (119,659; 53% in Afder zone, Somali region) and `Other Foreign
Language` (20,102; a third in Asosa) span several families and sit on `other`.

## Tree

New nodes are in `taxonomy/tree.d/et.txt`: Ethiopian Semitic, Cushitic (Lowland East, Highland
East, Agaw), Omotic (Ometo, Gonga, Dizoid, South Omotic, plus Bench, Yem and Mao on their own),
and a **new root, Nilo-Saharan** (Nilotic, Surmic, Koman, Berta, Gumuz, Kunama). These are the
conventional groupings for the region. Glottolog agrees on each language's branch but splits
Omotic out of Afroasiatic and Nilo-Saharan into six families. That is ask 001 (a new
top-level family is hers to approve). By family: Cushitic 38.4M, Ethiopian Semitic 28.4M,
Omotic 5.9M, Nilo-Saharan 0.88M, `other` 0.16M, English 1,867.

**Colours.** Every language has a hand-picked `L C h` in the fragment, not only the big ones.
`taxonomy/build.py`'s generator currently gives every uncoloured sibling the same colour: in
`colour()`, `free` is rebuilt from `lch` inside the loop, so `free.index(nid)` is always 0. Before
I coloured them by hand, all twelve smaller Lowland East Cushitic languages came out #85eba3.
That is build.py's bug to fix, and it probably hits other countries too. Within Ethiopia I set
neighbours apart on the ground: Oromo mid green, Somali light yellow-green, Afar dark teal,
Amharic gold, Tigrinya orange-red, Sidama cyan, Wolaytta red-pink, Gamo light pink, Gofa violet,
Dawro pale lavender. Zones that hold several languages (Segen, South Omo, Kembata Tembaro, Gamo
Gofa) get spread lightness within the group.

## Placement

religiondots' Kontur hexes for 738 woredas (`religiondots/data/geo/et/et_hexes.gpkg`, read-only).
Each woreda's id is keyed to its zone by `place_unit`, so a zone's dots spread over its
woredas' hexes by Kontur population. Kontur is 2023 data against a 2007 census, so it is used
only as a weight inside each zone. **Consequence:** within a zone, every language is spread the
same way. Towns are not told apart from the countryside around them, so the Amharic-speaking
towns of Oromia's zones are diluted across those zones. `note_public` says so.

Six Kontur cap blocks print as UNREVIEWED: religiondots registered them (Mekele outskirts,
four in Somali, one in Afar). They are drawn as Kontur has them. Nothing new was registered.

## Corroboration

Not sought beyond the census's own tables. The 1994 census (mother tongue) and IPUMS
microdata (2007, MTONGET) exist. IPUMS is the account that is blocked.
