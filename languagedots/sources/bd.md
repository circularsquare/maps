# Bangladesh: 2011 census ethnic groups, drawn as languages

Built 2026-10-05, session `edd42a8c-bd`. Scripts: `sources/bd_census.py` (table),
`sources/bd_camps.py` (camp outlines, placement only), `taxonomy/bd2011.py` (crosswalk),
`taxonomy/tree.d/bd.txt` (nodes), `countries/bd.py`. `data/` is gitignored; this is the record.

**This is a proxy.** Bangladesh asks no language question. Anita allowed drawing ethnic groups
as languages on 2026-10-05 ("yes bangladesh mapping fine", relayed by the languagedots
supervisor; the model was Indonesia's ask 009). Every row is `tier=derived`, and `how` and
`note_public` say so.

## 1. The search for a language table (none usable)

| source | what it has | verdict |
|---|---|---|
| 2011 census (BBS; USCB tabulation on HDX) | religion, 27 ethnic groups + other, by upazila | no language item |
| 2022 census, national volume and HDX district dataset (`populationa-and-housing-census-dataset`) | total ethnic population by district only | no language item; `bbs.portal.gov.bd`, which hosted the national PDF, now says "Domain is not available" |
| 2022 census district reports | ethnic population by union, ~50 groups | no language item seen (Wikipedia upazila articles quote the ethnicity tables only) |
| **SEDS 2023** (Socio-Economic and Demographic Survey, BBS, released 2025-06-04) | **mother tongue, Table 3.6, by division, two columns: Bangla / Others** | the only language measurement; too coarse to build from (below) |
| IPUMS Bangladesh samples | no LANG variable (coverage sweep) | none |

SEDS 2023: Bangla 99.17%, others 0.83% nationally; Chattogram division 2.89% others, Khulna
0.00%. These figures are from secondary reports of Table 3.6 (a GitHub PR on another project and
search summaries); the report itself is on `nsds.bbs.gov.bd`, which refused connections from
here (ECONNREFUSED 203.112.218.101), so they are not opened at source. Not built from because it
names no language but Bangla, at eight divisions, from a sample survey: drawing it would put one
unnamed 0.83% remainder on the map and lose the Hill Tracts entirely. It is used as a check
(§4). **Call someone might reverse**: a stricter reading of "build from any language table" would
draw SEDS (Bangla vs unnamed) and use ethnicity only to place the "Others" inside divisions.

## 2. The table

U.S. Census Bureau, `bangladesh_uscb_202107.xlsx`, HDX dataset
`bangladesh-subnational-boundaries-and-tabular-data` (CC BY), sheet `Religion and Ethnicity`:
BBS's 2011 census at 544 upazilas and thanas, with total population, total ethnic population,
27 named groups and `Other ethnicity`. Religiondots draws its religion columns from the same
sheet (`../religiondots/sources/bd.md`), and its Kontur hex layer is keyed on the same
`GEO_MATCH` ids, so the join is an identity (544 = 544).

Why 2011 and not 2022: the 2022 census names ~50 groups by union (~4,500 units), which would be
better, but only in 64 district-report PDFs on `203.112.218.101`, which timed out. The Wayback
Machine holds about 40 of the 64 (CDX query on
`203.112.218.101/storage/files/1/Publications/PHC_2021/*`); Rangamati and Bandarban, the two
districts that matter most, are not among them. A union build would also need union boundaries of
the 2022 vintage. That is the upgrade path; 2011 is complete and reconciles exactly.

Checks in `bd_census.py`, all pass: the 28 group columns sum to `ETH_ETOTL` on every row; the 544
upazilas sum to the national row column by column and to 144,043,696; no nulls or negatives.
Ethnic minorities: 1,586,183 (1.10%). The remainder, 142,457,513, is written as
`Not an ethnic minority`.

## 3. The crosswalk (`taxonomy/bd2011.py`)

| census group | people | drawn as |
|---|---|---|
| Not an ethnic minority | 142,457,513 | Bengali |
| Chakma | 444,755 | Chakma |
| Marma | 203,009 | Marma |
| Other ethnicity | 202,818 | `bangladesh_other` |
| Sawntal | 147,112 | Santali |
| Tripura | 133,798 | Kokborok |
| Garo | 84,565 | Garo |
| Orao | 80,386 | Kurukh |
| Barmon | 53,792 | `bangladesh_other` |
| Tanchaynga | 44,254 | Tanchangya (new) |
| Mro | 39,004 | Mru (new, under new group Mruic) |
| Monda | 38,212 | Mundari |
| Monipuri | 24,695 | Meitei |
| Coach | 16,903 | Koch |
| Rakhain | 13,254 | Rakhine (new) |
| Bawm | 12,424 | Bawm (new) |
| Khasia | 11,697 | Khasi |
| Hajong | 9,162 | Hajong |
| Pahari | 5,908 | Malto (Sauria Paharia) |
| Khiyang | 3,899 | Khyang (new) |
| Khumi | 3,369 | Khumi (new) |
| Cool | 2,843 | Kol (new) |
| Malpahari | 2,840 | Mal Paharia (new) |
| Chak | 2,835 | Chak (new, under new group Luish) |
| Pangkhua | 2,274 | Pangkhua (new) |
| Lusai | 959 | Mizo |
| Dalu | 806 | `bangladesh_other` |
| Uchai | 347 | Usoi (new) |
| Mong | 263 | `bangladesh_other` |

Calls:
- **`bangladesh_other`** is a new areal root ("Other languages of Bangladesh's ethnic
  minorities"), as `indonesia_other` is. "Other ethnicity" is concentrated in Chunarughat,
  Madhabpur and Bahubal (Habiganj tea gardens: communities speaking Sadri, Odia, Telugu and others)
  and Naogaon (Mahali, Pahan and others), so no family node holds it.
- **Barmon, Dalu, Mong** have no established language of their own (Barman are described as
  Bengali-dialect or Sadri speakers depending on the source; Dalu speech as Bengali-Hajong; "Mong",
  263 people scattered over Naogaon, Dhaka, Bandarban and Sunamganj, is unidentified), so they
  sit on the remainder rather than being guessed into Bengali.
- **Monipuri as Meitei.** The group covers Meitei and Bishnupriya Manipuri speakers (mostly
  Kamalganj, 15,672). They share no node short of the root, the census does not split them, and
  the name usually means Meitei. Bishnupriya are therefore under-drawn.
- **Pahari as Malto, Malpahari as Mal Paharia.** Two separate census groups; Glottolog lists
  Sauria Paharia (Malto) in Bangladesh, and Mal Paharia (ISO mkb) as Indo-Aryan.

## 4. Known misfits, and what a better source would fix (room for improvement)

Anita asked for this to be said plainly; `note_public` says it too.

- **Sylheti and Chittagonian** (languages in Glottolog, roughly 11 and 13 million speakers) are
  counted as Bengali by the census and drawn as Bengali. This is the biggest error on the map by
  far: the whole of Sylhet division and most of Chattogram district would change colour.
- **Urdu speakers** ("Biharis", a few hundred thousand, in camps such as Geneva Camp in Mohammadpur
  and in Saidpur) are not an ethnic category, so they are drawn as Bengali.
- **Language shift.** Groups are drawn on their heritage language. Many Garo, Santal, Oraon and
  Munda households use Bengali, and many Oraon and Munda use Sadri; Koch and Hajong are largely
  Bengali-speaking. So the minority languages are overstated, the Munda and Dravidian ones most.
- **The census's own count** of ethnic groups is disputed: ethnic leaders say 1.65 million (2022)
  is far too low (claims of 3 million).

What would fix it: a census or large survey asking **mother tongue with Sylheti, Chittagonian and
Urdu as answers**, tabulated by upazila. SEDS 2023 asks mother tongue but publishes only
Bangla/Others by division; its microdata (BBS, not open) or a future census language question
would be the source. The 2022 census district reports (union-level ethnicity, 50 groups) would
fix the grain and vintage but not the misfits.

**Check against SEDS 2023.** The crosswalk gives 1.10% non-Bengali nationally (all ethnic groups,
including `bangladesh_other`); SEDS measured 0.83% non-Bangla mother tongue. The gap is the
direction language shift predicts. In Chattogram division (where the Hill Tracts' groups mostly
kept their languages) SEDS's 2.89% is close to the 2011 ethnic share of 3.16%; in Khulna SEDS
found 0.00% against 40,530 ethnic-group members in 2011 (0.26%; Munda and others there are
largely Bengali-speaking, and a sample survey can miss a group that small). Not a check at unit level, and the SEDS figures are secondhand (§1).

## 5. Geography and placement

Religiondots' `bd_hexes.gpkg` (Kontur 2023 r8, 145,658 hexes, `unit` = GEO_MATCH), read-only;
its `../religiondots/sources/bd_geo.md` has the checks. Population weight within each upazila.

**Rohingya camps (brief §7).** The camps (about a million people, mostly since 2017) are in no
census. Kontur 2023 puts ~140,000 people in the Ukhia camp area, nearly half of Ukhia's
weight, so the census's Ukhia and Teknaf residents would have been drawn mostly inside the
camps. `sources/bd_camps.py` fetches the ISCG/RRRC/UNHCR/IOM A1 camp outlines (HDX
`outline-of-camps-sites-of-rohingya-refugees-in-cox-s-bazar-bangladesh`, 2023-04-12, CC0; 33
camps, 23.6 km²), and `countries/bd.py` gives every hex whose centre is in a camp zero weight
(28 hexes, 100,046 Kontur people on this build). **The Rohingya are not drawn.** Drawing them
would need a UNHCR figure rather than a census, and who is drawn for a persecuted group is
Anita's to decide; `gap` and `note_public` say the camps are left empty. Rohingya living outside
the camps before 2011 are in the census as whatever ethnicity they gave, which in practice is the
Bengali remainder.

**Chittagong Hill Tracts (brief §7).** Drawn at upazila, the grain BBS itself published, with
nothing finer modelled. The groups are the Hill Tracts' own recognised peoples, and their
geography is in every account of the region; religiondots draws the same upazilas by religion.
The census undercount claim is in `note_public`. No hold-back.

Kontur cap: the same five unreviewed Mymensingh/Kishoreganj/Noakhali blocks religiondots lists
warn on scatter; nothing changed (`../religiondots/sources/bd.md`, last section).

## 6. Build

```
python sources/bd_census.py --fetch      # USCB workbook, 4.8 MB -> data/normalized/bd.csv
python sources/bd_camps.py --fetch       # camp outlines, 0.1 MB -> data/geo/bd/bd_camps.gpkg
python taxonomy/build.py
python tools/check_country.py bd         # ok: 144,043,696 people, 544 units, 26 languages
python scatter.py --country bd           # 144,030 dots, 2 rings
```

Colours (`tree.d/bd.txt`): Chakma hand-set to orange-red and Tanchangya to pink, because the
generated Eastern colours sat next to Bengali's yellow in the Hill Tracts. Marma (periwinkle) and
Rakhine (orchid) part. Bawm, Khyang, Pangkhua and Kokborok are all light blues and close to each
other; they mostly do not share upazilas (Bawm in Ruma and Rowangchhari, Tripura in
Khagrachhari), so left for now.
