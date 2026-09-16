# DR Congo: Enquête 1-2-3 household heads (2005 and 2012), religion by province

**Built 2026-09-15** (session `cb8b206e-cd`), on Anita's 2026-09-15 priority (biggest visual holes first).
26 provinces, 8 answers, 117,740,329 people on COD-PS 2024, every row `modelled`. §14 ask 039 is
filed on the eastern provinces.

- `sources/cd.py` -> `data/normalized/cd.csv` (USCB workbook and COD-PS 2024 in `data/raw/cd/`)
- `sources/cd_geo.py` -> `data/geo/cd/cd_units.gpkg`, `cd_hexes.gpkg`, `cd_lookup.csv`,
  `cd_territoires.csv` (COD-AB v01 + Kontur 400 m calibrated to the 164 territoires)
- `taxonomy/cd2012.py` -> the mapping; `countries/cd.py` -> the wiring; `taxonomy/branches.py`
  `other.cd` is the one new node
- sources.md **§cd-2026-09-15** is the write-up.

```
python sources/cd.py     --fetch
python sources/cd_geo.py --fetch
```

## 1. What exists, checked 2026-09-15

| source | religion | below the nation | access |
|---|---|---|---|
| census | last one 1984; UNSD oracle has no DRC row | none | |
| **USCB, *DRC Subnational Population and Housing Data Tables* (HDX, 2021-03)** | Enquête 1-2-3 2005 and 2012, **household heads** by 9 answers | **26 provinces, 164 districts** | open |
| EDS-RDC III 2023-24 (DHS, FR393) | women and men 15-49, 10-code card (v130) | report: **national only**, Tableau 3.1 (PDF p.78) | recode files behind DHS registration |
| MICS-Palu 2017-18 (MICS6, report on bv-assk.org) | household head, `HC1A`, 10 codes incl. Église de réveil | report: **no religion table** (questionnaire PDF p.452) | UNICEF MICS registration |
| MICS 2010 (World Bank catalog 1313) | not checked | 11 old provinces | licensed through UNICEF |
| Enquête 1-2-3 2012 results report (IHSN 9258, doc 92977) | none printed | | open |
| Enquête 1-2-3 microdata, 2005 and 2012 (University of Antwerp, Great Lakes Africa Centre) | 2012 roster `M27 Religion pratiquée`, every member | district | registration form |
| Pew 2008-09 *Tolerance and Tension* (19 countries incl. DR Congo) | self | region variable not checked | Pew account |
| Afrobarometer, WVS, Global Flourishing Study | DR Congo not in any of them | | |

`catalog.ihsn.org/catalog/9258/get-microdata` says no data is available there. ARDA's copy of the Pew
survey (`fid=SUBSAHAR`) was not read past its page chrome.

## 2. The source, and why §11h's refusal is reversed

USCB's `Tribe and Religion` sheet (`CD_TRIBE_AND_RELIGION_2005_2012surveys_uscb_202103`) counts
**31,755 household heads**: 11,636 households in 2005 (`Fichier phase 1.dta`) and 20,119 in 2012
(`Phase1 IND Final.dta`), pooled by USCB on the joint panel's household ids, summed at district,
province and nation. Its Metadata sheet also says:

- 16 of 164 districts have no response: Kiri, Bolobo and Yumbi (Maï-Ndombe), Moanda and Kimvula
  (Kongo-Central), Bomongo and Ingende (Equateur), Befale (Tshuapa), Sakania (Haut-Katanga), Nyunzu
  (Tanganyika), Nyiragongo (Nord-Kivu), Niangara (Haut-Uele), Opala and Bafwasende (Tshopo), Shabunda
  and Idjwi (Sud-Kivu). Their provinces take the shares of the districts that were sampled.
- Kolwezi city is added to Mutshatsha and Tshikapa city to Kamonia; 169 responses from unknown
  districts in Kongo-Central, Tanganyika, Bas-Uele and Sud-Kivu went to the district with the largest
  sample in each. That slightly misweights those four provinces' district weighting (§4); not corrected.
- Answers, as the data dictionary's original field names: Catholique, Protestant, Kimbanguiste,
  Musulman, Autre chrétien, Animiste, Autre réligion, Sans religion, Manquant. The 2012 household
  questionnaire (IHSN 9258, doc 92969, PDF p.8) prints the card: 1 Catholique, 2 Protestante,
  3 Kimbanguiste, 4 Musulmane, 5 Autre Chrétiens, 6 Animiste, 7 Autre religion, 8 Sans religion. The
  2005 card was not read.

**§11h (2026-09-05) refused this as "not a census and not people".** That was before any survey was
drawn here. `do` and `hn` are now drawn from the same shape (the head's religion applied to the
household, MICS `HC1A`), and every survey country draws sample shares on a population base, so the
refusal no longer describes current practice. The heads basis is stated in `basis` and the note.

## 3. Checks (all in `sources/cd.py`, every run)

| check | result |
|---|---|
| level counts | 1 nation, 26 provinces, 164 districts |
| sentinel | no negative cell |
| unsampled districts | exactly 16, every count blank |
| nine answers = sample size | every sampled row, exact |
| districts sum to provinces, provinces to nation | every answer, exact |
| national sample | 31,755 |
| labels | each column's original field name, read off the data dictionary |
| COD-PS 2024 | 519 health zones, 117,808,872 |
| name join, USCB -> COD-PS pcodes | 26 of 26 by folded name; witness COD-PS 2019 (USCB's sheet) against 2024, Spearman +0.902, best of 2,000 shuffles +0.709; ratio 0.84 (Ituri) to 1.98 (Kongo-Central) inside `POP_RATIO_BAND` |

## 4. Construction

1. **District shares weighted by district population inside each province** (COD-PS 2019, USCB's
   `Population Estimates` sheet on the same GEO_MATCH). The 2012 design fixed the sample per
   district, so pooling heads straight over-weights small districts. It moves Ituri's Catholic share
   9.5 points, Protestant 7.7 (Ituri), Autre réligion 4.4 (Tshopo), Musulman 3.5 (Maniema).
2. **Split-half on districts** (no cluster ids survive): 400 random halvings of each province's
   sampled districts, median Spearman across provinces, against 2,000 regroupings of the 147 halvable
   districts into provinces (`stability.cluster_null`). Kinshasa is one district, out of the rank test.

| answer | heads | median | null 95% | p | chi-square p | largest district |
|---|---:|---:|---:|---:|---:|---|
| Catholique | 11,114 | +0.737 | +0.245 | 0.0005 | 0 | Kinshasa 8% |
| Protestant | 8,757 | +0.504 | +0.259 | 0.0005 | 6e-261 | Kinshasa 6% |
| Kimbanguiste | 974 | +0.654 | +0.248 | 0.0005 | 4e-159 | Kinshasa 11% |
| Musulman | 563 | +0.428 | +0.260 | 0.0035 | 4e-260 | Kasongo 17% |
| Autre chrétien | 7,329 | +0.785 | +0.255 | 0.0005 | 0 | Kinshasa 14% |
| **Animiste** | 167 | +0.264 | +0.278 | **0.059** | 2e-35 | Kinshasa 12% |
| Autre réligion | 1,361 | +0.566 | +0.244 | 0.0005 | 2e-99 | Kinshasa 8% |
| Sans religion | 1,470 | +0.420 | +0.257 | 0.0015 | 3e-64 | Kinshasa 8% |

3. **Animiste goes flat** at its national share (0.61%), the carried answers scaled so each row
   sums. `EXPECT_FLAT` pins the verdict.
4. **Shares on COD-PS 2024 by province**, largest-remainder rounding inside each row. `Manquant`
   (68,543 people, 0.058%) stays in the file and is `EXCLUDED`.

National as drawn, against EDS-RDC III 2023-24 (Tableau 3.1, women / men 15-49):

| this map (heads, 2005-12) | | EDS 2023-24 | women / men |
|---|---:|---|---:|
| Catholique | 35.29% | Catholique | 23.7 / 25.8 |
| Protestant | 27.85% | Protestante | 27.4 / 26.7 |
| Autre chrétien | 22.28% | Église non dénominationelle | 39.1 / 33.2 |
| Kimbanguiste | 3.18% | Autre religion chrétienne | 5.8 / 6.8 |
| Musulman | 1.69% | (not printed apart) | |
| Animiste | 0.61% | Animisme/religion traditionnelle | 3.0 / 2.5 |
| Sans religion | 4.78% | Sans religion | 1.0 / 3.5 |
| Autre réligion | 4.32% | Autre | 0.1 / 1.6 |

The Protestant level holds; Catholics fall about 11 points and the revival churches gain about 17.
Different universes (heads against people 15-49) and a decade apart, so a witness, not a rescale.
Tableau 3.1 has no Muslim or Kimbanguist row although the card offers both; where they went in the
regrouping was not traced.

## 5. Mapping (`taxonomy/cd2012.py`)

Catholique `christianity.catholic`; Protestant `christianity.protestant` (the ECC's member
communities); Kimbanguiste `christianity.africaninstituted.kimbanguist`; Musulman `islam`; **Autre
chrétien `christianity`** (the root: mostly revival churches by every account, plus JW, Salvation
Army, Adventists, Orthodox and Branhamists, none with a code; REVIEW); Animiste `indigenous.african`;
Autre réligion **`other.cd`** (new); Sans religion `unaffiliated`; Manquant excluded.

## 6. §14 (ask 039)

Drawn at all 26 provinces, the eastern conflict provinces included, as ask 018 drew Burkina Faso and
Mali. Nothing below the province is drawn. The ask gives the options (keep, merge Nord-Kivu, Sud-Kivu
and Ituri, flatten Muslims there, hold).

## 7. Placement (`sources/cd_geo.py`)

- COD-AB v01: the 26 province and 164 admin2 pcodes equal COD-PS 2024's, names and parents included;
  every province polygon within 2% of COD-AB's own `area_sqkm`.
- Raw Kontur (2023-11): 302,954 hexes, 102.5 million people; 1,742 centroids outside every admin2 unit
  (410,436 people, 0.40%) dropped. 0.867 of COD-PS nationally; by province Sankuru 0.09x,
  Kongo-Central 0.35x, Kinshasa 0.47x, Kwilu 0.55x, up to Haut-Lomami 1.61x.
- Cap blocks, read on raw Kontur (`BLOCKS`, GeoNames `CD.zip`): Kinshasa (165 hexes, 4,287,016, left),
  Mbuji-Mayi (101 hexes, 3,434,223, left), Kilwa in Pweto territoire (2 hexes, 73,449, capped to 3,858).
- Calibrated to the 164 territoires' COD-PS 2024 totals; factor over the national one p10 0.60,
  median 1.02, p90 2.42. **Five Sankuru territoires are Kontur holes** (Lubefu 59.1x, Lomela 50.7,
  Lodja 19.8, Kole 15.3, Katako-Kombe 11.9) and their totals are spread evenly over Kontur's hexes
  there (383 to 1,809/km2). Scaled like the rest, Lubefu drew a 214,456/km2 hex. Lusambo, the sixth,
  reads 0.40 and is scaled.
- Left as is: the calibrated densest hex is 112,563/km2 in Katanda (factor 4.5), the territoire
  around Mbuji-Mayi, where Kontur's city edge meets a COD-PS total five times Kontur's; its dots stay
  inside Katanda. Likasi reads 0.24 (Kontur 663,218 against COD-PS 179,860) and is scaled down; not
  investigated.

## 8. The upgrades, for whoever has an account

- **DHS EDS-RDC III 2023-24**: `CDIR81FL.ZIP` (women, `v130`, `v024` = 26 provinces) and `CDMR81FL.ZIP`
  (men, `mv130`), from `dhsprogram.com/data/dataset/Congo-Democratic-Republic_Standard-DHS_2023.cfm`;
  registration with a project description (Anita's identity). Weighted, person-level, current, with
  `Église de réveil` and `Témoins de Jéhovah` codes. Ages 15-49 only (`playbooks/dhs_mics.md`). The
  better source by every measure but age.
- **MICS-Palu 2017-18** (`HC1A`, household heads, 26 provinces): UNICEF MICS, where Anita already has
  an account (ask 006); dataset file name not read.
- **Enquête 1-2-3 person files** (2012 `M27` for every household member): the University of Antwerp's
  form, `iob-online-application.com/machform/view.php?id=75465`. Same survey, people instead of heads,
  and the cluster ids for a real split-half.

## 9. Not checked

The 2005 questionnaire's card; MICS 2010's household questionnaire; Pew 2008-09's region variable;
Catholic diocesan rolls (`roll` basis, a witness only); where EDS 2023-24's Tableau 3.1 put Muslims.

## 10. Review, 2026-09-15 (cb8b206e-rev9, full pass)

`check_md.py` clean; both editions built; `check_rollup.py cd` all modelled, nothing orphaned;
`gap_share.py --check` agrees at 0.06%. Screenshot at the entry's `view`: dots along the rivers and
roads, dense in Kinshasa, Kasaï and along the eastern border, none outside the country; Sankuru is
thinner than its neighbours but not blank.

Checked in USCB's `Tribe and Religion` sheet before reading this file. The column labels hold against
known geography (`RLG_KIMB` peaks in Kongo-Central, `RLG_MUS` in Maniema), so USCB's column order, which
is not the card's, is not a shift. Sankuru's 12.7% Kimbanguist is 76 of 695 heads over five of its six
districts (Lodja 20, Katako-Kombe 19, Lubefu 18, Lomela 10, Kole 6, Lusambo 3), so it is in the data and
not one district. §11h's "not a census and not people" was a session's table cell and
`ask/RULINGS.md` has no `cd` line, so the reversal was the builder's call. Mapping agreed: `Autre
chrétien` on the root matches the residual precedent (au2021, be2024, Puerto Rico); `other.cd` is the
usual per-country residual (other.cg, other.bf, other.td); `Sans religion` on `unaffiliated` is right
because the card offers `Animiste` separately, so the 2026-09-14 lumped-box procedure does not apply.

**Two wording points in `note_public`, for the fix batch; neither needs Anita.**
- "35.3% Catholic means that share of Congolese live in a household whose head gave Catholic" is more
  than USCB's table says. Its cells are heads (the nine answers sum to the 31,755 sample) with no
  household size, so the drawn share is a share of heads, and it equals the share of people only if
  households are the same size whatever the head's religion. Suggested: "35.3% Catholic means 35.3% of
  household heads gave Catholic, with everyone in their households drawn the same way, not 35.3% of
  Congolese asked one by one."
- The note names the other-Christian box as the revival churches twice ("the revival churches this
  survey could only record as other Christian"; "the revival churches of Kasaï"). The survey does not
  say what the box holds, the REVIEW names no source for it, and DHS 2023-24 prints "non-denominational
  churches". Suggested: "the other Christian churches of Kasaï" in the second, and "which are probably
  much of what this survey recorded as other Christian" in the first.
