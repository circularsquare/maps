# South Sudan: HFSSS 2015-2016 tribe by former state; Jonglei, Unity, Upper Nile by homeland county

Drawn 2026-10-05 (session edd42a8c-ss): 8,252,228 people in 7 of 10 former states (2025
county-based estimates), every row `modelled`. **Filled 2026-10-07 (fix-ss)**: Jonglei, Unity and
Upper Nile, 5,043,986 people in 32 counties, each county on its homeland group's language
(section 7). Now 13,296,214 people, 39 units (7 states + 32 counties), 73 nodes, every row
`modelled`. 13,260 dots at 1:1000, 6 rings. Ask 020 (the 2025 phone survey, Anita's login)
would still replace section 7 with a measured language question.

```
python sources/ss_hfs.py       # religiondots' HFSSS zips, read-only -> ss.csv, ss_birth.csv
python sources/ss_homeland.py  # -> ss_homeland.csv (the three states, by county)
python taxonomy/build.py
python tools/check_country.py ss
python scatter.py --country ss
```

## 1. What exists

- **Census.** No census has asked a language: 2008 (Sudan's fifth, the south included) did not,
  nor did 1973-1993; IPUMS's 2008 sample has no language or ethnicity variable. The 1955-56
  Sudan census did (sources/sd.md section 1), volumes search-only on HathiTrust.
- **Open surveys.** South Sudan is in no round of Afrobarometer, Arab Barometer or WVS. CLEAR
  Global has no South Sudan layer on HDX (only Sudan's). HDX searches for language, tribe and
  ethnicity in South Sudan (2026-10-05) find nothing; REACH's and IOM's MSNA tables on HDX are
  Sudan's. IOM DTM's terms forbid derivative works (religiondots ruling 2026-09-16).
- **World Bank Microdata Library** (variable lists read through its open API, 2026-10-05):
  - HFSSS wave 1 (2015, catalog 2778) and wave 2 (2016, catalog 2777): hhq C.8 "Which tribe does
    [head] belong to?", C.8.1 the verbatim of "Other", plus the head's state and county of
    birth. Already downloaded by Anita for religiondots (its ask 042). **Used.**
  - HFSSS wave 3 (2016, catalog 2914): the same tribe item; not downloaded (login). Wave 4
    (2017, catalog 2916, direct download, no login): tribe, but only towns of seven states
    revisited from waves 1-2; it adds nothing wave 2 lacks. Wave 4 + Crisis Recovery Survey
    (catalog 3392): tribe, with the Protection of Civilians camps as strata; camps are not a
    state's population.
  - **High Frequency Phone Survey 2025, waves 1 and 2** (catalog 8589, 8590): b1_7 "What is the
    primary language spoken in your household?", with state and county, all 10 states and 3
    administrative areas, about 2,000 and 4,000 adults. **The best source there is; needs a
    login. Ask 020.**
  - HFS 2012-2014 panel (catalog 2576): "which language do you usually use talking to family
    and friends", Juba, Wau, Rumbek, Malakal only; login. A retention check (section 4).
  - Forced Displacement Survey 2023 (catalog 6639): mother tongue, but refugees and their hosts
    in five strata, no state composition; remote access only.

## 2. How the counts are made (`sources/ss_hfs.py`)

- Person weight = household weight x hhsize (religiondots' construction; hhsize equals the hhm
  roster in every household of both waves, asserted). Wave 1 for Central, Eastern and Western
  Equatoria, Lakes, Northern and Western Bahr el Ghazal (towns and countryside, 3,550 heads);
  wave 2 for Warrap (towns, 173 heads). Shares per state times religiondots' `ss_lookup.csv`
  `pop` (OCHA/NBS 2025 county-based estimates), largest remainder; each state sums to its
  estimate (asserted). State labels join COD-AB names exactly (asserted).
- **Warrap from a town sample.** Religiondots left Warrap empty because town and countryside
  answered religion very differently. For the group question the gap is small in the two Dinka
  states wave 1 sampled both ways: Dinka share of persons, Northern Bahr el Ghazal towns 0.889,
  countryside 0.972; Lakes towns 0.966, countryside 0.828 (the countryside's Atuot and Beli).
  Warrap's towns are 0.975 Dinka. The big fact is not in doubt; Warrap's rural minorities (Bongo,
  Jur, Luwo in Tonj) are under-drawn.
- **Witness**: wave 2's towns against wave 1's towns, six states, every (state, answer) cell:
  r = +0.976 over 828 cells. Top answers agree (Lakes Dinka 0.967 vs 0.966; Central Equatoria
  Kakwa 0.291 vs 0.299); Western Equatoria's Azande 0.774 vs 0.584 and Western Bahr el Ghazal's
  Belanda Viri 0.381 vs 0.225 differ most (dissimilarity 0.27-0.33): towns are 12 EAs each.
- 2 heads with no answer (980 people) are not drawn.
- **Placement** (`ss_birth.csv`, `countries/ss.py`): the survey has no county of residence. 89%
  of heads were born in their state of residence with a county named; their birth counties
  (joined to COD-AB v03's 78 counties within the birth state, exactly, after five spelling
  aliases: Rumbek Center, Lafon/Lopa, Raga, Meluth, Panriang) give each language a county
  share, (heads born there + 12 x state share) / (heads born there + 12), times Kontur
  population (religiondots' county-calibrated hexes). Rural EAs' modal birth county holds 94%
  of their in-state-born heads on average, so birth county is a fair stand-in for residence in
  the countryside; in towns less so (73%). Result: Kakwa 252 of 383 dots in Yei, Kuku 187 in
  Kajo-Keji, Mundari 184 in Terekeka, Toposa in the three Kapoetas, Reel in Yirol West. Cost:
  county dot density runs 0.55 (Mvolo) to 2.17 (Torit) of the county estimate; Juba 0.74.

## 3. Mapping and tree (`taxonomy/ss2015.py`, `taxonomy/tree.d/ss.txt`)

The docstring of `ss2015.py` lists every call. In short: one node per card group; spelling
variants merged (Kakowa into Kakwa, its 34 heads in Central Equatoria, born in Yei; Atwot,
Bviri, Olubo); two names for one people merged (Jurchol and the card's "Luo", answered in Jur
River county, on Luwo; Shatt, Jur Shat and Thuri on Thuri; Boya and Larim on Narim). "Moro Nuba"
(3 heads, all born in Equatorian or Bahr el Ghazal counties) drawn on Moru as a mis-tap. Bari
varieties (Pojulu, Mundari, Nyangwara, Kuku, Nyepu) and Lotuxo groups (Ifoto, Imatong,
Ketebo, Logir, Lokoya, Lango, Lopit, Dongotono) each a leaf, as the card names them. Arab
write-ins (Misiriya, Habaniya, Halba) on sd.txt's Sudanese Arabic; Ugandan write-ins on Ganda,
Konzo, Lugbara. Nuba, Darfur alone, Fertit, Arua district and 12 unidentified names on
`africa_other`. No new families or groups; four hand-picked colours (fragment's last comment).

## 4. Retention: not checked

AGENT_BRIEF section 2 asks how many of each group speak its language. No source on disk says.
The 2012-2014 HFS panel (catalog 2576, ask 020) asks the language used with family in four
towns and would give town retention for Dinka, Bari, Zande and others. Juba Arabic is the
lingua franca of the towns and the first language of many town-born children; English is
taught. Rural retention of the home language is believed high, but nothing here measures it,
and no one is moved to Arabic or English.

## 5. An outside-estimate version (ask 019's question; drawn since 2026-10-07 as section 7)

If ask 019 is answered yes for Sudan, the same construction would fill South Sudan's three empty
states: per-group speaker estimates (Ethnologue's as Wikipedia carries them; the 1955-56 census
by district if the volumes are fetched) placed on homeland counties, scaled to each county's
2025 estimate. Jonglei: Dinka (Bor, Twic East, Duk), Nuer (Lou, Gawaar: Akobo, Uror, Nyirol,
Ayod, Fangak), Murle (Pibor), Anyuak (Pochalla), Jie and Kachipo. Unity: Nuer (Bul, Leek,
Jikany: Rubkona, Guit, Koch, Leer, Mayendit, Panyijiar, Mayom), Dinka (Pariang, Abiemnhom).
Upper Nile: Shilluk (Panyikang, Fashoda, Manyo), Nuer (Jikany: Nasir, Ulang, Longochuk, Maiwut),
Dinka (Padang: Baliet, Melut, Renk), Maban and Uduk (Maban), Burun. It would draw counts no
source measured, which is why it is Anita's call; ask 020's phone survey would make it
unnecessary.

## 6. Calls someone might reverse

- Ethnicity read as language with no retention check (section 4).
- Warrap drawn from 173 town heads (section 2).
- Birth county as a placement proxy, K = 12.
- Kakowa merged into Kakwa; card "Luo" on Luwo; "Moro Nuba" on Moru.
- Jonglei, Unity, Upper Nile: one homeland group per county (section 7).
- Dinka recoloured 0.53 0.16 268 (indigo, #4463c7) in `tree.d/ss.txt`, away from et.txt's Nuer.

## 7. Jonglei, Unity and Upper Nile by homeland county (fix-ss, 2026-10-07)

Anita (2026-10-07) asked for "a stab" at the hatched states without ask 020's login. Built under
ask 019's ruling (published estimates placed by homeland, rows `modelled`, `sources/eg.md` the
model), at county grain since the 2025 estimates are per county. `sources/ss_homeland.py`.

**Searched, nothing measured and open** (6 web searches): CLEAR Global/TWB (still no South Sudan
layer); the 2025 HFPS (catalog 8589) variable page over the open NADA API
(`/index.php/api/catalog/SSD_2025_HFPS-W1_v01_M/variable/V13`) prints only national unweighted
frequencies of b1_7, 2,020 adults: Dinka 1,029, Juba Arabic 248, Other 228, Nuer 200, Arabic 115,
Zande 108, English 92, **Shilluk 0, Bari 0** (two card codes empty nationally: a coding or
routing fault, worth knowing before ask 020's data is used). 22% name Arabic, Juba Arabic or
English, a phone-owner sample's town lean and a hint at the retention question in section 4. No
state split, so not drawn. The HFSSS's own 111 heads living elsewhere but born in the three states
(waves 1-2, born-county x tribe; scratch tabulation) agree where they are many: Bor South 49 of 53
Dinka, Twic East 21/21, Duk 8/8 Dinka; Pibor 3/3 Murle; Pochalla 1/1 Anyuak. Too few and too
migrant-biased (Fangak's 3 born heads are all Dinka) to draw from. REACH's Malakal City Profile
(Feb 2025, `data/raw/ss/`) names Shilluk, Dinka, Nuer without shares.

**Sources used**
- **CSRF county profiles** (Conflict Sensitivity Resource Facility, csrf-southsudan.org/county_profile/<slug>/,
  read 2026-10-07; the "Ethnic groups" line of each saved in `data/raw/ss/csrf_county_profiles_ethnic.json`).
  The profiles also print the 2025 OCHA estimate the map already uses (Malakal 204,041 both). The
  first-named group is the county's; the script asserts each against the saved line.
- **JICA**, *Project for Comprehensive Planning and Support for Urgent Development on Social
  Economic Infrastructure in Malakal Town*, final report July 2014, appendix I, attachment II,
  A1-270 "Tribe and Ethnic Group" (openjicareport.jica.go.jp/pdf/12228961_05.pdf, saved in
  `data/raw/ss/`): 884 households in 24 bomas of Malakal Town, Shilluk 445 (50.3%), Nuer 201
  (22.7%), Dinka 134 (15.2%), others 104 (11.8%). Fieldwork date not printed in the appendix;
  it must predate the town's destruction in 2014. Applied to all Malakal county (town, PoC,
  countryside); "others" on `africa_other`.
- **Joshua Project** (`data/raw/pg/joshuaproject_pgic.csv`, ROG3 = OD), scaled 1.0769 (2025
  estimate 13,297,196 / JP 12,348,000), for two small groups the profiles name second and whose
  JP point falls in that county: Kacipo-Balesi (JP "Suri", koe) 5,061 in Pibor; Opo (JP "Opo" +
  "Buldit", lgn) 15,938 in Maiwut (the profile's "Koma").

| county (2025 est.) | drawn |
|---|---|
| Akobo, Nyirol, Uror (Lou), Ayod (Gawaar), Fangak (Thiang, Lak); Guit, Koch, Leer, Mayendit, Mayom, Panyijiar, Rubkona; Longochuk, Luakpiny/Nasir, Ulang (Jikany) | Nuer |
| Maiwut 147,823 | Nuer 131,885, Opo 15,938 |
| Bor South, Duk, Twic East (Southeastern Dinka); Canal/Pigi, Abiemnhom, Pariang, Baliet (Padang); Melut (Ageer, Nyiel); Renk (Abialang) | Dinka |
| Fashoda, Manyo, Panyikang | Shilluk |
| Malakal 204,041 | Shilluk 102,713, Nuer 46,394, Dinka 30,929, other African 24,005 |
| Pibor 237,649 | Murle 232,588, Kacipo-Balesi 5,061 |
| Pochalla | Anyuak |
| Maban | Mabaan |

**Not split** (named second by a profile, no figure in that county): Anyuak in Akobo; Gawaar and
Lou Nuer and Shilluk in Canal/Pigi; Jie in Pibor (Glottolog jiye1239, a Toposa dialect); Nuer
in Maban; Koma in Longochuk (JP's Komo, 26,000 with its point there, would be 35% of the county;
the script caps a small group at a fifth); Padang Dinka in Panyikang; Shilluk in Renk. Refugees
from Sudan (Maban's Blue Nile camps: Uduk, Jumjum; Pariang's Nuba camps) are outside the county
estimates, as in the other states.

**Checks** (drawn in the three states against JP's whole-country figure, scaled): Nuer 2,800,928
vs 1,581,923; Dinka 1,458,351 vs 4,425,937 (with the seven states, 5.10M); Shilluk 339,997 vs
575,049; Murle 232,588 vs 213,220; Mabaan 84,550 vs 97,995; Anyuak 82,568 vs 160,454. The Nuer gap
is the 2025 county estimates, not the rule: the ten Nuer homeland counties' estimates (Rubkona
361,802 with Bentiu's IDP camp, Nasir 350,742) run high against JP's older figure, and the phone
survey's unweighted 10% Nuer is nearer JP. National now: Dinka 38.4%, Nuer 21.1%. Shilluk low
against JP fits Craze (Small Arms Survey 2019): many Shilluk were displaced to Sudan.

**Placement**: the three states' units are counties (`countries/ss.py` `_place_unit`: the hexes'
`county` column for SS03/06/07), placed on religiondots' county-calibrated Kontur population.

**Tree**: new `nilosaharan.nilotic.mabaan` (maba1273) and `nilosaharan.surmic.kacipo`
(kaci1244); Anuak, Koman, Opo repeated from et.txt. `taxonomy/ss2025.py` is the identity.

**Room for improvement**: ask 020's 2025 phone survey (language by state and county) replaces
all of this. Short of it, any open survey or IDP-camp registration by ethnicity per county would
split the mixed counties above.

## Terms

HFSSS: World Bank Microdata Library public-use terms (no redistribution of microdata; aggregates
may be reported, study cited): only state and county aggregates leave the zips, which stay in
religiondots' raw folder. CSRF county profiles: public web pages, cited per county, one line of
each kept. JICA report: public on openjicareport. Joshua Project: as sources/sd_estimate.py.
COD-AB and COD-PS: OCHA, CC BY-IGO. Glottolog CC BY. Kontur CC BY 4.0.
Citation: Pape, Utz J. (World Bank) and National Bureau of Statistics, *South Sudan High
Frequency Survey 2015, Wave 1* (DOI 10.48529/bn2b-8q88) and *2016, Wave 2* (DOI
10.48529/xz60-7w58).

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Toposa is in an Ateker group with Turkana, Karamojong and Teso. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
