# Malaysia — Banci Penduduk dan Perumahan Malaysia 2020

**32,447,385 people, 160 administrative districts, 7 categories, one census year.**
Ingested 2026-09-07. `sources/my.py` builds `data/normalized/my.csv`;
`sources/my_geo.py` builds the boundaries and the placement grid.

`sources.md` §11r is how Malaysia was chosen, §11s is the acquisition and the traps,
§11s-iii is the boundary work. This file is the per-country reference.

---

## 1. The source

Department of Statistics Malaysia (DOSM), *Penemuan Utama Banci Penduduk dan Perumahan
Malaysia 2020* — Key Findings of the Population and Housing Census of Malaysia 2020.
Released 2022-05-29. Distributed through the **eStatistik** portal
(`newss.statistics.gov.my`), which needs a free registration and a browser; a scripted
session is bounced to `epTimeout.seam`.

**The table is Jadual/Table 7**, *Bilangan penduduk mengikut agama, jantina dan daerah
pentadbiran/ jajahan* — population by religion, sex and administrative district. The
national volume's equivalent is **Table 6**, by state.

### Which file, out of about twenty-one per state

Each state publication's download list holds ~21 files. **Twenty carry the subject
`MYLOCAL STATS`** (tables 17–102: housing, employment, income) and **none of them has
religion in it.** The one that matters is the single row whose subject is
**`<STATE> JADUAL 1 HINGGA 16`**. In Perak's list it was record 21 of 21, alone on page 3.

Sabah and Sarawak title their volumes **"State Sabah"** and **"State Sarawak"** where every
peninsular one says *"Negeri X"*, and each sits among 27 and 40 per-district publications.
**Do not download the district volumes** — the state file reproduces them exactly, which
was verified: the standalone Kampar district volume agrees with its row in the Perak state
volume on all eight cells.

### Files on disk — `data/raw/my/`

| file | what |
|---|---|
| `<STATE> JADUAL 1 HINGGA 16.xlsx` × 16 | the counts. Table 7 is religion × district |
| `JADUAL 1 HINGGA 29.xlsx` | the **national** volume. Table 6 is religion × state — the outside check |
| `JADUAL BANCI MALAYSIA 2020 MUKIM BANDAR PEKAN.xlsx` | all 1,756 mukim, **no religion** — kept as the evidence for §3 below |
| `DP Kampar JADUAL 1 HINGGA 12.xlsx` | one district volume, kept as the cross-check |
| `geoBoundaries-MYS-ADM1.geojson`, `-ADM2.geojson` | boundaries |

---

## 2. The categories, and what each one actually holds

| Malay | English | count | share | node |
|---|---|---|---|---|
| Islam | Islam | 20,610,060 | 63.52% | `islam` |
| Buddha | Buddhism | 6,066,784 | 18.70% | `buddhism` |
| Kristian | Christianity | 2,941,049 | 9.06% | `christianity` |
| Hindu | Hinduism | 1,969,471 | 6.07% | `hinduism` |
| Tidak Diketahui | Unknown | 303,070 | 0.93% | `unknown` |
| Lain-lain\* | Others | 285,152 | 0.88% | `other.my` |
| Tiada Agama | No Religion | 271,799 | 0.84% | `unaffiliated` |

\* The footnote is printed on every sheet and is load-bearing: *"Others include Sikhism,
Taoism, Confucianism, Bahai, Tribal/ folk/ other traditional Chinese religion, Animisme
and Others."*

**The basis is total population including non-citizens** — roughly 2.7 million in 2020.
§3.1: do not compare these against a citizens-only figure.

`taxonomy/my2020.py` has the reasoning per cell. The two that need reading before use:

- **`Tiada Agama` is not a secular geography.** It peaks at **35.94% in Kecil Lojing,
  Kelantan** and runs Rompin 13.4%, Selangau 12.5%, Pekan 10.3%, Cameron Highlands 8.8% —
  every one an Orang Asli or interior indigenous district, and none of them a city (Kuala
  Lumpur is 0.9%). Read it as indigenous practice with no box on the form. Mapped as
  printed anyway, because remapping asserts a magnitude DOSM does not publish; the reader
  gets it in `note_public`.
- **`Tidak Diketahui` is 97% male** — 67,664 men to 25 women in Perak, 34,437 to 1 in
  Kinta. Almost certainly non-citizen labour enumerated without the religion item. **Not
  distributed** (§3.5); it stays its own node.

---

## 3. What is NOT available, and it was checked properly

**Religion stops at administrative district.** `JADUAL BANCI MALAYSIA 2020 MUKIM BANDAR
PEKAN` is one 200-sheet workbook covering all 1,756 mukim for the whole country, and it
carries exactly three tables per state: population/sex/households (1.1), **ethnicity**
(1.2) and age (1.3). The state volumes' Table 11 is mukim-level too and is population and
households only. **1,756 mukim is not available at any price.**

**And 2010 is not a cheaper fallback.** The 2010 census published religion at **national
and state only** — its district tables are ethnicity, and *Population Distribution by Local
Authority Areas and Mukims 2010* has no religion at all. **2020 is the first Malaysian
census to publish religion below the state.**

---

## 4. Reconciliation

Three checks, and the third is external:

1. within each state volume, the seven categories sum to the state total;
2. within each state volume, the districts sum to the state total;
3. across volumes, the sixteen states sum **category by category** to the national
   volume's Table 6 — a different publication, so this is not the file agreeing with
   itself.

All three pass exactly, on all seven categories and all sixteen states, to 32,447,385.
`sources/my.py` asserts every one and refuses to write a file that fails.

**The summary release's "Others 860 thousand" is the three residual categories added
together** (285,152 + 271,799 + 303,070 = 860,021). Reading it as the `Lain-lain` cell
alone overstates that cell threefold.

---

## 5. Geography

`geoBoundaries gbOpen MYS ADM2`, vintage **2020 — the census year**. 159 polygons against
160 counted districts.

- **Three renames aliased**: Kulaijaya→Kulai, Ledang→Tangkak (both Johor, 2015),
  Nabawan / Persiangan→Nabawan (Sabah).
- **The name join is verified spatially**, not trusted: every polygon is given a state by
  point-in-polygon against ADM1 and that state must match the census. It does for all 159.
  The only disagreements are geoBoundaries writing the English *Penang* and *Malacca*.
- **Putrajaya is missing from ADM2 and lies entirely inside Sepang.** Appending the ADM1
  polygon would double-count 48.7 km², because Sepang still covers 100.0% of it — the ADM2
  layer never registered the 2001 carve-out. So Putrajaya is **subtracted from Sepang**
  and then added. W.P. Kuala Lumpur and W.P. Labuan are ordinary ADM2 features.

Placement is **Kontur** `kontur_population_MY_20231101`, 144,439 hexes after the centroid
join, 0.96% of hex population outside every district and dropped. Country ratio **1.050**,
which is right for a 2020 census against a 2023 grid. **125 of 160 districts, 91.6% of the
population, fall in 0.8–1.25.** The eleven outliers were checked against polygon area and
are genuine Kontur/census differences, not bad boundaries — see `sources/my_geo.py`.

---

## 6. What it shows

| State | Pop | Muslim | Christian | Buddhist | Hindu |
|---|---|---|---|---|---|
| Terengganu | 1,149,440 | **97.3%** | 0.3 | 2.0 | 0.2 |
| Kelantan | 1,792,501 | 95.5 | 0.4 | 2.8 | 0.2 |
| Kedah | 2,131,427 | 78.5 | 0.8 | 12.4 | 5.9 |
| Sabah | 3,418,785 | 69.6 | **24.7** | 5.1 | 0.1 |
| Selangor | 6,994,423 | 61.1 | 4.9 | 21.6 | 10.3 |
| Pulau Pinang | 1,740,405 | 45.5 | 4.3 | **37.6** | 8.4 |
| W.P. Kuala Lumpur | 1,982,112 | 45.3 | 6.4 | 32.3 | 8.2 |
| **Sarawak** | 2,453,677 | 34.2 | **50.1** | 12.8 | 0.1 |

**Malaysia holds a wider religious range inside one border than any other country on this
map** — 97.3% Muslim in Terengganu against 50.1% Christian in Sarawak.

District peaks worth knowing: Christian — Tebedu 93.1%, Kapit 89.6%, Lubok Antu 88.0%,
Belaga 86.2%, Kanowit 84.8% (all Sarawak), Tambunan 79.8% (Sabah). Buddhist — Timur Laut
52.5%, Kampar 45.8%, Seberang Perai Tengah 34.0%. Hindu — Bagan Datuk 21.5%, Port Dickson
17.3%, Klang 16.9%.

**What it cannot show**: no Christian denomination (the Sidang Injil Borneo, the largest
Protestant body in the interior, appears on no map here), no branch of Islam (Shia and
Ahmadi are both legally suppressed and both inside the one cell), no branch of Buddhism
(Mahayana and Theravada are not separated), and **no Chinese folk religion, Taoism or
Confucianism** — all three are inside `Lain-lain`, so a tradition this map draws for China,
Vietnam and Singapore cannot be brought out for Malaysia at all.
