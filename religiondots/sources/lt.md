# Lithuania — Statistics Lithuania, Gyventojų surašymas 2021

Wired 2026-09-05. 2,810,761 people, 60 municipalities, 16 religion categories.

| | |
|---|---|
| source | Lietuvos statistikos departamentas, SDMX dataflow **`S3R778_GBS010306_1`** |
| basis | `self_id`, voluntary question |
| geography | 60 savivaldybės |
| categories | 16 plus the universe total |
| drawn | **2,424,984 people, 86.28%** |
| licence | Statistics Lithuania open data, free reuse with attribution |

**The deepest census religion question in Europe for a country this size.** Sixteen answers
for 2.8 million people, and they include three splits nobody else on this map makes: Roman
against Greek Catholic, Orthodox against Old Believer, and the Karaims as their own answer.
Fifteen taxonomy nodes carry Lithuanian dots; Serbia's 6.6 million carry eight.

---

## 1. The host that was "403 to scripted clients" is not the host with the data

`sources.md` §11 recorded Lithuania as blocked: *"`osp.stat.gov.lt`'s REST API returns 403
to scripted clients; `osp-rs.stat.gov.lt/rest_xml/` 404s"*. Both halves were true and the
conclusion was wrong.

- **`osp.stat.gov.lt` is the human web UI** and sits behind Cloudflare. It still returns
  `Just a moment…` to everything. Nothing here needs it.
- **`osp-rs.stat.gov.lt` is a plain SDMX REST endpoint with no protection at all.** The
  earlier probe asked for `/rest_xml/` and `/api/v1/lt/`, which are not endpoints; the
  endpoints are `/rest_xml/dataflow/` and `/rest_json/data/<flow>/`. One path segment
  short of the answer.

**A statistical office's UI host and its API host are two machines, and usually only one of
them is walled.** That is the finding, and it costs nothing to check: strip the hostname
back and try the sibling. Lithuania's counts have been one request away for the whole
project.

Two asymmetries in the API worth writing down. **The catalogue is XML-only** —
`/rest_json/dataflow/` 404s while `/rest_xml/dataflow/` returns 7.4 MB — and **the data is
available as either**. And the catalogue is the searchable index: 9,521 dataflows with
bilingual names, so grepping it for `religi|tikyb` finds everything the office holds on the
subject in one pass. Nine dataflows mention religion; one of them is the census.

**TLS verifies normally.** The scouting pass reached this host with verification off and
`sources/lt.py` nearly shipped that way, which is §9h's reflex exactly. `requests` and
`curl` both verify `osp-rs.stat.gov.lt` fine, so nothing is disabled.

## 2. The dataflow, and the three it is not

| dataflow | what it is |
|---|---|
| **`S3R778_GBS010306_1`** | **Population \| Administrative territory \| Religion, 2001/2011/2021. The one to use.** |
| `S3R778_GBS010306` | the same, plus age group and sex — bigger, and nothing here needs the breakdown |
| `S3R778_GBS010502` | religion with **no geography at all** |
| `S3R0162`, `S3R422` | registered religious *organisations*, a §4.4 institution layer, not people |

113 KB, no key, no login, three censuses in one cube.

## 3. Four nested levels, told apart by the shape of the code

One geography dimension, 73 members: the country (`00`), two NUTS2 regions (`LT01`,
`LT02`), ten counties (`01`–`10`) and sixty municipalities (`11`–`94`). Summing it whole
gives **4× the country exactly**, which is India's C-01 and Hungary's WBS003 again and is
caught the same way. Codes are numeric for the administrative hierarchy and alphanumeric for
the statistical regions, so the levels separate on the shape of the code rather than on a
list.

`check()` asserts the member count at every level and that each level's total reproduces
2,810,761 exactly. The total is never suppressed, so those are equalities and not bands.

## 4. The nulls mean two opposite things, and one of them is people

**414 of the 1,020 municipality cells for 2021 are null**, and reading them all as zero
would be wrong for 298 of them. `OBS_STATUS` separates:

- `konfidencialūs duomenys` — **withheld**, disclosure control on a cell small enough to
  identify someone. Not zero. Not drawn.
- `tokio reiškinio (rodiklio) atitinkamu laikotarpiu nebuvo` — **there was no such thing**,
  i.e. a true zero.

`sources/lt.py` resolves the two on their Lithuanian text rather than on their index in the
attribute array, because an index is a fact about one download and the text is a fact about
the source. If a third status ever appears the run stops rather than guessing.

**This is the sharpest case of spec §3.8 in the project.** What is withheld is 1,683 people,
0.06% of Lithuania — and:

| category | drawn at municipality | withheld | in how many of 60 |
|---|---|---|---|
| Karaimų | 101 | **154 (60.4%)** | 31 |
| Graikų apeigų katalikų | 555 | 230 (29.3%) | 39 |
| Septintos dienos adventistų | 518 | 203 (28.2%) | 25 |
| Naujosios apaštalų Bažnyčios | 306 | 106 (25.7%) | 25 |
| Baptistų ir laisvųjų bažnyčių | 948 | 144 (13.2%) | 30 |
| Judėjų | 794 | 105 (11.7%) | 37 |
| Musulmonų sunitų | 2,007 | 158 (7.3%) | 32 |
| Sekmininkų | 2,822 | 210 (6.9%) | 24 |
| everything larger | | under 3% | |

**The withholding is proportional to how interesting a category is.** Roman Catholic,
Orthodox, no-religion and not-stated lose nothing at all; the four smallest religions lose a
quarter to two thirds. Nothing is filled back in (§3.5), so the map understates Lithuania's
small religions and understates them more the smaller they are, and `note_public` says so.

**The clearest illustration is the Karaims.** Their historic seat is Trakai, where the
community has lived since 1397 and where their kenesa still stands. Trakai's cell is
withheld. So all 101 Karaims this map can draw are in Vilnius city, and the one place in the
world most associated with them is blank. That is not an error to fix — it is what
disclosure control does to a community of 255 people — but it is worth knowing before
reading the map.

## 5. The categories, national

| source category | 2021 | % | node |
|---|---|---|---|
| Iš viso pagal religiją | 2,810,761 | 100.00 | *universe* |
| Romos katalikų | 2,085,340 | 74.19 | `christianity.catholic.latin` |
| Nenurodyta | 384,094 | 13.67 | *excluded* |
| Nė vienai | 171,810 | 6.11 | `unaffiliated` |
| Stačiatikių (ortodoksų) | 105,326 | 3.75 | `christianity.orthodox.canonical` |
| Sentikių | 18,196 | 0.65 | `christianity.orthodox.oldbeliever` |
| Evangelikų liuteronų | 15,741 | 0.56 | `christianity.lutheran` |
| Kitų | 15,353 | 0.55 | `other.lt` |
| Evangelikų reformatų | 5,540 | 0.20 | `christianity.reformed` |
| Sekmininkų | 3,032 | 0.11 | `christianity.pentecostal` |
| Musulmonų sunitų | 2,165 | 0.08 | `islam.sunni` |
| Baptistų ir „laisvųjų bažnyčių“ | 1,092 | 0.04 | `christianity.baptist` |
| Judėjų | 899 | 0.03 | `judaism` |
| Graikų apeigų katalikų (unitų) | 785 | 0.03 | `christianity.catholic.eastern` |
| Septintos dienos adventistų | 721 | 0.03 | `christianity.adventist` |
| Naujosios apaštalų Bažnyčios | 412 | 0.01 | `christianity.other` |
| Karaimų | 255 | 0.01 | **`judaism.karaite` — a node added for this source** |

`judaism.karaite` is the one new node and the one arguable call; `taxonomy/lt2021.py`'s
REVIEW entry is where to argue with it, and the short version is that filing the Karaims
under Judaism is a claim about the religion and not about the community, many of whom
describe themselves as a distinct people rather than as Jews.

## 6. Three censuses in one cube, and what moved

The dataflow carries 2001, 2011 and 2021. Only 2021 is written to `lt.csv` — the map draws
one vintage — but the trend is the reason `Nenurodyta` is excluded rather than read as
irreligion:

| | 2001 | 2011 | 2021 |
|---|---|---|---|
| population | 3,483,972 | 3,043,429 | 2,810,761 |
| Roman Catholic | 79.00% | 77.23% | 74.19% |
| **not stated** | **5.35%** | **10.11%** | **13.67%** |
| no religion | 9.51% | 6.13% | 6.11% |
| Orthodox | 4.07% | 4.11% | 3.75% |
| Old Believer | 27,073 | 23,330 | 18,196 |

**Catholic identification has fallen 4.8 points and irreligion has not risen at all.**
Everything that left the Catholic column went into the blank. A map that read `Nenurodyta`
as "no religion" would show Lithuanian secularisation tripling since 2001; the census
actually shows the no-religion answer falling slightly and the refusal rate tripling. Those
are different claims and only the second is in the data.

## 7. What the map shows

Every minority here is a border that stopped moving.

- **Biržai is the Reformed municipality.** 8.9% Evangelical Reformed against 0.49% in the
  next-highest place in Lithuania — the Radvila (Radziwiłł) family's Calvinist estate,
  granted in the 1560s, still a single bright spot with nothing around it 460 years later.
- **The Lutherans are the Prussian border.** Tauragė 9.2%, Pagėgiai 5.4%, Šilutė 4.6%,
  Jurbarkas 3.4% — the band along the Nemunas that was Lithuania Minor under Prussia. Across
  the line, Catholic Samogitia runs 89–91% (Šilalė 91.5%, Plungė 89.5%). A sixteenth-century
  state boundary, legible in a 2021 census.
- **Visaginas is 49.1% Orthodox**, the only municipality in Lithuania that is not majority
  Catholic — a town built in the 1970s for the Ignalina nuclear plant and populated from
  across the Soviet Union. It is also the most religiously mixed place in the country.
- **The Old Believers are the north-east**: Zarasai 12.1%, Švenčionys 5.0%, Visaginas 3.2%,
  Rokiškis 2.5% — refugees from the Nikonian reforms of the 1650s, still on the border they
  crossed.
- **Irreligion is the north, not the capital.** Joniškis 12.5%, Akmenė 12.3%, Klaipėda
  11.8%. This is the opposite of Serbia's pattern the same week, where irreligion was four
  central Belgrade municipalities and nothing else.

## 8. Not done

- **Romuva has no cell.** The Baltic-faith revival is inside `Kitų` (15,353 people, and
  large for a residual on a form this detailed). It was state-recognised in 2025, after this
  census, so a 2031 census may name it.
- **2001 and 2011 are in the cube and not written out.** Lithuania is a candidate for
  anything the project ever does with change over time — three censuses on one geography,
  same categories, no reconciliation needed.
- **The age/sex breakdown exists** (`S3R778_GBS010306`) and would give the cohort gradient
  Chile's file argues from. Not fetched.
