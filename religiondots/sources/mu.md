# Mauritius — Statistics Mauritius, 2022 Housing and Population Census, Volume II Table D6

Wired 2026-09-07. 1,233,097 people, 182 units, 13 drawn categories.

| | |
|---|---|
| source | Statistics Mauritius, **2022 HPC Volume II, Table D6** (report pages 144-149) |
| basis | `self_id`, *"Religion as reported by respondent"* |
| geography | **182 Municipal Council Wards and Village Council Areas** — ~6,800 people each |
| categories | **13** plus the universe total; all 13 drawn |
| drawn | **1,233,097 people, 100%** of the resident population enumerated |
| licence | Statistics Mauritius publication, free to download and cite |

**The only source on this map that counts Hinduism as more than one thing.** Five of the
thirteen groups are Hindu — Marathi, Tamil, Telugu, Vedic/Arya Samaj, and the unspecified
majority — published at village level in a country that is 47.9% Hindu. India's census does
not do this. Guyana's, a quarter Hindu, does not. Before Mauritius the tree divided Hinduism
not at all: one family node, and Vietnam's Cham Balamon under it. **Four nodes were added to
`branches.py` for this one table.**

---

## 1. The site is SharePoint and its pages are not guessable; its documents are

`statsmauritius.govmu.org` is SharePoint. **Every `.aspx` path that ought to exist 404s** —
`Pages/Census_and_Surveys/HPC/HPC.aspx`, `Pages/Census_and_Surveys/Census2022/Census2022.aspx`
— and `Pages/Statistics/By_Subject/Population/SB_Population.aspx` **redirect-loops a scripted
client to 30 hops** while rendering fine in a browser.

The `/Documents/` tree beside it is wide open and is what to fetch:

```
statsmauritius.govmu.org/Documents/Census_and_Surveys/Census2022/HPC_TR_Vol2_Demography_Yr22.pdf
```

**The file naming is inconsistent inside one release**, so a guessed sibling path will miss:
Volume II is `HPC_TR_Vol2_Demography_Yr22.pdf` and Volume III is
`HPC_Vol3_Geo_Migration_Yr22_ 120126.pdf` — a different prefix (`HPC_TR_` vs `HPC_`), a date
suffix on one and not the other, and **a literal space in the filename**. Find the exact URL
rather than deriving it.

## 2. Both halves of §3.9's trade are published, eight pages apart

The report's contents page labels every table with the geographic level it is presented at:
`I` island, `D` district, `R` municipal ward and village council area. Religion appears
twice:

| table | level | units | categories |
|---|---|---|---|
| **D5** | `I` | 3 (Republic, Island of Mauritius, Rodrigues) | **~60 individually named bodies** |
| **D6** | `R` | **182 drawn** | **13 groups** |

D5 is the deepest religion question on this map — `L'Assemblée de Dieu`, `Mission Salut et
Guérison`, `La Voix de la Delivrance`, `Peniel Tabernacle`, `Full Gospel Church`, `Église
Chrétienne`, `Christian Tamil`, `Church of England`, `Presbyterian`, `Methodist`, `Mormon`,
`Arya Samajist`, `Témoin de Jehovah` — and it has three units, which is not a geography.

**D6 is drawn.** D5 is read only as a check on the national column and as the source of the
splits recorded in §5 below. A source that gave both at once would be the best religion table
anywhere here; this one gives either.

## 3. The category list

```
Total                                  1,233,097   universe
Buddhist/Chinese                           5,053    0.41%
L'Assemblée de Dieu / M.S et Guérison     11,357    0.92%
Church of England/Protestant               2,457    0.20%
Roman Catholic                           307,515   24.94%
Other Christian                           76,883    6.23%
Marathi/Marathi Hindu                     19,052    1.55%
Tamil/Tamil Hindu                         63,950    5.19%
Telugu/Telugu Hindu                       25,216    2.04%
Vedic/Hindu Vedic & Aryan                  7,422    0.60%
Hindu & Other Hindu                      474,623   38.49%
Islam/Muslim & Other Muslim              224,885   18.24%
No religion                                7,753    0.63%
Other & Not stated                         6,931    0.56%
```

**The three headline shares reconcile exactly against the office's own published figures**,
which is what proves the column order was read correctly: Hindu (five cells) 47.9%, Christian
(four cells) 32.3%, Muslim 18.2%. Statistics Mauritius publishes 47.9 / 32.3 / 18.2.

**Three of the four Hindu splits are COMMUNITIES and the fourth is a MOVEMENT**, and the
difference is spec §2.1's two relations showing up in one census question. Marathi, Tamil and
Telugu Hindus descend from indentured labourers out of three different parts of India and
have kept separate temples, priesthoods, languages and festival calendars for a century and a
half. **Arya Samaj** is Dayananda Saraswati's 1875 reform — Vedas alone, no image worship, no
caste birth-right — which reached Mauritius in 1910 and split Mauritian Hinduism bitterly
enough to shape its politics for decades. Anyone can join it; nobody joins being Telugu.

`taxonomy/mu2022.py` has the per-category reasoning. The two arguable calls are
`Buddhist/Chinese` (§5) and `Other & Not stated` (§6).

## 4. The parse: a six-level stub with no codes, and a sixth level hidden in the fifth

There is **no code column anywhere in D6**. Level is indentation and nothing else:

```
24.6  REPUBLIC OF MAURITIUS
31.8    ISLAND OF MAURITIUS
39.0      PAMPLEMOUSSES DISTRICT-Wholly Rural
46.2      MOKA DISTRICT - Urban
53.4        Arsenal VCA                          <- the drawn tier
```

**The parse rule is "the last fourteen tokens are the figures".** Nothing else survives the
row shapes: `Town of Port Louis-Ward 3 12,612 …` puts the **ward number inside the name**
where it reads as a fifteenth figure; `Region 1 - La Ferme 7,578 …` does the same on
Rodrigues; and `Town of Port Louis-Ward 2-North (South in Moka & West in B/R) 8,055 …`
carries a parenthetical naming *other districts* inside the unit's own name.

**Four parent rows put their name and their figures on different lines** — `PORT LOUIS
DISTRICT-Wholly Urban` at y=179 and its fourteen numbers at y=180, likewise `GRAND PORT
DISTRICT`, `RODRIGUES` and `ISLAND OF RODRIGUES - Wholly Rural`. Grouping words by exact y
splits each into a nameless number row and a numberless name row and **both then fail the row
test silently**, so the district disappears from the parent-sum check that is the only thing
verifying the units beneath it. Rows are clustered with a y tolerance instead.

### A SIXTH TIER HIDES INSIDE THE FINEST ONE — Serbia's §9p in a second country

`Town of Curepipe` (70,008) is printed at **x0=53.4, the drawn-unit indentation**, and so are
`Town of Curepipe-Ward 1` … `Ward 5`, which sum to it exactly. Four towns do this — Beau
Bassin/Rose Hill, Curepipe, Quatre Bornes and Vacoas/Phoenix — **334,496 people, 27.1% of the
country, counted twice** by any rule that trusts the indentation.

Nothing marks it. All 209 per-row totals stay perfect either way; the only thing that sees it
is the units-sum-to-the-republic check, and the excess it reported was exactly the four town
rows added together.

The fix is Serbia's: **a parent's children are the consecutive following rows that sum to it
exactly**, which doubles as a parse check. Two conditions are required rather than one — the
exact sum *and* the name prefix — so a false positive would need a coincidence in two
unrelated spaces at once.

**Port Louis is not one of them**, and that is a real fact rather than an exception: Port
Louis is a district *and* a town, so its wards hang off `PORT LOUIS DISTRICT-Wholly Urban` one
level up and there is no town row beside them. OSM agrees — it has four town relations at
`admin_level=8`, not five.

## 5. `Buddhist/Chinese` is one cell for two things, and D5 says how many of each

D6 gives 5,053. D5, nationally only, splits it: **Buddhist 2,178, Chinese 2,434, Other Chinese
441** — summing exactly. There is no table anywhere that gives the split a geography.

It is drawn on **`chinesefolk`**, per §3.3: the cell is the Sino-Mauritian community that kept
its ancestral religion, and Sino-Mauritian practice is the ordinary Chinese combination of
Mahayana Buddhism, Guanyin and ancestor observance rather than two separable things.

**What is wrong with that, stated rather than hidden:** `chinesefolk`'s own note calls it
"China's own tradition", and 2,178 people here answered `Buddhist` and are now drawn as
Chinese religion. Sending all 5,053 to `buddhism` instead misfiles 2,875 the other way and
denies the folk practice outright. Both are wrong; this one is wrong about fewer people, and
it is 0.41% either way.

## 6. `Other & Not stated` is the only residual here that mixes an answer with a non-answer

6,931 people, 0.56%. **Every other source on this map keeps `not stated` in a cell of its own
and spec §3.5 takes it off the tree.** Mauritius pools the two and publishes no split at any
geography, so §3.5's usual move is unavailable and the choice is between drawing an unknown
number of non-responders as a religion or deleting an unknown number of real adherents.

It is drawn, and the node's note is the marking §3.5 asks for: **read 6,931 as a ceiling on
Mauritius's other religions, not a count of them.** What is genuinely in it is small and
knowable in kind if not in number — the Bahá'ís, present since the 1950s; a few hundred
Sikhs; and the island's tiny Jewish community, of which the Beau Bassin detainees of 1940-45
are the reason there is one at all.

## 7. What the map shows

**Each Hindu community has its own map and none is the one you would guess.**

* **Marathi** is the tightest cluster of the four and it is the **southwest coast**: La
  Gaulette 27.7%, Baie du Cap 27.0%, against 1.5% nationally — eighteen times the national
  rate on a stretch of Black River and Savanne coastline that is otherwise unremarkable.
* **Tamil** is southern and central, strongest in Savanne (8.2%) and Plaines Wilhems (7.1%),
  and **thinner in Port Louis (3.7%) than in the country as a whole** — the opposite of what
  the usual account of an early-arriving urban community predicts. Unit peak: St Julien
  d'Hotman VCA-East, 18.0%.
* **Telugu** is Savanne (4.7%) and Rivière du Rempart (3.8%).
* **Arya Samaj** is thin everywhere, which is what a movement rather than a community looks
  like on a map.
* **The unspecified Bhojpuri-descended majority is the cane belt**: Camp Thorel 95.4%,
  La Laura-Malenga 93.2%, Laventure 93.2%.

**Port Louis Ward 5 is 96.81% Muslim** — 17,058 people, and one of the most nearly total
single-religion units drawn anywhere on this map. The city as a whole is 40.9% against 18.2%
nationally; Moka is 24.2% and Savanne 21.7%.

**Rodrigues is a different country.** All six regions run **84.9% to 91.9% Roman Catholic**
and 0.5% Hindu, against 24.9% and 38.5% on the main island — a Creole Catholic population 600
km east of a Hindu-majority republic, and the sharpest internal contrast any country here
holds.

**And no-religion is 0.63%, the lowest share of any country on this map** — below Ghana's
4.50% and Malawi's 2.15%. What geography it has is the west-coast resort strip (Flic en Flac
4.93%, Tamarin 4.31%), which is a resident-foreigner pattern rather than a Mauritian one.

## 8. The five Hindu rows share one orange at L2 — measured, and decided to leave

> **DECIDED 2026-09-07, Anita:** *"i think the colors of mauritius look fine tbh. they seem
> quite similar and if people want to see detail they can click on hinduism to disperse the
> colors, and that looks fine."* **Closed** — §6.14's argument reaching a family with no
> LINEAGE group. What follows is what the decision was made against.

At the overview level the five Hindu rows do not separate. Measured off the rendered legend:

```
Hindu unspecified / Tamil Hindu      dE  3.6      <- 474,623 and 63,950 people
Tamil Hindu       / Marathi Hindu    dE 15.9
Hindu unspecified / Marathi Hindu    dE 16.5
Telugu Hindu      / Arya Samaj       dE 18.8
```

`check_palette.py`'s own threshold is 25. Telugu and Arya Samaj do separate from the rest;
the top three do not.

**The cause is diagnosed and it is the one `ROOT_BAND` warns about.** Hinduism has no band, so
it falls through to the ±2° default *around its own root hue* — and the family's own dots draw
at exactly that hue. The comment in `index.html` says a family without a band is fine because
such families "have two or three rows each and none of them is ever a large area of colour".
Mauritius breaks both halves of that.

**`hinduism: [28, 44]` was tried and reverted.** It helps inside the family
(unspecified/Tamil 3.6 → 12.7) and costs more elsewhere: 28→44 sits under Catholic's
`OVERVIEW_ARC` gold, putting **Catholic at dE 18.6 from Arya Samaj, 19.2 from Marathi and 21.8
from Tamil** — and Catholic is 24.9% of Mauritius. One within-family collision traded for
three cross-family ones in the same country.

The warm end is full (Sikhism 0, Ravidassia 17, Hinduism 23, ancient communions 16→26,
Catholic's arc, Christianity 50) and Hinduism is 1.21bn dots in India, so it cannot move
quietly. The three alternatives — flatten the family into a LINEAGE group, move it to a free
arc, or separate the five on tier rather than hue — are priced in `index.html` beside the
reverted band and **none was taken**.

## 9. Ethics (§14)

**Raised rather than settled, and it is Anita's per §14's opening line.** Nothing here is a
persecuted-minority case: Statistics Mauritius asks religion, publishes these categories at
this exact geography itself, and §14.4's *"no resolution finer than the state's own
publication"* is satisfied by construction.

What is worth naming is that **four of the five Hindu cells are ethno-linguistic communities**
— Marathi, Tamil, Telugu and, by residue, the Bhojpuri-descended majority — and Mauritian
politics has run along exactly these lines since independence, with communal arithmetic
written into the electoral system's Best Loser mechanism. Drawing them at village level maps
the country's own political fault lines at high resolution.

Three things make that acceptable here rather than merely arguable. The state publishes it
itself, at this tier, as a religion question rather than an ethnicity one — so this is not
§14.5's derive-religion-from-ethnicity route. The communities are not persecuted minorities;
they are the constituent parts of the majority. And the map draws what the census asked, with
no inference added.

## 10. Not done

* **D5's ~60 named bodies are not ingested.** They have three units and would need §3.10-style
  allocation against D6 to reach any geography; that is a real piece of work and would make
  Mauritius the deepest Christian source in the southern hemisphere.
* **Ahmadi Muslims are in D5 and not in D6**, so they are inside `islam` here.
* **Agalega** (~330 people) is excluded by Statistics Mauritius from the published tables and
  is therefore absent. Saint Brandon has no resident population.
* **The 2022 census is current.** The previous one was 2011.
