# Kazakhstan — BNS, National Population Census 2021. **Religion by oblast, counted**

Wired 2026-09-07 as a **modelled** country; **redrawn from counts 2026-09-08**. 19,186,015
people, **17 regions**, 9 drawn nodes, **100.00% drawn**, basis `counted`.

| | |
|---|---|
| figures | Bureau of National Statistics, **NPC 2021**, religion × oblast, from the census dashboard's **Qlik engine** at `qap.stat.gov.kz` (app `4c82a5bb…`) |
| checked against | the printed volume **«Национальный состав, вероисповедание и владение языками»** (Astana 2023), **chapter 12**, whose nine national totals it reproduces to the person |
| basis | `counted` — BNS's own cross-tabulation, no tier flag on any row |
| geography | **17 regions**, 14 oblasts + Astana, Almaty, Shymkent; ~1.13M people each |
| drawn | **19,186,015 people, 100.00%** — every answer Question 11 offers is on the tree (§4) |
| licence | BNS publications and the engine both open, no registration, no terms gate |

**This country was modelled for one day and that is the interesting thing about it.**
`sources.md` §11u established four ways that BNS publishes religion nationally and nowhere
else, and every one of those checks was correct. All four were about **documents**. The
figures were in the census dashboard, which is backed by the microdata and will cross-tabulate
on request. The model is kept in `sources/kz_model.py`, where it is now scored rather than
drawn; sources.md §9cd has the route and spec §14.25 has what the score means.

---

## 1. Where the numbers come from

`stat.gov.kz/ru/national/2021/` links eleven volumes and **two interactive dashboards**. The
dashboards are Bitrix tab strips whose thirteen tabs each hold an iframe onto a Qlik Sense
sheet; the religion tab is `28443` and the sheet is `063175c3-9502-4a72-a506-69ec3d5f3a99`.

**A Qlik app ships its data model, not its charts.** The published sheets show what BNS chose
to show. The engine underneath answers questions about everything the app loaded, and this app
loaded the census:

```
    table `Население 2009_2021`   35,195,612 rows = 16,009,597 (2009) + 19,186,015 (2021)
    one row per enumerated person, keyed HASH_IIN, 124 fields, among them
        Вероисповедание · Область · КАТО РАЙОН · Тип местности · Национальность · Возраст
```

Anonymous access is enabled: the socket's first frame is `OnAuthenticationInformation` with
`mustAuthenticate: false`. `sources/kz.py` has the whole route in about forty lines, and the
practical notes are in sources.md §9cd (ask for the field list before writing a query; filter
with set analysis rather than a selection).

## 2. The four margins, which are the proof

`kz.py` asserts the first three on every run and refuses to write the CSV if any fails.

1. **all nine national religion totals reproduce volume ch.12 to the person** — Ислам
   13,297,775, Православие 3,269,143, Католицизм 18,988, Протестантизм 9,419, Иудаизм 7,192,
   Буддизм 15,458, Другое 23,247, Отказались указать 2,112,653, Неверующие 432,140;
2. they sum to **19,186,015**, with no residual;
3. urban and rural are **11,741,342 / 7,444,673**, the volume's own constants;
4. the two censuses' row counts are their published populations.

And the fifth, which is what makes it conclusive rather than merely consistent: it **differs
regionally** from the ethnicity model by 7.50% of the country. A table that matches the
published national figures to the person while disagreeing with everything derivable from them
is their source and not a derivative of them.

## 3. The join is on population, and that is not fussiness

The engine labels oblasts in caps and its `ОБЛАСТЬ КАТО` field is another name string rather
than a code, while the geography is keyed on KATO. **A name join matches sixteen of seventeen
and fails silently on the capital**, which the census enumerated as Nur-Sultan and the
dashboard calls Astana; no total would move.

So `kz.py` asserts the seventeen oblast populations **distinct** — which is what turns a
population match into a proof — joins on the population to the person, and only then checks
the names, allowing exactly the one documented rename. `[[reference_name_join_wrong_neighbour]]`
is the general case.

> **The Latin/Cyrillic fold is kept even though this route does not need it.** BNS types the
> ethnos workbook's German column as **`Hемцы`** with a Latin `H` (U+0048), not Cyrillic `Н`
> (U+041D). A dict lookup on the name misses Germans and only Germans, and every printout
> looks identical. `fold()` is still in both files because `kz_model.py` needs it and because
> **any Cyrillic or Greek source can do this.**

## 4. The 11.01% who declined, and why they ARE drawn

**Anita, 2026-09-07:** *"is there any way we could try to draw the 11% who refused to state?"*
The census form settles it. Question 11 of `Переписной лист 3-И`:

```
    11. Укажите Ваше вероисповедание (религию)
        1. Ислам                    5. Другое (укажите) ______
        2. Христианство             6. Отказываюсь указать
           2.1 Православие          7. Неверующий
           2.2 Католицизм
           2.3 Протестантизм
        3. Иудаизм
        4. Буддизм
```

**`Отказываюсь указать` is option SIX: printed, numbered, first person, and chosen by
2,112,653 people out of seven offered.** It is not a blank, not item non-response, not an
enumerator's residual code, and that is exactly what separates it from Trinidad's derived
`Not Stated` (tt2011.py, the case §3.5 is written about). An answer that was offered and
picked is an answer. So it goes on `unknown` and Kazakhstan is **100.00% drawn**. It is
drawn in its own colour and nobody is redistributed, which is §3.5 satisfied rather than bent.

### And the counts changed what this layer means

Modelled, refusal came out **almost flat, every region between 9.35% and 12.19%**, and this
file said at the time that the flatness was the model's and not Kazakhstan's. It was.
Counted:

| | | | |
|---|---:|---|---:|
| Mangystau | **22.07%** | Jambyl | 6.94% |
| Shymkent | 21.29% | Turkistan | 5.76% |
| Almaty city | 19.94% | Aktobe | 4.33% |
| Almaty region | 19.41% | North Kazakhstan | 4.05% |
| Akmola | 14.97% | West Kazakhstan | 3.92% |
| Kostanay | 13.92% | Pavlodar | 2.68% |
| Karaganda | 11.05% | Kyzylorda | 1.60% |
| Atyrau | 10.88% | East Kazakhstan | **1.19%** |
| Astana | 7.92% | | |

**An eighteenfold spread on one census form**, and it is the largest range of any layer here.
It does not follow the north-south line; it barely moves between town and country (11.84%
urban against 9.71% rural); and it does not follow ethnicity — Mangystau is one of the most
Kazakh oblasts and Kazakhs decline at 9.3% nationally, while Turkistan, which has the large
Uzbek population that declines at 20.4%, comes in at 5.76%.

**Nothing BNS publishes explains it.** A range that wide between neighbouring administrative
units is as consistent with fieldwork practice as with reticence, and `note_public` says that
rather than choosing a story. Two things that are true and do not add up to an explanation:
Kazakhstan requires religious groups to register, refuses registration to Jehovah's Witnesses
and to Ahmadi Muslims, and prosecutes unregistered worship; and refusal is much higher among
the non-Kazakh Muslim minorities than among Kazakhs (Azerbaijanis 27.3%, Kurds 26.7%, Turks
25.4%, Dungans 24.3%, Uyghurs 23.4%, Tajiks 22.6%, Uzbeks 20.4%, against Kazakhs 9.3% and
Russians 7.5%).

**Palette note:** `check_palette.py` reports two pairs under the usual 25 here, and both are
inside the grey and residual family rather than between religions: **`unknown` / `secular` at
dE 21.0** and **`secular` / `other` at dE 19.8**. Accepted on that family's own documented
terms — its floor is about 15, not 25, because a mix-up between two neighbours on one spectrum
is a small error where reading either as a religion is not. `unknown` and `secular` also
separate warm against cool (hue 51 against 204), and `unknown` clears every religion
Kazakhstan draws by 71–82. **Unchanged by the 2026-09-08 rebuild**: the counts moved and the
palette did not, and dropping the `modelled` tier only means the same colours now draw at full
saturation instead of desaturated.

## 5. What the map says

**The north-south split is the country.** North Kazakhstan region is **55.16%** Christian and
38.72% Muslim; Turkistan, on the Uzbek border, is **1.57%** Christian and 92.38% Muslim. Those
are the two ends. Islam runs from **96.21%** of Kyzylorda to **36.95%** of Kostanay, which is
the least Muslim region in the country. The line is the Slavic settlement of the northern
steppe, first under the Empire and again under the Virgin Lands campaign of the 1950s.

**Christianity is split three ways, which most censuses here do not manage.** 99.14% of
Kazakhstani Christians are Orthodox. The 18,988 Catholics sit where the deportations left
them, 0.92% of North Kazakhstan region and 0.52% of Akmola against 0.00% of Turkistan; 63.5%
of them are ethnic Germans and Poles, which is why Karaganda has a cathedral. **82.2% of the
country's 15,458 Buddhists are Koreans**, from the 1937 deportation of the Soviet Korean
population out of the Far East.

**Non-belief is small and northern.** 432,140 people, 2.25%, from **4.84%** of Kostanay down
to **0.24%** of Turkistan.

## 6. Do not read the Russian border

Kazakhstan looks vastly less secular than Russia 200 km away — **1.86%** non-believers in
North Kazakhstan against **52.13%** of Omsk reporting no religious institution — and almost
all of that cliff is the questionnaire. Russia's Arena survey offers *believes in God,
professes no religion*, which **24.93%** of Russians choose and which Kazakhstan's census does
not offer at all. **85.3%** of Kazakhstan's Russians are recorded Orthodox against **43.17%**
of Russia's population. Russia's largest non-institutional answer has not vanished at the
border; it is inside Kazakhstan's Orthodox count, and the 11% who declined are the rest of it.

Five of those figures reproduce from `data/normalized/kz.csv` and `data/normalized/ru.csv`.
The sixth, the 85.3%, is the engine's own religion × `Национальность` cut and is not in either
CSV; it is 2,542,994 Orthodox of 2,981,946 ethnic Russians, and the same query gives Kazakhs
89.2% Muslim, the 82.2% Korean share of the Buddhists and the 63.5% German-and-Polish share of
the Catholics quoted above.

## 7. 17 regions, and the 218 that are now available

The engine returns religion × `КАТО РАЙОН` for **218 units** summing to 19,186,015, cached in
`data/raw/kz/kz_qlik_rayon_religion.csv`, and the same app carries the rayon **boundaries**
(table `карта_район`, 190 rows with a `Район.Line` geometry) on its own key.

**Both of §9aq's reasons for refusing 218 are gone.** It is no longer 218 units of pure
inference, and there is no 218-way fuzzy transliteration join, because the source ships the
geometry beside the data. What is left is a real geography build: extract the boundaries,
rebuild the Kontur hex grid, redo the lookup, retile. That is a session's work and was left
for one rather than bolted onto this. **`queue.md` carries the row.** Whoever takes it should
note that the 218 rayon labels are bare numbers and the oblast is a separate dimension, so the
unit key is the pair, and that `карта_район` has 190 rows against the data's 218 units.

At ADM1 the boundary file is the 2023 COD-AB, which ships 20 polygons because Kazakhstan
created Abay, Jetisu and Ulytau in 2022; `kz_geo.py` dissolves the three pairs back to the
census's 17. See `sources/kz_geo.md` §1.

## 8. Vintage and licence

Enumerated 1 September 2021; the volume is Astana 2023 and the engine serves the same round.
The three oblasts created in 2022 postdate both. BNS publications carry a Civil Code notice
requiring attribution when the statistics are reused, and no other restriction; the engine has
no terms gate of its own. Same footing as the other census offices here, and worth re-checking
before any commercial use ([[reference_poster_commercial_licences]]).

---

## 9. Review, 2026-09-08 — the provenance holds, one figure was wrong, and the refusal layer leans

Second pass by a reviewer who read the caches and the printed volume rather than this file's
account of them. Everything below was recomputed from `data/raw/kz/*` and
`data/normalized/kz.csv` without running `sources/kz.py`.

### What was confirmed

**The nine national totals are the printed volume's, checked against the PDF and not against
`kz.py`'s constants.** `kz2021_ethnic_religion_language.pdf` is 542 pages and its page 506 is
table 12, *Население по вероисповеданию и национальности*. Every figure in `PUBLISHED` is
there as printed, including the volume's own typo-tight `13 297775`, and христиан 3,297,550 is
exactly православие + католицизм + протестантизм, so the three drawn branches are an exact
partition of the Christian parent with no residual. §2's four margins all reproduce.

**The rayon cut is the discriminating test and it passes.** A re-served published aggregate
could not produce it: the engine returns 1,851 non-empty (rayon, religion) cells over **218
units**, they sum to 19,186,015, they reproduce every one of the 153 (oblast, religion) cells
of the coarse cut exactly, and 384 of them are under ten people with a minimum non-zero cell
of **one**. BNS prints religion at no geography at all, so there is no aggregate this could be
a re-serving of. It is the microdata.

**The population join could not have paired a wrong neighbour.** The seventeen oblast
populations are distinct, and the closest pair is Atyrau 673,601 against West Kazakhstan
675,655 — **2,054 people apart**, on a join made on exact equality. The meta file's
`oblast_pop` (a cube over `Область` alone) equals the sum over religions in every oblast, so
no row carries a null religion and the join denominators are the same number twice.

**Spec §14.25's arithmetic reproduces**, table and all: 313,745 held out (1.635%), 1,439,367 at
oblast level (7.502%), ratio **4.59**. Both use the same halved-absolute-difference convention,
so the comparison is between like things.

**All 26 reader-facing figures reproduce**, in `note_public` and in §§4–6 here, including the
six Russia ones: 24.93% and 43.17% are the 79 drawn subject rows of `ru.csv` (Arena's own
national column says 25.16% and 42.96%, which is the same thing seen through a different
weighting), 52.13% of Omsk is its 39.13% *believe in God, profess no religion* plus its 13.00%
*do not believe in God*, and 85.3%, 89.2%, 82.2% and 63.5% are all on volume page 506 as well
as in the engine. Both superlatives are right on the base the map draws: North Kazakhstan and
Turkistan are the most and least Christian of the seventeen, and 1.19% is the actual minimum of
the refusal range.

### The one figure that was wrong, now fixed

§4 said refusal was **12.18% urban against 9.84% rural**. Those are refusal over
`total − неверующие`, a denominator nothing else in this country uses; the plain shares are
**11.84% and 9.71%** and the file now says so. The conclusion it supports is unaffected — the
gap is 2.13 points instead of 2.34 — but the numbers were not the ones they claimed to be.

### The §3.5 lean check, which nobody had run here

Kazakhstan excludes nobody, so there is no dropped residual to correlate. The live form of the
question is whether the drawn `unknown` layer, running eighteenfold across the oblasts, eats
evenly out of the religions. Correlating each oblast's refusal share against each category's
share **of its answering population**, n=17, permutation p over 20,000 shuffles:

```
    Неверующие      rho +0.54   p 0.027      <- the only one
    Иудаизм         rho +0.39   p 0.12
    Протестантизм   rho +0.17   p 0.51
    Ислам           rho -0.01   p 0.97
    Православие     rho -0.04   p 0.87
```

**Refusal is not disproportionately Muslim or Orthodox in any detectable way**, which is the
reading a reader is most likely to reach for and it is not supported. The one lean that shows
is toward **non-belief**, and it is the same sign as Serbia's §3.5 finding (+0.60 against the
declared-atheist share). Say it at its real strength and no higher: **rho +0.54 at n=17 is one
result out of eight tests and does not survive a correction for that**, so it is a hint, not a
finding, and this file's own instruction to resist supplying an explanation the data does not
carry applies to it too. It is recorded because it is the direction to test first if the 218
rayons are ever built, where n is thirteen times larger.

### The refusal layer does move a superlative, on a base the map does not draw

Because refusal is a flat proportional dilution of everything else in an oblast, a region at
22% refusal has all its named shares multiplied by 0.78. That is large next to Islam's own
range, and both ends of *"Islam is highest in Kyzylorda at 96.2% and lowest in Kostanay at
37.0%"* flip if shares are taken over the answering population instead: Turkistan 98.03% at the
top and North Kazakhstan 40.36% at the bottom. The Christian and non-belief superlatives do not
flip. Nothing here should change — every country on this map draws shares of the whole
population and Kazakhstan should not be the exception — but the Islam extremes are a statement
about the drawn base rather than about where Islam is, and the Christian ones are robust either
way.

### For the 218-rayon question, which stays Anita's

Not an ask, per §7's own instruction. Three measurements that bear on it:

- **The finer cut individuates far more than the oblast one does.** 218 rayons, median 41,898
  people, **45 of them under 20,000** and the smallest 5,468. In that smallest rayon the engine
  returns 2 Protestants and 2 Buddhists by name. There are **240 cells of one to four people**
  across the country — 64 Jewish, 65 Buddhist, 58 Protestant, 39 Catholic. At 17 oblasts the
  smallest drawn cell of any religion is in the hundreds.
- **The map itself would not print those numbers.** At 1:1,000 a four-person cell draws no dot,
  and §4.3 gives no ring either, because Protestantism and Judaism both clear a dot elsewhere in
  Kazakhstan and rings are one per country. So the exposure is the unit size and the file, not a
  mark a reader can point at.
- **`карта_район` has 190 rows against the data's 218 units**, so 28 units would need their
  geometry from somewhere else, and that is where a name join would have to be reintroduced —
  the one thing this country was careful to avoid.
