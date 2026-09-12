# Serbia — RZS, Попис становништва 2022

Wired 2026-09-05. 6,647,003 people, 168 municipalities, 11 drawn categories.

| | |
|---|---|
| source | Републички завод за статистику, Census 2022, workbook `6_stanovnistvo-prema-veroispovesti.xlsx` |
| basis | `self_id`, voluntary question |
| geography | 168 units — opštine and gradovi, with Belgrade as its 17 city municipalities and Niš as its 5 |
| categories | 13 published, 11 drawn |
| drawn | **6,122,033 people, 92.07%** |
| licence | RZS open data, free reuse with attribution |

**No API and no hunt.** The census results portal has a page of Excel tables and religion
by municipality is one of them, at 83 KB. The boundaries were already on disk from North
Macedonia. What Serbia costs instead is *reading the sheet correctly*, and there are three
traps in it, all of which pass every national check while being wrong.

---

## 1. Where it is

`https://popis2022.stat.gov.rs/sr-latn/popisni-podaci-eksel-tabele/` lists every published
census table as a direct `.xlsx` under `/media/<id>/`. The religion one is:

    https://popis2022.stat.gov.rs/media/31329/6_stanovnistvo-prema-veroispovesti.xlsx

83 KB, one sheet (`OpstinePol`), 620 rows × 18 columns, TLS verifies with no special
handling. **`www.stat.gov.rs` and `data.stat.gov.rs` are a different matter** — both fail
certificate verification from this machine and `data.stat.gov.rs` is a server-rendered
ASP.NET dissemination app whose only data route is `/Home/Result/<datasetId>`, so the
static workbook is the right door and not a workaround. The portal has two other religion
tables and neither is finer: `2_stanovnistvo-prema-nacionalnoj-pripadnosti-i-veroispovesti`
is religion × ethnicity **by region** (4 units), and there is nothing at settlement level.
Municipality is RZS's ceiling for religion, as it is for Croatia and North Macedonia.

## 2. The sheet is five levels deep and the sixth one is inside the finest

Column 0 holds the geography and nothing marks the level — no code, no indentation, no
level column. The nesting is Republic → **Srbija-sever / Srbija-jug** → 4 regions →
25 oblasti → municipalities, and summing the sheet as delivered counts Serbia five times.
That much is India's C-01 trap and Hungary's WBS003 trap, and it is caught the usual way.

**The one that is new is the sixth copy, and it hides in the finest tier.** Four rows
inside the municipality level are themselves parents:

| row | total | its own children |
|---|---|---|
| `Grad Niš` | 249,501 | Crveni krst, Medijana, Niška Banja, Palilula, Pantelej |
| `Grad Požarevac` | 68,648 | Požarevac, Kostolac |
| `Grad Užice` | 69,997 | Užice, Sevojno |
| `Grad Vranje` | 74,381 | Vranje, Vranjska Banja |

462,527 people, counted twice, at the level you would draw. And there is **no structural
marker at all**: after Niš's five city municipalities the next row is Aleksinac, an
ordinary municipality of the same oblast, in the same column at the same indent. The
`Grad ` prefix is the only hint and it is a name, not a field.

**The test is arithmetic.** `_nest()` takes the consecutive rows after a `Grad X` until
their totals sum to it exactly, and refuses to continue if they do not. That both resolves
the nesting and checks that the sheet parsed — a misread column would fail it immediately.
Belgrade is the same shape one level up: it is an *oblast* (`Beogradska oblast (Grad
Beograd)`) whose 17 children are city municipalities.

Once the four parents are dropped the drawn tier is 168 rows summing to 6,647,003 exactly.

## 3. Palilula is two places

Belgrade has a Palilula (69,113 people) and so does Niš (69,811), and **the source
publishes no codes of any kind** — only names, as Romania and Ghana do. Resolving on the
bare name gives one of them both, and every national and oblast total still reconciles.

Ghana's `TMA` again, and worth noting that it looks *less* dangerous than Ghana's: an
acronym invites suspicion and a real place name does not. The fix is the same in shape and
better in kind — `sources/rs_geo.py` computes which bare names repeat instead of listing
them, so a name that starts colliding in a future census is caught by the same code rather
than by somebody noticing.

## 4. Kosovo is in the sheet, and it is empty

`Регион Косовo и Метохија` is present as a region row whose every cell is `...`, RZS's
symbol for data not available, because the 2022 census did not enumerate it. It carries no
sex breakdown either, so the natural row filter — keep the rows marked `с`/`t` — drops it
silently.

It is kept and asserted on instead. A source that publishes a unit as **empty** is saying
something different from a source that omits it, and the assertion is that there is
**exactly one** such row and that it is a region: if RZS ever publishes Kosovo, or ever
suppresses something else, the run fails rather than quietly changing what is drawn.

## 5. The Christian parent is a duplicate, not a parent with a remainder

`Хришћанска / свега` (5,758,719) sits beside its four children — Orthodox, Catholic,
Protestant, Other Christian — and equals their sum **in all 204 rows**, checked per row and
not nationally (§12). So it is Ghana's case: drop it. Drawing it would double 5.76 million
people; Hungary's version of the same shape needed a 77,629-person remainder emitted
instead, and the two are indistinguishable from the table of contents.

## 6. The two things not drawn, and which way they bias the map

| | people | share | where it peaks |
|---|---|---|---|
| `Нису се изјаснили` — did not declare | 169,486 | 2.55% | Dimitrovgrad 10.2%, Subotica 10.1%, Sombor 9.2%, Bački Petrovac 7.2% |
| `Непознато` — unknown | 355,484 | 5.35% | Savski venac 17.3%, Stari grad 14.4%, Vračar 11.3%, Zvezdara 9.6% |

RZS keeps them apart and so does this file. Article 47 of the Serbian constitution says
nobody is obliged to declare a religion, so the first is a refusal (§3.5) — and its
geography is the ethnic-minority towns of Vojvodina, where declaring anything carries the
most weight and the ethnicity question gets the same treatment.

**The second is a completely different pattern and it is the one worth knowing about.**
`Непознато` is a *central Belgrade* phenomenon: 17.3% in Savski venac against 0.97% in
Preševo and under 1.7% across rural central Serbia. Across the 168 municipalities it
correlates **+0.60 with the declared-atheist share** and only +0.28 with the refusal share.
So it is not a group hiding inside a residual — it is missingness concentrated among the
young, urban and secular.

**Which means excluding it is not neutral, and this is the general point.** §3.5 says
undercounting is marked rather than filled, and Serbia adds the half that was missing:
*say which way it leans*. Dropping `Непознато` removes proportionally more people from the
least religious municipalities than from the most, so every share drawn here is slightly
more religious than Serbia is. Nothing corrects for it — correcting would be inventing a
magnitude (§14.4) — but `note_public` says it out loud.

## 7. The categories, national

| source category | people | % | node |
|---|---|---|---|
| Total | 6,647,003 | 100.00 | *universe* |
| Christian - All | 5,758,719 | 86.64 | *duplicate of its four children* |
| Christian - Orthodox | 5,387,426 | 81.05 | `christianity.orthodox.canonical` |
| Unknown | 355,484 | 5.35 | *excluded* |
| Islam | 278,212 | 4.19 | `islam` |
| Christian - Catholic | 257,269 | 3.87 | `christianity.catholic` |
| Did not declare | 169,486 | 2.55 | *excluded* |
| Not believers (atheists) | 74,139 | 1.12 | `unaffiliated` |
| Christian - Other Christian | 59,346 | 0.89 | `christianity.other` |
| Christian - Protestant | 54,678 | 0.82 | `christianity.protestant` |
| Agnostics | 8,654 | 0.13 | `secular` |
| Eastern religions | 1,207 | 0.02 | `other.rs` |
| Judaism | 602 | 0.01 | `judaism` |
| Otherreligions | 500 | 0.01 | `other.rs` |

The header cell for `Otherreligions` really is written unspaced in the workbook; the key
travels as RZS wrote it (§2.4). The Christian columns are keyed `Christian - <child>`
because the sheet's header is two rows deep and the child cell alone reads `All`,
`Orthodox`, `Catholic` — labels that mean nothing without their family.

Every category is `measured`. Nothing is allocated, because there is nothing to allocate
from: RZS publishes these categories at this geography and at no other, which is §3.9's
trade made by the office. Croatia's file, one country over, has the same shape.

## 8. What the map shows

**81% Orthodox, and the whole interesting map is in the other fifth**, in two places.

*Vojvodina is the old Habsburg side of the border and still reads that way.* Kanjiža 85.3%
Catholic, Senta 74.2%, Ada 72.2%, Subotica 48.2%, Čoka 45.0%, Bačka Topola 44.7% — the
Hungarian towns along the Tisza. And **Bački Petrovac 57.3% Protestant, Kovačica 41.4%**,
which are the Slovak Lutheran colonies settled in the 1740s, two dark spots in an otherwise
Orthodox province. The census offers one Protestant cell so the tree cannot say Lutheran;
the geography says it instead.

*The Muslim map is two places 300 km apart.* The Sandžak is Bosniak — Tutin 93.5%, Novi
Pazar 82.9%, Sjenica 78.3%, Prijepolje 46.8%, Priboj 21.8% — and the Preševo valley is
Albanian: Preševo 94.2%, Bujanovac 68.8%, Medveđa 14.7%. **Tutin (2.0%) and Preševo (4.5%)
are the two least Orthodox municipalities in Serbia.**

*Irreligion is 1.25% and is four Belgrade municipalities.* Stari grad 7.7% atheist or
agnostic, Vračar 6.2%, Savski venac 5.4%, Novi Beograd 4.7%, against 0.32% in Lazarevac.
That is very low by this map's standards — Czechia is twenty points higher.

*And 602 Jews in the country*, 78 of them in Stari grad and 66 in Novi Sad.

One pattern this file cannot explain: **`Other Christian` runs 5–7% in a band of rural
western and central Serbia** — Ljubovija 7.1%, Arilje 7.1%, Malo Crniće 6.9%, Žitorađa
5.5%, Lučani 5.4% — against 0.89% nationally. The category is a published catch-all and
RZS says nothing about what is in it, so this is recorded as a fact about the map and not
explained. Anyone who knows what those places are should say.

## 9. Not done

- **The religion × ethnicity table exists at region level** (4 units) and is not ingested.
  It would say how much of `Orthodox` is Serb against Romanian and Vlach, and how much of
  `Catholic` is Hungarian against Croatian — at a geography too coarse to draw and useful
  only as a note.
- **Nothing separates the Protestant cell.** Slovak Lutheran, Hungarian Reformed and the
  Nazarenes are all in it and the geography is suggestive, but §12's Philippine rule says
  co-location audits a mapping and never builds one.
- **Kosovo is not drawn.** It has its own 2024 census with a religion question, run by ASK
  in Pristina; that is a separate source, a separate `source_id`, and a §14 conversation
  before anything else.
