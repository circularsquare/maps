# Croatia — DZS, Popis stanovništva 2021

Ingested and drawn 2026-09-03. Rebuild with `python sources/hr.py --fetch`.

12 categories over **555 towns and municipalities plus Zagreb's 17 gradske četvrti** —
572 units for 3.87M people. Reconciles exactly at every level: DZS neither rounds nor
suppresses.

**Croatia is drawn shallow by choice, not by necessity.** That is unusual here and §4 is
the important section of this file.

---

## 1. The file, and finding it

One XLSX, 18.3MB, no login, no bot protection:

```
https://podaci.dzs.hr/media/td3jvrbu/popis_2021-stanovnistvo_po_gradovima_opcinama.xlsx
```

**The data portal is a dead end and the old site is not.** `podaci.dzs.hr` is a JavaScript
navigation shell; every guessed PxWeb path (`/pxweb/api/v1/hr/`, `/px/`, `/api/`) returns
404, and the census page under it lists no files. The actual downloads are linked from the
*old* site's feature page:

```
https://dzs.gov.hr/naslovna-blokovi/u-fokusu/popis-2021/88
```

which carries sixteen `podaci.dzs.hr/media/<hash>/...xlsx` links with opaque hashes. This
is now the third office where the modern portal is a shell over the real files (with
`data.gov.sk` and `data.stat.gov.rs`) — see spec §12.

The workbook has 24 sheets. Two carry religion:

| sheet | what | used |
|---|---|---|
| **`2.`** | **STANOVNIŠTVO PREMA VJERI** — population by religion, 607 rows × 31 cols | **yes** |
| `5.` | OSOBE KOJE SU SE IZJASNILE KAO KRŠĆANI ILI VJERNICI, PREMA VJERSKOJ ZAJEDNICI — **54 named churches**, 2,466 rows × 64 cols | **no — see §4** |

### Every count column is followed by its own percentage

`Katolici` is column 7 and `Katolici, %` is column 8, for all twelve categories. Taking the
wrong one of each pair produces a map where every unit has about 100 people in it and
nothing else complains. `hr.py` lists the count columns explicitly.

### Headers are bilingual in one cell

`Katolici Catholics`, `Ostali kršćani1) Other Christians1)` — Croatian and English
concatenated, footnote markers included. The Croatian half is kept as the mapping key, and
`Ostali kršćani1)` **keeps its `1)`**, because the key in `taxonomy/hr2021.py` has to match
what the normaliser writes rather than a tidied version of it (§2.4).

## 2. Grad Zagreb is not in the municipality list

DZS publishes the capital **only as its 17 gradske četvrti**. So the census has 555
municipalities where Croatia has 556, and

```
555 municipalities  3,104,702
 17 Zagreb districts  767,131
                    ---------
            572 units 3,871,833  = the published national total, exactly
```

Checking `municipality` against the national total therefore fails by exactly Zagreb, which
is the correct answer to the wrong question. **The complete cover is the two levels
together**, and `hr.py` asserts that rather than either level alone.

This is the first source in the project that has already done the city-district
substitution itself. Czechia and Estonia publish both levels and leave the choice to us;
Croatia has made it.

## 3. Reconciliation

```
municipality    555 units   3,104,702      partial by design
city_district    17 units     767,131      partial by design
DRAWN COVER     572 units   3,871,833      exact
county           21 units   3,871,833      exact
country           1 unit     3,871,833      exact
categories partition the total in all 594 units — 0 mismatched
```

No rounding, no suppression, no sentinels. `-` is a true zero and is the only non-numeric
value in the count columns.

## 4. THE REAL SOURCE IS SHEET 5, AND IT IS NOT INGESTED

Sheet `5.` names **54 individual churches and religious communities** at the same
geography as sheet 2. It is the largest available upgrade for Croatia and the reason the
drawn map should not be taken as the country's ceiling. What it contains:

- **Four Orthodox jurisdictions kept apart** — Serbian, Macedonian, Montenegrin and
  Bulgarian Orthodox Churches in Croatia. Almost nothing else in the project separates
  Orthodox jurisdictions at all.
- **Eleven separate Jewish communities**, by city — Zagreb, Split, Rijeka, Osijek,
  Dubrovnik, Čakovec, Koprivnica, Virovitica, Daruvar, Slavonski Brod, and Bet Israel.
  Croatia's whole Jewish population is 573 people, so this is enumeration at the level of
  a few dozen individuals — exactly what §R3 asks for.
- **Two Old Catholic churches**, Croatian and Liberal.
- A dozen Pentecostal, Baptist and Evangelical bodies by name, plus Adventists, Nazarenes,
  Latter-day Saints, the New Apostolic Church, ISKCON, the Dharmaloka Buddhist community,
  the Hindu Religious Community, the Baha'i community, and the Church of Scientology.

**Why it is not simply used instead.** Sheet 5 does not replace sheet 2's partition; it
*refines two of its residual categories*. Each unit gets four rows:

```
Ostali kršćani                        186,960   <- sheet 2's category total
Od toga – kršćani                     180,368   <- of which, by named church
Ostale religije, pokreti i svjetonazori 37,066  <- sheet 2's category total
Od toga – vjernici                     25,957   <- of which, by named church
```

and the 54 church columns are populated only on the two `Od toga` rows. So the named
churches account for the ~224k people in two residual categories, **not** for the 3.06M
Catholics or the 128k Orthodox, who are only in sheet 2.

The confusing part, and the reason this needs care rather than a quick pass: the largest
column on `Od toga – kršćani` is **`Katolička crkva` at 157,388** — people whose *religion*
answer was "other Christian" but who named the Catholic Church as their *community*. Those
are disjoint from sheet 2's 3,057,735 `Katolici`, and adding the two would be wrong while
drawing them as separate nodes would be confusing. Resolving that is a design decision, not
a parsing one, which is why it is deferred rather than guessed.

## 5. What is drawn

12 categories → 9 taxonomy nodes → **3,716 dots and 1 ring** at 1:1,000.

Croatia is 79% Catholic and one node carries 2.49M of the 3.0M drawn people. The map's
content is the remainder: the Serbian Orthodox belt along the Bosnian and Serbian borders
(Lika, Banovina, eastern Slavonia), Muslims in the cities, and Istria, which has by far the
country's largest irreligious and agnostic share.

One new node: `other.hr`, holding `Ostale religije, pokreti i svjetonazori` and —
reluctantly — `Istočne religije`. See the REVIEW note in `taxonomy/hr2021.py`.

## 6. Not done

- **Sheet 5.** §4. This is the one that matters.
- **Zagreb is one polygon.** The data would allow 17; the boundaries were not found. See
  `hr_geo.md` §4 — this is the cheapest outstanding capital fix in the project.
- `Nisu vjernici i ateisti` merges non-believers with atheists into one answer, where
  Czechia, Romania and Estonia ask them apart. It goes to `unaffiliated` whole.
