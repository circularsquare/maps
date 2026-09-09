# Slovenia — SURS, Popis 2002

**Drawn 2026-09-08.** 192 municipalities, 1,964,036 people, 1,514,744 of them on the map.

Files: `sources/si.py`, `sources/si_geo.py`, `sources/si_grid.py`, `taxonomy/si2002.py`.

---

## 1. Why this country was closed twice and was not closed

`sources.md` §11c and §11k both struck Slovenia off, and the queue row said *"correctly"*,
and that was right about what they checked. Slovenia's 2021 census is **register-based and
carries no religion question**, so there is no recent figure at any geography, national
included. Neither sweep was careless: §11k walked the SURS PxWeb catalogue, recorded that it
is live and 908 KB, and still concluded there was nothing here.

**The catalogue it downloaded contains `05W1006S.px`, `Prebivalstvo po veroizpovedi, občine,
Slovenija, popis 2002`.** Ten religion answers on 192 municipalities, from the last
conventional census Slovenia ran. The UNSD oracle's row for Slovenia already said `2002` and
`14 cats`; nobody had matched the oracle's year against the year the sweeps had tested.

That is the whole finding, and it generalises: **a closure is about a TIER, and a country
usually has more than one.** Finland landed on the same shape the same day. What makes it
worth writing down is that the second sweep did not fail to reach the data; it reached the
data, downloaded it, and searched it for the wrong census.

The 2002 census microsite is also still up at `www.stat.si/popis2002/`, with the same table
as a frameset over `si/rezultati_html/OBC-T-06SLO.htm` and as `si/rezultati/OBC-T-06si.xls`.
**WebFetch cannot see either**, because it drops frames; the API is what to use anyway,
because the HTML prints a withheld cell as the letter `z` and gives no way to tell it from a
value.

## 2. The tables

| table | what | where |
|---|---|---|
| `05W1006S.px` | religion x 192 občine, 10 answers | **drawn** |
| `05W1606S.px` | religion x settlement type, **14 answers**, 1991 AND 2002 | check level, national only |
| `05W0405S.px` | population x 6,152 naselja, popis 2002 | the občina CODES, and a population check |
| `05W1002S.px` | ethnicity x 192 občine | used once, see §6 |

All four are `https://pxweb.stat.si/SiStatData/api/v1/sl/Data/<table>`; GET returns the
variables, POST returns json-stat2. No key, no login, no rate limit met.

## 3. The trap, and it is the one that does not announce itself

**The `OBČINA` dimension of the religion table is not Slovenia's municipality code.** It runs
`001`-`193` where `001` is SLOVENIJA and `002`-`193` are the 192 municipalities in
alphabetical order. Slovenia's real codes are alphabetical only for `001`-`147`, assigned in
1994, with `148`-`193` given to the municipalities created in 1998 and slotted into the
alphabet. So Ajdovščina is `002` in this table and `001` everywhere else, Beltinci is `003`
and `002`, and **every unit is one place out with every total still reconciling** —
`[[reference_name_join_wrong_neighbour]]` in its purest form, on a set where the wrong
neighbour is always the alphabetically previous one.

The codes are recovered from `05W0405S.px`, whose `NASELJA` labels are of the form
`001  AJDOVŠČINA` for a municipality row and `001 001 Ajdovščina` for a settlement — the
office's own code, in its own dimension, for the same enumeration. The join is by name, and
**the check is population**: the two tables must agree to the person on all 192
municipalities, and they do. 192 distinct populations cannot all survive a shifted join.

Two names differ between the two tables of the same census and both are SURS's own older
forms: `Destrnik` is printed `DESTERNIK`, and `Sveti Jurij ob Ščavnici` is printed `SVETI
JURIJ`. Each resolves to exactly one remaining code and each is confirmed by the population
check rather than assumed.

## 4. Boundaries: free, for once

Slovenia had **192 municipalities from 1998 until 2006** and has 212 now, all twenty of the
additions carved out of existing units. Eurostat's GISCO **Communes 2001** layer has exactly
192 Slovenian polygons keyed on `NSI_CODE`, which is the official občina code, so the
enumeration's own boundary set is a file already on disk for Austria. No crosswalk is written
and none is needed. The check is set equality against the counts, not a count of rows.

GISCO's `SABE_NAME` writes carons as a preceding `^` (`Ajdov^s^cina`), and the `.dbf` is
cp1250 while the `.cpg` claims otherwise, so pyogrio's default read raises. Nothing joins on
those names; the names on the map are SURS's.

Placement is Kontur 400 m hexes, `kontur_population_SI_20231101`. Census-vs-grid correlation
across the 192 units is **r = 0.9889** against a best of 0.2237 over 500 shuffles. The ratio
band is 0.74 (Razkrižje) to 1.78 (Škofljica, the Ljubljana commuter belt), which is a fact
about twenty-one years of Slovenian internal migration rather than an error term.

## 5. Suppression, and the cell that was NOT reconstructed

`status` separates `z` (zaupno, withheld) from `-` (ni pojava, a true zero). 269 cells are
withheld, holding **2,317 people, 0.118% of the country**, and per spec §3.8 the headline is
the wrong number to report:

| category | withheld | in how many of 192 |
|---|---|---|
| druge veroizpovedi | **4.86%** | 64 |
| pravoslavna | **2.80%** | 79 |
| evangeličanska in druge protestantske | **1.74%** | 66 |
| islamska | 0.47% | 40 |
| katoliška, Neznano | **0%** | 0 |

**In seven municipalities exactly one detail cell is withheld**, which makes it recoverable
by subtraction from the declared subtotal, and `sources/si.py` deliberately does not recover
it. Undoing an office's disclosure control to gain a few dozen people is not a trade worth
making, and it would leave the file doing something the office decided against. The
shortfall is reported per category instead.

## 6. The hole, and which way it leans (spec §3.5)

**22.88% of Slovenia is not drawn**: 307,973 refusals (15.68%), 139,097 never established
(7.08%), and 2,222 in withheld cells of drawn categories. `tools/gap_share.py` computes
22.88% from the universe route and the authored figure matches.

Two causes, and the questionnaire says both. Declaring a religion is voluntary under article
41 of the constitution. And **questions 29 and 30 (ethnicity and religion) had to be answered
by the person themselves, aged 14 or over**, with no household member permitted to answer by
proxy; an enumerator who found nobody in left form P-3/NV and a prepaid envelope. That is
`Neznano`. (`www.stat.si/popis2002/si/kdo_lahko_posreduje.html`.)

**Nobody is missing by age.** Parents answered for children under 15, the blocks partition
1,964,036 exactly, and `Neznano` at 7.08% is well under the under-15 share, so there is no
Peru-shaped second hole here and `gap_share` is the whole of it.

The lean, over the 192 municipalities, correlating each unit's excluded share of its
population against each drawn category's share of that unit's drawn base:

| | |
|---|---|
| Je vernik, ne pripada nobeni veroizpovedi | **+0.553** |
| Ni vernik, ateist | **+0.384** |
| druge veroizpovedi | +0.286 |
| pravoslavna | +0.170 |
| islamska | +0.138 |
| **katoliška** | **+0.043** |
| **evangeličanska in druge protestantske** | **−0.466** |

So the map slightly understates the two irreligious answers and overstates Protestantism, and
Catholicism is almost exactly neutral. Nothing corrects for it (§14.4).

**And the hole is about the religion question rather than about the census.** The same form
asked ethnicity under the same voluntary, answer-for-yourself rule, and `05W1002S.px` puts
that non-response at 48,588 refusals plus 126,325 unknown, **174,913 people, 8.9%**, against
22.76% for religion. Two questions, one form, one population, and 2.6x the non-response on
one of them. That comparison is not in `si.csv` and is therefore kept out of `note_public`
and recorded in `countries.py`'s internal `note` instead.

## 7. What Slovenia does not publish, in case anyone comes back

- **Religion below občina.** The 2002 naselje tables (`05W04xx`, `05W05xx`, `05W06xx`) cover
  age, education, activity, migration, households, families and dwellings. There is no
  religion table at naselje and there is no reason to expect one: the municipality table is
  already suppressing four-fifths of its small cells.
- **Religion recalculated to the 2007 municipalities.** SURS republished a good deal of Popis
  2002 on the 210-municipality set (`0558xxx`), and religion is not among it. The 2002 set is
  the only geography this answer has.
- **1991 below the country.** `05W1606S.px` carries 1991 by settlement type only; the 1991
  municipality tables in the catalogue (`0556501S`-`0556507S`) are age, ethnicity, education,
  activity, households and families, with no religion among them.
- **A newer survey.** Slovenia is in the European Social Survey for every round, which would
  give a modern national figure at NUTS-2, and Slovenia's NUTS-2 is **two regions**. 192
  municipalities at twenty-four years old beats two regions at current, and the census is
  counted rather than sampled. Not attempted.

## 8. Licence

The XLS carries *"Uporaba in objava podatkov dovoljena le z navedbo vira"* — use and
publication permitted with attribution to the source. Attribution only; nothing to raise.

## 9. Review pass, 2026-09-08

A second agent read `sources/si.py`, `taxonomy/si2002.py`, `data/normalized/si.csv` and the
`countries.py` entry against the raw json-stat2 on disk, re-did the join from scratch, and
recomputed every reader-facing figure. Nothing was changed. Five findings.

**The join is right, and the trap is bigger than §3 above says.** Re-derived independently
from `05W1006S.json` and `05W0405S.json`: the `OBČINA` dimension is `001`-`193` contiguous
with `001` = SLOVENIJA, the settlement table's embedded codes are the official ones (spot
checks: `011` Celje, `050` Koper, `061` Ljubljana, `070` Maribor, `133` Velenje, all
correct), all 192 units matched by name, all 192 populations agreed to the person, no code
was claimed twice, and the national total came back 1,964,036.

But **the naive join is not "one place out"**. It is one place out for exactly two units and
arbitrary after that, because the 1998 municipalities carry codes `148`-`193` while sitting
in their alphabetical position in the table: `004` is Benedikt here and Bohinj officially,
and Benedikt is really `148`. Counted directly, using the `OBČINA` value as the official code
**mis-assigns 190 of 192 municipalities**. The phrase "shifts every unit one place" appears
in `sources/si.py`'s docstring, in `countries.py`'s internal `note`, in §3 above, in
`sources.md` §9cg and in the `queue.md` row, and it understates the trap in all five: a
future reader who believes it might think spot-checking two units proves the join. The true
figure is stronger and is the one worth carrying.

Two smaller corrections to §3's proof as stated: the 192 municipality populations are **191
distinct, not 192** (3,640 appears twice), and the settlement-table code list runs `001`-`193`
with `145` absent. Neither weakens the join, which is checked per unit and guarded against
crossing, but the sentence "192 distinct populations cannot all survive a shifted join" is
not quite the proof it claims to be.

**Two figures in `note_public` read a withheld cell as a zero.** `Ni želel odgovoriti` is
suppressed in exactly two municipalities, Hodoš and Gornji Petrovci, and those are the two
the note names: *"It runs from 3.1% of Hodoš to 38.3% of Žetale"* and *"at 3.1% and 3.2%
non-response in Hodoš and Gornji Petrovci"*. Both shares are computed with the withheld
refusal cell counted as zero, so they are floors and not measurements. Hodoš's true
non-response is between **3.1% and 6.7%** (11 to 24 of 356) and Gornji Petrovci's between
**3.2% and 7.0%** (70 to 155 of 2,217). The lowest unit with both hole cells published is
**Šalovci at 4.9%**, then Rogašovci at 5.9%. The qualitative claim survives untouched, since
those are the same corner of Prekmurje, but the range's low anchor should be a number the
office printed. `Žetale` at the top is 38.27% and is barely affected (38.3% to 38.8%).

The same reading is in `taxonomy/si2002.py`'s `Ni želel odgovoriti` note, which says the
refusal *"runs from 2.0% in Rogašovci to 25.6% in Miklavž na Dravskem polju"*: the maximum is
right, and Rogašovci at 2.03% is the lowest **published** value, with Hodoš and Gornji
Petrovci suppressed rather than lower.

**The §3.5 lean check reproduces exactly and is robust.** Pearson of each unit's excluded
share against that category's share of the unit's drawn base, over 192 units: unchurched
+0.553, atheist +0.384, `druge veroizpovedi` +0.286, Orthodox +0.170, Islam +0.138, Catholic
+0.043, Protestant −0.466. Reassigning every withheld person in a unit to the hole instead
moves all seven by at most 0.03. Dropping the two units with a suppressed refusal cell leaves
six of them within 0.08 and moves Catholic from +0.043 to −0.125, which is still the "almost
neutral" the note describes. The disclosure as written stands.

**The `Ni vernik, ateist` -> `unaffiliated` call is consistent with the three neighbours, and
Albania is consistent too.** Checked against the mapping files rather than the accounts of
them: `hr2021.py` `Nisu vjernici i ateisti` -> `unaffiliated`, `mk2021.py`
`Не е верник (атеист)` -> `unaffiliated`, `rs2022.py` `Not believers (atheists)` ->
`unaffiliated`, against bare labels `ba2013.py` `Ateist` -> `secular` and `al2023.py`
`Atheists` -> `secular`. The house rule is really what those five files do and Slovenia lands
on the right side of it. Albania is not in tension: its row is a bare `Ateist` and its form
offers no "no religion" answer at all, which is the reason `al2023.py` gives.

**Not reconstructing the seven recoverable cells is right.** Recomputed: seven municipalities
have exactly one of the five religion cells withheld with the subtotal published, and the
recoverable values are 1 or 2 people each, **11 people in total** (Dobje, Gornji Petrovci and
Grad, 2 Orthodox each; Kobilje, Razkrižje and Trnovska vas, 1 in `druge veroizpovedi`;
Žetale, 2). That is 0.5% of the 2,222 the map does not draw and 0.0006% of Slovenia, and the
cells are small precisely because they identify one or two households. §5's "a few dozen
people" is generous; it is eleven, which makes the call easier rather than harder.

**One suggestion, not a defect.** The ethnicity control is the strongest evidence for a claim
`note_public` already makes to the reader, that the hole is *"the question's doing rather than
the country's"*. As written the note supports that with the constitution and the
answer-for-yourself rule, which explain why non-response was possible, not why religion's was
2.6x ethnicity's on the same form. §6's stated reason for keeping it out, that it is not in
`si.csv`, is not a rule this project has: Austria's `note_public`, the next entry in
`countries.py`, quotes four 2021 Mikrozensus shares that are not in `at.csv` at all. A clause
would do it. Left alone here because the wording is the builder's.

Everything else reproduced from `data/normalized/si.csv`: 57.82 / 10.15 / 2.42% against 1991's
71.60 / 4.42 / 1.53%; 307,973 at 15.68% and 139,097 at 7.08%; 2,222 withheld across the seven
drawn categories and 1,974 across the five religion cells; drawn 1,514,744 = 77.12% and
`gap_share` 0.2288 exactly. The superlatives hold: Jesenice is the highest Muslim share
(22.72%) **and** the highest Orthodox share (10.08%) of drawn, Piran (26.00%) and Ljubljana
(25.76%) are the top two on the atheist answer, Osilnica is the only unit at 100% Catholic of
drawn, and the eight highest-share Protestant municipalities do hold 11,432 of the 15,855
placed. `check_md.py` clean, `built_countries.py --check` clean, `check_rollup.py si` clean,
no derived or modelled rows. One screenshot at the country's own `view`: dots inside the
outline, no dots in the sea, Ljubljana, Maribor, Celje and Kranj where they should be, legend
totals matching the CSV.
