# Kazakhstan — BNS, National Population Census 2021. **A modelled country**

Wired 2026-09-07. 19,186,015 people, **17 regions**, 9 drawn nodes, **100.00% drawn**,
and **every row `modelled`** — the first country on this map with no measured tier at all.

| | |
|---|---|
| magnitude | Bureau of National Statistics, **NPC 2021**, *Численность населения по этносам, населенным пунктам и по возрастам*, **sheet 2.1** — ethnicity × 17 regions |
| coefficients | the same census, **«Национальный состав, вероисповедание и владение языками»** (Astana 2023), **chapter 12** — religion × nationality, national |
| basis | `modelled` (spec §7, §14.10) — ethnicity-derived, from the census's own cross-tab |
| geography | **17 regions** — 14 oblasts + Astana, Almaty, Shymkent; ~1.13M people each |
| drawn | **19,186,015 people, 100.00%** — every answer Question 11 offers is on the tree (§4) |
| licence | BNS publications, free to download and cite; no wall, no registration |

**Kazakhstan publishes religion for the country and nowhere else.** `sources.md` §11u
established that four ways. So the choice was a modelled country or no country, and Anita's
call was to build it — *"i think it'd be interesting to see. if we can get it working we might
be able to fill out a bunch of other countries this way, with disclosures."*

---

## 1. The model, in one line

```
    count(region, religion) = Σ_ethnicity  pop(region, ethnicity) × share(religion | ethnicity)
```

Both inputs are **the same census, the same office, the same round**. The magnitude is BNS's
count of ethnicities in each region; the coefficients are BNS's national cross-tabulation of
religion against nationality, collected from the same people in the same interview.

**No magnitude is estimated.** §14.4's rule 1 — *never estimate a magnitude a source does not
publish* — holds by identity here: every person the map places is a person BNS counted in that
region, and the model only decides which column they go in. It never creates anybody.

### The two margins come back exact, and that is arithmetic rather than luck

Because the coefficients are conditional on ethnicity, and the two publications' ethnic
margins agree to the person on all eighteen groups, the model necessarily reproduces:

* **every religion's national total, to the person** — Islam 13,297,775, Orthodoxy 3,269,143,
  and so on down to Judaism's 7,192;
* **every region's population, to the person**.

`sources/kz.py` asserts both. That is not evidence the model is *right* — a model that
reproduces its own margins is doing what it was built to do — but it does rule out the whole
family of arithmetic errors, and it means no dot is invented or lost.

## 2. The coefficients are the best any modelled country here has

Greece, Spain and France all multiply a state's count by a **third party's** national
composition (Pew's). Kazakhstan's coefficients are its own census's own cross-tab, an exact
partition over 18 nationalities. §14.10's condition 2 — *documented and attributable, not
fitted* — is met about as strongly as it can be.

**The pairing between the two publications is asserted on the counts, not on the names.** The
volume names nationalities in Kazakh (`Қазақтар`) and the workbook in Russian (`Казахи`), and
`kz.py` requires all eighteen to agree **to the person** before a single coefficient is used.
They do.

> **A mixed-script header cost half an hour and would have failed exactly one row.** BNS types
> the workbook's German column as **`Hемцы`** — Latin `H` (U+0048), not Cyrillic `Н` (U+041D).
> A dict lookup on the name misses Germans and only Germans, and every printout looks
> identical. `kz.py` folds the Latin/Cyrillic confusables before matching anything by name.
> **This is worth carrying to other countries: any Cyrillic or Greek source can do it.**

## 3. The held-out test — §14.10's fifth condition, and the reason to believe any of this

§14.10 asks what the output was checked against, and says a model with no check is not
forbidden but is *required to say so*. Kazakhstan can do better than say so.

**The volume publishes religion × nationality separately for URBAN and RURAL Kazakhstan.**
That is the model's own assumption — *share(religion | ethnicity) does not vary by place* —
written down as a testable claim about a partition the model never sees.

Predicting religion × urban/rural from ethnicity × urban/rural and the **national**
coefficients puts **313,745 people, 1.64% of the country, on the wrong side of the
town/country line**:

| | urban | rural | |
|---|---|---|---|
| `Ислам` | +1.7% | −2.3% | 69.3% of the country |
| `Православие` | +0.9% | −2.4% | 17.0% |
| `Буддизм` | +1.3% | −6.0% | |
| `Отказались указать` | −7.8% | **+15.0%** | 11.0%, drawn on `unknown` (§4) |
| `Неверующие` | −13.0% | **+41.8%** | 2.3%, and the weakest thing on the map |
| `Католицизм` | +43.5% | −34.3% | 18,988 people, so ~9,000 either way |
| `Протестантизм` | −17.1% | +105.6% | 9,419 people |

**The finding is clean and it is the one that generalises.** The cells that are about
**ancestry** are predicted well; the cells that are about **attitude** are not. Non-belief and
refusal are urban behaviours *inside every ethnic group at once* — Kazakhs are 1.4%
non-believing in town and 0.6% in the country, Koreans 17.3% and 10.7% — and an ethnicity
model cannot see that by construction.

### The urban/rural coefficients are deliberately NOT used

They would fit better. They are also the only independent evidence this country has, and
**a check you have spent is not a check.** Using them would leave Kazakhstan in exactly the
position §14.10 warns about: a plausible model with nothing to falsify it. So the build uses
the national coefficients and reports the error.

## 4. The 11.01% who declined, and why they ARE drawn

**This reverses the first build, and the thing that reversed it is the census form.** Anita,
2026-09-07: *"is there any way we could try to draw the 11% who refused to state?"*

Question 11 of the 2021 individual questionnaire (`Переписной лист 3-И`, on stat.gov.kz):

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

**`Отказываюсь указать` — "I decline to state" — is option SIX: printed, numbered, first
person, and chosen by 2,112,653 people out of seven offered.** It is not a blank, not item
non-response, and not an enumerator's residual code.

**That is exactly what separates it from Trinidad's `Not Stated`** (tt2011.py, 11.10%), which
is the derived residual §3.5 is written about. **An answer that was offered and picked is an
answer.** So it goes on the tree, and Kazakhstan is **100.00% drawn**.

### Why `unknown` and not a node of its own

`branches.py` defines `unknown` as *"the one that reports nothing at all — people the source
counted and whose religion it did not establish… the claim is only that these people are
here"*, and names Vietnam as **the first** customer rather than the only one. A refusal is the
plainest possible instance: counted, religion not established, nothing else claimed. Vietnam's
cell arose differently — its answer set never reached what those people practise — and the two
countries never appear together, so one legend row carries one meaning in each. A dedicated
`undisclosed` node is the cleaner long-run answer and would need an eighth colour in a grey
ramp whose tightest pair is already dE 8; it is not built.

### Drawing it is §3.5 satisfied, not bent

The rule is *marked, not filled, and never redistributed*. These people are **not**
redistributed into any religion — they are drawn where they are, in their own colour, saying
precisely what the census recorded. Excluding them marked the absence in a legend line;
drawing them marks it on the map, which is stronger and is what a dot map is for.

**Two things to carry:**

* It varies enormously by ethnicity — 9.25% of Kazakhs, 7.48% of Russians, 26.7% of Kurds,
  **65.7% of the residual "other nationalities"** (which is less mysterious than it looks: the
  residual carries the 24,806 people who gave no *ethnicity* either). So the model puts more
  of it in some regions than others for reasons about ancestry rather than about belief.
* **Kazakhstan requires religious groups to register**, refuses registration to Jehovah's
  Witnesses and to Ahmadi Muslims, and prosecutes unregistered worship. Some part of an 11%
  refusal is plausibly about that, and nothing published says which part.

**It is the model's second-worst cell** (−7.8% urban / +15.0% rural), behind `Неверующие`.
That is a reason to label it carefully — `note_public` does — and was never a reason to leave
2.1 million people off the map.

### The layer is nearly flat, and that flatness is the model's

**Every region comes out between 9.35% and 12.19% — a spread of 2.84 points.** That is not a
finding about Kazakhstan; it is the arithmetic. Refusal is placed by ancestry, Kazakhs (9.25%)
and Russians (7.48%) decline at similar rates, and the residual group that really does differ
(65.7%) is only 1.4% of the country — so every region is dragged to the national 11%.

**The real variation is by town and country, and the model flattens exactly that**: the
held-out test misses urban refusal by −7.8% and rural by +15.0%. So the honest reading of this
layer is *"about a ninth of Kazakhstan declined, roughly everywhere"* and **not** *"refusal is
uniform across Kazakhstan"*, which is what a reader would otherwise take from an even wash.
`note_public` says so in those words.

**It is still worth drawing.** An even wash that is honestly labelled beats 2.1 million people
silently absent — §6.12's argument, which is that a blank cannot tell "nobody here" from
"nobody counted here".

**Palette note:** `unknown` sits at dE 21.0 from `secular` in Kazakhstan, under the usual 25.
Accepted on the grey family's own documented terms — the floor inside that family is about 15,
not 25, because a mix-up between two neighbours on one spectrum is a small error where reading
either as a religion is not. They also separate warm against cool (hue 51 against 204), and
`unknown` clears every religion Kazakhstan draws by 71–82.

## 5. What to distrust, in order

1. **`Неверующие` — non-believers, 432,140.** The held-out test misses it by +42% in rural
   Kazakhstan. If any single thing on this map should be read as "roughly this many, roughly
   around here", it is this.
2. **`Отказались указать` — declined to state, 2,112,653.** Second-worst (−7.8% / +15.0%), and
   for the same reason: declining is an act of a person in a moment, and ancestry is being
   asked to predict it. The national total is exact; the geography is the model's.
3. **`Иудаизм` — 7,192.** The ethnicity `Евреи` sits inside the residual group, so this cell's
   geography comes from the residual's composition rather than from Jews directly.
4. **`Католицизм` and `Протестантизм`** — small, and the held-out test says the model gets
   their town/country balance badly wrong in opposite directions.
5. **Islam and Orthodoxy** — 86% of the country, predicted within 3%, and the big north-south
   pattern they make is real.

## 6. 17 regions, and why not 218

The census publishes ethnicity at **218 level-2 units** (rayons and cities) as an exact
partition, and COD-AB has exactly 218 ADM2 polygons. It was not taken:

* **§14 restraint.** 218 units of pure inference is a much stronger claim than 17, in a
  country where nothing is counted at any of them.
* **The join would be 218 fuzzy cross-script matches.** COD names ADM2 in English
  transliteration (`Arshaly District`, `Korgalzhyn District`) and the census in Russian
  adjectival form (`Аршалынский район`, `Коргалжынский район`), with **no shared code** —
  COD's is `KAZ###`, the census's is KATO. §12's shape 2 with 218 chances to fire.

At ADM1 the same join is seventeen names, checked by rule and by eye. See `sources/kz_geo.md`
for the 2022 three-oblast reform and how it is undone.

## 7. Vintage and licence

The census was enumerated 1 September 2021; the religion volume is Astana 2023. **The three
oblasts created in 2022 (Abay, Jetisu, Ulytau) postdate the census** and are merged back —
§8.1, and `sources/kz_geo.md` §1.

BNS publications carry a Civil Code notice requiring attribution when the statistics are
reused, and no other restriction; downloads are open with no registration. Same footing as the
other census offices here, and worth re-checking before any commercial use
([[reference_poster_commercial_licences]]).
