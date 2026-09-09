# 001 — eg: Copts at governorate level from the Arab Barometer, or not at all

*Filed 2026-09-08 by session `f60e4589-09f4-4c85-b4ec-693275bcf079`. Anita's call; nothing is waiting on it.*

## What I did

**I did not build Egypt.** I scouted it to the point where it is buildable, measured it against
every bar this project applies to a survey-built country, found it clears all of them, and
stopped there rather than drawing it, because §14.4 rule 2 is about a persecuted minority and
that is not a call to make while building. Everything needed to build it is written down in
§11af and nothing about it is guessy any more.

## What it costs to reverse

**To draw it: about a day.** The route is `sources/lapop.py`'s construction pointed at a
different survey, plus a governorate name-harmoniser that already exists in scratch. To leave
it undrawn: nothing, Egypt stays blank as it is today. Neither direction touches a drawn
country.

## Why it is yours rather than mine

AGENT_BRIEF §3, first bullet, and it is the clean case: **whether a country may be drawn at
all and at what resolution, when a group's safety is affected by publishing where they live.**
Copts are a persecuted minority, the Egyptian state collected religion in 1986, 1996, 2006 and
2017 and released none of it, and §14.4 rule 2 says *no resolution finer than the state's own
publication*. §11d called this "the central example" of that rule and it was right to.

**What is new since §11d, and why it is worth asking again rather than leaving closed.** §11d
closed Egypt on four grounds and three of them have moved:

- *"CAPMAS has never published a religion tabulation and its site has nothing behind it."* The
  second half is wrong. CAPMAS ran a full NADA microdata catalogue at
  `censusinfo.capmas.gov.eg`, 544 studies with data dictionaries; §11d tested only the React SPA
  on the other host. It is DNS-dead now and archived to March 2026, and its dictionary carries a
  religion variable in the **Marriage and Divorce bulletins of 2008, 2011 and 2013**. That is a
  flow rather than a stock and I am not proposing to draw from it. **The first half of §11d's
  sentence is confirmed rather than contradicted**, and now from CAPMAS's own variable list: the
  2017 census file it deposited has **13 variables** — governorate, station, marital status,
  sex, age, work status, education — **and religion is not among them.** So there is no state
  microdata route to a Coptic map at any geography. It is the Arab Barometer or nothing.
- *"The only route is IPUMS, and the account is still not live."* There is now a second route
  that needs no account at all.
- *"The depth is not there anyway"* — Muslim / Christian / Jewish / Other. Still true, and it
  is two real categories.

What has **not** moved is §14, which is why this is the whole of the ask.

## The detail

**The source.** Arab Barometer, waves III, IV, V and VII, pooled — **6,840 Egyptian
respondents**, `Q1012` *"What is your religion?"*, answered by 100% of them, cut by `Q1`
Governorate. Public files, no account, no registration actually enforced; the URLs are printed
in the page HTML and AB's own FAQ says the data are public at no charge. This is the same
construction as Guatemala and El Salvador (§11ad), with a different barometer.

**It clears every bar, and it is checked from outside.**

| test | Egypt | bar / comparison |
|---|---|---|
| split-half rank correlation, 23 governorates | **+0.495** | bar +0.418 — **passes**, p=0.016 |
| governorates differ at all (chi-square) | p=**8.9e-37** | — |
| Upper Egypt vs the rest | **11.9% vs 4.7%** | p=7.9e-26 |
| weighted national Christian share | **5.93%** | 1986 census **5.7-5.8%** |
| Cairo governorate | **8.50%** | census Cairo **9.3% (1986), 8.57% (1996)** |

The last two rows are the ones that matter. **This instrument is not undercounting Copts** —
which was the thing to fear, because §11ad measured LAPOP reading 0.21x a census on exactly
this kind of cell in Suriname. It agrees with the last census that published, nationally and
at the one governorate where a published census figure exists to check against.

The pooled ordering is the real Coptic geography and nothing had to be assumed to get it:
**Sohag 16.9%, Minya 16.1%, Asyut 14.8%**, then Aswan, Luxor, Qena; Kafr El Sheikh 1.9% and
Sharqia 2.3% at the bottom. 24 governorates of 27 are sampled (North Sinai, South Sinai and
New Valley are not).

**The case for drawing it.** Egypt is 107 million people and the largest blank on the map after
South Africa; the Copts are among the largest undrawn religious minorities anywhere. The
governorate is a tier the Egyptian state has itself tabulated religion at — Cairo's 1976, 1986
and 1996 shares are all in the academic literature, sourced to CAPMAS — so drawing at
governorate is arguably *not* finer than the state's own publication, which is the Türkiye
argument (§11ac: twelve İBBS-1 regions IS the state's own publication, so rule 2 held by
construction).

**The case against, and I think it is the stronger one.** Türkiye's argument worked because the
Diyanet *published* the twelve-region table. Egypt published a **national** number in 1986 and
has published nothing since; the governorate figures exist in academic work because a researcher
obtained them, not because the state released them. So rule 2 on its plain reading permits
Egypt at **one national unit** and no finer. And the interesting thing about a Coptic map — that
Minya and Asyut are a fifth Christian — is exactly the resolution the rule withholds.

**Three ways to rule, and the middle one is real.**

1. **Draw at governorate.** Best map, clearest §14.4 rule 2 problem.
2. **Draw Egypt as one national unit**, 5.9% Christian, no internal geography, on the 1986
   published national figure with AB carrying the level forward. Rule 2 satisfied on its plain
   reading. 107M people stop being blank; the Upper Egypt concentration is stated in
   `note_public` and not drawn, which is how Türkiye's Alevi figure is handled.
3. **Leave it closed**, and record that the route exists so nobody re-derives it.

I have no view I would defend against yours on this one. If it helps: option 2 is the one that
follows this project's existing precedents most closely, and it is what I would do if the call
were mine and I had to make it today.

---

## Ruled 2026-09-08 by Anita

**Draw at governorate — option 1.** Her words: governorates are pretty big.

She asked whether anything finer is even possible. It is not, and that is worth recording so
nobody re-derives it: the Arab Barometer cuts by `Q1` Governorate and carries no finer
geography, and CAPMAS deposited a 2017 census file whose 13 variables do not include religion
at any tier. So governorate is simultaneously the ruling and the ceiling of the instrument.

---

## Built 2026-09-08, at governorate

`sources.md` §9bz and `sources/eg.md`. 24 of 27 governorates, **107,658,120 people**, Muslim
and Christian, every row `modelled`.

Every check in this ask reproduced, and **two numbers moved for reasons worth knowing**:

* the **+0.495** split-half was the unweighted one. The build weights, because the weighted
  share is what it draws, and reports **+0.518** against the same +0.418 bar on the same 23
  units. Putting the weights back reproduces +0.495 exactly.
* **the ordering in this ask is the unweighted one, and the map's top governorate is Minya at
  16.41%, not Sohag.** Weighted, Sohag reads 13.90% and Asyut 13.58%. The three sit inside one
  another's 95% intervals either way, so `note_public` names Minya, which is the figure in
  `data/normalized/eg.csv`, and tells the reader to take the three as a group.

The magnitude did not come from COD-PS, whose Egypt file is the **2012 COMPAS estimate at
81.4M**. It comes from **CAPMAS's own governorate population API**, which §11d and §11af had
both concluded did not exist: it is on **port 8080**, not on a path of the 443 host, and the
port is named in the React bundle's constants block.
