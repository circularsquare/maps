# 002 — za: A free DataFirst account would take South Africa from 9 provinces to 234 municipalities

*Filed 2026-09-08 by session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-za`. Anita's call; nothing is waiting on it.*

## What I did

**Drew South Africa at nine provinces from the openly published Community Survey 2016
provincial profiles, and shipped it.** That is 55M people over 24 categories at 6.1M people
per unit, which is the coarsest counting geography on this map. It is the finest tier of
this variable that exists anywhere open, and §2 of `sources/za.md` is the record of checking
that rather than assuming it.

## What it costs to reverse

Nothing that exists is thrown away. A finer build would reuse `taxonomy/za2016.py` and
`sources/za_geo.py`'s pattern unchanged and replace `sources/za.py` with a microdata reader,
so it is roughly a day's work for whoever picks it up, plus a retile. **The country stays on
the map either way**; this decides only whether it is drawn at 9 units or about 234.

## Why it is yours rather than mine

AGENT_BRIEF.md §3, last bullet: **anything needing an account, money, or her identity.**
DataFirst's download requires a registered account *and* a signed confidentiality
declaration in your name. It is not the Korean-ID wall and not IPUMS; it is the cheap kind,
one form and a declaration. But it is your signature on a terms document, so it is not mine
to give.

## The detail

**What the account buys.** DataFirst catalogue 611, Community Survey 2016 person file:
3,328,867 records × 99 variables. The two variables that matter are already confirmed from
the browsable metadata (the metadata needs no login; the files do):

- `ReligionBelief` — the same 13 codes as the published table 2.10a: Christianity, Islam,
  Traditional african religion, Hinduism, Buddism [sic], Bahaism, Judaism, Atheism,
  Agnosticism, No religious affiliation/belief, Other, Do not know, Unspecified.
- `Christianity` — **15** denominations, the fourteen published in table 2.10b plus
  *Do not know*.
- Geography in the same file: `DC_MDB_C_2016` (district/metro), `MN_CODE_2016` (local
  municipality), `PR_CODE_2016` (province).

CS 2016 was *designed* to be representative at local-municipality level, so this is not
over-reach on a survey. **Same 24 categories, same year, same instrument, 26 times the
spatial detail**, and it would reconcile against the nine provincial profiles already on
disk, which is a free correctness check most builds do not get.

**Why nine provinces is the open ceiling.** Everything below was opened and read, not
inferred from a search result; the full table is `sources/za.md` §2.

- Report 03-01-84 *Cultural dynamics in South Africa*, the report named as the likely home
  of a finer cut, is **province-only and coarser** (8 categories, Christianity undivided).
- **Census 2011 asked no religion question at all** — stated in 03-01-84 p.48 — which closes
  Wazimap and every 2011 municipal product.
- Stats SA's own keyless Census 2022 dissemination API serves 24 topics down to **Main
  Place**, and religion is on none of them (0 hits for `religio` in the whole front-end
  bundle). Language goes to main place; religion is not served at any level.
- In the Census 2022 provincial profiles, religion is **the single variable published
  province-only** while population, language, education, dwelling, water and the rest are
  all tabulated by district and local municipality. That reads as a decision, not a gap.

**How much it would change.** At nine units the map can say South Africa is 32.6% African
Independent Church and that Limpopo is 50.8% while Western Cape is 15.8%. It cannot say
anything about Soweto against Sandton, or about the old Transkei against the Eastern Cape's
coastal towns, and on a country this internally divided that is most of what a reader wants.
Six million people per unit is four times Zimbabwe's grain, which the spec already calls the
coarsest on the map.

**Two things worth knowing before you decide.**

1. Census 2022's 10% sample (DataFirst 982) is behind the *same* account and is more recent,
   but leaves Christianity undivided, so it would trade the whole denominational split for
   six years of currency. If you sign up, 611 is the file to take, not 982.
2. There is a second route that needs your identity in a different way: **Stats SA accepts
   custom tabulation requests**, and religion by district plainly exists in their database.
   That is an email rather than a download. It is mentioned only so the option is on the
   record; the account is much the cheaper of the two.

**IPUMS International also has RELIGION for South Africa 1996, 2001 and 2016** and is
separately dead for this project (`[[reference_ipums_account]]`), so it is not an
alternative route to the same thing.
