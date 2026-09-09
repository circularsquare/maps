# 006 — pa: A free UNICEF MICS account would give Panama the two comarcas LAPOP cannot reach

*Filed 2026-09-08 by session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-pa`. Anita's call; nothing
is waiting on it.*

## What I did

Drew Panama from the LAPOP AmericasBarometer, 6,105 respondents over ten of its thirteen
first-order units, and put **Comarca Guna Yala and Comarca Emberá-Wounaan into `gap=`** —
44,374 people, 1.09% of the country — because LAPOP's `prov` has no code for either in any
wave. They keep their polygons and their hexes and draw no religion, which is Ecuador's
Galápagos precedent (§9bn).

## What it costs to reverse

If the MICS microdata arrives, Panama is a **rebuild rather than an edit**: a new
`sources/pa_mics2013.py`, a new mapping, and the LAPOP build demoted to a cross-check the way
`do` and `uy` demoted theirs. Two to three hours, and the existing files are the template.
Nothing on the map is wrong meanwhile.

## Why it is yours rather than mine

AGENT_BRIEF §3, *"anything needing an account, money, or her identity"*. `mics.unicef.org`
wants a registration with a name, an email address and a stated research purpose before it
releases a country's datasets. This is the same shape as open ask **002** (za, DataFirst) and
I am filing it separately only because it is a different registry and a different country;
if the answer to 002 was no, the answer here is probably no too.

## The detail

**What Panama's own state has, established this session and written up in `sources/pa.md`:**
INEC's five censuses ask no religion question (§11x, re-confirmed), but the office has asked
it twice on household surveys and published neither answer at the geography it holds.

- **MICS 2013.** `www.inec.gob.pa/archivos/MICS_FINAL.pdf` page 125 is the household listing
  form, and it carries `HC1.A ¿QUÉ RELIGIÓN PROFESA (nombre)?` with a card of ten-plus named
  options, asked of **every member** of **11,100 households**, beside `HC1.D` on indigenous
  group. The survey is representative of **all twelve provinces and comarcas**, with
  oversampling in Colón, Darién and Panamá. That is roughly **forty thousand people's
  religion at a geography that includes both comarcas this map leaves blank**, against
  LAPOP's 6,105 across ten units. Grep the 230-page report for `religi` and there are two
  hits, both on that questionnaire page: the office tabulated none of it.
- The World Bank catalogue lists it as study **2921** with access type `remote`, meaning
  the data lives with the producer. The producer is UNICEF: `mics.unicef.org/surveys`,
  free of charge, account required.
- **The other route needs no account and needs a person.** INEC's own April 2022 slide deck
  (*Módulo de Percepción de la EPM*) publishes a national religion bar — 65% Catholic, 22%
  Evangelical, 2% Adventist, 1% Jehovah's Witnesses, 2% other, 8% none — and says on page 2,
  in its own words, that results **can be obtained by province and comarca**. The director
  then quoted two provincial figures to a television reporter. The table exists and has never
  been printed. Asking INEC for it is a letter from a person, not an agent.

**What it would fix, in order of size:**

1. The two comarcas would be measured rather than blank.
2. `Religiones Tradicionales` would stop being a floor. LAPOP reads it at **0.4%** of Panama
   against the **9.4%** of INEC's own 2022 informants who identify as indigenous, and the two
   units that would carry it are exactly the two not drawn.
3. Ngäbe-Buglé is currently drawn on **n=138**, and it is the most interesting unit on the
   map: 58.8% Evangelical against 29.5% Catholic, the only unit where Evangelicals lead.
4. MICS's card names Buddhism and Orthodoxy separately, so part of `other.pa` would resolve.

**What it would NOT fix.** Panamá Oeste. MICS 2013 predates the 2014 province split as well,
so the 2.09 million people currently drawn as one unit would stay one unit.
