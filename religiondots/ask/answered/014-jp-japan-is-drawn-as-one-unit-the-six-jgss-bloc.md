# 014 — jp: Japan is drawn as one unit; the six JGSS blocks need microdata that only researchers can get

*Filed 2026-09-14 by session `f95259a4-jp2`. Anita's call; nothing is waiting on it.*

## What I did

Drew Japan as **one national unit** from the Japanese General Social Surveys' published national
religion table, pooled over 2021 to 2024 (10,612 respondents): 71.7% no religion, 18.8% Buddhist,
3.1% Japanese new religions (Soka Gakkai 1.8%), 1.1% Christian, 0.9% Shinto. That is step one of
your baseline and it stays inside the spec: nine countries are already drawn as one unit.

**Step two, the six regional blocks, is not drawn**, because JGSS does not publish religion by
block. It exists only in the microdata.

## What it costs to reverse

Nothing to undo. If a block table arrives, `sources/jp.py` gains a block tier, `grain` changes,
and the build tail reruns, about an hour.

## Why it is yours rather than mine

§3's *"anything needing an account, money, or her identity"* and *"a source whose terms are
unclear or which bans what we are doing"*. Every route to the block figures is one of those:

| route | what it needs |
|---|---|
| SSJDA (Univ. of Tokyo) microdata, JGSS to 2005 and later waves | an application from a university researcher or supervised student |
| JGSSDDS (JGSS's own download system, and the only place the 4-digit religion recode is) | an application, approved, with an academic advisor or guarantor |
| ICPSR, JGSS cumulative files | membership of an ICPSR member institution |
| GESIS | a personal account; academic-use terms |
| SSJDA Data Analysis, the anonymous online cross-tab tool | **does not carry JGSS** (247 surveys listed, checked 2026-09-14); its consent screen also limits use to academic secondary analysis |

## The detail

**One published block table exists, and using it would be a method the spec does not cover.**
JGSS's director showed JGSS-2015 by block in a 2017 Pew talk: the share who believe in a religion
or have a family religion is 25.9% Hokkaido/Tohoku, 24.7% Kanto, 34.3% Chubu, 31.8% Kinki, 36.0%
Chugoku/Shikoku, 36.9% Kyushu. It is one wave, percentages without counts, and not split by
religion, so the split-half and the chi-square cannot run on it. Drawing it would mean: that block
share for "has a religion", with the national mix inside it. NHK's 1996 chart, aggregated to the
same blocks, runs in the same west-high, east-low order (`sources/jp.md` §3), which is a check from
a different instrument twenty years earlier, not the test the spec asks for.

**So the options are:**

1. **Stay national** (what is built). Honest, flat.
2. **Draw the JGSS-2015 block split for the religious share**, national mix inside each block.
   Moves roughly 6 points of people between east and west; one wave, no counts.
3. **An academic route in your name** (GESIS account is the cheapest), which would give all 18
   waves by block and let the normal stability test decide which religions carry a block
   geography. Pure Land and Soka Gakkai are the likely ones.

**Checked and not offered as an option:** the Agency for Cultural Affairs roll's Christian line.
Its national total matches self-identification (1.51% against NHK 1996's 1.46%), but Tokyo holds
46.9% of the roll's Christians on 11.5% of the population, about 747,000 people beyond what its
churches' count explains, because independent corporations report their whole membership at their
registered address. Outside Tokyo the roll runs below the survey almost everywhere. It cannot place
Christians. `sources/jp.md` §4.


---

## Anita, 2026-09-14: partial answer, ask stays open

- **Not option 1.** She does not want Japan left at one national unit.
- **Option 3 is being explored.** She asked for guidance on which account route to open and what
  it would realistically yield. Keep this ask open until that resolves.
- **Buddhist schools as Japan-only legend rows: yes.** This reverses the build's call to keep one
  Buddhism node. Split the schools out of the national JGSS table, if its 164 codes carry them,
  whenever Japan is next rebuilt.

---

## Routes for option 3, checked 2026-09-14 by the supervisor

| route | where respondents are placed | religion detail | access |
|---|---|---|---|
| WVS wave 7, Japan 2019, n = 1,353 | `N_REGION_ISO` at prefecture; 45 listed, JP-31 Tottori and JP-45 Miyazaki absent. `N_REGION_WVS` is 5 blocks | `Q289CS` Japan: none, Buddhism, Catholic, Orthodox, Protestant nfd, Other. No Shinto code, no schools | download form, no account; non-profit use only; publications cited and reported to the WVSA |
| ISSP Japan, fielded by NHK; checked on the 2011 background-variable report only | `JP_REG`, 9 regions from the Basic Resident Register: Hokkaido, Tohoku, Kanto, Koshinetsu, Tokai-Hokuriku, Kinki, Chugoku, Shikoku, Kyushu | `JP_RELIG` from F22: Buddhism, Shinto, Christianity, other, none. No schools | GESIS account, access category A: released for academic research and teaching; other purposes need GESIS written agreement |
| JGSS prefecture codes, supplemental application at `jgss.daishodai.ac.jp/english/data/dat_application_pref.html` | prefecture from JGSS-2000 on; block variables are in the archived files for JGSS-2010 and earlier | the full JGSS religion list | institutional affiliation with a department head, plus a guarantor signature. Not open to Anita |

**Not checked:** whether every ISSP year carries `JP_REG`; ISSP Japan annual sample sizes; the purpose-of-use options on a GESIS download. **No route gives Buddhist schools below national level.** The realistic best is ISSP pooled over many years at 9 regions with broad categories.

---

## Anita, 2026-09-14, second reply, and what was done with it

- **1996 for allocating: yes.** *"actually i guess if nothing is more recent, 1996 is better than
  nothing. we can use it for allocating."*
- **A block pattern may come from a different survey: yes.**
- **Buddhist split: left to the agent.** Done, as five nodes (sources/jp.md §7).
- **Christians and the Agency roll:** she asked whether the roll is really that unrealistic,
  since it is official and is what everyone cites for Tokyo. Answered with numbers rather than
  decided: believers per Christian body are 886 in Tokyo and 578 in Kanagawa against a median of
  76, and on the roll's pattern Tokyo would be Japan's most Christian prefecture at 4.7% with
  Saitama at 0.4%. A per-church count (United Church of Christ in Japan, members by district)
  puts Tokyo plus Chiba at 1.68x the national rate, NHK 1996 at 1.74x, the roll at 2.96x.
  **Built on NHK 1996; the roll version is `python sources/jp_alloc.py --christians roll`.** Her
  call.
- **Retry the source search:** done again (sources/jp.md §8).
- **ISSP:** she asked what she would need to do. The walk-through is in sources/jp.md §8: GESIS
  releases ISSP for academic research and teaching, so a public hobby map needs GESIS's written
  agreement first. Her call whether to write.

**Built the same day**: 47 prefectures, sources/jp_alloc.py and sources/jp_grid.py. The ask
stays open on the two calls above.

---

## Anita, 2026-09-14, third reply: closed

Having looked at the built map: *"nhk christianity seems decent, and not too far off so lets keep
it yeah"*, and *"for ISSP i think 9 regions is not that helpful. i agree lets skip."* Christians stay
on NHK 1996; no GESIS request. Nothing left open.
