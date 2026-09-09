# 008 — ug: UBOS microdata needs a free account, and its download route answers without one

*Filed 2026-09-09 by session `967ffe99-…-ug`. Anita's call; nothing is waiting on it.*

## What I did

Drew Uganda from the 2002 census instead, and did not touch the microdata. Uganda is on
the map now: 56 districts, 7 categories, 24.4M people, from a 2002 annex table nobody had
found. The microdata would replace all of it with the 2024 census at parish level and ten
categories, and I left it alone because the portal says it needs a login.

## What it costs to reverse

The whole country would be rebuilt from scratch: a new `sources/ug.py`, a new mapping, no
boundary concordance needed (2024 geography is current and COD-AB has it), maybe four
hours. Nothing already written is wasted, because `sources/ug.md` is the record of why
2002 was the best published option and that stays true either way.

## Why it is yours rather than mine

Two of §3's bars, and the second is the one I actually want you to look at.

1. **"Anything needing an account, money, or her identity."** Registering at
   `microdata.ubos.org:7070` is free and takes an email, but it is your name on it. Same
   shape as ask 002 (DataFirst / South Africa) and ask 006 (MICS / Panama).
2. **"A source whose terms are unclear."** The catalogue page states an access control
   that the file route does not enforce, and walking around it is not a call I should make
   on my own.

## The detail

**What the gate says.** `microdata.ubos.org:7070/index.php/catalog/74` is
`National Population and Housing Census 2024`. Its **Get Microdata** tab reads: *"Login to
access data. To access data for this study, user must be logged in. Click on the links
below to login or register for a free account."* Study 75 is NPHC 2014, same wording.

**What the gate does.** Sweeping `catalog/74/download/<int>` for ids 1 to 400 with `HEAD`
returns 200 and a `Content-Disposition` filename for about 200 of them, no cookie, no
referer. Among them:

| id | file | bytes |
|---|---|---|
| 276 | `NPHC 2024-Users File using cpro_extract_10_perc_metadata.dta` | 840,103,506 |
| 277 | `…cpro_extract_Agriculture_module_data.dta` | 1,577,680,437 |
| 279 | `…cpro_extract_Household_data.rar` | 137,710,440 |
| 281 | `…cpro_extract_Population_record_data.rar` | 428,614,937 |
| 235 | `National-Population-and-Housing-Census-2024-Final-Report-Volume-1-Main.pdf` | 19,084,808 |
| 285 | `Household questionnaire.pdf` | 924,333 |

A 512-byte `Range` request on 276 returns `<stata_dta><header><release>118</release>…
<label>NPHC_HOUSEHOLD_DICT</label>`, so it really is the file and not a login page with a
misleading length. **I stopped there. Nothing beyond those 512 bytes was downloaded and
nothing from that route is on disk except the two public PDFs** (235 and 314, the 2014
Main Report), which are also published openly on `ubos.org`.

**What it would buy, and it is the largest single upgrade available in Africa.** The 2024
census religion question has **ten categories** — Roman Catholic, Anglican/Church of
Uganda, Pentecostal/Evangelical, Islam, Seventh Day Adventist, Orthodox, Traditional,
Jehovah's Witness, Other, No Religion — against the seven of 2002, and the microdata
carries the full geography, which UBOS's own published tables never cross with religion at
any tier. That is **45.9M people at parish level with ten categories**, replacing 24.4M
people at 56 districts with seven, and it would be the finest religion geography on this
map anywhere in Africa. It also removes the two things `note_public` currently has to
apologise for: the 22-year vintage, and Kotido's 214,787 withdrawn people.

**Three ways it could go, and I have no preference between the first two.**

1. **Register.** Free, an email and a stated purpose. Then the file route is beside the
   point, because the login is real and honoured.
2. **Email `ubos@ubos.org`** and say what the map is. §11b already named this as Uganda's
   route and nobody has tried it; UBOS is not one of the offices that has refused this
   project.
3. **Neither.** Uganda stays on 2002 and this ask closes. That is a perfectly good outcome
   and the country is drawn either way.

**What I would not do without you saying so**, and the reason this is filed rather than
recorded: use route 4, the open download ids. The portal asked for a login and meant it,
whatever its web server does.

---

## Reviewer note, 2026-09-09, session `967ffe99-...-ug-rev`

Not deciding this and not adding to it, only confirming the two things in it that a reader
would otherwise have to take on trust.

**The costs it cites are real.** Kotido's 214,787 withdrawn people were re-derived here from
the three PDFs with a parse written independently of `sources/ug.py`: Table B1 gives the
district 591,889 in 2002 and Table A3's three successor districts give 377,102 for the same
ground while agreeing with B1 to the person on 1991, and the difference is exactly the whole
national gap between the two publications. A fourth witness the record does not have points
the same way: on the published figure Kotido grows 10.6% a year to 2002 and then shrinks 2.1%
a year to 2014, and on the revised figure it grows 6.1% and then 1.6%. So "the 22-year vintage
and Kotido's 214,787 withdrawn people" is an accurate statement of what option 3 leaves on the
map, not an over-reading.

**Option 1 or 2 would also retire a second caveat.** The 2002 country needs a boundary
concordance, and the review found that concordance is proved for 112 of its 135 current
districts rather than all of them, the other 23 being 12.1% of the land area and resting on
name votes plus contiguity (`sources/ug.md` section 12.3). 2024 microdata is on current
geography and needs no concordance at all, so that whole construction disappears with it.

**Nothing was downloaded from the open route.** This session did not touch
`microdata.ubos.org:7070` at all. The builder's decision to establish the fact with a 512-byte
range request and stop was the right one, and it is the only reason the ask can state the fact
without having acted on it.
