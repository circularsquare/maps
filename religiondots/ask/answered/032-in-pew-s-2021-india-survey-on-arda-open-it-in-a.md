# 032 — in: Pew's 2021 India survey on ARDA: open it in a browser

Summary: An all-India Christian split needs Pew's 2021 Religion in India dataset. ARDA hosts it (PWIND20) but its page needs a browser. Could you open it and say whether it downloads freely or needs a login? Nothing waits.

*Filed 2026-09-15 by session `d743fc47`. Anita's call; nothing is waiting on it.*

## What I did

Started the India Christian upgrade on the two routes that need nothing from you: Kerala at 14
districts and Mizoram at state level (`sources.md` §scout-2026-09-15-india-christians). The rest of
India keeps its Christians on one node.

## What it costs to reverse

Nothing to undo. If the file is open, a later builder adds regional splits for the rest of India.

## Why it is yours rather than mine

A page that only loads in a browser is a browser job (`CLAUDE.md`, Blocked downloads), and if it
turns out to need an account, that is yours to decide (`[[feedback_gated_data_last_resort]]`).

## The detail

- **The page:** https://www.thearda.com/data-archive?fid=PWIND20 (the URL shape ARDA uses for its
  other archives, such as Taiwan's `TSC09`).
- **What to look for:** a download link for the data file, and whether it asks you to log in or
  register first.
- **Why it matters:** Pew's *Religion in India* (2021) asked 1,011 Christians their church, but the
  published results are all-India only (Catholic 37%, Baptist 13%, Church of North India 7%, Church
  of South India 7%). A regional split needs the respondent file.
- **Pew's own copy** asks for an account, so ARDA's was the open mirror to try first.
- If ARDA also needs a login, the options are to skip it or register; both are yours.


---

## Answered 2026-09-15 by Anita: downloaded

*"downloaded, the file looks really small though"*

The file is complete: `Pew India Survey Dataset.DTA` is 10.2 MB, 29,999 respondents and 312
variables, which is the whole survey (a Stata file of answers is small). It has `qdenomrec` (church)
and `weight`; the only geography variable found is `region`, not state. Copied to
`data/raw/in/pew_india_2021.dta`. The second download, `ICPSR_38489-V1.zip`, holds only the
codebook, questionnaire and ICPSR's terms of use, copied beside it as
`data/raw/in/ICPSR_38489-V1.zip`. The builder reads the terms before using the file.
