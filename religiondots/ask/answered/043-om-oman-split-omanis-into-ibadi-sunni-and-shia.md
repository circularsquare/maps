# 043 — om: Oman: split Omanis into Ibadi, Sunni and Shia by wilaya?

Summary: Omanis are drawn on plain Islam. Figures are national only (21-75% Ibadi); Dhofar is described as entirely Sunni. A Shia mosque in Muscat was attacked in 2024. Split, partly split, or keep one Islam?

*Filed 2026-09-15 by session `cb8b206e-om`. Anita's call; nothing is waiting on it.*

## What I did

Drew all 2,984,793 Omanis on `islam` in each of the 61 wilayat, with no Ibadi, Sunni or Shia split.
Expatriates carry no sect either (their Muslim branches are folded to `islam` in `sources/om.py`).

## What it costs to reverse

Change the Omanis block in `sources/om.py` and the MAP in `taxonomy/om2024.py`, rerun it and both
scatters, then the build tail: under an hour. The hard part is that nothing below the nation has a
number (below).

## Why it is yours rather than mine

Two bars. Your priority line of 2026-09-15 names Oman for its Ibadis, so whether they are drawn is
a call you have already signalled you care about. And AGENT_BRIEF §3's first bar (spec §14): on 15
July 2024 gunmen attacked the Imam Ali mosque, a Shia mosque in Wadi al-Kabir, Muscat, during
Ashura; six people and the three attackers were killed and Islamic State claimed it (as Wikipedia's
article summarises BBC, Reuters and AP, 16 July 2024; the reports themselves not opened). A Shia
split would place Oman's Shia in Muscat and on the Batinah coast. Ibadis and Sunnis: I found no
safety point.

## The detail

**Levels, from documents opened.**
- J.E. Peterson, "Oman's Diverse Society: Northern Oman", *Middle East Journal* 58(1), 2004, p.32
  note 1: the indigenous population "may be some 45% Ibadi (concentrated in the historic heartland of
  the country in the northern interior), approximately 50% Sunni (scattered elsewhere in northern
  Oman and the overwhelming majority in far eastern and southern Oman), and probably less than 5%
  Shi'i and Hindu". No source is given for the figures. Shia: the Lawatiyya (mostly Matrah, some
  in Saham, Barka, al-Masna'a and al-Khabura), under a dozen Baharina families (Muscat), and the
  'Ajam (Muscat and Matrah, largely assimilated).
- Peterson, "Oman's Diverse Society: Southern Oman", *MEJ* 58(2), 2004, p.254: "Dhufaris are
  entirely Sunni" (nearly all Shafi'i). No figure.
- A.K. Majidyar, *Is Sectarian Balance in the UAE, Oman and Qatar at Risk?* (AEI, 2013): "about
  three-quarters of Omani nationals ... are Ibadi"; Shia 5% of citizens; citing Pew, the State
  Department and Wikileaks cables.
- US State Department, *2023 Report on International Religious Freedom: Oman* (the ecoi.net copy;
  state.gov returns 403): 95% Muslim, "45 percent Sunni, 45 percent Ibadhi, and 5 percent Shia", of
  everyone, not of citizens; the government publishes no figures.
- CIA World Factbook (the factbook.json mirror): citizens "Ibadhi and Sunni sects each constitute
  about 45% and Shia about 5%".
- Badr al-Abri's blog (2023): 77% Sunni, 21% Ibadi, 2% Shia, quoted from a television programme,
  with no source and his own doubts.

**Searched, nothing below the nation:** English and Arabic web searches for sect by governorate or
wilaya; Marc Valeri on Oman's Shia (paywalled; its abstract's 3% is national); the Review of
Nationalities 2024 article (CIA figures); a UCLA paper on Ibadi Oman (none); the Ministry of
Endowments' mosque statistics (not published by madhhab as far as found); ARDA's national profile
(World Religion Database, national only, and ascription).

**Options.**
- (a) Keep one Islam, as built.
- (b) Dhofar's 238,811 Omanis on `islam.sunni` on Peterson's "entirely Sunni", everyone else on
  `islam`. A named split with the remainder on the parent; one author, 2004, no figure, and
  Dhofar's register also holds northern Omanis working there.
- (c) A three-way split everywhere from Peterson's description scaled to a national figure. I would
  not: the national Ibadi share itself runs from 21% to 75%, so the scaling has no anchor, and it
  would place the Shia.

My lean is (a); (b) is defensible if you want the Ibadi question visible at all. Nearby rulings:
Iran was split from Masaili's province table (2026-09-16), a real per-province estimate that Oman
has no counterpart to; Pakistan's partial Punjab Shia shares were refused (2026-09-16); ask 040
asks the same question for Saudi Arabia.
