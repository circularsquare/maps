# 046 — bt: Bhutan's Hindus placed by Nepali mother tongue, a census-withheld religion after the 1990s expulsions

Summary: Bhutan is drawn: Hindus (18%, the Centre for Bhutan Studies' survey) placed by dzongkhag on Nepali mother tongue, which the Centre publishes. The state asked religion in 2005 and withheld it. Keep, or one national share?

*Filed 2026-10-03 by session `fafd1067-bt`. Anita's call; nothing is waiting on it.*

## What I did

Bhutan is drawn at its 20 dzongkhags. The level is the Centre for Bhutan Studies' estimate from its
2010 Gross National Happiness survey (81% Buddhist, 18% Hindu, 1.2% Christian, Bhutanese 15+). The
Hindus are placed in proportion to Nepali mother tongue by dzongkhag, which the Centre's 2015 survey
report prints (Table A1.5): 55% of Samtse's Bhutanese are drawn Hindu, 14% of Thimphu's, under 1% in
the east. Christians are drawn at 1.2% in every dzongkhag. Non-Bhutanese (45,425, mostly Indian
hydropower workers) are drawn by dzongkhag from the census, at UN DESA's origin mix through Pew.

## What it costs to reverse

One word: `HINDU_PLACEMENT = "national"` in `sources/bt.py` draws the Hindus at 18% in every
dzongkhag; re-run it, two scatters and the note's placement paragraph, about 15 minutes. Taking
Bhutan off the map is `ORDER` and the countries entry.

## Why it is yours rather than mine

AGENT_BRIEF §3, first bar, and spec §14 rule 2. The Hindus drawn are the Lhotshampa, the
Nepali-speaking people of the south, over 100,000 of whom left or were expelled to camps in Nepal in
the early 1990s. The state asked everyone's religion in the 2005 census and never published it,
which is the shape spec §14.5 calls the Egypt case (collected and withheld).

## The detail

- **Why I think it is the reflect case, not the reveal case.** The input that places them is
  published: the Centre (a government-funded research institute) prints Nepali mother tongue by
  dzongkhag, and the map's Hindu share in each dzongkhag is that column times 0.985. Nothing finer
  than dzongkhag is used (about 34,000 Bhutanese each), and that Lhotshampa live in the southern
  foothills is in every account of the country. Whether the Centre counts as "the state" for rule 2
  is the open point from the Zanzibar ruling; I read it as yes.
- **What I did not do.** The 2005 census microdata hold religion by gewog (205 units); they are
  released only on a written application with a signed undertaking under Bhutanese law, which needs
  your identity, and using them would publish what the state chose not to. I recommend not applying.
- **Christians are the group the state restricts now** (no registered church; proselytising is an
  offence), so they are held at one national share, which places nobody.
- **The model's weak point**, said in the note: no source crosses religion with language, so
  "Nepali-speaking means Hindu" is assumed; Buddhist Tamang and Gurung families and Hindus with
  another mother tongue are misplaced, and nothing by dzongkhag can check it.
- **The national share option** puts 18% Hindu dots in Bumthang, Gasa and Trashigang, where the
  language table has about 1-4% Nepali speakers; the note would then say the Hindus' location is
  not drawn.
- Record: `sources/bt.md`; `sources.md §bt-2026-10-03`.
