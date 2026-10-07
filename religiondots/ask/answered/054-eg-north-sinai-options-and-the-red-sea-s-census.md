# 054 — eg: North Sinai options, and the Red Sea's census figure

Summary: South Sinai and New Valley are now drawn from old census tables. North Sinai: stay empty (my pick), or a borrowed rate? And should the Red Sea move from the survey's 0% (28 interviews) to the 1996 census's 10.5%?

*Filed 2026-10-03 by session `fafd1067-gaps`. Anita's call; nothing is waiting on it.*

## What I did

Drew South Sinai at the 1996 census's 5.03% Christian and New Valley at the 1976 census's 1.8%,
because both were measured (your Ecuador line: "whether anything measured the place"). Left North
Sinai empty, because nothing printed measures it. Left the Red Sea on the survey's 0%.

## What it costs to reverse

A few lines in `sources/eg_census.py` and a rescatter of Egypt, about two minutes, plus the build tail.

## Why it is yours rather than mine

Filling North Sinai would draw a place nothing measured, and the brief told me not to do that on my
own. Moving the Red Sea changes a governorate that is already drawn (AGENT_BRIEF §3).

## The detail

**North Sinai** (475,331 people, 0.44% of Egypt). Options, with what each draws:

- stay empty, hatched (as built);
- South Sinai's 1996 rate, 5.03%: about 23,900 Christians. Probably too high: South Sinai is a
  tourist-coast workforce (Sharm el-Sheikh), North Sinai is al-Arish and Bedouin country, and Coptic
  families were widely reported leaving al-Arish in 2017;
- the national rate, 6.01%: about 28,600;
- the 1976 census's "Sinai" figure, 1.0%: about 4,750. The right size, but it measured the strip
  Egypt held in 1976 (about ten thousand people), not today's governorate.

The real route would be the 1986 census governorate volumes, which did print religion; I could not
find them online.

**Red Sea** (417,930 people). The Arab Barometer drew it 0% Christian because none of its 28
respondents said Christian. The 1996 census table (copied in the statistics agency's library by
Arab-West Report) counts 10.48%, 16,315 of 155,695, of whom 11,078 were foreigners; the 1976 census
had 4.4%. The note now mentions the 10.5%. Moving the Red Sea to the 1996 rate would add about 43,800
Christians. I lean towards moving it, since 28 interviews against a full census count is not close,
but it is the same kind of call as the frontier.

Sources and checks: `sources/eg.md`, "Frontier governorates from the census, 2026-10-03".
