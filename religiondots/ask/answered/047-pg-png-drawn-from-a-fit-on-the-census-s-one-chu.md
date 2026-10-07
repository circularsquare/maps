# 047 — pg: PNG drawn from a fit on the census's one church per province; DHS 2016-18 registration would replace it

Summary: PNG is drawn: each province's largest church is the 2011 census's figure, the other churches are fitted. Registering with DHS (your name) would give a real church mix for all 22 provinces. Register, or leave it?

*Filed 2026-10-03 by session `fafd1067-pg`. Anita's call; nothing is waiting on it.*

## What I did

Drew Papua New Guinea at its 22 provinces, every dot `modelled`. Each province's largest church is
the 2011 census's own printed figure (Morobe 67.0% Lutheran, Oro 60.6% Anglican, Bougainville 68.4%
Catholic...); the other ten churches are fitted so that each adds up to its national 2011 total,
with Catholics shaped by the Catholic Church's 2004 diocese figures. The note says which parts are
the census's and which are estimates. `sources/pg.md` is the record.

## What it costs to reverse

Registering: you sign up at `dhsprogram.com/data/new-user-registration.cfm` (name, country, a
stated research purpose; "Individual Researchers" is an accepted organisation type) and request
PNG 2016-18; an agent then rebuilds `pg` from the women's and men's files, about an hour. Not
registering costs nothing: PNG stays as drawn.

## Why it is yours rather than mine

AGENT_BRIEF §3: it needs an account in your name, and DHS terms bar redistributing the data.

## The detail

- **What PNG publishes.** Religion only for the whole country, plus one line per province in the
  2011 and 2000 National Reports: the largest church and its share. Everything else in this build
  is an estimate, and its known weakness is that a church strong in a few provinces without being
  the largest anywhere (Lutherans in the Highlands, Adventists, the United Church outside Papua) is
  spread at one rate everywhere else.
- **What DHS 2016-18 would give.** Religion asked of about 15,200 women and 7,300 men aged 15-49,
  representative for each of the 22 provinces, on eleven codes. Two limits, both in the final
  report: there is no Baptist code, and "Other Christian church" is 21.3% (against 9.7% in the 2011
  census), so a fifth of each province would sit on a catch-all; and it covers adults of 15-49
  only. It would still be measured shares in every province instead of one measured share and ten
  fitted ones.
- **The other route, for completeness.** The 2000 census printed "Basic Tables" for each province
  (three volumes a province, National Statistical Office 2002). Google Books' index of them lists
  religion terms such as Salvation Army and Jehovah's Witness, so a full religion table for each province
  (possibly by district too; not seen) is in print. None is online: Google Books is snippet-only, HathiTrust is
  search-only behind a Cloudflare check, and the National Library of Australia, University of
  Michigan and UC Berkeley hold paper copies. A scan request (NLA Copies Direct, for instance) would
  cost money and your details. Older than DHS, but a census count of everyone, and possibly at
  district level; the only reason I put DHS first is that it can be had today without a library.
