# 055 — bh: Bahrain: split Bahraini Muslims into Shia and Sunni by governorate?

Summary: Sunni and Ja'fari Endowments' mosque counts place Shia at about 78% in Capital and Northern, 20% in Muharraq and Southern; OSM tags agree; level 57% from Arab Barometer 2009. Sensitive (2011 crackdown). Draw it, or keep one Islam?

*Filed 2026-10-03 by session `fafd1067` (supervisor). Anita's call; nothing is waiting on it.*

## What I did

Kept Bahrain as drawn: Bahraini Muslims on one `islam` node, per asks 040/043 (Gulf citizens stay one
Islam unless a cited regional source exists). Did not spawn the split. The source now exists, so the
question is only whether to draw it.

## What it costs to reverse

Saying yes is one builder, about an hour: a partial rebuild of `bh` with `islam.shia` and `islam.sunni`
by governorate, a note, a rescatter.

## Why it is yours rather than mine

AGENT_BRIEF.md §3 and spec §14: a sect map of a country whose Shia majority was the target of the 2011
crackdown, drawn from the state's own endowment records. The 040/043 rulings anticipated "update if
regional estimates turn up", which this is, but the §14 weight is yours.

## The detail

- Source: `sources.md` §scout-2026-10-03-sect-registers; `sources/branches.md`, last section.
- Ja'fari Endowments 2016, mosques by governorate: Northern 344, Capital 332, Muharraq 44, Southern 33
  (ma'tams 306/211/71/31). Sunni Endowments, about 2022: Capital 92, Muharraq 168, Southern 130,
  Northern 89. By mosques: 78-79% Shia in Capital and Northern, 20-21% in Muharraq and Southern.
- Second placement: OSM's Shia-tagged mosques (604 of 841) fall the same way.
- Level: Arab Barometer wave I (2009), 249 Shia and 183 Sunni of 435 Bahraini respondents, 57.2%;
  the mosque shares applied to the 2020 census's Bahraini Muslims give 56.7%. Not checked: whether that
  sample had a sect quota (Lebanon's surveys did).
- Caveat recorded by the scout: mosques rank places but do not count people; the mosque counts alone
  range 57-73% Shia nationally, so the survey sets the level and the registers only the geography.
- Options: (a) draw it as above, rolled back to `islam` when inferred dots are hidden, with a note
  naming both endowments and the survey; (b) keep one Islam.
