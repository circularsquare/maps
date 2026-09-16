# 038 — lb: Lebanon: WVS wave 7 (and 6) Lebanon download

Summary: For the Lebanon build you approved: please download WVS Wave 7 Lebanon (CSV), and wave 6 Lebanon if offered, from worldvaluessurvey.org's data page, the same form as Puerto Rico's, into data/raw/lb/. A builder starts once it lands.

*Filed 2026-09-15 by session `cb8b206e` (supervisor). Anita's call; nothing is waiting on it.*

## What I did

Nothing is built. You approved Lebanon on ask 035; the build waits for the data file, which sits behind the WVS download form.

## What it costs to reverse

Nothing. Without the file Lebanon stays off the map.

## Why it is yours rather than mine

It is a download only you can make: the WVS form asks for a name and purpose, as it did for Puerto Rico.

## The detail

- **Where:** worldvaluessurvey.org, Data & Documentation, WVS wave 7 (2017-2022), then the per-country files; pick **Lebanon, CSV**. Puerto Rico's came as `F00013157-WVS_Wave_7_Puerto_Rico_Csv_v5.1.zip`, so Lebanon's should look the same with a different number.
- **Also, if it is offered:** WVS wave 6 (2010-2014) Lebanon. A second wave lets the builder check the sect shares by splitting the respondents in half, which one wave alone cannot do.
- **Put them in** `data/raw/lb/` (or leave them in Downloads and say so).
- **Why the file and not the online tool:** the first thing the builder does is test whether the sect mix by governorate was a fieldwork quota, like the Arab Barometer's. That is quicker and more reliable from the file than by scripting the website's analysis tool.
