# Language coverage sweep — brief

Goal: for every country, find out what its census (or register, or best open substitute) says
about **which language people speak**, and how finely that is published. This decides whether a
world first-language dot map (a sister to `religiondots`) is worth building. This is a
**coverage sweep, not a build**: find and describe sources, do not download or process data.

## What to record per country

One CSV row per country in your region file, columns exactly:

```
iso3,country,question,census_year,finest_level,n_units,access,categories,licence,source_url,older_census,fallback,confidence,notes
```

- `question` — one of:
  - `mother_tongue` (first language learned in childhood)
  - `home_language` (language spoken at home / most often at home)
  - `native_language` (ex-USSR style "rodnoy yazyk" — often identity, not use; say so)
  - `main_language` (main / best-known language, e.g. Switzerland)
  - `languages_spoken` (multi-response "languages you can speak", no single first language)
  - `ability_minority` (only asks about one or a few languages, e.g. Welsh, Irish, Basque)
  - `indigenous_only` (asks only about indigenous languages, e.g. Mexico, Brazil)
  - `ethnicity_only` (no language question, but an ethnicity question a crosswalk could use)
  - `none` (neither)
  - `unknown` (could not determine — say what you tried)
  If a census asks several (Canada asks mother tongue AND home language), list the best one for
  a one-dot-per-person first-language map first, then the others in `notes`.
- `census_year` — latest census (or register year) with that question.
- `finest_level` — the finest geography at which language counts are **openly published**
  (e.g. "district (C-16 tables)", "municipality", "tract", "province only", "national only").
- `n_units` — approximate count of units at that level (e.g. 640). Blank if unknown.
- `access` — `open_table` (xls/csv/API), `pdf`, `microdata_server` (REDATAM, PopGIS,
  table builder — free tabulations), `uscb_hdx` (US Census Bureau geodatabase on HDX), `gated`
  (account/application), `none`.
- `categories` — rough count and character of language categories ("~270 mother tongues",
  "9 incl. 'other'", "Arabic only, no varieties", "lumps Bhojpuri into Hindi").
- `licence` — what reuse terms say if visible, else blank.
- `source_url` — the single most useful URL.
- `older_census` — an older census with a language question if the latest lacks one
  (e.g. Turkey 1965 by province). Blank otherwise.
- `fallback` — if no census language data: what could stand in (ethnicity question +
  crosswalk, Afrobarometer home language by region, a national survey, a dialect atlas,
  population registers…). Name it specifically.
- `confidence` — `checked` (you saw the table/questionnaire), `reported` (a secondary source
  says so), `guess` (from general knowledge only — flag it, it's still useful).
- `notes` — anything that matters: identity-vs-use caveats, politically sensitive question,
  multi-response, data only for one minority language, notable lumping.

**Negatives are records, not verdicts.** If you find nothing, write *what you asked, of what,
and what came back* in `notes` (e.g. "searched INS site + IPUMS variable list, 2026-10-03; 2018
census questionnaire has ethnicity but no language item"). Never write "not possible".
Stopping early on a country is fine — a coarse answer for every country beats a perfect answer
for a few. Mark it `guess` and move on.

## Cheap oracles — use these before searching per country

- **IPUMS International variable pages are public** even though the data is gated:
  `https://international.ipums.org/international-action/variables/group?id=...` and the
  per-variable pages (search "IPUMS international LANG variable availability" or open
  `https://international.ipums.org/international-action/variables/LANG#availability_section`
  and similar: `LANG`, `LANGSEC`, plus country-specific `XX####_LANG...` source variables).
  The availability table says which census samples asked a language question. This is the
  single best "did census X ask language?" oracle. Ethnicity is `ETHNIC`.
- **US Census Bureau country geodatabases on HDX** carry census tables *with* boundaries.
  Already checked: language tables in Indonesia (2020 census), Pakistan (2017), Ethiopia
  (2007), Ukraine (2001), Central African Republic (estimates), Mali (projections).
- **REDATAM web servers** (Latin America, some Caribbean/Africa) run free tabulations of
  census microdata down to fine units; the variable picker on
  `.../RpWebStats.exe/Frequency?BASE=<base>&ITEM=FREQPOB&lang=esp` lists every variable.
  Known live: Nicaragua, Panama, Peru, Ecuador, Cuba, Argentina, Colombia, Bolivia, Uruguay,
  Paraguay, Trinidad, Suriname (prod.redatam.org hosts several; BASE is not namespaced by
  country — check the CGI dir per deployment).
- **SPC PopGIS** (`<country>.popgis.spc.int`) for Pacific islands.
- **Census questionnaires** (often on the UNSD census knowledgebase or the office site) settle
  "was it asked" definitively.

## Rules

- **WebSearch is capped per session and shared across all agents.** Your budget is stated in
  your prompt — stick to it. WebFetch is not capped; fetch pages you already know of freely.
- **Never put any personal name or email in a User-Agent, URL, or request.** Use a generic
  browser UA if one is needed.
- **Do not fetch citypopulation.de** (its robots.txt bans ClaudeBot).
- Gated sources (registration, application): note them, don't sign up or apply.
- Bot walls (Cloudflare, 403, 418): retry once with a browser UA, then try the Wayback Machine
  (`http://archive.org/wayback/available?url=...`); otherwise record the URL and move on.
- Don't download large files. Small probes (a questionnaire PDF, an index page, a variable list)
  are fine. Write only to your own region file.
- Prefer the office's own publication over Wikipedia, but Wikipedia "Languages of X" pages are
  a fine pointer to which census asked what.

## Output

Write `C:\Users\anita\projects\maps\languagedots\coverage\<region>.csv` (the CSV above, UTF-8,
quote fields containing commas) and a short `<region>.md` beside it: 5–15 lines on the
region's overall picture, the best sources found, the biggest gaps, and any surprises. Country
names in prose in Latin characters; source labels in CSV may keep native script.
