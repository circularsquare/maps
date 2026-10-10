# Operator colours, logos and short names: sources

Research note, 2026-10-07. Nothing in the build or app was changed. Scratch scripts and raw results are in
the session scratchpad (`ops.py`, `rels.py`, `sample.py`, `props.py`, `adj2.py`, `logocol.py`).

## How the app gets operators today

- `dist/data/<cc>/lines.json`: each line has `operator` and `operator_en`, copied by `build_model.py` from
  the OSM route / route_master tags `operator` and `operator:en` (register lines get them from the
  matched OSM line). There is no operator id anywhere.
- The app groups by `opKey()` in `dist/index.html`: the raw string, each `;` part swapped for an English
  name learned from any line that has both tags (`OP_EN`), then sorted and joined.
- Counted the same way over all 73 regions (non-service lines): **1,808 operator keys**, 1,760 of them
  single operators. Per country: jp 215, de 278, ch 158, cn 128, us 136, ru 114, fr 86, gb 64, in 46,
  kr 18. Lines with no operator at all come to about 170,000 km, the biggest row by far.
- `extract.py` keeps the relation's `wikidata` (the line item) but drops `operator:wikidata`, so no id
  reaches the build today. On OSM, `operator:wikidata` is on 30% of `route=train` relations,
  `network:wikidata` on 52%, `operator:short` on 9% (taginfo, today).
- `colours/<cc>.csv` and `line_colours.py` colour lines, not operators. `GENERIC` holds Korail's
  corporate blue (`0066B3`, `0066BC`) only so that it can be skipped as a line colour.

Two bugs turned up while counting. Neither was fixed.

- **Network Rail shows as "Transport for Wales" with 18,045 km.** One gb line has `operator=Network Rail`
  with `operator:en=Transport for Wales`. `OP_EN` learns that one pairing and applies it to all 447
  Network Rail register lines.
- **Keys collide across countries.** India's Southern Railway and South Western Railway zones share
  rows with the British train companies of the same names.

An operator table keyed on (country, raw string) would fix both.

## Sources

| Source | Gives | Licence | Coverage / notes |
|---|---|---|---|
| Wikidata operator item | P154 logo, P1813 short name, P465 colour, P749 parent | CC0 data; logo files carry their own Commons licence | Logos good, short names fair, colours almost none (trial below) |
| Wikidata line items → P137 | an operator QID for an OSM operator string | CC0 | The join path (below) |
| OSM `operator:wikidata`, `operator:short` on routes | QID and short name per route | ODbL | 30% / 9% of train routes; needs `extract.py` to keep the tags |
| en-Wikipedia `Module:Adjacent stations/*` | `system color`, `system icon` per system | CC BY-SA (text); a colour is a fact | 1,341 modules, 431 with a system colour, 540 with an icon. Mixed quality: some are `000`, `FF0000`, `008000` |
| Colour taken from the logo file | most-used saturated fill in the SVG/PNG | derived | 10 of 12 test logos gave a plausible brand colour (DB `EC0016`, CN `DA291C`, China Railway `E60012`) |
| Name Suggestion Index, `data/transit/route/{train,subway,light_rail,tram}.json` | network/operator + QID + `*:short` | BSD-3 | About 600 entries, mostly networks (tariff unions); 109 train entries name an operator. Hit 18 of the 64 trial operators. Useful as a cross-check, not as a base |
| NSI `data/operators/route/railway.json` | infrastructure operators on track ways | BSD-3 | 72 entries, 69 with a QID |
| Transitland Atlas (DMFR feeds) | operator `name`, `short_name`, `tags.wikidata_id` | CC BY 4.0 | Mostly North America; no colours. Good for US/CA short names |
| GTFS `route_color` | per-route colour | feed licence | A line colour, not a brand colour. Not useful here |
| OSM operator-level objects | none | | OSM has no operator object; `colour` sits on routes. `brand:*` is for shops |

No source found had brand colours in bulk. Wikidata barely records them for companies. P465 is used on
lines, parties and sports teams.

## Coverage trial (64 operators)

Picked from our own operator keys: biggest by km, at most two per country, plus JR West, Korail, SBB,
Tokyo Metro, Seoul Metro, JR Central and some urban "Metro" operators.

**Joining to a QID.** First join: the line items on the operator's OSM routes, then their P137, counting
votes. Second join: Wikidata search on the raw string in the country's language, then on the English
name, keeping only hits whose P17 is the country.

- 62 of 64 joined automatically, and 55 of those were right (86%).
- Both joins gave the same item on 23 of the 25 operators where both ran.
- Line votes are reliable at 3 or more. With 1-2 votes they were wrong for RFI and SNCF Réseau (both
  gave the SNCF group).
- Search on its own got NS wrong (missed), ScotRail wrong (the 2015-2022 Abellio franchise) and US
  Metrolink wrong (St Louis).
- Nothing found for АО «Экспресс-пригород» or Nanjing Metro Group.
- Expect about 1 in 7 to need a hand fix.

**What the 64 items carry:**

| Field | On the item | Note |
|---|---|---|
| Logo (P154) | 52 / 64 (81%) | Parent-company logos are not a usable fallback: they turned up government seals (UK, US, Romania, Serbia) |
| Short name (P1813), any language | 24 / 64 (38%) | Often only native: `JR東日本`, `ČD`, `ZSSK`, `NMBS`/`SNCB` |
| Short name in English | 7 / 64 (11%) | JR East, JR West, JR Central, CN, CPPK, ScotRail, SBB |
| Colour (P465) | 2 / 64 (3%) | ÖBB `ED1834`, SBB `EC1B24`. 3 with the parent's |
| Colour from Adjacent stations | 15 / 64 matched by name | About 11 usable. Metrorail (South Africa) matched LA Metro Rail; Metrolinx and China Railway are `000` |
| Colour from the logo | 10 of 12 tested | Logos exist for 81%, so this is the route to wide coverage |

**Logo licences.** The 93 logo files found across the sample (including parents) are all free on
Commons: 88 public domain (73 of them tagged PD-textlogo, which is most of them), 2 CC0, 2 CC BY-SA 4.0
(Metro Trains Melbourne, John Holland) and 1 Korean government licence. 68 carry the Commons trademark
warning.

Fetching the files at scale is slow. After about a dozen downloads `upload.wikimedia.org` started
returning 429 to a plain User-Agent, so the colour-from-logo test stopped at 12. A bulk fetch needs slow
pacing and a policy-compliant User-Agent: a project URL as the contact, not a personal name or email.

## Joining our operator strings to an id

1. Key every operator by **(country, raw `operator` string)**, split on `;`. This also fixes the two bugs
   above.
2. Get a QID in this order:
   1. `operator:wikidata` from OSM, once `extract.py` keeps it. Add `operator:wikidata` and
      `operator:short` to `REL_TAGS`.
   2. Line items → P137, with 3 or more votes.
   3. Search in the country's language, filtered on P17.
   4. A hand entry.
3. Collapse different raw strings that reach the same QID into one operator: 東京地下鉄 / 東京メトロ,
   東日本旅客鉄道 / East Japan Railway Company. This is the alias table the comment in `index.html` asks
   for.

## Recommended plan

- Add a hand-kept table, **`colours/operators.csv`**, with columns
  `cc, operator, qid, short, colour, colour_src, logo, logo_licence, group, note`. `operator` is the raw
  string as it appears in the data, and `group` is optional (see the question below).
- Write a `--seed` script that fills it from Wikidata: the QID from the join above, then label, P1813,
  P465, P154 plus its Commons licence.
  - `short`: take the English P1813, then a Latin-script P1813. Otherwise leave it blank for a hand entry.
  - `colour`: take P465, then the Adjacent stations system colour (dropping black and white), then the
    colour taken from the logo. Mark each with `colour_src` (`wikidata` / `adjstations` / `logo` /
    `hand`), as `line_colours.py` already does.
- Hand-check the top 100-150 operators by km, roughly 80% of the km that has an operator. Leave the long
  tail to the seed, falling back to today's blue when there is no colour.
- Like `colours/<cc>.csv`, the table wins over everything else and the seed never overwrites a filled
  cell.
- The build writes a small `dist/data/operators.json` (key → short, colour, logo path) and the app reads
  it in `opKey`/`buildOps`.
- Logos, if used: copy the chosen files into `dist/` as small SVG/PNG and credit them on an about page.
  Do not hotlink Commons.

## Logos on a public site: caveats

- **Copyright.** PD-textlogo means the logo is too simple to be copyrighted, under US law and Commons
  policy. That makes copying it fine in the US. In some countries (Germany, the UK, Japan) the threshold
  is lower, and Commons often marks a file "PD in the US, maybe not at home". CC BY-SA logos need an
  attribution line.
- **Trademark.** This applies to all of them, whatever the copyright status. A small logo beside the
  operator's own name, used to identify that operator, is ordinary nominative use and generally fine.
  Problems start if it suggests endorsement, gets altered, is used as the site's own branding, or is
  sold (for example on posters).
- **Safest choice.** Colour plus short name gives most of the benefit with no logo risk. If logos are
  added, keep them small, unaltered, labelled, and only from Commons files marked free.
- **Logos go out of date.** Renfe's file is "2005-2026" and ScotRail's item is new. Record the file name
  and check it now and then.

## 30 proposed rows

Colour sources: `wd` = Wikidata P465, `adj` = Adjacent stations system colour, `logo` = taken from the
logo file in this trial, `kr` = Korail's corporate blue already in `line_colours.py`. A blank colour means
no source yet (the logo exists but was not fetched). Short names were filled by hand where Wikidata has
no English one. All logos are on Commons (`https://commons.wikimedia.org/wiki/File:<name>`).

| cc | operator (as in our data) | QID | short | colour | src | logo file | licence |
|---|---|---|---|---|---|---|---|
| cn | 中国铁路 (and the 中国铁路…局集团 rows) | Q1073489 | China Railway | #E60012 | logo | China Railways.svg | PD, TM |
| fr | SNCF Voyageurs | Q93090957 | SNCF | #F00000 | logo (JPG, approx.) | LOGO SNCF GROUPE CMJN.jpg | PD-textlogo |
| de | DB Fernverkehr | Q452140 | DB | #ED1C24 | logo | Db-bahn.svg | PD-textlogo, TM |
| de | DB InfraGO | Q122870674 | DB InfraGO | #EC0016 | logo | DB InfraGo logo.svg | PD-textlogo, TM |
| fr | SNCF Réseau | Q21605526 | SNCF Réseau | #D80810 | logo | SNCF Réseau.png | PD-textlogo, TM |
| gb | Network Rail | Q1501071 | Network Rail | | | (none on item) | |
| pl | PKP Polskie Linie Kolejowe | Q1344677 | PKP PLK | #004681 | logo | PKP PLK logo.svg | PD |
| it | RFI | Q1060049 | RFI | #006A6A | logo | Rete Ferroviaria Italiana logo.svg | PD-textlogo, TM |
| kz | Қазақстан темір жолы | Q1069105 | KTZ | #0090D8 | logo | Logo Kazakh railway.jpg | PD-textlogo, TM |
| es | Renfe | Q2476154 | Renfe | #830065 | adj | Logotipo de Renfe Operadora (2005-2026).svg | PD-textlogo, TM |
| ca | Canadian National | Q624798 | CN | #DA291C | logo | CN Railway logo.svg | PD-textlogo, TM |
| pl | Polregio | Q1139703 | Polregio | #C52121 | adj | Logo PolRegio.svg | PD-textlogo, TM |
| jp | 東日本旅客鉄道 | Q499071 | JR East | #0A8C0D | adj | JR logo (east).svg | PD-textlogo, TM |
| us | Amtrak | Q23239 | Amtrak | #00537E | adj | Amtrak logo.svg | PD-textlogo, TM |
| us | BNSF Railway | Q267122 | BNSF | | | BNSF Railway Company logo.svg | PD-textlogo, TM |
| us | Union Pacific Railroad | Q725793 | Union Pacific | #00377C | adj | Union pacific railroad logo.svg | PD-textlogo, TM |
| at | ÖBB-Personenverkehr AG | Q83822 (group) | ÖBB | #ED1834 | wd | Logo ÖBB.svg | PD-textlogo, TM |
| ch | SBB | Q83835 | SBB | #EC1B24 | wd (adj says FF0000) | SBB CFF FFS logo.svg | PD-textlogo, TM |
| nl | Nederlandse Spoorwegen | Q23076 | NS | | | Nederlandse Spoorwegen logo.svg | PD-textlogo, TM |
| hu | MÁV-Start | Q1180332 | MÁV | | | MÁV Személyszállítási Zrt.png | PD-textlogo |
| be | NMBS/SNCB | Q524255 | SNCB | | | SNCB logo white.svg | PD-textlogo, TM |
| jp | 西日本旅客鉄道 | Q502125 | JR West | #006CC5 | adj | JR logo (west).svg | PD-textlogo, TM |
| it | Trenitalia | Q286650 | Trenitalia | | | Trenitalia logo.svg | PD-textlogo, TM |
| kr | 한국철도공사 | Q18169 | Korail | #0066B3 | kr | Korail logo.svg | PD-textlogo, TM |
| au | Queensland Rail | Q379439 | Queensland Rail | #FA6432 | adj | Logo QR.svg | PD-textlogo, TM |
| ie | Iarnród Éireann | Q73043 | Irish Rail | | | Irish Rail Logo.svg | PD, TM |
| jp | 東海旅客鉄道 | Q513679 | JR Central | #EE6D00 | adj | JR logo (central).svg | PD-textlogo, TM |
| jp | 東京地下鉄 / 東京メトロ | Q682894 | Tokyo Metro | #0C9ED4 | adj | Tokyo Metro logo (full).svg | PD-textlogo, TM |
| cz | České dráhy | Q304944 | ČD | | | Ceske drahy-logo.svg | PD-textlogo, TM |
| kr | 서울교통공사 | Q28699048 | Seoul Metro | | | Seoul Metro.svg | PD-textlogo, TM |

Questions for Anita:

- **Should subsidiaries fold into one row?** DB Fernverkehr, DB Regio AG, DB Regio Bayern, DB Regio NRW
  and so on would all become "DB". The `group` column would allow it.
- **Should infrastructure managers stay in the Operators list?** Adif, DB InfraGO, SNCF Réseau,
  Trafikverket, Väylävirasto and PKP PLK are there because register lines carry the infrastructure
  manager as `operator`.
