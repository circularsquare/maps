# Legal check before release (T-1824)

2026-10-29, agent t1824. Anita's question: "briefly check for legal stuff? are there any laws we have to be
aware of". This is research, not legal advice. Nothing in the code was changed. Sources are linked in the
text; "unverified" marks what I could not open or only know from memory.

Scope assumed: a free game, no ads, no payments, run by one person in the US, on her own site, with optional
accounts on her own server. Several things below change if that changes (see "If you ever earn money").

## The short version

The game is in better shape than most. Credits, a terms page and a privacy page already exist, and the OSM
and OpenFreeMap credits are on the map itself. The problems are all about the privacy page no longer matching
what the published game does:

1. **The privacy page says "no tracking cookies", but the published game page loads Google Analytics.**
   `website/_includes/analytics.html` (GA4) is included in `website/trainsworld/index.html` line 8, and the deploy
   script adds it on purpose. The 2026-10-29 decision for the new playtime heartbeat said "no Google Analytics",
   but the include is still there.
2. **The game page loads Google Fonts on every visit** (`app/index.html` lines 9-11, Zen Maru Gothic). That sends
   each visitor's IP address to Google. The privacy page does not say so. A Munich court fined a site owner 100
   euros for exactly this in January 2022 ([report](https://wptavern.com/german-court-fines-website-owner-for-violating-the-gdpr-by-using-google-hosted-fonts),
   case 3 O 17493/20). The fix is small: the app already ships Nunito as a local file, and Zen Maru Gothic is under
   the same open font licence.
3. **The privacy page does not mention** the playtime heartbeat (added 2026-10-29, the page is dated 25 October),
   or the feedback form (name, email and message are stored on the server and mailed on).
4. **A near-identical game name exists**: "Train World", a railroad simulator by Turn Point Games on
   [Steam](https://store.steampowered.com/app/2132350) (July 2023, 34.99 dollars) and Epic. Same genre, one letter
   apart. Probably fine for a free hobby game, but it is the one name clash I found.
5. **Credits miss several sources**, one with a required wording: TfL's licence requires "Powered by TfL Open Data".

## 1. Data, maps, fonts, sounds

"Done" means checked against the app: `app/src/ui/Credits.tsx` (Settings, Data sources, closed by default),
the map's corner attribution (`map/mapView.ts`, `map/realNetwork.ts`), `public/fonts/OFL-*.txt`.

| Item | What it asks | Done today | To add | Priority |
|---|---|---|---|---|
| OpenStreetMap (ODbL) | Say "OpenStreetMap", link to the copyright page or say it is under ODbL; visible in a corner of the map (collapsed behind an (i) is allowed if the licence is findable) or on a credits screen ([OSMF guidelines](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines)). A derivative database (extracts you transform and publish) must be offered under ODbL, with the attribution and a pointer to the licence | Map corner has "Data from OpenStreetMap" linking to the copyright page; credits say "Map data © OpenStreetMap contributors, under the ODbL" and that the data files are ODbL; the packs are fetchable from data.anita.garden | Nothing needed for the game itself (it is a "produced work", its code stays yours). Optional: one line in credits saying where the data files can be fetched | done |
| Pack files as an ODbL database | The packs (outlines, places, rail, parks, schools, hospitals from OSM) are a derivative database, so the whole file is offered under ODbL, and must not contain data you cannot license that way | Credits claim ODbL. T-924 notes: 9 outlines are filled with EuroGeographics NUTS polygons (non-transferable), and every pack still carries an unread `nocar` layer drawn from about 15 sources, several with no licence stated (AIRIA, Senatran, China public security ministry, NBS Nigeria) | Rebuild without the NUTS fills (T-925) and strip `nocar` (T-926) when the packs are next rebuilt. The `nocar` figures are national aggregates, so the risk is low | P3, with the next pack rebuild |
| OpenFreeMap and OpenMapTiles (basemap) | Credit "OpenFreeMap © OpenMapTiles Data from OpenStreetMap" (their stated wording). Their [terms](https://openfreemap.org/tos/): free, as is, may stop at any time, Hungarian law; the terms page does not discuss attribution or commercial use (ask info@openfreemap.org) | Exact credit is in the map corner and in credits | Nothing. Not legal: if OpenFreeMap stops, the basemap breaks. The Positron style's own licence was not checked (unverified) | done |
| Terrain Tiles (hill shading, AWS Open Data) | Link to Mapzen's attribution page, which lists the underlying sources | Link in map attribution and credits | Nothing | done |
| GHSL (population, built area, city list) | "Reuse authorised with proper acknowledgment of the source", © European Union 1995-2025 ([GHSL](https://human-settlement.emergency.copernicus.eu/download.php)) | In credits, with the © line | Nothing | done |
| Kontur Population | CC BY 4.0, credit | In credits | Nothing | done |
| GISCO NUTS boundaries (Europe zones) | Non-commercial, non-transferable; notice "© EuroGeographics for the administrative boundaries" on the map legend and the intro page ([GISCO](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units)) | In map attribution and credits | The intro page: the start menu does not show it, only the map corner and Settings do. Decide if that counts. Any ads, payments or sponsors need a EuroGeographics licence first | P3 (P1 if you earn money) |
| geoBoundaries, Natural Earth, OurAirports, Wikidata | geoBoundaries per-country licences (CC BY, ODbL, public domain); the others public domain or CC0 | In credits | Nothing | done |
| Wikidata and Wikipedia page views | CC0 per the data register | Wikidata named in credits; page views not named | Optional one line | P3 |
| World Bank, Maddison Project | CC BY 4.0 | In credits | Nothing | done |
| Eurostat | EC reuse, credit | In credits | Nothing | done |
| ASPECT population grid (China townships, Ju et al. 2025, CC BY 4.0 per helper1m) | Credit the authors | **Not in credits.** The register lists it as the source of every Chinese township count | Add "China township population: ASPECT (Ju et al. 2025, Scientific Data), CC BY 4.0". I could not open figshare to confirm the exact citation (403); the 2018 township outlines it is summed over: source not recorded in the files I read | P2 |
| Census and statistics offices by country | Per licence: OGL v3 (UK), Licence Ouverte 2.0 (France), dl-de/by-2-0 (Germany), StatCan Open Licence, Government of Japan terms 2.0 / CC BY 4.0, US public domain | In credits, with the wordings those licences ask for (OGL, StatCan non-endorsement, BKG link) | Brazil (IBGE, Metro SP survey: no licence stated), Korea (MOIS, KTDB: none stated), India 2011, Shanghai bureau: facts and aggregates only, low risk. T-924 listed these as unverified and nothing has changed | P3 |
| TfL timetables (London) | Licence is OGL v2 with TfL changes: **"Powered by TfL Open Data"** required when you adapt or use it ([TfL terms](https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service)) | **Not in credits** | Add the exact sentence | P1 |
| Network Rail SCHEDULE (London National Rail) and the GB GTFS from aubin.app | Network Rail's [data feeds licence](https://www.networkrail.co.uk/who-we-are/transparency-and-ethics/transparency/open-data-feeds/network-rail-infrastructure-limited-data-feeds-licence) works inside the OGL, with account terms. aubin.app's own terms were not read | Not in credits (OGL line names ONS and others, not these) | Add "Contains Network Rail data, OGL v3" plus aubin.app if it asks. Read the account terms you accepted | P2 |
| MTA GTFS (New York subway, LIRR, Metro-North) | [Terms](https://www.mta.info/developers/terms-and-conditions): free; do not say or imply MTA licensed your app or the data came from MTA's server; do not say the data is accurate or timely. They say you may not modify the data itself, and you may build an app that uses only some of it. Logos and trademarks are not mentioned | "US Census Bureau, NCES, FAA, BTS" only; no MTA line | Add "New York subway, LIRR and Metro-North timetables from MTA open data. Not affiliated with or endorsed by the MTA." The game turns the timetable into its own save (a change of form, not a copy of the feed), which I read as fine, a lawyer would be the one to say | P2 |
| NJ Transit and PATH GTFS | NJ Transit's developer terms sit behind a login ([portal](https://developer.njtransit.com/registration/login)); I could not read them. PATH's feed is on Trillium, terms not read | Not in credits | Read the terms you accepted when you registered for the NJ Transit feed, then add one credit line | P2 |
| TTC and Open Toronto; Metrolinx GO and UP | Metrolinx says its data is under the Open Government Licence Ontario, but its GTFS download adds a separate revocable Access and Use Agreement ([page](https://www.metrolinx.com/en/about-us/open-data)). TTC's Open Toronto licence not read | "Open Government Licence (Ontario)" is in credits | Name Toronto Transit Commission and Metrolinx. Read the Metrolinx agreement | P3 |
| Paris (IDFM GTFS) | Timetables are ODbL, with the Licence Mobilites added for some data ([PRIM](https://prim.iledefrance-mobilites.fr/en/licences), summary from search results, text not read). ODbL means credit and share-alike, which the ODbL claim on the data files already covers | "INSEE and geo.api.gouv.fr" only | Add "Timetables: Ile-de-France Mobilites (ODbL)" | P2 |
| Seoul and Tokyo timetables | Seoul from Seoul Metro and Korail open data via koreariders; Tokyo from the 2015 transport census and KSJ N02 and S12 | Japan census and MLIT in credits; Korea's Ministry of the Interior in credits | Fine as it stands | done |
| Fonts: Zen Maru Gothic, Noto Sans, Nunito | SIL OFL 1.1: licence text goes with the fonts, no selling the font alone | OFL texts in `public/fonts/`, credit line in Settings | Nothing for the licence. Stop loading Zen Maru Gothic from Google (see privacy) | P1 (privacy, not licence) |
| Sounds in `public/sounds/` | Pixabay licence: free, credit appreciated but not required, no selling or giving away the sound on its own ([summary](https://pixabay.com/service/license-summary/)). Your own recordings need nothing | Credits name the four Pixabay sounds. One credit links freesound.org though the file came from Pixabay's freesound_community account. `track.wav` still ships, unused | Point that link to Pixabay. Keep each sound's Pixabay page URL somewhere, since the licence is the one in force when you downloaded. Drop `track.wav` if it is not yours to ship | P3 |
| Icons and logo | Favicon is drawn by you; `package.json` has no icon library. Station and line symbols are drawn in code | Nothing to credit | Nothing | done |
| Code libraries | MapLibre GL and pmtiles (BSD-3), Preact and signals (MIT), earcut (ISC); the RTL text plugin from unpkg (BSD) | Not listed. The Rust crates inside the WASM were not audited | Optional "Open source libraries" line, generated from `cargo about` or `cargo license`. MIT and Apache licences ask for their notice when shipped in a binary | P3 |

## 2. Privacy

What the game stores, from the code and the server docs:

- No account: saves in the browser (localStorage and IndexedDB), a random id per browser that is sent only when
  signed in, settings, load timings. No cookies set by the game.
- Account: Google account number, or username plus password hash, optional recovery email (sent through Resend),
  cloud saves (up to two older copies), points per game, public display name if on a leaderboard. Daily server
  backups kept 7 days.
- Playtime heartbeat (T-1771): once a minute while a game is open, a random per-page-load id, start city, sandbox
  flag, build. Signed in, it also counts minutes per account. The server keeps daily totals, never the IP, and
  hashes the id for the open day only (`shonei-server/docs/trainworld-api.md`).
- Feedback form: optional name, optional email, message. Stored on the server and mailed to you; a signed-in
  account id and name ride along.
- Third parties that see a visitor: Cloudflare (data), OpenFreeMap (map), Google (fonts on every visit, sign-in
  script when the account panel opens, and Analytics through the website include), unpkg (the RTL text plugin).

What a US individual with a free site needs:

| Item | What it needs | Done today | To add | Priority |
|---|---|---|---|---|
| A privacy notice (US) | No federal law demands one for a site like this, but California's CalOPPA requires a posted policy for any site collecting personal information from California residents ([summary](https://www.iubenda.com/en/blog/do-not-track-california-privacy-policy-changes/)), and Google sign-in expects one | `public/privacy.html`, linked from the account panel. It is clear and short | Make it match the list above (heartbeat, feedback, GA or not, fonts). Change the date. Say how you handle Do Not Track (one sentence; with no cross-site tracking it can say the game does not track across sites) | P1 |
| Analytics | The page says no tracking cookies. GA4 sets cookies | Contradicts the published page | Pick one. A: remove the analytics include from the trainsworld page (the heartbeat replaces it); the privacy page then stays true and no consent banner is ever needed. B: keep GA, name it on the page, and show EU visitors a consent choice before it loads. I recommend A | P1 |
| Fonts from Google | See above | Not disclosed | Self-host Zen Maru Gothic as a local file, as Nunito already is. Same for the RTL plugin from unpkg if you want no third party at load | P1 |
| Feedback form | The page should say what is kept and for how long | Not mentioned | Add "Feedback: the name, email and message you type are stored on the server and emailed to Anita. Account name is attached if you are signed in." Say how long you keep them | P1 |
| Under-13 users (COPPA) | The FTC rule covers sites aimed at children, and general-audience sites only once they have **actual knowledge** a user is under 13 ([FTC FAQ](https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions)). Child-directed factors: subject matter, cartoon characters, music, audience. A rail economy game with money and city data reads as general audience; the pink theme and cartoon faces on trains are the only child-leaning factors. The rule was amended in 2025 (written security programme, retention policy), compliance date April 22, 2026 ([White & Case](https://www.whitecase.com/insight-alert/unpacking-ftcs-coppa-amendments-what-you-need-know)), but that only bites operators the rule covers. A February 2026 FTC policy statement on mixed audiences was mentioned in one source and not verified | Nothing said | Terms and privacy page: "Accounts are for people 13 and over. If you are under 13, play without an account." And "if we learn an account belongs to a child under 13 we delete it." Delete when told. Do not add features that ask for age or ask children for more | P2 |
| EU visitors and GDPR | GDPR reaches a non-EU site if it **targets** people in the EU or monitors their behaviour there. Being reachable, or using English, is not enough ([Recital 23, as quoted here](https://www.privacyworld.blog/2019/12/territorial-scope-of-the-gdpr-following-edpbs-final-guidelines-part-2/)). The game is English-only, has no EU domain, price or ads, but it does cover every European country as a map, which is a mild pointer. Separately, the EU's cookie rule (ePrivacy) depends on where the visitor's device is (general knowledge, not checked in search) | Nothing | Minimal sensible steps, which also cost nothing: keep data to what the game needs (already so); list in the privacy page what is stored, why (account: to run it; stats: keep the game working and see use), how long, who processes it (Resend, Cloudflare, Google, the host), the rights to see, fix and delete (delete exists in the panel), and a contact. State where the server is. With A above there is no cookie consent question. No EU representative or registration is realistic to need at this size | P2 |
| Accounts and passwords | Reasonable security | bcrypt hashes, HTTPS, token epochs, rate limits | Nothing | done |
| If someone is hurt by a breach | US states have breach laws for email plus password | Not planned | Keep the server patched; if the box is ever broken into, email affected users | P3 |

Not a lawyer, but I would not hire one for privacy at this size. A lawyer is worth a one-off hour if the game
passes a few thousand EU users, gets an actual complaint, or collects anything from children on purpose.

## 3. Names and trademarks

| Item | What I found | Do | Priority |
|---|---|---|---|
| Real railway and operator names, station and line names | Naming a real thing to say what it is (nominative use) is generally fine. Station names, line names and timetables are facts. The MTA terms add one real rule: do not imply endorsement. Colours are not protected as such; the colours come from the feeds' route colours, then OSM, then a palette | Add one sentence in the terms or credits: "trainsworld is not affiliated with or endorsed by any railway, transit agency or operator named in it." | P2 |
| Logos and bullets | The game draws its own circles and pills with a letter, in the line's colour (`ui/bullet.ts`, `GroupPill.tsx`); no operator logo, roundel, wordmark or typeface was found in the app code or `public/` | Keep it that way. Do not add the MTA bullet set, the TfL roundel, JR or Tokyo Metro symbols or wordmarks. That is where a takedown would come from | standing rule |
| Operator names inside the saves | Seven real-city saves carry line and station names tidied from the feeds ("lines lose their feed word"). Tokyo Metro, JR and so on appear only as names | Fine | none |
| The game's name | "Train World" (Turn Point Games, 2023, Steam and Epic, railroad sim) is the closest. There is also, from memory and not searched, a railway museum in Brussels called Train World. I could not query the USPTO or EUIPO databases from here (no trademark search through a fetch tool) | Search "train world" and "trains world" in the [USPTO trademark search](https://tmsearch.uspto.gov/) classes 9, 41 yourself; a free, no-ads game rarely draws a claim, but if the owners of "Train World" write, rename quickly rather than argue. Cheap insurance: a subtitle or a distinct logo, and not using "Train World" in the page text or tags. The file and code still say anitabuilder or trainworld; those are internal | P2 |
| Domain and site | `anita.garden/trainsworld/` is your own path | Nothing | none |

## 4. Other things a free game on a personal site commonly has

| Item | Verdict | Do | Priority |
|---|---|---|---|
| Terms of use | Exist (`public/terms.html`, 23 October). They are short and sensible: as is, no warranty, leaderboard rules, names may be removed | The sentence "Your games are kept only in your browser" is out of date since cloud saves (T-988). Add: 13 and over, no affiliation, how to contact you, that saves on the server may be lost. Optional: a governing-law line naming your US state | P2 |
| A contact | Privacy page links to anita.garden for contact | Make sure that page has a way to write to you, and say how fast you answer (a feedback form exists in the game too) | P2 |
| DMCA agent | The safe harbor from being sued for what users post needs a registered agent ($6 at [copyright.gov/dmca-directory](https://www.copyright.gov/dmca-directory), renewed every 3 years per the sources I read). The only public user content today is leaderboard display names; games and their pictures are private. Names are rarely infringing | Not needed now. Do it the day you ship public sharing of games by link (T-989), pictures, or comments | P3 (P2 with T-989) |
| Leaderboard names | You can remove names and entries (`hidden.json`); the terms say so. US law (section 230) protects hosting what users post. EU's Digital Services Act asks hosts for a contact point and a way to report content, but it is in scope only if you are offering the service in the EU | A "report a name" line: the contact is enough | P3 |
| Age rating, app stores | Not applicable on the web | none | none |
| Taxes, business registration | Free, no ads, no income: none | If you add donations: check your state | none now |
| Accessibility law (ADA and similar) | Suits mostly target commercial sites; a free hobby game is a very low risk | None required; colour contrast and keyboard use are a courtesy | none |
| Open-source licence of your own code | The repo and data licence are your call. Data files: ODbL (stated) | Decide if the code is public. Nothing is owed | none |
| The AWS account | A friend's account, used for builds only; no data of visitors goes there | none | none |

## If you ever earn money

The free, no-ads state is what keeps several things simple. Ads, payments, a paid tier, sponsorship or
donations would need these first:

- GISCO NUTS (non-commercial; EuroGeographics licence needed) and any other non-commercial source (SHRUG is
  checks-only already; Kyrgyz statistics are CC BY-NC-SA in helper1m, if used).
- Rereading the licences marked "not read" above (NJ Transit, Metrolinx, MTA, IBGE, India).
- A real privacy policy for ad networks, consent in the EU, and likely a lawyer.

## What to do before release

Must (small edits, an hour or two altogether):

1. Decide Analytics: remove the GA include from the trainsworld page (my pick), or disclose it and add EU consent.
2. Host Zen Maru Gothic yourself instead of from Google Fonts (and optionally the unpkg RTL plugin).
3. Update the privacy page: heartbeat, feedback form, third parties as they really are, retention, where the
   server is, Do Not Track sentence, accounts are for 13 and over, a new date.
4. Update the terms: fix the "only in your browser" sentence, 13 and over, no affiliation, contact.
5. Add credits: "Powered by TfL Open Data"; MTA (with the no-endorsement sentence); IDFM (ODbL); Network Rail
   (OGL); ASPECT (CC BY 4.0); NJ Transit and PATH once you have read their terms.
6. Search "train world" and "trains world" in the USPTO trademark search and decide if the name stays.

Later: the pack rebuild without the NUTS fills and the `nocar` layer (T-925, T-926); a DMCA agent when public
sharing ships; an open-source libraries line; keeping each sound's Pixabay page URL.

Needs a lawyer (otherwise not): a claim or letter about the name or a logo; any plan to earn money; a real EU
audience; a complaint from a parent; anything the NJ Transit or Metrolinx agreements forbid.
