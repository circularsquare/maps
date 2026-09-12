# Traincounts country difficulty estimates

Research checkpoint: September 10, 2026. These are engineering estimates for this project's **scheduled trains per physical corridor** metric, not rankings of railway quality. They are provisional until an actual feed is downloaded and its geometry is inspected. A timetable being available does not establish the path taken by express trains or national completeness.

## Rough tiers

| Tier | Country / starting scope | What looks promising | Main remaining difficulty | Evidence level |
|---|---|---|---|---|
| A — easiest pilots | **Finland**, Fintraffic passenger rail | One official passenger feed with intermediate pass-by points and shapes | 16 geometry fallbacks; differently segmented shared corridors still need audit | Downloaded, implemented and reconciled |
| A — easiest pilots | **Ireland**, Iarnród Éireann including published Enterprise services | One official rail feed, detailed shapes, manageable network | Project express paths through skipped stations; review malformed shapes and cross-border scope | Downloaded; second adapter under validation |
| B — good next candidates | **Norway**, mainline passenger rail first | Entur centralizes national public-transport data | Inspect current static export, rail-only coverage and shape quality before committing | Official portal verified; ZIP not inspected |
| B — good next candidates | **Austria**, ÖBB feed first | Official ÖBB GTFS plus official railway geometry datasets | ÖBB feed is explicitly a subset; join schedules to corridors and add other operators later | Official source/scope verified; ZIP not inspected |
| B — good next candidates | **Netherlands**, rail first | Public NDOV NS schedule archive; wider public-transport data services | Inspect the IFF/archive format and pass-through detail, geometry, and coverage beyond NS | Current archive listing verified; contents not inspected |
| C — substantial integration | **Switzerland**, national rail | Official national public-transport GTFS | Publisher explicitly omits shapes; need an external rail graph and route matching; mixed modes and frequency-based services need care | Official GTFS technical documentation inspected |
| C — substantial integration | **France**, SNCF mainline first | SNCF publishes TER, Intercités and TGV GTFS | National completion needs regional/IDFM and other-operator coverage policies; geometry and overlaps need audit | Official timetable source/scope verified; ZIP not inspected |
| C — substantial integration | **Germany**, DELFI rail subset | Nationwide aggregate includes local public transport and long-distance rail | Registration; large mixed-mode dataset, geometry and feed completeness audit | Official DELFI documentation inspected; ZIP not inspected |
| C — substantial integration | **Great Britain**, Network Rail passenger services | Central operational timetable source rather than many isolated operator calendars | Account access, CIF/operational schedule semantics, passenger filtering, overrides and route mapping; urban systems outside Network Rail remain separate | Official Network Rail access/format documentation inspected |
| D — multi-system programme | **USA**, national coverage | Individual agencies have useful public GTFS; MTA has separate LIRR and Metro-North downloads | Country-wide work needs an operator inventory and shared-corridor conflation; Amtrak feed access still needs verification | MTA official feeds verified; nationwide discovery incomplete |
| D — multi-system programme | **Japan**, national coverage | ODPT is an official collaboration/catalog route for participating operators | Operator-by-operator access, usage terms and timetable coverage audit; through-running and geometry reconciliation | ODPT access/catalog documentation verified; nationwide discovery incomplete |

Tier A means a bounded source-specific adapter plus validation. Tier B adds at least one meaningful source/geometry/coverage problem. Tier C likely needs several implementation and review passes. Tier D describes a national programme, not a single adapter. These are relative effort bands, not wall-clock promises. Access approval can dominate elapsed time independently of coding difficulty.

An **individual US operator**, such as LIRR, can be a B-tier task even though complete US coverage is D-tier. Similarly, an ODPT-supported Japanese system can be much more approachable than nationwide Japan. Starting with Amtrak alone would be an operator pilot, not “USA complete.”

## Suggested order

1. Finish Ireland and retain its adapter/geometry lessons alongside Finland.
2. Inspect Norway's actual current static export; proceed if shapes and rail coverage check out.
3. Try an Austria ÖBB pilot, then Netherlands rail. Reverse this order if the actual Dutch archive offers better pass-through geometry.
4. Build a reusable railway-graph matching approach for Switzerland, then use those lessons for France and Germany.
5. Treat Great Britain as its own timetable-semantics task once source access is available.
6. Start a parallel **roadmap**, not concurrent shared-file edits, for US operators if US coverage is a priority. MTA is a concrete accessible starting point; continue Amtrak source discovery separately.

Denmark, Sweden, Belgium, Luxembourg, Spain, Italy and countries outside Europe/North America/Japan have not been evaluated in this pass. Their omission is not a difficulty judgment. Re-rank after inspecting actual feeds; do not extrapolate from country size or operator count alone.

## Source notes

- **Finland:** [Fintraffic railway documentation](https://www.digitraffic.fi/en/railway-traffic/) identifies passenger GTFS variants with and without pass-by points, daily publication and CC BY 4.0.
- **Ireland:** [NTA GTFS catalog](https://data.gov.ie/dataset/nta-gtfs) and [official Irish Rail ZIP](https://www.transportforireland.ie/transitData/Data/GTFS_Irish_Rail.zip). The inspected feed contains 19 rail routes and 152 stations; read `IRELAND.md` for measured routing diagnostics and the precise scope.
- **Norway:** [Entur open-data overview](https://developer.entur.no/open-data) states that Entur maintains the national registry and makes the data openly available. This pass did not verify an actual static export URL or its geometry.
- **Austria:** [ÖBB datasets](https://data.oebb.at/de/datensaetze) describes the GTFS scope as ÖBB-Personenverkehr plus CAT and Montafonerbahn services within Austria, and separately lists Geo Netz data. This is not evidence of all-operator coverage.
- **Netherlands:** [NDOV NS archive](https://data.ndovloket.nl/ns/) exposes `ns-latest.zip`; the inspected listing was updated September 8, 2026. [REISinformatiegroep's catalog](https://reisinformatiegroep.nl/ndovloket/datacollecties) lists GTFS and other products with an application process. Verify the terms of the specific source chosen, rather than treating these two access routes as identical.
- **Switzerland:** [Official GTFS cookbook](https://opentransportdata.swiss/en/cookbook/timetable-cookbook/gtfs/) states national coverage, explicitly explains the absence of shapes, and documents `frequencies.txt`. Geometry is the main reason for its C-tier estimate.
- **France:** [SNCF timetable dataset](https://data.sncf.com/explore/dataset/horaires-sncf/) publishes a GTFS download and describes TER/Intercités/TGV scope; it recommends regional and IDFM data for the corresponding authoritative services.
- **Germany:** [DELFI official catalog](https://www.opendata-oepnv.de/ht/de/organisation/delfi/startseite) describes the national GTFS/NeTEx aggregate, weekly publication, registration and dataset-specific completeness notes.
- **Great Britain:** [Network Rail open feeds](https://www.networkrail.co.uk/who-we-are/transparency-and-ethics/transparency/open-data-feeds/) explains registration and access conditions; its [data architecture reference](https://www.networkrail.co.uk/wp-content/uploads/2019/12/Data-Architecture-Reference-Model.pdf) describes CIF and working/applicable timetable distinctions. Northern Ireland is a separate scope, not silently included in this row.
- **USA:** [MTA developer resources](https://www.mta.info/developers) provides distinct LIRR/Metro-North feeds and notes that some Metro-North-branded lines are in NJ TRANSIT data. This illustrates the source-boundary problem without establishing a complete US operator inventory.
- **Japan:** [ODPT](https://www.odpt.org/en/) documents developer registration, usage conditions and the participating-provider catalog. Existing Japanrail passenger-flow data is not a replacement for train schedules.

Keep the table current in project handoffs. Promote a candidate from “documentation verified” to “download inspected” only after recording the feed checksum, publication/window, rail filters and geometry findings.
