# Education locations and enrollment (T-150, 2026-10-11)

The useful first expansion is **US public schools across the NYC metro**, using
the national NCES source outside the five boroughs and keeping the better DOE
building inputs inside them. National source availability is not global coverage,
and only the NYC pack is enriched/enabled in this pass.

## Implemented source

[NCES EDGE public-school administrative data 2023-24](https://nces.ed.gov/opengis/rest/services/K12_School_Locations/EDGE_ADMINDATA_PUBLICSCH_2324/MapServer/1)
combines school locations and CCD administrative attributes. The pipeline reads
NCESSCH, SCH_NAME, SURVYEAR, STATUS, VIRTUAL, MEMBER, LATCOD and LONCOD. MEMBER excludes
adult education (TOTAL includes it). Positive enrollment is required; operational
statuses 1/3/4/5/8 and Not Virtual/Supplemental Virtual are retained. Full Virtual,
unknown virtual status, closed/future sites and negative missing-value codes do not
create trips. Supplemental virtual schools remain a rough attendance approximation.

NCESSCH is the stable deduplication key. Distinct schools sharing coordinates have
distinct pupil counts, unlike DOE's repeated whole-building counts. Repeated NCES
IDs fail the build. The spatially excluded DOE area prevents adding the national
records on top of DOE building enrollment. This deliberately leaves unmatched DOE
buildings unresolved rather than guessing which NCES school duplicates them.

County membership uses the actual city/borough polygons. The NCES administrative
layer's CNTY field was observed as a truncated code, so it is not used. The game's
2020 Connecticut boundary remains authoritative. Absent H3 cells can snap at most
500 m to existing cells; omissions are audited and at least 95% of positive eligible
NCES enrollment must be located. No cells, commutes, jobs or other demand inputs are
fabricated or reordered.

This is published federal aggregate administrative data, treated as a US government
public-use work, with NCES attribution retained. The service has an empty
copyrightText field, not a specific CC license. Relevant federal publisher guidance:
[IES public-access policy](https://nces.ed.gov/about/public-access-research) and
[Department of Education copyright notice](https://www.ed.gov/about/ed-overview/required-notices/website-policies/copyright-status-notice).
The ED notice explicitly limits its statement to www.ed.gov; do not mislabel this
NCES dataset as CC0 based solely on that notice. No restricted-use student records
are used. Existing DOE and Census provenance is retained in the pack.

## University sources for a separate attendance layer

| Source | Geography / useful fields | Reuse and limitations |
|---|---|---|
| [IPEDS downloadable survey files](https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx?year=-1) | US institutions: HD directory coordinates and UNITID, Fall Enrollment totals and distance-education status | Official federal public-use data; strongest next step for current US cities. Join a consistent survey year by UNITID. Institution totals may span multiple campuses; do not place a system total at every campus. |
| [Wikidata](https://www.wikidata.org/wiki/Property:P2196) | Semi-global: P625 coordinates and P2196 student counts, sometimes dated with P585 and qualified by P518 | [Structured data is CC0](https://www.wikidata.org/wiki/Wikidata:Copyright). Broad but incomplete and uneven; counts may mean undergraduate only, a whole system, or a historical year. Preserve qualifiers/references, select one comparable total, never sum overlapping statements. Best rough global fallback after country sources, with missing enrollment explicit. |
| [ROR](https://ror.readme.io/docs/ror-data-structure) | Global research-organization IDs, education type, Wikidata crosswalks, geographic locations | [CC0 data dump](https://ror.readme.io/docs/data-dump). No enrollment field. Location details come from GeoNames and may identify a city rather than a campus entrance. Useful identity/join aid, not a standalone attendance source. |
| [ETER / EHESO](https://eter-project.com/data/overview-data/data-collection-process/) | European institutional student counts and institutional information from national statistical authorities | Promising regional supplement; inspect each export's access and reuse terms before shipping. Not researched sufficiently here to claim unrestricted reuse or complete campus geocodes. |

For IPEDS, use Fall Enrollment rather than annual unduplicated enrollment for a
typical-day budget. [The official enrollment survey](https://nces.ed.gov/ipeds/use-the-data/download-survey-material/2024/fall%20enrollment/package_6_51.pdf)
separates exclusively distance-education students, which should not produce campus
attendance trips. Students taking some online courses need an explicit attendance
assumption. College homes, attendance factors, spatial decay, and timing must be
separate from ages 3-17 and the K12 morning/dismissal schedule. Staff remain in Work.

## General schools beyond the US

[England's Get Information about Schools](https://www.get-information-schools.service.gov.uk/AcceptableUsePolicy)
supports bulk establishment downloads and describes OGL reuse where stated. It is a
useful next national source; verify enrollment dates and location columns in the
actual export before integration. There is no verified, uniformly complete global
school-enrollment source in this work. Wikidata/OSM can help locate institutions but
missing enrollment is not evidence of zero pupils, and OSM reuse has its own license.
Country adapters plus explicit coverage metadata are preferable to silently applying
one guessed school size everywhere.
