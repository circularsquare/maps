# Australia: the record

Drawn 2026-10-04 (agent d9e44929-au). 23,881,399 people on 2,411 SA2s (placed on their 61,811
SA1 polygons), 381 languages, 23,762 dots and 71 single-dot languages. 716,865 people (3.0%) are
`derived`.

## Source

Australian Bureau of Statistics, Census of Population and Housing 2021, CC BY 4.0. The question is
**language used at home** (LANP): "Does the person use a language other than English at home?",
one answer, the language used most often. Not mother tongue. Asked of everyone, place of usual
residence, overseas visitors excluded. Coded to ASCL 2016 (about 430 four-digit languages).

No open table has the full classification at small-area level (TableBuilder does, behind a
registered login; not used). Three tables of the one census are combined
(`sources/au_census.py` normalises and checks, `countries/au.py` combines):

| level | table | what it has |
|---|---|---|
| `sa2` | General Community Profile **G13**, 2,472 SA2s | English only, 35 named languages, five remainders: other Chinese, other Indo-Aryan, other Southeast Asian Austronesian, Australian Indigenous languages (all as one), Other |
| `sa2_top` | **QuickStats**, one page per SA2 | "Language used at home, top responses (other than English)": the SA2's five largest languages at four-digit detail |
| `state` | **Cultural diversity data summary 2021, Table 5** | every four-digit language by state and territory |

- G13 is read from religiondots' SA2 DataPack (`religiondots/data/raw/au/2021_GCP_SA2_for_AUS_short-header.zip`,
  release R2), read in place; `--fetch` downloads it into `data/raw/au/` if that copy is gone.
- Table 5: `data/raw/au/Cultural diversity data summary.xlsx` (released 28 June 2022).
- The classification: `data/raw/au/Language classification.xlsx` (Census dictionary 2021, LANP).
- QuickStats: 2,454 pages (every SA2 but the 18 pseudo SA2s) scraped into
  `data/raw/au/quickstats_sa2.jsonl`, generic browser User-Agent, 4 threads with a pause, allowed
  by robots.txt. The first run hung on one stuck connection; the scraper now gives up after three
  minutes without a page and resumes on the next `--fetch` (it took three runs).
- SA1 DataPack (365 MB) not taken: SA1 G13 has the same 35 languages, so it would only sharpen
  placement inside an SA2, which equal shares over ~400-person SA1s already do.

## Checks (numbers from the run)

1. **G13 leaves sum to each SA2's total**: worst 64 people; nationally 25,414,753 against
   25,422,677 (-0.03%). **G13's "Other" includes the Australian Indigenous languages**, which the
   table also shows as their own cell (Yarrabah: Other 1,958, Indigenous 1,961, all languages but
   English 1,966). The metadata does not say so; this check failed by up to 6,079 people per SA2
   until "Other" was taken as Other minus Indigenous.
2. **The remainders hold what they should**: per state, G13's SA2 sum for each named language and
   each remainder against Table 5's sum of that remainder's four-digit members, 179 cells over
   5,000. The first bar (1%) failed on 8 cells, plain named languages among them (Polish in
   Queensland -2.4%, Serbian, Sinhalese, Croatian), so it is the two tables' separate
   perturbation; bar now 2.5% or 500 people. Worst: NT "Other" -8.5% (G13 4,935, Table 5 5,391),
   because Other minus Indigenous is a difference of two cells perturbed by up to ~90 each in the
   big remote SA2s (six NT SA2s come out negative and are clipped to 0).
3. **A second table per unit**: all 10,518 QuickStats rows that name a G13 language equal G13's
   cell exactly (0 people difference): QuickStats is drawn from the same perturbed tables, so its
   four-digit rows can be trusted at SA2 level.
4. QuickStats covers all 2,454 real SA2s. 24 pages have no language table; 23 are SA2s of under
   130 people (non-English totals 3-25), and one is **Acton** (ANU campus, 2,848 people, 569
   non-English), whose page omits the table. Their remainders are shared out like any other.
5. Scatter: every counted SA2 has SA1 polygons (religiondots' ASGS 2021 SA1 file, `SA2_CODE21`);
   34 empty SA1s dropped (all children of the pseudo SA2s); 43 SA2s have polygons but no people.

## How the remainders are shared out (`countries/au.py`)

The five G13 remainders hold 969,000 people (3.8%).

- **Measured, 210,184 people.** Inside each SA2, QuickStats rows whose language falls in a
  remainder are taken as they stand (17 SA2 cells where they sum above G13's remainder are
  trimmed to it). This places 54,814 of the 77,874 Indigenous-language speakers drawn (70%): the
  core communities are measured (Tiwi 1,804 of 2,086 on the Tiwi Islands SA2; Murrinh Patha
  2,007 of 2,079 in Wadeye and its neighbour; Assyrian's five largest SA2s all in Fairfield).
- **Derived, 759,145 people (716,865 drawn).** What is left of each SA2's remainder is shared
  over that state's languages in the remainder, each reduced by what QuickStats already placed,
  by raking (iterative proportional fitting) to both margins: each SA2's leftover and each
  language's state leftover. The seed is 90% a 30 km distance-decay pull towards SA2s where
  QuickStats names the language and 10% even, so Tiwi speakers not in the Tiwi Islands' top five
  land around Darwin rather than in Alice Springs. Caps from the QuickStats list: a language
  missing from an SA2's top five is at most the fifth entry, and a list shorter than five is
  complete (nothing unlisted is there). 136 people found no room under the caps and sit on the
  remainder's group node. Other Territories (Christmas, Cocos, Jervis Bay, Norfolk) have no
  Table 5 column; Australia minus the eight states stands in.
- "Inadequately described" (18,150) and "Non-verbal, so described" (24,625) are members of
  "Other" and are shared out with it, then not drawn: 42,416 people.

## Calls someone might reverse

- **Sharing out the remainders at all.** The alternative is drawing 969,000 people as five
  "language not named" blobs, 79,000 of them as one "Australian Indigenous" colour. Drawing the
  measured QuickStats part and leaving the rest unnamed is a one-line change in
  `countries/au.py` (drop the derived rows); the viewer's inferred-dots switch already shows it.
- **The 30 km kernel and 90/10 seed** (`KERNEL_KM`, `LAMBDA`): picked, not fitted. They move
  derived dots within a state only; every margin is the census's.
- **The `australian` root** (tree.d/au.txt): an areal root, "Australian Indigenous languages", as
  the ABS draws it, holding Pama-Nyungan, ~20 small northern families, Meriam Mir (Papuan in
  Glottolog) and the Indigenous creoles. A root per family would put two dozen roots in the legend
  for a few thousand people each, and the census's unnamed Indigenous remainders mix all of them.
  The Daly families are one group ("Daly"); Tiwi (an isolate) stays under the root rather than
  `isolate`.
- **Kriol, Yumplatok, Gurindji Kriol and Aboriginal English under `australian.contact`**, not
  under `creole` (where Krio, Haitian and the rest sit): the ABS counts them as Indigenous and its
  nfd Indigenous remainder can hold their speakers.
- "X, nfd" on the group: **Arrernte, nfd (1,437)** on Arrernte, Karen (13,181) on the Karen
  group, as ca2021 does for Cree and Karen. "Cape York Peninsula Languages, nec" (2,701) on
  Pama-Nyungan. "Burmese and Related Languages, nec/nfd" on Sino-Tibetan.
- **Ndebele on Zimbabwean Ndebele**; **Nyanja (Chichewa) on zm.txt's Nyanja**; Bemba on zm.txt's
  Bemba leaf (both repeated identically in au.txt).
- **Chaldean Neo-Aramaic recoloured** (us.txt's node, was generated identical to Arabic,
  `#96d5b2`) to a light amber in au.txt: Chaldeans live among Arabic speakers in Fairfield and in
  Detroit, so the US map gains too.

## Geography

Religiondots' `data/geo/au/SA1_2021_AUST_GDA2020/SA1_2021_AUST_GDA2020.shp` (ASGS Edition 3),
read-only; `unit` = `SA2_CODE21`, no weighter: an SA2's dots are shared equally over its SA1s,
which ABS builds to about 400 people (religiondots `sources/au_geo.md` §4 measured median 406,
IQR 359-447). Its sea clip was reused. Remote SA1s are large, so a dot there lands anywhere in the
SA1 polygon, not on the community; no Kontur layer for Australia is on disk.

## Gap

- Not stated: 1,438,071 people (5.7%) in real SA2s.
- 42,416 answers naming no language (above).
- 52,920 people in the 18 pseudo SA2s (offshore, shipping, no usual address), which have no
  polygon.

## Colours

The Indigenous families and big languages are hand-picked in `taxonomy/tree.d/au.txt`
(Western Desert reds and oranges, Arandic yellows, Ngumpin-Yapa crimsons, Yolngu greens,
Gunwinyguan violets, Maningrida blue, Daly magenta, Tiwi teal, Kriol pale yellow-green). Checked
pairwise in OKLab against co-location by SA4: no Indigenous pair under 0.06 shares an area.
Also hand-picked, because generated colours sat within 0.02 of a neighbour in the same suburbs:
Hazaragi (was beside Pashto's khaki in Dandenong), Fiji Hindi (beside Turkish), Māori and Cook
Islands Māori (beside Cantonese and Afrikaans), Hakha Chin (beside Samoan). Many other close pairs
in the cities are between other countries' nodes (Telugu/Oromo, Somali/Ukrainian, Hungarian/
Armenian); not touched.
