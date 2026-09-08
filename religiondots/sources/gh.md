# Ghana — GSS, 2021 Population and Housing Census

Wired 2026-09-05. 30,753,327 people, 272 units, 7 drawn categories. **The first African
country on the map.**

| | |
|---|---|
| source | Ghana Statistical Service, PHC 2021, StatsBank PxWeb table **`religion_table.px`** |
| basis | `self_id` |
| geography | **272 units** — 255 districts + 17 sub-metros, replacing 6 metropolitan parents |
| categories | 9, of which 7 are drawn (one is the universe, one is a duplicated parent) |
| drawn | **30,753,327 people, 99.74% of the census count** |
| licence | GSS open data; StatsBank is a free public dissemination platform, attribution expected |

Cheap, like North Macedonia, and for the same reason: a PxWeb API and a boundary file the
statistical office ships itself. The two things that cost real time were both *finding*
rather than *parsing* — the API is not where PxWeb's own UI says it is (§1), and the
geography dimension nests three levels deep with an acronym collision inside it (§3).

---

## 1. The API is not under the UI prefix, and the wrong path returns 500 rather than 404

StatsBank's web interface is at `statsbank.statsghana.gov.gh/pxweb/en/...`, so the obvious
API root is `/pxweb/api/v1/en/`. That returns **HTTP 500 with a PX-Web ASP.NET error page**
("An error has occurred. The error has been registered"). A 404 would have read as "wrong
path, try another"; a 500 on a PxWeb-branded error page reads as "the API exists and is
broken", and this session nearly recorded Ghana as *PxWeb with the API disabled* on the
strength of it.

The API is at **`/api/v1/en/`**, with no `/pxweb` in front:

```
https://statsbank.statsghana.gov.gh/api/v1/en/                              -> the 10 databases
https://statsbank.statsghana.gov.gh/api/v1/en/PHC%202021%20StatsBank/       -> 11 topics
https://statsbank.statsghana.gov.gh/api/v1/en/PHC%202021%20StatsBank/Population/religion_table.px
```

**Generalises:** try the API prefix with AND without the UI prefix. And a 500 from an
application's own error template is evidence about the ROUTE, not about the feature — it
means the request reached the app and the app did not recognise it, which is the same
information a 404 carries and is much easier to misread.

The HTML tree is a dead end by comparison: it navigates by ASP.NET `__doPostBack` and every
folder page renders the same sibling list, so a link-following walk finds nothing and looks
like an empty database. The node ids are visible in the postback arguments
(`pPHC 2021 StatsBank__Population`), which is what pointed at the right shape.

## 2. What the question asks, and what that fixes forever

Nine categories:

```
Total  (universe)
  Christian  (parent — see §5)
    Protestant (Anglican, Lutheran, Presbyterian,  Methodist, etc.)   5,364,320   17.4%
    Catholic                                                          3,071,844    10.0%
    Pentecostal/ Charismatic                                          9,703,351    31.6%
    Other Christian                                                   3,793,193    12.3%
  Islam                                                               6,108,530    19.9%
  Traditionalist                                                        999,319     3.3%
  No Religion                                                         1,384,049     4.5%
  Other Religion                                                        328,721     1.1%
```

**Four Christian boxes and one box for everyone else** — the Philippines' lopsidedness
(§9m) in a nine-item list rather than a 129-item one. The consequences are fixed and no
amount of work moves them:

- **No Sunni/Ahmadi split**, in a country with one of the oldest and largest Ahmadiyya
  communities in West Africa, with its own hospitals and schools.
- **No African Independent Church is named.** The Musama Disco Christo Church, the Twelve
  Apostles Church, the African Faith Tabernacle and the Aladura-type bodies are a distinct
  stream with a distinct history, and they are the bulk of `Other Christian` — 3.79M people
  in one cell.
- **`Pentecostal/ Charismatic` is one box for two things**: the classical Pentecostal
  denominations (the Church of Pentecost is the largest Protestant body in Ghana, plus
  Assemblies of God and Christ Apostolic) and the neo-charismatic ministries founded from
  the 1980s (Lighthouse Chapel, Action Chapel, Perez Chapel). It goes to
  `christianity.pentecostal`, the parent, for that reason.
- **`Traditionalist` is one box for every tradition** between the Akan and the Dagomba.

The label `Protestant (Anglican, Lutheran, Presbyterian,  Methodist, etc.)` names four
bodies **as examples of the box, not as a decomposition of it** — the census does not ask
which. Mapping to `christianity.methodist` on the strength of a label would invent a split
the source does not make. (The double space after `Presbyterian,` and the space in
`Pentecostal/ Charismatic` are GSS's; both are kept verbatim per spec §2.4, and
`taxonomy/gh2021.py` keys on the exact strings.)

## 3. The geography nests three levels, and one acronym means two different cities

`Geographic_Area` has 295 values in one flat list with no level column and no codes at all:

| | |
|---|---|
| 1 | Ghana |
| 16 | regions |
| 255 | plain districts |
| 6 | metropolitan districts (parents) |
| 17 | sub-metros of those six |

Summing the file as delivered counts the country roughly three times. The structure is
recovered **positionally** — the cube lists each region header immediately followed by its
own districts — and then *asserted*: every region's children must sum to it and the regions
to Ghana, or `sources/gh.py` stops. (Romania's county-header misparse, §12, is what that
assert exists for.)

**`TMA` IS TEMA IN GREATER ACCRA AND TAMALE IN NORTHERN.** Both are "Tema/Tamale
Metropolitan Area (TMA)", both have sub-metros prefixed `TMA-`, and they are 600 km apart:

```
Greater Accra   Tema Metropolitan Area (TMA)      176,723   TMA-Tema Central, TMA-Tema East
Northern        Tamale Metropolitan Area (TMA)    373,469   TMA-Tamale South, TMA-Tamale Central
```

Resolving `TMA-` against a single global acronym hands all four to whichever parent is seen
last and orphans the other — and **every national and regional total still reconciles**,
because the four rows are all still there and all still in the same country. The only thing
that changes is which metropolis two of them belong to. The prefix is unique only *within a
region block*, which is where `gh.py` resolves it.

That is Sri Lanka's lesson (§12: a shared code is only trustworthy as far up the hierarchy
as you have independently verified it) arriving as an *acronym* rather than a numeric code,
and it is worth noticing that the acronym looked far more human-readable and was no safer.

The related near-miss: `Nkwanta North (Kpassa)` also ends in a parenthesised word and is a
plain district with an alias, not a metro. The rule is "a parenthesised acronym that some
row in the same region uses as a `ACR-` prefix", and the second half of that is what keeps
Kpassa out.

## 4. The universe is 30,753,327 and the missing 78,692 are missingness, not a group

The 2021 PHC counted **30,832,019** people (`population_table.px`, and GSS's headline
figure). The religion table's `Total` is **30,753,327** — 78,692 fewer, 0.26%.

There is no "not stated" category: GSS drops non-response rather than publishing a cell for
it. That it is genuinely non-response and not a defined sub-universe was checked two ways —
the same 0.26% shortfall appears in **every one of the 17 education bands** and across the
age distribution, against `population_table.px` read the same way. A restricted universe
(institutional population, a minimum age) would concentrate somewhere; this does not.

**It is not scaled up**, and the precedent is Chile's 15+ question (§9k): shares are shares
of the people who answered, and the map draws 99.74% of the country. Inventing a religion
for 78,692 people to make a total round is exactly what spec §14.4's first rule forbids.

## 5. `Christian` is a duplicate, and this is Hungary's trap with the easy ending

GSS publishes `Christian` (21,932,708) **beside** its own four children, and the four sum to
it **exactly, on all 295 rows of the cube**. So the parent is dropped: drawing it too would
count 71% of Ghana twice.

Worth putting next to Hungary (§9h), which is the same shape and does not end the same way.
KSH gives `Katolikus` (2,886,619) beside Roman and Greek Catholic and never publishes their
77,629-person difference, so there the remainder had to be *emitted* — and then emitted at
every level the allocation touches, or it vanished silently. **The distinguishing question
is arithmetic, not structure: do the published children sum to the published parent?** Here
they do, everywhere, so there is nothing to emit and the parent is simply a duplicate.
`sources/gh.py` checks the identity on every row rather than nationally, because a
parent/child relation that holds at the top and fails in one district is what a misparse
looks like.

## 6. What the map shows

Ghana is a good map, which is not guaranteed by a good source. Four things are visible:

- **The north–south divide is a wall, not a gradient.** Islam is 66.5% of the Northern
  region and 4.7% of Volta. Nanton is 98.7% Muslim, Kumbungu 96.7%, Tolon 96.0%, Savelugu
  95.7% — as near-total as anything on this map outside Sulu.
- **A Catholic island inside the Muslim north.** Upper West is 33.8% Catholic against a
  national 10.0%, and **Nandom is 88.6%**, the highest single-body share of any Ghanaian
  district. Its neighbours Jirapa (64.0%), Nadowli Kaleo (59.2%) and Lawra (48.7%) run with
  it. That is one mission field — the White Fathers from 1906, at Navrongo and Jirapa —
  still legible as a hard edge more than a century later, and it is the single most
  striking thing in the Ghanaian data.
- **Pentecostal/Charismatic Christianity is the largest answer in the country**, 31.6%, and
  it is a 20th-century arrival rather than a mission inheritance: it outnumbers Catholics
  three to one. Greater Accra is 47.3%, and the Ada districts touch 60%.
- **Traditional religion survives in a belt.** 3.25% nationally and a median district of
  0.5%, but 43.6% in Tatale Sanguli, 41.0% in Nabdam, 40.1% in Nanumba South — Northern and
  Upper East, with a second pocket in Oti and Volta (12.7% and 9.7%). It is near zero across
  the whole Akan south.

**Read `Traditionalist` as a floor.** The form makes it exclusive of the Christian and
Muslim boxes, and in Ghana traditional practice very often accompanies one of those rather
than replacing it, so everyone who would answer both is counted in the other column. That
caveat applies to every African census that asks this question the same way, which is all
of them.

## 7. Ethics (§14)

Nothing here is close to §14's line, and it is worth writing down why so the next African
country does not have to re-derive it. Ghana's state publishes religion itself, at district
level, on a free public platform; the map draws **exactly the tier GSS publishes** and
nothing finer, so §14.4's "for a persecuted group, no resolution finer than the state's own
publication" is satisfied trivially. The Muslim–Christian balance is politically salient in
Ghana — it is why Nigeria stopped asking (§3) — but Ghana has not stopped asking, and
reproducing an official published table at its published resolution is not the same act as
estimating one that was withheld.

## 8. Not done

- **Sub-metros are drawn; nothing finer exists.** GSS publishes religion at district and
  sub-metro and stops. The census has enumeration areas underneath, and no religion table
  reaches them.
- **The 2010 PHC is also on StatsBank** (`PHC2010` database, Population folder) and is not
  ingested. A 2010/2021 comparison would show the Pentecostal share moving, which is the
  most interesting time series available anywhere in the file — but spec §13 says no time
  slider, and a two-date country would need a place to put the second date.
- **Ethnicity is in the same database** (`ethnic_table.px`, same 295 rows). It is not read,
  and per §14.4 it must not be used to refine religion within a district.
