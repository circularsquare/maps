# Portugal — INE, Censos 2021

Ingested 2026-09-06. `sources/pt.py`, `sources/pt_geo.py`, `taxonomy/pt2021.py`.
Drawn at **freguesia**: 3,092 units, 11 categories, 8,781,900 people.

Summary: **the cheapest and cleanest country in the project.** One unauthenticated GET for the
counts, no download at all for the boundaries, a perfect partition at every level, zero
suppression, and a code join that needed no derivation. Everything hard about the last ten
countries was absent here. The two things worth reading are §1 — how it was nearly missed —
and §4, the universe caveat that the table does not state.

---

## 1. The find was the catalogue, and one digit hid it

An earlier probe of INE concluded Portugal published no subnational religion. It was wrong, and
the reason generalises.

INE has an indicator catalogue at `www.ine.pt/ine/xml_indic.jsp`, and it takes an `opc` mode:

| `opc` | bytes | indicators | hits for `religi` |
|---|---|---|---|
| 0, 1, 4, 5, 6, 7 | ~80 KB | 0 | 0 — the same nav page |
| **3** | 517 KB | **326** | **0** |
| **2** | **21 MB** | **13,098** | **6** |

`opc=3` is the one that looks like the catalogue. It is the *recently updated* indicators, and
a 2021 census table last touched in 2022 is not in it. **When a catalogue endpoint takes a mode
parameter, fetch the widest mode before concluding the data is absent.** The site's own search
UI does not rescue this either: `xpgid=ine_pesquisa` returns ten pages of results whose bodies
contain no usable indicator codes.

`opc=2` also carries **`geo_lastlevel` per indicator**, which is the part worth keeping. It turns
"which Portuguese census tables reach the freguesia" into a string search over a local file
rather than a crawl, and it is how the religion indicator was found in one pass.

The four census religion indicators:

| varcd | census | geography | note |
|---|---|---|---|
| **0012311** | 2021 | **freguesia** | NUTS-2024 codes. **Drawn.** |
| 0011644 | 2021 | freguesia | the NUTS-2013 twin |
| 0006396 | 2011 | freguesia | the previous census, same tier |
| 0014361 | historical | distrito/ilha | 1900–2021 series, coarse |

`0012311` is drawn rather than `0011644` because its codes are the ones GISCO LAU 2021 carries.

## 2. The download

```
https://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd=0012311&Dim1=S7A2021&lang=PT
```

7.4 MB, TLS verifies normally, no key, no User-Agent games. 41,268 rows = 3,439 units × 12
(11 categories + total). `pindicaMeta.jsp?varcd=…` gives the codelists.

`Dim2=<geocod>` filters to one territory, which is how the 15+ population in §4 was fetched
without pulling another 11 MB cube. Worth knowing: this API takes dimension filters.

## 3. Six nested levels in one dimension, told apart by code length

India's C-01 trap yet again, and the cheapest instance of it. Every level is in the same
`geocod` column:

| shape | n | level |
|---|---|---|
| `PT` | 1 | country |
| 1 digit | 3 | Continente, Açores, Madeira |
| 2 chars | 9 | NUTS2 (`11` Norte … `1A` Grande Lisboa) |
| 3 digits | 26 | NUTS3 |
| 4 digits | 308 | município |
| **6 digits** | **3,092** | **freguesia** |

Summing the file whole gives six copies of the country. `_level()` keys on the shape, not on
position — Hungary's rule (§9h).

## 4. THE UNIVERSE IS NOT THE POPULATION, AND THE TABLE DOES NOT SAY SO

**This is the one thing about Portugal that would have gone wrong quietly.**

The 11 categories sum to the published total exactly, on all 3,439 units, to the person. There
is no `não respondeu` cell anywhere in the file. That reads as a mandatory question answered by
everyone. It is not:

```
resident population, all ages   10,343,066
resident population 15+          9,011,878   (indicator 0011609, total minus the 0-14 group)
published in the religion table  8,781,900
removed as non-response            229,978   = 2.55% of the universe
```

The religion question is *de resposta facultativa* and INE **dropped the people who declined
from the denominator** rather than publishing them as a category. So every share in this table
is a share of those who answered, and the difference is invisible from inside the file.

This is §9r's Guyana finding without the footnote that gave Guyana away: **an office can prorate
its own non-response away, and a perfect partition is not evidence that it did not.** The test
that catches it is differencing against a separate table on the same universe. `sources/pt.py`
asserts the gap, so a vintage that changes the convention fails loudly.

Combined with the 15+ restriction, **the map draws 84.9% of Portugal**. Not scaled up — Chile
(§9k) is the other 15+ source here and takes the same line.

## 5. The boundaries cost nothing, and this is the only country where that is unqualified

GISCO LAU 2021 has been on disk since Poland (§9e). Portugal's LAU **is** the freguesia:

```
census freguesias      3,092
GISCO PT polygons      3,092
unmatched, either way      0
names disagreeing          0   (after folding case and accents)
```

No concordance, no re-cutting, no name resolution, no Kontur, no derivation of any kind. Six
digits of `LAU_ID` are six digits of `geocod`. §9e wrote that "the GISCO LAU file is the boundary
answer for most of Europe"; every country since has still paid something. Portugal did not.

The names are checked anyway, because §12's rule is that a count match is not a join — and here
the check is free and passes on all 3,092.

**GISCO's `POP_2021` sums to 10,562,178 against INE's census 10,343,066.** Different
measurements (§9i), reported only, never used for allocation.

## 6. What the source is worth

- **3,092 units at about 2,800 answering people each** — after Estonia, the finest European
  geography here, and finer than Poland's gmina, Romania's UAT or Hungary's settlement.
- **80.2% Catholic, the highest share of any directly-asked country on this map.** Higher than
  Poland, Croatia or Ireland.
- **A clean north-south gradient**: Açores 91.6%, Norte 88.1%, Centro 86.5%, Oeste e Vale do
  Tejo 80.1%, Grande Lisboa 68.4%, Algarve 65.9%, Península de Setúbal 65.3%. No religion runs
  the other way, 6.2% to 25.6%.
- **A fifteen-year-old immigration visible at village scale.** São Teotónio is **17.1% Hindu**;
  Longueira/Almograve is **17.1% Buddhist, 9.3% Hindu, 6.2% Muslim and 43.7% Catholic** — the
  least Catholic freguesia in the country. This is the Odemira berry-and-greenhouse belt and its
  South and Southeast Asian workforce, and nothing else in Western Europe on this map looks
  remotely like it.
- **A five-hundred-year-old one at the same scale.** Belmonte: **49 people, 1.61% of the
  freguesia**, report Judaism, against 0.03% nationally — the crypto-Jewish community that kept
  practising in secret after 1497 and returned openly in the 1970s. A table that publishes single
  people is what makes them drawable.
- **Two immigrations separating cleanly.** Orthodox to the Algarve (3.2%, Almancil 8.5%) because
  that is the tourism labour market; Muslims to inner Lisbon (Santa Maria Maior 18.2%).

## 7. What it cannot show

- **One Protestant cell** for the Lusitanian Anglicans, the historic Presbyterians and
  Methodists, and the Brazilian and African Pentecostal churches together. Its sharpest geography
  is a handful of Alentejo border villages (Póvoa de São Miguel 12.9%, Sobral da Adiça 9.7%)
  rather than the cities, and the census cannot say what that is.
- **One Muslim cell**, which costs most here of anywhere: Portugal's community is substantially
  **Nizari Ismaili**, from Mozambique after 1975, and the Imamat's seat has been in Lisbon since
  2018 — alongside Sunni communities from Guinea-Bissau, Bangladesh, Nepal and Morocco.
- **No Sikh cell**, and `other.pt` has its shape: 0.28% nationally, **2.99% in Odemira**, sitting
  beside the Hindu and Buddhist peaks. §9u's and §9r's rule again — *a residual with a sharp
  geography is a missing category, not a mixture.*
- **No rite split** on Catholicism, so `christianity.catholic` and not `.latin`.

## 8. Left undone

**Placement is uniform within the freguesia (§8.2).** The median freguesia is 16.5 km², which is
fine, but the Alentejo units reach 863 km² (União das freguesias de Alcácer do Sal) at
single-digit people per km², so dots there spread across empty cork forest. A Kontur PT extract
would fix it exactly as it did for Kenya and Ethiopia. This is an improvement rather than a
correction — the tier is already fine enough that it is not misleading — and it is the only
open item.
