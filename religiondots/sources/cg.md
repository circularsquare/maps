# Republic of the Congo — RGPH 2007, religion by département

**Drawn 2026-09-14** (session `f95259a4-cg`). 12 départements, 9 categories, 3,697,490 residents,
the whole census population, every row `measured`, in counts.

- `sources/cg.py` -> `data/normalized/cg.csv` (the PDF is `data/raw/cg/rgph2007pd.pdf`)
- `sources/cg_geo.py` -> `data/geo/cg/cg_departements.gpkg`, `cg_hexes.gpkg`, `cg_lookup.csv`
  (COD-AB ADM1 + Kontur 400 m)
- `taxonomy/cg2007.py` -> the mapping; `countries.py` `"cg"` -> the wiring;
  `taxonomy/branches.py` `other.cg` is the one new node
- sources.md **§9dv** is the write-up; the route is **§11aq**'s.

```
python sources/cg.py     --fetch
python sources/cg_geo.py --fetch
```

## 1. What CNSEE publishes

| release | religion | tier |
|---|---|---|
| **RGPH 2007, *Le RGPH-2007 en quelques chiffres*** (CNSEE with UNFPA, July 2010, 23 pp) | **Tableau 11, département x nine answers, counts**, with the national row in counts and shares | **12 départements** |
| the same brochure, Tableau 1 | district and commune populations, no religion | 86 districts and communes |
| UNSD Demographic Yearbook table 28 | **absent**: Congo forwarded no religion tabulation | none |
| RGPH-5 (2023) | preliminary total 6,142,180 only; no thematic volumes (§11aq) | none |

Scout `f95259a4-scout-af` found the table and the domain problem (§11aq). Nothing finer than the
département was found for 2007; `ins-congo.cg`'s keyless API and the CDX of `ins-congo.cg` and
`cnsee.org` hold no religion file (§11aq, not re-derived).

## 2. Cite and fetch only the Wayback copy

`cnsee.org` is squatted. The live `cnsee.org/pdf/rgph2007pd.pdf` serves the same brochure reflowed
to 20 pages with spam links injected into the text layer. **The only URL used or cited is**

    web.archive.org/web/20111113144639id_/http://www.cnsee.org/pdf/rgph2007pd.pdf

The `id_` form returns the archived bytes rather than the Wayback frame. The file is 1,120,560
bytes, `%PDF-1.5`, Word 2007, author `Léonard`, created 2010-07-02, 23 pages, and contains no
`http` string (the squatted copy is full of links). `cg.py --fetch` refuses a body of any other
size or with a URL in it.

**Its `%%EOF` is not in the last 2 KB.** There are four `%%EOF` markers, the last at byte
1,027,215, followed by 93,345 bytes of Word's free-object xref list that stops mid-entry, and
PyMuPDF opens the file as repaired. The content is complete: all 23 pages carry text, the page
count matches the brochure's own table list, and Tableau 11 closes to the person. The Wayback CDX
returned 503 and then 504 ("Temporarily Offline") three times on 2026-09-14, so whether another
capture has a clean trailer is **unchecked**.

## 3. The checks (all in `sources/cg.py`)

| check | result |
|---|---|
| Tableau 11 parsed off the page = the transcription | 12 départements, every cell identical |
| each département's nine answers vs Tableau 1 | equal to the person, all 12 |
| each column vs the printed national row | equal to the person, all 9 |
| national row vs the resident population | 3,697,490 = 3,697,490 |
| printed national shares vs counts | within 0.054 pp; `Sans religion` is 11.354% and printed 11,3 (the printed shares then sum to 100.0) |
| Tableau 1 totals | men + women = total; sum 3,697,490 (1,821,357 + 1,876,133); the 12 totals appear again in Tableaux 3, 20 and 21 |
| UNSD table 28 | no Congo row, so no external count check exists |
| COD-AB name join | 12/12; `Lekoumou` by folding, `Point-Noire` by alias |
| Kontur 2023 vs census 2007 | 1.645x overall; 0.45x (Likouala) to 5.04x (Cuvette-Ouest), used only inside each département |

**The questionnaire.** RGPH-06 *Feuille de ménage ordinaire* (the census was planned for 2006;
its reference date is 28 April 2007), UNSD's scan
`unstats.un.org/unsd/demographic/sources/census/quest/COG2007fr.pdf`, 4 pages, image only, read
from renders; IREDA holds a second scan, `cog-2007-rec-q1_quest_menage_ordinaire.pdf`, not read
separately. **P13 *Religion*, asked of every resident: CA=1, PR=2, SA=3, KI=4, MU=5, ER=6, AN=7,
AU=8, SR=9**, which is Tableau 11's column order and count exactly. There is no non-response
code, which is why the table covers the whole resident population and this country has no `gap`.
P12 asks ethnicity (Bantu ethnic group, or *pygmée*); it is not used.

## 4. What the table shows

| | Cath | Prot | Salv | Kimb | Musl | Réveil | Anim | Autres | None |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Congo | 33.1 | 19.9 | 2.2 | 1.5 | 1.6 | 22.3 | 0.7 | 7.4 | 11.4 |
| Brazzaville | 42.8 | 15.1 | 2.3 | 1.9 | 2.3 | 25.2 | 0.5 | 3.5 | 6.4 |
| Pointe-Noire | 33.2 | 23.8 | 1.8 | 1.3 | 2.1 | 19.4 | 0.9 | 8.6 | 9.0 |
| Pool | 42.3 | 21.2 | 4.4 | 2.7 | 0.2 | 12.7 | 1.1 | 9.5 | 6.0 |
| Bouenza | 30.6 | 30.3 | 1.1 | 1.0 | 0.3 | 11.6 | 0.6 | 10.8 | 13.7 |
| Niari | 27.0 | 33.0 | 3.6 | 1.1 | 0.8 | 11.6 | 0.7 | 11.1 | 11.2 |
| Lékoumou | 16.8 | 37.1 | 7.3 | 0.2 | 0.3 | 11.6 | 0.4 | 5.6 | 20.6 |
| Kouilou | 20.0 | 16.9 | 3.2 | 1.5 | 0.5 | 16.7 | 1.7 | 29.3 | 10.2 |
| Plateaux | 9.4 | 9.7 | 1.8 | 1.4 | 0.2 | 25.0 | 1.6 | 16.9 | 33.9 |
| Cuvette | 21.4 | 7.1 | 0.3 | 0.3 | 0.8 | 37.8 | 0.4 | 3.9 | 27.9 |
| Cuvette-Ouest | 19.9 | 14.5 | 0.2 | 0.1 | 0.6 | 33.6 | 1.3 | 4.4 | 25.3 |
| Sangha | 12.7 | 12.8 | 0.2 | 0.8 | 2.6 | 38.8 | 0.5 | 6.0 | 25.7 |
| Likouala | 19.3 | 23.6 | 0.3 | 1.4 | 2.8 | 38.7 | 1.1 | 4.5 | 8.4 |

The south (Pool, Bouenza, Niari, Lékoumou) is Catholic and mission Protestant, and carries almost
all the Salvation Army and Kimbanguists outside the two cities. The north (Cuvette, Cuvette-Ouest,
Sangha, Likouala) is revival churches and no religion. Brazzaville holds 48.1% of Catholics, 42.0%
of the revival churches, 39.0% of Salvationists, 47.9% of Kimbanguists and 53.8% of Muslims.

The mapping calls, with reasons, are in `taxonomy/cg2007.py`'s REVIEW: revival churches to
`christianity.pentecostal` (the family node, not `.charismatic`, not `christianity.evangelical`);
`Protestante` to the answer-node; `Autres` to a new `other.cg`; `Sans religion` drawn as printed
although Plateaux's 33.9% beside a 1.6% animist box is unlikely to be mostly secular.

## 5. Vintage and boundaries

**The twelve 2007 départements are COD-AB's twelve** (`cod-ab-cog` v01, boundaries of 2017, valid
2019-06-17). Brazzaville and Pointe-Noire were already départements in 2007.

**Congo has fifteen since October 2024**, and the new three are not drawn. Laws 24-2024 to
34-2024 of 8 October 2024, *Journal officiel* no. 42 of 17 October 2024, pp1304-1308 (read from
`sgg.cg/JO/2024/congo-jo-2024-42.pdf`):

| new département (chef-lieu) | districts (law) | in 2007 (Tableau 1) |
|---|---|---|
| Djoué-Léfini (Odziba) | Ignié, Mayama, Vindza, Kimba, Ngabé, Odziba (new, law 24-2024) | all Pool |
| Nkéni-Alima (Gamboma) | Gamboma, Abala, Allembé, Ollombo, Ongogni, Makotimpoko | all Plateaux |
| Congo-Oubangui (Mossaka) | Mossaka, Loukoléla, Liranga, Bokoma (new) | Mossaka and Loukoléla Cuvette; Liranga Likouala |

Laws 29-33 redefine the territory of Brazzaville, Cuvette, Likouala, Plateaux and Pool. A 2007
religion table cannot be carried onto the fifteen without the district cut, which was never
published.

**COD-AB's two cities are bigger than 2007's.** Brazzaville is 246 km² on COD-AB against 100 km²
in Tableau 3, Pointe-Noire 208 against 43.7. Hexes whose centroids fall between the two lines take
the city's religion mix rather than Pool's or Kouilou's. Left, because the rural mixes around both
cities are close to the city mixes in the large answers.

## 6. Placement

**Kontur's Brazzaville block is `real`** in `kontur_cap.csv`: 78 hexes, 15 at the cap, 91.1% of
the unit, which is the city itself. Its peak is 4.2 km from Brazzaville's point in
`maps/data/worldcities.csv`, where Kontur is already 41,452/km², and it holds 1.34x the city's
figure. Pointe-Noire does not reach the cap (peak 4,642/km²).

**Kontur's spread inside several départements disagrees with 2007's district table**, and it was
measured and left: the dots of a département all carry the same mix, so this moves density and not
colour. Pool puts 23% of its weight on the Brazzaville fringe around Kintélé and Ignié, against
Ignié district's 12.4% of Pool in 2007 (the city has since grown into Pool); Lékoumou puts 29% near
Sibiti against the district's 48%; Likouala puts 20% near Impfondo against 31%. The refinement, if
anyone wants it, is `_HT_COMMUNE_LEVEL`'s move (spec §12, Haiti): scale each district's hexes to
its Tableau 1 share. COD-AB's 89 districts line up with Tableau 1's except that Dolisie sits inside
Louvakou, Mossendjo inside Moutamba, and Kayes and Nkayi share one polygon.

## 7. §14 was considered and no ask was filed

Pool was the theatre of the 1998-2003 and 2016-17 conflicts involving Frédéric Bintsamou's Ninja
militia, whose movement has a religious identity of its own. The census has no code for it, the
tier is twelve départements of 308,000 people on average, and the table is the office's own
publication, archived since 2011. Nothing here places any group more finely than CNSEE did. The
autochthonous (Pygmy) population is counted by département in Tableau 24 of the same brochure and
is not used.

## 8. Gotchas

- **Tableau 11 prints `Cuvette-0uest` with a digit zero.** `cg.py`'s `norm` folds it.
- **`Sans religion` is printed 11,3% and is 11.354%.** Do not read a transcription error into it;
  the counts are exact and the printed shares were forced to 100.
- **COD-AB spells `Point-Noire`.** The join has one alias for it.
- **The two questionnaire scans have no text layer**, so a text search of them says nothing.
- **`cnsee.org` is not the office any more.** Anything fetched from it, including this brochure,
  carries injected links. `ins-congo.cg` is the office now.

## 9. Terms

CNSEE's brochure is a free public publication; the archived copy is cited, not redistributed.
UNSD's questionnaire collection and IREDA's inventory are public. COD-AB is CC BY-IGO. Kontur
Population is CC BY 4.0.
