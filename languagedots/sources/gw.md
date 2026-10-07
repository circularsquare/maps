# Guinea-Bissau: RGPH 2009, main ethnic language, national by etnia, placed by região

Drawn 2026-10-05 (session edd42a8c-gw). 1,310,587 Guinean nationals in ordinary households,
15 nodes from 15 table columns, every row `measured`, on one national unit. 1,304 dots at
1:1000, no rings. Across the nine regiões the dots are placed through the etnias that name each
language (the census's região x etnia table); inside a região, by Kontur population.

```
python sources/gw_rgph.py --fetch     # copies religiondots' PDF (SHA-1 pinned); all checks
python sources/gw_place.py            # religiondots' hexes, one national unit, região kept as `reg`
python taxonomy/build.py
python tools/check_country.py gw
python scatter.py --country gw
```

## CLEAR Global has no Guinea-Bissau

Checked 2026-10-05 on HDX's API: CLEAR Global's organisation (`clear`) holds 55 datasets, one
per country, and none is Guinea-Bissau (`guinea-languages` is Guinea; `cabo-verde`, `gambia`,
`senegal` and `sierra-leone` are there). A plain search for Guinea-Bissau language data finds
only boundary and settlement layers. Guinea and Sierra Leone's CLEAR files came from IPUMS
census samples, and IPUMS has no Guinea-Bissau sample, so none is likely. The placement below
uses the census's own região x etnia table instead.

## Source

- **Volume.** INE Guiné-Bissau, RGPH 2009, *Características socioculturais* (92 pp), the
  volume religiondots draws the country's religion from (`religiondots/sources/gw.md`, which
  has the fetch history and the truncated-Wayback trap). `--fetch` copied religiondots'
  verified file (2,725,536 bytes, SHA-1 `3SHQDVYZ...`).
- **Questions** (Anexo 2, the individual form; definitions PDF p18):
  - **P.15 "Qual é o principal Dialecto falado?"**, one write-in answer, coded. The volume
    defines a *dialecto* as the language of an etnia, and Kriol, Portuguese and foreign
    languages as *línguas*, asked in P.16. So P.15 is the main ethnic language. **Drawn.**
  - **P.16**, yes/no for Crioulo, Portuguese, French, English, Spanish, Russian and "another
    language": languages known, several allowed (Crioulo 90.4%, Portuguese 27.1%, French
    5.1%, Quadro 6). **Not drawn**: it never asks which comes first, it names no ethnic
    language, and a single-answer table from the same census is preferred (brief §2).
- **Table drawn: Anexo Quadro 4** (PDF pp73-74), etnia (16 rows incl. "Sem Etnia" and "ND") x
  principal dialecto (Total, "Sem dialecto", 14 ethnic languages, "NA"), counts. Its Total row
  is the national count per language. The coverage sweep found only P.16's tables (Quadros
  6-8A) and the Gráfico 5 shares; the counts table is in the annex.
- **Grain.** National. No região x language table exists in this volume, the regional
  booklets (religiondots §1 read all nine) or the structure volume. Microdata is not public
  (not in IPUMS).

## Placement across regiões (a weight only; counts are INE's)

Language is published only for the whole country, by etnia. Anexo Quadro 2 (p71) gives each
etnia's people by região. A language's speakers in região r are estimated as the sum over
etnias of [people of that etnia naming the language] x [the etnia's share living in r]. This
moves people only inside the unit the census counted them in (brief §4.4), and each language's
national count stays the table's.

- **Bissau (SAB) is all urban**, and town and country differ: "Sem dialecto" is 11.8% of the
  urban population and 2.1% of the rural (Quadros 6, 6A). So SAB's people of each etnia take
  that etnia's urban rates (the first page's 8 columns; the continuation's columns split as
  the etnia's split nationally), and the rest of each etnia's speakers go to the other eight
  regiões by Anexo Quadro 2. Without this, SAB's "Sem dialecto" share was 10%; with it, 14%.
- **Anexo Quadro 2's two misprints** (religiondots §5) are corrected: Fula in Oio 2,980 ->
  23,980, Mandinga in Cacheu 1,460 -> 11,460; every row and column then closes (check 3c), and
  the misprinted cells are what is on the page (3d).
- **What it assumes**: an etnia names the same languages in every região outside Bissau. A
  Balanta in Oio and in Tombali are taken to answer alike. The regional split is an estimate,
  and `how` and `note_public` say so.

What the placement gives (estimated % of the região's drawn people): Gabú Fula 78, Mandinka 14;
Bafatá Fula 59, Mandinka 22; Oio Balanta 42, Mandinka 30, Fula 12; Cacheu Manjak 35, Balanta
28, Ejamat 9; Biombo Papel 60, Balanta 19; Quinara Balanta 34, Biafada 32; Tombali Balanta 45,
Fula 21, Nalu 7, Susu 6; Bolama/Bijagós Bijagó 53, Sem dialecto 11; SAB Balanta 18, Fula 18,
Sem dialecto 14, Papel 13, Mandinka 12. The dots per região match the estimates within
sampling (one dot is 1,000 people).

## Checks (`sources/gw_rgph.py`, all pass)

| check | result |
|---|---|
| file | 92 pages, 2,725,536 bytes, SHA-1 pinned, %%EOF |
| Quadro 4 parse | 17 rows x 17 columns off pp73-74 = the transcription |
| closure | every etnia row and every column closes; total 1,442,227 |
| Quadro 4 vs Anexo Quadro 2 | etnia totals equal; Anexo Quadro 2 (corrected) closes on all 9 região totals |
| urban + rural (Quadros 6, 6A) | = Quadro 4 in all 153 cells of the first page, and on the continuation's Total row |
| men + women (Quadros 5, 5A) | = Quadro 4 on the continuation's Total row; men close on both pages (698,119). 5A's first page is missing from the volume |
| Gráfico 5 (p33) | each etnia's own-language share, and Sem Etnia's 69.2% "sem dialecto", = the 15 printed bars |
| NA | 6.6-9.9% of every etnia; r = +0.91 against each etnia's share aged 0-14 (Quadro 2) |
| join | the 9 regiões = religiondots' gw_hexes units; placement sums back to every national count |

## The universe and the gap

Guinean nationals in ordinary households, all ages, 1,442,227 (religiondots §3 has the
arithmetic). **NA, 131,640 (9.1%), is not drawn.** It is "no answer recorded": 100% of the
1,274 with no etnia recorded, 6.6-9.9% of every etnia, near equal by sex (66,244 men, 65,396
women), and it tracks each etnia's share of children (r = +0.91), so it is mostly infants with
no language yet. `gap` is 9.8% of the 1,452,926 enumerated: NA 9.1% plus the 0.7% outside every
table (foreign nationals, no nationality recorded, collective households). Not corrected for
the 4.6% post-enumeration omission, as in religiondots.

## Mapping calls (`taxonomy/gw2009.py`, `taxonomy/tree.d/gw.txt`)

- **Reused**: Balanta, Manjak, Mankanya (sn), Fula, Nalu (gn), Mandinka, Susu, Soninke.
- **New, flat under Atlantic**: Papel (pape1239), Biafada (biaf1240), Mansoanka (mans1259),
  Bijagó (one census answer for Glottolog's two Bijagó languages, one leaf as gn did for Baga),
  Balanta Mané (no glottocode; the census prints it apart from Balanta, so a sibling leaf).
- **Felupe -> Ejamat** (ejam1238), under sn.txt's Jola group, where Glottolog puts it.
- **"Sem dialecto" -> `other.no_ethnic_language`**, a grey leaf labelled for what the census
  says. 85,356 people (6.5% of those drawn), 79% of them urban. The census filed there anyone
  whose principal language is not an ethnic one: Kriol, Portuguese or foreign. By P.16 and the
  volume's prose (p34: these people "do not consider any dialect principal"), nearly all are
  Kriol-first, but the table cannot tell Kriol from Portuguese, and no node holds both, so
  it is not drawn as Kriol. Reversing it (onto a new Upper Guinea Crioulo node, uppe1455) is
  a one-line change in gw2009.py.

**Colours.** Papel purple (Biombo, Bissau, beside Balanta's light cyan and Manjak's dark
teal), Balanta Mané rose and Mansoanka pale yellow (Oio, Cacheu), Biafada ochre (Quinara),
Bijagó orange-brown, Ejamat a darker Jola blue. Closest pair that shares ground: Manjak and
Ejamat in Cacheu, about 0.11 in OKLab.

## Not escalated

Language here follows ethnicity closely, and the placement uses the census's own região x
etnia table, which INE publishes. Nine units averaging 160,000 people, no finer than the
religion layer religiondots already draws from the same volume; the country's instability has
been military and political, not linguistic.

## Terms

INE Guiné-Bissau's report is a public PDF with no licence text. Kontur CC BY 4.0; COD-AB CC
BY-IGO; Glottolog CC BY.
