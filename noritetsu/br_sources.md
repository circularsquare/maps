# Brazil register sources (built 2026-10-03)

What `br_register.py` reads, what runs and what does not, the line calls and why, and how the
build checks out. Downloads are in `data/raw/br/` (gitignored); this file is the record. Nothing
here needed a login, a key or an account.

## The short answer

- **Register lines: the passenger track of Brazil's trains, laid on OpenStreetMap.** Nearly all
  of Brazil's ~30,000 km of railway is freight (Rumo, VLI/FCA, MRS, Vale, Transnordestina) and
  is no line, as in the US and Mexico builds. 19 register lines, 2,136 km:
  - São Paulo's commuter lines 7 to 13 (CPTM, ViaMobilidade, TIC Trens, Trivia Trens), from
    OSM's track names, which are the lines' ("Linha 7 - Rubi" ... "Linha 13 - Jade");
  - SuperVia's eight Rio lines, Vale's two long-distance railways (EFVM, EFC), the Serra Verde
    Express's Curitiba - Morretes, and Teresina's metro, from the ways of their own OSM route
    relations (the El Chepe recipe of mx_register), because there the track is named for the
    freight railway (the EFVM's 1,100 named km include its ore branches) or for a track pair
    (SuperVia's "Via A" to "Via H").
- **Metros, light rail, monorails and trams stay OSM's route relations**, as in the US and UK
  builds: every city's routes are complete, with stops (survey below). 61 OSM lines.
- 80 lines in all (6 of them named trains), 728 stations, 5,463 route-km.
- `probe_kr_ways.py --region br`: 74.6% of main and branch rail km carry a name, but the names
  are freight railways' (Ferrovia Norte-Sul, Estrada de Ferro Carajás, Tronco Principal Sul),
  so Korea's recipe works only where the name is a passenger line's (São Paulo's CPTM track).

## What runs (checked 2026-10-03) and what this build does with it

| service | status | in the build |
|---|---|---|
| CPTM / ViaMobilidade / TIC Trens / Trivia Trens lines 7-13 | running (7 to TIC Trens since 26 Nov 2025; 11-13 passing to Trivia Trens from July 2026) | register lines, named track |
| Expresso Aeroporto (Barra Funda - Aeroporto over 11/12/13) | running | OSM line (an operating pattern over the register lines) |
| CPTM Expresso Turístico (Luz - Paranapiacaba / Jundiaí / Mogi, weekends) | running | named train (and its route_master has no routes in the extract) |
| SuperVia: Deodoro, Japeri, Santa Cruz, Paracambi, Belford Roxo, Saracuruna, Vila Inhomirim, Guapimirim | running | register lines (OSM's routes as operated stay, flagged dup) |
| Vitória - Belo Horizonte (Vale, EFVM), daily, 664 km, 30 stops; Desembargador Drumond - Itabira connecting train, daily | running | register line "Estrada de Ferro Vitória a Minas" (both); the train itself a named train |
| São Luís - Parauapebas (Vale, EFC), Mon/Thu/Sat out, Tue/Fri/Sun back, 892 km, 15 stops; daily from 2027 | running | register line "Estrada de Ferro Carajás"; the train a named train |
| Serra Verde Express, Curitiba - Morretes | Fri-Sun, daily 1 Dec - 6 Mar and the July holidays | register line "Curitiba - Morretes" |
| Maria Fumaça (Giordani Turismo), Bento Gonçalves - Garibaldi - Carlos Barbosa | Wed, Fri, Sat, Sun, two a day | OSM line, counted |
| Trem do Corcovado (rack), Bonde de Santa Teresa (tram) | daily | OSM lines, counted |
| São João del-Rei - Tiradentes (steam), Trem Republicano (Itu - Salto), Trem das Águas, Trem da Serra da Mantiqueira, Trem de Guararema, Campinas - Jaguariúna, Trem da Vale | weekends, or Fri-Sun | named trains (track counted nowhere) |
| Estrada de Ferro Campos do Jordão | no trains since 2024; concession auctioned April 2026, first stretch back about a year after signing | not built |
| Metrô SP 1-5, 15, 17 (Morumbi - Congonhas), 6 (João Paulo I - Perdizes, opened 2 Jul 2026) | running | OSM lines |
| Line 17's Washington Luís branch (opened June 2026) | running | missing: not in OSM's route yet |
| MetrôRio 1, 2, 4; VLT Carioca 1-4 | running | OSM lines |
| Trensurb (Porto Alegre) + Aeromóvel; Metrô BH 1 (+ OSM's 1.2 km line 2); Metrô-DF Verde/Laranja; Salvador 1, 2; Recife Centro 1/2, Sul, VLT Cabo, VLT Curado; Metrofor Sul, Oeste, Parangaba-Mucuripe, Ramal Aeroporto; Cariri, Sobral; Maceió Azul, Verde; João Pessoa; Natal Norte, Sul; Baixada Santista VLT 1, 2; Aeromóvel GRU | running | OSM lines |
| Teresina Linha 1, and its south-east branch to Colorado and Todos os Santos (from 11 May 2026, peak hours) | running | register line (OSM's route is broken, below) |
| VLT de Salvador (Calçada - Itacaranha, assisted operation weekdays since 29 Jun 2026) | running in trial | not built: OSM has no track or route for it yet |
| Teresina - Parnaíba railway | disused | not built (see Teresina) |

**No passenger train crosses a border.** The extract carries a few km of Argentina (Iguazú's
Tren Ecológico) and Bolivia (Expreso Oriental at Puerto Quijarro): named trains, counted
nowhere, left to their countries. No `borders.EXTRA` point.

## Sources

- **OSM** (Geofabrik `brazil-latest`, extracted into `data/proc/br` 2026-10-03; ODbL): 19,039
  track ways, 163 route relations, 68 route masters, 125 infrastructure relations.
- **Wikidata** (CC0): `python br_register.py --fetch` writes `data/raw/br/wikidata_stations.json`
  (1,423 rows of Brazilian station items). Used for one station OSM lacks: Marabá on the EFC
  (Q123459428). `data/raw/br/wikidata_lines.json`: line lengths (P2043) for the checks.
- **Vale** (vale.com, travel and news pages): the EFVM's 664 km and 30 stops, the Itabira
  connection, the EFC's 892 km, timetable and 15 stops (São Luís - Arari - Vitória do Mearim -
  Santa Inês - Alto Alegre - Mineirinho - Auzilândia - Altamira - Vila Pindaré - Nova Vida -
  Açailândia - São Pedro - Marabá - Itainópolis - Parauapebas).
- **Published lengths**: en.wikipedia's CPTM line infoboxes, pt.wikipedia "SuperVia",
  Wikidata P2043 (each in `check_model.REGISTER["br"]` and `KNOWN["br"]`).

Tried and not used:

- **ANTT's open data** (dados.antt.gov.br, CKAN `package_list` read 2026-10-03): road datasets,
  accident and performance reports, nothing with railway geometry or sections. ANTT's network
  data sits in each concession's Declaração de Rede (PDFs, freight track); the passenger trains
  run over only a sliver of it, which OSM's routes already pick out.
- **GTFS** (Transitous `feeds/br.json`): Rio (mdb-1791), São Paulo SPTrans (mdb-8), ARTESP, Belo
  Horizonte, Fortaleza ETUFOR and ARCE, Grande Recife, Porto Alegre, FlixBus. All city or state
  feeds. **No `gtfs_served.FEEDS["br"]` proposed**: gtfs_served judges every register section in
  the country, so a São Paulo or Rio feed would close the EFVM, the EFC and the other city's
  lines (the scope problem in HANDOFF thread 0). Worth trying once gtfs_served judges only a
  feed's own area: Rio's mdb-1791 (SuperVia) and SPTrans mdb-8 (CPTM in it).

## The line calls

- **Register lines are named as OSM's route masters** ("Linha 7 - Rubi", "Linha Japeri",
  "Linha 1 do Metrô de Teresina"), so each OSM line is matched to its register line: it hands
  over its colour and is dropped as a twin where it is the same line (lines 7-10, 12, 13,
  Deodoro, Paracambi, Vila Inhomirim), kept as the line as operated (flagged dup, not counted)
  where it runs further (Linha 11 from Barra Funda, the SuperVia branches from Central).
- **SuperVia's trunk Central - Deodoro is one line, "Linha Deodoro"**, all four tracks: the
  Deodoro locals use one pair, the Japeri and Santa Cruz trains the other, and the pairs lie
  side by side (Anita's rule on second tracks). `FOLD_INTO`: a SuperVia way within 45 m of the
  Deodoro line's track over 90% of its length is Deodoro's. So Linha Japeri is Deodoro -
  Japeri, Linha Santa Cruz Vila Militar - Santa Cruz (it shares the Japeri line's track from
  Deodoro to Vila Militar, ~1.5 km), Belford Roxo and Saracuruna start at Triagem where they
  leave the trunk. Elsewhere two SuperVia lines side by side stay two lines (folding the Santa
  Cruz branch into the Japeri line took Deodoro off it).
- **Each way belongs to one register line**, the first to claim it: named track (CPTM), then
  ROUTE_LINES in order (Line 8's unnamed Amador Bueno shuttle, the SuperVia lines in the order
  Deodoro, Japeri, Santa Cruz, Paracambi, Belford Roxo, Saracuruna, Vila Inhomirim, Guapimirim,
  then EFVM, EFC, Serra Verde, Teresina).
- **Vale's trains: the railways are lines, the trains named trains.** Both run more often than
  weekly (daily; three a week each way), so their track counts. The project's rule makes a
  single long-distance train a named train (Amtrak's, the Rocky Mountaineer); the line it
  runs over is the railway's passenger stretch, as NARN subdivisions are in the US and
  "Chihuahua al Pacífico" in Mexico. So "Estrada de Ferro Vitória a Minas" (Pedro Nolasco,
  Cariacica - Belo Horizonte, with the Desembargador Drumond - Itabira branch, whose daily
  connecting train is part of the same service) and "Estrada de Ferro Carajás" (São Luís -
  Parauapebas) are register lines, and OSM's "Trem de passageiros da EFVM / EFC" routes are
  named trains (`rules/br.py`).
- **Tourist trains: counted when they run on four or more days a week** (or daily in a season),
  named trains when on two or three. Anita's threshold is "more often than about once a week";
  Australia's build read it as daily heritage lines counting and 2-3-day ones not, and this
  follows it. Counted: the Serra Verde Express (Fri-Sun all year, daily December to early March
  and in July: seasonal lines are drawn as running), Giordani's Maria Fumaça (four days, two
  trains a day), the Corcovado and Santa Teresa trams (daily). Named trains: São João del-Rei -
  Tiradentes (Fri-Sun), and the weekend trains (Trem Republicano, Trem das Águas, Serra da
  Mantiqueira, Guararema, Campinas - Jaguariúna, CPTM's Expresso Turístico, Trem da Vale).
- **Serra Verde's line is Curitiba - Morretes only**, the part the train runs (the railway goes
  on to Paranaguá with freight). Its stations are the two ends, where the Serra Verde Express
  sells tickets; Cadeado, a station record on the track, was taken off.
- **Stations on the EFVM, EFC and Serra Verde lines are only the listed ones** (`STRICT`): the
  routes' stops plus `EXTRA_LISTS`. Their freight track carries stop nodes of stations no
  passenger train calls at (Aroaba, Acesita) and passes Belo Horizonte's metro station Central;
  those are taken off and their two sections joined. Belo Horizonte (the EFVM terminus, missing
  from OSM's route) and the Itabira connection's stop are added.
- **The EFC has 9 of its 15 stops**: OSM and Wikidata have no record of Arari, Auzilândia,
  Altamira, Vila Pindaré, Nova Vida and Itainópolis, so Anjo da Guarda - Vitória do Mearim is
  one 158 km section and Mineirinho - Açailândia 232 km.
- **Teresina's OSM route is broken**: relations 420628 and 10570394 run on from the city over
  the disused Teresina - Parnaíba railway to Luís Correia (359 km, 15 extra stops; the railway
  is "desativada e vandalizada" per pt.wikipedia, with only plans to reopen). The register line
  keeps the route's ways inside the city (`BBOX`) and adds the May 2026 branch's two stations
  (Colorado, Todos os Santos). Until build_model can leave a route out, the OSM line stays as a
  396 km "Linha 1 do Metrô de Teresina" flagged dup (not counted, but drawn): the proposed hook
  below removes it.
- "CTO", a railway=station record beside Central do Brasil in no route's stop list (a SuperVia
  operations point), is no stop (`NOT_STOPS`).

## Proposed: a build_model hook to leave a route out (managing session)

`rules/br.py` already sets `SKIP_ROUTES = {420628, 10570394}`; build_model does not read it yet.
Diff for `build_model.py`, in `build()` after `rules = country_rules(region)`:

```python
    # Route relations the country's rules leave out (rules/<cc>.py SKIP_ROUTES): stale or
    # broken OSM routes (Teresina's runs on over a disused railway to the coast).
    skip = set(getattr(rules, "SKIP_ROUTES", ()) or ())
    if skip:
        n0 = len(groups)
        groups = [(lid, mtags, [r for r in rids if r not in skip]) for lid, mtags, rids in groups]
        groups = [g for g in groups if g[2]]
        log(f"  {len(skip)} route relations left out by the country's rules (SKIP_ROUTES); "
            f"{n0 - len(groups)} lines with no route left")
```

and in `country_rules()`'s docstring:

```
      SKIP_ROUTES = {relation id, ...}
            Route relations the OSM half leaves out: stale or broken routes (Brazil's
            Teresina route runs on over a disused railway). Default: none.
```

A no-op for every country without the name. Trialled on br by wrapping `group_lines` the same
way (scratch `skip_trial.py`, output to a temp folder): 80 -> 79 lines, 728 -> 712 stations,
5,463 -> 5,066 route-km, the 396 km Teresina OSM line gone, register lines unchanged (19,
2,136 km). It also
answers HANDOFF's open "hook to leave out a stale OSM route" (Malaysia's Skypark Link, Mexico's
Línea Z route, the no-train routes in mk, xk, al).

## Checks

`python check_model.py --region br` (2026-10-03):

| line | built | published | ratio |
|---|---|---|---|
| Linha 7 - Rubi | 56.8 | 62.7 (en.WP; OSM's route 56.9, the figure is likely from Brás) | 0.91 |
| Linha 8 - Diamante | 41.7 | 42.0 | 0.99 |
| Linha 9 - Esmeralda | 35.9 | 39.1 (en.WP; OSM's route 35.9) | 0.92 |
| Linha 10 - Turquesa | 40.7 | 38.0 (WD; built from Barra Funda, OSM's Line 10 track) | 1.07 |
| Linha 11 - Coral | 50.5 | 54.1 less Barra Funda - Luz 3.6 | 1.00 |
| Linha 12 - Safira | 38.6 | 39.0 | 0.99 |
| Linha 13 - Jade | 8.8 | 12.2 (en.WP; the two ends are 7.7 km apart as the crow flies) | 0.72 |
| Linha Deodoro | 21.8 | 23.0 | 0.95 |
| Linha Japeri (Deodoro -) | 39.5 | 61.75 - 23 | 1.02 |
| Linha Santa Cruz (Vila Militar -) | 30.5 | 54.75 - 23 | 0.96 |
| Linha Paracambi | 8.4 | 8.26 | 1.02 |
| Linha Vila Inhomirim | 15.3 | 15.35 | 1.00 |
| Estrada de Ferro Vitória a Minas | 692.7 | 664 + Itabira 34 | 0.99 |
| Estrada de Ferro Carajás | 873.7 | 892 | 0.98 |
| Linha 1 do Metrô de Teresina | 16.9 | 13.5 + branch 3.3 | 1.00 |

Not checked: Linha Belford Roxo (26.4 from Triagem; pt.WP gives 27.7 from Central, which does
not square with OSM's 32.8 km route), Linha Saracuruna (29.5 from Triagem; 34.02 from Central),
Linha Guapimirim (40.4; pt.WP's 17.3 cannot be Saracuruna - Guapimirim, whose ends are 33 km
apart as the crow flies), Curitiba - Morretes (68.4; no published figure for the stretch).

OSM lines (`KNOWN["br"]`): São Paulo Metro 1, 3, 4, 5 at 0.99-1.00 with every station; MetrôRio
1 at 1.06, 2 at 1.01; Trensurb 0.99 (22/22 stations); Natal Norte 1.00; Corcovado 0.98;
Aeromóvel GRU 0.95; Baixada Santista VLT 1 0.89 (Wikidata's figure takes in the depot).

## Commands

    python br_register.py --fetch          # Wikidata stations, one SPARQL query
    python br_register.py --names          # track names and km in the extract
    python build_model.py --region br --register br_register:data/raw/br     # ~45 s
    python build_tiles.py --region br                                        # ~20 s, 3.3 MB
    python check_model.py --region br

No clip step: the extract's few foreign km are named trains only.

## Open

- The proposed SKIP_ROUTES hook (above): until it lands, Teresina's 396 km OSM line is drawn.
- OSM data to fix at source: Teresina's route (stops to Luís Correia), the EFVM route's missing
  Belo Horizonte stop, the EFC route's 7 missing stops, Line 17's Washington Luís branch, the
  VLT de Salvador (no track mapped yet).
- The EFC's six stops with no record anywhere (above).
- A city feed for SuperVia and CPTM once gtfs_served has a scope.
- Watch for: the EFC train going daily (2027), Campos do Jordão's return, Line 6's extension
  (mid 2027), Fortaleza's Linha Leste, the Trem Intercidades São Paulo - Campinas (2031),
  Salvador's VLT leaving trial operation.
