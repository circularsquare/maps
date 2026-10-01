# Portugal register sources (built 2026-09-30)

What the Portuguese build reads, where each piece came from, and what is still wrong with it.
Portugal is built with `rinf.py`; how the reader works is in its docstring, and its per-country
entry is `rinf_countries/pt.py`. Downloads live in `data/raw/rinf/pt/` (gitignored). Nothing
needed a login or a key.

## Run

```powershell
python rinf.py --fetch pt                                           # RINF + Wikidata, ~10 s
curl -L -o data/raw/portugal-260929.osm.pbf https://download.geofabrik.de/europe/portugal-260929.osm.pbf
python extract.py --region pt --pbf data/raw/portugal-260929.osm.pbf   # 40 s; delete the .pbf after
python inspect_region.py --region pt
python build_model.py --region pt --register rinf:data/raw/rinf/pt     # 20 s
python build_tiles.py --region pt                                      # 10 s
python check_model.py --region pt
python rinf.py --dry pt          # the reader alone, with its full log
```

Geofabrik's `portugal-latest.osm.pbf` redirect-looped on 2026-09-30; the dated file from
https://download.geofabrik.de/europe/portugal.html worked. The extract was run with
`OSMIUM_POOL_THREADS=2`.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-09-30 (`sections.json`: 781 sections, one version each; `points.json`: 749 points).
  131 line ids, 2,478 km, all Infraestruturas de Portugal (`0094_IM`). No point has a
  coordinate on `netReference`; all 749 have `geo:hasGeometry`, which `rinf.py` reads.
- **OpenStreetMap**, Geofabrik `portugal-260929.osm.pbf` (data to 2026-09-29), ODbL: track,
  stations, 197 passenger route relations, and 86 `route=railway` relations. IP's lines are
  mapped as `route=railway` with IP's own line number as `ref` ("8" Linha do Norte, "20" Linha
  da Beira Alta, "104" Ramal da Colpor) and the line's name; that is where the names come from.
- **Wikidata** (`wikidata.json`), CC0: 47 items with a route number (P1671) and P17 Portugal.
  They use IP's numbers too (33 Linha de Vendas Novas, 68 Variante de Alcácer), but only for a
  few, mostly closed or freight, lines. "1", "2", "3" and the letters are the Almada light rail
  and the Porto metro.
- **Published lengths** for `check_model.REGISTER["pt"]`: the line articles on pt.wikipedia and
  en.wikipedia, raw wikitext, retrieved 2026-09-30. Two cite IP's network statement (then
  REFER's Directório da Rede): Linha do Norte 336 km (2022, p.71) and Ramal de Tomar 14.8 km
  (2012, p.70). The rest cite nothing or old books, and a few infoboxes are wrong (Linha de
  Évora's infobox repeats the Algarve's 139.5; the Oeste's gives 215.1 against 197.9 in its own
  text), so the notes in `REGISTER` say which figure was taken. IP's current network statement
  (https://servicos.infraestruturasdeportugal.pt/pt-pt/parceiros/operacao-ferroviaria/os-nossos-servicos/diretorio-da-rede-ips)
  links only its addenda, not the main document, so it was not read.

## How `pt.py` reads the ids

IP's RINF id is its line number followed by one digit for the part of the line: `081` is
line 8, the Linha do Norte; `251` and `252` are the two parts of line 25, the Beira Baixa
(Entroncamento side, then Abrantes - Guarda); `011` and `012` are line 1, the Minho (São Bento -
Campanhã, then on to Valença). So the public number is the id less its last digit. OSM's
relations carry the same number, and 53 of the 60 ids kept confirm it that way
(`rule_certain` covers the rest).

Portuguese lines are known by name, so the number is only the `ref` ("8"); the name is the OSM
relation's ("Linha do Norte"), else Wikidata's label, else `PT_NAMES`.

Numbers are keyed zero-padded ("08") and shown bare ("8"). Wikidata's route numbers 1 and 3 are
the Almada light-rail lines, and `rinf.py` looks Wikidata up before OSM's name, so an unpadded
key named the Linha do Minho "Linha 1".

`skip_line` leaves out:

- every four-digit id: IP's private sidings and freight terminals, numbered 101-183 (66 ids,
  65 of them section nature 20, a link). A siding that leaves the main line and rejoins it
  gets the main line's ways in `build_model.register_way_lines`, so it read as ridden and came
  out as a 0.3 km "line" (Terminal de Loulé, Ramal Cacia Portucel).
- six numbered lines with no passenger train that survived for the same reason, or because
  both ends are stations: 65 Ramal do Barreiro-Terra, 66 Ramal Barreiro-Quimigal, 81 the Tadim
  terminal, 3 Concordância de São Gemil (no OSM passenger route on it: the Leixões trains run
  Leça do Balio - Contumil), 62 the 1.9 km stub of the Ramal da Figueira da Foz left at
  Pampilhosa (the line closed in 2009), 63 Linha da Matinha (the freight line beside the Norte
  out of Santa Apolónia).

`cut_at_junctions`: a section also ends where another line's sections meet. Without it,
Bifurcação de Águas de Moura-Sul (two neighbours on the Sul, but two other lines join there)
was merged through. The Intercidades' Águas de Moura - Pinheiro and the freight-only Praias do
Sado - Águas de Moura became one 26 km section, 41% ridden, and the Sul lost 12.4 km of the
Lisbon - Faro route. With it on, the Sul gains those 12.4 km, the Oeste loses 2.7 km that no
OSM route runs over (below), and no other Portuguese line changes. It is off by default because
in Belgium it costs ridden track (L.12 Antwerp - Essen 32.5 to 26.0 km, L.27 45.0 to 39.5):
there, stretches whose OSM routes have gaps were protected by being part of a stop-to-stop
section, which `drop_unridden_sections` never questions.

## What is in the register and what stays OSM

Kept: 24 register lines, 2,138 km, after `build_model` dropped junction-ended sections that no
OSM passenger route runs over (43 sections, 286 km). All IP's.

Left out as unridden, which is right: Linha de Vendas Novas (33; passenger service suspended,
Coruche, Muge and Marinhais have no OSM station), Linha de Sines (38), Ramal de Neves-Corvo
(79), Ramal do Pego (30), Ramal do Porto de Aveiro (90), the Linha do Alentejo beyond Beja (the
build's Alentejo is Barreiro - Beja), the old Sul through Alcácer do Sal (Pinheiro - Grândola
Norte; trains use the Variante de Alcácer), Praias do Sado - Águas de Moura, the freight
concordâncias (Águas de Moura, Bombel, Agualva, Funcheira, Ermidas, Norte do Setil), and the
Coimbra-B - Coimbra stub (211): Coimbra-A has no station in OSM any more, and its trace is
rejected.

Not in RINF at all: the closed Tua, Corgo, Tâmega, Sabor, Dão, Ramal de Cáceres and Ramal de
Portalegre. The **Linha do Vouga** (metre gauge, IP's, still running Espinho - Oliveira de
Azeméis and Aveiro - Sernada do Vouga) is not in RINF either, so it stays two OSM lines, "CP
Regional: Espinho - Vouga - Oliveira de Azeméis" and "CP Regional: Aveiro - Sernada do Vouga";
the suspended Oliveira de Azeméis - Sernada middle has no route and is not a line.

The **Linha do Leste** is kept: OSM has CP's Entroncamento - Badajoz regional over it.

Stay OSM lines, as every metro, tram and private operator does:

- **Metropolitano de Lisboa**: Azul, Amarela, Verde, Vermelha (subway).
- **Metro do Porto**: lines A to F (light rail). Some turnback stubs at the line ends have no
  route and are not drawn.
- **Metro Transportes do Sul** (Almada): Linhas 1-3 (light rail).
- **Fertagus**: one OSM line, Roma-Areeiro - Setúbal/Coina. It runs on IP track (the Sul
  over the Ponte 25 de Abril, and the Cintura), so riding it credits those register lines.
- **Trams**: Carris 12E, 15E, 18E, 24E, 25E, 28E; STCP 1, 18, 22; the Elétrico de Sintra;
  Viseu's funicular-tram (mapped as `route=tram`).
- **Funiculars**: Carris' Glória, Lavra, Bica and Graça, Porto's Guindais, Nazaré, Bom Jesus
  (Braga), Santa Luzia (Viana do Castelo). Six more funiculars and inclined lifts have track in
  OSM but no route relation, so they are not lines: Elevador do Mercado (Coimbra), Funicular de
  São João, Elevador da Goldra, Elevador de Santo André, Elevador do Castelo 2, and the
  Curtumes Aleu funicular.
- CP's services stay OSM lines over the register lines. Alfa Pendular, Intercidades and the
  Celta to Vigo are named trains (`build_model.looks_like_service`, the `pt` branch, 14
  relations); Regional, InterRegional and the Lisbon and Porto Urbanos are lines, as operating
  patterns are in Japan. `norm_line_name` strips "CP Lisboa"/"CP Porto", so "CP Lisboa: Linha
  de Cascais" merged into the register line and handed it its yellow; Linha de Guimarães and
  Linha de Leixões took the Porto Urbanos' colours the same way.

No `colours/pt.csv`: IP publishes no line colours, and CP's colours belong to services, which
keep the colours OSM gives them.

## Counts (2026-09-30)

- 24 register lines, 2,138 km.
- 102 lines in all: 24 register, 32 CP/Fertagus train lines, 14 named trains, 9 light rail, 4
  metro, 11 tram, 8 funicular. 870 stations, 462 of them on a register line; 9,813 route-km.
- 516 RINF passenger-typed points, of which 428 are an OSM station (424 distinct), 4 by
  distance alone (spelling: "Porto Rei" and "Porto de Rei"). The other 88 are closed halts,
  freight points and passing loops ("Alcácer do Sal", "Pinheiro", "Resguardo" points).

## Check

`python check_model.py --region pt`: against RINF's own section lengths, 23 lines of 2 km or
more, median 0.997, none off by more than 5%. Against the published figures, 17 lines, 15 of
them within 2%:

| line | built | published | ratio |
|---|---|---|---|
| Linha do Norte | 334.7 | 336.0 | 1.00 |
| Linha da Beira Baixa | 238.8 | 240.0 | 1.00 |
| Linha da Beira Alta | 200.0 | 202.0 | 0.99 |
| Linha do Oeste | 193.9 | 197.9 | 0.98 |
| Linha do Douro | 162.5 | 160.0 | 1.02 |
| Linha do Leste | 140.5 | 140.7 | 1.00 |
| Linha do Algarve | 139.3 | 139.5 | 1.00 |
| Linha do Minho | 133.0 | 133.6 | 1.00 |
| Linha de Guimarães | 30.1 | 30.1 | 1.00 |
| Linha de Sintra | 27.2 | 27.2 | 1.00 |
| Linha de Évora | 26.0 | 26.2 | 0.99 |
| Linha de Cascais | 25.2 | 25.4 | 0.99 |
| Ramal de Braga | 14.9 | 15.0 | 0.99 |
| Ramal de Tomar | 14.6 | 14.8 | 0.99 |
| Linha de Cintura | 11.0 | 10.5 | 1.05 |
| Ramal de Alfarelos | 14.6 | 16.5 | 0.89 |
| Linha de Leixões | 14.1 | 18.7 | 0.75 |

- **Linha de Leixões** (0.75): the last 4.6 km from Guifões into the port of Leixões is freight
  and is dropped as unridden. Leça do Balio - Guifões (3.5 km) has no passenger train either
  (the Urbanos turn at Leça do Balio), but both ends are OSM stations, so it is kept.
- **Ramal de Alfarelos** (0.89): RINF's own length is 14.7, so the gap is between the two
  published figures (the article's 16.5 runs to Bifurcação de Lares).
- **Linha de Cintura** (1.05): the build includes RINF's 1.0 km link from Alcântara-Terra on to
  Alcântara-Mar; the article's 10.5 stops at Alcântara-Terra.
- **Linha do Oeste** (0.98): Bifurcação de Lares - Amieira (2.7 km) is dropped, since no OSM
  passenger route runs over it (Coimbra - Caldas trains take the Verride curve). If CP still
  runs anything from Figueira da Foz south over it, OSM does not map it.
- **Linha do Sul** and **Linha do Alentejo** are not in `REGISTER`: their published figures
  (273.6 Campolide - Tunes; 217.6 Barreiro - Funcheira) include stretches the build rightly
  leaves out (the old line through Alcácer do Sal, Beja - Funcheira) or files elsewhere (the
  Variante de Alcácer is line 68), so they measure a different extent. Against RINF they are
  within 1%.

## Still off, and why

- **Frontier stubs** (Valença - Valença Fronteira 1.7 km, Elvas - Elvas Fronteira 10.7 km,
  Vilar Formoso 0.3 km) are kept, which is right: the Celta and the Badajoz regional cross
  them. No OSM line's section credits them, though, because those relations end at the
  extract edge, so riding the Celta to Vigo does not complete the Minho's last 1.7 km.
- **RINF's length is misallocated** in 2 places, kept because the trace runs on the line's own
  track: São Romão - São Frutuoso on the Minho (RINF 1.58, track 2.35) and a 0.2 km piece at
  Praias do Sado.
- **Two sections rejected**: Coimbra-B - Coimbra (the closed Coimbra-A stub) and the freight
  Ramal de Sines at the petrochemical plant.
- Lines have no English names: Wikidata's English labels exist only for lines the build leaves
  out, and OSM's relations carry none.
