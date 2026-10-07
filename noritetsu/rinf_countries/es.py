"""Spain: Adif's network, all of it under one RINF manager code (0071_IM), in all three gauges:
Iberian broad gauge, the standard-gauge high-speed lines, and the metre gauge that was FEVE's
until 2013 (Ferrol - Gijón - Santander - Bilbao, La Robla, Cercedilla - Los Cotos, Cartagena -
Los Nietos). RINF has no FGC, Euskotren, FGV, SFM, metro or tram line; those stay OSM lines.
2,520 sections, 466 line ids, 15,537 km (fetched 2026-10-02), no validity versions.

NUMBERS. Every RINF id is "ESL" + Adif's three-digit line number + six digits of an older
section code: ESL100100010 is a piece of line 100 (Madrid - Hendaya), ESL740874000 line 740
(Ferrol - Pravia, metre gauge), ESL050210000 line 050 (the Madrid - Barcelona high-speed line;
standard-gauge lines are 0xx). Those numbers are the catalogue's, Orden FOM/710/2015 and Adif's
Declaración sobre la Red, and Wikidata's P1671 uses them too. So the number is ids[3:6] and it
IS Adif's number (`rule_certain`). One exception: ESL893002600 is Lleida - La Pobla de Segur
beyond PK 1.9, which Adif handed to the Generalitat (FGC runs it) and RINF files under a
catch-all 893; it is joined to line 206, Adif's 1.9 km start of the same line (`fixed`).

NAMES are the catalogue's, from es.wikipedia "Anexo:Líneas de la Red Ferroviaria de Interés
General" (which cites Orden FOM/710/2015 and Adif's Declaración sobre la Red), written
"100 Hendaya – Madrid-Chamartín-Clara Campoamor", English "Line 100 (Hendaya – ...)". The
annex writes Castilian exonyms (Lérida, Gerona, La Coruña, Játiva); they are put back to the
official names Adif's own stations carry (Lleida-Pirineus, Girona, A Coruña, Xàtiva). ROUTES
also keeps the catalogue length (km), gauge and whether Adif AV administers the line (Adif's
high-speed arm; one RINF code covers both), which `im_of` reads.

LEFT OUT (`skip_line`): ids numbered 870-899 other than La Pobla: port, factory and yard sidings
(Puerto de Barcelona, Mercabarna, Cepsa, Ensidesa, Fasa-Renault, the gauge changers at Pedralba
and Taboadela, Cádiz-Puerto), none in the catalogue and none with a passenger train.

FIXES TO RINF (`fix`, es_fix), each logged by the build:
- Stretches RINF does not have at all, though the catalogue line runs through them and trains
  use them (checked against the ERA endpoint for any country: no section has these points as
  ends). 21 of them, 526 km, most on the high-speed lines (217 km of them on 982, Olmedo -
  Zamora - Ourense). Each is added as a section between the RINF points either side (GAPS), with its
  length from the catalogue where the line's shortfall is one gap, else crow-fly times 1.06
  (high-speed) or about 1.1 (conventional); the traced length decides whether it is kept, as
  for any RINF section. Two stations on them that RINF lacks too, Sanabria AV and A Gudiña,
  are added as stops (NEW_STOPS), and three Taboadela gauge-changer links are left out
  (DROP_SECTIONS), so Zamora - Ourense runs stop to stop (see NEW_STOPS for why it matters).
- Two border links RINF lacks, so borders.py can join the neighbours by the shared border
  point: line 508, Badajoz - the Portuguese border towards Elvas (5.3 km in the catalogue;
  border point EU00125 from Portugal's RINF), and the Spanish part of the Perpignan - Figueres
  high-speed line from RINF's "LÍMITE ADIF - LFPSA" to the French border point EU00121
  (LFP Perthus's concession, not Adif's; its own line with no number).
- Three section lengths that cannot be right (TYPO_KM: 333 km for 21 km, 0.21 km for 4.3 km,
  line 320 16.7 km long), one point whose coordinate contradicts its sections (NO_COORD), and
  three high-speed points RINF places nearer the conventional line beside them (MOVE).

STOPS (`stop_name`): six RINF points took the wrong OSM station: four named like a station of
the other gauge or of the conventional line nearby (found by comparing the gauge of the track
at RINF's coordinate with that at the matched station), La Sagrera (unopened), and Portbou,
typed technical. STOP_NAMES and NOT_STOPS say which.

TRACING (`osm_rel`): three OSM route=railway relations carry no ref; REL_REF reads them as
Adif's number, so the second trace pass keeps 982 and 984 on the high-speed line.

JUNCTIONS (`cut_at_junctions`, CUT_AT; for the timetable check, live since 2026-10-03): eight
points where a branch leaves a main line at a junction the reader would merge away inside the
main line, so the branch's end met nothing and gtfs_served found no path onto it. Cutting
there kept 984 (the Pajares base tunnel), 320 Chinchilla - Hellín, 422 Arahal - Utrera, 500
Cañaveral - Cáceres, 818 Padrón - Bif. Angueira and 828 A Portela. Riquelme-Sucina (NOT_STOPS),
where no train calls, is no stop, so 352 El Reguerón - Balsicas is one section the Murcia -
Cartagena trains run to the end of.

What is still off, and the numbers: es_sources.md.
"""
import re

from rinf_countries import osm_ref_default

# Adif line number -> (catalogue name, catalogue km, gauge, administered by Adif AV).
# Gauge: "ibé" Iberian 1668 mm, "est" standard 1435 mm, "mét" metre, "mix" mixed. Rows marked
# "gone" are lines the annex lists greyed (dismantled or left the network); harmless here.
ROUTES = {
    "010": ("Madrid-Puerta de Atocha-Almudena Grandes – Sevilla-Santa Justa", 470.5, "est", True),
    "012": ("Madrid-Puerta de Atocha-Almudena Grandes – Cambiador Atocha", 1.3, "est", True),  # gone
    "014": ("Bifurcación Gobantes – Bifurcación Bobadilla", 9.0, "est", True),
    "016": ("Majarabique – Cambiador Majarabique", 2.0, "est", True),
    "018": ("Bifurcación Cerro Negro/Santa Catalina – CTT Cerro Negro-Alta Velocidad", 0.3, "est", True),
    "020": ("La Sagra – Toledo", 21.4, "est", True),
    "022": ("Cambiador Alcolea – Bifurcación Cambiador Alcolea", 0.7, "est", True),
    "024": ("Yeles-aguja km 34,397 – Bifurcación Los Blancales", 5.7, "est", True),
    "026": ("Plasencia – Bifurcación San Nicolás", 175.3, "ibé", True),
    "030": ("Bifurcación Málaga-Alta Velocidad – Málaga-María Zambrano", 154.5, "est", True),
    "032": ("Antequera-Santa Ana – Cambiador Antequera", 0.4, "est", True),
    "036": ("Antequera-Santa Ana – Granada", 125.7, "est", True),
    "040": ("Madrid-Chamartín-Clara Campoamor – Valencia-Joaquín Sorolla", 397.6, "est", True),
    "042": ("Bifurcación Albacete – Alacant-Terminal", 237.8, "est", True),
    "044": ("Bifurcación Jesús – Bifurcación Joaquín Sorolla UIC", 0.5, "est", True),
    "046": ("Bifurcación Murcia – El Reguerón-Aguja km 522,1", 52.0, "est", True),
    "048": ("Bifurcación Vinalopó – Monforte del Cid AV", 2.1, "est", True),
    "050": ("Límite ADIF-LFPSA – Madrid-Puerta de Atocha-Almudena Grandes", 752.4, "est", True),
    "052": ("Cambiador de Plasencia de Jalón – Bifurcación Cambiador Plasencia de Jalón", 3.8, "est", True),
    "054": ("Bifurcación Moncasi – Bifurcación Canal Imperial", 25.9, "est", True),
    "056": ("Bifurcación Artesa de Lleida – Bifurcación Les Torres de Sanui", 16.5, "est", True),
    "058": ("Cambiador Lleida – Bifurcación Cambiador Lleida", 0.5, "est", False),  # gone
    "060": ("Bifurcación Cambiador Zaragoza-Delicias – Cambiador Zaragoza-Delicias", 0.4, "est", True),
    "064": ("Bifurcación Cambiador Roda – Roda de Bará-Cambiador de Ancho", 1.8, "est", False),  # gone
    "066": ("Bifurcación Can Tunis-Alta Velocidad – Can Tunis-Alta Velocidad", 0.2, "est", True),
    "068": ("Vallecas-Alta Velocidad-aguja km 12,300 – Los Gavilanes-aguja km 13,400", 5.6, "est", True),
    "070": ("Bifurcación Huesca – Huesca", 78.9, "est", False),
    "072": ("CTT de Fuencarral Alta Velocidad – Cambiador Madrid-Chamartín", 0.1, "est", True),
    "074": ("Cambiador de Medina del Campo – Olmedo AV-aguja km 133,9", 19.9, "mix", False),  # gone
    "076": ("Cambiador Valdestillas – Bifurcación Cambiador de Valdestillas", 1.0, "est", True),  # gone
    "078": ("Cambiador Valladolid-Campo Grande – Valladolid-Campo Grande", 0.9, "est", True),  # gone
    "080": ("Burgos-Rosa Manzano – Madrid-Chamartín-Clara Campoamor", 304.0, "est", True),
    "082": ("Bifurcación A Grandeira aguja km 85,0 – Bifurcación Coto da Torre", 84.0, "ibé", True),
    "084": ("León – Bifurcación Venta de Baños", 127.9, "est", True),
    "100": ("Hendaya – Madrid-Chamartín-Clara Campoamor", 640.9, "ibé", False),
    "102": ("Bifurcación Aranda – Madrid-Chamartín-Clara Campoamor", 280.6, "ibé", False),
    "104": ("Alcobendas-San Sebastián de los Reyes – Universidad de Cantoblanco", 6.9, "ibé", False),
    "106": ("Hendaya – Irún", 2.2, "ibé", False),
    "108": ("Complejo Ferroviario de Mercancías de Valladolid-aguja – La Carrera", 2.0, "ibé", False),
    "110": ("Segovia – Villalba de Guadarrama", 62.7, "ibé", False),
    "112": ("Valladolid-Campo Grande – Valladolid-Argales", 4.3, "ibé", False),
    "114": ("Complejo Ferroviario de Mercancías de Valladolid – Bifurcación Canal del Duero", 8.0, "mix", False),
    "116": ("Los Cotos – Cercedilla", 18.1, "mét", False),
    "120": ("Villar Formoso – Medina del Campo", 201.0, "ibé", False),
    "122": ("Salamanca – Ávila", 111.1, "ibé", False),
    "124": ("Salamanca – Valdunciel", 12.4, "ibé", False),
    "126": ("Aranda de Duero-Montecillo – Aranda de Duero-Chelva", 1.8, "ibé", False),  # gone
    # Not in the annex; ends and length from RINF.
    "128": ("Cambiador de Burgos – Burgos-aguja km 374,2", 0.2, "est", True),
    "130": ("Gijón-Sanz Crespo – Venta de Baños", 306.1, "ibé", False),
    "132": ("Bifurcación Tudela-Veguín – Ablaña", 5.3, "ibé", False),
    "134": ("León Clasificación – Bifurcación León Clasificación", 1.3, "ibé", False),
    "136": ("Cambiador de Burgos – Burgos-Rosa Manzano", 0.6, "est", True),
    "138": ("Bifurcación Galicia – Bifurcación Asturias", 0.9, "ibé", False),
    "140": ("Bifurcación Tudela Veguín – El Entrego", 21.5, "ibé", False),
    "142": ("Soto de Rey – Bifurcación Olloniego", 2.3, "ibé", False),
    "144": ("San Juan de Nieva – Villabona de Asturias", 20.8, "ibé", False),
    "146": ("Bifurcación Viella – Bifurcación Peña Rubia", 0.5, "ibé", False),
    "148": ("Trasona – Nubledo", 5.7, "ibé", False),
    "150": ("Aboño – Serín", 9.0, "ibé", False),
    "152": ("Gijón-Puerto (El Musel) – Veriña", 4.6, "ibé", False),
    "154": ("Lugo de Llanera – Tudela-Veguín", 14.1, "ibé", False),
    "156": ("Bifurcación Villamuriel de Cerrato – Cambiador Villamuriel", 1.3, "ibé", False),
    "158": ("Cambiador Villamuriel – Bifurcación Cerrato", 1.8, "est", True),
    "160": ("Santander – Palencia", 217.2, "ibé", False),
    "162": ("Solvay-Factoría – Sierrapando", 5.7, "ibé", False),
    "164": ("Palencia-Arroyo Villalobón – Magaz", 7.5, "ibé", False),
    "166": ("Bifurcación Rubena – Villafría", 3.7, "ibé", False),
    "168": ("Villafría – Bifurcación Rubena-aguja km 377,3", 3.5, "ibé", False),
    "170": ("Bifurcación Soto – Bifurcación Cerrato", 0.7, "est", True),
    "172": ("Cambiador Madrid-Chamartín – Madrid-Chamartín-Clara Campoamor", 0.7, "ibé", False),
    "174": ("Medina del Campo – Cambiador Medina del Campo", 2.1, "ibé", False),
    "176": ("Valdestillas – Cambiador Valdestillas", 0.8, "ibé", False),  # gone
    "178": ("Valladolid aguja km 250,2 – Cambiador Valladolid-Campo Grande", 0.6, "ibé", False),  # gone
    "180": ("Bifurcación Estadio Municipal – Cambiador Clasificación", 0.4, "est", True),
    "182": ("Cambiador Clasificación – Bifurcación Clasificación", 0.4, "ibé", False),
    "184": ("Bifurcación Río Bernesga – Cambiador Vilecha", 0.7, "ibé", False),
    "186": ("Cambiador Vilecha – Bifurcación Cambiador Vilecha", 0.6, "est", True),
    "188": ("Cambiador de Medina AV – Bifurcación Arroyo de La Golosa", 3.0, "ibé", False),
    "190": ("Medina AV – Cambiador de Medina AV", 1.1, "est", True),
    "200": ("Madrid-Chamartín-Clara Campoamor – Barcelona-Estació de França", 699.7, "ibé", False),
    "202": ("Torralba – Soria", 103.6, "ibé", False),
    "204": ("Bifurcación Canfranc – Canfranc", 138.5, "ibé", False),
    "206": ("Lleida-Pirineus – La Pobla de Segur", 1.9, "ibé", False),
    "208": ("San Juan de Mozarrifar – San Gregorio", 3.5, "ibé", False),
    "210": ("Miraflores – Sant Vicenç de Calders", 275.9, "ibé", False),
    "212": ("Hoya de Huesca-aguja km 2,3 – Bifurcación Hoya de Huesca", 1.7, "ibé", False),
    "214": ("CIM de Zaragoza – La Cartuja", 25.5, "ibé", False),
    "216": ("Bifurcación Plaza-aguja km 1,4 – Bifurcación Plaza-aguja km 8,7", 2.0, "ibé", False),
    "218": ("Bifurcación Plaza – Zaragoza-Plaza", 4.5, "ibé", False),
    "220": ("Lleida-Pirineus – Bifurcación Vilanova", 181.7, "ibé", False),
    "222": ("La Tor de Querol-Enveitg – Bifurcació Aigües", 149.7, "ibé", False),
    "224": ("Cerdanyola Universitat – Cerdanyola del Vallès", 3.6, "ibé", False),
    "230": ("La Plana-Picamoixons – Reus", 20.8, "ibé", False),
    "234": ("Reus – Constantí", 31.2, "ibé", False),
    "238": ("Castellbisbal-Agujas Llobregat – Barcelona-Morrot", 22.4, "mix", False),
    "240": ("Sant Vicenç de Calders – L'Hospitalet de Llobregat", 71.0, "ibé", False),
    "242": ("Martorell-Seat – Aguja km 71,185", 2.5, "ibé", False),
    "244": ("Aguja km 70,477 – Aguja km 0,500", 0.5, "ibé", False),
    "246": ("Mollet-Sant Fost – Castellbisbal-Agujas Rubí", 27.3, "mix", False),
    "248": ("Bifurcación Rubí – Aguja km 4,014", 1.9, "ibé", False),  # gone
    "250": ("Bellvitge-aguja km 674,8 – L'Hospitalet de Llobregat", 1.7, "ibé", False),
    "254": ("Aeroport – El Prat de Llobregat", 6.7, "ibé", False),
    "260": ("Figueres-Vilafant – Vilamalla", 6.4, "mix", False),
    "270": ("Cerbère – Bifurcación Aragó", 162.1, "ibé", False),
    "272": ("Bifurcación Girona-Mercaderies – Girona-Mercaderies", 1.8, "ibé", False),
    "274": ("Cerbère – Portbou", 2.2, "ibé", False),
    "276": ("Maçanet-Massanes – L'Hospitalet de Llobregat", 85.1, "ibé", False),
    "278": ("La Llagosta – Bifurcación nudo de Mollet", 2.3, "mix", False),
    "280": ("Bifurcación Mollet – Bifurcación nudo de Mollet", 2.5, "est", True),
    "282": ("Cambiador Plasencia de Jalón – Cambiador Plasencia-aguja km 308,6", 1.4, "ibé", False),
    "284": ("CIM-aguja km 337,1 – CIM-aguja km 0,7", 0.6, "ibé", False),
    "286": ("La Cartuja-aguja km 23,3 – La Cartuja-aguja km 351,1", 1.1, "ibé", False),
    "288": ("Miraflores-aguja km 345,6 – Miraflores-aguja km 0,9", 0.9, "ibé", False),
    "290": ("CIM-aguja km 337,1 – Cambiador Zaragoza-Delicias", 0.3, "ibé", False),
    "294": ("Roda de Bará-Cambiador de Ancho – Roda de Bará", 0.2, "ibé", False),  # gone
    "298": ("Girona-Mercaderies – Bifurcación Girona-Mercaderies", 0.6, "est", True),
    "300": ("Madrid-Chamartín-Clara Campoamor – València-Estació del Nord", 480.6, "ibé", False),
    "302": ("Aguja km 146,1 – Alcázar de San Juan", 2.0, "ibé", False),
    "304": ("Alfafar-Benetússer – València-Font de Sant Lluís-aguja km 1,3", 4.8, "ibé", False),
    "306": ("San Vicente de Raspeig – San Vicente de Raspeig-aguja km 448,7", 2.3, "ibé", False),  # gone
    "308": ("Albacete-Los Llanos – Cambiador Albacete", 0.5, "est", True),
    "310": ("Aranjuez – València-Font de Sant Lluís", 353.9, "ibé", False),
    "312": ("Castillejo-Añover – Algodor", 11.4, "ibé", False),
    "314": ("Xirivella-L'Alter – València-Sant Isidre", 1.9, "ibé", False),
    "316": ("Pinto – San Martín de la Vega", 15.4, "ibé", False),  # gone
    "318": ("Cambiador Albacete – Albacete-aguja km 279,4", 0.3, "ibé", False),
    "320": ("Chinchilla de Montearagón-aguja km 298,4 – Murcia del Carmen", 146.2, "ibé", False),
    "322": ("Águilas – Murcia-Cargas", 111.6, "ibé", False),
    "324": ("Aguja km 0,8 – Cartagena", 0.6, "ibé", True),
    "326": ("Aguja km 523,2 – Dársena de Escombreras", 11.4, "ibé", False),
    "328": ("Valencia-Alta Velocidad-aguja km 396,7 – Cambiador Valencia", 0.1, "est", True),
    "330": ("La Encina – Alacant-Terminal", 78.3, "ibé", False),
    "332": ("La Encina-Aguja km 3 – Caudete", 5.9, "ibé", False),
    "334": ("San Gabriel – Alicante-Benalúa", 1.7, "ibé", False),  # gone
    "336": ("El Reguerón-aguja km 525,3 – Alacant-Terminal", 73.7, "ibé", False),
    "338": ("Cambiador Valencia – Valencia-Joaquín Sorolla", 0.5, "ibé", False),
    "340": ("Moixent – Bifurcación Moixent", 0.5, "ibé", False),
    "342": ("Alcoi – Xàtiva", 63.7, "ibé", False),
    "344": ("Gandia – Silla", 50.8, "ibé", False),
    "346": ("Gandia-Puerto – Gandia-Mercancías", 2.9, "ibé", False),
    "348": ("Ford – Silla", 4.4, "ibé", False),
    "350": ("Bifurcación Benalúa – Bifurcación Alacant", 2.2, "ibé", False),
    "352": ("El Reguerón-aguja km 522,1 – Cartagena", 58.0, "ibé", True),
    "354": ("El Reguerón-aguja km 522,1 – Murcia del Carmen", 3.2, "mix", True),
    "360": ("Cartagena Plaza Bastarreche – Los Nietos", 19.6, "mét", False),
    "400": ("Alcázar de San Juan – Cádiz", 576.9, "ibé", False),
    "402": ("Espeluy-aguja km 340,1 – Jaén", 32.8, "ibé", False),
    "404": ("Espeluy-aguja km 338,8 – Espeluy-aguja km 150,5", 0.9, "ibé", False),
    "406": ("Las Aletas – Universidad de Cádiz (apeadero)", 2.4, "ibé", False),
    "408": ("Alcolea-aguja km 431,9 – Cambiador Alcolea", 0.4, "ibé", False),
    "410": ("Linares-Baeza – Almería", 240.8, "ibé", False),
    "412": ("Minas del Marquesado – Huéneja-Dólar", 14.4, "ibé", False),  # gone
    "414": ("Bifurcación Almería – Bifurcación Granada", 0.7, "ibé", False),
    "416": ("Moreda – Granada", 56.7, "ibé", False),
    "418": ("Santa Ana-aguja km 50,4 – Santa Ana-aguja km 48,3", 2.3, "ibé", False),
    "420": ("Bifurcación Las Maravillas – Algeciras", 179.6, "ibé", False),
    "422": ("Bifurcación Utrera – Fuente de Piedra", 113.4, "ibé", False),
    "424": ("Bifurcación Málaga – La Roda de Andalucía", 8.4, "ibé", False),  # gone
    "428": ("Cambiador Antequera – Santa Ana-Aguja km 50,4", 0.6, "ibé", False),
    "430": ("Bifurcación Córdoba-Mercancías – Los Prados", 188.8, "ibé", False),
    "432": ("Córdoba – El Higuerón", 6.5, "ibé", False),
    "434": ("Cambiador Córdoba – Valchillón", 5.3, "ibé", False),  # gone
    "436": ("Fuengirola – Málaga-Centro Alameda (apeadero)", 30.8, "ibé", False),
    "438": ("Huelva Mercancías – Puerto de Huelva", 6.0, "ibé", False),  # gone
    "440": ("Bifurcación Los Naranjos – Huelva", 109.1, "ibé", False),
    "442": ("Cambiador Majarabique – Bifurcación Los Naranjos", 1.8, "ibé", False),
    "444": ("Bifurcación Tamarguillo – La Salud", 11.5, "ibé", False),
    "446": ("Bifurcación La Cartuja – La Cartuja", 2.2, "ibé", False),
    "448": ("Bifurcación San Jerónimo – Bifurcación Los Naranjos", 0.8, "ibé", False),  # gone
    "450": ("Bifurcación La Negrilla – Bifurcación San Bernardo", 0.6, "ibé", False),
    "452": ("Puerto de Sevilla – La Salud", 5.4, "ibé", False),
    "454": ("Cambiador Majarabique – Bifurcación San Jerónimo", 1.4, "ibé", False),
    "456": ("La Salud-aguja km 6,2 – La Salud-aguja km 10,2", 0.8, "ibé", False),
    "458": ("Majarabique-Estación – Bifurcación San Jerónimo", 1.8, "ibé", False),
    "460": ("Bifurcación Las Maravillas – Fuente de Piedra", 11.8, "ibé", False),
    "464": ("Bifurcación Tocón – Bifurcación La Chana", 32.5, "ibé", False),
    "500": ("Bifurcación Planetario – Bifurcación Casa de la Torre", 322.7, "ibé", False),
    "502": ("Cáceres – km 428,5 (frontera)", 105.8, "ibé", False),
    "504": ("Villaluenga-Yuncler – Algodor", 16.8, "ibé", False),
    "506": ("Asland – Aguja km 5,7", 1.2, "ibé", False),  # gone
    "508": ("Badajoz – km 517,6 (frontera)", 5.3, "ibé", False),
    "510": ("Bifurcación Granja Las Encinas – Aljucén", 1.5, "ibé", True),
    "512": ("Zafra – Huelva-Cargas", 180.9, "ibé", False),
    "514": ("Zafra – Jerez de los Caballeros", 46.7, "ibé", False),
    "516": ("Mérida – Los Rosales", 14.2, "ibé", False),
    "518": ("Cáceres-aguja km 82,2 – Bifurcación Romanos", 4.2, "ibé", False),
    "520": ("Ciudad Real – Badajoz", 336.7, "ibé", False),
    "522": ("Manzanares – Ciudad Real", 64.5, "ibé", False),
    "524": ("Ciudad Real-Miguelturra – Bifurcación Poblete", 1.9, "ibé", False),
    "526": ("Puertollano – Puertollano-Refinería", 7.4, "ibé", False),  # gone
    "528": ("Almorchón – Mirabueno", 130.1, "ibé", False),
    "530": ("Monfragüe – Bifurcación El Chaparral", 6.6, "ibé", True),
    "532": ("Monfragüe-aguja km 255,4 – Monfragüe-aguja km 4,4", 2.7, "ibé", True),
    "534": ("Bifurcación El Chaparral – Arroyo de la Herrera", 2.7, "ibé", True),
    "536": ("Bifurcación San Esteban – Bifurcación El Chaparral", 2.6, "ibé", True),
    "600": ("València-Estació del Nord – Cambiador de La Boella", 254.1, "ibé", True),
    "602": ("València-Font de Sant Lluís-aguja km 2,3 – Valencia-Puerto Norte", 0.9, "ibé", False),
    "604": ("Les Palmes – Puerto de Castellón", 6.8, "ibé", False),
    "606": ("València-Font de Sant Lluís-aguja km 1,3 – Valencia-Puerto Sur", 2.3, "ibé", False),
    "608": ("Clasificación-València-Font de Sant Lluís – València-Font de Sant Lluís-aguja km 1,6", 0.8, "ibé", False),
    "610": ("Sagunt – Bifurcación Teruel", 314.5, "ibé", False),
    "612": ("Sagunt-aguja km 28,3 – Sagunt-aguja km 268,8", 0.6, "ibé", False),
    "614": ("València-aguja estación Alta Velocidad – Valencia-Joaquín Sorolla", 0.7, "ibé", False),
    "620": ("Tortosa – L'Aldea-Amposta-Tortosa", 13.1, "ibé", False),
    "622": ("Aguja-Clasificación km 272,0 – Tarragona-Classificació", 1.1, "ibé", False),
    "624": ("Aguja-Clasificación km 100,4 – Tarragona", 3.1, "ibé", False),
    # Not in the annex; ends and length from RINF (OSM: "FFCC Port Aventura - Tarragona", ref 630).
    "630": ("Tarragona – Salou-Port Aventura", 9.9, "ibé", False),
    "632": ("Bifurcación La Feredat – Bifurcación Vilaseca", 1.5, "ibé", True),
    "640": ("Cambiador de La Boella – Camp de Tarragona", 12.2, "est", True),
    "700": ("Intermodal Abando Indalecio Prieto – Casetas", 233.8, "ibé", False),
    "702": ("Cabañas de Ebro – Grisén", 6.0, "ibé", False),
    "704": ("Bifurcación Rioja – Bifurcación Castilla", 1.6, "ibé", False),
    "710": ("Altsasu – Castejón de Ebro", 139.2, "ibé", False),
    "712": ("Bifurcación km 534,0 – Bifurcación km 231,5 (Altsasu-Pueblo)", 2.5, "ibé", False),
    "720": ("Santurtzi – Intermodal Abando Indalecio Prieto", 13.6, "ibé", False),
    "722": ("Muskiz – Desertu-Barakaldo", 13.0, "ibé", False),
    "724": ("Bilbao-Mercancías – Santurtzi", 3.3, "ibé", False),
    "726": ("Bifurcación La Casilla – Aguja de enlace", 2.8, "ibé", False),
    "740": ("Pravia – Ferrol", 269.0, "mét", False),
    "750": ("Gijón-Sanz Crespo – Pravia", 29.0, "mét", False),
    "752": ("Laviana – Gijón-Sanz Crespo", 29.0, "mét", False),
    "754": ("Sotiello – Puerto El Musel", 2.0, "mét", False),
    "756": ("Aguja Enlace Sotiello – Aguja Enlace Veriña", 0.7, "mét", False),
    "758": ("La Maruca Mercancías – Puerto de Avilés", 2.0, "mét", False),
    "760": ("Oviedo – Trubia", 12.0, "mét", False),
    "762": ("Trubia – San Esteban de Pravia", 29.0, "mét", False),
    "764": ("Trubia – Collanzo", 55.0, "mét", False),
    "770": ("Santander – Oviedo", 216.0, "mét", False),
    "772": ("Liérganes – Orejo", 10.0, "mét", False),
    "774": ("Maliaño La Vidriera – Puerto de Raos", 2.0, "mét", False),
    "776": ("Ribadesella Puerto – Llovio", 3.0, "mét", False),
    "780": ("Bilbao Concordia – Santander", 110.0, "mét", False),
    "782": ("Ariz – Basurto Hospital", 8.0, "mét", False),  # gone
    "784": ("Lutxana-Barakaldo – Irauregui", 6.0, "mét", False),  # gone
    "790": ("Aranguren – Asunción Universidad", 310.0, "mét", False),
    "792": ("Matallana – La Robla", 11.0, "mét", False),
    "794": ("Guardo – Central Térmica de Velilla", 2.0, "mét", False),
    "800": ("A Coruña – León-Aguja km 123,6", 428.2, "ibé", False),
    "802": ("Toral de los Vados – Villafranca del Bierzo", 9.1, "ibé", False),
    "804": ("Betanzos-Infiesta – Ferrol", 42.8, "ibé", False),
    "806": ("La Bañeza – Astorga", 21.8, "ibé", False),
    "810": ("Bifurcación Chapela – Monforte de Lemos", 177.8, "ibé", False),
    "812": ("Vigo-Guixar – Bifurcación Chapela", 6.4, "ibé", True),
    "814": ("Guillarei – Valença", 7.8, "ibé", False),
    "816": ("Guillarei-aguja km 141,6 – Guillarei-aguja km 0,9", 1.0, "ibé", False),
    "818": ("Vilagarcía de Arousa – Bifurcación Angueira", 27.9, "ibé", True),
    "820": ("Zamora-aguja km 233 – Medina del Campo", 90.2, "ibé", False),
    "822": ("Bifurcación Valorio – A Coruña", 436.3, "ibé", False),
    "824": ("Redondela – Santiago de Compostela", 92.0, "ibé", True),
    "826": ("Central térmica de Meirama – Cerceda-Meirama", 11.9, "ibé", False),
    "828": ("Bifurcación San Amaro – Portas", 7.6, "ibé", False),
    "830": ("Bifurcación Uxes – Bifurcación San Cristóbal", 0.7, "ibé", False),
    "832": ("Aguja km 545,4 – Bifurcación San Diego", 0.5, "ibé", False),
    "834": ("A Coruña-San Diego – Bifurcación El Burgo", 1.7, "ibé", False),
    "836": ("Bifurcación León – Bifurcación Río Bernesga", 3.2, "ibé", False),
    "838": ("Bifurcación Torneros – Bifurcación Quintana", 3.1, "ibé", False),
    "840": ("Cerceda-Meirama-aguja km. 0,729 – Meirama-Picardiel", 0.5, "ibé", False),
    "842": ("Bifurcación Río Sar – Bifurcación A Grandeira aguja km 376,1", 1.1, "ibé", False),
    "848": ("Redondela AV – Bifurcación Redondela AV", 1.0, "ibé", True),
    "850": ("Vigo-Urzaiz – Bifurcación Arcade", 17.9, "ibé", True),
    "900": ("Madrid-Chamartín-Clara Campoamor – Madrid-Atocha Cercanías", 7.8, "ibé", False),
    "902": ("Pitis – Hortaleza", 8.3, "ibé", False),
    "904": ("Bifurcación Fuencarral – Fuencarral-aguja km 4,5", 1.7, "ibé", False),
    "906": ("Fuencarral-Complejo – Madrid-Chamartín-Clara Campoamor", 1.3, "ibé", False),
    "908": ("Hortaleza – Aeropuerto T-4", 5.3, "mix", False),
    "910": ("Madrid-Atocha Cercanías – Pinar de Las Rozas", 28.0, "ibé", False),
    "912": ("Las Matas – Pinar de Las Rozas", 3.6, "ibé", False),
    "914": ("Bifurcación Chamartín-aguja km 18,6 – Bifurcación Príncipe Pío", 0.7, "ibé", False),
    "916": ("Delicias – Madrid-Santa Catalina", 4.1, "ibé", False),
    "920": ("Móstoles-El Soto – Parla", 45.3, "ibé", False),
    # Not in the annex; ends and length from RINF.
    "924": ("Bifurcación Chamartín – Bifurcación Príncipe Pío", 1.3, "ibé", False),
    "922": ("Bifurcación Parla – Parla-Industrial", 1.1, "ibé", False),  # gone
    "930": ("Madrid-Atocha Cercanías – San Fernando de Henares", 18.3, "ibé", False),
    "932": ("Madrid-Atocha Cercanías – Madrid-Santa Catalina", 5.4, "ibé", False),
    "934": ("Madrid-Abroñigal – Bifurcación Rebolledo", 3.2, "ibé", False),
    "936": ("San Cristóbal Industrial – Villaverde Bajo", 3.0, "ibé", False),
    "940": ("O'Donnell – Vicálvaro-Clasificación", 4.0, "ibé", False),
    "942": ("Villaverde Bajo – Vallecas-Industrial", 7.2, "ibé", False),
    "944": ("Vicálvaro – Bifurcación Vicálvaro-Clasificación", 5.0, "ibé", False),
    "946": ("Madrid-Santa Catalina – Villaverde Bajo", 1.9, "ibé", False),
    "948": ("Vicálvaro-Clasificación – Bifurcación Vicálvaro-Clasificación", 2.9, "ibé", False),
    "950": ("Madrid-Puerta de Atocha-Almudena Grandes – Aguja km 1,4", 1.4, "ibé", False),  # gone
    "982": ("Taboadela aguja km 234,0 – Bifurcación Medina", 313.9, "est", True),
    "984": ("Pola de Lena – Bifurcación Pajares", 49.3, "mix", True),
}

# Stretches of catalogue lines RINF has no section for (module docstring, FIXES): line number,
# RINF point either side (uopid), length in km, and where the length comes from. "catalogue"
# means the line's catalogue length less RINF's sum, when this is the line's only gap.
GAPS = [
    ("982", "ESB0740", "ES08242", 69.1, "Bif. Medina - Toro AV (Olmedo - Zamora HSL), 1.06 x 65.2"),
    ("982", "ESB0843", "ESXSAN1", 86.5, "Bif. Los Conforcos - Sanabria AV (Zamora - Pedralba), "
                                        "1.06 x 81.6"),
    ("982", "ESXSAN1", "ESB3120", 6.3, "Sanabria AV - Bif. Pedralba, 1.06 x 5.9"),
    ("982", "ES08250", "ESXGUD1", 8.1, "Vilavella AV - A Gudiña-Porta de Galicia, 1.08 x 7.5"),
    ("982", "ESXGUD1", "ES08252", 47.0, "A Gudiña-Porta de Galicia - Miamán, 1.06 x 44.4"),
    ("050", "ES04103", "ES04104", 12.0, "Alcover AV - Camp de Tarragona, 1.08 x 11.1"),
    ("050", "ES71801", "ES04303", 58.0, "Barcelona-Sants - Riells (through La Sagrera), 1.09 x "
                                        "53.4; catalogue shortfall 68.7 less the 12.0 above is 56.7"),
    ("080", "ESB0813", "ES08014", 22.0, "Bif. Las Pajareras - Dueñas (Valladolid - Venta de Baños), "
                                        "1.08 x 20.4"),
    ("080", "ESB0814", "ESB0816", 4.8, "Bif. Venta de Baños - Bif. La Vega (towards Burgos), "
                                       "1.08 x 4.4"),
    ("084", "ESB0818", "ESA0823", 79.0, "Bif. Las Barreras - Bif. Cambiador Vilecha (Palencia - "
                                        "León), catalogue"),
    ("120", "ES33001", "ES33003", 17.4, "Tejares-Chamberí - Barbadillo y Calzada (Salamanca - "
                                        "Fuentes de Oñoro), catalogue"),
    ("300", "ES64004", "ESB6091", 30.3, "Vallada - La Encina (through Moixent), 1.12 x 27.1"),
    ("300", "ES64200", "ES64107", 9.4, "Silla - Benifaió, 1.08 x 8.7"),
    ("322", "ES06006", "ES06007", 11.2, "Lorca-Sutullena - Puerto Lumbreras (towards Águilas), "
                                        "catalogue"),
    ("230", "ES73102", "ES71400", 9.1, "La Selva del Camp - Reus, the traced length (9.08; "
                                       "6.7 crow-fly; catalogue shortfall 9.5)"),
    ("416", "ES57006", "ES05000", 6.8, "Albolote - Granada, catalogue"),
    ("026", "ESB3602", "ESB3750", 20.6, "Peñas Blancas - Bif. La Isla (Madrid - Extremadura HSL), "
                                        "1.06 x 19.5"),
    ("400", "ES51419", "ES51407", 5.9, "Río Arillo - Cortadura (San Fernando - Cádiz), 1.05 x 5.6"),
    ("700", "ES81006", "ES81005", 5.4, "Fuenmayor - Cenicero (Logroño - Haro), 1.08 x 5.0"),
    ("752", "ES05421", "ES05431", 7.3, "Tuilla - Ciaño Escobio (Langreo line), the traced "
                                       "length (7.35; 4.6 crow-fly, up the valley)"),
    ("800", "ES20002", "ES20003", 9.9, "Quintana-Raneros - Villadangos (León - Astorga), 1.06 x 9.4"),
]

# Stations on the gaps that RINF lacks with them, added as passenger points so the gap sections
# run stop to stop: the Zamora - Ourense high-speed stretch is otherwise one 213 km section
# ending at the Taboadela junction, and only one track of the pair carries OSM's Alvia
# relations, so build_model judged it 43% ridden and dropped it. Coordinates are the OSM
# stations' ("Sanabria Alta Velocidad", "A Gudiña-Porta de Galicia"), which they then match.
NEW_STOPS = {"ESXSAN1": ("Sanabria Alta Velocidad", -6.562912, 42.044769),
             "ESXGUD1": ("A Gudiña-Porta de Galicia", -7.134370, 42.063765)}

# The Taboadela gauge-changer links, left out so that A Gudiña - Miamán - Taboadela - Ourense
# is one stop-to-stop section rather than three junction-ended ones (section labels).
DROP_SECTIONS = {"Section of Line TABOADELA AV AG KM 446,1 - TABOADELA AG. KM. 447,1",
                 "Section of Line TABOADELA AG. KM. 447,1 - TABOADELA AG. KM. 234,0",
                 "Section of Line TABOADELA AV - TABOADELA AV AG KM 446,1"}

# Border points from the neighbours' RINF (border_points.json), for the two links RINF lacks.
BORDER = {"EU00125": ("Portugal – Spain border", -7.03015, 38.92024),
          "EU00121": ("France – Spain border", 2.86132, 42.45494)}
LINKS = [
    ("ESL508GAP", "ES37606", "EU00125", 5.5, "line 508 Badajoz - border towards Elvas "
                                             "(catalogue 5.3)"),
    ("ESLLFP", "ES04313", "EU00121", 20.5, "LFP Perthus, Límite Adif-LFPSA - border, "
                                           "1.05 x 19.5"),
]

# RINF lengths that cannot be right: section label -> (RINF's km, km used, why).
TYPO_KM = {
    "Section of Line MONTIJO - BIF. SAN NICOLAS": (333.0, 22.3, "21.0 km crow-fly; 1.06 x"),
    "Section of Line ALMASSORA - CASTELLO DE LA PLANA": (0.21, 4.4, "4.3 km crow-fly"),
    # Line 320's RINF sections sum to 163.1 km against the catalogue's 146.2; Hellín - Cieza
    # is 62.45 in RINF and 45.7 km of track, the same 16.7 km. Taken off its longest piece.
    "Section of Line KM. 378.0 - CIEZA": (32.06, 15.36, "line 320 is 16.7 km long in RINF"),
}

# RINF points and the OSM station they are (`stop_name`): a point named alike to a station of
# the other gauge nearer than its own takes the wrong one. Found by comparing the gauge of the
# track at RINF's coordinate with the track at the matched station (es_sources.md).
STOP_NAMES = {
    "CIAÑO": "Ciañu",                                   # the Iberian halt; "Ciaño" is FEVE's
    "ZORROTZA ZORROZGOITI": "Zorrotza Zorrotzgoiti",    # FEVE's; "Zorrotza" is Renfe's
    # Typed 40 (technical) in RINF, so line 270 ended at a junction called PORTBOU instead of
    # at the station every R11 and French TER calls at.
    "PORTBOU": "Portbou",
}
# Never stops: the high-speed technical points at l'Espluga and Campomanes ("A.V."), which took
# the conventional station of the town; La Sagrera, the unopened station 690 m from the OSM
# "La Sagrera" on the Meridiana tunnel; La Felguera on the Iberian line 140, whose only OSM
# station of that name is FEVE's (335 m). Each stays a section end at RINF's coordinate.
NOT_STOPS = {"LESPLUGA DE FRANCOLI-A. V.", "CAMPOMANES  AV", "LA SAGRERA", "LA FELGUERA",
             # No train calls (Renfe's feed, Oct 2026: Murcia > Balsicas-Mar Menor non-stop).
             # As a stop it left El Reguerón - Riquelme a junction-ended piece that only the
             # 41.5 km Murcia - Balsicas run crosses, "weak" by 1.5 km, and 22.6 km of the
             # Murcia - Cartagena line went undrawn.
             "RIQUELME-SUCINA"}


def es_stop_name(p):
    n = p.get("name")
    if n in NOT_STOPS:
        return False
    return STOP_NAMES.get(n)


# Points whose RINF coordinate is far from where their sections say: P.B. El Villar is 39.4 km
# crow-fly from Chinchilla aguja km 298,4, which its section puts 17.9 km away, and 4.8 km
# from Alpera, 21.5 km away. Left unplaced, so Chinchilla - Alpera is traced end to end.
NO_COORD = {"P.B. EL VILLAR"}

# High-speed points RINF places nearer the conventional line beside them, moved to the nearest
# point on the high-speed track (computed from the extract). Vilavella AV is 50 m from the
# Zamora - A Coruña line and 224 m from the high-speed line, so it snapped to the conventional
# track only and Pedralba - Vilavella - Miamán (90 km) was traced over it.
MOVE = {
    "VILAVELLA AV": (-7.046776, 42.046764),         # 224 m, way 971655316
    "MIAMAN": (-7.641187, 42.196704),               # 198 m, way 971676778
    "BIF. PEDRALBA (AVE)": (-6.634488, 42.046276),  # 180 m, way 954718589
}

# OSM route=railway relations with no ref that are one Adif line (`osm_rel`): read as that
# number, so the second trace pass prefers their ways. Without it the 982 gaps were traced over
# the conventional Zamora - A Coruña line beside the high-speed one (108.6 km near Lubián).
REL_REF = {"LAV Olmedo-Zamora-Galicia": "982", "LAV Variante de Pajares": "984",
           "Liña Zamora-A Coruña": "822"}


# Junctions where a branch leaves a main line at a point that is no stop (`cut_at_junctions`):
# the reader merges such a point away inside the main line (two neighbours on it), so the
# branch ends at a node nothing else touches and the timetable check finds no path onto it.
# Cut there, and only there: cutting at every junction (tried) split main lines into
# junction-ended pieces that a parallel line made "ambiguous" or long non-stop runs "weak",
# and dropped real track (Valladolid - Venta de Baños on 080, Albacete - Chinchilla on 300).
CUT_AT = {
    "CHINCHILLA MONTEAR.AG.KM.298,4",   # 320 to Hellín and Murcia leaves 300
    "BIF. PAJARES",                     # 984, the Pajares base tunnel, leaves 130 at La Robla
    "BIF. UTRERA",                      # 422 to Arahal - Osuna - Bobadilla leaves 400
    "BIF. CASA DE LA TORRE",            # 500 Cañaveral - Cáceres joins 026 near Cáceres
    "EL REGUERON AG KM 522,1",          # 352 to Cartagena leaves 336 (Murcia - Cartagena)
    "BIF. ANGUEIRA",                    # 818 Padrón - Vilagarcía leaves 824 (the Atlantic axis)
    "BIF. SAN AMARO",                   # 828 A Portela leaves 824 near Pontevedra
    "BIF. TERUEL",                      # 610 to Teruel leaves the Zaragoza lines at Cuarte
}


def es_osm_rel(tags):
    return (REL_REF.get(tags.get("name")) or osm_ref_default(tags.get("ref")), tags.get("name"))


def es_fix(secs, points):
    out = []
    for p in points.values():
        if p.get("name") in NO_COORD and "lon" in p:
            p.pop("lon")
            p.pop("lat")
            out.append(f"{p['name']}: RINF's coordinate dropped (inconsistent with its sections)")
        if p.get("name") in MOVE:
            p["lon"], p["lat"] = MOVE[p["name"]]
            out.append(f"{p['name']}: moved onto the high-speed track")
    for s in secs:
        t = TYPO_KM.get(s["label"])
        if t and s["km"] is not None and abs(s["km"] - t[0]) < 0.01:
            out.append(f"{s['base']} {s['label']}: {s['km']} km set to {t[1]} ({t[2]})")
            s["km"] = t[1]
    n0 = len(secs)
    secs[:] = [s for s in secs if s["label"] not in DROP_SECTIONS]
    out.append(f"{n0 - len(secs)} Taboadela gauge-changer links left out")
    by_uop = {p.get("uopid"): op for op, p in points.items()}
    for uop, (name, lon, lat) in NEW_STOPS.items():
        op = f"es:point:{uop}"
        points[op] = {"op": op, "uopid": uop, "name": name, "type": "10", "lon": lon, "lat": lat}
        by_uop[uop] = op
    for uop, (name, lon, lat) in BORDER.items():
        op = f"http://data.europa.eu/949/OperationalPoint_{uop}"
        if uop not in by_uop:
            points[op] = {"op": op, "uopid": uop, "name": name, "type": "90",
                          "lon": lon, "lat": lat}
            by_uop[uop] = op
    adds = [(f"ESL{num}GAP", a, b, km, why) for num, a, b, km, why in GAPS] + LINKS
    for lid, ua, ub, km, why in adds:
        a, b = by_uop.get(ua), by_uop.get(ub)
        if not a or not b:
            out.append(f"gap {lid} {ua} - {ub}: a point is missing, not added")
            continue
        secs.append({"sol": f"es:{lid}:{ua}-{ub}", "line": lid, "base": lid, "a": a, "b": b,
                     "km": km, "im": "0071_IM",
                     "label": f"{points[a].get('name')} - {points[b].get('name')} (added in es.py)"})
        out.append(f"added {lid}: {why}, {km} km")
    return out


ID = re.compile(r"ESL(\d{3})")


def es_num(lid):
    m = ID.match(lid or "")
    return m.group(1) if m else None


def es_skip(lid):
    n = es_num(lid)
    return bool(n) and 870 <= int(n) <= 899 and lid != "ESL893002600"


FGC = "Ferrocarrils de la Generalitat de Catalunya"


def es_im(sec):
    base = sec.get("base") or ""
    if base == "ESLLFP":
        return "LFP Perthus"
    if base == "ESL893002600":
        return FGC
    r = ROUTES.get(es_num(base))
    return "Adif AV" if r and r[3] else "Adif"


def es_id_name(lid, _uop):
    if lid == "ESLLFP":
        return ("Figueres – Perpignan (LFP)", "Figueres – Perpignan high-speed line (LFP)")
    return None


class EsName(str):
    """rinf.py writes a numbered line's name as COUNTRY["name"].format(ref=ref)."""
    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        r = ROUTES.get(ref)
        if r:
            return str.format(self, ref=ref, route=r[0])
        return str.format(self.split(" {route}")[0].split(" ({route})")[0], ref=ref)


COUNTRY = {
    "iso3": "ESP", "wikidata": "Q29", "langs": ["es"],
    "fix": es_fix,
    "skip_line": es_skip,
    "ref": es_num,
    "rule_certain": True,
    "fixed": {"ESL893002600": "206"},
    "no_ref": lambda lid: lid == "ESLLFP",
    "id_name": es_id_name,
    "stop_name": es_stop_name,
    "osm_rel": es_osm_rel,
    "im_of": es_im,
    "name": EsName("{ref} {route}"), "name_en": EsName("Line {ref} ({route})"),
    "im": {"0071_IM": "Adif"},
    "cut_at_junctions": CUT_AT,
}
