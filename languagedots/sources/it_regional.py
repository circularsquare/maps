"""Italy's local languages: which language a comune's "dialetto" is, and the minority figures.

Read by sources/it_istat.py. The record is sources/it.md. Written 2026-10-05 (edd42a8c-it).

THE SURVEY. ISTAT, "L'uso della lingua italiana, dei dialetti e delle lingue straniere", anno
2024 (indagine "I cittadini e il tempo libero", ~16,950 households, 6+), published 27 Jan 2026:
https://www.istat.it/wp-content/uploads/2026/01/Tavole_Report_lingue-e-dialetti.xlsx
Tavola 3 gives, by region (Bolzano and Trento apart), the language usually spoken IN THE FAMILY:
only or mainly Italian / only or mainly dialect / both Italian and dialect / another language.
The build reads it from data/raw/it/istat_lingue_2024_tavole.xlsx; nothing is retyped here.

"Dialetto" is not a language name: ISTAT's question leaves it to the respondent. Following
Glottolog (data/raw/glottolog) the Italo-Romance "dialects" are drawn as the languages they
are, decided by where the respondent lives (a label whose meaning depends on place, split by
geography, AGENT_BRIEF §3):
  Piedmontese (piem1238), Lombard (lomb1257), Ligurian (ligu1248), Emilian (emil1241),
  Romagnol (roma1328), Venetian (vene1258; Trentine and Triestine are Venetian dialects in
  Glottolog), Friulian (friu1240), Neapolitan (Glottolog "Continental Southern Italian",
  neap1235, which takes in Abruzzese, Molisan, Apulian, Lucanian and northern Calabrian),
  Sicilian (sici1248; with Salentino and southern Calabrian), Sardinian, Gallurese (gall1276),
  Sassarese (sass1235), Franco-Provencal (Arpitan, fran1260).
  Tuscany, Umbria, Lazio, the Aquila and central Marche: Glottolog files their dialects
  (Tuscan, Romanesco, Umbrian, central Marchigiano) under Italian itself (ital1282), so the
  "dialect" answers there are drawn as Italian.

THE MEASURE (as France): a "both Italian and dialect" answer counts half. See it_istat.py.
"""

# ---- the region's default dialect language, and provinces that differ ---------------------
# Provinces are named by their capoluogo comune (the build looks up its NUTS 3 unit).
REGION_DIALECT = {
    "Piemonte": "Piedmontese",
    "Valle d'Aosta/Vallée d'Aoste": "Franco-Provencal",
    "Liguria": "Ligurian",
    "Lombardia": "Lombard",
    "Trento": "Venetian",            # Glottolog tret1239 Trentine sits under Venetian
    "Veneto": "Venetian",
    "Friuli-Venezia Giulia": "Friulian",
    "Emilia-Romagna": "Emilian",
    "Toscana": "Italian",
    "Umbria": "Italian",
    "Marche": "Italian",
    "Lazio": "Italian",
    "Abruzzo": "Neapolitan",
    "Molise": "Neapolitan",
    "Campania": "Neapolitan",
    "Puglia": "Neapolitan",
    "Basilicata": "Neapolitan",
    "Calabria": "Sicilian",
    "Sicilia": "Sicilian",
    "Sardegna": "Sardinian",
}
PROVINCE_DIALECT = {
    "Novara": "Lombard", "Verbania": "Lombard",          # Novarese and Ossolano are Lombard
    "Mantova": "Emilian",                                 # Mantuan is Emilian
    "Ravenna": "Romagnol", "Forlì": "Romagnol", "Rimini": "Romagnol",
    "Pesaro": "Romagnol",                                 # Gallo-Picene, northern Marche
    "Fermo": "Neapolitan", "Ascoli Piceno": "Neapolitan", # southern Marchigiano
    "Frosinone": "Neapolitan",
    "L'Aquila": "Italian",                                # Aquilano is central Italian
    "Lecce": "Sicilian", "Brindisi": "Sicilian",          # Salentino
    "Cosenza": "Neapolitan",                              # northern Calabrian
    "Trieste": "Venetian",                                # Triestine
}

# ---- comuni whose local language differs from their province's -------------------------------
# Names as ISTAT spells them; the build stops on any it cannot find in the stated region.
COMUNE_LANG = {
    # southern Lazio (Gaeta, Formia, Fondi): Neapolitan-type dialects
    ("Lazio", "Neapolitan"): [
        "Gaeta", "Formia", "Minturno", "Fondi", "Itri", "Spigno Saturnia", "Castelforte",
        "Santi Cosma e Damiano", "Monte San Biagio", "Sperlonga", "Lenola", "Campodimele",
        "Ventotene", "Ponza"],
    # Friuli: the Venetian-speaking comuni (Bisiacco, Gorizia's Venetian, western Pordenone,
    # the lagoon). Everything else in Udine, Pordenone and Gorizia is drawn as Friulian.
    ("Friuli-Venezia Giulia", "Venetian"): [
        "Gorizia", "Monfalcone", "Grado", "Staranzano", "Ronchi dei Legionari",
        "Fogliano Redipuglia", "San Canzian d'Isonzo", "Turriaco", "San Pier d'Isonzo",
        "Pordenone", "Sacile", "Brugnera", "Caneva", "Fontanafredda", "Polcenigo",
        "Prata di Pordenone", "Pasiano di Pordenone", "Porcia", "Azzano Decimo", "Chions",
        "Pravisdomini", "Marano Lagunare", "Lignano Sabbiadoro"],
    # Sardinia: Gallura (Gallurese), the Sassari area (Sassarese), Alghero (Catalan),
    # Carloforte and Calasetta (Tabarchino, a Ligurian dialect)
    ("Sardegna", "Gallurese"): [
        "Olbia", "Tempio Pausania", "Arzachena", "Calangianus", "Luras", "Luogosanto",
        "Aglientu", "Santa Teresa Gallura", "Palau", "La Maddalena",
        "Sant'Antonio di Gallura", "Loiri Porto San Paolo", "Telti", "Golfo Aranci",
        "Bortigiadas", "Aggius", "Trinità d'Agultu e Vignola", "Badesi", "Viddalba"],
    ("Sardegna", "Sassarese"): ["Sassari", "Porto Torres", "Sorso", "Stintino", "Castelsardo"],
    ("Sardegna", "Catalan"): ["Alghero"],
    ("Sardegna", "Ligurian"): ["Carloforte", "Calasetta"],
    # Arbëreshë villages (Law 482/1999 communities that still speak it)
    ("Molise", "Arbereshe"): ["Campomarino", "Montecilfone", "Portocannone", "Ururi"],
    ("Puglia", "Arbereshe"): ["Casalvecchio di Puglia", "Chieuti", "San Marzano di San Giuseppe"],
    ("Basilicata", "Arbereshe"): ["Barile", "Ginestra", "Maschito", "San Costantino Albanese",
                                  "San Paolo Albanese"],
    ("Campania", "Arbereshe"): ["Greci"],
    ("Calabria", "Arbereshe"): [
        "Acquaformosa", "Castroregio", "Cerzeto", "Civita", "Falconara Albanese", "Firmo",
        "Frascineto", "Lungro", "Plataci", "San Basile", "San Benedetto Ullano",
        "San Cosmo Albanese", "San Demetrio Corone", "San Giorgio Albanese",
        "Santa Caterina Albanese", "Santa Sofia d'Epiro", "Spezzano Albanese",
        "Vaccarizzo Albanese", "San Martino di Finita", "Carfizzi", "Pallagorio",
        "San Nicola dell'Alto", "Andali", "Caraffa di Catanzaro", "Marcedusa"],
    ("Sicilia", "Arbereshe"): ["Piana degli Albanesi", "Contessa Entellina",
                               "Santa Cristina Gela"],
    # Molise Croatian (Slavomolisano)
    ("Molise", "Slavomolisano"): ["Acquaviva Collecroce", "Montemitro", "San Felice del Molise"],
    # Franco-Provencal outside the Aosta Valley: the Lanzo, Orco and Soana valleys and the
    # Cenischia; Faeto and Celle di San Vito in Apulia
    ("Piemonte", "Franco-Provencal"): [
        "Balme", "Ala di Stura", "Ceres", "Groscavallo", "Usseglio", "Lemie", "Chialamberto",
        "Cantoira", "Viù", "Mezzenile", "Pessinetto", "Traves", "Monastero di Lanzo",
        "Ceresole Reale", "Noasca", "Locana", "Ribordone", "Ronco Canavese",
        "Valprato Soana", "Ingria", "Novalesa", "Venaus", "Mompantero", "Giaglione",
        "Mattie"],
    ("Puglia", "Franco-Provencal"): ["Faeto", "Celle di San Vito"],
    # Occitan valleys of Piedmont (core valley comuni; the plains below speak Piedmontese)
    ("Piemonte", "Occitan"): [
        "Acceglio", "Argentera", "Bellino", "Canosio", "Casteldelfino", "Castelmagno", "Elva",
        "Entracque", "Frassino", "Limone Piemonte", "Macra", "Marmora", "Melle",
        "Pietraporzio", "Pontechianale", "Prazzo", "Sambuco", "Sampeyre", "Stroppo",
        "Valdieri", "Vinadio", "Vernante", "Demonte", "Aisone", "Roaschia",
        "Celle di Macra", "San Damiano Macra", "Oncino", "Crissolo", "Ostana",
        "Bobbio Pellice", "Villar Pellice", "Torre Pellice", "Angrogna", "Rorà", "Prali",
        "Perrero", "Pragelato", "Fenestrelle", "Usseaux", "Roure", "Salbertrand", "Oulx",
        "Cesana Torinese", "Claviere", "Sestriere", "Sauze d'Oulx", "Bardonecchia",
        "Exilles", "Chiomonte"],
}

# ---- minority languages with a figure of their own, placed on named comuni ------------------
# ISTAT 2024 tav. 14 gives the share of each region's 6+ who KNOW a language protected by Law
# 482/1999, and the report (p. 6) says 49.1% of those who know one use it always or often in
# the family. Knowledge x 49.1% is the family-language estimate. Used only where the region's
# "dialect" answer cannot stand for the language: Slovene in Friuli-Venezia Giulia (a Slavic
# language beside Friulian and Venetian) and Greek in Apulia and Calabria (Griko and Calabrian
# Greek, spoken in a handful of villages inside Salentino and Calabrian country).
# NOT used for Albanian (tav. 14's "Albanese" counts the 415,000 Albanian citizens too) or
# Sardinian/Friulian (the dialect answer covers them; see it_istat.py).
FAMILY_USE = 0.491
KNOW_FIXED = {
    # (region, label, tav14 column): placement comuni
    ("Friuli-Venezia Giulia", "Slovenian", "Sloveno o croato"): [
        "Trieste", "Duino Aurisina", "Sgonico", "Monrupino", "San Dorligo della Valle",
        "Muggia", "Gorizia", "Doberdò del Lago", "Savogna d'Isonzo", "San Floriano del Collio",
        "San Pietro al Natisone", "San Leonardo", "Savogna", "Stregna", "Grimacco",
        "Drenchia", "Pulfero", "Resia", "Taipana", "Lusevera", "Malborghetto Valbruna",
        "Tarvisio"],
    ("Puglia", "Griko", "Greco"): [
        "Calimera", "Carpignano Salentino", "Castrignano de' Greci", "Corigliano d'Otranto",
        "Martano", "Martignano", "Melpignano", "Soleto", "Sternatia", "Zollino"],
    ("Calabria", "Calabrian Greek", "Greco"): [
        "Bova", "Bova Marina", "Condofuri", "Roghudi", "Roccaforte del Greco"],
}

# ---- regions where "another language" holds a regional language, not only immigrants --------
# In these the immigrant languages are foreign citizens x the national retention share, and the
# rest of tav. 3's "altra lingua" is the regional language (Sardinian 11.4% "altra lingua"
# against 3% foreign citizens; Friuli 23.3% against 9.6%; Aosta 13.5% against 6.6%).
# Everywhere else "altra lingua" is taken as the immigrant languages themselves.
REGIONAL_IN_OTHER = {"Sardegna", "Friuli-Venezia Giulia", "Valle d'Aosta/Vallée d'Aoste"}

# ISTAT 2024 tav. 11: of people 6+ whose mother tongue is not Italian, 61.5% speak another
# language in the family (26.0% only or mainly Italian, 7.0% both Italian and dialect, 2.6%
# dialect). Applied to foreign citizens in REGIONAL_IN_OTHER and in South Tyrol and Trentino.
RETENTION = 0.615

# Children under 6 are outside the survey. Tav. 12: among 6-14s, dialect only 1.4% and both
# 16.6%, against 9.6% and 28.4% at all ages; the 6-14 ratio is applied to the under-6s.
YOUNG_RATIO = (0.014 + 0.166 / 2) / (0.096 + 0.284 / 2)
