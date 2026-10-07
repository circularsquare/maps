"""Spain: where inside each province each language's dots go (placement only, spec 8.2).

    python sources/es_place.py --fetch    download into data/raw/es/ whatever is missing
    python sources/es_place.py            write data/geo/es/es_weights.csv

ECEPOV gives each language's count per province (sources/es_ecepov.py). This file decides only
how a province's dots spread over its municipios, on religiondots' 8,131 GISCO LAU 2021
municipio polygons (religiondots/data/geo/es/es_municipios.gpkg, read-only), and never changes a
province's counts. Output: one weight per (municipio, node); countries/es.py's weighter reads it.

THE WEIGHTS, per municipio m (locals = Spanish nationals + nationals of Spanish-speaking
countries, Padron 1 Jan 2022):
  * an immigrant language: the municipio's residents of the nationalities that speak it
    (es2021.ORIGIN), from the Padron's municipal tables (INE "Poblacion por sexo, municipios y
    nacionalidad (principales nacionalidades)", 54 tables, about 30 named nationalities each).
    Nationalities a municipal table does not name sit in its continent remainder; that
    remainder is shared by the province's own mix of the unnamed ones (INE 03005).
  * a regional language L in a province where ECEPOV draws it: locals x g_m, where g_m is
    the share of locals with L as first language, f_m (below) scaled by one factor per province
    so that the province's sum matches ECEPOV's count of L. Spanish there: locals x (1 - g_m).
    f_m, the local evidence:
      Basque Country     Eustat, 2021 census-based "primera lengua" per municipio:
                         (Euskera + half of "las dos") / total. Measured, every municipio.
      Navarre            the Ley Foral del Vascuence's zones (2017 version): Basque speakers
                         are 62.1% of the Basque-speaking zone, 13.5% of the mixed zone and
                         2.7% of the rest (Nastat, 2021 census). Those shares are f.
      Catalonia          Idescat EULP 2023 first-language share of the area of the
                         territorial plan (eight areas, Barcelona city on its own, Aran):
                         (Catalan + half of "Catalan and Spanish") / total.
      Valencian provinces  the Valencian law's zones (Llei 4/1983, title V): f = 1 in the
                         Valencian-speaking zone, 0.05 in the Castilian-speaking one, whose
                         municipios the law lists.
      Huesca, Teruel     the Catalan-speaking municipios of the Franja (the 2001 draft
                         language law's list, the only official one): f = 1; elsewhere 0.03.
      Leon, Zamora       Galician: the Galician-speaking municipios of El Bierzo and As
                         Portelas (f = 1; Ponferrada 0.3, Galician in its western parishes
                         only); elsewhere 0.02.
      Aran               Aranese (Occitan) only in the Val d'Aran's nine municipios.
      everywhere else    f = 1 (Galician in Galicia, Catalan in the Balearics, Asturian,
                         Bizkaia's Galician, Tarifit in Melilla): no finer source.
  * Spanish elsewhere, and `other`: locals; `other` on all foreign residents.
A node with no weight anywhere in a province falls back to locals, then GISCO population.
"""

import json
import os
import re
import sys
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
sys.path.insert(0, ROOT)
RAW = os.path.join(ROOT, "data", "raw", "es")
NORM = os.path.join(ROOT, "data", "normalized", "es.csv")
OUT = os.path.join(ROOT, "data", "geo", "es", "es_weights.csv")
MUNI_NAT = os.path.join(RAW, "padron2022_muni_nationality.csv")
EMEX = os.path.join(RAW, "idescat_emex_com_mun.json")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

# INE Padron continuo (operation 188), table 33572: "Poblacion por sexo, municipios y
# nacionalidad (principales nacionalidades)", national, every municipio. The API refuses it whole
# ("restricciones de volumen"), so it is asked for one nationality at a time (about 25 s each).
PADRON_TABLES = [33572]
INE_API = "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/{t}?nult=1&tip=AM&tv=18:451"

# ---- zone lists, as the laws (or the draft law) print them; joined by name, asserted
NAVARRE_BASQUE_ZONE = """Abaurrea Alta, Abaurrea Baja, Alsasua, Anué, Araiz, Aranaz, Arano,
Araquil, Arbizu, Areso, Aria, Arive, Arruazu, Bacáicoa, Basaburúa Mayor, Baztán, Beinza-Labayen,
Bertizarana, Betelu, Burguete, Ciordia, Donamaría, Echalar, Echarri Aranaz, Elgorriaga, Erasun,
Ergoyena, Erro, Esteríbar, Ezcurra, Garayoa, Garralda, Goizueta, Huarte-Araquil, Imoz, Irañeta,
Ituren, Iturmendi, Lacunza, Lanz, Larráun, Leiza, Lesaca, Oiz, Olazagutía, Orbaiceta, Orbara,
Roncesvalles, Saldías, Santesteban, Sumbilla, Ulzama, Urdax, Urdiáin, Urroz de Santesteban,
Valcarlos, Vera de Bidasoa, Villanueva de Aézcoa, Yanci, Zubieta, Zugarramurdi, Lecumberri,
Irurzun, Atez"""
NAVARRE_MIXED_ZONE = """Abárzuza, Ansoáin, Aoiz, Arce, Barañáin, Burgui, Burlada, Ciriza,
Cendea de Cizur, Echarri, Echauri, Valle de Egüés, Ezcároz, Esparza de Salazar, Estella,
Ezcabarte, Garde, Goñi, Güesa, Guesálaz, Huarte, Isaba, Iza, Izalzu, Jaurrieta, Juslapeña,
Lezáun, Lizoáin, Ochagavía, Odieta, Oláibar, Olza, Ollo, Oronz, Oroz-Betelu, Pamplona,
Puente la Reina, Roncal, Salinas de Oro, Sarriés, Urzainqui, Uztárroz, Vidángoz, Vidaurreta,
Villava, Yerri, Zabalza, Berrioplano, Berriozar, Orcoyen, Zizur Mayor, Aranguren, Belascoáin,
Galar, Abáigar, Adiós, Aibar, Allín, Améscoa Baja, Ancín, Añorbe, Aranarache, Arellano, Artazu,
Bargota, Beriáin, Biurrun-Olcoz, Cabredo, Cirauqui, Dicastillo, Enériz, Eulate, Gallués,
Garínoain, Izagaondoa, Larraona, Leoz, Lerga, Lónguida, Mendigorría, Metauten, Mirafuentes,
Murieta, Nazar, Obanos, Olite, Oteiza, Pueyo, Sangüesa, Tafalla, Tiebas-Muruarte de Reta,
Tirapu, Unzué, Ujué, Urraúl Bajo, Urroz-Villa, Villatuerta, Zúñiga"""
VALENCIA_CASTILIAN_ZONE = {
    "03": """Albatera, Algorfa, Almoradí, Aspe, Benferri, Benejúzar, Benijófar, Bigastro,
Callosa de Segura, Catral, Cox, Daya Nueva, Daya Vieja, Dolores, Elda, Formentera del Segura,
Granja de Rocamora, Jacarilla, Monforte del Cid, Los Montesinos, Orihuela, Pilar de la Horadada,
Rafal, Redován, Rojales, Salinas, San Isidro, San Fulgencio, San Miguel de Salinas, Sax,
Torrevieja, Villena""",
    "12": """Algimia de Almonacid, Almedíjar, Altura, Arañuel, Argelita, Ayódar, Azuébar,
Barracas, Bejís, Benafer, Castellnovo, Castillo de Villamalefa, Caudiel, Cirat,
Cortes de Arenoso, Chóvar, Espadilla, Fanzara, Fuente la Reina, Fuentes de Ayódar, Gaibiel,
Geldo, Higueras, Jérica, Ludiente, Matet, Montán, Montanejos, Navajas, Olocau del Rey, Pavías,
Pina de Montalgrao, Puebla de Arenoso, Sacañet, Segorbe, Soneja, Sot de Ferrer, Teresa, Toga,
Torás, El Toro, Torralba del Pinar, Torrechiva, Vall de Almonacid, Vallat, Villahermosa del Río,
Villamalur, Villanueva de Viver, Viver, Zucaina""",
    "46": """Ademuz, Alborache, Alcublas, Alpuente, Andilla, Anna, Aras de los Olmos, Ayora,
Benagéber, Bicorp, Bolbaite, Bugarra, Buñol, Calles, Camporrobles, Casas Altas, Casas Bajas,
Castielfabib, Caudete de las Fuentes, Cofrentes, Cortes de Pallás, Chelva, Chella, Chera, Cheste,
Chiva, Chulilla, Domeño, Dos Aguas, Enguera, Fuenterrobles, Gátova, Gestalgar, Godelleta,
Higueruelas, Jalance, Jarafuel, Loriguilla, Losa del Obispo, Macastre, Marines, Millares,
Navarrés, Pedralba, Puebla de San Miguel, Quesa, Requena, Siete Aguas, Sinarcas, Sot de Chera,
Teresa de Cofrentes, Titaguas, Torrebaja, Tous, Tuéjar, Utiel, Vallanca, Venta del Moro,
Villar del Arzobispo, Villargordo del Cabriel, Yátova, La Yesa, Zarra""",
}
FRANJA = {
    "22": """Albelda, Alcampell, Altorricón, Arén, Azanuy-Alins, Baells, Baldellou, Benabarre,
Bonansa, Camporrells, Castigaleu, Castillonroy, Estopiñán del Castillo, Fraga, Isábena,
Lascuarre, Laspaúles, Monesma y Cajigar, Montanuy, Peralta de Calasanz, Puente de Montañana,
San Esteban de Litera, Sopeira, Tamarite de Litera, Tolva, Torre la Ribera, Torrente de Cinca,
Velilla de Cinca, Vencillón, Veracruz, Viacamp y Litera, Zaidín""",
    "44": """Aguaviva, Arens de Lledó, Beceite, Belmonte de San José, Calaceite,
La Cañada de Verich, Cerollera, La Codoñera, Cretas, Fórnoles, La Fresneda, Fuentespalda,
La Ginebrosa, Lledó, Mazaleón, Monroyo, Peñarroya de Tastavins, La Portellada, Ráfales,
Torre de Arcas, Torre del Compte, Torrevelilla, Valderrobres, Valdetormo, Valjunquera""",
}
GALICIAN_CYL = {
    "24": """Puente de Domingo Flórez, Benuza, Carucedo, Borrenes, Priaranza del Bierzo,
Ponferrada, Carracedelo, Toral de los Vados, Camponaraya, Arganza, Cacabelos, Sobrado,
Corullón, Oencia, Barjas, Vega de Valcarce, Trabadelo, Balboa, Villafranca del Bierzo,
Vega de Espinareda, Fabero, Candín, Peranzanes""",
    "49": """Porto, Lubián, Hermisende, Pías""",
}
F_OUTSIDE = {"valencia": 0.05, "franja": 0.03, "galician_cyl": 0.02}
F_PONFERRADA = 0.3
NAVARRE_F = {"basque": 0.621, "mixed": 0.135, "rest": 0.027}  # Nastat, 2021 census

# Idescat comarca code -> area of the territorial plan (EULP 2023's AT01-AT08). Penedes is the
# 2017 area (Alt Penedes, Anoia, Baix Penedes, Garraf); the area 15+ populations are checked.
COMARCA_AT = {
    "13": "AT01", "11": "AT01", "21": "AT01", "40": "AT01", "41": "AT01",           # Metropolita
    "02": "AT02", "10": "AT02", "19": "AT02", "20": "AT02", "28": "AT02", "31": "AT02",
    "34": "AT02",                                                                      # Gironines
    "01": "AT03", "08": "AT03", "16": "AT03", "29": "AT03", "36": "AT03",           # Camp
    "09": "AT04", "22": "AT04", "30": "AT04", "37": "AT04",                         # Ebre
    "18": "AT05", "23": "AT05", "27": "AT05", "32": "AT05", "33": "AT05", "38": "AT05",  # Ponent
    "07": "AT06", "14": "AT06", "24": "AT06", "35": "AT06", "42": "AT06", "43": "AT06",  # Centrals
    "04": "AT07", "05": "AT07", "15": "AT07", "25": "AT07", "26": "AT07", "39": "AT07",  # Pirineu
    "03": "AT08", "06": "AT08", "12": "AT08", "17": "AT08",                         # Penedes
}


def _get(url, dest):
    if os.path.exists(dest):
        return json.load(open(dest, encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=600).read()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "wb").write(raw)
    print(f"  fetched {os.path.relpath(dest, ROOT)} ({len(raw):,} bytes)")
    return json.loads(raw)


def fetch():
    _get("https://api.idescat.cat/emex/v1/nodes.json?tipus=com,mun&lang=es", EMEX)
    if os.path.exists(MUNI_NAT):
        return
    import time
    cache = os.path.join(RAW, "padron2022")
    os.makedirs(cache, exist_ok=True)

    def api(url):
        # curl, not urllib: urllib took ten minutes over a 4.6 MB answer curl reads in 25 s
        import subprocess
        for attempt in range(6):
            try:
                raw = subprocess.run(["curl", "-s", "-m", "300", "-A", UA, url],
                                     capture_output=True, check=True).stdout
                return json.loads(raw)
            except Exception as e:  # noqa: BLE001  INE's API drops the odd request
                print(f"    retry {attempt + 1}: {e}", flush=True)
                if attempt == 5:
                    raise
                time.sleep(15)

    def rows_of(t, d):
        out = []
        for s in d:
            md = {m["T3_Variable"]: m for m in s["MetaData"]}
            mun, nat = md.get("Municipios"), md.get("Nacionalidad")
            if mun is None or nat is None or not s["Data"] or s["Data"][0]["Anyo"] != 2022:
                continue
            out.append((t, mun["Codigo"], mun["Nombre"], nat["Nombre"], s["Data"][0]["Valor"]))
        return out

    for t in PADRON_TABLES:
        dest = os.path.join(cache, f"{t}.csv")
        if os.path.exists(dest):
            continue
        d = api(INE_API.format(t=t))
        if isinstance(d, dict):
            # "No puede mostrarse por restricciones de volumen": the big provinces' tables
            # are refused whole, so ask for them one nationality at a time.
            groups = api(f"https://servicios.ine.es/wstempus/js/ES/GRUPOS_TABLA/{t}")
            g = [x for x in groups if x["Nombre"].startswith("Nacionalidad")][0]
            vals = api(f"https://servicios.ine.es/wstempus/js/ES/VALORES_GRUPOSTABLA/{t}/{g['Id']}")
            rows = []
            for v in vals:
                part_csv = os.path.join(cache, f"{t}_{v['Id']}.csv")
                if not os.path.exists(part_csv):
                    part = api(INE_API.format(t=t) + f"&tv={v['FK_Variable']}:{v['Id']}")
                    assert isinstance(part, list), (t, v["Nombre"], part)
                    pd.DataFrame(rows_of(t, part), columns=["table", "muni", "name",
                                                            "nationality", "count"]).to_csv(
                        part_csv, index=False)
                    print(f"  {t} {v['Nombre']}", flush=True)
                rows += list(pd.read_csv(part_csv, dtype={"muni": str}).itertuples(
                    index=False, name=None))
        else:
            rows = rows_of(t, d)
        pd.DataFrame(rows, columns=["table", "muni", "name", "nationality", "count"]).to_csv(
            dest, index=False)
        print(f"  table {t}: {len(rows):,} rows", flush=True)
    df = pd.concat([pd.read_csv(os.path.join(cache, f"{t}.csv"), dtype={"muni": str})
                    for t in PADRON_TABLES], ignore_index=True)
    df.to_csv(MUNI_NAT, index=False)
    print(f"  wrote {os.path.relpath(MUNI_NAT, ROOT)}: {len(df):,} rows, "
          f"{df['muni'].nunique():,} municipios")


# ---------------------------------------------------------------- build

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


ARTICLES = r"(La|El|Los|Las|L'|Els|Les|A|O|As|Os|Es)"


def name_keys(name):
    """INE prints co-official names as "Altsasu/Alsasua" and articles as "Yesa, La"."""
    keys = set()
    for part in str(name).split("/"):
        part = part.strip()
        m = re.match(r"^(.*), " + ARTICLES + r"$", part)
        if m:
            part = f"{m.group(2)} {m.group(1)}"
        keys.add(norm(part))
        keys.add(norm(re.sub(r"^" + ARTICLES + r" ", "", part)))
    return keys


# names the lists spell differently from INE (each checked against the Padron's own name)
ALIASES = {
    ("31", "Araiz"): "Araitz",
    ("31", "Aranaz"): "Arantza",
    ("31", "Araquil"): "Arakil",
    ("31", "Arive"): "Aribe",
    ("31", "Bacáicoa"): "Bakaiku",
    ("31", "Basaburúa Mayor"): "Basaburua",
    ("31", "Beinza-Labayen"): "Beintza-Labaien",
    ("31", "Ciordia"): "Ziordia",
    ("31", "Echalar"): "Etxalar",
    ("31", "Echarri Aranaz"): "Etxarri Aranatz",
    ("31", "Erasun"): "Eratsun",
    ("31", "Ergoyena"): "Ergoiena",
    ("31", "Ezcurra"): "Ezkurra",
    ("31", "Garayoa"): "Garaioa",
    ("31", "Huarte-Araquil"): "Uharte Arakil",
    ("31", "Imoz"): "Imotz",
    ("31", "Lacunza"): "Lakuntza",
    ("31", "Lanz"): "Lantz",
    ("31", "Leiza"): "Leitza",
    ("31", "Lesaca"): "Lesaka",
    ("31", "Orbaiceta"): "Orbaizeta",
    ("31", "Sumbilla"): "Sunbilla",
    ("31", "Ulzama"): "Ultzama",
    ("31", "Urroz de Santesteban"): "Urroz",
    ("31", "Vera de Bidasoa"): "Bera",
    ("31", "Villanueva de Aézcoa"): "Villanueva de Aezkoa",
    ("31", "Yanci"): "Igantzi",
    ("31", "Lecumberri"): "Lekunberri",
    ("31", "Irurzun"): "Irurtzun",
    ("31", "Atez"): "Atetz",
    ("31", "Cendea de Cizur"): "Cizur",
    ("31", "Echauri"): "Etxauri",
    ("31", "Goñi"): "Val de Goñi",
    ("31", "Lizoáin"): "Lizoain-Arriasgoiti",
    ("31", "Olza"): "Cendea de Olza",
    ("31", "Ollo"): "Valle de Ollo",
    ("31", "Vidaurreta"): "Bidaurreta",
    ("31", "Yerri"): "Valle de Yerri",
    ("31", "Orcoyen"): "Orkoien",
    ("31", "Estella"): "Estella-Lizarra",
    ("31", "Lezáun"): "Lezaun",
    ("31", "Oláibar"): "Olaibar",
    ("31", "Mendigorría"): "Mendigorria",
    ("31", "Urraúl Bajo"): "Urraul Bajo",
    ("22", "Veracruz"): "Beranuy",  # renamed 2007
    ("44", "Valdetormo"): "Valdeltormo",
    ("24", "Candín"): "Valle de Ancares",  # renamed 2020
}


def join_list(text, prov, muni_names):
    """List of names -> set of 5-digit codes in that province. Every name must match once."""
    want = [w.strip() for w in re.split(r",\s*", text.replace("\n", " ")) if w.strip()]
    index = {}
    for code, name in muni_names.items():
        if code[:2] != prov:
            continue
        for k in name_keys(name):
            index.setdefault(k, set()).add(code)
    out, missing = set(), []
    for w in want:
        k = norm(ALIASES.get((prov, w), w))
        hits = index.get(k) or index.get(norm(re.sub(r"^" + ARTICLES + r" ", "", w)))
        if not hits or len(hits) != 1:
            missing.append(w)
            continue
        out |= hits
    if missing:
        raise SystemExit(f"es_place: {prov}: no single INE municipio for {missing}")
    return out


def origin_continents():
    """03005's country rows -> continent, from the order its headers print them in."""
    from es_ecepov import read_px
    s = read_px(os.path.join(RAW, "ine_03005.px"))
    labels = list(dict.fromkeys(s.index.get_level_values("Nacionalidad")))
    heads = {"EUROPA": "EU", "ÁFRICA": "AF", "AMÉRICA": "AM", "ASIA": "AS", "OCEANÍA": "OC",
             "APÁTRIDAS": "OC"}
    cont, out = None, {}
    for lab in labels:
        if lab in heads:
            cont = heads[lab]
        elif not (lab.isupper() or lab.startswith("UE(")):
            out[lab] = cont
    return out


def eulp(fn):
    e = json.load(open(os.path.join(RAW, fn), encoding="utf-8"))
    geo_d = [d for d in e["id"] if d in ("AT", "MUN", "COM", "CAT")][0]
    g = list(e["dimension"][geo_d]["category"]["index"])
    L = list(e["dimension"]["LAN_ISO"]["category"]["index"])
    v = np.array([np.nan if x is None else x for x in e["value"]], dtype=float)
    v = np.nan_to_num(v.reshape(len(g), len(L)))
    return {gg: dict(zip(L, r)) for gg, r in zip(g, v)}


def ca_share(d):
    return (d.get("CA", 0) + d.get("CA_ES", 0) / 2) / d["TOTAL"]


def build():
    import geopandas as gpd
    import es2021
    from es_ecepov import padron_nationality
    from rdlink import RD_GEO

    place = gpd.read_file(RD_GEO / "es" / "es_municipios.gpkg")
    place = pd.DataFrame(place.drop(columns="geometry"))
    place["muni"] = place["muni"].astype(str).str.zfill(5)
    assert place["muni"].is_unique and len(place) == 8131

    mn = pd.read_csv(MUNI_NAT, dtype={"muni": str})
    mn = mn[mn["muni"].str.len() == 5]
    names = mn.drop_duplicates("muni").set_index("muni")["name"].to_dict()
    wide = mn.pivot_table(index="muni", columns="nationality", values="count", aggfunc="first")
    wide = wide.fillna(0.0)
    print(f"Padron 2022 municipal nationality: {len(wide):,} municipios, "
          f"{wide.shape[1]} columns")
    miss = sorted(set(place["muni"]) - set(wide.index))
    extra = sorted(set(wide.index) - set(place["muni"]))
    print(f"  GISCO municipios without a Padron row: {len(miss)} {miss[:10]}; "
          f"Padron rows without a polygon: {len(extra)} {extra[:10]}")

    # ---- every 03005 nationality per municipio: named ones as printed, the rest from the
    #      municipio's continent remainder, shared by the province's mix of the unnamed ones
    cont_of = origin_continents()
    cont_col = {"EU": "Europa (sin España)", "AF": "De Africa", "AM": "De América",
                "AS": "De Asia", "OC": "Oceanía y Apátridas"}
    for c in cont_col.values():
        assert c in wide.columns, (c, list(wide.columns))
    countries = [c for c in wide.columns if c in es2021.ORIGIN]
    assert set(countries) <= set(cont_of), set(countries) - set(cont_of)
    pn = padron_nationality()
    pn = pn[pn["nat"].isin(cont_of)]
    est = []
    for prov, g in wide.groupby(wide.index.str[:2]):
        pmix = pn[pn["prov"] == prov].set_index("nat")["n"]
        e = g[countries].copy()
        for K, col in cont_col.items():
            inK = [c for c in countries if cont_of.get(c) == K]
            rem = (g[col] - g[inK].sum(axis=1)).clip(lower=0)
            unnamed = pmix[[n for n in pmix.index if cont_of[n] == K and n not in countries
                            and n in es2021.ORIGIN]]
            unnamed = unnamed[unnamed > 0]
            if unnamed.sum() > 0:
                for n, sh in (unnamed / unnamed.sum()).items():
                    e[n] = (e[n] if n in e.columns else 0.0) + rem * sh
        est.append(e)
    F = pd.concat(est).fillna(0.0)
    tot_f = F.sum(axis=1).sum()
    print(f"  foreign residents placed by nationality: {tot_f:,.0f} of "
          f"{wide['Extranjera'].sum():,.0f} (the rest are 'Resto de ...' and stateless)")

    from es_ecepov import es_mix   # the shared origin table, as es.csv drew it
    mixes = {n: es_mix(n) for n in es2021.ORIGIN}
    # Spanish-speaking nationalities: Spanish the majority of their origin mix
    hisp = [n for n, v in mixes.items() if dict(v).get(es2021.SPANISH, 0) >= 0.5]
    locals_ = wide["Española"] + F[[c for c in F.columns if c in hisp]].sum(axis=1)
    foreign = wide["Extranjera"]

    # ---- the nodes each province draws
    es = pd.read_csv(NORM, dtype={"geo_id": str})
    es = es[es["geo_level"] == "province"]
    es["node"] = es["source_category"].map(es2021.resolve)
    drawn = es.groupby(["geo_id", "node"])["count"].sum()

    W = {}
    for node in sorted(set(drawn.index.get_level_values("node"))):
        if node == es2021.SPANISH:
            continue
        cols = [(c, sh) for c, v in mixes.items() for n, sh in v
                if n == node and c in F.columns]
        if cols:
            W[node] = sum(F[c] * sh for c, sh in cols)
    W["other"] = foreign.copy()

    # ---- regional languages: f per municipio
    f = {}
    ez = json.load(open(os.path.join(RAW, "eustat_lm01_2021.json"), encoding="utf-8"))
    geo = [d for d in ez["id"] if "mbito" in d][0]
    lang = [d for d in ez["id"] if d.startswith("lengua")][0]
    gi = list(ez["dimension"][geo]["category"]["index"])
    li = list(ez["dimension"][lang]["category"]["index"])
    vals = np.array(ez["value"], dtype=float).reshape(len(gi), len(li))
    eus = {}
    for code, r in zip(gi, vals):
        if len(code) == 5 and code[:2] in ("01", "20", "48") and r[li.index("10")] > 0:
            eus[code] = (r[li.index("20")] + r[li.index("40")] / 2) / r[li.index("10")]
    pv = [m for m in wide.index if m[:2] in ("01", "20", "48")]
    print(f"Eustat 2021 first language: {len(eus)} municipios, Padron has {len(pv)}")
    assert not set(pv) - set(eus), sorted(set(pv) - set(eus))
    nav_b = join_list(NAVARRE_BASQUE_ZONE, "31", names)
    nav_m = join_list(NAVARRE_MIXED_ZONE, "31", names)
    assert not nav_b & nav_m and len(nav_b) == 64 and len(nav_m) == 98, (len(nav_b), len(nav_m))
    nav = {m: NAVARRE_F["basque"] if m in nav_b else NAVARRE_F["mixed"] if m in nav_m
           else NAVARRE_F["rest"] for m in wide.index if m[:2] == "31"}
    f[es2021.BASQUE] = pd.Series({**eus, **nav})

    # Catalonia: EULP 2023 area shares; Barcelona city and Aran on their own
    at = eulp("eulp2023_at.json")
    bcn = eulp("eulp2023_mun.json")["080193"]
    aran = eulp("eulp2023_com.json")["39"]
    emex = json.load(open(EMEX, encoding="utf-8"))["fitxes"]["v"]
    cat, at_pop, occ = {}, {}, {}
    for com in emex:
        a = COMARCA_AT[com["id"]]
        for m in com["v"]:
            code = m["id"][:5]
            sh = ca_share(at[a])
            if code == "08019":
                sh = ca_share(bcn)
            elif com["id"] == "39":
                sh = ca_share(aran)  # Aran's own figure: Catalan 1.3k of 9.2k aged 15+
                occ[code] = 1.0
            cat[code] = sh
            at_pop[a] = at_pop.get(a, 0) + wide["Total"].get(code, 0)
    catm = [m for m in wide.index if m[:2] in ("08", "17", "25", "43")]
    assert not set(catm) - set(cat), sorted(set(catm) - set(cat))[:10]
    for a, p in sorted(at_pop.items()):
        ratio = at[a]["TOTAL"] * 1000 / p
        print(f"  EULP area {a}: 15+ {at[a]['TOTAL'] * 1000:,.0f} / Padron {p:,.0f} = "
              f"{ratio:.3f}, Catalan first language {100 * ca_share(at[a]):.1f}%")
        assert 0.78 < ratio < 0.92, f"{a}: comarca -> area mapping looks wrong"
    fr = {}
    for prov, text in FRANJA.items():
        z = join_list(text, prov, names)
        for m in wide.index[wide.index.str[:2] == prov]:
            fr[m] = 1.0 if m in z else F_OUTSIDE["franja"]
    f[es2021.CATALAN] = pd.Series({**cat, **fr})
    f[es2021.OCCITAN] = pd.Series({m: occ.get(m, 0.0) for m in wide.index if m[:2] == "25"})

    val = {}
    for prov, text in VALENCIA_CASTILIAN_ZONE.items():
        cz = join_list(text, prov, names)
        for m in wide.index[wide.index.str[:2] == prov]:
            val[m] = F_OUTSIDE["valencia"] if m in cz else 1.0
    f[es2021.VALENCIAN] = pd.Series(val)

    gl = {}
    for prov, text in GALICIAN_CYL.items():
        z = join_list(text, prov, names)
        for m in wide.index[wide.index.str[:2] == prov]:
            gl[m] = 1.0 if m in z else F_OUTSIDE["galician_cyl"]
    assert names["24115"].startswith("Ponferrada"), names["24115"]
    gl["24115"] = F_PONFERRADA
    f[es2021.GALICIAN] = pd.Series(gl)

    regional = [es2021.CATALAN, es2021.VALENCIAN, es2021.GALICIAN, es2021.BASQUE,
                es2021.ASTURIAN, es2021.TARIFIT, es2021.OCCITAN]
    gsum = pd.Series(0.0, index=wide.index)
    for node in regional:
        provs = sorted({p for p, n in drawn.index if n == node})
        gm = pd.Series(0.0, index=wide.index)
        for prov in provs:
            idx = wide.index[wide.index.str[:2] == prov]
            fm = f.get(node, pd.Series(dtype=float)).reindex(idx).fillna(1.0)
            loc = locals_.reindex(idx)
            k = drawn[(prov, node)] / (loc * fm).sum()
            gm[idx] = (k * fm).clip(upper=0.95)
            got = (loc * gm[idx]).sum() / drawn[(prov, node)]
            if abs(got - 1) > 0.02:
                print(f"  note: {node} in {prov}: capped at 95% of locals in places; the "
                      f"weights carry {100 * got:.0f}% of the count (placement only)")
        W[node] = locals_ * gm
        gsum += gm
    over = gsum[gsum > 0.95]
    print(f"  {len(over)} municipios where regional shares exceed 95%; Spanish floored at 5%")
    W[es2021.SPANISH] = locals_ * (1 - gsum).clip(lower=0.05)

    rows = []
    for node, w in W.items():
        w = w[w > 0]
        rows.append(pd.DataFrame({"muni": w.index, "node": node, "weight": w.values}))
    out = pd.concat(rows, ignore_index=True)
    out["unit"] = out["muni"].str[:2]
    out = out[out["muni"].isin(set(place["muni"]))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[["unit", "muni", "node", "weight"]].to_csv(OUT, index=False, float_format="%.3f")
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(out):,} (municipio, node) weights, "
          f"{out['node'].nunique()} nodes")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    build()
