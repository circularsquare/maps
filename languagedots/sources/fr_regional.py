"""France's regional and overseas languages: one entry per survey, with its citation.

Read by sources/fr_build.py (counts) and sources/fr_geo.py (the Basque zones' placement). The
record is sources/fr.md §3. Surveys were read 2026-10-05 (session edd42a8c-fr).

THE MEASURE. The map draws first or home languages. Where a survey publishes the language
people learned first or spoke at home as children, that is used; a "both" answer (the regional
language and French together) counts half. Where a survey publishes only ability ("speaks very
or fairly well"), ability is multiplied by the survey's own share of speakers who learned the
language from their parents, which is the nearest thing to a first language it measures. Every
row is `derived`.

THE AGES. Each survey asked adults (15+, 16+ or 18+). Its share is applied to the département's
population at those ages (RP 2023, single years). Children below the survey's age are drawn at
the survey's own youngest adult band, as a ratio to its all-adult figure, where it publishes
one (`young_ratio`); otherwise they are left on French. Overseas, the creole surveys' figure for
the language parents passed to their children is used for children (`young_share`).

Shares are fractions. `deps` maps département -> share, or the zone name for Basque.
"""

# ---- Basque: Office public de la langue basque (EEP), VII Inkesta Soziolinguistikoa 2021 -----
# https://www.mintzaira.fr/fileadmin/documents/Enquete_sociolinguistique/Emaitzen_laburpena-eus.pdf
# Universe: Iparralde 16+, 255,940. First language (received from parents up to age 3):
#   BAB (Bayonne, Anglet, Biarritz)    Basque 5.1%,  Basque+French 3.3%
#   Lapurdi outside BAB                Basque 13.6%, Basque+French 7.1%
#   Behe Nafarroa + Zuberoa            Basque 38.3%, Basque+French 10.7%
#   Iparralde                          Basque 13.3%, Basque+French 6.0%
#   16-24                              Basque 3.7%,  Basque+French 11.1%
BASQUE_ZONES = {
    "bab": 0.051 + 0.033 / 2,
    "lapurdi": 0.136 + 0.071 / 2,
    "bn_zuberoa": 0.383 + 0.107 / 2,
}
BASQUE_YOUNG_RATIO = (0.037 + 0.111 / 2) / (0.133 + 0.060 / 2)
BAB = {"64102", "64024", "64122"}           # Bayonne, Anglet, Biarritz (INSEE codes)
# Labourd's 41 communes (fr.wikipedia "Liste des communes du Labourd", read 2026-10-05), by
# name; fr_geo.py matches them to INSEE codes inside the Communaute d'agglomeration du Pays
# Basque (EPCI 200067106) and stops on any name it cannot find there.
LAPURDI = [
    "Ahetze", "Ainhoa", "Anglet", "Arbonne", "Arcangues", "Ascain", "Bardos", "Bassussarry",
    "Bayonne", "Biarritz", "Bidart", "Biriatou", "Bonloc", "Boucau", "Briscous",
    "Cambo-les-Bains", "Ciboure", "Espelette", "Guéthary", "Guiche", "Halsou", "Hasparren",
    "Hendaye", "Itxassou", "Jatxou", "Lahonce", "Larressore", "Louhossoa", "Macaye",
    "Mendionde", "Mouguerre", "Saint-Jean-de-Luz", "Saint-Pée-sur-Nivelle",
    "Saint-Pierre-d'Irube", "Sare", "Souraïde", "Urcuit", "Urrugne", "Urt", "Ustaritz",
    "Villefranque",
]
CAPB_EPCI = "200067106"
BASQUE_BASE_16 = 255_940

SURVEYS = [
    dict(
        lang="Basque", deps={"64": "basque_zones"}, age=16,
        young_ratio=BASQUE_YOUNG_RATIO,
        measure="first language (from parents up to age 3), Basque alone plus half of Basque "
                "with French, by the survey's three zones",
        cite="EEP, VII Inkesta Soziolinguistikoa 2021 (Iparralde)",
    ),
    # ---- Catalan: Generalitat de Catalunya, EULCN 2015 -----------------------------------------
    # https://llengua.gencat.cat/web/.content/documents/dadesestudis/altres/arxius/EULCN_2015_principals_resultats.pdf
    # 15+ in private households, 369,590 (Pyrenees-Orientales less the Fenouilledes). Llengua
    # inicial: Catalan 9.2%, Catalan+French 3.5%. No age band for first language.
    dict(
        lang="Catalan", deps={"66": 0.092 + 0.035 / 2}, age=15, young_ratio=None,
        measure="first language (llengua inicial), Catalan alone plus half of Catalan with "
                "French; applied to the whole département, Fenouilledes included",
        cite="Generalitat de Catalunya, Enquesta d'usos lingüístics a la Catalunya Nord 2015",
    ),
    # ---- Corsican: Collectivite de Corse, enquete sociolinguistique 2021 ----------------------
    # https://www.isula.corsica/assemblea/docs/rapports/2022O2303-annexe.pdf (table 14), 18+.
    # Language spoken at home up to age 6: Corsican only 15.6%, French+Corsican 40.2%,
    # French+Corsican+foreign 0.9%. Youngest band: 2012 OpinionWay (same annex), main language
    # up to 6 among 18-24, Corsican 2%, used directly for under-18s (young_share).
    dict(
        lang="Corsican", deps={"2A": 0.156 + 0.402 / 2 + 0.009 / 3,
                               "2B": 0.156 + 0.402 / 2 + 0.009 / 3},
        age=18, young_share=0.02,
        measure="language spoken at home up to age 6, Corsican alone plus half of French with "
                "Corsican; under-18s at the 2% of 18-24s whose main childhood language was "
                "Corsican (2012)",
        cite="Collectivite de Corse, enquete sociolinguistique 2021 (Assemblee de Corse report "
             "2022/O2/303, annex)",
    ),
    # ---- Breton and Gallo: Region Bretagne / TMO Regions 2024 ---------------------------------
    # https://brezhoweb.bzh/article/produit/photo/dossier4629/8460_region_bretagne_enque%CC%82te_sociolinguistique_r2001_1.pdf
    # 15+, 4,011,805 in the five départements. Speak very or fairly well, by département:
    # Finistere 6.0, Cotes-d'Armor 3.9, Morbihan 3.4, Ille-et-Vilaine 1.3, Loire-Atlantique <1
    # (taken as 0.5). Of speakers, 44% learned it mainly with their parents. 15-24: 1.5% against
    # 2.7% of all 15+.
    # Gallo: Cotes-d'Armor 7.9, Ille-et-Vilaine 5.1 (no other département printed); 51% of
    # speakers learned it with their parents. No age band.
    dict(
        lang="Breton",
        deps={"29": 0.060 * 0.44, "22": 0.039 * 0.44, "56": 0.034 * 0.44,
              "35": 0.013 * 0.44, "44": 0.005 * 0.44},
        age=15, young_ratio=0.015 / 0.027,
        measure="speaks very or fairly well x the 44% of speakers who learned it mainly with "
                "their parents",
        cite="Region Bretagne / TMO Regions, enquete sociolinguistique 2024",
    ),
    dict(
        lang="Gallo", deps={"22": 0.079 * 0.51, "35": 0.051 * 0.51}, age=15, young_ratio=None,
        measure="speaks very or fairly well x the 51% of speakers who learned it with their "
                "parents; only the two départements the report prints",
        cite="Region Bretagne / TMO Regions, enquete sociolinguistique 2024",
    ),
    # ---- Alsatian: OLCA / EDinstitut 2012 by département, CeA 2022 for the age gradient ----------
    # https://www.olcalsace.org/sites/default/files/documents/etude_linguistique_olca_edinstitut.pdf
    # 18+: "sait bien parler" Bas-Rhin 46%, Haut-Rhin 38%; 89% of speakers learned it from their
    # parents. CeA 2022 (via https://www.olcalsace.org/fr/observer-et-veiller/le-dialecte-en-chiffres):
    # 46% of 18+ speak it, 18-24 9%.
    dict(
        lang="Alsatian", deps={"67": 0.46 * 0.89, "68": 0.38 * 0.89}, age=18,
        young_ratio=0.09 / 0.46,
        measure="speaks it well (2012) x the 89% of speakers who learned it from their parents",
        cite="OLCA / EDinstitut 2012; age gradient from the Collectivite europeenne d'Alsace "
             "survey 2022",
    ),
    # ---- Lorraine Franconian: DRAC Grand Est / TMO 2024 -----------------------------------------
    # https://www.culture.gouv.fr/mc/content/download/362210/file/DRAC%20Grand%20Est%20-%20Resultats%20enquete%20sociolinguistique%202024.pdf
    # 4.3% of Grand Est 18+ speak Francique very or fairly well, 190,000; 73% of those who
    # understand it live in Moselle; 49% of them learned it with their parents. Drawn in Moselle
    # only: 190,000 x 0.73 x 0.49 = 67,963 adults, as a share of Moselle's 18+ (set by the build).
    dict(
        lang="Lorraine Franconian", deps={"57": ("count", 190_000 * 0.73 * 0.49)}, age=18,
        young_ratio=None,
        measure="190,000 adult speakers x the 73% in Moselle x the 49% who learned it with "
                "their parents",
        cite="DRAC Grand Est / TMO, enquete sociolinguistique 2024",
    ),
    # ---- Occitan: OPLO 2020 (Nouvelle-Aquitaine + Occitanie), Midi-Pyrenees 2010 for family -----
    # https://www.ofici-occitan.eu/wp-content/uploads/2020/08/Synthese-4-pages-sans-traits-de-coupe.pdf
    # 7% speak Occitan without difficulty or enough for a simple conversation, "nearly 600,000";
    # 2% in Haute-Garonne, Gironde and Herault, 22% in Lozere; the other départements only as map
    # classes, so they share the rest evenly (the build solves for it so the two regions' total
    # stays 7%). 71% of those with any Occitan learned it in the family (Region Midi-Pyrenees
    # 2010, https://www.laregion.fr/IMG/pdf/EnqueteOccitan.pdf).
    dict(
        lang="Occitan", deps="occitan", age=15, young_ratio=None,
        measure="speaks it (2020) x the 71% who learned it in the family (2010)",
        cite="Office public de la langue occitane, enquete sociolinguistique 2020; Region "
             "Midi-Pyrenees 2010",
    ),
]
OCCITAN_SHARE = 0.07          # printed; its base is unstated, so the count below anchors it
OCCITAN_SPEAKERS = 600_000    # "pres de 600 000 personnes"
OCCITAN_FAMILY = 0.71
# Printed: Haute-Garonne, Gironde, Herault 2%, Lozere 22%. ASSUMED at the lowest printed share
# (2%), not read off the survey's map: the three langue d'oil départements of Nouvelle-Aquitaine
# (Charente-Maritime, Deux-Sevres, Vienne: Poitevin-Saintongeais country) and Catalan
# Pyrenees-Orientales, where Occitan is the Fenouilledes' alone.
OCCITAN_PINNED = {"31": 0.02, "33": 0.02, "34": 0.02, "48": 0.22,
                  "17": 0.02, "79": 0.02, "86": 0.02, "66": 0.02}
OCCITAN_DEPS = [  # Nouvelle-Aquitaine and Occitanie, every département (the survey's frame)
    "16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87",
    "09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82",
]

# ---- overseas: INED/INSEE Migrations, Famille et Vieillissement 2009-10 -----------------------
# Native-born parents 18-79 (summary: https://www.erudit.org/en/journals/cqd/2017-v46-n2-cqd04128/1054054ar/):
# childhood language Creole only / French+Creole: Martinique 23.5 / 57.2, Guadeloupe 34.2 /
# 47.7, Reunion 79.7 / 17.6. Passed to their children: Antilles about 9 / 47, Reunion 51.2 / 34.3.
# Applied to the non-immigrant population (the survey is of natives).
DOM_CREOLE = {
    "971": ("Antillean Creole", 0.342 + 0.477 / 2, 0.09 + 0.47 / 2),
    "972": ("Antillean Creole", 0.235 + 0.572 / 2, 0.09 + 0.47 / 2),
    "974": ("Reunion Creole", 0.797 + 0.176 / 2, 0.512 + 0.343 / 2),
}
DOM_AGE = 15
# Guyane: INSEE, enquete Pratiques culturelles 2019-20 (https://www.insee.fr/fr/statistiques/5543889):
# languages used in daily life, Guianese Creole 20%, Maroon (Bushinenge) languages 8%. Applied
# to the whole population; Maroon less the Suriname-born already drawn as Ndyuka.
GUYANE = {"Guianese Creole": 0.20, "Ndyuka": 0.08}

# ---- Mayotte: no RP 2023 table; census 2017 and INSEE 2019 -------------------------------------
# INSEE Premiere / Insee Analyses Mayotte (https://www.insee.fr/fr/statistiques/3713016): 256,518
# people in 2017; 36% born abroad, 6% born in metropolitan France or another DOM; foreigners are
# 95% Comorian and 4% Malagasy. Languages (INSEE 2019, https://www.insee.fr/fr/statistiques/6467148,
# ability only, 15+): natives speak Shimaore 82%, Kibushi 33%; Kibushi speakers nearly all speak
# Shimaore too, so the natives are split 82:33 between the two.
MAYOTTE = dict(pop=256_518, born_abroad=0.36, born_france=0.06, comorian=0.95, malagasy=0.04,
               shimaore=0.82, kibushi=0.33)
