"""New Caledonia, modelled (sources/nc.py) -> religiondots taxonomy.

Named for Pew Research Center's 2020 estimate, which sets every national level. The geography and
the split of Christians into churches come from J.-M. Kohler's count of church members at the start
of 1978 (ORSTOM 1979), by community and commune, laid on the 2019 census's communities by commune.
Every row is `modelled` (spec §7b). `sources/nc.md` has the construction.

    59.15%  Catholiques                                      -> christianity.catholic.latin
    24.03%  Protestants (Églises évangéliques océaniennes)    -> christianity.reformed.congregational
    10.52%  Sans religion                                    -> unaffiliated
     2.76%  Musulmans                                        -> islam
     0.62%  Bouddhistes                                      -> buddhism
     0.62%  Autres religions                                 -> other.nc          <- new
     0.57%  Protestants (autres)                             -> christianity.protestant
     0.39%  Assemblées de Dieu                               -> christianity.pentecostal
     0.37%  Baha'is                                          -> bahai
     0.32%  Témoins de Jéhovah                               -> christianity.witnesses
     0.32%  Adventistes                                      -> christianity.adventist
     0.32%  Mormons et Sanitos                               -> christianity.latterday
"""

EXCLUDED = {}

REVIEW = {
    "Protestants (Églises évangéliques océaniennes)":
        "-> christianity.reformed.congregational, the London Missionary Society's line, as the "
        "Cook Islands Christian Church (`.cicc`) and Samoa's churches are filed. Kohler's "
        "Protestants among Kanak, Tahitians and Ni-Vanuatu: the Église évangélique en "
        "Nouvelle-Calédonie et aux îles Loyauté (16% of the territory in 1978, now the Église "
        "protestante de Kanaky Nouvelle-Calédonie), its 1958 split the Église évangélique libre "
        "(6%), and the Tahitian Evangelical Church (2%), all LMS foundations; Kohler says the "
        "Ni-Vanuatu Presbyterians worshipped with the first (p.15). No node for either church: "
        "Kohler splits the two only nationally (Melanesians 70/30, Tableau 1 note), and the "
        "split would be a legend row for one territory. The World Religion Database counts 9.5% "
        "of New Caledonia as Independents, which probably holds the Église libre; not drawn "
        "apart.",
    "Protestants (autres)":
        "-> christianity.protestant. Kohler's Protestants among Europeans (1,000), Indonesians "
        "(100) and `Autres` (400) in 1978, whose church he does not name; European Protestants "
        "in Nouméa were mostly French Reformed. 0.57% of the drawn population.",
    "Catholiques":
        "-> christianity.catholic.latin. The Archdiocese of Nouméa, Latin rite.",
    "Assemblées de Dieu":
        "-> christianity.pentecostal. Kohler calls them `Assemblées de Dieu ou Eglise "
        "Pentecôtiste`; 680 members in 1978.",
    "Mormons et Sanitos":
        "-> christianity.latterday. The Church of Jesus Christ of Latter-day Saints (530 in 1978) "
        "and the Sanitos, the Reorganized Church (260; now the Community of Christ), mostly "
        "Tahitian; the Community of Christ goes to the same node in Australia.",
    "Sans religion":
        "-> unaffiliated. Pew's `Religiously_unaffiliated` (the database's agnostics 9.49% and "
        "atheists 1.03%), placed on Kohler's `Divers`, which held people outside every church "
        "including the atheists (Tableau 1 note). The weakest drawn cell: an attitude, placed "
        "by the communities' 1978 church membership (spec §14.12).",
    "Musulmans":
        "-> islam, no branch. Pew's 2.76%, placed on Kohler's Muslims, 94% of them of Indonesian "
        "origin in 1978. Nothing says which branch.",
    "Bouddhistes":
        "-> buddhism. Pew's 0.62% (the database files them all as Mahayana; plain `buddhism` "
        "because no count names a school, the Taiwan ruling). Kohler has no Buddhist row, so "
        "they are placed at a flat share of each commune's `other and not declared` census "
        "column, where the Vietnamese and other Asian communities are counted.",
    "Autres religions":
        "-> other.nc, new. Pew's `Other_religions` less the database's Bahá'í share, plus Pew's "
        "Jews: ethnic religionists 0.18% and new religionists 0.41% in the database's split. "
        "Flat across every commune.",
}

MAP = {
    "Catholiques": "christianity.catholic.latin",
    "Protestants (Églises évangéliques océaniennes)": "christianity.reformed.congregational",
    "Protestants (autres)": "christianity.protestant",
    "Assemblées de Dieu": "christianity.pentecostal",
    "Adventistes": "christianity.adventist",
    "Témoins de Jéhovah": "christianity.witnesses",
    "Mormons et Sanitos": "christianity.latterday",
    "Sans religion": "unaffiliated",
    "Musulmans": "islam",
    "Baha'is": "bahai",
    "Bouddhistes": "buddhism",
    "Autres religions": "other.nc",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`.


def resolve(category):
    """religiondots branch for a model category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
