"""Angola — the 2024 census report set, and where each PDF lives.

Shared by `sources/ao.py` (religion by municipality, 21 provincial volumes) and by any
later module that wants the national volume. Kept apart from `ao.py` so the URL table can
be read without importing the parser.

INE publishes every file under one flat directory, `/Arquivos/arquivosCarregados/Carregados/`,
named `Publicacao_<18-digit .NET tick stamp>.pdf`. The stamp is the upload time and carries
no meaning, so THE FILENAME CANNOT BE DERIVED FROM THE PROVINCE and this table is the only
index. It was built by parsing the publications sidebar on
`ine.gov.ao/publicacoes/detalhes/NDc0MTE=`, which lists every census volume with its title.

If a fetch 404s, that sidebar is where to look: re-reading it costs one request and the
titles are unambiguous.
"""

BASE = "https://www.ine.gov.ao/Arquivos/arquivosCarregados/Carregados/"

# The national volume, published 2025-11-20. Religion is Quadro 7.1/7.2 by PROVINCE, and
# `ao.py` reads it only as a check on the 21 provincial volumes.
NATIONAL = "Publicacao_638996687409619846.pdf"

# Province -> its volume, published 2026-01 (Cunene, Huila, Namibe) or 2026-02 (the rest).
# The key is INE's own spelling in the report title; `ao.py` maps it to the boundary
# layer's spelling.
PROVINCES = {
    "Bengo": "Publicacao_639070145024903075.pdf",
    "Benguela": "Publicacao_639064071318925115.pdf",
    "Bié": "Publicacao_639071861895047488.pdf",
    "Cabinda": "Publicacao_639086316474352489.pdf",
    "Cuando": "Publicacao_639175197254207294.pdf",
    "Cuanza Norte": "Publicacao_639074613297362596.pdf",
    "Cuanza Sul": "Publicacao_639065784580611643.pdf",
    "Cubango": "Publicacao_639175237509463838.pdf",
    "Cunene": "Publicacao_639050104455150531.pdf",
    "Huambo": "Publicacao_639064975879258682.pdf",
    "Huíla": "Publicacao_639051372856668517.pdf",
    "Icolo e Bengo": "Publicacao_639081910852232827.pdf",
    "Luanda": "Publicacao_639175196705255655.pdf",
    "Lunda Norte": "Publicacao_639062413002295499.pdf",
    "Lunda Sul": "Publicacao_639064347606305071.pdf",
    "Malanje": "Publicacao_639071901376985768.pdf",
    "Moxico": "Publicacao_639064111089603097.pdf",
    "Moxico Leste": "Publicacao_639065854702898637.pdf",
    "Namibe": "Publicacao_639051805987160989.pdf",
    "Uíge": "Publicacao_639120325078684256.pdf",
    "Zaire": "Publicacao_639175220682196239.pdf",
}

# Quadro 9 of the national volume, `Número de municípios, comunas e localidades por
# província à luz da Lei 14/24 de 5 de Setembro`. Asserted against the municipality rows
# actually parsed out of each provincial volume, and against the boundary layer.
MUNICIPALITIES = {
    "Cabinda": 10, "Zaire": 11, "Uíge": 23, "Bengo": 12, "Luanda": 16,
    "Cuanza Norte": 17, "Cuanza Sul": 24, "Malanje": 27, "Lunda Norte": 19,
    "Lunda Sul": 14, "Moxico": 12, "Bié": 19, "Huambo": 17, "Benguela": 23,
    "Namibe": 9, "Huíla": 23, "Cunene": 14, "Cubango": 11, "Icolo e Bengo": 7,
    "Moxico Leste": 9, "Cuando": 9,
}

assert sorted(PROVINCES) == sorted(MUNICIPALITIES), "province tables disagree"
assert sum(MUNICIPALITIES.values()) == 326, sum(MUNICIPALITIES.values())
