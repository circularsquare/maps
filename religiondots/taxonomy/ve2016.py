"""
LAPOP AmericasBarometer `q3c` (Venezuela, single-country files, waves 2010-2016/17) -> taxonomy.

**The same card as `gt2023.py`, `co2023.py` and `bo2023.py`.** Read `gt2023.py` for the shared
arguments. Venezuela's last wave is 2016/17, so the card change after 2016 never reaches it: 2016/17
adds `Otro` (77) and still offers Witnesses (12) and Mormons (6), Bolivia's pattern. No respondent in
any wave chose Jewish (10). Named for the last wave in the pool.

    as drawn, 21 of 25 federal entities, post-stratified to the 2011 census
    74.03%  Católico                                    -> christianity.catholic.latin
     9.76%  Evangélica y Pentecostal                    -> christianity.evangelical
     8.41%  Ninguna (Cree en un Ser Superior…)          -> unchurched
     3.44%  Protestante, Protestante Tradicional o…     -> christianity.protestant
     1.49%  Testigos de Jehová                          -> christianity.witnesses
     0.89%  Otro                                        -> other.ve
     0.78%  Religiones Tradicionales                    -> other.ve   (NOT indigenous, see REVIEW)
     0.56%  Agnóstico o ateo (no cree en Dios)          -> secular
     0.46%  Religiones Orientales no Cristianas         -> other.ve
     0.19%  Iglesia de los Santos de los Últimos Días   -> christianity.latterday

Every row is `modelled` in §7's sense.
"""

EXCLUDED = {}

REVIEW = {
    "Religiones Tradicionales":
        "-> other.ve, NOT `indigenous` as in gt2023.py and ec2023.py, and not an `afrodiasporic` "
        "node either. **27 of its 44 Venezuelan respondents are in Caracas** (Distrito Capital 11, "
        "Miranda 16), none in Zulia, home of the Wayuu, and Amazonas and Delta Amacuro were never "
        "sampled, so it does not read as indigenous practice. What it does read as is unknowable "
        "from the file: the value label names no tradition, and a Caracas answer to a box called "
        "traditional could be Santería, the María Lionza cult or espiritismo, which a map node "
        "would be guessing between. Colombia's call (co2023.py, 18 of 37 in Bogotá). It passes the "
        "split-half at the states (median +0.581, p=0.003, chi-square 1.5e-8), so it is drawn on "
        "state shares: 3.5% of the Distrito Capital and 2.1% of Miranda. **26 of the 44 are the "
        "2016/17 wave**, which `lapop.wave_flags` flags high (26 against 6 expected); kept "
        "(sources/ve.py WAVE_FLAGS), and the pooled 0.78% is inflated by it.",
    "Católico":
        "-> christianity.catholic.latin. 74.03%, **drawn on 2010's six LAPOP design regions, not "
        "the states**: it fails the split-half at the 17 states (median +0.343, p=0.052 on 20,000 "
        "draws) and passes at the regions (+0.657, p=0.044), from 62.9% in los llanos to 81.1% in "
        "the Andes (occidental). On 2,000 draws the state verdict was seed-dependent (p 0.045 to "
        "0.058 over six seeds), which is why this country runs 20,000. Falls from 78.1% in 2010 to "
        "67.5% in 2016/17; the pooled level is kept (ask 015).",
    "Evangélica y Pentecostal":
        "-> christianity.evangelical, gt2023.py's argument. 9.76%, drawn on its own state shares "
        "(median +0.547, p=0.002, chi-square 2.5e-19): 20.4% of Bolívar and 16.2% of Portuguesa "
        "against 2.9% of the Distrito Capital. 6.1% in 2010 and 13.0% in 2016/17.",
    "Protestante, Protestante Tradicional o Protestante no Evangélico":
        "-> christianity.protestant, the 'unspecified' node. 3.44%, fails (+0.093), residual. The "
        "two Protestant boxes pass together (+0.529) but their state shares are not opposed "
        "(Spearman +0.27), so they are kept as the card offers them (bo2023.py). **The residual "
        "reverses it** (Spearman drawn against measured -0.28): Bolívar is drawn 2.5% against 9.8% "
        "measured and Portuguesa 5.1% against 11.2%, because Catholic at the region share leaves "
        "those states a small remainder (sources/ve.md §8).",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)":
        "-> unchurched, gt2023.py's reason. 8.41%, fails at the states (+0.296, p=0.079) and at "
        "the regions, residual. **The residual reverses it in places** (Spearman drawn against "
        "measured +0.13): Falcón is drawn 11.7% against 1.3% measured on 95 interviews, and "
        "Anzoátegui 10.1% against 3.2%, because both are more Catholic than their region.",
    "Testigos de Jehová":
        "-> christianity.witnesses. 1.49%, drawn on the design region (fails at the states at "
        "+0.043, passes at the regions at +0.771, p=0.021): 3.2% of Zulia and Falcón's region, "
        "where 32 of its 95 respondents are. 49 of the 95 are the 2016/17 wave.",
    "Agnóstico o ateo (no cree en Dios)":
        "-> secular, following mx2020.py. 0.56%, 32 respondents, fails (+0.268), residual.",
    "Iglesia de los Santos de los Últimos Días (Mormones)":
        "-> christianity.latterday. 0.19%, 11 respondents, residual.",
    "Otro":
        "-> other.ve. 2016/17 only, 50 respondents (3.49% of that wave), so its pooled 0.89% is "
        "one wave's figure spread over four. Kept apart from the other two answers on this node "
        "in `source_category`.",
    "Religiones Orientales no Cristianas":
        "-> other.ve, following mx2020.py and gt2023.py: one bucket from Islam to Buddhism that "
        "cannot be split without inventing the split. 0.46%. It passes the rank test (p=0.042) "
        "and the states do not differ (chi-square 0.25), so it is refused and drawn in the "
        "residual. 19 of its 29 respondents are the 2016/17 wave, which `lapop.wave_flags` flags "
        "high. Nueva Esparta (Margarita Island) was never sampled.",
    "Judío (Ortodoxo, Conservador o Reformado)":
        "-> judaism. Offered in 2010 and 2012 and chosen by nobody, so it draws nothing; mapped "
        "so a rebuild that finds one resolves.",
}

MAP = {
    "Católico": "christianity.catholic.latin",
    "Evangélica y Pentecostal": "christianity.evangelical",
    "Protestante, Protestante Tradicional o Protestante no Evangélico": "christianity.protestant",
    "Testigos de Jehová": "christianity.witnesses",
    "Iglesia de los Santos de los Últimos Días (Mormones)": "christianity.latterday",
    "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)": "unchurched",
    "Agnóstico o ateo (no cree en Dios)": "secular",
    "Judío (Ortodoxo, Conservador o Reformado)": "judaism",
    "Religiones Tradicionales": "other.ve",
    "Religiones Orientales no Cristianas": "other.ve",
    "Otro": "other.ve",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, nothing was counted anywhere.


def resolve(category):
    """religiondots branch for a LAPOP answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
