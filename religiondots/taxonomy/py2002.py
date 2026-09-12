"""
DGEEC 2002 Censo Nacional de Población y Viviendas, variable P17 -> religiondots taxonomy.

**FIFTY-FOUR CATEGORIES, WHICH IS THE LONGEST LIST ON THIS MAP.** Austria's 31 was what
`queue.md` called the deepest undrawn religion list in the world. Paraguay asked one question
of everyone aged 10 and over and coded the answers into 54 cells, and it did it in 2002 and
then never asked again.

    Católica                        3,488,086  89.61%  -> christianity.catholic.latin
    Otras - Evangélica                186,107   4.78%  -> christianity.evangelical
    Sin religión                       44,334   1.14%  -> unaffiliated
    No especificado                    37,206   0.96%  §3.5 residual, EXCLUDED
    Religión indígena                  23,741   0.61%  -> indigenous
    Pueblo de Dios                     12,114   0.31%  -> christianity
    Testigos de Jehova                 11,805   0.30%  -> christianity.witnesses
    Bautista. Bautista Maranata        10,355   0.27%  -> christianity.baptist
    Asamblea de Dios                    9,879   0.25%  -> christianity.pentecostal.trinitarian
    Mormones                            9,374   0.24%  -> christianity.latterday
    Luterana                            8,849   0.23%  -> christianity.lutheran
    Pentecostal                         8,631   0.22%  -> christianity.pentecostal
    Mennonita                           8,445   0.22%  -> christianity.anabaptist.mennonite
    Adventista                          7,804   0.20%  -> christianity.adventist
    Otra religión No Especificada       6,139   0.16%  -> other.py
    Budismo                             2,088   0.05%  -> buddhism
    Anglicana                           1,858   0.05%  -> christianity.anglican
    ... and 37 more, down to Independientes and Rosacruces at 7 people each.

(Percentages are of the 3,892,603-person universe.)

**THE ONE CELL THAT IS NOT A BODY IS THE SECOND LARGEST.** `Otras - Evangélica` is 186,107
people, **4.78% of the country**, and it sits at the end of a block of nineteen named
Protestant churches rather than in place of them. So it is a residual WITHIN Protestantism and
not a coarse cell standing in for the named ones: whatever is in it, DGEEC had Asamblea de
Dios, Bautista, Luterana, Mennonita, Pentecostal and Presbiteriana available and did not use
them. It goes to `christianity.evangelical`, the family node, where it can neither claim a
denomination nor fall out of Protestantism. It is far and away the largest uncertainty in this
file and it is twenty-five times the size of `other.py`.

**FIVE CATEGORIES COUNT SYNCRETISM, AND THE TREE CANNOT.** `Indígena + católica` (223),
`+ anglicana` (29), `+ evangélica` (1,203), `+ mennonita` (8) and `+ otras religiones` (15)
are answers from people who declared both an indigenous religion and a church. Nothing else
drawn on this map offers that answer at all, and the Anglican and Mennonite pairings are the
Chaco missions written into a census form. All five go to `indigenous`, following the census's
own block structure (they are codes 4 to 8, immediately after `Religión indígena` at code 3
and before the Orthodox block at 9), and the cost is stated in REVIEW: 1,478 people whose
church affiliation the source recorded are drawn without it. Splitting them the other way
would have erased the indigenous half instead, and there is no third option in a tree that
puts each person in one place.

**AND THE TAIL IS WHY THE COUNTRY IS WORTH DRAWING.** Reyukai (72) and Sintoismo (30) are the
Japanese agricultural colonies at La Colmena, Yguazú, Pirapó and La Paz, planted from 1936, and
those four districts top the Buddhist ranking too; `Umbanda` (54) and
`Espiritualistas - E.C.Basilio` (289) come over the Brazilian and Argentine borders;
`Fe Bahía` (225) is DGEEC's spelling of Bahá'í. No other census on this map counts a Japanese
new religion at all.

**THE CENSUS CALLS NINE OF ITS OWN CATEGORIES `Pseudo-Cristianos`** (codes 32 to 40:
Adventista, Dios es amor, Iglesia Universal, Moon, Mormones, Pueblo de Dios, Testigos de
Jehova, Monte de Sión, and the residual). That is DGEEC's judgement and not a structural fact,
and it is not carried here: the Adventists go to `christianity.adventist` and the Igreja
Universal to `christianity.pentecostal` because that is what those bodies are. The block is
recorded because it explains why the list is ordered the way it is, and because the Unification
Church really does sit outside Christianity in this tree and is the only one of the nine that
does.
"""

EXCLUDED = {
    "No especificado":
        "37,206 people, **0.96% of the universe** — the real non-response to P17, as "
        "distinct from `No Aplica` (the under-10s, which `sources/py.py` drops before this "
        "file ever sees it). Spec §3.5: marked, not filled. Paraguay is 99.04% drawn.",
}

REVIEW = {
    "Otras - Evangélica":
        "-> christianity.evangelical. **186,107 people, 4.78%, and the largest uncertainty "
        "in this file by an order of magnitude.** The judgement is that it is a residual "
        "within Protestantism rather than a coarse cell: it is code 30, at the end of a run "
        "of nineteen named Protestant bodies (codes 12 to 29), so a coder who meant Asamblea "
        "de Dios or Bautista had those codes to hand. What it cannot tell us is the shape of "
        "what is inside, and in a country where Pentecostalism grew fast through the 1990s "
        "the honest reading is that much of it is Pentecostal. It stays on the family node "
        "because §14.4 rule 1 forbids inventing the split. "
        "**AND IN ONE DISTRICT IT IS ALMOST CERTAINLY MENNONITE.** Boqueron reads **39.5% "
        "`Otras - Evangélica` against 4.8% nationally**, eight times the rate, in the one "
        "district holding the Fernheim, Menno and Neuland colonies, and only 12.2% "
        "`Mennonita`. The colonies carry two church bodies and only one of them is called "
        "Mennonite in Spanish: the *Mennonitische Brüdergemeinde* self-describes as "
        "*evangélica*. Crediting Boqueron's excess to Mennonites would take the national "
        "figure from 8,445 to about 19,000, still only 0.49% of the country. Not done "
        "here, because §14.4 rule 1 applies to a plausible split as much as to a guessed "
        "one, but it is the shape of what this cell hides.",
    "Religión indígena":
        "-> indigenous, undivided, following br2010.py. 23,741 people. DGEEC names no "
        "individual people here even though the same census identifies twenty indigenous "
        "peoples in its `PUEBLO` variable, so the tree stays on the family node.",
    "Indígena + católica":
        "-> indigenous, and this is the call worth a second opinion. 223 people who told the "
        "census they are both. The five `Indígena + …` codes (4 to 8) sit inside the "
        "census's own indigenous block, between `Religión indígena` and the Orthodox, which "
        "is the reason the indigenous half is taken as primary. It is a reading of the "
        "form's structure and not of the people's answer, and the alternative reading would "
        "put these 223 in `christianity.catholic.latin` and lose the other half. 1,478 "
        "people across the five codes, 0.08% of the universe.",
    "Indígena + anglicana":
        "-> indigenous. 29 people, and geographically the most specific cell in the census: "
        "the Anglican mission in the Paraguayan Chaco (Makthlawaiya and Sanapaná country) "
        "is the only reason this pairing exists. Same reading as `Indígena + católica`.",
    "Indígena + mennonita":
        "-> indigenous. 8 people, the smallest cell here that is not a rounding artefact, "
        "and it names the Enlhet and Nivaclé congregations of the Mennonite colonies in "
        "Boquerón. Same reading as `Indígena + católica`.",
    "Pueblo de Dios":
        "-> christianity, the bare family node, and the least identified large cell in the "
        "file. 12,114 people, 0.31%, which makes it the sixth-largest category in Paraguay. "
        "DGEEC files it in its `Pseudo-Cristianos` block (codes 32 to 40) beside the Mormons "
        "and the Jehovah's Witnesses, so the office read it as outside mainstream "
        "Christianity, but the label names no body this file can identify with confidence "
        "and the block heading is a judgement rather than a description. It sits on "
        "`christianity` where it can neither claim a denomination nor be pushed out of the "
        "religion. Worth a second opinion from someone who knows Paraguayan church history.",
    "Monte de Sión":
        "-> christianity.pentecostal. 233 people. `Monte de Sión` is the usual Spanish name "
        "of a Pentecostal congregation type and the surrounding codes are Brazilian "
        "neo-Pentecostal imports, but DGEEC publishes nothing about it. On the family node, "
        "so a wrong reading misplaces 233 people by one branch.",
    "Iglesia Universal - Pare de Sufri":
        "-> christianity.pentecostal. 714 people. The label is DGEEC's truncation of "
        "**Iglesia Universal del Reino de Dios**, the Brazilian neo-Pentecostal church whose "
        "Spanish slogan is *Pare de sufrir*. Neo-Pentecostal rather than classical "
        "Pentecostal, and the tree has no node for that, so it sits on the family.",
    "Dios es amor":
        "-> christianity.pentecostal. 1,290 people. **Igreja Pentecostal Deus é Amor**, "
        "founded in São Paulo in 1962 and the other large Brazilian Pentecostal body in "
        "Paraguay. Classical Pentecostal, but DGEEC publishes nothing on its doctrine of the "
        "Godhead, which is what `trinitarian` and `oneness` divide on.",
    "Hinduismo(Tao)":
        "-> hinduism, and the parenthesis is DGEEC's, not this file's. **The census merges "
        "Hinduism and Taoism into one code**, 151 people. Hinduism is taken as the primary "
        "reading because the label leads with it and because Paraguay's South Asian "
        "community is the larger of the two, but a Taoist counted in 2002 is drawn here as a "
        "Hindu and nothing in the source separates them.",
    "Rusa":
        "-> christianity.orthodox.canonical. 470 people, and the label means the Russian "
        "Orthodox Church rather than a nationality: Paraguay took in about 2,000 White "
        "emigres in the 1920s and the parish in Asunción dates from then. The tree has no "
        "plain `russian` node (its Orthodox children are the American jurisdictions), so it "
        "goes to the canonical family, which is where the Moscow Patriarchate belongs.",
    "Mentalistas(Meditación Transcende":
        "-> esoteric. 164 people, DGEEC's truncation of *Meditación Trascendental*. "
        "Transcendental Meditation presents itself as a technique rather than a religion and "
        "the census counted it as one; `esoteric` is where this tree puts movements of that "
        "shape, beside the Rosicrucians two codes below.",
    "Fe Bahía":
        "-> bahai. 225 people. DGEEC's spelling; §2.4 keeps the source's own string in "
        "`source_category`, so it reaches the reader as printed.",
    "Reyukai":
        "-> eastasiannew.japanese. 72 people, and **the only Japanese new religion counted "
        "by any census on this map.** Reiyūkai is a lay Nichiren movement; it is in Paraguay "
        "because the Japanese agricultural colonies are, at La Colmena from 1936 and Yguazú "
        "Pirapó and La Paz from the 1950s. `Sintoismo` (30) is the same community, and those "
        "four districts hold the four highest Buddhist shares in Paraguay.",
    "Hermanos Libres":
        "-> christianity.plymouth. 665 people. `Hermanos Libres` is the standard Spanish "
        "name of the Open Brethren, which is what `christianity.plymouth` holds; it is NOT "
        "the Schwarzenau Brethren under `christianity.anabaptist.brethren`, which in Spanish "
        "would be *Hermanos de Schwarzenau* or *Dunkers*.",
    "Iglesia de Dios":
        "-> christianity.pentecostal.trinitarian.cog-unspecified. 1,550 people, and the "
        "label is exactly as ambiguous in Spanish as `Church of God` is in English. The "
        "unspecified Pentecostal node exists for this case. `Iglesia de Dios de la Profecía` "
        "(149) is a separate code and is unambiguous, so the two do not collide.",
    "Alianza Cristiana y Misionera":
        "-> christianity.holiness.cma. 87 people. The Christian and Missionary Alliance, "
        "unambiguous in Spanish.",
    "Centro Fam. de Adoración. Aposent":
        "-> christianity.pentecostal. 513 people. DGEEC has truncated two church names "
        "into one code: *Centro Familiar de Adoración* and what is almost certainly "
        "*Aposento Alto*. Both are Pentecostal, which is why the truncation does not matter "
        "here, but the cell is two bodies and not one.",
    "Independientes":
        "-> christianity.nondenominational.independent. **7 people, the joint-smallest category "
        "in the census**, and it is listed among the named Protestant bodies rather than "
        "among the residuals, so it means congregations that describe themselves as "
        "independent rather than 'not stated'.",
    "Comunidad Cristiana":
        "-> christianity.nondenominational. 1,046 people. The label names a congregation "
        "type rather than a denomination; it sits between `Bautista` and `Hermanos Libres` "
        "in the code order, among named bodies.",
    "Neotestamentaria":
        "-> christianity. 276 people. *Iglesia Neotestamentaria* names no body this file can "
        "identify and the bare family node is where an unidentifiable Christian label goes "
        "rather than a guessed branch.",
    "Otros grupos Pseudo-Cristianos":
        "-> christianity. 825 people, and the node is chosen against the census's own "
        "heading. `Pseudo-Cristianos` is DGEEC's judgement about the block it closes "
        "(Adventists, Mormons, Jehovah's Witnesses and the Brazilian neo-Pentecostals); the "
        "bodies in it are Christian by this tree's structure, so the residual goes to "
        "`christianity` and not off the branch.",
    "Espiritualistas - E.C.Basilio":
        "-> spiritualism. 289 people. The **Escuela Científica Basilio**, the Argentine "
        "spiritualist school founded in Buenos Aires in 1917, which is a body and not a "
        "residual; `Otras, Espiritismo` (66) is the residual and goes to the same node "
        "because the tree has nowhere finer that fits both.",
}

MAP = {
    # --- Catholic, and the census's two non-religion answers -------------------------
    "Católica": "christianity.catholic.latin",
    "Sin religión": "unaffiliated",

    # --- the indigenous block, codes 3 to 8 ------------------------------------------
    "Religión indígena": "indigenous",
    "Indígena + católica": "indigenous",
    "Indígena + anglicana": "indigenous",
    "Indígena + evangélica": "indigenous",
    "Indígena + mennonita": "indigenous",
    "Indígena + otras religiones": "indigenous",

    # --- Orthodox, codes 9 to 11 -----------------------------------------------------
    "Ortodoxa": "christianity.orthodox",
    "Rusa": "christianity.orthodox.canonical",
    "Otras - Ortodoxa": "christianity.orthodox",

    # --- the named Protestant bodies, codes 12 to 30 ---------------------------------
    "Alianza Cristiana y Misionera": "christianity.holiness.cma",
    "Anglicana": "christianity.anglican",
    "Asamblea de Dios": "christianity.pentecostal.trinitarian",
    "Bautista. Bautista Maranata": "christianity.baptist",
    "Centro Fam. de Adoración. Aposent": "christianity.pentecostal",
    "Comunidad Cristiana": "christianity.nondenominational",
    "Hermanos Libres": "christianity.plymouth",
    "Independientes": "christianity.nondenominational.independent",
    "Iglesia de Dios": "christianity.pentecostal.trinitarian.cog-unspecified",
    "Iglesia de Dios de la Profecía": "christianity.pentecostal.trinitarian",
    "Luterana": "christianity.lutheran",
    "Mennonita": "christianity.anabaptist.mennonite",
    "Metodista": "christianity.methodist",
    "Metodista Libre": "christianity.methodist.free",
    "Nazarena": "christianity.holiness.nazarene",
    "Neotestamentaria": "christianity",
    "Pentecostal": "christianity.pentecostal",
    "Presbiteriana": "christianity.reformed.presbyterian",
    "Otras - Evangélica": "christianity.evangelical",

    # --- Judaism, then the block DGEEC heads `Pseudo-Cristianos`, codes 31 to 40 ------
    "Judaismo": "judaism",
    "Adventista": "christianity.adventist",
    "Dios es amor": "christianity.pentecostal",
    "Iglesia Universal - Pare de Sufri": "christianity.pentecostal",
    "Iglesia de la Unificación - Moon": "unification",
    "Mormones": "christianity.latterday",
    "Pueblo de Dios": "christianity",
    "Testigos de Jehova": "christianity.witnesses",
    "Monte de Sión": "christianity.pentecostal",
    "Otros grupos Pseudo-Cristianos": "christianity",

    # --- everything else, codes 41 to 53 ---------------------------------------------
    "Islamica - Musulmana": "islam",
    "Hinduismo(Tao)": "hinduism",
    "Espiritualistas - E.C.Basilio": "spiritualism",
    "Fe Bahía": "bahai",
    "Rosacruces": "esoteric",
    "Umbanda": "afrodiasporic.umbanda",
    "Otras, Espiritismo": "spiritualism",
    "Budismo": "buddhism",
    "Reyukai": "eastasiannew.japanese",
    "Sintoismo": "shinto",
    "Relig. no incluidas en las anteri": "other.py",
    "Otra religión No Especificada": "other.py",
    "Mentalistas(Meditación Transcende": "esoteric",
}

# No COLUMNS dict (spec §7a-i-1): every one of the 54 categories is measured at the
# district it is drawn on, so no row is `derived` and nothing ever needs to roll up.


def resolve(category):
    """religiondots branch for a DGEEC category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
