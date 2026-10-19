"""
Zensus 2022 religion classification -> religiondots taxonomy.

**The shallowest mapping in the project, and the only one where that is a property of the
instrument rather than of the table.** Three categories, and the reason there are three is
that nobody was asked: Zensus 2022 has no religion question, and the figures are read off
the Melderegister, which records religious-body membership because it determines church-tax
liability. So the data can see the two churches that levy church tax and nothing else.
`basis` is `roll` (spec §3.1) and sources/de.py's docstring has the full argument.

Croatia is drawn shallow ON PURPOSE — DZS publishes 54 named churches at the same geography
and joining them has not been done (hr2021.py). Germany is different: there is no deeper
table anywhere, in this census or in 2011, and `sources/de.md` §2 records why the 2011
attempt failed. Nothing here is deferred work.

WHAT THIS COSTS, stated plainly because the map cannot say it and the about panel must:
Germany's roughly four million Muslims, its two million Orthodox Christians, its Jewish
communities and its Freikirchen are all inside `unrecorded`, indistinguishable from the
people who belong to nothing. That is not a modelling choice — the register never knew.
Drawing them would mean inventing both a magnitude and a location, which spec §14.3 forbids
in exactly these words: never model at a finer resolution, or a stronger claim, than the
source publishes its magnitude at.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Einwohnerzahl":
        "the unit's own population total, not a category. Note it is NOT the sum of the "
        "three categories: the Cell-Key disclosure method perturbs the category cells and "
        "leaves the Einwohnerzahl untouched, so the three fall short of it by 174 people "
        "nationally (sources/de.py reconciles this).",
}

REVIEW = {
    "Römisch-katholische Kirche (öffentlich-rechtlich)":
        "-> christianity.catholic, the PARENT, not christianity.catholic.latin. Croatia's "
        "`Katolici` is filed the same way for the same reason: the category is the Roman "
        "Catholic Church as a legal body, and Germany's Eastern Catholics pay their church "
        "tax through it, so it is not the Latin rite alone. destatis is explicit that it "
        "EXCLUDES the Alt-Katholiken ('nicht aber der Altkatholiken und verwandter "
        "Gruppen'), who are a separate public-law body and therefore sit, uncounted, "
        "inside `unrecorded` — christianity.catholic.independent exists and gets nothing.",

    "Evangelische Kirche (öffentlich-rechtlich)":
        "-> christianity.protestant, whose own note in branches.py already named Germany "
        "as a case it would be wanted for. The category is the EKD, which destatis "
        "defines as 'der Zusammenschluss der zwanzig selbständigen lutherischen, "
        "reformierten und unierten Landeskirchen'. That spans three of the tree's "
        "families, and the tree has NO Protestant super-node on purpose — Lutheran and "
        "Reformed are siblings, not children of one. So this is the second kind of thing "
        "`christianity.protestant` holds: not 'the answer Protestant with no body named' "
        "(Czechia, Croatia) but 'a named body that spans the Protestant families'. "
        "christianity.united was considered and rejected: that node is for churches formed "
        "BY union, like the Uniting Church in Australia, and the EKD is a federation whose "
        "twenty members stayed independent. Splitting the 19.1M across lutheran / reformed "
        "/ united by Landeskirche would be possible — the Landeskirchen have territories — "
        "but destatis publishes one number and the split would be an allocation inventing "
        "structure the source does not have (spec §3.10). RE-EXAMINED AND UPHELD "
        "2026-09-07, Anita's call, and the reason turned out to be stronger than the one "
        "written here. The territories are real for eighteen of the twenty — but the "
        "Evangelisch-reformierte Kirche is NOT a territorial church at all, and EKD's own "
        "footnote says population cannot be assigned to it: 151,083 members in 141 "
        "congregations scattered inside Lutheran Hannover's borders and as far as Bavaria. "
        "The united Landeskirchen are internally mixed by construction too, since the 1817 "
        "Prussian Union merged administration and left congregations Lutheran or Reformed. "
        "So a territorial join would draw which church body GOVERNS a place, not what "
        "confession its people hold — and it would miss the Reformed almost entirely, who "
        "are ~1.5% of the 19.1M. See sources/de.md §8.\n\n"
        "REVERSED 2026-10-16, Anita's call (spec §2.7b, sources/de.md §12). The column now stays on "
        "this node only where a row is not split, which is nowhere: sources/de_ekd.py writes every "
        "Gemeinde's Protestants again under its Landeskirche's name. What changed the call is church "
        "law, not a new table. Kirchenmitgliedschaftsgesetz §1 Abs. 2: 'Die Kirchenmitgliedschaft "
        "besteht zur Kirchengemeinde und zur Gliedkirche des Wohnsitzes', and EKD's Table 1 footnote 3 "
        "puts membership across Landeskirche borders under 0.14 per mille. So the register's count in a "
        "Gemeinde IS members of the church whose territory holds it, and 'which body governs a place' "
        "is what a roll basis claims. The Reformed objection is met by placing the ERK from its own "
        "regional figures, the one modelled step. The node is still reachable through the roll-up "
        "(COLUMNS) and through ESS's `Andere protestantische Konfession`.",

    "Evangelische Kirche: Rheinland (uniert)":
        "-> christianity.united, and the nine Union churches with it (Westfalen, Hessen und Nassau, "
        "Baden, Berlin-Brandenburg-schlesische Oberlausitz, Kurhessen-Waldeck, Pfalz, Bremen, Anhalt, "
        "and Mitteldeutschland outside Thüringen). Anita's call, 2026-10-16. The node is 'churches "
        "formed by union across the Protestant families', and these are that literally: the Prussian "
        "Union of 1817, the Palatinate's of 1818, Baden's of 1821. The 2026-09-07 rejection of this "
        "node was about the EKD as a whole, a federation, which is a different thing. THE KNOWN COST: "
        "a Union church's congregations kept their own confession, so a Lutheran parish in Wuppertal is "
        "drawn United, as a Lutheran parish inside the PKN is drawn as the PKN (nl2014.py).",

    "Evangelische Kirche: Mitteldeutschland, Thüringen (lutherisch)":
        "-> christianity.lutheran; the rest of the EKM -> christianity.united. Anita's call, 2026-10-16. "
        "The EKM (2009) joined the Lutheran church of Thuringia and the Union's Kirchenprovinz Sachsen and "
        "belongs to both the VELKD and the UEK, so it is divided by Land: Thuringia Lutheran, Sachsen-"
        "Anhalt and the EKM's corners of Saxony and Brandenburg (Torgau, Delitzsch, Bad Liebenwerda) "
        "United. THE KNOWN COST: Erfurt, Mühlhausen, Nordhausen and the other Prussian parts of Thuringia "
        "were Kirchenprovinz, not Thuringian church, and are drawn Lutheran.",

    "Evangelische Kirche: Evangelisch-reformierte Kirche (reformiert)":
        "-> christianity.reformed.continental, with Lippe. THE ONLY MODELLED STEP in the split: the ERK has "
        "no territory, so sources/de_ekd.py takes its members out of Hannover's count in its two home "
        "regions, from the church's own 2010 report scaled to its 2022 total of 159,309: 61,954 in "
        "Ostfriesland (23.1% of the Protestants there), 53,103 in the Grafschaft Bentheim and Emsland "
        "(48.3%), spread by Protestants. The scattered third (about 44,000, as far as Bavaria) stays with "
        "the church whose territory it lives in. Rows are `derived` and not `modelled` all the same "
        "(spec §2.7a): these people were counted, as Protestants, in the Gemeinde they are drawn in.",

    "Sonstige, keine, ohne Angabe":
        "-> unrecorded, a node added for this source (branches.py, 2026-09-04) and the "
        "single most consequential call in this file: 42,845,220 people, 51.8% of Germany. "
        "Every existing home for it asserts something false. `unaffiliated` is a person "
        "reporting no religion, and nobody was asked. `other.de` would be a religion the "
        "source named but the tree cannot place, and the source named nothing. "
        "`unchurched` is a positive report of belief without institution. destatis' own "
        "definition is the argument for a separate node — the bucket holds people in OTHER "
        "public-law bodies too, because 'für diese anderen öffentlich-rechtlichen "
        "Religionsgesellschaften liegen nur in sehr begrenztem Umfang Einträge im "
        "Melderegister vor'. So it is three different things at once (another body, no "
        "body, no entry) and its composition is a fact about the register rather than "
        "about the people. It gets the greyest treatment in the §6.3a family for that "
        "reason.",

    "Jüdische Gemeinde (ZWST-Mitgliedsgemeinde)":
        "-> judaism, and THIS CATEGORY IS NOT FROM THE ZENSUS. It is the one part of "
        "`unrecorded` that a second source can count rather than estimate: the "
        "Zentralwohlfahrtsstelle der Juden in Deutschland publishes membership per "
        "community, 87,934 people in 2025, and sources/de_zwst.py seats them. It lives in "
        "this file rather than in a de_zwst2025.py because taxonomy/registry.py discovers "
        "one module per country and a second vintage for `de` would need an OVERRIDE entry "
        "to say which one is drawn — when the honest answer is BOTH, one for each source.\n\n"
        "`judaism` and not a movement below it: ZWST's members are mostly Einheitsgemeinden, "
        "single communities that span orthodox and liberal on purpose, and the statistic "
        "names no movement anywhere. Israel's four observance nodes are the wrong axis and "
        "the wrong question (branches.py, §6.15) and are not reachable from a membership "
        "roll.\n\n"
        "THE BASIS IS THE REASON THIS IS ALLOWED AT ALL. Mixing a survey into the Zensus "
        "columns would break spec §3.1, which is why the ESS split scouted in "
        "sources/de.md §6 is still unbuilt. ZWST is a register too — an association's own "
        "membership list, exactly the kind of thing the Melderegister is — so `roll` meets "
        "`roll` and nothing is mixed. countries.py SUBTRACTS these people from the same "
        "Gemeinde's `unrecorded` before adding them here, because that is the cell they are "
        "sitting in today; adding without subtracting would give Germany 87,934 extra "
        "people.\n\n"
        "What it does not fix: this is affiliated membership, so the unaffiliated and much "
        "of the post-2022 Ukrainian arrival stay inside `unrecorded` (spec §3.5 — marked, "
        "not filled), and a roll counts the institution's location and not the member's "
        "(§3.6), which here means a regional catchment drawn at its seat.",

    "Muslimisch/Islam":
        "-> islam, the PARENT, and not islam.sunni the way fr2024.py files the same ESS "
        "answer for France. Germany's Muslim population is not simply Sunni: the Alevis "
        "are a large minority of the Turkish-origin community and there is a substantial "
        "Shia population from Iran, Iraq, Lebanon and Afghanistan. Zensus 2011 knew this — "
        "its Frage 8 offered Sunni, Shia and Alevi as three separate answers — so naming "
        "one of them off a survey that offers only `Muslimisch/Islam` would assert exactly "
        "the distinction the German instrument that DID ask it declined to publish "
        "usably. UPDATED 2026-10-16: the ESS cell is now divided by a source that asks, "
        "BAMF's MLD 2020 (the `MLD 2020: ...` rows below); `islam` keeps the share MLD "
        "leaves unnamed.",

    "MLD 2020: Alevitisch":
        "-> alevism, not islam.shia, following the map's existing rule for a source that "
        "names Alevis: England and Wales's write-in `Alevi` (uk2021.py) and every foreign "
        "half's Turkish nationals (origin_religion.py) are on `alevism`. Bulgaria's "
        "Kızılbaş are on islam.shia only because that census's cell says Shia (§2.7). MLD "
        "counts Alevis among Muslims, so some German Alevis who would not call themselves "
        "Muslim are probably in ESS's other or none answers and are not drawn here; only "
        "the share MLD gives of ESS's Muslims is moved.",

    "Eine evangelische Freikirche":
        "-> christianity.evangelical, whose own note says it is 'for sources that collect "
        "`Evangelical` as an answer distinct from both `Protestant` and a named body' and "
        "that it is deliberately NOT a parent of anything. The German Freikirche is that "
        "category exactly: the VEF umbrella spans Baptists, Methodists, Pentecostals, "
        "Brethren and Mennonites, so it cuts across the tree's families rather than "
        "sitting inside one. It must NOT go to christianity.protestant — that node is "
        "already carrying Germany's 19.1M EKD members, and the whole point of the "
        "Freikirchen is that the register and the ESS both treat them as NOT the EKD "
        "(destatis' column is 'Evangelische Kirche', ESS's is 'EKD, ohne Freikirchen').",

    "Andere christliche Konfession":
        "-> christianity, the root, as an unspecified Christian answer — the same call "
        "fr2024.py makes for ESS's `Other Christian denomination`. NOT christianity.other, "
        "whose note is explicit that it holds 'bodies with no branch to belong to, not a "
        "residual', and this answer is a residual. `Christlich, aber fühlt sich keiner "
        "spezifischen Religionsgemeinschaft zugehörig` joins it here rather than at "
        "christianity.nondenominational: that node is a positive American answer about "
        "belonging to a non-denominational church, and this one is the opposite — a person "
        "saying no denomination fits them.",

    "Östlich-orthodox":
        "-> christianity.orthodox, the PARENT, and not .canonical the way France files it. "
        "Germany's Orthodox are mostly the canonical churches of the Greek, Serbian, "
        "Romanian and Russian diasporas, but it also holds Europe's largest Syriac "
        "Orthodox community, plus Copts and Armenians — Oriental Orthodox, a different "
        "branch of the tree, who have no other answer on this form to give. Filing the "
        "whole cell as canonical would assert something false about tens of thousands of "
        "people; the parent asserts only what the answer says.",

    "Östliche Religionsgemeinschaft":
        "-> other.de, together with `Andere nicht-christliche Religionsgemeinschaft`. See "
        "that node in branches.py: the tree has no home for 'some Eastern religion, "
        "unspecified', Zensus 2011 did print Buddhism and Hinduism as their own boxes and "
        "the answers are unusable, and the cell's likely contents are knowable without "
        "being separable.",
}

MAP = {
    "Römisch-katholische Kirche (öffentlich-rechtlich)": "christianity.catholic",
    "Evangelische Kirche (öffentlich-rechtlich)": "christianity.protestant",
    # --- the same column split by Landeskirche, sources/de_ekd.py (2026-10-16). `derived`, rolling
    # back to `christianity.protestant` through COLUMNS. The confession is the BODY's.
    "Evangelische Kirche: Bayern (lutherisch)": "christianity.lutheran",
    "Evangelische Kirche: Hannover (lutherisch)": "christianity.lutheran",
    "Evangelische Kirche: Württemberg (lutherisch)": "christianity.lutheran",
    "Evangelische Kirche: Nordkirche (lutherisch)": "christianity.lutheran",
    "Evangelische Kirche: Sachsen (lutherisch)": "christianity.lutheran",
    "Evangelische Kirche: Oldenburg (lutherisch)": "christianity.lutheran",
    "Evangelische Kirche: Braunschweig (lutherisch)": "christianity.lutheran",
    "Evangelische Kirche: Schaumburg-Lippe (lutherisch)": "christianity.lutheran",
    "Evangelische Kirche: Mitteldeutschland, Thüringen (lutherisch)": "christianity.lutheran",
    "Evangelische Kirche: Rheinland (uniert)": "christianity.united",
    "Evangelische Kirche: Westfalen (uniert)": "christianity.united",
    "Evangelische Kirche: Hessen und Nassau (uniert)": "christianity.united",
    "Evangelische Kirche: Baden (uniert)": "christianity.united",
    "Evangelische Kirche: Berlin-Brandenburg-schlesische Oberlausitz (uniert)": "christianity.united",
    "Evangelische Kirche: Kurhessen-Waldeck (uniert)": "christianity.united",
    "Evangelische Kirche: Pfalz (uniert)": "christianity.united",
    "Evangelische Kirche: Bremen (uniert)": "christianity.united",
    "Evangelische Kirche: Anhalt (uniert)": "christianity.united",
    "Evangelische Kirche: Mitteldeutschland, outside Thüringen (uniert)": "christianity.united",
    "Evangelische Kirche: Lippe (reformiert)": "christianity.reformed.continental",
    "Evangelische Kirche: Evangelisch-reformierte Kirche (reformiert)": "christianity.reformed.continental",
    "Sonstige, keine, ohne Angabe": "unrecorded",
    # a SECOND SOURCE, not a fourth Zensus column — see the REVIEW note above
    "Jüdische Gemeinde (ZWST-Mitgliedsgemeinde)": "judaism",
    # --- a THIRD source: ESS `rlgdnade`, splitting the residual (sources/de_ess.py).
    # Verbatim ESS labels. The two register answers and `Jüdisch` are not here on
    # purpose — they are measured elsewhere and must never be modelled on top.
    "Muslimisch/Islam": "islam",
    # --- a FOURTH source, dividing that ESS `islam` by Land: BAMF's Muslimisches Leben in
    # Deutschland 2020 (sources/de_mld.py, 2026-10-16). MLD's own answer labels. Still
    # `modelled`, like the ESS rows they are carved from; MLD's other, just-Muslim and
    # don't-know answers stay on `islam`. See the REVIEW note on `MLD 2020: Alevitisch`.
    "MLD 2020: Sunnitisch": "islam.sunni",
    "MLD 2020: Schiitisch": "islam.shia",
    "MLD 2020: Alevitisch": "alevism",
    "MLD 2020: Ahmadi": "islam.ahmadiyya",
    "Östlich-orthodox": "christianity.orthodox",
    "Eine evangelische Freikirche": "christianity.evangelical",
    "Andere protestantische Konfession": "christianity.protestant",
    "Andere christliche Konfession": "christianity",
    "Christlich, aber fühlt sich keiner spezifischen Religionsgemeinschaft zugehörig":
        "christianity",
    "Östliche Religionsgemeinschaft": "other.de",
    "Andere nicht-christliche Religionsgemeinschaft": "other.de",
}


def _key(cat):
    return " ".join(str(cat).split())


# spec §7a-i-1: the Zensus column each derived row was split out of -> the node it names. The
# Landeskirche rows (sources/de_ekd.py) roll back to the undivided Protestant column the register
# counted at the same Gemeinde. The ESS rows are `modelled` and are removed, not rolled.
EV_COLUMN = "Evangelische Kirche (öffentlich-rechtlich)"
COLUMNS = {
    EV_COLUMN: "christianity.protestant",
}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
