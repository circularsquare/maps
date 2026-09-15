# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cn_place_weight(place):
    """countries.py hook. `place` is the 3km hex layer scatter.py has read."""
    return _kontur_place_weight(place, "cn_grid_3km.gpkg", "sources/cn_geo.py")


def _cn_counts():
    """China 2000 census ETHNICITY at county, rescaled onto 2010 provincial totals.

    THE ONLY COUNTRY HERE WHOSE RELIGION IS NOT IN ITS SOURCE AT ALL. China has never
    asked, so what is counted is nationality and what is drawn is spec §14.5's permitted
    derivation: an ethnic category may imply a religion where the category was itself
    constituted religiously, at no finer geography than the ethnicity is published at.
    `taxonomy/cn2000.py` argues each of the 56 nationalities. Ten Muslim, three Tibetan
    Buddhist, one Theravada, six with a fractional Protestant share, and the other
    thirty-six claim nothing. Nothing here is a count of a religious person, and §7a's
    control removes all the colour at once.

    AND SINCE 2026-09-08 THAT IS NO LONGER THE WHOLE COUNTRY. A fourth layer, below, carves
    Han Buddhism and Protestantism out of the grey at province grain from the pooled Chinese
    General Social Survey — `self_id`, the first thing in China drawn from what people said
    about themselves. It is much the largest colour here: **62.4M Mahayana Buddhists and 24.7M
    Protestants**, against the 30.7M the ethnic derivation reaches. China goes from 2.5%
    coloured to 8.8%. See sources/cn_cgss.py, whose docstring carries the whole argument.

    THAT LAYER IS FIVE WAVES SINCE LATER THE SAME DAY, not three. CGSS 2010 and 2013 were
    found openly mirrored in a replication package after CNSDA turned out to gate every
    download behind a reviewed application, taking the pool from 32,495 respondents to 55,637
    and Protestantism from 585 to 1,016. **The point was never the extra dots, which are few.
    It is that Protestantism's provincial ordering had been judged on a SINGLE pair of waves
    returning +0.17; over ten pairs the median is +0.559 and that pair is the worst of them.**
    Hainan gains a survey reading for the first time; Xizang is sampled by 2010 and dropped on
    purpose, because CGSS has no 藏传佛教 answer and would file Tibetan Buddhists as Mahayana
    in the one province §14.5 already draws from ethnicity.

    AND HAINAN HAD 3.34 MILLION PEOPLE IN THE WRONG COUNTY UNTIL LATER THE SAME DAY, spec
    §14.23. The 2000 volume carries eleven of Hainan's county-level units as a name and
    nothing else, so the per-group provincial rescale was giving their people to the ten
    counties that survived: Danzhou drawn at 3,268,523 against a real 932,362, Sanya given
    595,912 Li where the 2000 census counted 183,865. Hainan is now the one province
    reconciled to its own
    published 2010 COUNTY totals, and the eleven are written at theirs on `Unpublished`,
    which claims no nationality because none was published. **The lesson generalises and is
    §14.17's twin**: a margin cannot be enforced against a structure that is missing rows,
    because the enforcement does not fail, it pushes the missing mass into whatever remains.
    Nothing changes colour by it — Li, Han and Miao all resolve to `unknown` — but 229,000
    Buddhists and 290,000 folk-religion dots move off the coast and into the interior, since
    the CGSS layer below carves each unit's own residual and could not reach a unit with no
    rows.

    THE COUNTRY GREW BY 73.5 MILLION PEOPLE THE SAME DAY, and it is spec §14.17 rather than
    anything about religion: 168 county names in the census volumes matched no adcode and
    were being read and then discarded — 5.47% of China, 142 of them urban districts. The
    join is repaired in sources/cn.py, every province now reconciles to its 2010 total, and
    the national figure lands 17 people from the published census. cn.csv went from 2,654
    units to 2,768 and from 1,259,316,206 people to 1,332,810,852.

    91.7% OF THE COUNTRY IS ONE GREY NODE, AND THAT IS THE POINT RATHER THAN A DEFECT.
    Until 2026-09-07 only 2.2% of China was drawn at all and the eastern half of the map
    was blank; spec §6.12's coverage wash was the only thing standing between that blank
    and a reader concluding nobody in eastern China believes anything. §14.7 decided to
    fill it and §14.25 unblocked it. The 1.14 billion Han are now on `unknown` — counted,
    located, and nothing claimed — because Han religion is §14.5's religiously-mixed row
    and the folk-religion / irreligious boundary is mostly an artefact of how the question
    is asked. **Refusing to draw that boundary is the decision; showing the people is not
    the same as guessing about them.**

    THE MONGOLS ARE NOT DRAWN, AND IT IS THE LARGEST SINGLE CALL IN THE COUNTRY. spec §12
    and §14.5 both send Mongol to Tibetan Buddhism; at 5.81M in 2000 that is more people
    than Tibetans, so Inner Mongolia rather than Tibet would have been the largest block
    of Vajrayana dots in China. §14.5 requires the coefficient to be "documented rather
    than fitted" and for Mongols there is nothing to document — Anita's call, 2026-09-05.
    Tu (241k) goes with them for the same reason. See taxonomy/cn2000.py.

    THE GEOGRAPHY IS 2000 AND THE MAGNITUDES ARE 2010, per spec §3.4. No county-level
    ethnic table newer than 2000 is in the open; the NBS publishes the 2020 provincial
    one as a JPEG scan and the 2010 one as HTML. So each group's county figures are
    scaled to its 2010 provincial total, which moves magnitude and not shape.

    ---- REWRITTEN 2026-09-07, spec §14.25, and the country went from 2.2% to 100% ----

    EVERYBODY IS DRAWN NOW, AT THREE STRENGTHS OF CLAIM, AND ONE SOURCE ROW CAN BECOME
    TWO. `cn2000.shares()` returns a list of (node, share, tier) per nationality instead
    of a single node, so this is a fan-out rather than a `.map()`:

      * 15 nationalities -> `islam` / `buddhism.vajrayana` / `buddhism.theravada` at 1.0,
        tier `derived`. Unchanged, and still spec §14.5's derivation.
      * 6 nationalities -> `christianity.protestant` at a fractional share with the
        remainder on `unknown`, BOTH tier `modelled`. The southwestern mission peoples,
        permitted by §14.9 and applied by us under §14.10; the coefficients and the three
        conditions they had to meet are argued in cn2000.py.
      * everyone else, 1.21 billion people -> `unknown` at 1.0, tier `derived`. Nothing
        is claimed about them; the row says only that the census counted them and where.

    WHY `unknown` IS `derived` AND NOT `measured`, WHICH IS THE ONE SUBTLE CALL HERE.
    Nobody's religion was measured, but that is not what the tier is about — the row makes
    no religious claim at all, so on the religion question it could arguably be anything.
    What settles it is §3.4: the county figure is a 2000 count carried onto a 2010
    provincial total, and §7's table puts "somebody counted this and the number was
    carried to a finer place" squarely in `derived`. Brazil is 41.2% derived for exactly
    this reason. So China stays 100% not-measured and `inferred dots: hidden` still empties
    it — which §14.6 called the honest test of the country and it survives the rewrite.
    """
    from cn2000 import shares

    df = pd.read_csv(HERE / "data" / "normalized" / "cn.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[(df["geo_level"] == "county") & (df["count"] > 0)]

    parts = []
    for cat, sub in df.groupby("source_category", sort=False):
        for node, share, tier in shares(cat):
            if share <= 0:
                continue
            parts.append(pd.DataFrame({
                "unit": sub["geo_id"].to_numpy(),
                "node": node,
                "count": sub["count"].to_numpy(dtype=float) * share,
                "tier": tier,
            }))
    out = pd.concat(parts, ignore_index=True)

    # ---- the CGSS self-id layer, sources/cn_cgss.py -------------------------------
    #
    # THE ONLY THING IN CHINA DRAWN FROM WHAT SOMEBODY SAID ABOUT THEMSELVES. Everything
    # above is spec §14.5's derivation from the nationality column — a claim the map makes
    # about people. This is the pooled Chinese General Social Survey (2010 + 2012 + 2013 +
    # 2017 + 2021, n = 55,637, 30 of 31 provinces), which asks 您的宗教信仰是什么
    # and is therefore §3.1's `self_id`, the same basis as Vietnam's census and Russia's Arena.
    #
    # ONLY THE `unknown` RESIDUAL IS CARVED, AND THAT IS THE WHOLE OF THE ARITHMETIC. If the
    # provincial share were applied to a province's entire population, Qinghai and Gansu would
    # count their Tibetans twice — once as `buddhism.vajrayana` from ethnicity and again as
    # generic 佛教 here. So the share eats the grey and nothing else. The cost runs the other
    # way and is named in note_public: in the four provinces where the derived population is
    # large, some of CGSS's Buddhist respondents WERE those people, so re-spreading over the
    # residual alone runs a little high there. Everywhere else the derived share is under 2%.
    #
    # TWO CATEGORIES OF SIX SURVIVE §14.10. Buddhism passes cleanly (χ² p = 4e-184; Zhejiang
    # 15.7% CI 14.0–17.5 against Anhui 0.9% CI 0.4–1.4) and Protestantism is drawn on Anita's
    # call with its weakness disclosed — its spatial variation is highly significant but its
    # 2012↔2021 rank stability is only +0.17. Islam is NOT taken from CGSS: its provincial
    # subsamples find 0.07x Qinghai's Muslims and 2.7x Ningxia's, which is a lottery over
    # sampling units rather than a bias a weight could fix, and §14.5's county derivation is
    # better. Nationally the two agree (CGSS 1.87–2.56%, derivation 1.83%) and that is the
    # first external check §14.5 has ever had. folk, Daoism and Catholicism are too thin.
    #
    # THE LEVEL IS THE POOLED ONE BECAUSE OF THE DENOMINATOR, NOT BECAUSE OF THE SAMPLE SIZE.
    # cn.csv is 2000 structure on 2010 provincial totals, so the people being coloured are the
    # 2010 census's people; pooled and n-weighted, CGSS centres on about 2015, where the 2021
    # wave alone is eleven years downstream of its own denominator. Reported religiosity fell
    # steadily across the waves (any religion 14.5% → 10.6% → 7.5%), so this layer is about
    # half again larger than 2021 alone would draw and note_public says which way it leans.
    cgss = pd.read_csv(HERE / "data" / "normalized" / "cn_cgss.csv")
    prov = df.assign(_p=df["note"].str.extract(r"province=([^;]+)")[0]) \
             .drop_duplicates("geo_id").set_index("geo_id")["_p"]
    piv = cgss.pivot(index="province", columns="node", values="share")

    unk = out["node"] == "unknown"
    province = out["unit"].map(prov)
    carve = {}
    total_share = pd.Series(0.0, index=out.index)
    for node in piv.columns:
        s = province.map(piv[node]).fillna(0.0).where(unk, 0.0)
        carve[node] = s
        total_share = total_share + s
    if (total_share > 1).any():
        raise ValueError("cn_cgss shares sum past 1 in some province")

    carved = [out.assign(count=out["count"] * (1.0 - total_share))]
    for node, s in carve.items():
        part = out.loc[s > 0].copy()
        part["count"] = part["count"] * s[s > 0]
        part["node"] = node
        part["tier"] = "modelled"      # the coefficient is CGSS's, not the census's
        carved.append(part)
    out = pd.concat(carved, ignore_index=True)

    # Several nationalities land on `unknown` in the same county — Han, Miao, Manchu and
    # the non-Christian remainder of the Lisu are four rows saying the same thing. scatter.py
    # would group them anyway; doing it here takes ~60,000 rows to ~10,000 and keeps the
    # `modelled` remainder separate from the `derived` one, which is §7's "the tier keys
    # the row, it does not aggregate over it".
    out = out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    out = out[out["count"] > 0]
    out["congregations"] = 0
    return out[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "cn": dict(
        name="China",
        source=("2000 census nationality by county (NBS) on 2010 provincial totals; "
                "religion shares from the Chinese General Social Survey, "
                "2010+2012+2013+2017+2021"),
        basis=("ethnicity, derived — no census asked about religion; plus self-identified "
               "religion from a pooled national survey, at province"),
        view=[73.0, 17.5, 135.5, 54.0],
        note_public=(
            "**China has never asked anybody on this map what their religion is.** No "
            "Chinese census has carried a religion question, so unlike every other "
            "country here there is no answer to draw. What the state does count, and "
            "publish by county, is **nationality** — and for a small number of China's 56 "
            "nationalities the ethnic category and a religion are the same historical "
            "object rather than two correlated ones. A Hui person is *defined* as a "
            "Chinese-speaking Muslim, with no separate language or territory; the Turkic "
            "and Iranian peoples of the northwest were Muslim before any of them was a "
            "census category; Tibetan Buddhism is what the word Tibetan carries. Those are "
            "what is drawn here, and nothing else is. "
            "**The second thing drawn here is what people say about themselves when a "
            "survey asks, and it is deliberately a small number.** China's censuses do not "
            "ask, but its main academic social survey does — *which religion do you belong "
            "to* — and pooling five waves of it gives about 55,000 answers across 30 of "
            "the 31 provinces. Roughly **one person in eight** is drawn with a religion "
            "here. Those are the Buddhist, folk-religion and Protestant dots across "
            "eastern China: Buddhism heaviest in Zhejiang, Fujian and Jiangxi at ten to "
            "fifteen percent, Protestantism heaviest "
            "in Henan, which is the province usually described as China's Christian "
            "heartland, and in the northeast. "
            "**The largest thing here that is not an imported religion is the folk one.** "
            "The survey offers *popular belief, worshipping Mazu or Guandi and the like* as "
            "an answer of its own, and **41 million** people are drawn on it: the sea "
            "goddess of the Fujian coast, the deified general, the lineage hall and the "
            "earth-god shrine, which is what most religious practice in China has always "
            "looked like and which no census anywhere counts. It is overwhelmingly southern "
            "and coastal, around **18%** in Guangdong and Fujian against under one "
            "percent across most of the north. "
            "**These are the least certain dots on this map, and it is worth saying exactly "
            "why.** Every other layer here counts people who gave a religion a name, and so "
            "does this one, but folk practice is the thing people are least likely to call a "
            "religion. Asked to tick everything that applies, about three in a hundred "
            "Chinese name folk belief; asked in 2021 to choose one single religion, **fewer "
            "than three in a thousand** did. That is a **sixteenfold** swing on question "
            "wording alone, where Buddhism moves by half and Protestantism by four fifths. "
            "All five waves are pooled here rather than the ones that flatter the layer, so "
            "the number sits in the middle of that range and is a floor rather than a count. "
            "**The gap it cannot show is much larger than the layer itself.** In the one "
            "survey that asks both questions of the same people, the Spiritual Life Study of "
            "2007, **16%** of Chinese said they had a religious belief while only **38%** "
            "said they never worshipped a god, spirit or ancestor. So roughly four times as "
            "many people tend graves, burn incense and visit temples as will call any of it "
            "their religion. The dots are the naming, not the doing. "
            "Two smaller things: sixteen of the thirty provinces have fewer than ten people "
            "giving this answer, so the southern block is firm and the thin northern shading "
            "is a direction rather than a quantity; and about one in twelve of these people "
            "also named Buddhism, so a few of them are drawn twice. "
            "**Daoism and Confucianism are not drawn, and the reasons differ.** Only about "
            "one person in three hundred names Daoism when asked, which is far below the "
            "figures usually quoted for it, because those count practice and temple-going "
            "rather than what people call themselves. Confucianism is not drawn because the "
            "survey does not offer it as an answer at all. "
            "**The other seven in eight are still grey, and that is a choice rather than "
            "a finding.** They are drawn as *Religion unknown* — counted by the census, "
            "placed where it puts them, nothing claimed about what they believe. **They are "
            "emphatically not drawn as irreligious**, because in China the two halves of "
            "that survey question are not equally trustworthy. Ask people to name a "
            "religion and about 88% name none; ask instead whether they tend graves, visit "
            "temples or believe in deities and most of it comes back. Pew's *Measuring "
            "Religion in China* puts Buddhism alone at 4% by self-identification and 33% by "
            "belief, from the same two surveys in the same year. **So the answer *yes, I am "
            "a Buddhist* is a measurement and the answer *none* is mostly an artefact of "
            "the wording.** This map draws the first and leaves the second grey. Daoism, "
            "most of a Christian population usually estimated in the tens of millions, and "
            "a great deal more folk practice than the 41 million who name it are all still "
            "inside that grey. "
            "**And the number that is drawn is falling, which may not be about belief.** "
            "Across the five survey waves used here, running from 2010 to 2021, the share "
            "naming any religion at all fell from a peak of 14.5% in 2012 to **7.5%** in "
            "2021, in every category at once, "
            "including Islam in a population whose Muslim nationalities were growing. Some "
            "of that is likely a real change in what people are willing to tell an "
            "interviewer. This map pools the waves, so it sits nearer the middle of that "
            "range than the end of it. "
            "**The Christian dots in the northeast are the newest thing here and the least "
            "certain.** About 549,000 of China's 1.8 million ethnic Koreans are drawn as "
            "Protestant, densest in Yanbian on the North Korean border, where they are half "
            "the population of Yanji and Longjing. Korean-Chinese churches are real and well "
            "documented — but the share used, 30%, comes from a missionary dataset and lands "
            "within two points of South Korea's own Christian share, so it may be that "
            "country's figure carried across the border rather than a measurement of this "
            "one. If so it is wrong twice, because a third of South Korea's Christians are "
            "Catholic and these dots are all Protestant. Read the northeast as *there is a "
            "substantial Korean Christian population here*, and not as a count. "
            "**Arunachal Pradesh is drawn as India's**, which is where its people were "
            "counted. China claims it and its own county boundaries cover it, so until "
            "2026-09-08 this map put about 197,000 Chinese-counted people on the Indian side "
            "of the line. Aksai Chin goes the other way, to China, on the same rule: "
            "whoever administers a place is who counted the people in it. "
            "**Every coloured dot here is an inference, and the `inferred dots` control "
            "removes all of them.** That is the honest test of this country: turn it on and "
            "China loses its colour entirely, because nothing in it was counted as religion "
            "by anybody. "
            "**The Mongols are deliberately absent too**, and they are the biggest "
            "judgement call in the country. Tibetan Buddhism among Mongols is real history, "
            "but at 5.8 million people they would have outnumbered Tibetans and made Inner "
            "Mongolia the largest Buddhist region in China on the strength of an assumption "
            "nothing measures — decades after the monastic system they would have been "
            "counted through was dismantled. The same reasoning leaves out the Tu, and it "
            "is the reason to trust what remains. "
            "**What the map does show is real and is not obvious.** Islam in China is not "
            "only Xinjiang: the Hui live in every province, so the Muslim layer runs from "
            "Kashgar to Kaifeng and down to a single village cluster in Sanya on Hainan. "
            "Linxia in Gansu and the Ningxia countryside are as densely Muslim as anywhere "
            "in the northwest. Tibetan Buddhism reaches far outside Tibet, across western "
            "Sichuan, Qinghai and southern Gansu. And China's only Theravada population is "
            "the Dai of Xishuangbanna and Dehong, whose monasteries belong to the "
            "Southeast Asian world rather than the Chinese one. "
            "**If you want to know which single layer here to trust least, it is "
            "Protestantism.** Where it is heaviest, in Henan, Heilongjiang, Zhejiang, "
            "Jiangsu and Jilin, is well attested outside this survey, and the survey picks "
            "Henan out unaided at **6.8%**. The five waves broadly agree with each other "
            "about the order of the provinces, so the shape of this layer is better "
            "supported than its size; the one wave that disagrees is 2021, which reached 19 "
            "provinces rather than 30 and counted the fewest believers of any of them. What "
            "stays genuinely uncertain is how many. The share naming Protestantism fell from "
            "2.3% in 2012 to 1.0% in 2021, and some of that is likely a change in what "
            "people were willing to tell an interviewer rather than a change in belief, so "
            "read these dots as a good guide to where Chinese Protestantism is and a rough "
            "one to how much of it there is. Buddhism is on firmer ground on both counts, "
            "and the coastal southeast really is the Buddhist part of China. "
            "**The Protestant dots in the far southwest are built differently from the rest "
            "and are the least certain thing here.** Six peoples of the Yunnan border — the "
            "Lisu, Lahu, Jingpo, Wa, Nu and "
            "Derung — were reached by Protestant missions between about 1900 and 1935 and "
            "large parts of them have been Christian ever since. Nujiang, the Lisu "
            "prefecture on the Myanmar border, contains what is usually described as the "
            "first majority-Christian county in China. Nobody counts them: the shares used "
            "here are Joshua Project's, a missionary organisation's estimates, applied to "
            "the census's own count of each nationality. **They probably run high** — for "
            "the Lisu, Joshua Project says 80% where the figure usually reported as the "
            "official one works out to about 43% — so read this layer as *where*, "
            "confidently, and *how many*, within a factor of about two. "
            "**And half a million Christians are missing from it, in a place this map "
            "cannot put them.** The A-Hmao and Gha-Mu of northwestern Guizhou, converted by "
            "the Pollard mission in the 1900s, are as Christian as the Lisu — but the "
            "census counts them as *Miao* along with nine million other people spread over "
            "five provinces, and there is no way to pull them out. They are in the grey. "
            "**One part of China is drawn from a different table, and it is the middle of "
            "Hainan.** The 2000 census volume for that province is incomplete: eleven of "
            "its counties, the Li ones across the centre and west of the island, are in it "
            "as a name and nothing else. Their 3.3 million people are drawn here from their "
            "own published county populations instead, so this map has how many of them "
            "there are and where they live but nothing at all about nationality for them. "
            "Almost nobody there belongs to a nationality this map draws a religion from, so "
            "little is lost by it — but until 2026-09-08 those eleven counties were empty and "
            "their people were being drawn in Danzhou and Sanya instead, which made Danzhou "
            "three and a half times its real size. "
            "**The vintage is split.** Where people are comes from the 2000 census, the "
            "last one whose county-level ethnic tables are public; how many there are comes "
            "from 2010. Twenty-five years is long enough that the cities have grown and "
            "these dots do not know it, and the migrant communities of the coast — a few "
            "thousand Uyghurs and Dai in Zhejiang and Shanghai — are placed by a pattern "
            "that predates them."),
        how=("no census question; ethnicity for the minorities, a pooled survey by province for "
             "everyone else"),
        fill="from the 2000 census's ethnicity table",
        grain=("counties, 478,000 people on average; the survey layer's shares are provincial, so "
               "Buddhism and Protestantism vary between provinces and not within them"),
        # `gap` (§6.12). It said "Han majority not shown" until §14.14 drew them, then
        # "97% of these dots say only that somebody was counted" until the CGSS layer of
        # 2026-09-08 took the grey from 97.5% to 91.6%. The failure mode it guards against
        # has never moved: this is still the country where a reader is most likely to read
        # the grey as irreligion, which is exactly what it is not.
        gap=("a religion for 88% of these dots; with no census question they say only that "
             "somebody was counted"),
        counts=_cn_counts,
        # Counts are on the GB/T 2260 county adcode; the Kontur hexes carry no adcode, so
        # sources/cn_geo.py assigns and clips every hex and writes the `unit` this reads.
        # Russia's, Kenya's, Serbia's and Lithuania's wiring exactly.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cn" / "cn_grid_3km.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cn_place_weight,
        note="**100% of the census population is drawn and 88.1% of it is `unknown`** — "
             "rewritten 2026-09-07, spec §14.25, after CFPS refused the data access §14.7 "
             "had planned a Han Buddhist share around. Nothing is measured: the derived "
             "rows are §14.5's ethnic derivation and the modelled rows are §14.9's "
             "fractional Christian share over six Yunnan border peoples, whose "
             "coefficients are Joshua Project's and whose three admission conditions are "
             "argued in taxonomy/cn2000.py. **The gate is NOT a threshold over Joshua "
             "Project's own numbers** — that was tried, and it returns 121 million Han "
             "Christians in the Wu- and Min-speaking southeast against CGSS's 1.7% "
             "nationally, because an interested source's bias is not uniform across its "
             "own rows. Selection is made on evidence outside the missionary literature and "
             "only the coefficient comes from JP. "
             "The output check, which is all this country has: the six peoples come to "
             "~876,000 Christians, essentially all in Yunnan, against a province usually "
             "reported to hold over a million Protestants — consistent, and leaving room "
             "for the A-Hmao Miao and Han that this map cannot place. "
             "Structure is the 2000 census at "
             "county (Harvard's `chinacensus` dataverse, CC0, 2,859 counties — the table "
             "reaches township and that is deliberately NOT used, per §14.5's ceiling); "
             "magnitudes are the 2010 census by province (NBS table 1-6), per §3.4. The "
             "2020 edition of that table exists only as a JPEG scan. "
             "**The county join is by name and it is the fragile part**: the census carries "
             "romanised names and no codes, DataV carries Chinese names and the adcode, and "
             "**all 2,859 counties now resolve** — 2,546 by name, 28 by code order (both "
             "lists are in GB/T 2260 order, which is the only thing separating Yining city "
             "from Yining county), 285 by a hand table of administrative changes. It was "
             "2,691 of 2,859 until spec §14.17, and the 168 that failed were eastern urban "
             "districts holding 73.5M people that were read and then discarded. Nothing is "
             "stranded now. "
             "**HAINAN IS RECONCILED TO COUNTY TOTALS AND IS THE ONLY PROVINCE THAT IS** — "
             "spec §14.23, 2026-09-08. Its volume carries eleven of its 24 county-level "
             "units as a name and a tab and nothing else (all 111 tables of the Dataverse "
             "dataset do), so the per-group provincial rescale was handing their 3.34M "
             "people to the ten counties that survived: Danzhou drawn at 3,268,523 against "
             "a real 932,362, Sanya given 595,912 Li where the 2000 census counted 183,865, every county's "
             "Han inflated 1.841x and its Li 3.241x. **A margin cannot be enforced against "
             "a structure that is missing rows — the enforcement does not fail, it pushes "
             "the missing mass into whatever remains**, and the file-sum denominator that "
             "guards §14.17's case is mute about this one. The ten present units now keep "
             "their 2000 nationality shares and scale by their own county_2010/county_2000; "
             "the eleven are written at their published 2010 total on `Unpublished`, which "
             "claims no nationality because none was published. Hainan's per-nationality "
             "provincial reconciliation is reported as a residual rather than enforced, and "
             "it costs a label rather than a dot: Li, Han and Miao all resolve to `unknown` "
             "anyway, the Hui reconcile to within 200, and the whole claiming residual is "
             "Kazakh 1,535 and Dai 779, both §6 migration cases the old code was scaling "
             "110x. 8,671,074 of a published 8,671,518 is drawn; the 444 are "
             "西南中沙群岛. "
             "**geoBoundaries CHN ADM2 was tried and rejected** — duplicated polygons, "
             "counties abolished in the 1980s, units in the wrong province, corrupted "
             "romanisation, 59.9% match. sources/cn_geo.py has the evidence. "
             "Placement is Kontur's 3km H3 grid, 167,890 hexes; its totals reproduce the "
             "2010 census at 1.063x nationally with a per-province median of 1.048 and "
             "everything inside a factor of two, which is also what says the DataV polygons "
             "are WGS84 rather than GCJ-02 offset.",
    ),
}
