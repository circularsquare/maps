"""China 2000 census ETHNICITY -> religiondots taxonomy. spec §14.5, §14.9 and §14.14.

**THIS FILE IS NOT LIKE THE OTHERS AND THE DIFFERENCE IS THE WHOLE POINT.** Every other
mapping in this directory takes a religion category a source published and finds it a home
on the tree. This one takes an ETHNIC category and asserts something about religion for it.
China has never asked about religion in a census; what is counted is nationality, by the
state, published by the state at county level.

**EVERY PERSON IN THE FILE IS NOW DRAWN, AND THEY ARE DRAWN AT THREE DIFFERENT STRENGTHS
OF CLAIM.** That is the change of 2026-09-07 (spec §14.13) and it is what took China from
2.2% of its own population to 100% of it.

| | categories | people | node | tier |
|---|---|---|---|---|
| **religio-ethnic**, spec §14.5 | 14 | ~30.6M | `islam`, `buddhism.vajrayana`, `buddhism.theravada` | `derived` |
| **mission peoples**, spec §14.9 | 6 | ~0.88M of 1.81M | `christianity.protestant` + remainder | `modelled` |
| **everyone else** | 41 | ~1.21 billion | `unknown` | `derived` |

**AND THIS FILE IS NO LONGER THE WHOLE OF CHINA.** From 2026-09-08 a fourth layer is applied
downstream of it, in `countries.py::_cn_counts` from `sources/cn_cgss.py`: Han Buddhism and
Protestantism, carved out of the `unknown` residual at province grain from the pooled Chinese
General Social Survey. That layer is `self_id` — the first thing in this country drawn from
something people said about themselves rather than derived from their nationality — and it is
much the largest colour on the Chinese map. Nothing in THIS file changed to accommodate it;
the residual it eats is the same residual §14.14 built.

**`unknown` asserts nothing and that is why it is allowed to hold a billion people.** It is
§6.3a-ii's node — the source counted these people and did not establish what they practise.
It is not `unaffiliated`, which would be the largest false claim available on this map, and
it is not `unrecorded`, which is Germany's register-shaped absence. The one thing the Chinese
census is excellent at is counting people and saying where they are; that is the whole of
what the grey says.

**Nothing here is a count of a religious person.** The map's §7a control strips the derived
and modelled rows out in one click, and after 2026-09-07 doing so leaves the grey standing —
China does not empty, it decolours, which is the more honest demonstration of the same fact.

MAP / MISSION hold what is asserted. NOT_ASSERTED holds the categories that go to `unknown`,
with the reason. REVIEW holds calls that are defensible but arguable. EXCLUDED holds the one
row that is not a category at all.
"""

# --- 1. the religio-ethnic derivations, spec §14.5 --------------------------------

# Muslim by descent. The ten nationalities whose formation as categories is inseparable
# from Islam: the Hui are *defined* as Chinese-speaking Muslims and have no other
# distinguishing marker, and the nine Turkic, Iranian and Mongolic groups of the northwest
# were Muslim before any of them was a census category. spec §14.5's coefficient is nearest
# to 1 here of anywhere in the file.
_MUSLIM = ["Hui", "Uyghur", "Kazakh", "Dongxiang", "Salar", "Kyrgyz", "Tajik", "Uzbek",
           "Bonan", "Tatar"]

# Tibetan Buddhist by descent, and a much shorter list than spec §12 proposed -- see
# REVIEW for Mongol and Tu, which are Anita's call of 2026-09-05 and are NOT drawn, and for
# PUMI, which was drawn until 2026-09-08 and is Anita's call too.
_VAJRAYANA = ["Tibetan", "Yugur", "Monba"]

# Theravada by descent. The Dai of Xishuangbanna and Dehong are the northern edge of the
# mainland Southeast Asian Theravada world, not an outpost of Chinese Buddhism.
_THERAVADA = ["Dai"]

MAP = {}
MAP.update({g: "islam" for g in _MUSLIM})
MAP.update({g: "buddhism.vajrayana" for g in _VAJRAYANA})
MAP.update({g: "buddhism.theravada" for g in _THERAVADA})


# --- 2. the southwestern mission peoples, spec §14.9 ------------------------------
#
# BUILT 2026-09-07. spec §14.8 parked these because §14.5's permission ("the ethnic
# category was itself CONSTITUTED religiously") did not reach them -- Lisu identity
# predates the Fraser mission by centuries -- while its ban ("never where the group is
# religiously mixed") did not reach them either, since they are not mixed so much as
# overwhelmingly Christian. §14.9 withdrew the ban, made a fractional share over a mixed
# category legal, and said in its last bullet that consistency therefore permits these too.
# §14.10 permits the map to apply the shares itself. This is that, and it is the FIRST
# TIME `christianity` reaches China.
#
# THE COEFFICIENTS ARE JOSHUA PROJECT'S AND THAT IS A CHOICE WITH A COST.
# JP publishes `PercentAdherents` -- Christian adherents as a share of the group -- for
# every people-group in every country, free, no key, at
# joshuaproject.net/resources/datasets/1. It is an evangelical missions organisation and
# therefore the interested party; §14.9 accepted exactly that trade for Spain's UCIDE, and
# there is no disinterested source that covers all six of these peoples on one basis.
# **It probably runs high** -- see the Lisu cross-check in REVIEW.
#
# THE SHARE IS PER NATIONALITY, NOT PER PEOPLE-GROUP, AND THE DIFFERENCE IS THE WHOLE
# METHOD. The census names a nationality; JP names a people-group; the Chinese state packs
# several of the second into each of the first. So each figure below is the
# population-weighted mean of JP's shares over the groups the state classifies under that
# nationality -- which is why Jingpo comes out at 24% and not at Kachin Jingpo's 54%: the
# Zaiwa are the larger half of the nationality and JP puts them at 0.25%.
#
# ***A THRESHOLD OVER JP'S OWN NUMBERS IS NOT A RULE, AND FINDING THAT OUT IS THE MOST
# USEFUL THING IN THIS SECTION.*** The first draft of this gate was mechanical -- draw any
# nationality whose weighted share clears 10% -- on the reasoning that a stated threshold
# beats a hand-picked list (which is how the Lahu were found, below). Applied to all 546 of
# JP's China groups it returns, at the top of the list:
#
#       Han Chinese, Wu        80,857,000   13.4%
#       Han Chinese, Min Nan   22,446,000   10.0%
#       Han Chinese, Min Dong  10,143,000   10.0%
#       Han Chinese, Min Bei    8,198,000   10.0%
#
# **121 million Han Christians in the southeast**, against CGSS 2021's 1.5% Protestant and
# 0.2% Catholic nationally. JP's shares are not one quantity measured consistently: for a
# small evangelised minority they are close to a church's count of its own members, and for
# the Han they are a national estimate spread over dialect groups. **An interested source's
# bias is not uniform across its own rows**, so a threshold over it inherits the bias
# instead of controlling for it. spec §14.12 in a new costume.
#
# SO THE GATE IS THREE CONDITIONS AND ONLY THE COEFFICIENT COMES FROM JP.
#   (a) SELECTION is made on evidence OUTSIDE the missionary literature: the nationality
#       is one whose Christianity is attested as its predominant or major pattern by
#       Chinese state and academic sources -- Nujiang's "Christian county" reporting,
#       Yunnan's >1M Protestants, the recognised minority-language churches. This is
#       §14.8's own proposed clean form, *"an independently measured share ... from a
#       source with no stake in the answer"*, used to CHOOSE the groups rather than to
#       number them.
#   (b) the JP groups under the nationality are CO-LOCATED, so that spreading one share
#       over the nationality's counties does not move anybody. This is spec §14.3's rule
#       -- never model at a finer resolution than the source publishes its magnitude at --
#       and it is checkable from JP's own centroids, which is the `gap` column below.
#   (c) the COEFFICIENT is JP's population-weighted share, because nothing else covers
#       these six on one basis. It is marked `modelled` and note_public says which way it
#       leans.
#
# Condition (b) is what keeps the MIAO out, and they would have been the largest addition
# here: A-Hmao (448,000 at 80%) and Gha-Mu (142,000 at 80%) are the Pollard mission's
# harvest and are as Christian as the Lisu, but they are 6% of a 9.4-million nationality
# spread over five provinces, and their centroid is **427 km** from that of the Northern
# Hmu, who are 2.1 million at 0.3%. The census column says `Miao` and cannot tell them
# apart, so a nationality-wide share would put half a million Christians into Hunan and
# eastern Guizhou where there are none. Same argument excludes the Yi, whose Christian
# Lipo, Naluo and Laka are ~2% of 8.7 million. **Both are drawn as `unknown` instead**,
# which is the honest answer: they are there, and this source cannot place them.
#
# THE KOREANS ARE THE ONE ARGUABLE OMISSION AND ARE FLAGGED FOR ANITA -- see REVIEW.
# JP puts them at 30%, they are co-located, and 1.83M x 30% would be ~549,000 people,
# comparable to the Lisu. They fail (a) rather than (b) or (c), and that is a judgement.
#
# Computed 2026-09-07 from the JP PGIC file; the group lists are ethnography, not a field
# in it. Every one of these rows is `modelled` -- BOTH of them, because the split between
# the Christian share and the remainder is what the coefficient decides.
MISSION = {
    # nationality: (share, [Joshua Project groups], census 2010, centroid gap km)
    # ADDED 2026-09-08, Anita's call, and it closes the flag REVIEW has carried since
    # 2026-09-07 -- see REVIEW["Korean"] for the argument on both sides. 1.83M people,
    # concentrated in Yanbian and spread through Jilin, Heilongjiang and Liaoning; JP has
    # them as ONE group, so condition (b) is trivial. It is the only row here whose
    # condition (a) rests on attestation outside the missionary literature that is
    # ANALOGICAL rather than local -- the Yanbian church is well documented in Chinese and
    # Korean academic work, but the 30% itself is JP's and sits suspiciously close to South
    # Korea's own self-identified Christian share (~28% in its 2015 census). Drawn with
    # that stated, not hidden.
    "Korean": (0.3000, ["Korean 30%"], 1_830_929, 0),
    "Lisu":   (0.7972, ["Lisu 80%", "Lemo 0%"], 702_839, 1),
    "Lahu":   (0.4195, ["Lahu 55%", "Lahu Shi 10%"], 485_966, 23),
    "Derung": (0.2806, ["Drung 25%", "Rawang 60%"], 6_930, 5),
    "Jingpo": (0.2390, ["Kachin Jingpo 54%", "Zaiwa 0.25%", "Maru 79%",
                        "Lashi 35%"], 147_828, 139),
    "Nu":     (0.2315, ["Nu 17%", "Ayi/Anong 59%", "Zauzou 5%"], 37_523, 12),
    "Va":     (0.1533, ["Wa Parauk 19%", "Wa Vo 0.2%"], 429_709, 13),
}

MISSION_NODE = "christianity.protestant"

# ~876,000 people over the six, against 1.81M in the nationalities themselves. See REVIEW
# for the check on that number, which is the only one this country has.


# --- 3. everyone else: counted, and nothing claimed about them --------------------

UNKNOWN_NODE = "unknown"

NOT_ASSERTED = {
    "Han":
        "1.14 BILLION PEOPLE, AND UNTIL 2026-09-07 THE LARGEST DELIBERATE ABSENCE ON THIS "
        "MAP. They are now drawn, on `unknown`, and NOTHING about their religion is "
        "claimed -- spec §14.7 decided that and §14.13 unblocked it when CFPS refused the "
        "data access that was to have carved a Buddhist share out first. "
        "**The reason no share is carved is unchanged and is not the refusal.** Han "
        "religion is spec §14.5's religiously-mixed row, and the folk-religion / "
        "irreligious boundary is not a boundary in the world so much as an artefact of the "
        "question: ask Chinese respondents to name a religion and about 92% name none (CGSS "
        "2021), ask instead about ancestor rites, temple visits and belief in deities and "
        "most of it comes back -- Pew's *Measuring Religion in China* puts Buddhism alone at "
        "4% by self-identification and 33% by belief, from the same two instruments. A dot "
        "map cannot hold both answers and spec §3.1 says pick a basis. `chinesefolk` is "
        "still the node waiting for them if that ever changes.",
    "Unidentified":
        "734,438 people the census could not assign to any of the 56 nationalities, "
        "overwhelmingly in Guizhou. No religion follows from 'not classified' -- and none "
        "is claimed, which is why they can now be drawn rather than dropped.",
    "Naturalised":
        "941 foreign nationals who took Chinese citizenship. Too few to say anything "
        "about, and nothing about their religion follows from the category.",
    "Unpublished":
        "3,336,751 PEOPLE IN ELEVEN HAINAN COUNTIES, AND THE ONE CATEGORY HERE THAT IS NOT "
        "A NATIONALITY -- added 2026-09-08, spec §14.23. The Harvard digitisation of the "
        "Hainan volume carries 澄迈, 临高, 定安, 屯昌, 东方, 乐东, 陵水, 昌江, 白沙, 琼中 "
        "and 保亭 as a name and a tab and nothing else, in all 111 of its tables, so for "
        "these eleven counties there is a published population and no nationality at all. "
        "`sources/cn.py` writes them at their own 2010 county total under this name rather "
        "than guessing a composition. **Nothing is lost by it.** Li, Han, Miao and Zhuang -- "
        "which is essentially everyone there -- all resolve to `unknown` anyway, and every "
        "one of Hainan's ~13,600 religio-ethnic people is in a county the volume does cover. "
        "So this row draws the same colour a full nationality table would have drawn, and "
        "the alternative was 3.34 million people in the wrong county.",
}

# The nationalities for which no religion is asserted: religiously mixed, religiously
# indigenous, or religiously indistinguishable from their Han neighbours. Before
# 2026-09-07 this list was an EXCLUSION and these people were not on the map at all;
# they are now drawn as `unknown`, which claims exactly as little and shows them.
_NOT_RELIGIO_ETHNIC = [
    "Miao", "Yi", "Zhuang", "Bouyei", "Manchu", "Dong", "Yao", "Bai", "Tujia",
    "Hani", "Li", "She", "Gaoshan", "Sui", "Naxi", "Daur", "Mulao", "Qiang", "Blang",
    "Maonan", "Gelao", "Xibe", "Achang", "Russian", "Evenk", "De'ang", "Gin", "Oroqen",
    "Hezhen", "Lhoba", "Jino", "Mongol", "Tu",
]
for _g in _NOT_RELIGIO_ETHNIC:
    NOT_ASSERTED.setdefault(
        _g, "not a religio-ethnic category: religiously mixed, indigenous, or "
            "indistinguishable from the surrounding population. spec §14.5's derivation "
            "is not available here, so these people are drawn on `unknown` and no "
            "religion is named for them.")

NOT_ASSERTED["Miao"] = (
    "9.43M people, and THE LARGEST THING THIS FILE DECLINES TO DRAW IN COLOUR. The "
    "A-Hmao (Big Flowery Miao, 448,000) and Gha-Mu (142,000) of northwestern Guizhou and "
    "northeastern Yunnan are the Pollard mission's harvest and Joshua Project puts both at "
    "80% Christian -- as Christian as the Lisu, and about half a million people. They are "
    "not drawn because the census column says `Miao` and cannot tell them from the 2.1M "
    "Northern Hmu at 0.3%, whose centroid is 427 km away. Applying one nationality-wide "
    "share would put them in Hunan and eastern Guizhou, which is spec §8.1's failure in a "
    "new hat. **A source that placed the A-Hmao by county would add them immediately**, "
    "and that is the single highest-value missing input for China.")
NOT_ASSERTED["Yi"] = (
    "8.71M people, and the Miao case one size down. The Eastern Lipo (116,000 at 67%), "
    "Naluo (49,000 at 32%) and Laka (7,900 at 41%) are Christian and are officially Yi; "
    "together they are about 2% of the nationality, so Christianity is nowhere near the "
    "pattern of the Yi, and they are a distinct geographic subset besides.")
# NOTE: `Korean` used to sit here. It moved to MISSION on 2026-09-08 (Anita's call) and is
# now drawn at christianity.protestant 30%. See MISSION and REVIEW.
NOT_ASSERTED["Manchu"] = (
    "10.68M people, and the clearest case of a group whose religio-ethnic past does not "
    "survive into its present. Manchu shamanism was a real institution with a state cult "
    "attached; it did not outlast the Qing, and Manchu religious practice today is not "
    "distinguishable from that of the Han among whom they live.")
NOT_ASSERTED["Russian"] = (
    "15,609 people in Xinjiang and Inner Mongolia, historically Orthodox, and the one "
    "case where the derivation would probably be true. Not named because the community is "
    "small, dispersed and three generations into a place where its churches were closed, "
    "so 'probably true' is doing more work than spec §14.5 allows. If it is ever drawn it "
    "is `christianity.orthodox`, and it is 16 dots.")
NOT_ASSERTED["Achang"] = (
    "39,555 people in Dehong, beside the Jingpo and reached by the same missions. Joshua "
    "Project's weighted share is 6.28% -- a minority rather than the pattern of the group, "
    "which is what MISSION's condition (a) asks for. About 2,500 people who are probably "
    "Christian, drawn as `unknown`. Note JP's `Xiandao`, 300 people at 95%, is officially "
    "Achang: the sharpest share in China sits inside the nationality with the weakest one, "
    "which is the per-group/per-nationality problem in miniature.")
NOT_ASSERTED["Mongol"] = (
    "5.98M people, and it stays where Anita put it on 2026-09-05 -- see REVIEW. What "
    "changed on 2026-09-07 is only that they are now VISIBLE, on `unknown`, rather than "
    "absent from the map; no Buddhist share is claimed for them and none was before.")
NOT_ASSERTED["Tu"] = (
    "241,161 people in Qinghai's Huzhu and Minhe. With Mongol, and for the same reason. "
    "Now drawn on `unknown`.")
NOT_ASSERTED["Pumi"] = (
    "33,599 people in Yunnan's Ninglang and Lanping, and the one group this file has ever "
    "taken BACK off a religion -- Anita's call, 2026-09-08. Pumi religion is Hangui, an "
    "indigenous tradition held alongside Gelug Buddhism, which is spec §14.5's "
    "'religiously mixed' row and not its 'religio-ethnic' one. It goes with Mongol and Tu, "
    "on the same argument, and this REVIEW had flagged it as the one drawn group that "
    "probably should not be. See REVIEW.")

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category. It is the ONLY row in the file "
        "that must not be drawn -- everything else is now either a religion or `unknown`, "
        "and drawing this one too would count every county's people twice.",
}


REVIEW = {
    "Lisu":
        "-> christianity.protestant at 79.72%, and IT IS THE LARGEST MODELLED CLAIM IN "
        "CHINA at about 560,000 people. It is also the one row with an independent check, "
        "and the check does not fully agree. Joshua Project puts the Lisu at 80% "
        "Christian; the figure usually reported as China's OFFICIAL one is 300,000 "
        "Christian Lisu in Yunnan, which against a 702,839 nationality is 43%; and the "
        "churches' own claim is about 700,000, which is essentially 100%. **The truth is "
        "somewhere in a factor of two and this map takes the missionary source's number, "
        "which sits at the high end.** That is recorded rather than corrected, because "
        "spec §14.12's lesson is that the adjustment which feels more careful is usually "
        "the error, and because splitting sources per group would be worse than one source "
        "applied consistently. The direction of the error is stated in note_public. "
        "Fugong county, where the census puts the Lisu at 73%, is reported at about 70% "
        "Christian and is the first county in China to have been described as a Christian "
        "one; that is the pattern this row draws and the part nobody disputes.",
    "Jingpo":
        "-> christianity.protestant at 23.90%, and the arithmetic is the point. Joshua "
        "Project's Kachin Jingpo are 54% Christian and its Maru 79% -- but the Chinese "
        "`Jingpo` nationality also contains the Zaiwa, who are its largest component at "
        "119,000 and whom JP puts at 0.25%. Taking the headline group's figure would have "
        "drawn 80,000 Christians instead of 35,000. **A people-group share is not a "
        "nationality share and the census column is the nationality.** "
        "Its centroid gap of 139 km is the worst of the six and is driven entirely by JP's "
        "coordinate for the Maru at 101.32°E, which is central Yunnan and almost certainly "
        "wrong -- the Langsu live in Dehong with the rest. Flagged rather than corrected.",
    "Va":
        "-> christianity.protestant at 15.33%. The Baptist mission of William Marcus Young "
        "reached the Wa of Ximeng and Cangyuan from Kengtung in the 1900s-30s and the "
        "Parauk are its result at 19%; the Vo, the other half of the nationality, are at "
        "0.2%. The weighted figure is the lowest of the six drawn and the least "
        "comfortable, because 15% of a nationality is exactly the case where a wrong "
        "coefficient is invisible on the map.",
    "Lahu":
        "-> christianity.protestant at 41.95%, ~204,000 people, and it is NOT on spec "
        "§14.8's list. §14.8 named Lisu, Jingpo, Derung, Nu and Va; the Lahu were reached "
        "by the same Baptist mission as the Wa, JP puts the main Lahu group at 55%, and "
        "the group is 486,000 strong. This is §14.6's lesson arriving from the other "
        "direction: **a list written to illustrate a rule is not the rule**, and applying "
        "the stated test to all 56 nationalities found a group the list had missed.",
    "Korean":
        "-> christianity.protestant at 30%, ~549,000 people. **DRAWN 2026-09-08 on Anita's "
        "call, and this entry is kept in full because the argument against is still live "
        "and a reader should be able to see what was traded.** What follows was written "
        "while it was still `unknown`. "
        "**The one thing to add now that it is drawn**: of the seven MISSION rows this is "
        "the only one whose coefficient is not corroborated locally, and JP's 30% sits "
        "within two points of South Korea's own self-identified Christian share (2015 "
        "census: 19.7% Protestant + 7.9% Catholic). That may be a real convergence or it "
        "may be an estimate carried across the border; nothing here can tell. It also "
        "means the row is Protestant-only where a third of South Korea's Christians are "
        "Catholic, so if the analogy IS the source then the split is wrong as well as the "
        "level. note_public says the northeast is the least certain block on the map. "
        "1.83M people "
        "in 2010, concentrated in Yanbian and spread through Jilin, Heilongjiang and "
        "Liaoning. Joshua Project puts them at 30% Christian, they pass the co-location "
        "condition trivially (JP has them as one group), and 30% of 1.83M is ~549,000 "
        "people -- which would make them the second-largest Christian population on the "
        "Chinese map and would put a real block of dots in the northeast, where this "
        "country currently has nothing but grey. "
        "**What changed and what did not.** The old reason for excluding them was §14.5's "
        "'constituted religiously' test, which §14.9 withdrew as a ban; that objection is "
        "dead. What is left is MISSION's condition (a): for the Lisu, Lahu, Jingpo, Va, Nu "
        "and Derung the Christian pattern is attested outside the missionary literature, "
        "in Chinese state and academic reporting on Nujiang, Dehong and Lancang. For the "
        "Korean-Chinese the outside attestation is thinner and mostly by analogy with "
        "South Korea, and the population is the most urban and most migratory of the "
        "candidates, so one nationality-wide share carries more weight than it does for a "
        "people who all live in two valleys. **That is a judgement, not a rule**, and it "
        "is the kind spec §14 asks to be raised rather than decided alone.",
    "Mongol":
        "-> NO religion asserted; drawn on `unknown`. Anita's call, 2026-09-05, and it "
        "reverses spec §12's list and §14.5's table, both of which send Mongol to Tibetan "
        "Buddhism. 5.81M people in 2000 -- MORE THAN TIBETANS (5.42M) -- so this single "
        "call is the difference between Inner Mongolia or Tibet being the largest block of "
        "Vajrayana dots in China. §14.5's own test is that the coefficient be 'near 1 and "
        "DOCUMENTED rather than fitted'. For Tibetans that documentation is everywhere. "
        "For Mongols in Inner Mongolia there is nothing comparable to point at: the Gelug "
        "monastic system they would have been counted through was dismantled, and the "
        "surveys that exist put a large share of Mongols at no religion. "
        "**2026-09-07 note: the fractional-share machinery §14.9 and §14.10 now permit "
        "would make a partial Mongol claim legal where it was not before.** It is still "
        "not made, because the missing thing was never the permission but a documented "
        "coefficient, and there still is not one.",
    "Tu":
        "-> NO religion asserted, with Mongol and for the same reason. 241,161 people in "
        "Qinghai's Huzhu and Minhe, historically Gelug, and a stronger case than Mongol on "
        "the history -- but the same silence in the present.",
    "Pumi":
        "-> NO religion asserted; drawn on `unknown`. **UNDRAWN 2026-09-08, Anita's call, "
        "and it closes the flag this entry used to carry.** Until then it was "
        "buddhism.vajrayana and this REVIEW called it THE ONE DRAWN GROUP THAT PROBABLY "
        "SHOULD NOT BE. 33,599 people in Yunnan's Ninglang and Lanping. Pumi religion is "
        "Hangui, an indigenous tradition, held alongside Gelug Buddhism -- which is spec "
        "§14.5's 'religiously mixed' row rather than its 'religio-ethnic' one, so the same "
        "argument that keeps Mongol and Tu out keeps Pumi out. It had been drawn only "
        "because spec §12's list named it. **A list written to illustrate a rule is not the "
        "rule** -- the same lesson §14.6 and the Lahu record from the opposite direction, "
        "where applying the stated test ADDED a group the list had missed. 34 dots either "
        "way; the point is the consistency, not the size.",
    "Tajik":
        "-> islam, NOT islam.shia, and the restraint is deliberate. China's 41,016 Tajiks "
        "are Sarikoli and Wakhi speakers in Taxkorgan and they are ISMAILI SHIA -- the "
        "only substantial Shia community in the country, and the one genuinely novel thing "
        "China would add to the `islam` branch that Russia opened. It is not drawn, "
        "because drawing it would force the parallel claim for everyone else: if the "
        "Tajiks are Shia then the Hui, Uyghurs, Kazakhs, Salar, Dongxiang and Bonan are "
        "Sunni, and that is a second derivation nobody asked for. Being Sunni is not what "
        "makes a person Hui, so it fails §14.5's test even though it is true. Same shape "
        "as lk2024.py's refusal to file Sri Lanka's Buddhists as Theravada. See "
        "sources/ru.md §5, where Anita made the same call for Chechnya and Ingushetia.",
    "Hui":
        "-> islam. 9.82M in 2000 and the group that carries this map's national reach: the "
        "Hui are in every province, so the Muslim layer is not a northwestern regional "
        "story the way Tibetan Buddhism is. The category is the cleanest religio-ethnic "
        "case anywhere in the world -- a Hui person is defined as a Chinese-speaking "
        "Muslim, with no language, territory or appearance separating them from the Han, "
        "so 'Hui' and 'Muslim' name the same set by construction and the derivation adds "
        "no information the ethnonym did not carry. That is precisely §14.5's argument for "
        "why this is safe and also why it is honest.",
    "Uyghur":
        "-> islam. 8.40M in 2000, Sunni of the Hanafi school, concentrated in southern "
        "Xinjiang. THIS IS THE ROW spec §14 IS ABOUT. The reasoning that permits it is not "
        "that the map is obscure but that it reflects rather than reveals: the only input "
        "is the Chinese state's own published county tabulation of its own territory, and "
        "no compilation of that can tell that state anything it does not already hold. "
        "The county-level resolution is the state's own, per Anita's call of 2026-09-05 -- "
        "the source goes a level finer, to township, and that is deliberately not used.",
    "Tibetan":
        "-> buddhism.vajrayana, and the vehicle is part of the claim rather than an "
        "addition to it. lk2024.py refuses to file Sri Lanka's Buddhists as Theravada "
        "because the source said only 'Buddhist' and the school would be an extra "
        "inference; here the source says nothing about religion at all and the whole "
        "assertion is 'Tibetan implies Tibetan Buddhism', which is Vajrayana by "
        "definition. Filing it on the bare `buddhism` parent would understate what is "
        "being claimed, not overstate it.",
    "Dai":
        "-> buddhism.theravada. 1.16M in 2000, and the only Theravada population in China. "
        "The Dai of Xishuangbanna and Dehong sit inside the mainland Southeast Asian "
        "Theravada world and their monastic tradition is continuous with Laos, Myanmar and "
        "northern Thailand rather than with Han Buddhism. Note the category is broader than "
        "the practice at its edges -- some Dai in Yunnan's north hold indigenous "
        "traditions -- but the centre of the group is unambiguous.",
    "Bonan":
        "-> islam. 16,505 people, almost all in Gansu's Jishishan, and worth a line "
        "because they are the map's sharpest illustration that a row count is not a people "
        "count: 86% of them sat in ONE county the automatic name join could not resolve. "
        "Note the group is not entirely Muslim -- the Bonan of Qinghai's Tongren are "
        "Tibetan Buddhist -- but those are counted as Tu or Monguor there, so the census "
        "category `Bonan` is the Muslim one.",
    "Han":
        "-> unknown, 1.14 BILLION PEOPLE AND ABOUT 97% OF THE COUNTRY'S DOTS. See "
        "NOT_ASSERTED for the argument. The thing worth checking before touching it: this "
        "row makes China the largest country on the map, and the reason the map can carry "
        "it honestly is that `unknown` says nothing -- if anybody is ever tempted to split "
        "it, spec §14.7's 'refusing to draw the boundary is the point' is the sentence to "
        "read first.",
}


# --- resolution -------------------------------------------------------------------

def _key(cat):
    return " ".join(str(cat).split())


def shares(cat):
    """Source category -> [(node, share, tier)], or [] if not a category at all.

    The list sums to 1.0 for every category except `Total`. This is what
    countries.py::_cn_counts reads; `resolve` below is the single-node view the
    tools/ checks and spec §12's playbook expect.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return []
    if c in MAP:
        return [(MAP[c], 1.0, "derived")]
    if c in MISSION:
        s = MISSION[c][0]
        return [(MISSION_NODE, s, "modelled"), (UNKNOWN_NODE, 1.0 - s, "modelled")]
    return [(UNKNOWN_NODE, 1.0, "derived")]


def resolve(cat):
    """Source category -> the taxonomy node id it principally lands on, or None.

    `None` means the row is not a category (only `Total`). Every real category now
    resolves, because the ones no religion is claimed for land on `unknown` rather
    than being dropped -- see the module docstring.
    """
    sh = shares(cat)
    return sh[0][0] if sh else None
