"""Hand-entered national estimates for the §15 layer: one row per (country, node), each argued.

Anita, 2026-09-14: small religions come from figures entered one by one. *"we will probably have
to compile lots of sources for specific religions. that is ok. though we should try to find
self-identification ones so that we stay somewhat consistent."* Pew's seven families are the
only thing the layer takes wholesale; everything below or beside them arrives here.

A ROW
  cc       the country, lowercase ISO alpha-2 as everywhere else in this project ("uk", "xk")
  node     a node in taxonomy/religions.json
  low      the figure as a SHARE OF THE WHOLE COUNTRY, or of the family named in `of` where
  high     that is set. A single published figure has low == high. Where sources disagree, the
           range they span; never a midpoint (§15.4)
  basis    `self_id` wherever such a source exists. `estimate` or `roll` only with the reason
           in `note`, because Pew's families are self-identification and a card that mixes the
           two without saying so is §3.1's failure
  year     the fieldwork or reference year, or the span of them
  source   who says so, specifically enough to find it again: survey, wave, table. The viewer's
           card prints this string as the whole citation, so it carries the year too
  of       optional. One of Pew's families (`islam`, `christianity`, ...), and `low` and `high`
           are then shares of Pew's 2020 figure for that family in the country rather than of the
           country. This is §15.4's chain, a sect's share of Muslims times Pew's Muslims, and the
           node must sit directly under the family. Use it when the source's own denominator is
           the family: a share of Muslims converted to a share of the country by hand would bake
           in Pew's Muslim share and go stale when Pew's file does
  within   optional. The Pew family these people are INSIDE in Pew's own count. The taxonomy
           keeps some groups apart that Pew folds into a family (Anita, 2026-09-14: *"ideally
           we show alevis separately from muslims. would like to be consistent with
           taxonomy"*), so `estimates.py` takes this row's range out of that family, in the
           country and in the world total
  note     why this figure and not another

`estimates.py` multiplies the share by Pew's 2020 population for the country (or Pew's 2020 figure
for the `of` family), which is the §15.4 chain: a published share times a published total. §15.3
still applies, so a row for a node the country's own source already measured ships nothing.
"""

# Pew's 2009 Shia table, used for the Gulf where no self-identified sect covers the country.
# Its Appendix B (p. 38) says the ranges rest mainly on ethnographic studies and the World
# Religion Database, so these rows are `estimate`. Sunni is Pew's own two-way split (same page:
# "simplified into two categories: Sunni and Shia"), so a Sunni row is the rest of Muslims.
PEW2009 = "Pew Research Center 2009, Mapping the Global Muslim Population"
PEW2009_SHIA = PEW2009 + ", Shia share of Muslims"
PEW2009_SUNNI = PEW2009 + ", Muslims less the Shia share"
AB_YE = "Arab Barometer 2013 and 2018-19, Muslims naming their sect"

# Pew's sect question, self-identified: "Are you Sunni (for example, Hanafi, Maliki, Shafi, or
# Hanbali), Shia (for example, Ithnashari/Twelver or Ismaili/Sevener), or something else?" Pew
# 2012 asked it as Q31 in 2011-12; its sub-Saharan rows reprint Tolerance and Tension's Q37 from
# 2008-09. The schools are examples inside the Sunni answer, so no row reaches a school (§2.6).
PEW2012 = "Pew Research Center 2012, The World's Muslims: Unity and Diversity, Q31, p. 30"
PEW2010 = ("Pew Research Center 2010, Tolerance and Tension: Islam and Christianity in Sub-Saharan "
           "Africa, Q37, p. 21 (reprinted in Pew 2012, The World's Muslims, p. 30)")

HAND = [
    dict(
        cc="tr", node="alevism", low=0.04, high=0.0573, basis="self_id", year="2007-2021",
        source="KONDA surveys, 5.73% (2007), 5% (2018), 4% (2021); ISSP 2010, 5.47%",
        within="islam",
        note="sources.md §11ac. KONDA's 2007 figure was read at source; the 2018 and 2021 ones "
             "are as reported in the Turkish press. ISSP 2010 is the one Türkiye wave that asked "
             "the Alevi follow-up. Alevi organisations' own figure, about 15%, is a community "
             "claim and not a self-identification measure, so it is not the upper bound here.",
    ),
    dict(
        cc="cn", node="daoism", low=0.0019, high=0.0028, basis="self_id", year="2016-2021",
        source="CGSS 2021, 0.19% of adults; CLDS 2016, 0.28% of adults",
        note="sources/cn_cgss.md and sources/cn_clds.md. Both are shares of ADULTS applied here to "
             "the whole population, which assumes children identify as their parents do. CGSS "
             "2012 and 2017 read 0.26% and 0.23%, so the range covers the whole run. §14.16 "
             "found 80 respondents in 32,495 far too few to place, which is exactly the case "
             "§15 exists for.",
    ),

    # ---- Islam's branches, first batch: the Gulf and Yemen (sources/estimates.md) ----------
    dict(
        cc="ye", node="islam.shia", of="islam", low=0.1815, high=0.1902, basis="self_id",
        year="2013-2019", source=AB_YE,
        note="Wave III (Nov-Dec 2013, 1,198 Muslims, all 20 governorates) offers Sunni and Shia: "
             "19.0% Shia, with Saada 98.9% and every southern governorate 0%. Wave V (2018-19, "
             "2,399 Muslims) logs a volunteered answer against a master list with no Zaydi box, "
             "and nobody in Yemen was logged `Shia`; 18.2% were logged `Alawi`, which runs Saada "
             "62.5%, Amran 51.3%, Dhamar 39.9% and zero across the south, so it is the Zaydi "
             "answer. Wave V also has `Just a Muslim` at 24.4%, heaviest in Sanaa and the "
             "highlands, and nobody who named no sect is counted here. Pew 2009 gives 35-40% of "
             "Muslims, ascribed from ethnographic data rather than asked, and is not used.",
    ),
    dict(
        cc="ye", node="islam.sunni", of="islam", low=0.5694, high=0.8006, basis="self_id",
        year="2013-2019", source=AB_YE,
        note="The same two waves as the Shia row. 80.1% in 2013, when the answer list had no way "
             "to name no sect; 56.9% in 2018-19 (Shafi'i 35.2%, Sunni 21.7%), when 24.4% said "
             "`Just a Muslim`. The range is that difference between the two lists.",
    ),
    dict(
        cc="sa", node="islam.shia", of="islam", low=0.10, high=0.15, basis="estimate",
        year="2009", source=PEW2009_SHIA,
        note="Pew 2009 p. 41. No self-identified figure covers the country. Arab Barometer wave II "
             "(2010-11) asked Saudis their sect in a Saudi-only item, sa1012, and reads 3.85% of "
             "Muslims Shia, but it sampled 6 of 13 regions and Najran and Medina are not among "
             "them; its Eastern Region reads 13.9%. Pew's share is of all Muslims, citizens or "
             "not, and is applied here to Pew's 2020 Muslims.",
    ),
    dict(
        cc="sa", node="islam.sunni", of="islam", low=0.85, high=0.90, basis="estimate",
        year="2009", source=PEW2009_SUNNI,
        note="The rest of Muslims after the Shia row. Arab Barometer wave II reads 96.2% "
             "(Sunni, Hanbali, Shafi'i) on the six-region sample the Shia row describes.",
    ),
    dict(
        cc="kw", node="islam.shia", of="islam", low=0.20, high=0.25, basis="estimate",
        year="2009", source=PEW2009_SHIA,
        note="Pew 2009 p. 40. Nothing self-identified exists. Arab Barometer wave III records "
             "Kuwaiti sect only as the interviewer's opinion of the respondent (q2005kw: Sunni "
             "66.7%, Shia 14.3%, could not tell 19.0%), and waves V, VII and VIII leave the sect "
             "column empty for Kuwait. Those waves survey citizens only; Pew's share is of all "
             "Muslims and is applied here to Pew's 2020 Muslims.",
    ),
    dict(
        cc="kw", node="islam.sunni", of="islam", low=0.75, high=0.80, basis="estimate",
        year="2009", source=PEW2009_SUNNI,
        note="The rest of Muslims after the Shia row.",
    ),
    dict(
        cc="qa", node="islam.shia", of="islam", low=0.10, high=0.10, basis="estimate",
        year="2009", source=PEW2009_SHIA,
        note="Pew 2009 p. 41, printed as ~10%. Arab Barometer fielded Qatar in waves IV and V and "
             "the sect column is empty both times. Pew 2009 counted 1.09M Muslims in Qatar and "
             "Pew 2020 counts 2.13M; the share is assumed to hold across that growth, most of it "
             "migration.",
    ),
    dict(
        cc="qa", node="islam.sunni", of="islam", low=0.90, high=0.90, basis="estimate",
        year="2009", source=PEW2009_SUNNI,
        note="The rest of Muslims after the Shia row.",
    ),
    dict(
        cc="ae", node="islam.shia", of="islam", low=0.10, high=0.10, basis="estimate",
        year="2009", source=PEW2009_SHIA,
        note="Pew 2009 p. 41, printed as ~10%. The UAE is in no Arab Barometer wave. Pew 2009 "
             "counted 3.50M Muslims and Pew 2020 counts 6.89M; the share is assumed to hold across "
             "that growth, most of it migration.",
    ),
    dict(
        cc="ae", node="islam.sunni", of="islam", low=0.90, high=0.90, basis="estimate",
        year="2009", source=PEW2009_SUNNI,
        note="The rest of Muslims after the Shia row.",
    ),
    dict(
        cc="om", node="islam.shia", of="islam", low=0.05, high=0.10, basis="estimate",
        year="2009", source=PEW2009_SHIA,
        note="Pew 2009 p. 40. Oman is in no Arab Barometer wave. There is deliberately no Sunni "
             "row: Pew's p. 9 names \"Kharijites in Oman\" (the Ibadis) as fitting neither of its "
             "two categories, so the rest of Muslims is not Sunni. `islam.ibadi` exists since "
             "2026-09-14 and has no row either: the Ibadi figures found for Oman (45% to 75% "
             "across sources, per the State Department's reports) are shares of citizens, and a "
             "citizens-only figure is not a national estimate here.",
    ),

    # ---- Islam's branches, third batch: Pew 2012 Q31 (sources/estimates.md) ----------------
    # Anita, 2026-09-14: "we can add the national sunni/shia figures for sure." Six built
    # countries that draw Muslims undivided, where Pew's named share is a majority. Shares of
    # Muslims; "just a Muslim", nothing and don't know stay on `islam` (spec §2.7a), and a 0 has
    # no row.
    dict(
        cc="pk", node="islam.sunni", of="islam", low=0.81, high=0.81, basis="self_id",
        year="2011", source=PEW2012,
        note="Nov. 10-30, 2011, 1,450 Muslims. The sample left out FATA, Gilgit-Baltistan, Azad "
             "Jammu and Kashmir and unstable parts of Khyber Pakhtunkhwa and Balochistan, and "
             "represents 82% of adults (p. 125). 12% said just a Muslim and 1% something else.",
    ),
    dict(
        cc="pk", node="islam.shia", of="islam", low=0.06, high=0.06, basis="self_id",
        year="2011", source=PEW2012,
        note="A floor. The same sample left out FATA, Gilgit-Baltistan, Azad Jammu and Kashmir and "
             "unstable parts of Khyber Pakhtunkhwa and Balochistan (p. 125, 82% of adults), "
             "which include Shia areas, and 12% said just a Muslim. Pew 2009 gives 10-15% of "
             "Muslims (p. 40), compiled from ethnographic sources rather than asked, so it is "
             "not the range here. No Ahmadiyya row: Pakistan's census counts Ahmadis.",
    ),
    dict(
        cc="bd", node="islam.sunni", of="islam", low=0.92, high=0.92, basis="self_id",
        year="2011-2012", source=PEW2012,
        note="Nov. 21, 2011 to Feb. 5, 2012, 1,918 Muslims, all seven divisions. 4% said just a "
             "Muslim and 2% nothing or don't know. The State Department's \"91 percent\" Sunni "
             "is its own gloss on the 2022 census, which asks no sect.",
    ),
    dict(
        cc="bd", node="islam.shia", of="islam", low=0.02, high=0.02, basis="self_id",
        year="2011-2012", source=PEW2012,
        note="The same survey as the Sunni row: 2% of 1,918 Muslims, within its margin of error "
             "of 4.4 points.",
    ),
    dict(
        cc="eg", node="islam.sunni", of="islam", low=0.88, high=0.88, basis="self_id",
        year="2011", source=PEW2012,
        note="Nov. 14 to Dec. 18, 2011, 1,798 Muslims, 24 of 29 governorates (the five frontier "
             "ones, 2% of people, left out). 12% said just a Muslim. No Shia row: 0% in Pew "
             "2012. How much the answer card matters: the Global Flourishing Study's sect item "
             "reads 25.6% Sunni and 72.7% just a Muslim (sources/branches.md).",
    ),
    dict(
        cc="jo", node="islam.sunni", of="islam", low=0.93, high=0.93, basis="self_id",
        year="2011", source=PEW2012,
        note="Nov. 3 to Dec. 3, 2011, 966 Muslims, all 12 governorates. 7% said just a Muslim. No "
             "Shia row: 0% in Pew 2012. The fieldwork is before most of the Syrian refugees "
             "arrived, and Pew's 2020 Muslims include them.",
    ),
    dict(
        cc="ke", node="islam.sunni", of="islam", low=0.73, high=0.73, basis="self_id",
        year="2008", source=PEW2010,
        note="Dec. 18-27, 2008, 340 Muslims (a national sample of 1,300 plus 200 extra interviews "
             "with Muslims), margin of error 7 points, and the Muslim sample is mostly men "
             "(p. 65). A small sample and old fieldwork. 8% said just a Muslim and 7% nothing "
             "or don't know.",
    ),
    dict(
        cc="ke", node="islam.shia", of="islam", low=0.08, high=0.08, basis="self_id",
        year="2008", source=PEW2010,
        note="The same 340 Muslims as the Sunni row, fieldwork December 2008, margin of error 7 "
             "points. A small sample and old fieldwork.",
    ),
    dict(
        cc="gh", node="islam.sunni", of="islam", low=0.51, high=0.51, basis="self_id",
        year="2009", source=PEW2010,
        note="Jan. 17-30, 2009, 339 Muslims (a national sample of 1,300 plus 200 extra interviews "
             "with Muslims), margin of error 7 points, and the Muslim sample is mostly men "
             "(p. 65). A small sample and old fieldwork. 13% said just a Muslim and 11% nothing "
             "or don't know.",
    ),
    dict(
        cc="gh", node="islam.shia", of="islam", low=0.08, high=0.08, basis="self_id",
        year="2009", source=PEW2010,
        note="The same 339 Muslims as the Sunni row, fieldwork January 2009, margin of error 7 "
             "points. A small sample and old fieldwork.",
    ),
    dict(
        cc="gh", node="islam.ahmadiyya", of="islam", low=0.16, high=0.16, basis="self_id",
        year="2009", source=PEW2010,
        note="Its own column on p. 21, a volunteered answer to the sect question; Pew 2012 p. 30 "
             "prints the same 16 as \"something else\". The same 339 Muslims, margin of error 7 "
             "points. Ghana's 2010 and 2021 census forms print Ahmadi as an answer, but every "
             "published table folds it into Islam (sources/branches.md).",
    ),
]
