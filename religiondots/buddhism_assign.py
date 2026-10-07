"""Buddhists drawn on a school by assignment, where every count is of Buddhism as one answer.

WHY THIS EXISTS
---------------
Spec §2.6a, decided 2026-09-15 and built 2026-10-04 (Anita: *"ok lets go ahead"*, and *"bhutan is
fine ya"*). No census or survey in these countries asks which school a Buddhist follows, and each
country's Buddhist institutions are one school, so the census's Buddhist cell is ASSIGNED: the
school takes it, less a remainder that stays on bare `buddhism` for Buddhists of Chinese and
Vietnamese descent (or, in South Korea, Theravada and Tibetan), spread at one share across the
country because nothing places them. The evidence is `sources/branches.md`, "Thailand's 7.9%
Mahayana", and §2.6a's per-country check.

THE REMAINDER is the World Religion Database's (2025, via ARDA's rankings ADH_709 Mahayanists,
ADH_720 Theravadins, ADH_722 Lamaists) other traditions as a share of all three, Anita's choice
("lets take databases"). WRD's split is itself ethnic ascription at a fixed ratio, which is why it
sizes only what is NOT asserted.

THREE SHAPES BESIDE THE SIX THERAVADA COUNTRIES
-----------------------------------------------
* South Korea goes to `buddhism.mahayana`, its remainder WRD's Theravada and Lamaist. Won
  Buddhism is its own census answer and does not move.
* Mongolia's census Buddhists were already on `buddhism.vajrayana` (taxonomy/mn2020.py), filed
  there by the mapping; they are moved back to `buddhism` first and assigned like the rest, so
  they carry the tier and the row that say so. WRD lists no tradition there but Lamaist.
* Bhutan: WRD 82.72% Lamaist, 0.02% Mahayana, no Theravada (ARDA, read 2026-10-04), so all to
  `buddhism.vajrayana`, as Mongolia.
* Russia uses its survey's own geography instead of an even spread (Anita, 2026-09-15: "we should
  use the surveys geography. why would we not?"): `buddhism.vajrayana` in Tuva, Buryatia, Kalmykia
  and Zabaykalsky only, which hold 85.7% of the drawn Buddhists; the other subjects' Buddhists stay
  unassigned, one or two Arena respondents each.

TIERS are branch_assign.split's: a counted Buddhist row becomes `assigned` and rolls back to
`buddhism`; Russia's and Bhutan's are `modelled` (survey and compiler shares) and stay so.
"""

SCHOOL = {
    "th": "buddhism.theravada",
    "mm": "buddhism.theravada",
    "kh": "buddhism.theravada",
    "lk": "buddhism.theravada",
    "la": "buddhism.theravada",
    "bd": "buddhism.theravada",
    "kr": "buddhism.mahayana",
    "mn": "buddhism.vajrayana",
    "bt": "buddhism.vajrayana",
    "ru": "buddhism.vajrayana",
}
PARENT = "buddhism"

# Share of Buddhists left on bare `buddhism`: WRD 2025's other traditions over all three (spec
# §2.6a's table). Russia's 1.0 is "nothing assigned" outside UNIT_REMAINDER's four subjects.
REMAINDER = {
    "th": 0.091,    # Mahayana 7.90% of Theravada 78.98% + Mahayana
    "mm": 0.009,    # Mahayana 0.65%, Lamaist 0.02%, Theravada 73.54%
    "kh": 0.012,    # Mahayana 1.01%, Theravada 85.96%
    "lk": 0.002,    # Mahayana 0.11%, Theravada 67.88%
    "la": 0.023,    # Mahayana 1.19%, Theravada 51.61%
    "bd": 0.0,      # Mahayana 0.00%, Theravada 0.72%
    "kr": 0.007,    # Theravada 0.18% + Lamaist 0.00% of Mahayana 24.37%
    "mn": 0.0,      # Lamaist only
    "bt": 0.0,      # Lamaist 82.72%, Mahayana 0.02%: 0.02% of Buddhists, under a dot
    "ru": 1.0,      # see UNIT_REMAINDER
}
UNIT_REMAINDER = {("ru", u): 0.0 for u in ("RU-TY", "RU-BU", "RU-KL", "RU-ZAB")}

# Moved from the school back to `buddhism` before the split, because the mapping already filed
# them there (Mongolia's census `Будда` on buddhism.vajrayana, mn2020.py).
RETAG = {"mn"}

ASSIGNED = {
    "th": "Theravada", "mm": "Theravada", "kh": "Theravada", "lk": "Theravada",
    "la": "Theravada", "bd": "Theravada", "kr": "Mahayana", "mn": "Vajrayana",
    "bt": "Vajrayana",
}
ASSIGNED_EXCEPT = {"ru": "Vajrayana in the four Buddhist republics"}
SUFFIX = ", from national estimates"


def assigned_text(cc):
    return ASSIGNED_EXCEPT.get(cc) or ASSIGNED[cc] + SUFFIX


# The note paragraph each country gains. Panel voice: no em dashes, bold only on a figure.
WHY = {
    "th": "No source counts Thai Buddhists by school. On 1 March 2025, 44,153 of Thailand's "
          "44,195 temples were Theravada (National Office of Buddhism); the World Religion "
          "Database puts Mahayana at 7.90% of Thais, a share that stands for Thai Chinese "
          "families and has not moved since 1900 in its figures.",
    "mm": "No source counts Myanmar's Buddhists by school, and its sangha is Theravada. The World "
          "Religion Database puts Mahayana and Tibetan Buddhists at 0.9% of Buddhists.",
    "kh": "No source counts Cambodia's Buddhists by school, and its sangha is Theravada. The World "
          "Religion Database puts Mahayana at 1.2% of Buddhists, mostly Vietnamese and Chinese "
          "families.",
    "lk": "No source counts Sri Lanka's Buddhists by school, and its sangha is Theravada. The "
          "World Religion Database puts Mahayana at 0.2% of Buddhists.",
    "la": "No source counts Laos's Buddhists by school, and its sangha is Theravada. The World "
          "Religion Database puts Mahayana at 2.3% of Buddhists, mostly Vietnamese and Chinese "
          "families.",
    "bd": "No source counts Bangladesh's Buddhists by school. They are the Chakma, Marma, "
          "Tanchangya, Barua and Rakhine, and their temples are Theravada; the World Religion "
          "Database records no other school here.",
    "kr": "No source counts South Korea's Buddhists by school, and every large order, Jogye, "
          "Taego, Cheontae and Jingak, is Mahayana. Won Buddhism is its own census answer and "
          "keeps its own colour. The World Religion Database puts Theravada and Tibetan "
          "Buddhists at 0.7% of Buddhists, mostly migrant workers' temples.",
    "mn": "The census counts Buddhists as one answer, and Mongolia's monasteries are Tibetan "
          "Buddhist; the World Religion Database records no other school here.",
    "bt": "Bhutan's Buddhists are Drukpa Kagyu and Nyingma, both Tibetan Buddhist, and the World "
          "Religion Database records almost no other school (0.02% Mahayana).",
    "ru": "The survey counts Buddhists as one answer. In Tuva, Buryatia, Kalmykia and Zabaykalsky, "
          "where 85.7% of them live, the tradition is Tibetan Buddhist, and there they are drawn "
          "as Vajrayana. Elsewhere, one or two respondents per region, they are left with no "
          "school, since some will be Buryat, Kalmyk or Tuvan and some not.",
}

# Places a reader should know are drawn on the school although many of their Buddhists may not
# follow it (spec §2.6a, "Per-country check"; Anita: "sure just a note for mae hong son etc.").
CAVEAT = {
    "th": "In the northern highlands many people counted as Buddhist keep spirit and ancestor "
          "religion or Mien Taoist ritual: perhaps a fifth to a third of Mae Hong Son's "
          "Buddhists, and fewer in Tak, Chiang Rai, Chiang Mai and Nan. They are drawn as "
          "Theravada with everyone else. The share left with no school is spread evenly, so "
          "it is too large in the Northeast and too small in Bangkok, where most Thai Chinese "
          "live.",
    "mm": "The Kokang zone of northern Shan, about 121,000 Buddhists by Joshua Project's estimate, "
          "keeps Chinese folk religion and Mahayana, and is drawn as Theravada with the rest of "
          "Shan.",
    "kh": "About a quarter of Ratanakiri's Buddhists may be highland folk religion that moved into "
          "the Buddhist answer since 2008, and about a quarter of Preah Sihanouk's are likely "
          "recent Chinese migrants. Both are drawn as Theravada.",
    "la": "One village in Bokeo's Golden Triangle zone, about 2,300 Buddhists, is mostly people "
          "outside every Lao ethnic group, probably Chinese nationals, and is drawn as Theravada.",
}


def note(cc):
    r = REMAINDER[cc]
    school = {"buddhism.theravada": "Theravada", "buddhism.mahayana": "Mahayana",
              "buddhism.vajrayana": "Vajrayana"}[SCHOOL[cc]]
    parts = [f"**Buddhists are drawn as {school} by assignment.**", WHY[cc]]
    if cc != "ru":
        parts.append(f"**{_pct(r)}** of Buddhists are left with no school, spread evenly."
                     if r else "No Buddhists are left without a school.")
    if cc in CAVEAT:
        parts.append(CAVEAT[cc])
    return " ".join(parts)


def _pct(r):
    p = r * 100
    return f"{p:.2f}".rstrip("0").rstrip(".") + "%"


def apply(cc, df):
    if cc not in SCHOOL:
        return df
    import branch_assign
    if cc in RETAG:
        df = df.copy()
        df.loc[df["node"] == SCHOOL[cc], "node"] = PARENT
    return branch_assign.split(
        cc, df, PARENT, SCHOOL[cc], REMAINDER[cc],
        unit_share={u: s for (c, u), s in UNIT_REMAINDER.items() if c == cc})


def check():
    for cc in SCHOOL:
        if cc not in REMAINDER or cc not in WHY:
            raise SystemExit(f"buddhism_assign: {cc} is missing a REMAINDER or WHY")
        if not (cc in ASSIGNED or cc in ASSIGNED_EXCEPT):
            raise SystemExit(f"buddhism_assign: {cc} has no `assigned` text")
        for t in (WHY[cc], CAVEAT.get(cc, "")):
            if "—" in t:
                raise SystemExit(f"buddhism_assign: {cc} note has an em dash")
