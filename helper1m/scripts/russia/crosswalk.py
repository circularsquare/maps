"""Carry an older year's municipalities onto the 1 January 2025 set.

Between 2024 and 2025 about 180 municipal districts and urban okrugs became
municipal okrugs. Most kept their territory and only changed code (the type
digit: 14710000 Alekseyevsky urban okrug -> 14510000 Alekseyevsky municipal
okrug); some merged (Kasimov town + Kasimovsky district -> Kasimovsky okrug;
Polysayevo and Leninsk-Kuznetsky -> one okrug); some towns were folded into
the district around them, which keeps its code but grows (Chuvashia's
Alatyr, Kanash, Shumerlya). So within each subject:

1. A code present in both years whose population moved by under 8% is the
   same unit.
2. Every other old unit is linked to a new unit by name stem (first 6
   letters of the distinctive word, so "Касимов" finds "Касимовский"), then,
   failing that, by the two code digits that number the unit within its type
   (CC in AA B CC 000) where exactly one new unit shares them.
3. Old units still unlinked go, largest first, to the new unit whose
   population is most short of its linked old units, as long as that does
   not overshoot it by more than 8%.
4. Every new unit must then sit within 8% of the sum of its old units; any
   that does not is reported, and MANUAL decides it.
"""
import re
from collections import defaultdict

TOL = 0.08

# old OKTMO -> new OKTMO, for links the rules above cannot make. Filled from
# the report the rules print.
MANUAL = {
    # Pushchino urban okrug was merged into Serpukhov in 2024, with Protvino;
    # rule 3 alone would put it into Lyubertsy, which grew by about as much.
    "46762000": "46770000",
}

# Groups of new units whose old figures are on a different territory and
# cannot be split back: each member's old figure is set to its new figure
# times the group's old/new ratio, so the pair carries the group's change
# and no jump that is only a boundary move. Found by the RATIO lines of the
# report; units with a RATIO line that are real growth are left alone
# (Moscow's Troitsky okrug, +16%, is New Moscow's building boom;
# Svobodnensky district of Amur, +46%, has no neighbour that lost people).
REBASE_GROUPS = [
    # Saratov city took part of Tatishchevsky district: -5,398 there, +3,716
    # in the city, in a region otherwise losing people.
    ["63646000", "63701000"],
]
# Subjects where the whole subject is rebased: Chechnya moved ~66,000 people
# from every district into Grozny between the two estimates (Grozny +20%,
# every district -3% to -4%), a re-allocation rather than a year's migration.
REBASE_SUBJECTS = ["RU-CE"]


def rebase(old_values, new):
    """old_values: {new_oktmo: old-year total on that unit's territory}.
    Applies REBASE_GROUPS and REBASE_SUBJECTS in place; returns report lines."""
    groups = [list(g) for g in REBASE_GROUPS]
    for iso in REBASE_SUBJECTS:
        groups.append([n for n, v in new.items() if v[0] == iso])
    out = []
    for g in groups:
        o_tot = sum(old_values[n] for n in g)
        n_tot = sum(new[n][2] for n in g)
        before = {n: old_values[n] for n in g}
        for n in g:
            old_values[n] = round(new[n][2] * o_tot / n_tot)
        # Put the rounding remainder on the biggest member, so the group (and
        # the subject) still sums to the published total.
        big = max(g, key=lambda n: new[n][2])
        old_values[big] += o_tot - sum(old_values[n] for n in g)
        for n in g:
            out.append(f"rebased {n} {new[n][1]}: {before[n]} -> {old_values[n]}")
    return out

STOP = {
    "муниципальный", "муниципального", "муниципальное", "образование", "район",
    "района", "округ", "округа", "городской", "город", "г", "поселок", "посёлок",
    "пгт", "рп", "курорт", "город-курорт", "и", "им", "имени", "закрытое",
    "административно-территориальное", "зато", "сельское", "поселение",
    "административный", "городского", "значения", "федерального",
    "с", "подведомственной", "подведомственными", "территорией", "территориями",
    "подчиненной", "подчиненными", "рабочий", "-",
}


def key(name):
    """The distinctive words of a unit's name, type words dropped."""
    words = re.findall(r"[а-яё0-9-]+", name.lower().replace("ё", "е"))
    # startswith("муни") also catches the workbook's typo "муниипальный".
    return " ".join(w for w in words if w not in STOP and not w.startswith("муни"))


def unit_type(name):
    """'go' urban okrug, 'mo' municipal okrug, 'mr' municipal district, or ''.
    Breaks ties between twins such as Kemerovsky municipal okrug (the
    district) and Kemerovsky urban okrug (the city)."""
    n = name.lower()
    if re.search(r"городск\w* округ", n):
        return "go"
    if re.search(r"муни\w* округ", n):
        return "mo"
    if re.search(r"муни\w* район|\bрайон\b", n):
        return "mr"
    return ""


ENDINGS = ("ского", "ский", "ской", "ское", "ская", "кий", "кой", "кое", "кая",
           "ий", "ый", "ой", "ое", "ая")


def stem(name):
    """key() with adjectival endings cut, so "Касимов" meets "Касимовский"."""
    out = []
    for w in key(name).split():
        for e in ENDINGS:
            if w.endswith(e) and len(w) - len(e) >= 4:
                w = w[:-len(e)]
                break
        out.append(w)
    return " ".join(out)


def link(old, new):
    """old, new: {oktmo: (iso, name, pop)} for one year each.
    Returns ({new_oktmo: [old_oktmo, ...]}, report_lines)."""
    links = defaultdict(list)
    report = []
    pending_old = []
    for o, (iso, name, pop) in old.items():
        if o in MANUAL:
            links[MANUAL[o]].append(o)
        elif o in new and abs(new[o][2] / max(pop, 1) - 1) < TOL:
            links[o].append(o)
        else:
            pending_old.append(o)
    taken = set(links)
    by_iso_new = defaultdict(list)
    for n, (iso, name, pop) in new.items():
        if n not in taken:
            by_iso_new[iso].append(n)

    leftovers = []
    for o in pending_old:
        iso, name, pop = old[o]
        cands = by_iso_new.get(iso, [])
        hit = [n for n in cands if key(new[n][1]) == key(name)]
        if len(hit) != 1:
            hit = [n for n in cands if stem(new[n][1]) == stem(name)]
        if len(hit) == 0:
            hit = [n for n in cands if n[3:5] == o[3:5]]
        if len(hit) == 1:
            links[hit[0]].append(o)
        else:
            leftovers.append(o)

    # Rule 3: fold leftovers into the new unit with the largest shortfall.
    for o in sorted(leftovers, key=lambda o: -old[o][2]):
        iso, name, pop = old[o]
        best, best_gap = None, None
        for n, (niso, nname, npop) in new.items():
            if niso != iso:
                continue
            have = sum(old[x][2] for x in links.get(n, []))
            gap = npop - have
            if gap > 0 and have + pop <= npop * (1 + TOL) and (best is None or gap > best_gap):
                best, best_gap = n, gap
        if best is None:
            report.append(f"UNLINKED old {o} {name} {pop} ({iso})")
        else:
            links[best].append(o)
            report.append(f"folded old {o} {name} {pop} into {best} {new[best][1]}")

    for n, (iso, name, pop) in sorted(new.items()):
        olds = links.get(n, [])
        have = sum(old[x][2] for x in olds)
        if not olds:
            report.append(f"NO OLD UNIT for new {n} {name} {pop} ({iso})")
        elif abs(pop / have - 1) >= TOL:
            report.append(f"RATIO new {n} {name} {pop} vs old {have} from "
                          + ", ".join(f"{x} {old[x][1]} {old[x][2]}" for x in olds))
        elif olds != [n]:
            report.append(f"linked new {n} {name} {pop} <- "
                          + ", ".join(f"{x} {old[x][1]} {old[x][2]}" for x in olds))
    return dict(links), report
