"""Uganda: NPHC 2024 tribe by subcounty (UBOS 10% sample), read as home language through
Afrobarometer R4-R9.

    python sources/ug_nphc_extract.py   # once, ~25 min: the sample's tribe counts by parish
    python sources/ug_afro.py           # once: Uganda's Afrobarometer rows
    python sources/ug_census.py         # -> data/normalized/ug.csv

THE CENSUS. The 2024 census asked "tribe / nationality" (HH_P10), not language. Its 10% person
sample (Anita's UBOS download, the file religiondots' Uganda is drawn from) gives every household
member's answer with district, county, subcounty and parish. Each drawn unit's tribe shares come
from its sampled household members; its people are religiondots' full-count household population
for the same unit (`Total population` minus `Not in a household`, Subcounty Profiles workbook),
2,200 units (2,207 subcounties, Bidi Bidi's three camps merged with their hosts, as religiondots).

THE SURVEY. Each tribe's count is shared across home languages by Afrobarometer's answers from
respondents of that tribe (R4-R9, 2008-2022, ~12,000 respondents), by region (Central, Kampala,
East, North, West), each tribe-region share shrunk towards the tribe's national share with a
prior of K respondents, and the tribe's national share towards its own language with K0.

LINGUA FRANCAS (ask 018, ruled 2026-10-05: LF_ROUNDS = [7], R7's mother-tongue question Q2A,
which sources/ug_afro.py now extracts for R7; the rest of this paragraph is the earlier call). R4-R6 asked "Which language is your home language?", R7-R9
"Language spoken in home". In Uganda English went from 0.0% to 2.2% between the two wordings,
and non-Baganda answering Luganda from 0.6% to 8.5% (78% of R9's non-Baganda in Buganda). So
the shares of the lingua francas (English, Swahili, and Luganda for anyone not Muganda) come
from LF_ROUNDS only; every other answer's share from all rounds among the non-lingua-franca
answers, scaled to what the lingua francas leave. LF_ROUNDS = [7, 8, 9] or ROUNDS reverses it.
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
RAW = HERE / "data" / "raw" / "ug"
SAMPLE = RAW / "ug2024_parish_ethnic.csv"
P10_LABELS = RAW / "ug2024_p10_labels.txt"
AB = RAW / "ab_ug_language.csv"
RD_UG = RD / "data" / "normalized" / "ug.csv"
DISTRICTS = RD / "data" / "raw" / "ug" / "nphc2024_portal" / "district.json"
OUT = HERE / "data" / "normalized" / "ug.csv"

UNITS = 2200
HHPOP = 44_378_756          # 44,387,526 less Apaa's 8,770 (no polygon, no sample record)
SOURCE_ID = "ubos_nphc2024_tribe_x_afrobarometer_r4_r9"
YEAR = "2024 (census), 2008-2022 (survey)"

ROUNDS = [4, 5, 6, 7, 8, 9]
# Anita's ruling on ask 018 (2026-10-05): the lingua francas' shares from R7's mother-tongue
# question (Q2A, R7's "lang" in the extract since then). Before: [4, 5, 6].
LF_ROUNDS = [7]
K = 30.0                    # tribe-region share shrunk to the tribe's national share
K0 = 10.0                   # tribe's national share shrunk to its own language

# Bidi Bidi (religiondots sources/ug_2024.py, read-only, copied): camps drawn with their hosts.
CAMP_HOSTS = {
    "UG3130106": ["UG3130102"],
    "UG3130207": ["UG3130203", "UG3130205", "UG3130206"],
    "UG3130409": ["UG3130402", "UG3130403", "UG3130408"],
}


def drawn_unit(uid):
    for camp, hosts in CAMP_HOSTS.items():
        if uid == camp or uid in hosts:
            return hosts[0] + "BB"
    return uid


# Census tribe (HH_P10) -> the language read as its own. Glottolog for the pairs (tree.d/ug.txt).
OWN = {
    511: "Acholi", 512: "Ma'di",            # Aliba: survey's 9 Aliba answered Madi 7, Lugbara 2
    513: "Alur", 514: "Aringa", 515: "Amba", 516: "Luhya", 517: "Bwisi",
    518: "Rufumbira", 519: "Ganda", 520: "Masaaba", 521: "Gungu",
    522: "Saamia",                           # Bagwe: Lugwe, a Saamia-Lugwe (Luhya) variety
    523: "Gwere", 524: "Hehe",
    525: "Nyankore",                         # Bahororo: Hororo, a dialect of Nyankole (horo1246)
    526: "Kenyi", 527: "Chiga", 528: "Konzo",
    529: "Tooro",                            # Banyabindi of Kasese: a Tooro variety
    530: "Nyankore",                         # Banyabutumbi (Rukungiri): Nyankore-Kiga
    531: "Nyankore", 532: "Nyara",
    533: "Kupsabiny",                        # Benet (Ndorobo of Mt Elgon): speak Kupsabiny
    534: "Nyankore",                         # Banyaruguru (Rubirizi): a Nyankore variety
    535: "Kinyarwanda", 536: "Nyole", 537: "Nyoro", 538: "Ruuli", 539: "Kirundi",
    540: "Saamia", 541: "Soga", 542: "Songora", 543: "Tagwenda", 544: "Tooro",
    545: "Tooro",                            # Batuku: Tuku, a dialect of Tooro (tuku1253)
    546: "BATWA",                            # by district, below
    547: "Chope",
    548: "Karamojong",                       # Dodoth: Dodos, a Karamojong dialect (dodo1240)
    549: "Labwor",                           # Ethur: Lebthur, Labwor (labw1238)
    550: "Other Ugandan language",           # Gimara: language not identified
    551: "Ik", 552: "Teso",
    553: "Karamojong",                       # Jie: a Karamojong dialect (jiee1239)
    554: "Alur",                             # Jonam: a dialect of Alur (jona1239)
    555: "Adhola", 556: "Luhya", 557: "Kakwa", 558: "Karamojong",
    559: "Ndo",                              # Kebu (Okebu): Ndo (ndoo1242)
    560: "Kuku", 561: "Kumam", 562: "Lango", 563: "Lendu", 564: "Lugbara", 565: "Ma'di",
    566: "Karamojong",                       # Mening: a Karamojong-Toposa variety
    567: "Mvuba",
    568: "Karamojong",                       # Napore and Nyangia: Nyang'i is nearly extinct
    571: "Karamojong",                       #   (Glottolog AES); they speak Dodos/Karamojong
    569: "Karamojong",                       # Ngikutio: a Dodos section
    570: "Nubi", 572: "Pokot",
    573: "Other Ugandan language",           # Reli: language not identified
    574: "Kupsabiny",
    575: "Other Ugandan language",           # Shana: language not identified
    576: "Karamojong",                       # So (Tepeth): Soo is moribund (Glottolog AES)
    577: "Other Ugandan language",           # Vonoma: language not identified
    578: "Other Ugandan language",           # "Other Ugandan"
    579: "Other Ugandan language",           # Bakingwe: language not identified
    580: "Soga",                             # Bagabu: Gabula, a dialect of Soga (gabu1247)
    581: "Sabaot", 582: "Sabaot",            # Sabot; Mosopisyek (Mosop, Sabaot)
    583: "Haya",                             # Baziba: the Ziba of Kagera speak Haya
}
UGANDA = 500                                 # "Uganda": no tribe given; shared like the unit
UNKNOWN = 998                                # nationality "Unknown": likewise
FOREIGN = {"Rwanda": "Kinyarwanda", "Burundi": "Kirundi", "Somalia": "Somali"}
AFRICA = {"Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde",
          "Cameroon", "Central African Republic", "Chad", "Comoros",
          "Congo, Democratic Republic of the", "Congo, Republic of the", "Côte d’Ivoire",
          "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
          "The Gambia", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho",
          "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
          "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
          "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia",
          "South Africa", "South Sudan", "Sudan, South", "Sudan", "Tanzania", "Togo", "Tunisia", "Zambia",
          "Zimbabwe"}

# Afrobarometer: the card's labels -> answer (spellings merged across rounds).
CODED = {
    "Luganda": "Ganda", "Lusoga": "Soga", "Runyankole": "Nyankore", "Runyankore": "Nyankore",
    "Langi": "Lango", "Ateso": "Teso", "Rukiga": "Chiga", "Lugbara": "Lugbara",
    "Acholi": "Acholi", "Runyoro": "Nyoro", "Runyolo": "Nyoro", "Lumasaaba": "Masaaba",
    "Lumasaba": "Masaaba", "Alur": "Alur", "Rutooro": "Tooro", "Lukhonjo": "Konzo",
    "Lukhonzo": "Konzo", "Ngakarimajong": "Karamojong", "Ngakarimojong": "Karamojong",
    "Madi": "Ma'di", "English": "English", "Japadhola": "Adhola", "Japhadhola": "Adhola",
    "Lusamia": "Saamia", "Lugwere": "Gwere", "Rufumbira": "Rufumbira",
    "Kupsabinyi": "Kupsabiny", "Sabiny": "Kupsabiny", "Lunyole": "Nyole", "Kakwa": "Kakwa",
    "Runyarwanda": "Kinyarwanda", "Kumam": "Kumam", "Lebtur": "Labwor",
    "Rutagwenda": "Tagwenda", "Swahili": "Swahili", "Kiswahili": "Swahili",
    "Lululi": "Ruuli", "Rululi": "Ruuli", "Lwamba": "Amba", "Nubian": "Nubi",
    "Aringa": "Aringa",
    # R5 only: "Lunyoro" beside "Runyoro"; 19 of its 20 answers are Banyole -> Lunyole
    "Lunyoro": "Nyole",
    # R5 only: "Luo" in place of Acholi (120 of 123 answers Acholi): the respondent's Luo
    "Luo": "LUO",
}
NON_ANSWERS = {"Don't know", "Missing", "Refused"}
VERBATIM = {
    "KUMAM": "Kumam", "AKUMU": "Kumam", "RUNYARWANDA": "Kinyarwanda",
    "KINYARWANDA": "Kinyarwanda", "KAKWA": "Kakwa", "LULAMOGI": "Soga", "MULAMOGI": "Soga",
    "LUFUMBIRA": "Rufumbira", "RUFUMBIRA": "Rufumbira", "LUGUNGU": "Gungu",
    "RUNGUNGU": "Gungu", "RUGUNGU": "Gungu", "BAGUNGU": "Gungu", "RUKONJO": "Konzo",
    "RUBWISI": "Bwisi", "LUBWISI": "Bwisi", "MUBWISI": "Bwisi", "LEBTHUR": "Labwor",
    "LEBTUR": "Labwor", "POKOT": "Pokot", "LUO": "LUO", "RULUULI": "Ruuli", "RURULI": "Ruuli",
    "LURURU": "Ruuli", "LULURI": "Ruuli", "BAKENYI": "Kenyi", "LUKENE": "Kenyi",
    "LENDU": "Lendu", "RUSONGORA": "Songora", "MUSUNGORA": "Songora", "LUDAMA": "Adhola",
    "KEBU": "Ndo", "KEBU.": "Ndo", "OKEBU": "Ndo", "RULUNDI": "Kirundi", "MURUNDI": "Kirundi",
    "LWAMBA": "Amba", "MWAMBA": "Amba", "NUBI": "Nubi", "KINUBI": "Nubi",
    "LUNYALA": "Nyara", "LUNYARA": "Nyara", "MUNYABINDI": "Tooro",
    "KARAMAJONG": "Karamojong", "ARINGA": "Aringa", "LUGWERE": "Gwere",
    "LUJALUWO": "Luo (Dholuo)", "BALUYA": "Luhya", "CHOPE": "Chope", "LUGISU": "Masaaba",
    "RUTAGWENDA": "Tagwenda", "RUTAGWENDA/RUTOORO": "Tagwenda", "MADI": "Ma'di",
    "LUSUDAN": "Other African language", "LUSUDANI": "Other African language",
    "TANZANIAN": "Other African language",
}
LUO_OF = {562: "Lango", 513: "Alur", 555: "Adhola"}   # else Acholi

# Afrobarometer tribe (label or upper-cased verbatim) -> census HH_P10 code.
ETH = {
    "Muganda": 519, "Munyankole": 531, "Musoga": 541, "Ateso": 552, "Langi": 562,
    "Mukiga": 527, "Mugishu": 520, "Acholi": 511, "Lugbara": 564, "Munyoro": 537, "Alur": 513,
    "Mutooro": 544, "Karamajong": 558, "Karimojong": 558, "Mukhonjo": 528, "Mukhonzo": 528,
    "Madi": 565, "Japhadhola": 555, "Musamia": 540, "Mugwere": 523, "Munyole": 536,
    "Mufumbira": 518, "Sabini": 574, "Sabinyi": 574, "Kupsabiny": 574, "Munyarwanda": 535,
    "Kakwa": 557, "Kumam": 561, "Mululi": 538, "Mutagwenda": 543, "Aliba": 512,
    "Nubian": 570, "Aringa": 514, "Mwamba": 515, "Mulamogi": 541,
    # verbatims under Other / Others
    "MUFUMBIRA": 518, "MUFUBIRA": 518, "BAFUMBIRA": 518, "RUFUMBIRA": 518, "KUMAM": 561,
    "AKUM": 561, "MUNYARWANDA": 535, "MUNYALWANDA": 535, "MUGUNGU": 521, "MWAMBA": 515,
    "MUBWISI": 517, "MULAMOGI": 541, "KAKWA": 557, "BAKAKWA": 557, "ARINGA": 514,
    "MUKENYI": 526, "MUKENYE": 526, "BAKENYE": 526, "BAKINYI": 526, "MUKENE": 526,
    "MUHORORO": 525, "MURULI": 538, "MULULI": 538, "MULUULI": 538, "MUKONJO": 528,
    "MUDAAMA": 555, "MUDAMA": 555, "MUNYALA": 532, "MUNYARA": 532, "KEBU": 559, "OKEBU": 559,
    "LENDU": 563, "POKOT": 572, "MURUNDI": 539, "MULUNDI": 539, "NUBIAN": 570,
    "MUNYABINDI": 529, "THUR": 549, "LABURO": 549, "MUSONGORA": 542, "MUSUNGORA": 542,
    "MUGWERE": 523, "MUGWE": 522, "BABUKUSU": 516, "BALUYA": 516, "CHOPE": 547, "JONAM": 554,
    "MUTAGWENDA": 543, "MUHIMA": 531, "MUTUKU": 545, "GIMARA": 550, "MUSEBEI": 574,
    "KARAMAJONG": 558, "MUGISU": 520, "LUO": 511,
}

# Afrobarometer region -> the five regions used here.
REGION5 = {"Central": "Central", "Kampala": "Kampala", "East": "East", "North": "North",
           "West": "West", "Buganda": "Central", "Central/Buganda": "Central",
           "Eastern": "East", "Busoga": "East", "Tororo": "East", "Acholi": "North",
           "Lango": "North", "Karamoja": "North", "West Nile": "North", "Ankole": "West",
           "Ankore": "West", "Kigezi": "West", "Tooro": "West", "Bunyoro": "West"}
CENSUS_REGION = {"1": "Central", "2": "East", "3": "North", "4": "West"}
DISTRICT_ALIAS = {"SEMBABULE": "SSEMBABULE", "MADI OKOLLO": "MADI-OKOLLO", "LUWERO": "LUWEERO",
                  "ENTEBBE CITY": "WAKISO"}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def districts():
    d = json.load(open(DISTRICTS, encoding="utf-8"))
    rows = d["population_data"]
    out = {}
    for r in rows:
        name = r["name"].strip().upper()
        reg = "Kampala" if name == "KAMPALA" else CENSUS_REGION[str(r["region_code"])]
        out[int(r["code"])] = (name, reg)
    return out


# ---------------------------------------------------------------- census

def census():
    s = pd.read_csv(SAMPLE)
    say(int(s["persons"].sum()) == 4_693_190, f"sample rows {int(s['persons'].sum()):,} = 4,693,190")
    hh = s[s["qrtype"] == 1]
    say(int(hh["persons"].sum()) == 4_540_805,
        f"household members {int(hh['persons'].sum()):,} = 4,540,805 (religiondots' QRTYPE 1)")
    blank = int(hh.loc[hh["p10"] < 0, "persons"].sum())
    print(f"  household members with no tribe/nationality: {blank:,}")
    hh = hh[hh["p10"] >= 0].copy()
    hh["unit"] = [drawn_unit(f"UG{d:03d}{c:02d}{x:02d}") for d, c, x in
                  zip(hh["district"], hh["county"], hh["subcounty"])]
    # base: religiondots' full-count household population per drawn unit
    rd = pd.read_csv(RD_UG)
    tot = rd[rd["source_category"] == "Total population"].groupby("geo_id")["count"].sum()
    nh = rd[rd["source_category"] == "Not in a household"].groupby("geo_id")["count"].sum()
    base = (tot - nh.reindex(tot.index, fill_value=0)).rename("pop")
    names = rd.drop_duplicates("geo_id").set_index("geo_id")["geo_name"]
    say(len(base) == UNITS, f"{len(base):,} drawn units in religiondots' ug.csv")
    say(abs(base.sum() - HHPOP) < 1, f"household population {base.sum():,.0f} = {HHPOP:,}")
    su = set(hh["unit"])
    say(su == set(base.index), f"sample units = base units both ways "
        f"({len(su - set(base.index))} sample-only, {len(set(base.index) - su)} base-only)")
    tab = hh.groupby(["unit", "p10"])["persons"].sum().unstack(fill_value=0)
    return tab, base, names


def p10_labels():
    lab = {}
    for line in open(P10_LABELS, encoding="utf-8"):
        k, v = line.rstrip("\n").split("\t", 1)
        lab[int(k)] = v.strip()
    return lab


# ---------------------------------------------------------------- survey

def survey(dist):
    a = pd.read_csv(AB, dtype=str, keep_default_na=False)
    a["round"] = a["round"].astype(int)
    a["w"] = a["w"].astype(float)
    say(len(a) == 12_031, f"{len(a):,} respondents in the extract")
    byname = {n: r for n, r in dist.values()}

    def reg(row):
        if row["round"] in (7, 9) and row["district"]:
            k = row["district"].strip().upper()
            k = DISTRICT_ALIAS.get(k, k)
            if k not in byname:
                k = re.sub(r" CITY$", "", k)
            if k in byname:
                return byname[k]
            if row["round"] == 7:
                raise SystemExit(f"R7 district {row['district']!r} not a census district")
        return REGION5.get(row["region"].strip())
    a["reg"] = a.apply(reg, axis=1)
    say(a["reg"].notna().all(), f"every respondent has a region "
        f"({sorted(set(a.loc[a['reg'].isna(), 'region']))})")
    # R7's REGION column is scrambled (its "Acholi" holds Buikwe and Kayunga): region from
    # its district instead. Show how far R7's REGION and district disagree.
    r7 = a[a["round"] == 7]
    agree = (r7["region"].map(REGION5) == r7["reg"]).mean()
    print(f"  R7: REGION agrees with its district's region for {agree:.0%} (so district used)")
    r9 = a[a["round"] == 9]
    agree9 = (r9["region"].map(REGION5).replace({"Central": "Central"}) == r9["reg"]).mean()
    print(f"  R9: subregion agrees with district region for {agree9:.0%}")

    def eth(row):
        e = row["eth"].strip()
        if e in ("Other", "Others"):
            return ETH.get(" ".join(row["eth_verbatim"].upper().split()))
        return ETH.get(e)
    a["g"] = a.apply(eth, axis=1)

    def ans(row):
        lab = row["lang"].strip()
        if lab in NON_ANSWERS:
            return None
        if lab in ("Other", "Others"):
            v = " ".join(row["verbatim"].upper().split())
            x = VERBATIM.get(v, "OWN")       # an unidentified "Other": the group's own
        else:
            if lab not in CODED:
                raise SystemExit(f"card label {lab!r} not in CODED")
            x = CODED[lab]
        if x == "LUO":
            x = LUO_OF.get(row["g"], "Acholi")
        if x == "OWN":
            x = OWN.get(row["g"]) if row["g"] == row["g"] and row["g"] is not None else None
            x = x or "Other Ugandan language"
        return x
    a["answer"] = a.apply(ans, axis=1)
    n0 = len(a)
    a = a[a["answer"].notna()]
    print(f"  {n0 - len(a)} non-answers dropped")
    nog = a["g"].isna()
    print(f"  {int(nog.sum())} respondents name no census tribe (national identity, refused, "
          f"foreign, unidentified): used only in the regional lingua-franca rates")
    return a


def lf_set(g):
    return {"English", "Swahili"} | ({"Ganda"} if g != 519 else set())


def retention(a, groups):
    """(group, region) -> Series answer -> share."""
    regions = ["Central", "Kampala", "East", "North", "West"]
    lfr = a[a["round"].isin(LF_ROUNDS) & (a["g"] != 519)]
    gen_nat = lfr.groupby("answer")["w"].sum() / lfr["w"].sum()
    gen_reg = {r: (lambda s: s.groupby("answer")["w"].sum() / s["w"].sum())(lfr[lfr["reg"] == r])
               for r in regions}
    out, diag = {}, []
    for g in groups:
        own = OWN[g]
        L = sorted(lf_set(g))
        sg = a[a["g"] == g]
        # A: lingua-franca shares, LF_ROUNDS only
        sa = sg[sg["round"].isin(LF_ROUNDS)]
        na = sa["w"].sum()
        pa = sa[sa["answer"].isin(L)].groupby("answer")["w"].sum().reindex(L, fill_value=0)
        qa = (pa + K0 * gen_nat.reindex(L, fill_value=0)) / (na + K0)
        # B: everything else, all rounds
        sb = sg[~sg["answer"].isin(L)]
        nb = sb["w"].sum()
        cb = sb.groupby("answer")["w"].sum()
        idx = cb.index.union([own])
        delta = pd.Series(0.0, index=idx)
        delta[own] = 1.0
        qb = (cb.reindex(idx, fill_value=0) + K0 * delta) / (nb + K0)
        for r in regions:
            mix = (qa + gen_reg[r].reindex(L, fill_value=0) - gen_nat.reindex(L, fill_value=0)
                   ).clip(lower=0)
            sar = sa[sa["reg"] == r]
            nar = sar["w"].sum()
            par = sar[sar["answer"].isin(L)].groupby("answer")["w"].sum().reindex(L, fill_value=0)
            A = (par + K * mix) / (nar + K)
            sbr = sb[sb["reg"] == r]
            nbr = sbr["w"].sum()
            cbr = sbr.groupby("answer")["w"].sum()
            i2 = qb.index.union(cbr.index)
            B = (cbr.reindex(i2, fill_value=0) + K * qb.reindex(i2, fill_value=0)) / (nbr + K)
            B = B / B.sum()
            P = pd.concat([A[A > 0], (1 - A.sum()) * B]).groupby(level=0).sum()
            out[(g, r)] = P
            diag.append((g, r, len(sg), round(nar), round(nbr), round(float(P.get(own, 0)), 4),
                         round(float(A.sum()), 4)))
    return out, pd.DataFrame(diag, columns=["p10", "region", "n_group", "n_lf_region",
                                            "n_other_region", "own_share", "lf_share"])


# ---------------------------------------------------------------- compose

def main():
    dist = districts()
    tab, base, names = census()
    lab = p10_labels()
    a = survey(dist)
    groups = [c for c in tab.columns if c in OWN]
    unknown = [c for c in tab.columns if c not in OWN and c != UGANDA and c < 700]
    say(not unknown, f"every Ugandan tribe code has an own language ({unknown})")
    foreign = [c for c in tab.columns if c >= 700 and c != UNKNOWN]
    for c in foreign:
        if c not in lab:
            raise SystemExit(f"nationality code {c} has no label")
    ret, diag = retention(a, groups)
    diag["tribe"] = diag["p10"].map(lab)
    diag.to_csv(RAW / "ug_retention.csv", index=False)

    shares = tab.div(tab.sum(axis=1), axis=0)
    rows = []
    kisoro = [k for k, (n, _) in dist.items() if n == "KISORO"]
    say(len(kisoro) == 1, "Kisoro district found (Batwa)")
    for u in shares.index:
        d = int(u[2:5])
        dname, reg = dist[d]
        pop = base[u]
        acc = {}
        for c in groups:
            s = shares.at[u, c]
            if s <= 0:
                continue
            P = ret[(c, reg)]
            if OWN[c] == "BATWA":
                own = "Rufumbira" if d == kisoro[0] else "Chiga"
                P = P.rename(index={"BATWA": own}).groupby(level=0).sum()
            for k, v in P.items():
                acc[k] = acc.get(k, 0.0) + s * pop * v
        ug_total = sum(acc.values())
        s500 = sum(shares.at[u, c] for c in (UGANDA, UNKNOWN) if c in shares.columns)
        if s500 > 0:
            # "Uganda", no tribe: shared like the unit's other Ugandans
            for k in list(acc):
                acc[k] += s500 * pop * acc[k] / ug_total
        for c in foreign:
            s = shares.at[u, c]
            if s <= 0:
                continue
            nat = lab[c]
            k = FOREIGN.get(nat) or ("Other African language" if nat in AFRICA
                                     else "Other language")
            acc[k] = acc.get(k, 0.0) + s * pop
        for k, v in acc.items():
            rows.append((u, k, v))
    df = pd.DataFrame(rows, columns=["geo_id", "source_category", "count"])
    say(not df["source_category"].isin(["BATWA"]).any(), "Batwa resolved")
    t = df.groupby("geo_id")["count"].sum()
    say((t - base.reindex(t.index)).abs().max() < 0.01, "every unit sums to its population")
    say(abs(df["count"].sum() - HHPOP) < 1, f"drawn total {df['count'].sum():,.0f} = {HHPOP:,}")
    df["geo_level"] = "subcounty_2024"
    df["geo_name"] = df["geo_id"].map(names)
    df["tier"] = "modelled"
    df["source_id"] = SOURCE_ID
    df["year"] = YEAR
    df["note"] = ""
    df["count"] = df["count"].round(2)
    df = df[["geo_id", "geo_level", "geo_name", "source_category", "count", "tier",
             "source_id", "year", "note"]].sort_values(["geo_id", "source_category"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} rows, {df['geo_id'].nunique():,} units")

    # report
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\n  drawn, national:")
    for k, v in nat.items():
        print(f"    {k:28s} {v:12,.0f}  {100 * v / HHPOP:5.2f}%")
    eth = tab.sum() / tab.sum().sum()
    print("\n  census tribe shares (sample, household members), top 25:")
    for c, v in eth.sort_values(ascending=False).head(25).items():
        print(f"    {c} {lab.get(c, '?'):28s} {100 * v:5.2f}%")
    print("\n  retention (own-language share, national mix of regions), big groups:")
    big = eth[eth > 0.004].index
    for c in big:
        if c in OWN:
            d = diag[diag["p10"] == c]
            print(f"    {lab[c]:16s} n={int(d['n_group'].iloc[0]):5d}  own "
                  + " ".join(f"{r[:3]} {o:.2f}" for r, o in zip(d["region"], d["own_share"])))


if __name__ == "__main__":
    main()
