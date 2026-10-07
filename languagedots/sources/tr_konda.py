"""Türkiye: mother tongue modelled for the 81 provinces from KONDA's *Biz Kimiz?* survey (2006),
placed with the 1965 census, plus Syrians under temporary protection (DGMM, 1 Oct 2026)
-> data/normalized/tr.csv.

    python sources/tr_konda.py

No Turkish census has asked mother tongue since 1965, and TÜİK publishes no survey table of it.
What this builds from, all in data/raw/tr/ (the record is sources/tr.md):

  KONDA 2006   *Biz Kimiz? Toplumsal Yapı Araştırması* (KONDA for Milliyet, Sept 2006, 47,958
               adults face to face in 79 provinces), Tablo 7 p.19: mother tongue ("annenizden
               öğrendiğiniz konuşma diliniz"), 15 answers, national only.
               2006_09_KONDA_Toplumsal_Yapi.pdf, the Wayback copy of konda.com.tr's file.
  KONDA 2008   *Kürtler ve Kürt Sorunu* (Nov 2008) p.5: from the 2006 survey, the share of
               "Kürt ve Zaza" in each of the 12 İBBS-1 regions, children included (its Toplam
               row is 15.7, KONDA's whole-population figure on p.4); p.4 gives the adult figure
               13.40. KONDA counts as Kurd or Zaza anyone who said so OR whose mother tongue is
               Kurdish or Zazaki. The same regional table is reprinted in the 2010 report.
  1965 census  mother tongue by province (67 then), 13 languages, as transcribed on Wikipedia's
               "1965 Turkish census" from Ahmet Buran, *Türkiye'de Diller ve Etnik Gruplar* (2012).
               Used ONLY to place people: never to set a count (AGENT_BRIEF §4.4).
  DGMM 2026    Göç İdaresi's "Geçici koruma kapsamındaki Suriyelilerin illere göre dağılımı",
               1 Oct 2026, transcribed from its image to dgmm_gecici_koruma_iller_20261001.csv:
               Syrians under temporary protection and the province's register population
               (ADNKS, 86,092,168 in all, which leaves the Syrians out), per province.

THE MODEL, per province p in region r (population P = DGMM's il nüfusu):
  1. Kurdish + Zazaki (Kürtçe, Zazaca): region total = P_r x KONDA's regional Kurd-and-Zaza
     share x 0.9687, the adult ratio of mother tongue (11.97 + 1.01) to identity (13.40). Zazaki's
     part of it is the 1965 census's Zazaki / (Kurdish + Zazaki) in Kuzeydoğu, Ortadoğu and
     Güneydoğu Anadolu (where the speakers are native) and the 1965 national ratio elsewhere
     (where they are migrants from all three), all scaled by one factor so the national result
     is KONDA's 1.01 / 12.98. Inside the three eastern regions, provinces get them by an IPF
     seeded with 1965 rates (Hakkari's Kurdish, Tunceli's Zazaki); elsewhere by population.
  2. The other twelve answers: KONDA's national share x the national population, spread over
     provinces by the 1965 rate of the matching language x today's population (Arabic by
     Arabic, Laz by Laz, Balkan by Pomak + Bosnian + Albanian, Kafkas by Georgian, Yahudice by
     the census's "Jewish"); Türki Diller, Kıptice, Batı Avrupa and Diğer by population.
  3. Turkish is the rest of each province.
  4. Syrians under temporary protection, on top of the register population, as Arabic: an
     immigrant proxy (AGENT_BRIEF §2), `derived`. Everything else is `modelled`.

THE CHECKS: KONDA's table sums to 100 with two anchors pinned; the regional table's 12 rows and
its 15.7 total; the 1965 table's 67 rows and its parents cover all 81 provinces; DGMM's rows
satisfy syrians + il nüfusu = toplam on every row and sum to the 2,206,483 and 86,092,168 it
prints; no province's Turkish remainder goes negative; the output sums to the population.
"""
import csv
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "tr"
OUT = HERE / "data" / "normalized" / "tr.csv"
KONDA06 = RAW / "2006_09_KONDA_Toplumsal_Yapi.pdf"
KONDA08 = RAW / "2008_11_KONDA_Kurtler_ve_Kurt_Sorunu.pdf"
WIKI = RAW / "wikipedia_1965_Turkish_census.wikitext"
DGMM = RAW / "dgmm_gecici_koruma_iller_20261001.csv"
DGMM_SYRIANS, DGMM_POP = 2_206_483, 86_092_168
ADULT_KZ_IDENTITY = 13.40          # KONDA 2008 p.4, "Kürt ve Zaza", adults, 2006 survey
SYRIAN = "Arapça (Suriyeli, geçici koruma)"

# İBBS-1, as religiondots' sources/tr.py writes it out (ASCII names = COD-AB adm1_name)
IBBS1 = {
    "TR1": ["Istanbul"],
    "TR2": ["Tekirdag", "Edirne", "Kirklareli", "Balikesir", "Canakkale"],
    "TR3": ["Izmir", "Aydin", "Denizli", "Mugla", "Manisa", "Afyonkarahisar", "Kutahya", "Usak"],
    "TR4": ["Bursa", "Eskisehir", "Bilecik", "Kocaeli", "Sakarya", "Duzce", "Bolu", "Yalova"],
    "TR5": ["Ankara", "Konya", "Karaman"],
    "TR6": ["Antalya", "Isparta", "Burdur", "Adana", "Mersin", "Hatay", "Kahramanmaras",
            "Osmaniye"],
    "TR7": ["Kirikkale", "Aksaray", "Nigde", "Nevsehir", "Kirsehir", "Kayseri", "Sivas", "Yozgat"],
    "TR8": ["Zonguldak", "Karabuk", "Bartin", "Kastamonu", "Cankiri", "Sinop", "Samsun", "Tokat",
            "Corum", "Amasya"],
    "TR9": ["Trabzon", "Ordu", "Giresun", "Rize", "Artvin", "Gumushane"],
    "TRA": ["Erzurum", "Erzincan", "Bayburt", "Agri", "Kars", "Igdir", "Ardahan"],
    "TRB": ["Malatya", "Elazig", "Bingol", "Tunceli", "Van", "Mus", "Bitlis", "Hakkari"],
    "TRC": ["Gaziantep", "Adiyaman", "Kilis", "Sanliurfa", "Diyarbakir", "Mardin", "Batman",
            "Sirnak", "Siirt"],
}
EAST = {"TRA", "TRB", "TRC"}       # where Kurdish and Zazaki speakers are native
KONDA_REGION = {"İstanbul": "TR1", "Batı Marmara": "TR2", "Ege": "TR3", "Doğu Marmara": "TR4",
                "Batı Anadolu": "TR5", "Akdeniz": "TR6", "Orta Anadolu": "TR7",
                "Batı Karadeniz": "TR8", "Doğu Karadeniz": "TR9", "Kuzeydoğu Anadolu": "TRA",
                "Ortadoğu Anadolu": "TRB", "Güneydoğu Anadolu": "TRC"}

# Today's provinces split from a 1965 one take its 1965 rates. Batman and Şırnak were cut from
# Siirt and Mardin (and a little of Hakkari) and take the two pooled; Karabük's Eskipazar came from
# Çankırı, the rest of it from Zonguldak. Yalova was a district of İstanbul.
PARENT_1965 = {"Osmaniye": ["Adana"], "Kirikkale": ["Ankara"], "Duzce": ["Bolu"],
               "Karabuk": ["Zonguldak"], "Bartin": ["Zonguldak"], "Bayburt": ["Gumushane"],
               "Ardahan": ["Kars"], "Igdir": ["Kars"], "Aksaray": ["Nigde"],
               "Karaman": ["Konya"], "Kilis": ["Gaziantep"], "Yalova": ["Istanbul"],
               "Batman": ["Siirt", "Mardin"], "Sirnak": ["Siirt", "Mardin"]}
W65_COLS = ["Turkish", "Kurdish", "Arabic", "Zazaki", "Circassian", "Greek", "Georgian",
            "Armenian", "Laz", "Pomak", "Bosnian", "Albanian", "Jewish"]

# KONDA label -> the 1965 columns that place it (None: by population)
PLACE_BY = {"Arapça": ["Arabic"], "Ermenice": ["Armenian"], "Rumca": ["Greek"],
            "Yahudice": ["Jewish"], "Balkan": ["Pomak", "Bosnian", "Albanian"],
            "Kafkas": ["Georgian"], "Lazca": ["Laz"], "Çerkesçe": ["Circassian"],
            "Türki Diller": None, "Kıptice": None, "Batı Avrupa": None, "Diğer": None}


def ascii_tr(s):
    t = str.maketrans("ÇĞİIÖŞÜçğıöşüâÂîÎ", "CGIIOSUcgiosuaAiI")
    return s.translate(t)


def num(s):
    return float(s.replace(".", "").replace(",", "."))


def read_konda06():
    import fitz
    lines = [x.strip() for x in fitz.open(KONDA06)[22].get_text().split("\n") if x.strip()]
    if lines[0] != "Anadil" or "Tablo 7: Anadil Dağılımı" not in lines:
        raise SystemExit("KONDA 2006 p.23 is not Tablo 7")
    out, i = {}, lines.index("1")
    while lines[i + 1] != "Toplam":
        out[lines[i + 1]] = num(lines[i + 2])
        i += 4
    if abs(sum(out.values()) - 100) > 0.02 or out["Türkçe"] != 84.54 or out["Kürtçe"] != 11.97:
        raise SystemExit(f"Tablo 7 off: sum {sum(out.values())}, {out}")
    return out


def read_konda08():
    import fitz
    txt = fitz.open(KONDA08)[4].get_text()
    # the PDF's font maps İ and ş to Ġ and Ģ; undo for matching
    txt = txt.replace("Ġ", "İ").replace("Ģ", "ş")
    lines = [x.strip() for x in txt.split("\n") if x.strip()]
    out = {}
    for name, code in KONDA_REGION.items():
        i = lines.index(name)
        out[code] = num(lines[i + 2])            # "Bölge içinde Kürtler %"
    t = lines.index("Toplam")
    if num(lines[t + 2]) != 15.7 or out["TRB"] != 79.1 or out["TR1"] != 14.8:
        raise SystemExit(f"KONDA 2008 p.5 off: {out}")
    return out


def read_1965():
    t = WIKI.read_text(encoding="utf-8")
    a = t.index("=== Provincial level ===")
    b = t.index("=== Maps of provinces", a)
    blocks = t[a:b].split("|-")
    head = blocks[0]
    cols = re.findall(r"!\[\[[^|\]]*\|([^\]]*)\]\]", head)
    if cols[1:] != W65_COLS:
        raise SystemExit(f"1965 table columns changed: {cols}")
    out = {}
    for blk in blocks[1:]:
        ls = [x for x in blk.strip().split("\n") if x.startswith("|")]
        if not ls or ls[0].startswith("|}"):
            continue
        m = re.search(r"\[\[([^|\]]*)(?:\|([^\]]*))?\]\]", ls[0])
        name = ascii_tr((m.group(2) or m.group(1)).replace(" Province", ""))
        vals = [int(x.lstrip("|").strip().replace(",", "") or 0) for x in ls[1:14]]
        out[name] = dict(zip(W65_COLS, vals))
    if len(out) != 67:
        raise SystemExit(f"1965 table: {len(out)} provinces, expected 67")
    return out


def read_dgmm():
    rows = list(csv.DictReader(open(DGMM, encoding="utf-8")))
    for r in rows:
        if int(r["syrians"]) + int(r["il_nufusu"]) != int(r["toplam"]):
            raise SystemExit(f"DGMM row does not add up: {r}")
    s = sum(int(r["syrians"]) for r in rows)
    p = sum(int(r["il_nufusu"]) for r in rows)
    if (s, p, len(rows)) != (DGMM_SYRIANS, DGMM_POP, 81):
        raise SystemExit(f"DGMM totals {s:,} / {p:,} / {len(rows)} rows")
    return ({r["province"]: int(r["il_nufusu"]) for r in rows},
            {r["province"]: int(r["syrians"]) for r in rows})


def ipf(seed, rows, cols, iters=500):
    """seed[p][c] >= 0; returns a table with row sums `rows` and column sums `cols`."""
    x = {p: dict(seed[p]) for p in seed}
    for _ in range(iters):
        for p in x:
            s = sum(x[p].values())
            for c in x[p]:
                x[p][c] *= rows[p] / s if s else 0
        for c in cols:
            s = sum(x[p][c] for p in x)
            for p in x:
                x[p][c] *= cols[c] / s if s else 0
    err = max(abs(sum(x[p].values()) - rows[p]) for p in x)
    if err > 1:
        raise SystemExit(f"IPF did not converge ({err:.1f})")
    return x


def largest_remainder(vals, total):
    fl = {k: int(v) for k, v in vals.items()}
    rest = total - sum(fl.values())
    for k in sorted(vals, key=lambda k: vals[k] - fl[k], reverse=True)[:rest]:
        fl[k] += 1
    return fl


def main():
    k06 = read_konda06()
    kreg = read_konda08()
    w65 = read_1965()
    pop, syr = read_dgmm()
    prov_region = {p: r for r, ps in IBBS1.items() for p in ps}
    if set(prov_region) != set(pop) or len(prov_region) != 81:
        raise SystemExit("İBBS-1 crosswalk and DGMM disagree")
    parents = {p: PARENT_1965.get(p, [p]) for p in pop}
    missing = sorted({q for ps in parents.values() for q in ps} - set(w65))
    unused = sorted(set(w65) - {q for ps in parents.values() for q in ps})
    if missing or unused:
        raise SystemExit(f"1965 parents: missing {missing}, unused {unused}")

    def rate(p, langs):
        tot = sum(sum(w65[q].values()) for q in parents[p])
        return sum(w65[q][c] for q in parents[p] for c in langs) / tot

    P = sum(pop.values())
    counts = {p: {} for p in pop}

    # 2. the minor answers, national total placed by 1965 rate x today's population
    for lab, langs in PLACE_BY.items():
        total = k06[lab] / 100 * P
        w = {p: pop[p] * (rate(p, langs) if langs else 1.0) for p in pop}
        sw = sum(w.values())
        for p in pop:
            counts[p][lab] = total * w[p] / sw

    # 1. Kurdish and Zazaki
    ratio_mt = (k06["Kürtçe"] + k06["Zazaca"]) / ADULT_KZ_IDENTITY
    nat65 = sum(v["Zazaki"] for v in w65.values()) / sum(v["Zazaki"] + v["Kurdish"]
                                                        for v in w65.values())
    kz_r, z_raw = {}, {}
    for r, ps in IBBS1.items():
        kz_r[r] = sum(pop[p] for p in ps) * kreg[r] / 100 * ratio_mt
        if r in EAST:
            # Batman/Şırnak share parents with Siirt/Mardin: count each 1965 province once
            qs = {q for p in ps for q in parents[p]}
            z = sum(w65[q]["Zazaki"] for q in qs)
            k = sum(w65[q]["Kurdish"] for q in qs)
            z_raw[r] = z / (z + k)
        else:
            z_raw[r] = nat65
    want = k06["Zazaca"] / (k06["Kürtçe"] + k06["Zazaca"])
    scale = want * sum(kz_r.values()) / sum(kz_r[r] * z_raw[r] for r in kz_r)
    print(f"  Kurd+Zaza mother tongue / identity, adults: {ratio_mt:.4f}; Zazaki share of the "
          f"two {want:.4f}, 1965 national {nat65:.4f}, factor {scale:.3f}")
    for r, ps in IBBS1.items():
        zt = kz_r[r] * z_raw[r] * scale
        cols = {"Kürtçe": kz_r[r] - zt, "Zazaca": zt}
        rows = {p: pop[p] - sum(counts[p].values()) for p in ps}
        cols["rest"] = sum(rows.values()) - cols["Kürtçe"] - cols["Zazaca"]
        if r in EAST:
            seed = {p: {"Kürtçe": rate(p, ["Kurdish"]), "Zazaca": rate(p, ["Zazaki"]),
                        "rest": 1 - rate(p, ["Kurdish", "Zazaki"])} for p in ps}
            # a province the 1965 census gave no Zazaki still gets a trace, so IPF can move
            for p in ps:
                for c in seed[p]:
                    seed[p][c] = max(seed[p][c], 1e-4) * pop[p]
        else:
            seed = {p: {c: pop[p] for c in cols} for p in ps}
        fit = ipf(seed, rows, cols)
        for p in ps:
            counts[p]["Kürtçe"] = fit[p]["Kürtçe"]
            counts[p]["Zazaca"] = fit[p]["Zazaca"]
        pr = sum(pop[p] for p in ps)
        print(f"  {r}: Kurd+Zaza {kreg[r]:5.1f}% -> Kurdish {cols['Kürtçe'] / pr * 100:5.1f}%, "
              f"Zazaki {cols['Zazaca'] / pr * 100:4.1f}%  (1965 Zazaki/(K+Z) {z_raw[r]:.3f})")

    # 3. Turkish is the rest; integers by largest remainder within each province
    out = []
    for p in sorted(pop):
        rest = pop[p] - sum(counts[p].values())
        if rest < 0:
            raise SystemExit(f"{p}: languages exceed the population by {-rest:,.0f}")
        counts[p]["Türkçe"] = rest
        ints = largest_remainder(counts[p], pop[p])
        for lab, n in sorted(ints.items()):
            if n:
                out.append(dict(geo_id=p, geo_level="province", geo_name=p, region=prov_region[p],
                                source_category=lab, count=n, tier="modelled",
                                source_id="konda_2006_tr", year=2006))
        if syr[p]:
            out.append(dict(geo_id=p, geo_level="province", geo_name=p, region=prov_region[p],
                            source_category=SYRIAN, count=syr[p], tier="derived",
                            source_id="dgmm_tp_20261001", year=2026))

    tot = sum(r["count"] for r in out)
    if tot != P + DGMM_SYRIANS:
        raise SystemExit(f"output sums to {tot:,}, expected {P + DGMM_SYRIANS:,}")
    nat = {}
    for r in out:
        nat[r["source_category"]] = nat.get(r["source_category"], 0) + r["count"]
    print(f"  {len(out)} rows, {tot:,} people ({P:,} register + {DGMM_SYRIANS:,} Syrians)")
    for lab, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        print(f"    {lab:<34} {n:>12,}  {n / P * 100:6.2f}%  (KONDA {k06.get(lab, '-')})")
    by = {}
    for r in out:
        if r["source_category"] in ("Kürtçe", "Zazaca"):
            by[r["geo_id"]] = by.get(r["geo_id"], 0) + r["count"]
    top = sorted(by.items(), key=lambda kv: -kv[1] / pop[kv[0]])[:12]
    print("  highest Kurdish+Zazaki shares: " + ", ".join(f"{p} {n / pop[p] * 100:.0f}%"
                                                         for p, n in top))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    tmp.replace(OUT)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
