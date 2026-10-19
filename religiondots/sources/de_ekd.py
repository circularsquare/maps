"""Germany: the Zensus `Evangelische Kirche` column split by Landeskirche. Built 2026-10-16.

WHAT THIS DOES
--------------
Zensus 2022 counts 19,127,360 members of the Evangelische Kirche at 10,786 Gemeinden, one undivided
column (`christianity.protestant`). The EKD is twenty Landeskirchen, eight Lutheran, nine United, two
Reformed and one (Mitteldeutschland) both. This script gives every Gemeinde its Landeskirche and writes
one row per Gemeinde naming it, so the column can be drawn on `christianity.lutheran`,
`christianity.united` and `christianity.reformed.continental` (taxonomy/de2022.py).

WHY A TERRITORY IS A COUNT AND NOT A MODEL
------------------------------------------
Kirchengesetz über die Kirchenmitgliedschaft (EKD, 1976, as of 2022) §1 Abs. 2: "Die Kirchenmitgliedschaft
besteht zur Kirchengemeinde und zur Gliedkirche des Wohnsitzes des Kirchenmitgliedes", Wohnsitz being the
Hauptwohnung under state Melderecht, which is the register the Zensus read. EKD's own Table 1 footnote 3:
membership across Landeskirche borders "würde saldiert weniger als 0,14 ‰ der Mitglieder ausmachen". So
a Gemeinde's Protestants are members of the church whose territory holds it; the split is a lookup.
This reversed the 2026-09-07 refusal (sources/de.md §12, spec §2.7b).

THE ONE MODELLED STEP: the Evangelisch-reformierte Kirche (ERK) is not territorial. Its members live
inside Hannover's territory and EKD says population cannot be assigned to it. The ERK's own 2010 report
(Bericht des Moderamens, April 2010) gives about 70,000 members in Ostfriesland, 60,000 in the Grafschaft
Bentheim and Emsland and 50,000 scattered. Scaled to its 2022 total (159,309) the first two are taken out
of Hannover's count in those six Kreise in proportion to each Gemeinde's Protestants; the scattered third
stays with the church whose territory it sits in.

THE CHECK, AND IT IS STRONG: the Zensus count summed over each territory against EKD's Table 1 for
31.12.2022 (Hannover with the ERK, since they share ground). Printed every run. A wrongly assigned Kreis
shows as a miss in two neighbouring churches. Population summed the same way is printed beside it; Zensus
populations run 1.9% under EKD's 2011-based ones nationally, so 0.98 there is a match.
Known misses, each a church border inside one Gemeinde that cannot be split: Schaumburg-Lippe (Steinhude
and Großenheidorn are in Wunstorf; Steinbergen in Rinteln; Frille in Petershagen), Braunschweig
(Wolfsburg's Fallersleben, Heiligendorf and Hattorf are Hannover's; the city is drawn whole as
Braunschweig's, whose Propstei Vorsfelde holds its core), Anhalt (Gernrode in Quedlinburg).

Usage:
    python sources/de_ekd.py --fetch    save EKD's report to data/raw/de/ekd/ and build
    python sources/de_ekd.py            build data/normalized/de_ekd.csv and print the check
"""

import os
import sys
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DE_CSV = os.path.join(HERE, "data", "normalized", "de.csv")
OUT = os.path.join(HERE, "data", "normalized", "de_ekd.csv")
RAW = os.path.join(HERE, "data", "raw", "de", "ekd", "Ber_Kirchenmitglieder_2022_r.pdf")
URL = "https://www.ekd.de/ekd_de/ds_doc/Ber_Kirchenmitglieder_2022_r.pdf"
EV = "Evangelische Kirche (öffentlich-rechtlich)"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AN, BA, BY, EKBO, BS, BRE, HAN, EKHN, EKKW, LIP, EKM, NK, OL, PF, RL, SN, SL, WF, WUE = (
    "Anhalt", "Baden", "Bayern", "Berlin-Brandenburg-schlesische Oberlausitz", "Braunschweig", "Bremen",
    "Hannover", "Hessen und Nassau", "Kurhessen-Waldeck", "Lippe", "Mitteldeutschland", "Nordkirche",
    "Oldenburg", "Pfalz", "Rheinland", "Sachsen", "Schaumburg-Lippe", "Westfalen", "Württemberg")
ERK = "Reformierte Kirche"

# EKD, Kirchenmitgliederzahlen Stand 31.12.2022, korrigierte Ausgabe Juli 2024, Table 1 (p. 6):
# members, population of the territory. Transcribed by hand from the PDF; they sum to 19,152,827.
EKD = {
    AN: (26236, 268548), BA: (1035524, 4729790), BY: (2143233, 13369393), EKBO: (833846, 6417325),
    BS: (293979, 823062), BRE: (163710, 594516), HAN: (2302547, 6238435), EKHN: (1361322, 5375784),
    EKKW: (730478, 1883669), LIP: (140313, 347757), EKM: (615855, 4212959), NK: (1772953, 6490771),
    OL: (371688, 1102701), PF: (456498, 1596080), ERK: (159309, None), RL: (2266794, 12525101),
    SN: (610503, 3745258), SL: (45764, 93424), WF: (2001009, 7993805), WUE: (1821266, 6550467),
}
assert sum(v[0] for v in EKD.values()) == 19152827

# The confession of each BODY (EKD's own grouping; VELKD and UEK membership). Mitteldeutschland is
# divided by Land on Anita's call, 2026-10-16: Thuringia's congregations came from the Lutheran
# Thuringian church, the rest from the Union's Kirchenprovinz Sachsen. (The Prussian parts of Thuringia,
# Erfurt, Mühlhausen and Nordhausen among them, were Kirchenprovinz too; the call is by Land.)
LABEL = {
    **{c: f"Evangelische Kirche: {c} (lutherisch)" for c in [BY, HAN, WUE, NK, SN, OL, BS, SL]},
    **{c: f"Evangelische Kirche: {c} (uniert)" for c in [AN, BA, EKBO, BRE, EKHN, EKKW, PF, RL, WF]},
    LIP: f"Evangelische Kirche: {LIP} (reformiert)",
    ERK: "Evangelische Kirche: Evangelisch-reformierte Kirche (reformiert)",
}
EKM_T = "Evangelische Kirche: Mitteldeutschland, Thüringen (lutherisch)"
EKM_U = "Evangelische Kirche: Mitteldeutschland, outside Thüringen (uniert)"

# ---- Land defaults (first two digits of the ARS)
LAND = {"01": NK, "02": NK, "03": HAN, "04": BRE, "05": RL, "06": EKHN, "07": RL, "08": WUE, "09": BY,
        "10": RL, "11": EKBO, "12": EKBO, "13": NK, "14": SN, "15": EKM, "16": EKM}

# ---- Kreis defaults (first five digits)
KREIS = {
    # Niedersachsen: the former Free States of Oldenburg and Braunschweig; Bremerhaven is Hannover's
    **{k: OL for k in ["03401", "03403", "03405", "03451", "03453", "03455", "03458", "03460", "03461"]},
    **{k: BS for k in ["03101", "03102", "03103", "03154", "03158", "03255"]},
    "04012": HAN,
    # NRW: Westphalia is Regierungsbezirke Münster, Detmold and Arnsberg; Kreis Lippe is Lippe
    **{k: WF for k in ["05512", "05513", "05515", "05554", "05558", "05562", "05566", "05570", "05711",
                       "05754", "05758", "05762", "05770", "05774", "05911", "05913", "05914", "05915",
                       "05916", "05954", "05958", "05962", "05966", "05970", "05974", "05978"]},
    "05766": LIP,
    # Hessen: Kurhessen, Waldeck and Hanau
    **{k: EKKW for k in ["06611", "06631", "06632", "06633", "06634", "06635", "06636", "06435", "06534"]},
    # Rheinland-Pfalz: Rheinhessen and Nassau to EKHN, the Palatinate to Pfalz; the rest Rhineland
    **{k: EKHN for k in ["07315", "07319", "07331", "07339", "07141", "07143"]},
    **{k: PF for k in ["07311", "07312", "07313", "07314", "07316", "07317", "07318", "07320", "07332",
                       "07333", "07334", "07335", "07336", "07337", "07338", "07340"]},
    "10045": PF,                                    # Saarpfalz-Kreis, Bavarian Palatinate until 1919
    # Baden-Württemberg: the Baden Kreise; the rest default to Württemberg (Hohenzollern included)
    **{k: BA for k in ["08211", "08212", "08215", "08216", "08221", "08222", "08225", "08226", "08231",
                       "08311", "08315", "08316", "08317", "08326", "08335", "08336", "08337"]},
    "15001": AN,                                    # Dessau-Roßlau
}

# ---- Gemeinde overrides: (Kreis, Zensus name prefixes, church). Each prefix must match exactly one
# Gemeinde in its Kreis, or the build stops.
GEMEINDE = [
    ("03157", ["Vechelde", "Wendeburg"], BS),
    ("03153", ["Bad Harzburg", "Braunlage", "Seesen"], BS),
    ("03155", ["Bad Gandersheim"], BS),
    ("03159", ["Walkenried"], BS),
    ("03257", ["Bückeburg", "Stadthagen", "Niedernwöhren", "Nienstädt", "Sachsenhagen", "Lindhorst",
               "Bad Eilsen", "Ahnsen", "Buchholz", "Heeßen", "Luhden", "Hagenburg", "Wölpinghausen",
               "Pollhagen", "Meerbeck", "Lauenhagen", "Nordsehl", "Wiedensahl", "Helpsen", "Hespe",
               "Seggebruch", "Lüdersfeld", "Heuerßen", "Auhagen", "Beckedorf"], SL),
    # Hessen: the Rhineland's former Kreis Wetzlar; the Biedenkopf hinterland is Hessen und Nassau's
    ("06532", ["Wetzlar", "Aßlar", "Braunfels", "Ehringshausen", "Hohenahr", "Hüttenberg", "Lahnau",
               "Leun", "Schöffengrund", "Solms", "Waldsolms"], RL),
    ("06531", ["Biebertal", "Wettenberg"], RL),
    ("06534", ["Biedenkopf", "Breidenbach", "Dautphetal", "Gladenbach", "Steffenberg", "Angelburg",
               "Bad Endbach"], EKHN),
    # Baden-Württemberg
    ("08125", ["Bad Rappenau", "Eppingen", "Gemmingen", "Ittlingen", "Kirchardt", "Siegelsbach"], BA),
    ("08128", ["Wertheim", "Tauberbischofsheim", "Lauda-Königshofen", "Boxberg", "Grünsfeld", "Külsheim",
               "Freudenberg", "Großrinderfeld", "Werbach", "Wittighausen", "Ahorn", "Königheim"], BA),
    ("08236", ["Eisingen", "Ispringen", "Kämpfelbach", "Keltern", "Kieselbronn", "Königsbach-Stein",
               "Neulingen", "Niefern-Öschelbronn", "Remchingen", "Tiefenbronn", "Neuhausen"], BA),
    # Villingen-Schwenningen is one Gemeinde across the old border; its Protestants are mostly
    # Schwenningen's, and Württemberg is the better fit by the check (Baden 1.023 -> 1.005).
    ("08326", ["Tuningen", "Villingen-Schwenningen"], WUE),
    ("08327", ["Immendingen", "Geisingen", "Emmingen-Liptingen", "Buchheim", "Irndorf",
               "Neuhausen ob Eck"], BA),
    ("08435", ["Überlingen", "Meersburg", "Salem", "Uhldingen-Mühlhofen", "Daisendorf", "Stetten",
               "Hagnau", "Frickingen", "Heiligenberg", "Owingen", "Sipplingen", "Bermatingen",
               "Markdorf", "Deggenhausertal", "Immenstaad"], BA),
    ("08437", ["Pfullendorf", "Meßkirch", "Sauldorf", "Wald", "Herdwangen", "Illmensee", "Stetten am kalten",
               "Leibertingen"], BA),
    # Thüringen: Schmalkalden is Kurhessen-Waldeck's
    ("16066", ["Schmalkalden", "Steinbach-Hallenberg", "Brotterode-Trusetal", "Floh-Seligenthal",
               "Fambach", "Breitungen", "Oberhof"], EKKW),
    # Sachsen-Anhalt: Anhalt
    ("15082", ["Köthen", "Zerbst", "Südliches Anhalt", "Osternienburger Land", "Aken", "Raguhn"], AN),
    ("15089", ["Bernburg", "Nienburg", "Güsten", "Ilberstedt"], AN),
    ("15085", ["Ballenstedt", "Harzgerode"], AN),
    ("15091", ["Coswig", "Oranienbaum"], AN),
    # Brandenburg: southern Elbe-Elster was the Prussian Province of Saxony
    ("12062", ["Bad Liebenwerda", "Elsterwerda", "Falkenberg", "Herzberg", "Mühlberg", "Uebigau",
               "Schönewalde", "Schlieben", "Kremitzaue", "Fichtwald", "Hohenleipisch", "Plessa", "Gröden",
               "Schraden", "Merzdorf", "Röderland", "Großthiemig", "Hirschfeld", "Lebusa",
               "Heideland", "Gorden-Staupitz"], EKM),
    # Sachsen: the Silesian Oberlausitz is EKBO's; the Prussian north of Nordsachsen is EKM's
    ("14626", ["Görlitz", "Niesky", "Weißwasser", "Rothenburg", "Bad Muskau", "Boxberg", "Krauschwitz",
               "Gablenz", "Hähnichen", "Horka", "Kodersdorf", "Mücka", "Neißeaue", "Quitzdorf",
               "Rietschen", "Schleife", "Trebendorf", "Groß Düben", "Hohendubrau", "Waldhufen", "Vierkirchen",
               "Markersdorf", "Schöpstal", "Weißkeißel", "Kreba-Neudorf", "Reichenbach", "Königshain",
               "Neusalza"], EKBO),
    ("14625", ["Hoyerswerda", "Lauta", "Bernsdorf", "Elsterheide", "Lohsa", "Spreetal", "Wittichenau"], EKBO),
    ("14730", ["Torgau", "Belgern", "Dommitzsch", "Arzberg", "Beilrode", "Mockrehna", "Trossin", "Elsnig",
               "Dreiheide", "Eilenburg", "Delitzsch", "Bad Düben", "Schkeuditz", "Laußig", "Löbnitz",
               "Rackwitz", "Schönwölkau", "Wiedemar", "Zschepplin", "Krostitz", "Doberschütz",
               "Jesewitz"], EKM),
]

# The ERK's two home regions (Bericht des Moderamens, April 2010): Kreise and 2010 members.
ERK_2010_TOTAL = 180000
ERK_REGIONS = {
    "Ostfriesland": (["03402", "03452", "03457", "03462"], 70000),
    "Grafschaft Bentheim and Emsland": (["03456", "03454"], 60000),
}

# The bar: every church's Zensus/EKD ratio inside this, and the churches over 500,000 inside the
# tighter one. Set after the first run (2026-10-16), so post hoc; it exists to catch a regression.
BAR_ALL = (0.85, 1.15)
BAR_BIG = (0.97, 1.03)


def fetch():
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "religiondots-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    tmp = RAW + ".tmp"
    with open(tmp, "wb") as f:
        f.write(body)
    os.replace(tmp, RAW)
    print(f"  {len(body):,} bytes -> {RAW}")


def assign(names):
    """ARS -> Landeskirche for every Gemeinde."""
    ars = pd.Series(names.index, index=names.index)
    lk = ars.str[:2].map(LAND)
    lk = ars.str[:5].map(KREIS).fillna(lk)
    krs = ars.str[:5]
    for k, prefixes, church in GEMEINDE:
        inside = names[krs == k]
        for p in prefixes:
            hit = inside[inside.str.startswith(p)]
            exact = hit[hit.str.split(",").str[0] == p]
            if len(hit) > 1 and len(exact) == 1:
                hit = exact
            if len(hit) != 1:
                raise SystemExit(f"de_ekd: {p!r} matches {len(hit)} Gemeinden in Kreis {k}: "
                                 f"{list(hit)[:4]}")
            lk[hit.index[0]] = church
    if lk.isna().any():
        raise SystemExit(f"de_ekd: {lk.isna().sum()} Gemeinden have no Landeskirche")
    return lk


def main():
    if "--fetch" in sys.argv:
        fetch()
    df = pd.read_csv(DE_CSV, dtype={"geo_id": str}, low_memory=False)
    g = df[df["geo_level"] == "gemeinde"]
    names = g.drop_duplicates("geo_id").set_index("geo_id")["geo_name"].astype(str)
    ev = g[g["source_category"] == EV].set_index("geo_id")["count"].astype(float)
    pop = g[g["source_category"] == "Einwohnerzahl"].set_index("geo_id")["count"].astype(float)
    lk = assign(names)

    # ---- the check
    t = pd.DataFrame({"ev": ev, "pop": pop, "lk": lk}).fillna({"ev": 0.0})
    s = t.groupby("lk")[["ev", "pop"]].sum()
    print(f"  Zensus 2022 Protestants by Landeskirche territory against EKD's Table 1 (31.12.2022)")
    print(f"  {'Landeskirche':<44}{'Zensus':>11}{'EKD':>11}{'ratio':>7}{'pop ratio':>11}")
    bad = []
    for church, (mem, epop) in EKD.items():
        if church == ERK:
            continue
        mem2 = mem + (EKD[ERK][0] if church == HAN else 0)
        z, zp = s.loc[church, "ev"], s.loc[church, "pop"]
        r = z / mem2
        name = church + (" + Reformierte Kirche" if church == HAN else "")
        print(f"  {name:<44}{z:>11,.0f}{mem2:>11,}{r:>7.3f}{zp / epop:>11.3f}")
        lo, hi = BAR_BIG if mem2 > 500000 else BAR_ALL
        if not lo <= r <= hi:
            bad.append(f"{church} {r:.3f}")
    print(f"  {'Germany':<44}{s.ev.sum():>11,.0f}{sum(v[0] for v in EKD.values()):>11,}")
    if bad:
        raise SystemExit(f"de_ekd: outside the bar: {bad}")

    # ---- rows: one per Gemeinde, named by its Landeskirche
    t = t[t["ev"] > 0].copy()
    t["label"] = t["lk"].map(LABEL)
    ekm = t["lk"] == EKM
    t.loc[ekm, "label"] = [EKM_T if a.startswith("16") else EKM_U for a in t.index[ekm]]
    t["note"] = "landeskirche=" + t["lk"] + "; parent_column=" + EV
    rows = [pd.DataFrame({"geo_id": t.index, "source_category": t["label"], "count": t["ev"],
                          "tier": "derived", "note": t["note"]})]

    # ---- the ERK, carved out of Hannover's count in its two home regions
    scale = EKD[ERK][0] / ERK_2010_TOTAL
    for region, (kreise, n2010) in ERK_REGIONS.items():
        m = t.index.str[:5].isin(kreise) & (t["lk"] == HAN)
        target = n2010 * scale
        share = t.loc[m, "ev"] / t.loc[m, "ev"].sum()
        take = share * target
        if (take > t.loc[m, "ev"]).any():
            raise SystemExit(f"de_ekd: the ERK takes more than a Gemeinde's Protestants in {region}")
        rows[0].loc[rows[0]["geo_id"].isin(t.index[m]), "count"] -= take.to_numpy()
        rows.append(pd.DataFrame({
            "geo_id": t.index[m], "source_category": LABEL[ERK], "count": take.to_numpy(),
            "tier": "derived",
            "note": f"landeskirche={ERK}; placement modelled: {region}'s members from the ERK's 2010 report "
                    f"scaled to 2022, spread by Protestants; parent_column={EV}"}))
        print(f"  ERK: {target:,.0f} of Hannover's {t.loc[m, 'ev'].sum():,.0f} Protestants in {region} "
              f"({100 * target / t.loc[m, 'ev'].sum():.1f}%)")
    out = pd.concat(rows, ignore_index=True)
    if abs(out["count"].sum() - ev.sum()) > 0.5:
        raise SystemExit("de_ekd: the rows do not sum to the Zensus column")
    by = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\n  rows written, by Landeskirche:")
    for k, v in by.items():
        print(f"    {v:>12,.0f}  {k}")
    tmp = OUT + ".tmp"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"\n  {len(out):,} rows, {out['count'].sum():,.0f} people -> {OUT}")


if __name__ == "__main__":
    main()
