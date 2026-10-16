"""T-458: the OD 2017 survey's home-based trips by purpose (lengths, centre share, rates), for the
comparison with the game's everyday rule in notes/T-458.md. Read only.

    python notes/t458_survey.py
"""
import glob
import json
import os
import unicodedata

import numpy as np
import pyogrio

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sao", "od", "OD-2017")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "trainworld", "T-458-survey.json")
# Centre: the Se and Pinheiros subprefectures (the old centre, Paulista, Faria Lima)
CENTRE = {"se", "republica", "bela vista", "consolacao", "liberdade", "bom retiro", "santa cecilia", "cambuci",
          "pinheiros", "alto de pinheiros", "itaim bibi", "jardim paulista"}
PURPOSES = {"errands": [5, 10], "social": [7, 11], "medical": [6], "school": [4], "work": [1, 2, 3]}
BANDS = [0, 2, 5, 10, 15, 20, 30, 1e9]


def plain(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2)])


def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
    import brazil
    brazil._survey_dir("sao")  # extracts the microdata from the kept zip when it was deleted
    df = pyogrio.read_dataframe(glob.glob(os.path.join(BASE, "Banco*", "OD_2017_v1.dbf"))[0], read_geometry=False,
                                columns=["F_PESS", "FE_PESS", "FE_VIA", "MUNI_DOM", "ZONA_O", "ZONA_D", "MOTIVO_O", "MOTIVO_D",
                                         "MODOPRIN", "DISTANCIA", "TIPVG"])
    zones = pyogrio.read_dataframe(glob.glob(os.path.join(BASE, "Mapas*", "Shape*", "Zonas_2017_region.shp"))[0],
                                   read_geometry=False, encoding="cp1252")
    centre_zones = {int(z) for z, d in zip(zones.NumeroZona, zones.NomeDistri) if plain(d) in CENTRE}
    people = float(df.FE_PESS[df.F_PESS == 1].sum())
    sp_people = float(df.FE_PESS[(df.F_PESS == 1) & (df.MUNI_DOM == 36)].sum())
    t = df[df.FE_VIA.notna() & df.MOTIVO_D.notna()]
    hb = t[t.MOTIVO_O == 8]
    res = {"people": people, "centre_zones": len(centre_zones), "purposes": {}}
    print(f"people {people:,.0f} (Sao Paulo municipality {sp_people:,.0f}); trips {t.FE_VIA.sum():,.0f}, "
          f"home-based {hb.FE_VIA.sum():,.0f}; centre zones {len(centre_zones)}")
    for name, codes in PURPOSES.items():
        s = hb[hb.MOTIVO_D.isin(codes)]
        d, w = s.DISTANCIA.to_numpy() / 1000.0, s.FE_VIA.to_numpy()
        band = np.bincount(np.searchsorted(BANDS, d, side="right") - 1, weights=w, minlength=len(BANDS) - 1) / w.sum()
        cen = s.ZONA_D.isin(centre_zones).to_numpy()
        home_c = s.ZONA_O.isin(centre_zones).to_numpy()
        sp = (s.MUNI_DOM == 36).to_numpy() & ~home_c
        oth = (s.MUNI_DOM != 36).to_numpy()
        walk = float(w[(s.TIPVG == 3).to_numpy()].sum() / w.sum())
        row = {"trips": float(w.sum()), "a_resident": float(w.sum() / people), "mean_km": float(np.average(d, weights=w)),
               "median_km": wmedian(d, w), "bands": band.round(3).tolist(), "centre_share": float(w[cen].sum() / w.sum()),
               "centre_stay": float(w[cen & home_c].sum() / w[home_c].sum()),
               "centre_share_from_rest_of_sp": float(w[cen & sp].sum() / w[sp].sum()),
               "centre_share_from_others": float(w[cen & oth].sum() / w[oth].sum()), "walk_share": walk}
        res["purposes"][name] = row
        print(f"{name}: {row['trips']:,.0f} home-based trips, {row['a_resident']:.3f} a resident, mean {row['mean_km']:.1f} km, "
              f"median {row['median_km']:.1f} km, bands {BANDS[:-1]} {row['bands']}, to centre {row['centre_share']:.1%} "
              f"(centre stays {row['centre_stay']:.1%}, from the rest of SP {row['centre_share_from_rest_of_sp']:.1%}, "
              f"other municipalities {row['centre_share_from_others']:.1%}), walk {walk:.1%}")
    modes = t.groupby("TIPVG").FE_VIA.sum() / t.FE_VIA.sum()
    res["trip_type_shares"] = {int(k): round(float(v), 3) for k, v in modes.items()}
    print("all trips by type (1 collective, 2 individual, 3 walk, 4 bike):", res["trip_type_shares"])
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
