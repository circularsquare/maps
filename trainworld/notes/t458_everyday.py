"""T-458: the game's everyday outings in the `sao` pack (the sim's rule redone in numpy, as T-451's
region_od.py), against the OD 2017 survey's home-based trips (t458_survey.py): lengths by band and
the centre's share from three origin areas. Read only.

    python notes/t458_everyday.py     (needs the res-9 or merged pack; reads ../artifacts/trainworld/T-458-survey.json)
"""
import glob
import json
import os
import sys
import unicodedata

os.environ["OMP_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
import h3
import numpy as np
import pyogrio
import shapely
from shapely.geometry import shape
from read_pack import load_pack

CENTRE = {"se", "republica", "bela vista", "consolacao", "liberdade", "bom retiro", "santa cecilia", "cambuci",
          "pinheiros", "alto de pinheiros", "itaim bibi", "jardim paulista"}
BANDS = [0, 2, 5, 10, 15, 20, 30, 1e9]
AREAS = ["centre", "rest of Sao Paulo", "other municipalities"]


def plain(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    h, a = load_pack("sao")
    n = h["cells"]
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]])
    pts = shapely.points(ll[:, 1], ll[:, 0])
    base = os.path.join(HERE, "..", "data", "raw", "sao", "od", "OD-2017")
    dist = pyogrio.read_dataframe(glob.glob(os.path.join(base, "Mapas*", "Shape*", "Distritos_2017_region.shp"))[0],
                                  encoding="cp1252").to_crs("EPSG:4326")
    centre = shapely.union_all([g for nm, g in zip(dist.NomeDistri, dist.geometry) if plain(nm) in CENTRE])
    with open(os.path.join(HERE, "..", "data", "raw", "sao", "sao_units.geojson"), encoding="utf-8") as f:
        sp = next(shape(ft["geometry"]) for ft in json.load(f)["features"] if ft["properties"]["code"] == "3550308")
    shapely.prepare(centre)
    shapely.prepare(sp)
    region = np.full(n, 2)
    region[shapely.contains(sp, pts)] = 1
    region[shapely.contains(centre, pts)] = 0
    zone = a["zone"].astype(np.int64)
    gx, gy = a["zone_gx"].astype(np.float64), a["zone_gy"].astype(np.float64)
    zm, nz = h["gravity"]["zone_m"], len(a["zone_gx"])
    pop = a["pop"].astype(np.float64)
    leis, park, errand = (a[k].astype(np.float64) for k in ("leisure_jobs", "park_draw", "errand_jobs"))
    med = a["medical_visits"].astype(np.float64)
    social = 0.46 * pop / pop.sum() + 0.46 * leis / leis.sum() + 0.08 * park / park.sum()
    print("residents by area (M):", [round(float(pop[region == r].sum()) / 1e6, 2) for r in range(3)])
    with open(os.path.join(HERE, "..", "..", "artifacts", "trainworld", "T-458-survey.json")) as f:
        sv = json.load(f)["purposes"]
    zpop = np.stack([np.bincount(zone, weights=pop * (region == r), minlength=nz) for r in range(3)], 1)
    out = {}
    for name, w, scale in (("errands", errand, 3000.0), ("social", social, 6000.0), ("medical", med, 5000.0)):
        mass = np.bincount(zone, weights=w, minlength=nz)
        R = np.stack([np.bincount(zone, weights=w * (region == r), minlength=nz) for r in range(3)], 1)
        R = np.divide(R, mass[:, None], out=np.zeros_like(R), where=mass[:, None] > 0)
        od = np.zeros((3, 3))
        bands = np.zeros(len(BANDS) - 1)
        dsum = 0.0
        for oz in np.nonzero(zpop.sum(1) > 0)[0]:
            d = np.hypot(gx - gx[oz], gy - gy[oz]) * zm
            d[oz] = 1040.0
            pz = mass * np.exp(-d / scale)
            pz /= pz.sum()
            od += np.outer(zpop[oz], pz @ R)
            bands += zpop[oz].sum() * np.bincount(np.searchsorted(BANDS, d / 1000, side="right") - 1, weights=pz,
                                                  minlength=len(BANDS) - 1)
            dsum += zpop[oz].sum() * float((pz * d).sum()) / 1000
        tot = zpop.sum()
        s = sv[name]
        out[name] = {"mean_km": dsum / tot, "bands": (bands / tot).round(3).tolist(),
                     "centre_stay": od[0, 0] / od[0].sum(), "centre_from_rest_of_sp": od[1, 0] / od[1].sum(),
                     "centre_from_others": od[2, 0] / od[2].sum()}
        g = out[name]
        print(f"{name}: game mean {g['mean_km']:.1f} km (survey {s['mean_km']:.1f}); bands game {g['bands']} survey {s['bands']}; "
              f"centre stays game {g['centre_stay']:.1%} survey {s['centre_stay']:.1%}; to centre from the rest of SP game "
              f"{g['centre_from_rest_of_sp']:.1%} survey {s['centre_share_from_rest_of_sp']:.1%}; from other municipalities "
              f"game {g['centre_from_others']:.1%} survey {s['centre_share_from_others']:.1%}")
    with open(os.path.join(HERE, "..", "..", "artifacts", "trainworld", "T-458-everyday.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
