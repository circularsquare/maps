"""T-1150: Shanghai's homes and jobs as the game has them, by district and densest cells."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
from read_pack import load_pack
import china

sig = load_sig("sha")
pop, jobs, idx, ds = sha_rebuild(sig)
h, a = load_pack("sha")
print(f"rebuilt res-9: pop {pop.sum():,.0f} jobs {jobs.sum():,.0f}; pack pop {a['pop'].sum():,.0f} jobs {a['jobs'].sum():,.0f}")
area = cell_area_km2(sig["h3"])
np.save(os.path.join(os.path.dirname(__file__), "sha_area.npy"), area)
np.save(os.path.join(os.path.dirname(__file__), "sha_pop.npy"), pop)
np.save(os.path.join(os.path.dirname(__file__), "sha_jobs.npy"), jobs)
np.save(os.path.join(os.path.dirname(__file__), "sha_idx.npy"), idx)

# pack check by district (pack cells' centres)
import england
plat, plon = [], []
for c in a["h3"]:
    la, lo = h3.cell_to_latlng(h3.int_to_str(int(c)))
    plat.append(la); plon.append(lo)
pidx, _ = england.cell_units(np.array(plat), np.array(plon), [(d[0], d[1]) for d in ds])
print("\nDistrict | census pop | pack pop | GHSL raw | GHSL/census | census jobs | pack jobs | land km2 | pop/km2 | jobs/km2")
for i, (name, g, cp) in enumerate(ds):
    py = china.SHANGHAI[name][0]
    sel = idx == i
    km2 = area[sel & ((sig["pop_ghsl"] > 0) | (sig["nres_v"] > 0))].sum()
    gh = sig["pop_ghsl"][sel].sum()
    print(f"{py} | {cp:,} | {a['pop'][pidx == i].sum():,.0f} | {gh:,.0f} | {gh / cp - 1:+.0%} | {china.JOBS['sha'][name]:,.0f} | "
          f"{a['jobs'][pidx == i].sum():,.0f} | {km2:,.0f} | {cp / km2:,.0f} | {china.JOBS['sha'][name] / km2:,.0f}")

dens = pop / area
o = np.argsort(-pop)
print("\nDensest res-9 cells by people (pop, per km2, GHSL raw, district, lat lon, nres_v, osm):")
for i in o[:25]:
    print(f"  {pop[i]:8,.0f} {dens[i]:9,.0f}/km2 ghsl {sig['pop_ghsl'][i]:8,.0f} {china.SHANGHAI[ds[idx[i]][0]][0]:10s} "
          f"{sig['lat'][i]:.4f} {sig['lon'][i]:.4f} nres_v {sig['nres_v'][i]:,.0f}")
print("\npop per res-9 cell quantiles (cells with pop>0):")
p = pop[pop > 0]
for q in (0.5, 0.9, 0.99, 0.999, 1.0):
    print(f"  q{q}: {np.quantile(p, q):,.0f} ({np.quantile(p, q) / 0.105:,.0f}/km2)")
o = np.argsort(-jobs)
print("\nDensest res-9 cells by jobs:")
for i in o[:20]:
    print(f"  {jobs[i]:8,.0f} pop {pop[i]:7,.0f} {china.SHANGHAI[ds[idx[i]][0]][0]:10s} {sig['lat'][i]:.4f} {sig['lon'][i]:.4f} "
          f"nres_v {sig['nres_v'][i]:,.0f} h {sig['nres_v'][i] / max(sig['nres_s'][i], 1):.0f} osm_land {sig['osm_land'][i]:.1f} osm_other {sig['osm_job'][i] + sig['osm_errand'][i] + sig['osm_leisure'][i]:.0f}")
# pack merged cells densest
pa = cell_area_km2(a["h3"])
d = a["pop"] / pa
o = np.argsort(-d)
print("\nPack cells by density (merged cells included):")
for i in o[:10]:
    print(f"  {a['pop'][i]:8,.0f} {d[i]:9,.0f}/km2 res {h3.get_resolution(h3.int_to_str(int(a['h3'][i])))} {plat[i]:.4f} {plon[i]:.4f}")
