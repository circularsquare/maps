"""Look-image of a city pack's water mask (T-030): the whole extent and close-ups of the places
that must come out right, with the city boundary.

    python pipeline/look_water.py nyc [out.png]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from read_pack import PACKS, R_EARTH, ROOT, load_pack, water_mask
from look_pack import boundary_xy

# (title, lon, lat, half-width km) of close-ups
VIEWS = {
    "nyc": [
        ("Manhattan, East River, Governors Island", -73.975, 40.745, 7.5),
        ("Harlem River, Randalls Island, Rikers", -73.915, 40.815, 4.5),
        ("Jamaica Bay and JFK", -73.83, 40.62, 9.0),
        ("Newark Bay, Kill van Kull, Arthur Kill", -74.14, 40.64, 7.5),
    ],
    "tor": [  # T-277
        ("Toronto Harbour and the Islands", -79.38, 43.63, 5.0),
        ("Humber River and Etobicoke shore", -79.48, 43.63, 4.5),
        ("Hamilton Harbour", -79.83, 43.29, 7.5),
        ("Lake Simcoe shore", -79.45, 44.25, 9.0),
    ],
    "lon": [  # T-041
        ("Central London, Westminster to Tower Bridge", -0.10, 51.505, 4.5),
        ("Isle of Dogs and the Royal Docks", 0.01, 51.505, 5.0),
        ("Staines reservoirs and Heathrow", -0.48, 51.45, 7.5),
        ("Lee Valley reservoirs", -0.03, 51.62, 5.0),
    ],
    "par": [  # T-289
        ("Central Paris, the Seine and its islands", 2.335, 48.858, 4.5),
        ("Seine loops west: Boulogne to Gennevilliers", 2.24, 48.88, 9.0),
        ("Marne and Seine meeting, Charenton", 2.43, 48.81, 6.0),
        ("Saint-Quentin-en-Yvelines", 2.03, 48.79, 5.0),
    ],
    "sha": [  # T-291
        ("Central Shanghai, the Huangpu and Suzhou Creek", 121.48, 31.235, 6.0),
        ("Yangtze mouth: Baoshan, Changxing, Hengsha", 121.70, 31.40, 30.0),
        ("Dianshan Lake, Qingpu", 120.98, 31.12, 12.0),
        ("Pudong's coast and Dishui Lake", 121.90, 30.92, 16.0),
    ],
    "mum": [  # T-292
        ("Mumbai Harbour and the island city", 72.88, 18.96, 16.0),
        ("Thane Creek, Vashi to Thane", 72.98, 19.10, 14.0),
        ("Vasai Creek and the Ulhas river to Kalyan", 73.00, 19.27, 24.0),
        ("Powai, Vihar and Tulsi lakes", 72.91, 19.16, 8.0),
    ],
    "tyo": [  # T-420
        ("Central Tokyo, the Sumida and the bay shore", 139.78, 35.68, 8.0),
        ("Arakawa, Edogawa and the Tokyo Bay landfill", 139.86, 35.66, 14.0),
        ("Tama River, Kawasaki and Haneda", 139.70, 35.56, 12.0),
        ("Lake Inba and Teganuma", 140.12, 35.80, 20.0),
        ("Kasumigaura, Kitaura and the lower Tone", 140.45, 35.98, 40.0),  # T-426
    ],
    "ber": [  # T-290
        ("Central Berlin, the Spree and the Landwehrkanal", 13.40, 52.51, 6.0),
        ("Havel lakes: Spandau, Wannsee, Tegeler See", 13.20, 52.48, 16.0),
        ("Mueggelsee and the Dahme", 13.62, 52.43, 10.0),
        ("Potsdam's lakes", 13.03, 52.38, 12.0),
    ],
    "osa": [  # T-431
        ("Central Osaka, the Yodo and the bay shore", 135.47, 34.69, 12.0),
        ("Kobe, the Muko and Amagasaki", 135.30, 34.72, 14.0),
        ("Kyoto, the Katsura, Uji and Kamo", 135.74, 34.95, 16.0),
        ("Lake Biwa's south basin", 135.95, 35.10, 30.0),
    ],
    "ngo": [  # T-431
        ("Central Nagoya, the Shonai and the port", 136.86, 35.12, 14.0),
        ("Kiso, Nagara and Ibi to Ise Bay", 136.72, 35.15, 30.0),
        ("Gifu and the Nagara", 136.77, 35.43, 12.0),
    ],
    "fuk": [  # T-432
        ("Hakata Bay, Tenjin and the Naka", 130.36, 33.62, 16.0),
        ("Kitakyushu: Dokai Bay and the Kanmon strait", 130.88, 33.92, 18.0),
        ("Chikugo river and the Ariake Sea", 130.45, 33.15, 30.0),
    ],
    "spk": [  # T-432
        ("Central Sapporo and the Toyohira", 141.36, 43.06, 10.0),
        ("Ishikari river to Ishikari Bay", 141.40, 43.20, 30.0),
        ("Otaru and its harbour", 141.00, 43.20, 10.0),
    ],
    "sdj": [  # T-432
        ("Central Sendai and the Hirose", 140.88, 38.26, 8.0),
        ("Matsushima Bay", 141.10, 38.35, 16.0),
        ("Ishinomaki and the Kitakami mouth", 141.30, 38.43, 16.0),
    ],
    "hij": [  # T-432
        ("The Ota delta, central Hiroshima", 132.45, 34.38, 8.0),
        ("Hiroshima Bay, Miyajima and Kure", 132.40, 34.25, 30.0),
        ("Onomichi and the Shimanami islands", 133.15, 34.33, 24.0),
    ],
}


def dense(m):
    """Decode row runs to a boolean raster (H, W), row 0 south."""
    off, xs = m["row"].astype(np.int64), m["x"].astype(np.int64)
    W, H = m["size"]
    d = np.zeros((H, W + 1), np.int8)
    r = np.repeat(np.arange(H), np.diff(off) // 2)
    np.add.at(d, (r, xs[0::2]), 1)
    np.add.at(d, (r, xs[1::2]), -1)
    return np.cumsum(d, axis=1, dtype=np.int8)[:, :W] > 0


def main(city, out, packs=PACKS):
    """`packs`: another folder of packs, such as T-041's water-only ones (data/work/osm_water)."""
    header, a = load_pack(city, packs)
    m = water_mask(header, a)
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    k = math.cos(math.radians(lat0))
    w = dense(m)
    x0, y0 = m["origin_m"]
    cell = m["cell_m"]
    W, H = m["size"]
    views = VIEWS.get(city, [])
    fig = plt.figure(figsize=(18, 12), dpi=130, facecolor="white")
    gs = fig.add_gridspec(2, 4)
    cmap = ListedColormap(["#f3efe6", "#7fa7d9"])
    ax = fig.add_subplot(gs[:, :2])
    f = 8  # overview at 8 x the pixel size
    small = w[: H // f * f, : W // f * f].reshape(H // f, f, W // f, f).mean(axis=(1, 3))
    ext = [x0 / 1e3, (x0 + W // f * f * cell) / 1e3, y0 / 1e3, (y0 + H // f * f * cell) / 1e3]
    ax.imshow(small, origin="lower", extent=ext, cmap="Blues", vmin=-0.3, vmax=1.2, interpolation="nearest")
    if os.path.exists(os.path.join(PACKS, f"{city}.boundary.geojson")):  # a water-only pack has none
        for bx, by in boundary_xy(city, lon0, lat0):
            ax.plot(bx, by, color="#c0392b", lw=0.5)
    ax.set_title(f"{city} water mask: {W} x {H} pixels of {cell:.0f} m, {m['runs']:,} runs, "
                 f"{m['bytes'] / 1e6:.2f} MB; water {m['water_share']:.1%} of the extent", fontsize=10)
    ax.set_xlabel("km east of the pack origin", fontsize=8)
    ax.tick_params(labelsize=7)
    for i, (title, lo, la, hw) in enumerate(views[:4]):
        axz = fig.add_subplot(gs[i // 2, 2 + i % 2])
        cx = R_EARTH * math.radians(lo - lon0) * k
        cy = R_EARTH * math.radians(la - lat0)
        c0 = max(0, int((cx - hw * 1e3 - x0) / cell)); c1 = min(W, int((cx + hw * 1e3 - x0) / cell))
        r0 = max(0, int((cy - hw * 1e3 - y0) / cell)); r1 = min(H, int((cy + hw * 1e3 - y0) / cell))
        axz.imshow(w[r0:r1, c0:c1], origin="lower", cmap=cmap, interpolation="nearest",
                   extent=[(x0 + c0 * cell) / 1e3, (x0 + c1 * cell) / 1e3, (y0 + r0 * cell) / 1e3, (y0 + r1 * cell) / 1e3])
        axz.set_title(title, fontsize=9)
        axz.tick_params(labelsize=6)
        rect = plt.Rectangle(((x0 + c0 * cell) / 1e3, (y0 + r0 * cell) / 1e3), (c1 - c0) * cell / 1e3,
                             (r1 - r0) * cell / 1e3, fill=False, color="#333333", lw=0.6)
        ax.add_patch(rect)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc"
    main(city, sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "..", "artifacts", "trainworld", "T-030.png"))
