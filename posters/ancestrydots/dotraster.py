"""
Shared dot rasteriser for the ancestrydots poster.

Dots are splatted last-write-wins in shuffled order, which is unbiased: in a
pile-up, whichever dot happens to be on top is random rather than being decided
by input order (which is state-by-state and would bias the result).

Antialiasing is done by supersampling rather than by alpha-blending each dot.
Alpha-blending would have to composite dots sequentially to get occlusion right,
which is not vectorisable; and averaging overlapping dots instead of occluding
them desaturates dense areas toward a muddy mean. Rendering hard-edged at ss x
and box-downsampling keeps correct occlusion AND gives soft round dots, at the
cost of ss^2 memory — so it runs in horizontal strips to stay bounded.

Note `disc()` quantises hard at ss=1: radius 0.9 -> 1 px, 1.0 -> 5 px (a plus),
1.5 -> 9 px. Fractional radii only become meaningful once ss > 1.
"""

from __future__ import annotations

import numpy as np

EMPTY = 0xFFFF


def disc(r: float):
    """Integer pixel offsets within radius r of the origin."""
    n = int(np.ceil(r))
    dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
    m = (dx * dx + dy * dy) <= r * r
    return np.stack([dy[m].ravel(), dx[m].ravel()], axis=1)


def splat(px, py, idx, W, H, radius, lut, ss=1, rng=None, max_sup_px=20_000_000):
    """Rasterise dots to (rgb uint8 HxWx3, alpha uint8 HxW).

    px, py  float pixel coordinates in the *output* raster
    radius  dot radius in output pixels (can be fractional when ss > 1)
    ss      supersample factor; 1 reproduces the old hard-edged look
    """
    rng = rng or np.random.default_rng(0)
    order = rng.permutation(len(px))
    px, py, idx = px[order], py[order], idx[order]

    offs = disc(radius * ss)
    r_sup = int(np.ceil(radius * ss))
    Ws = W * ss
    pxs = (px * ss).astype(np.int32)
    pys = (py * ss).astype(np.int32)

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    alpha = np.zeros((H, W), dtype=np.uint8)

    rows = max(1, min(H, int(max_sup_px / max(Ws, 1) / max(ss, 1))))
    for y0 in range(0, H, rows):
        y1 = min(y0 + rows, H)
        sy0, sy1 = y0 * ss, y1 * ss

        sel = (pys >= sy0 - r_sup) & (pys < sy1 + r_sup)
        buf = np.full((sy1 - sy0, Ws), EMPTY, dtype=np.uint16)
        if sel.any():
            bx, by, bi = pxs[sel], pys[sel] - sy0, idx[sel]
            for dy, dx in offs:
                yy, xx = by + dy, bx + dx
                ok = (yy >= 0) & (yy < buf.shape[0]) & (xx >= 0) & (xx < Ws)
                buf[yy[ok], xx[ok]] = bi[ok]

        if ss == 1:
            solid = buf != EMPTY
            rgb[y0:y1] = lut[buf]
            alpha[y0:y1] = np.where(solid, 255, 0)
            continue

        solid = buf != EMPTY
        # premultiply so partly-covered output pixels average only the covered
        # subpixels — otherwise empty subpixels would drag the colour to black
        sub = lut[buf].astype(np.uint16) * solid[..., None]
        h_out = y1 - y0
        csum = sub.reshape(h_out, ss, W, ss, 3).sum(axis=(1, 3))
        acnt = solid.reshape(h_out, ss, W, ss).sum(axis=(1, 3))

        nz = acnt > 0
        out = np.zeros((h_out, W, 3), dtype=np.uint8)
        out[nz] = (csum[nz] // acnt[nz][:, None]).astype(np.uint8)
        rgb[y0:y1] = out
        alpha[y0:y1] = (acnt * 255 // (ss * ss)).astype(np.uint8)

    return rgb, alpha


def over(rgb, alpha, base):
    """Composite premultiplied-ish dots over an RGB base."""
    a = alpha[..., None].astype(np.uint16)
    return ((rgb.astype(np.uint16) * a +
             base.astype(np.uint16) * (255 - a)) // 255).astype(np.uint8)
