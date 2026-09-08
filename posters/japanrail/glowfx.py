"""
Glow behind the rail lines.

The interactive map had this and lost it twice over: it cost too much per
frame, and drawing a halo *per segment* left a bright notch at every joint,
worst on curves, because each segment's halo ended at its own cap instead of
continuing round the corner.

Neither problem exists here. The whole network is already one raster by the
time this runs, so blurring it blurs a continuous shape — corners included —
and it happens once per render rather than once per frame.

Two things matter for it not to look muddy:

- **Premultiplied alpha.** Blurring straight RGBA drags the colour of fully
  transparent pixels (which is black, or whatever matplotlib left there) into
  the halo, so every glow greys off toward the background. Multiplying by alpha
  first, blurring, then dividing back out keeps each halo the colour of the
  line that cast it.
- **Two radii.** One blur gives either a tight rim or a broad haze, not both.
  A tight pass at roughly twice the line width reads as the line glowing; a
  wide pass at several times that reads as light in the air around it, and in
  Tokyo the two stack into a hot core because the alpha of forty lines adds up.

The glow is computed at reduced resolution and scaled back. A Gaussian this
wide is smooth by construction, so a quarter-scale pass is visually identical
to a full-scale one and sixteen times cheaper — which matters at 68 megapixels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageFilter

Image.MAX_IMAGE_PIXELS = None


def _premultiplied_small(im, sw, sh):
    """(rgb*a, a) at (sw, sh), as float arrays in 0..1."""
    r, g, b, a = im.split()
    parts = [ImageChops.multiply(ch, a) for ch in (r, g, b)]
    # BOX is a plain area average, which is what premultiplied compositing
    # wants; LANCZOS would ring and put negative alpha round every line.
    small = [ch.resize((sw, sh), Image.BOX) for ch in parts + [a]]
    arr = [np.asarray(ch, np.float32) / 255.0 for ch in small]
    return np.dstack(arr[:3]), arr[3]


def glow(lines_png, out_png, radius_in, dpi, scale=1.0, strength=0.85,
         wide_mult=3.5, wide_strength=0.45, downscale=4, cap=1.0):
    """Write the glow layer for a rendered lines layer.

    radius_in is the tight pass, in inches of the printed sheet, so it means
    the same thing at any dpi or preview scale.
    """
    im = Image.open(lines_png).convert("RGBA")
    W, H = im.size
    ds = max(1, int(downscale))
    sw, sh = max(1, W // ds), max(1, H // ds)

    pm, a = _premultiplied_small(im, sw, sh)

    r_px = radius_in * dpi * scale / ds
    out_pm = np.zeros_like(pm)
    out_a = np.zeros_like(a)
    for rad, amt in ((r_px, strength), (r_px * wide_mult, wide_strength)):
        if rad < 0.3 or amt <= 0:
            continue
        f = ImageFilter.GaussianBlur(radius=float(rad))
        bpm = np.dstack([
            np.asarray(Image.fromarray((pm[:, :, i] * 255).astype(np.uint8))
                       .filter(f), np.float32) / 255.0 for i in range(3)])
        ba = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(f),
                        np.float32) / 255.0
        out_pm += bpm * amt
        out_a += ba * amt

    # Un-premultiply against the accumulated alpha, BEFORE it is clipped —
    # clipping first would darken the colour of exactly the dense areas the
    # cap exists to tame. Where almost nothing blurred in, the ratio is noise,
    # but it is multiplied by a near-zero alpha, so it never shows.
    rgb = np.clip(out_pm / np.maximum(out_a, 1e-4)[:, :, None], 0.0, 1.0)
    out_a = np.clip(out_a, 0.0, cap)

    small = Image.fromarray(np.concatenate([
        (rgb * 255).astype(np.uint8),
        (out_a * 255).astype(np.uint8)[:, :, None]], axis=2), "RGBA")
    small.resize((W, H), Image.BICUBIC).save(out_png)
    return out_png
