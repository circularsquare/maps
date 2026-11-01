"""Export asia1m.ase as separate web layers for the website's layered viewer (layout: layeredmap).

    python asia1m/weblayers.py                 # writes ../website/assets/maps/asia1m/
    python asia1m/weblayers.py --out DIR --preview

Copies asia1m.ase (the original is never opened; Aseprite's GUI may stay open), exports the layers
below at full strength, and writes:

    base.png          coastglobal + main + insets, no text (what you see with everything off)
    plain.png         the same with region fills white (the viewer's colors-off map; its own image
                      rather than a whitening mask, which tints borders once mipmaps mix them)
    adm1.png          'adm' lines (grey mask), cut where insets cover them
    adm2.png          'adm2' lines; red = coverage, green = coverage of the darker Sri Lanka green
    insetadm.png      'inset adm', cropped (RGBA, drawn above the insets with the adm2 toggle)
    labels_eng.png    'text', cropped
    labels_local.png  'textlocal', cropped
    flat_eng.png      base + english labels in one image, and flat_local.png with local labels:
                      the pages' fallback where WebGL2 can't draw the layers, and their preview image
    layers.json       sizes, crop offsets, colours and opacities (read from the .ase)

Stacking matches the .ase: coastglobal, main, adm2, adm, insets, inset adm, text. A few minutes.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
ASE = os.path.join(HERE, "asia1m.ase")
ASEPRITE = os.environ.get("ASEPRITE", r"C:/Program Files/Aseprite/Aseprite.exe")
DEFAULT_OUT = os.path.join(HERE, "..", "..", "website", "assets", "maps", "asia1m")

LAYERS = {"coastglobal": "coastglobal", "main": "main", "adm": "adm", "adm2": "adm2", "insets": "insets",
          "inset adm": "inset_adm", "text": "text", "textlocal": "textlocal"}

# Writes each layer's cel onto a full-canvas PNG, raw pixels, and the layer opacities to opacity.txt.
LUA = """
local spr = app.activeSprite
local outdir = app.params["outdir"]
local want = { %s }
local f = io.open(outdir .. "/opacity.txt", "w")
local function walk(layers)
  for _, l in ipairs(layers) do
    if l.isGroup then walk(l.layers)
    elseif want[l.name] then
      local img = Image(spr.width, spr.height, spr.colorMode)
      local cel = l:cel(1)
      if cel then img:drawImage(cel.image, cel.position) end
      img:saveAs(outdir .. "/" .. want[l.name] .. ".png")
      f:write(want[l.name] .. " " .. l.opacity .. "\\n")
    end
  end
end
walk(spr.layers)
f:close()
""" % ", ".join(f'["{k}"]="{v}"' for k, v in LAYERS.items())

ADM1 = (0xf1, 0x56, 0x56)
ADM2 = (0xaf, 0xe3, 0x64)
ADM2_DARK = (0x4a, 0x8c, 0x3f)  # Sri Lanka's divisions, on undrawn white land


def export(tmp):
    copy = os.path.join(tmp, "copy.ase")
    shutil.copyfile(ASE, copy)
    lua = os.path.join(tmp, "export.lua")
    with open(lua, "w") as f:
        f.write(LUA)
    print("exporting layers from a copy of asia1m.ase ...", flush=True)
    subprocess.run([ASEPRITE, "-b", copy, "--script-param", f"outdir={tmp}", "--script", lua], check=True)
    missing = [v for v in LAYERS.values() if not os.path.exists(os.path.join(tmp, v + ".png"))]
    if missing:
        sys.exit(f"Aseprite did not export {missing}; was a layer renamed?")
    with open(os.path.join(tmp, "opacity.txt")) as f:
        return {k: int(v) for k, v in (line.split() for line in f if line.strip())}


def load(tmp, name):
    return np.asarray(Image.open(os.path.join(tmp, name + ".png")).convert("RGBA"))


def is_colour(a, rgb):
    return np.all(a[..., :3] == np.array(rgb, np.uint8), axis=-1) & (a[..., 3] > 0)


def keep_mask(a):
    """Pixels of main/insets that stay in the no-colours view: lines, coast, water, frames."""
    r, g, b = (a[..., i].astype(np.int16) for i in range(3))
    border = (r == 0x53) & (b == 0x8d) & (g >= 0x58) & (g <= 0x60)
    keep = border
    for c in [(0x45, 0x4c, 0x56), (0x64, 0xa0, 0xd2), (0xd9, 0xf6, 0xff), (0xa0, 0x99, 0x97)]:
        keep |= is_colour(a, c)
    return keep


def over(dst, src):
    """Straight-alpha 'over' of uint8 RGBA src onto an opaque float RGB dst (in place, banded)."""
    for y0 in range(0, dst.shape[0], 1000):
        s = src[y0:y0 + 1000].astype(np.float32) / 255
        a = s[..., 3:4]
        dst[y0:y0 + 1000] = s[..., :3] * a + dst[y0:y0 + 1000] * (1 - a)


def crop_box(alpha, pad=2):
    ys, xs = np.nonzero(alpha)
    h, w = alpha.shape
    return max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + 1 + pad), min(h, ys.max() + 1 + pad)


def save(img, path):
    """Write beside the temp exports, then move into place in one step: a watching `jekyll serve`
    otherwise copies a half-written PNG into _site and the viewer gets bad image data."""
    part = os.path.join(save.tmp, os.path.basename(path))
    img.save(part, optimize=True)
    os.replace(part, path)
    print(f"  {os.path.basename(path)}: {os.path.getsize(path) / 1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--preview", action="store_true", help="also write preview_*.png composites to --out")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    tmp = save.tmp = tempfile.mkdtemp(prefix="asia1m_web_")
    try:
        opacity = export(tmp)
        coast, mainl, insets = load(tmp, "coastglobal"), load(tmp, "main"), load(tmp, "insets")
        H, W = mainl.shape[:2]

        print("compositing base and plain ...", flush=True)
        base = np.ones((H, W, 3), np.float32)
        over(base, coast)
        plain = base.copy()
        over(base, mainl)
        mp = mainl.copy()
        mp[~keep_mask(mainl) & (mainl[..., 3] > 0), :3] = 255
        over(plain, mp)
        del mp
        over(base, insets)
        ip = insets.copy()
        ip[~keep_mask(insets) & (insets[..., 3] > 0), :3] = 255
        over(plain, ip)
        del ip
        base8 = np.clip(np.rint(base * 255), 0, 255).astype(np.uint8)
        del base
        save(Image.fromarray(base8), os.path.join(out, "base.png"))
        save(Image.fromarray(np.clip(np.rint(plain * 255), 0, 255).astype(np.uint8)), os.path.join(out, "plain.png"))
        del plain

        print("admin lines ...", flush=True)
        uncovered = 255 - insets[..., 3].astype(np.uint16)  # insets sit above adm and adm2
        adm = load(tmp, "adm")
        m1 = (is_colour(adm, ADM1) * uncovered).astype(np.uint8)
        del adm
        save(Image.fromarray(m1), os.path.join(out, "adm1.png"))
        adm2 = load(tmp, "adm2")
        light, dark = is_colour(adm2, ADM2), is_colour(adm2, ADM2_DARK)
        del adm2
        m2 = np.zeros((H, W, 3), np.uint8)
        m2[..., 0] = ((light | dark) * uncovered).astype(np.uint8)
        m2[..., 1] = (dark * uncovered).astype(np.uint8)
        save(Image.fromarray(m2), os.path.join(out, "adm2.png"))
        del uncovered

        crops = {}
        for name, layer in [("insetadm", "inset_adm"), ("labels_eng", "text"), ("labels_local", "textlocal")]:
            a = load(tmp, layer)
            x0, y0, x1, y1 = crop_box(a[..., 3] > 0)
            c = a[y0:y1, x0:x1].copy()
            c[c[..., 3] == 0] = 0
            save(Image.fromarray(c), os.path.join(out, name + ".png"))
            crops[name] = {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)}
            if name.startswith("labels_"):
                flat = base8.copy()
                region = flat[y0:y1, x0:x1].astype(np.float32)
                a = c[..., 3:4].astype(np.float32) / 255
                flat[y0:y1, x0:x1] = np.rint(c[..., :3] * a + region * (1 - a)).astype(np.uint8)
                save(Image.fromarray(flat), os.path.join(out, name.replace("labels_", "flat_") + ".png"))
                del flat, region

        hexc = lambda c: "#%02x%02x%02x" % c
        meta = {
            "width": W, "height": H, "version": time.strftime("%Y%m%d%H%M%S"),
            "base": {"src": "base.png"},
            "plain": {"src": "plain.png"},
            "lines": [  # bottom to top; one checkbox per id
                {"id": "adm2", "src": "adm2.png", "kind": "mask", "color": hexc(ADM2), "color2": hexc(ADM2_DARK),
                 "opacity": opacity["adm2"] / 255},
                {"id": "adm1", "src": "adm1.png", "kind": "mask", "color": hexc(ADM1), "opacity": opacity["adm"] / 255},
                {"id": "adm2", "src": "insetadm.png", "kind": "rgba", **crops["insetadm"],
                 "opacity": opacity["inset_adm"] / 255},
            ],
            "labels": [
                {"id": "eng", "name": "english", "src": "labels_eng.png", **crops["labels_eng"]},
                {"id": "local", "name": "local", "src": "labels_local.png", **crops["labels_local"]},
            ],
        }
        with open(os.path.join(tmp, "layers.json"), "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(os.path.join(tmp, "layers.json"), os.path.join(out, "layers.json"))
        print(f"  layers.json (version {meta['version']})")

        if args.preview:
            print("previews ...", flush=True)
            preview(out, meta)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("done:", out)


def preview(out, meta):
    """Composite a few toggle states from the written files, the way the viewer's shader does."""
    rd = lambda n: np.asarray(Image.open(os.path.join(out, n))).astype(np.float32) / 255
    col = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255

    def draw(colours, lines, labels):
        img = rd("base.png" if colours else "plain.png")
        for L in meta["lines"]:
            if L["id"] not in lines:
                continue
            m = rd(L["src"])
            if L["kind"] == "mask":
                cov = m[..., 0] if m.ndim == 3 else m
                c = col(L["color"])
                if m.ndim == 3:
                    t = np.where(cov > 0, m[..., 1] / np.maximum(cov, 1e-6), 0)[..., None]
                    c = c * (1 - t) + col(L["color2"]) * t
                a = (cov * L["opacity"])[..., None]
                img = img * (1 - a) + c * a
            else:
                s = img[L["y"]:L["y"] + L["h"], L["x"]:L["x"] + L["w"]]
                a = m[..., 3:4] * L["opacity"]
                img[L["y"]:L["y"] + L["h"], L["x"]:L["x"] + L["w"]] = s * (1 - a) + m[..., :3] * a
        if labels:
            L = next(l for l in meta["labels"] if l["id"] == labels)
            m = rd(L["src"])
            s = img[L["y"]:L["y"] + L["h"], L["x"]:L["x"] + L["w"]]
            img[L["y"]:L["y"] + L["h"], L["x"]:L["x"] + L["w"]] = s * (1 - m[..., 3:4]) + m[..., :3] * m[..., 3:4]
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    views = {"colours_eng": (True, set(), "eng"), "plain_adm12_local": (False, {"adm1", "adm2"}, "local"),
             "colours_adm1_none": (True, {"adm1"}, None)}
    for name, v in views.items():
        Image.fromarray(draw(*v)).save(os.path.join(out, f"preview_{name}.png"))
        print("  preview_" + name + ".png")


if __name__ == "__main__":
    main()
