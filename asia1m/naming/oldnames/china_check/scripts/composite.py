import os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
S = os.path.dirname(os.path.abspath(__file__))
base = None
for l in ["coastglobal", "main", "insets", "textlocal"]:
    im = Image.open(os.path.join(S, l + ".png")).convert("RGBA")
    a = np.array(im)
    print(l, im.size, "alpha values:", np.unique(a[:, :, 3])[:10])
    if l == "textlocal":
        a[:, :, 3] = np.where(a[:, :, 3] > 0, 255, 0)
        im = Image.fromarray(a)
    if base is None:
        base = Image.new("RGBA", im.size, (0, 0, 0, 0))
    base.alpha_composite(im)
# background: dark? use white under transparent
bg = Image.new("RGBA", base.size, (255, 255, 255, 255))
bg.alpha_composite(base)
bg.convert("RGB").save(os.path.join(S, "composite.png"))
print("done")
