"""Greenland, population register 1 January 2026 (sources/gl_census.py, sources/gl.md). No
language question: the Greenland-born on Greenlandic by district (Tunumiisut in the east, Inuktun
around Qaanaaq, Kalaallisut elsewhere); the born-outside by citizenship through origin_mix, whose
rows arrive as "born outside: <node>" already resolved."""
NAMES = {
    "Kalaallisut": "eskimoaleut.greenlandic",
    "Tunumiisut": "eskimoaleut.tunumiisut",     # Glottolog tunu1234, a language of its own
    "Inuktun": "eskimoaleut.inuktun",           # Glottolog's Polar Eskimo (pola1254)
}
PREFIX = "born outside: "


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    if label.startswith(PREFIX):
        return label[len(PREFIX):]
    raise KeyError(f"gl2026: unmapped label {label!r}")
