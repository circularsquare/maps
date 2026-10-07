"""A points-only Mapbox Vector Tile encoder, because the general one is 26x slower.

WHY THIS EXISTS.  `mapbox_vector_tile.encode` is written for arbitrary geometry: for every
feature it builds a shapely object, checks its validity, enforces a ring winding order and
runs it through `shapely.ops.transform`.  Every feature in this archive is a Point, for
which all of that is dead weight — a point's geometry is three integers.  Measured on a
200,000-feature tile of the real shape (a node string, a country string, a count):

    mapbox_vector_tile      11,400 features/s
    this module            294,000 features/s

A full --coarse build encodes roughly 28 million features, because the `atomic` layer is
every dot at every zoom, so that is the difference between about 40 minutes and about 90
seconds.  The output decodes to an identical feature set, and is fractionally smaller,
since this writes no feature id.

THIS IS THE ONLY HAND-ROLLED WIRE FORMAT IN THE PROJECT, so it has a check that shares no
code with it: `python tools/check_tiles.py <countries>` rebuilds every tile from the
geojson with the old pandas-and-mapbox_vector_tile path and compares the archive against
it feature by feature.  Run it after touching anything below.

THE VALUE TABLE IS PER TILE AND HAS TO BE.  An MVT layer carries its own key and value
tables and every feature refers into them by index.  Interning the project's ~150 node
names once for the whole archive would be faster still and would put all 150 strings in
every one of the 28,000 tiles — about 100 MB of tables for tiles that use three nodes
each.  So values are interned once per zoom (`intern`, which does the string hashing) and
the table is then PRUNED to what each tile actually references.

Wire format (MVT v2 protobuf), for reading the byte literals below:

    Tile.layers      field 3  len-delimited   0x1a
    Layer.name       field 1  len-delimited   0x0a
    Layer.features   field 2  len-delimited   0x12
    Layer.keys       field 3  len-delimited   0x1a
    Layer.values     field 4  len-delimited   0x22
    Layer.extent     field 5  varint          0x28
    Layer.version    field 15 varint          0x78
    Feature.tags     field 2  packed varints  0x12
    Feature.type     field 3  varint          0x18   (1 = POINT)
    Feature.geometry field 4  packed varints  0x22
    Value.string     field 1  len-delimited   0x0a
    Value.int        field 4  varint          0x20

A single point's geometry is always [MoveTo(1 point), zigzag(x), zigzag(y)] with the
cursor at the origin — command integer (1 & 0x7) | (1 << 3) = 9.
"""
import numpy as np
import pandas as pd

OMIT = -1          # a property this feature does not carry, e.g. `t` on a measured dot


def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _len_delim(tag: bytes, payload: bytes) -> bytes:
    return tag + _varint(len(payload)) + payload


_ZZ_CACHE = {}


def _zigzag_table(extent: int):
    """Every coordinate in a tile is 0..extent-1, so its zigzag varint is a lookup."""
    t = _ZZ_CACHE.get(extent)
    if t is None:
        t = _ZZ_CACHE[extent] = [_varint(v << 1) for v in range(extent)]
    return t


def value_str(s) -> bytes:
    """One Layer.values entry holding a string."""
    return _len_delim(b"\x22", _len_delim(b"\x0a", str(s).encode("utf-8")))


def value_int(v) -> bytes:
    """One Layer.values entry holding an integer."""
    return _len_delim(b"\x22", b"\x20" + _varint(int(v)))


def intern(col, kind="string"):
    """Factorise one property column and pre-encode each distinct value as Value bytes.

    Do this ONCE per zoom over the whole column, not once per tile: it is the only part
    of the encode that hashes, and the codes it returns are what makes the per-tile work
    pure integer arithmetic.
    """
    codes, uniq = pd.factorize(col)
    enc = value_int if kind == "int" else value_str
    return np.asarray(codes, dtype=np.int64), [enc(v) for v in uniq]


def encode(layers, extent: int = 4096) -> bytes:
    """Encode one tile.

    layers: (name, px, py, keys, codes, tables) per layer, where
        px, py   int arrays in [0, extent)
        keys     list of property names
        codes    (n, len(keys)) int array of indices into `tables`, or OMIT
        tables   list, per key, of pre-encoded Value bytes from `intern`
    Feature order is preserved: it is the draw order, and tiles.py shuffles it on purpose.
    """
    zz = _zigzag_table(extent)
    out = bytearray()
    for name, px, py, keys, codes, tables in layers:
        n = len(px)
        if not n:
            continue
        codes = np.asarray(codes, dtype=np.int64).reshape(n, len(keys))

        # ---- prune each key's value table to what this tile references, and renumber
        # into one flat table, as the format requires.
        local = np.empty_like(codes)
        vals, offset = [], 0
        for j, table in enumerate(tables):
            col = codes[:, j]
            used = np.unique(col)
            used = used[used != OMIT]
            vals.extend(table[g] for g in used)
            pos = np.searchsorted(used, col)
            np.clip(pos, 0, max(len(used) - 1, 0), out=pos)
            local[:, j] = np.where(col == OMIT, OMIT, pos + offset)
            offset += len(used)

        # ---- one tag blob per distinct property combination. Features far outnumber
        # combinations (a tile is a few hundred marks over a handful of religions), so
        # build each blob once and index into it.
        shifted = local + 1                                  # OMIT becomes 0
        sizes = shifted.max(axis=0) + 1 if n else np.ones(len(keys), np.int64)
        packed = np.zeros(n, dtype=np.int64)
        for j in range(len(keys)):
            packed = packed * sizes[j] + shifted[:, j]
        uniq, first, inv = np.unique(packed, return_index=True, return_inverse=True)
        inv = np.asarray(inv).ravel()
        blobs = []
        for row in local[first]:
            body = b"".join(_varint(j) + _varint(int(v))
                            for j, v in enumerate(row) if v != OMIT)
            blobs.append(_len_delim(b"\x12", body))

        feats = bytearray()
        pxl, pyl, invl = (np.asarray(px).tolist(), np.asarray(py).tolist(), inv.tolist())
        for i in range(n):
            geom = b"\x09" + zz[pxl[i]] + zz[pyl[i]]
            body = blobs[invl[i]] + b"\x18\x01\x22" + bytes((len(geom),)) + geom
            feats += b"\x12" + _varint(len(body)) + body

        layer = bytearray()
        layer += _len_delim(b"\x0a", name.encode("utf-8"))
        layer += feats
        for k in keys:
            layer += _len_delim(b"\x1a", k.encode("utf-8"))
        for v in vals:
            layer += v
        layer += b"\x28" + _varint(extent)
        layer += b"\x78\x02"
        out += b"\x1a" + _varint(len(layer)) + bytes(layer)
    return bytes(out)
