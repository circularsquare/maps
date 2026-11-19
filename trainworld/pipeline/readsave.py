"""Read an anitabuilder .save (gzip, TWG1 header, TWT4/5/6 network) into plain Python data.

Mirrors sim/src/track/save.rs `decode`. Returns nodes, edges, lines (stops as node ids, path as
(edge, dir) pairs) and links. Edge lengths are straight-line through the PIs (good enough here).
"""
import gzip, json, math, struct, sys

QUANT = 0.001


class R:
    def __init__(self, b, o):
        self.b, self.o = b, o

    def u(self):
        v, sh = 0, 0
        while True:
            c = self.b[self.o]; self.o += 1
            v |= (c & 0x7F) << sh
            if not c & 0x80:
                return v
            sh += 7

    def i(self):
        u = self.u()
        return (u >> 1) ^ -(u & 1)

    def mm(self):
        return self.i() * QUANT

    def s(self):
        n = self.u(); t = self.b[self.o:self.o + n]; self.o += n
        return t.decode("utf-8")

    def tenth(self):
        return self.u() / 10.0


def read(path):
    raw = gzip.decompress(open(path, "rb").read())
    assert raw[:4] == b"TWG1"
    n = struct.unpack("<I", raw[4:8])[0]
    header = json.loads(raw[8:8 + n])
    t = raw[8 + n:]
    magic = t[:4].decode()
    v4 = magic in ("TWT4", "TWT5", "TWT6")
    r = R(t, 20)
    r.u()  # right hand
    nodes = []
    px = py = 0.0
    for _ in range(r.u()):
        x = px + r.mm(); y = py + r.mm(); px, py = x, y
        level = r.i(); flags = r.u()
        nd = {"x": x, "y": y, "level": level, "flying": bool(flags & 1)}
        if flags & 2:
            nd["platform"] = r.u(); nd["name"] = r.s(); nd["built"] = bool(flags & 4)
        nodes.append(nd)
    edges = []
    prev_a = 0
    for _ in range(r.u()):
        a = prev_a + r.i(); b = a + r.i(); prev_a = a
        tb = r.u()
        pts = [(nodes[a]["x"], nodes[a]["y"])]
        qx, qy = nodes[a]["x"], nodes[a]["y"]
        for _ in range(r.u()):
            qx += r.mm(); qy += r.mm(); r.u(); r.i(); pts.append((qx, qy))
        if tb & 8:
            qx, qy = nodes[a]["x"], nodes[a]["y"]
            for _ in range(r.u()):
                qx += r.mm(); qy += r.mm(); r.i()
        pts.append((nodes[b]["x"], nodes[b]["y"]))
        length = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
        edges.append({"a": a, "b": b, "tracks": tb & 3, "built": bool(tb & 4), "crossover": bool(tb & 32), "hsr": bool(tb & 64), "len": length, "pts": pts})
    lines = []
    for _ in range(r.u()):
        name = r.s(); colour = r.u()
        tph = [r.tenth(), r.tenth(), r.tenth()]
        trains = None
        if v4 and r.u():
            trains = [r.u(), r.u(), r.u()]
        r.tenth(); r.tenth(); r.u()
        stops = []; p = 0
        for _ in range(r.u()):
            p += r.i(); stops.append(p)
        path = []; p = 0
        for _ in range(r.u()):
            v = r.i(); p += v >> 1; path.append((p, v & 1))
        lines.append({"name": name, "colour": "#%06x" % colour, "tph": tph, "trains": trains, "stops": stops, "path": path})
    symbols = None
    links = []
    if t[r.o:r.o + 4] == b"LUI1":
        r.o += 4; r.u(); n_l = r.u(); symbols = [r.s() for _ in range(n_l)]
        for l, s in zip(lines, symbols):
            l["symbol"] = s
    if t[r.o:r.o + 4] == b"LNK1":
        r.o += 4
        for _ in range(r.u()):
            links.append((r.u(), r.u()))
    return {"magic": magic, "header": header, "nodes": nodes, "edges": edges, "lines": lines, "links": links}


if __name__ == "__main__":
    d = read(sys.argv[1])
    print(d["magic"], len(d["nodes"]), "nodes", len(d["edges"]), "edges", len(d["lines"]), "lines", len(d["links"]), "links")
    for i, l in enumerate(d["lines"]):
        names = [d["nodes"][s].get("name", "?") for s in l["stops"]]
        print(i, repr(l["name"]), l.get("symbol"), l["colour"], len(l["stops"]), "stops", len(l["path"]), "edges", names[0], "->", names[-1])
