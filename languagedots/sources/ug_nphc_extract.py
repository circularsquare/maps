"""One streaming pass over Uganda's NPHC 2024 10% population file, straight out of Anita's RAR.

The same reader as religiondots' `data/raw/ug/nphc2024_aggregate_script.py` (read-only there),
keyed on ethnicity / nationality (HH_P10) instead of religion. Nothing is extracted to disk
(religiondots ask 008: extract what is needed, keep the compact aggregate). ~17 minutes, one core.

    python sources/ug_nphc_extract.py --header   # variable names and labels only, seconds
    python sources/ug_nphc_extract.py            # the full pass

Writes data/raw/ug/:
  ug2024_parish_ethnic.csv   district, county, subcounty, parish codes + names, qrtype,
                             P10 code, persons
  ug2024_variables.txt       every variable's name, type and label
  ug2024_p10_labels.txt      the HH_P10 value labels
"""
import collections
import csv
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RAR = r"C:\Users\anita\Downloads\NPHC 2024-Users  File using cpro_extract_Population_record_data.rar"
UNRAR = r"C:\Program Files\WinRAR\UnRAR.exe"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "raw", "ug")
os.makedirs(OUT, exist_ok=True)
HEADER_ONLY = "--header" in sys.argv

errf = open(os.path.join(OUT, "unrar_stderr.txt"), "wb")
p = subprocess.Popen([UNRAR, "p", "-inul", RAR], stdout=subprocess.PIPE, stderr=errf,
                     bufsize=16 * 1024 * 1024)
f = p.stdout
pos = 0


def read(n):
    global pos
    parts, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            break
        parts.append(b)
        got += len(b)
    pos += got
    return b"".join(parts)


head = read(4 * 1024 * 1024)
m = re.search(rb"<release>(\d+)</release><byteorder>(\w+)</byteorder><K>", head)
rel, e = int(m.group(1)), ("<" if m.group(2) == b"LSF" else ">")
q = m.end()
K = struct.unpack(e + ("I" if rel == 119 else "H"), head[q:q + (4 if rel == 119 else 2)])[0]
q = head.index(b"<N>", q) + 3
N = struct.unpack(e + "Q", head[q:q + 8])[0]
mp = head.index(b"<map>") + 5
offsets = struct.unpack(e + "14Q", head[mp:mp + 112])
data_off = offsets[9]
t0 = head.index(b"<variable_types>") + len(b"<variable_types>")
types = struct.unpack(e + f"{K}H", head[t0:t0 + 2 * K])


def block(tag, width):
    s = head.index(b"<" + tag + b">") + len(tag) + 2
    return [head[s + i * width:s + (i + 1) * width].split(b"\0")[0].decode("utf-8", "replace")
            for i in range(K)]


names = block(b"varnames", 129)
varlabels = block(b"variable_labels", 321)
NUM = {65526: "f8", 65527: "f4", 65528: "i4", 65529: "i2", 65530: "i1"}
fields = []
for n, t in zip(names, types):
    if t in NUM:
        fields.append((n, e + NUM[t]))
    elif 1 <= t <= 2045:
        fields.append((n, f"S{t}"))
    elif t == 32768:
        fields.append((n, e + "V8"))
    else:
        raise SystemExit(f"unknown type {t} for {n}")
dt = np.dtype(fields)
width = dt.itemsize
with open(os.path.join(OUT, "ug2024_variables.txt"), "w", encoding="utf-8") as fh:
    for (n, t), lab in zip(fields, varlabels):
        fh.write(f"{n}\t{t}\t{lab}\n")
print(f"release {rel} K={K} N={N:,} row width {width}", file=sys.stderr)
if HEADER_ONLY:
    p.kill()
    sys.exit(0)

skip = data_off - pos
if skip >= 0:
    read(skip)
    first = read(6)
    assert first == b"<data>", first
    buf = b""
else:
    buf = head[data_off:]
    assert buf[:6] == b"<data>", buf[:6]
    buf = buf[6:]


def as_int(a):
    if a.dtype.kind == "f":
        a = a.astype(float)
        return np.where(np.isfinite(a) & (np.abs(a) < 1e30), a, -1).astype(np.int64)
    return a.astype(np.int64)


counts = collections.Counter()
names_by_key = {}
rows = 0
CH = 100_000
t_start = time.time()
while rows < N:
    need = min(CH, N - rows) * width
    if len(buf) < need:
        buf += read(need - len(buf))
    take = min((len(buf) // width) * width, need)
    if take == 0:
        raise SystemExit(f"stream ended at row {rows:,}")
    arr = np.frombuffer(buf[:take], dtype=dt)
    buf = buf[take:]
    keys = np.stack([as_int(arr[c]) for c in
                     ("HH_DISTRICT", "HH_COUNTY", "HH_SUBCOUNTY", "HH_PARISH", "QRTYPE", "HH_P10")],
                    axis=1)
    uk, cnt = np.unique(keys, axis=0, return_counts=True)
    for k, n in zip(map(tuple, uk.tolist()), cnt.tolist()):
        counts[k] += n
    geo = keys[:, :4]
    ug, idx = np.unique(geo, axis=0, return_index=True)
    for g, i in zip(map(tuple, ug.tolist()), idx.tolist()):
        if g not in names_by_key:
            names_by_key[g] = tuple(arr[col][i].split(b"\0")[0].decode("utf-8", "replace").strip()
                                    for col in ("HH_COUNTYN", "HH_SUBCOUNTYN", "HH_PARISHN"))
    rows += len(arr)
    if rows % 500_000 < CH:
        print(f"{rows:,}/{N:,} rows, {time.time() - t_start:.0f}s", file=sys.stderr, flush=True)

tail = buf + f.read()
p.wait()
labels = {}
vs = tail.find(b"<value_labels>")
if vs >= 0:
    j = vs + len(b"<value_labels>")
    while tail[j:j + 5] == b"<lbl>":
        j += 5
        ln = struct.unpack(e + "I", tail[j:j + 4])[0]; j += 4
        name = tail[j:j + 129].split(b"\0")[0].decode(); j += 129 + 3
        body = tail[j:j + ln]; j += ln
        j += len(b"</lbl>")
        n = struct.unpack(e + "I", body[:4])[0]
        txtlen = struct.unpack(e + "I", body[4:8])[0]
        offs = struct.unpack(e + f"{n}I", body[8:8 + 4 * n])
        vals = struct.unpack(e + f"{n}i", body[8 + 4 * n:8 + 8 * n])
        txt = body[8 + 8 * n:8 + 8 * n + txtlen]
        labels[name] = dict(zip(vals, [txt[o:].split(b"\0")[0].decode("utf-8", "replace")
                                       for o in offs]))
with open(os.path.join(OUT, "ug2024_p10_labels.txt"), "w", encoding="utf-8") as fh:
    for v, l in sorted(labels.get("HH_P10_VS1", {}).items()):
        fh.write(f"{v}\t{l}\n")
with open(os.path.join(OUT, "ug2024_parish_ethnic.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["district", "county", "subcounty", "parish", "county_name", "subcounty_name",
                "parish_name", "qrtype", "p10", "persons"])
    for k in sorted(counts):
        g = k[:4]
        w.writerow(list(g) + list(names_by_key.get(g, ("", "", ""))) + [k[4], k[5], counts[k]])
print("done", rows, file=sys.stderr)
