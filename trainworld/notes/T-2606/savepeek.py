"""T-2606: a save's lines with their names, cars and any riders figures it keeps.
python savepeek.py <save> <line ids, comma (network order index not used; prints all long ones)>"""
import gzip, json, sys, zlib

raw = open(sys.argv[1], 'rb').read()
for f in (lambda b: gzip.decompress(b), lambda b: zlib.decompress(b, -15), lambda b: zlib.decompress(b)):
    try:
        txt = f(raw)
        break
    except Exception:
        continue
else:
    sys.exit('not gzip/deflate')
print('bytes', len(txt), txt[:300])
n = int.from_bytes(txt[4:8], 'little')
j = json.loads(txt[8:8 + n])
print('keys', list(j))


def walk(o, path, depth=0):
    if depth > 3:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if any(w in k.lower() for w in ('rider', 'demand', 'board', 'today')):
                print(path + '.' + k, type(v).__name__, str(v)[:200])
            walk(v, path + '.' + k, depth + 1)
    elif isinstance(o, list) and o:
        walk(o[0], path + '[0]', depth + 1)


walk(j, '')
