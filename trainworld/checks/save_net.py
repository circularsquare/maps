"""A game save's network bytes (the track model's own save), for Rust tests that load a real save.

    python trainworld/checks/save_net.py <game.save> <out file>

A .save is gzip of "TWG1", a u32 header length, the JSON header, then the network (saveFile.ts).
"""
import gzip
import struct
import sys

src, dst = sys.argv[1], sys.argv[2]
b = gzip.decompress(open(src, 'rb').read())
assert b[:4] == b'TWG1', 'not a game save'
n = struct.unpack('<I', b[4:8])[0]
open(dst, 'wb').write(b[8 + n:])
