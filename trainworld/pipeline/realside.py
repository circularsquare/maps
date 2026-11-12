"""T-3297: set the running side of a real-network game file (`realnet.py gamesave`'s, data/packs/real/<id>.save).

    python pipeline/realside.py <in.save> <out.save> left|right

A game file is gzip of a 4-byte magic, the header's length (u32 little-endian), the header's JSON and the
network's bytes; the network's running side is the varint after its 4-byte magic and two f64 origin fields
(`save::encode`). For a network `realsave` built (every junction that is not a station flying, so no move
crosses on either side) this is the file `realsave --left` gives: only the drawing changes (measured on
London, Tokyo, Hong Kong and Singapore, notes/T-3297.md).
"""
import gzip
import struct
import sys

SIDE_AT = 4 + 8 + 8


def main():
    src, dst, side = sys.argv[1], sys.argv[2], sys.argv[3]
    if side not in ("left", "right"):
        sys.exit("side is left or right")
    raw = bytearray(gzip.decompress(open(src, "rb").read()))
    head = struct.unpack_from("<I", raw, 4)[0]
    track = 8 + head
    if raw[track:track + 3] != b"TWT" or raw[track + SIDE_AT] not in (0, 1):
        sys.exit(f"{src}: not a game file with a track network")
    was = "right" if raw[track + SIDE_AT] == 1 else "left"
    raw[track + SIDE_AT] = 1 if side == "right" else 0
    open(dst, "wb").write(gzip.compress(bytes(raw), 9))
    print(f"{src}: {was} -> {side}, {dst}")


if __name__ == "__main__":
    main()
