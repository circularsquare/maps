"""python patterns_at.py <cc> <station name substring> [max] : stop patterns through a station."""
import sys

sys.path.insert(0, r"C:\Users\anita\projects\maps\noritetsu")
sys.stdout.reconfigure(encoding="utf-8")
import gtfs_served  # noqa: E402

gtfs_served.WRITE_CACHE = False     # inspecting data only reads it
cc, sub = sys.argv[1], sys.argv[2].casefold()
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 15
feed = gtfs_served.load_feeds(cc, print)
st = feed["stations"]
hits = [f for f, v in st.items() if sub in v[0].casefold()]
print("stations:", [(f, st[f]) for f in hits])
n = 0
for seq, trips, days in sorted(feed["patterns"], key=lambda p: -p[1]):
    if any(f in hits for f in seq):
        print(trips, days, " > ".join(st[f][0] if f in st else f"?{f}" for f in seq))
        n += 1
        if n >= mx:
            break
