"""T-2606: a line's room a day (crush, 240 a car) beside its market boardings at free flow.
python cap.py <net.json> <err.json> <lines, comma>"""
import json, sys

net = json.load(open(sys.argv[1]))
err = json.load(open(sys.argv[2]))
# hours a day at each level: high 6 (two peaks), medium 10, low 8 (an approximation of the periods)
H = (6, 10, 8)
off = 0
for l in (int(v) for v in sys.argv[3].split(',')):
    tph = net['tph'][3 * l:3 * l + 3]
    cars = net['cars'][l]
    trains = sum(t * h for t, h in zip(tph, H)) * 2
    print(f'line {l}: {cars:.0f} cars, trains a day both ways {trains:.0f}, room a day at crush {trains * cars * 240:.0f} (one seat-trip a train end to end); market boardings at free flow {err["whole"][l]:.0f}, lost when cut {err["whole"][l] - err["parts"][l]:.0f}')
