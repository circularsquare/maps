"""Test lines centred on each city's centre (lon, lat below), offset from the pack origin."""
import json, math, os, subprocess, sys
D = os.path.dirname(os.path.abspath(__file__))
reg = {x['id']: x for x in json.load(open('C:/Users/anita/projects/maps/trainworld/app/src/game/registry.json', encoding='utf-8'))['cities']}
CENTRE = {
    "bei": (116.3975, 39.9087, 0), "sha": (121.4737, 31.2304, 0), "shz": (114.0579, 22.5431, 0),
    "cho": (106.5770, 29.5583, 0), "nag": (108.3200, 22.8170, 0), "tay": (112.5492, 37.8570, 90),
    "nyc": (-73.985, 40.758, 61), "lon": (-0.1278, 51.5074, 0), "par": (2.3522, 48.8566, 0),
    "tyo": (139.7671, 35.6812, 0), "mun": (11.5755, 48.1374, 0),
}
os.makedirs(f"{D}/nets", exist_ok=True)
for c, (lon, lat, ang) in CENTRE.items():
    o = reg[c]['origin']
    dx = (lon - o['lon']) * 111320 * math.cos(math.radians(o['lat']))
    dy = (lat - o['lat']) * 110540
    print(c, round(dx), round(dy))
    subprocess.run([sys.executable, f"{D}/testline.py", c, "20", "1.25", f"{D}/nets/{c}_line.json", str(ang), str(dx), str(dy)], check=True)
