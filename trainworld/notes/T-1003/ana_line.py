"""T-1003: the test line: rail trips a day, and how many of them walked before."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
names = {"bei": "Beijing", "sha": "Shanghai", "shz": "Shenzhen", "cho": "Chongqing", "nag": "Nanning", "tay": "Taiyuan",
         "nyc": "New York", "lon": "London", "par": "Paris", "tyo": "Tokyo", "mun": "Munich"}
print("| Zone | Walk with no rail | Rail trips a day | From walking | From the car |")
print("|---|---|---|---|---|")
for c, n in names.items():
    e = json.load(open(f"{D}/calib/{c}_empty.json"))["day"]
    l = json.load(open(f"{D}/calib/{c}_line.json"))["day"]
    trips = e["commute"][0] + e["everyday"][0]
    w0 = e["commute"][2] + e["everyday"][2]
    w1 = l["commute"][2] + l["everyday"][2]
    rail = l["commute"][1] + l["everyday"][1]
    fw = (w0 - w1) / rail
    print(f"| {n} | {100*w0/trips:.0f}% | {rail/1e3:,.0f}k | {100*fw:.0f}% | {100*(1-fw):.0f}% |")
