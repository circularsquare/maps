"""Fetch every sample box (fetch_box.py) one after another; log to stdout."""
import os
import sys
import shutil
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BOXES = {
    # name: (west, south, east, north), what it samples
    "amazon_manaus": (-61.5, -3.8, -58.5, -2.0),      # forest lowland
    "amazon_coari": (-64.5, -4.6, -62.5, -3.2),
    "amazon_peru": (-75.0, -5.0, -73.0, -3.5),
    "congo_cuvette": (17.5, -1.5, 20.0, 0.5),
    "borneo_kapuas": (111.0, -0.5, 113.0, 1.0),
    "newguinea_fly": (141.0, -8.0, 143.0, -6.5),
    "laos_mountains": (102.0, 18.0, 104.0, 20.0),     # forest, hilly
    "cerrado": (-48.0, -16.0, -46.0, -14.0),          # tropical, little forest
    "sahel_savanna": (-2.0, 9.0, 1.0, 11.0),
    "ganges": (81.5, 25.0, 84.0, 26.5),
    "siberia_taiga": (70.0, 58.5, 73.0, 60.0),
    "canada_lowland": (-85.0, 50.0, -82.0, 51.5),
    "finland": (25.0, 61.0, 28.0, 62.5),
    "poland_plain": (18.0, 51.5, 21.0, 53.0),
    "alps": (9.0, 46.0, 11.0, 47.5),
    "us_midwest": (-92.0, 41.0, -89.5, 42.5),
    "us_mississippi_low": (-91.5, 32.0, -90.0, 34.0),
    "colorado_plateau": (-113.5, 35.8, -111.5, 36.9),
    "beaverhead": (-113.1, 44.9, -111.4, 46.0),
    "itasca": (-95.6, 46.8, -94.0, 47.7),
    "murray_darling": (144.0, -36.0, 147.0, -34.5),
    # north of 60N RiverATLAS comes from HYDRO1k (1 km)
    "yakutia_north": (125.0, 62.0, 128.0, 63.5),
    "canada_nwt": (-120.0, 61.0, -117.0, 62.5),
    "sweden_north": (18.0, 65.5, 21.0, 66.5),
    # more tropical forest, and dry country
    "amazon_south": (-56.0, -10.0, -54.0, -8.5),
    "congo_east": (24.0, 0.0, 26.0, 1.5),
    "sumatra_lowland": (102.0, -2.0, 104.0, -0.5),
    "kenya_dry": (38.0, -1.5, 40.0, 0.5),
    # more north of 60N
    "yenisei_north": (88.0, 62.0, 91.0, 63.5),
    "lena_60": (115.0, 60.0, 118.0, 61.5),
    "yukon_alaska": (-155.0, 64.0, -151.0, 65.5),
    "mackenzie": (-128.0, 64.0, -125.0, 65.5),
    "ob_north": (70.0, 63.0, 73.0, 64.5),
    "pechora": (52.0, 63.0, 55.0, 64.5),
}

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, (w, s, e, n) in BOXES.items():
        if only and name not in only:
            continue
        free = shutil.disk_usage(HERE).free / 1e9
        if free < 45:
            print("stopping: %.1f GB free" % free, flush=True)
            break
        t0 = time.time()
        for attempt in range(3):
            p = subprocess.run([sys.executable, os.path.join(HERE, "fetch_box.py"), name, str(w), str(s), str(e), str(n)],
                               capture_output=True, text=True)
            if p.returncode == 0:
                break
            print("retry", name, p.stderr[-500:], flush=True)
            time.sleep(10)
        print(time.strftime("%H:%M:%S"), p.stdout.strip(), "(%.0f s)" % (time.time() - t0), flush=True)
    print("done", flush=True)
