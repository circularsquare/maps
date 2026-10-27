"""Download, build and clean up GHS-POP epochs one after another.

    python -I run_epochs.py 2020 2025 2000 1975

Per epoch: download the 100 m zip (resuming, restarting stalls), claim cores in the shared
ledger (../cores.py), unzip the GeoTIFF and delete the zip, run build.py, release the claim,
delete the GeoTIFF. The next epoch's zip is fetched while the current one builds, so at most
one GeoTIFF and one zip are on disk (about 12 GB).
"""
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.dirname(HERE)
PY = sys.executable
RAW = os.path.join(HERE, "raw", "ghsl")
LOGS = os.path.join(HERE, "logs")
URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
       "GHS_POP_E{e}_GLOBE_R2023A_54009_100/V1-0/GHS_POP_E{e}_GLOBE_R2023A_54009_100_V1_0.zip")
NAME = "GHS_POP_E{e}_GLOBE_R2023A_54009_100_V1_0"
CURL = r"C:\Windows\System32\curl.exe"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def size_of(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download(e):
    url, out = URL.format(e=e), os.path.join(RAW, NAME.format(e=e) + ".zip")
    want = size_of(url)
    for n in range(200):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if have >= want:
            log(f"{e}: zip complete ({have:,} bytes)")
            return out
        subprocess.run([CURL, "-sS", "-L", "-C", "-", "-Y", "100000", "-y", "60", "-o", out, url])
        log(f"{e}: pass {n + 1}, {os.path.getsize(out):,} of {want:,}")
    raise RuntimeError(f"download of {e} did not finish")


def cores(*args):
    r = subprocess.run([PY, "-I", os.path.join(MAPS, "cores.py"), *args],
                       capture_output=True, text=True)
    log("cores: " + r.stdout.strip().replace("\n", " | "))
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def unzip(e, zpath):
    """Extract the GeoTIFF and delete the zip (reading through /vsizip/ was 4x slower)."""
    tif = os.path.join(RAW, NAME.format(e=e) + ".tif")
    if not os.path.exists(tif):
        subprocess.run([r"C:\Windows\System32\tar.exe", "-xf", zpath, "-C", RAW,
                        NAME.format(e=e) + ".tif"], check=True)
    os.remove(zpath)
    log(f"{e}: unzipped, zip deleted")
    return tif


def build(e, zpath):
    tif = unzip(e, zpath)
    n = cores("claim", "6", "liveddensity", "--pid", str(os.getpid()), f"GHSL subnational build {e}")
    try:
        workers = max(1, int(n))
    except ValueError:
        workers = 1
    log(f"{e}: building with {workers} workers")
    with open(os.path.join(LOGS, f"build_{e}.log"), "w", encoding="utf-8") as f:
        r = subprocess.run([PY, "-I", "-u", os.path.join(HERE, "build.py"), str(e), tif,
                            "--workers", str(workers)], stdout=f, stderr=subprocess.STDOUT)
    cores("release", "liveddensity")
    if r.returncode != 0:
        raise RuntimeError(f"build {e} failed, see logs/build_{e}.log")
    log(f"{e}: build done")
    os.remove(tif)
    log(f"{e}: tif deleted")


def main():
    epochs = [int(a) for a in sys.argv[1:]]
    os.makedirs(RAW, exist_ok=True)
    nxt = None
    zpath = download(epochs[0])
    for i, e in enumerate(epochs):
        if i + 1 < len(epochs):
            # fetch the next zip while this one builds
            nxt = subprocess.Popen([PY, "-I", __file__, "--download-only", str(epochs[i + 1])])
        build(e, zpath)
        if nxt:
            nxt.wait()
            zpath = os.path.join(RAW, NAME.format(e=epochs[i + 1]) + ".zip")
            nxt = None
    log("all epochs done")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--download-only":
        download(int(sys.argv[2]))
    else:
        main()
