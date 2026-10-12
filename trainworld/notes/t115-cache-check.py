"""Check that ACS city extracts reuse a shared cache without blending city rates."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import wfh

with tempfile.TemporaryDirectory() as raw:
    src = Path(raw) / "source.dat"
    pd.DataFrame({"GEO_ID": ["1400000US42101000100", "1400000US36061000100"],
                  "B08301_E001": [100, 200], "B08301_E021": [10, 100]}).to_csv(src, sep="|", index=False)
    calls = []
    def fetch(url):
        calls.append(url)
        return str(src)
    cols = ["B08301_E001", "B08301_E021"]
    nyc = wfh.extract("b08301", cols, ["36061000100"], raw, fetch)
    phl = wfh.extract("b08301", cols, ["42101000100"], raw, fetch)
    assert nyc.B08301_E021.sum() / nyc.B08301_E001.sum() == .5
    assert phl.B08301_E021.sum() / phl.B08301_E001.sum() == .1
    cached = pd.read_csv(Path(raw) / "acs2023_5y_b08301_tracts.csv", dtype={"key": str})
    assert set(cached.key) == {"36061000100", "42101000100"}
    again = wfh.extract("b08301", cols, ["36061000100"], raw, fetch)
    assert len(again) == 1 and again.iloc[0].B08301_E021 == 100
    assert len(calls) == 2, "both cities should now rebuild without fetching"
    print("ACS shared cache: city-isolated rates, both extracts retained, cached NYC reuse: OK")
