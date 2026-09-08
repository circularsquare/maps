# -*- coding: utf-8 -*-
"""인천공항's hourly passengers, turned into the two shapes an airport rail
trip actually has.

`fetch_airport.py` writes the raw file; this turns it into what `build_od.py`'s
IPF seed wants. Read `fetch_airport.py` first for why the airport is the only
possible source -- the line-level shape is already right, and the artefact is
confined to the airport journeys themselves.

## The two directions are not the same shape, and that is the whole point

An airport is not a commuter station. People fly out in the late morning --
인천's departures peak at 10:00 -- so they travel *to* the airport early, and
the airport-bound profile has **no evening peak at all**. Arrivals peak in the
late afternoon, so the airport-*outbound* profile peaks at 17:00-18:00 and
stays high until the last train. The commuter double peak the build used to
impose on both directions is wrong in a different way in each.

## The one thing here that is modelled rather than measured

The passenger counts are measured. Converting them into *train* boardings needs
two lags, and neither is published:

  LANDING_TO_TRAIN  how long after touchdown someone is on the platform:
                    deplane, immigration, baggage, walk to the station
  TRAIN_TO_FLIGHT   how long before their flight they want to be at the
                    terminal

Both are spread over a few hours rather than applied as a single offset,
because passengers are not synchronised and a hard shift puts a spike in the
profile that the real data does not have. The numbers below are judgement, not
measurement -- they are the second invented thing in this pipeline, after
`EXTEND_LAST_HOUR`, and they are deliberately in one place so they can be
argued with.

**What is *not* assumed is the mode share.** Only the shape is used downstream
-- `build_od.py` rescales every pair to its own total -- so whatever fraction
of air passengers take the train divides out, as long as that fraction does not
itself swing across the day. Night is the exception and the timetable already
handles it: there are no trains, so nobody is put on one.

    python airport.py               # print the profiles for a month
    python airport.py --month 202312
"""

import argparse
import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

HOURLY_CSV = os.path.join(D, "incheon_airport_hourly.csv")

# Hours between the wheels touching down and the passenger being on the
# platform. Deplaning and the walk to a stand are 10-15 min, immigration runs
# 15-30, baggage another 15-25, and the station is a walk from either terminal.
# So an hour is typical, two is common with a bag, and under an hour happens
# only hand-luggage-only and a quiet queue.
LANDING_TO_TRAIN = {0: 0.15, 1: 0.55, 2: 0.30}

# Hours between arriving at the terminal and the flight leaving. 인천 advises
# three hours for an international departure and most of its traffic is
# international; observed behaviour clusters a little tighter than the advice.
TRAIN_TO_FLIGHT = {1: 0.10, 2: 0.40, 3: 0.40, 4: 0.10}

# The complexes this applies to. Deliberately only the two terminals: they are
# the only stations on the line whose traffic is all air passengers. 운서 and
# 영종 are 영종도 housing, 공항화물청사 is a workplace, and 김포공항 is mostly
# an interchange -- all three are ordinary commuter stations and already have
# the right shape.
AIRPORT_COMPLEXES = (u"인천공항1터미널", u"인천공항2터미널")


def _read(month):
    """(arrivals, departures) per clock hour, transit netted out."""
    if not os.path.exists(HOURLY_CSV):
        return None
    tot = {"arr": np.zeros(24), "dep": np.zeros(24)}
    tra = {"arr": np.zeros(24), "dep": np.zeros(24)}
    seen = False
    with io.open(HOURLY_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["month"] != str(month):
                continue
            seen = True
            into = tot if r["pas"] == "all" else tra
            h = int(r["hour"])
            into["arr"][h] += float(r["arr"])
            into["dep"][h] += float(r["dep"])
    if not seen:
        return None
    # A transit passenger never leaves the airport, so they are not rail
    # demand. At 인천 they are about a tenth of everyone, and their banks do
    # not sit at the same hours as origin-destination traffic.
    return (np.maximum(tot["arr"] - tra["arr"], 0.0),
            np.maximum(tot["dep"] - tra["dep"], 0.0))


def _smear(v, kernel, sign):
    """Spread `v` over a kernel of whole-hour offsets, wrapping at midnight."""
    out = np.zeros(24)
    for off, w in kernel.items():
        out += w * np.roll(v, sign * off)
    return out


def profiles(month, quiet=False):
    """Two normalised 24-hour shapes, or None if the file is not there.

    board_at_airport[h]  when an airport-origin rider boards the train
    reach_airport[h]     when an airport-bound rider gets off at the airport;
                         the caller shifts this back by the train ride to put
                         it in origin-departure time
    """
    got = _read(month)
    if got is None:
        if not quiet:
            print("   no 인천공항 hourly file for %s -- airport pairs will "
                  "take the system profile. Run fetch_airport.py" % month)
        return None
    arr, dep = got
    board = _smear(arr, LANDING_TO_TRAIN, +1)
    reach = _smear(dep, TRAIN_TO_FLIGHT, -1)
    if board.sum() <= 0 or reach.sum() <= 0:
        return None
    if not quiet:
        print("   인천공항 %s: %s arriving and %s departing passengers, "
              "transit netted out"
              % (month, format(int(arr.sum()), ","), format(int(dep.sum()), ",")))
    return board / board.sum(), reach / reach.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="202311")
    args = ap.parse_args()
    got = profiles(args.month)
    if got is None:
        raise SystemExit("no data for %s" % args.month)
    board, reach = got
    print("\n%-5s %12s %12s" % ("hour", "from airport", "to airport"))
    for h in range(24):
        if board[h] < 1e-4 and reach[h] < 1e-4:
            continue
        print("   %02d %10.2f%% %11.2f%%   %s" %
              (h, 100 * board[h], 100 * reach[h],
               "#" * int(round(200 * board[h]))))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
