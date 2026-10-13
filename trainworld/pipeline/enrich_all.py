"""T-142: add the non-work inputs (everyday sectors, NCES schools, IPEDS universities, airport
ground trips, medical visits, parks T-196) to finished city packs in place, one city after another. Commutes
and gravity are untouched.

    enrich_all.py bos wdc bal      (no arguments: every registered city)
"""
import sys
import time

import build_city as b
import enrich_airports
import enrich_everyday
import enrich_medical
import enrich_parks
import enrich_schools
import enrich_universities


def main(cities):
    for city in cities:
        t = time.time()
        print(f"=== {city}", flush=True)
        if b.CITIES[city].get("country") == "ca":  # T-277: Canadian adapters, pipeline/canada.py
            import canada
            canada.everyday_header(city)
            canada.schools(city)
            canada.universities(city)
            canada.enrich_osm(city)  # airports, medical, parks with OSM from the Geofabrik extract
        else:
            enrich_everyday.main(city)
            enrich_schools.main(city)
            enrich_universities.main(city)
            enrich_airports.main(city)
            enrich_medical.main(city)
            enrich_parks.main(city)
        print(f"=== {city} done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(b.CITIES))
