"""Bangladesh: Bangladesh Railway's passenger lines as a hand-written list (bd_lines.py), traced
over OSM track by rinf.py through lk_register.py's engine (nafrica_register's recipe).

    python bd_register.py --clip bd           # after every extract (drops India's track)
    python bd_register.py --convert bd        # the conversion alone, with its log
    python build_model.py --region bd --register bd_register:data/raw/rinf/bd
    python bd_register.py --trace bd "Kamalapur" "Tongi"     # a trace, to write lists

WHY A LIST. Bangladesh has no open line register with geometry or chainage (bd_sources.md):
the lines are written here from en.wikipedia's route diagrams and BR's timetables, each as
its stations in order (ends, junctions and enough stations between to hold the trace to the
right track), and `osm_stops: "all"` makes every OSM station on a traced section a stop. The
lengths are our traces (`no_chain`); BR's fare-list distances and the Information Book's
totals are the outside checks (check_model.REGISTER["bd"]).

THE LINE UNIT is BR's line as the "List of railway lines in Bangladesh" names it, cut where
part runs and part does not (rinf.py greys a whole line or none). bd_lines.py says what runs.
"""
import lk_register as eng
from bd_lines import BORDERS, LINES, NOT_SERVICE, PATH_CHECKS

eng.register_country("bd", LINES, BORDERS, NOT_SERVICE, ["bn", "en"], "BGD", False,
                      PATH_CHECKS)

build = eng.build
split_pieces = eng.split_pieces


def country_conf(cc="bd"):
    return eng.country_conf(cc)


if __name__ == "__main__":
    eng.main()
