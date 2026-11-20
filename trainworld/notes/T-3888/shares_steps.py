"""T-3888 build: the hand model (notes/T-3888/shares.py) on Anita's 11-20 steps.

master: bonus 10; step 1: bonus 0; step 2: bonus 0 + train 2+2 (buses unchanged, so the bus rows
equal step 1's); step 3: + A (10% past 800 m, let off the walking rule). Bus inputs as shares.py
(T-3856's rules: no bus boarding cost, bus fare 0.8x)."""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("sh", sys.argv[1])
sh = importlib.util.module_from_spec(spec)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(sh)

DS = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]


def run(bonus, tier, **kw):
    sh.BONUS = bonus
    f = (lambda d, p: sh.tier_a(d, p, **kw)) if tier else (lambda d, p: sh.shares(d, p, **kw))
    return f


for rules, kw in (("today's bus (boarding 4, fare 1.0x)", {"board": 4.0, "fare_mult": 1.0}), ("T-3856's bus (no boarding, 0.8x)", {})):
    for place in ("dense", "suburb"):
        print(f"\n{rules}, {place}: bus % (walk %) of all trips by straight km")
        print(f"{'case':28s}" + "".join(f"{d:>12.1f}" for d in DS))
        for name, bonus, tier in (("master (bonus 10)", 10.0, False), ("bonus 0 (= with 2+2)", 0.0, False), ("bonus 0 + 2+2 + A", 0.0, True)):
            f = run(bonus, tier, **kw)
            print(f"{name:28s}" + "".join(f"{100 * f(d, place)[0]:5.1f} ({100 * f(d, place)[1]:4.1f})" for d in DS))
