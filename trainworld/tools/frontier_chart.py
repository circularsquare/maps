"""T-3887: the two frontier charts from tools/frontier.py's result.json. One panel per stop spacing, trip length
across; each mode at the service that makes its line the most money a day.

- share.png: the share of commutes of that length along the line that take the mode, when it alone runs there
  (against the car and walking), so the line on top is the mode most people would take. Every mode at one service
  (`SHARE_SERVICE`), so frequency does not decide; metro and rail are one line, since they carry the same trips.
- profit.png: what one trip of that length makes: its fare minus its passenger-km at the line's cost per
  passenger-km (running, upkeep, and building spread over 60 game days).

    python trainworld/tools/frontier.py charts [result.json]
"""
import json, os

ORDER = ["bus", "tram", "metro", "rail", "hsr"]
NAMES = {"bus": "Bus", "tram": "Tram", "metro": "Metro", "rail": "Rail", "hsr": "High speed"}
# the dataviz reference palette's categorical slots 1-5, light, in order; a marker shape each as well
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
AMORTIZE_DAYS = 60
SHARE_SERVICE = "B"  # a train every 4 minutes at the peaks, 8 in the evening, 15 at night


def spacing_label(m):
    return f"{m} m" if m < 1000 else f"{m / 1000:g} km"


def at_service(result, service):
    """(mode, spacing) -> the result row of `service`."""
    return {(r["run"]["mode"], r["run"]["spacing_m"]): r for r in result["results"] if "error" not in r and r["run"]["service"] == service}


def best(result):
    """(mode, spacing) -> the result row of the service with the highest net a day."""
    out = {}
    for r in result["results"]:
        if "error" in r:
            continue
        k = (r["run"]["mode"], r["run"]["spacing_m"])
        r["net_day"] = r["fares_day"] - r["running_day"] - r["upkeep_day"] - r["built"] / AMORTIZE_DAYS
        if k not in out or r["net_day"] > out[k]["net_day"]:
            out[k] = r
    return out


def trip_profit(r, fares, km):
    """A trip of `km` on the line of row `r`: its fare minus its share of the line's costs, game US$."""
    cost = r["running_day"] + r["upkeep_day"] + r["built"] / AMORTIZE_DAYS
    per_pkm = cost / r["pkm_day"] if r["pkm_day"] > 0 else float("inf")
    fare = (fares["base"] + r["fare_mul"] * fares["perKm"] * km) * 100  # the x100 economy
    return fare - per_pkm * km


def _style(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _panels(result, rows, series, value, ylabel, title, subtitle, png, zero=False, fmt=None, ylim=None, tie=0.0):
    """`rows`: (mode, spacing) -> result row; `series`: (mode, label) pairs to draw (each mode keeps its colour);
    `ylim`: a fixed range, lines running past it clipped; `tie`: values this close to the top count as level with it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    _style(plt)
    b = rows
    spacings = result["spec"]["runs"] and sorted({r["run"]["spacing_m"] for r in result["results"]})
    lengths = result["spec"]["trips"]["lengths_km"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True)
    ys_all = []
    for ax, sp in zip(axes.flat, spacings):
        ax.set_title(f"Stops {spacing_label(sp)} apart", loc="left", fontsize=11, color=INK, pad=6)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if zero:
            ax.axhline(0, color=INK2, lw=1)
        tops = {}  # per length: every (value, name) drawn there
        for mode, name in series:
            i = ORDER.index(mode)
            r = b.get((mode, sp))
            if not r:
                continue
            ys = [value(r, km) for km in lengths]
            ys_all += ys
            for km, y in zip(lengths, ys):
                if (ylim and not ylim[0] <= y <= ylim[1]) or (not zero and y < 0.5):
                    continue
                tops.setdefault(km, []).append((y, name))
            ax.plot(lengths, ys, color=COLOURS[i], lw=2, marker=MARKERS[i], ms=7, mec=SURFACE, mew=1.5, label=name, zorder=3 + i % 2)
        # the mode on top at each length named over its point; modes within `tie` of it named with it
        for km, vals in tops.items():
            y = max(v for v, _ in vals)
            names = [n for v, n in sorted(vals, reverse=True) if y - v <= tie]
            ax.annotate(" / ".join(names), (km, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK2)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(lengths))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} km"))
        if fmt:
            ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    lo, hi = ylim or (min(ys_all), max(ys_all))
    pad = 0.12 * (hi - lo)
    axes.flat[0].set_ylim((lo if ylim else min(0, lo - pad)) if zero else 0, hi + pad)
    axes.flat[0].set_xlim(lengths[0] / 1.35, lengths[-1] * 1.35)
    for ax in axes[1]:
        ax.set_xlabel("Trip length")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.012, 0.915), ncol=len(l), frameon=False, fontsize=10, handlelength=2.2)
    fig.text(0.012, 0.975, title, fontsize=14, color=INK, weight="bold", va="top")
    fig.text(0.012, 0.94, subtitle, fontsize=9.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"-> {png}")


def charts(path, out_dir):
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    fares = result["fares"]
    city = result["spec"]["city"]
    where = (f"A made-up city of {city['residents_km2']:,} residents and {city['jobs_km2']:,} jobs a km² everywhere, "
             f"one straight {result['spec']['line_km']} km line of one mode at a time.")
    share_series = [("bus", "Bus"), ("tram", "Tram"), ("metro", "Metro or rail"), ("hsr", "High speed")]
    _panels(result, at_service(result, SHARE_SERVICE), share_series, lambda r, km: 100 * next(s["transit"] for s in r["shares"] if s["trip_km"] == km),
            "Share of trips (%)", "Who would take each mode",
            where + " Every mode with a train or bus every 4 minutes at the peaks.\nMorning commutes along the line by people "
            "living within 800 m of it; the rest drive or walk. Metro and rail carry the same trips.",
            os.path.join(out_dir, "share.png"), fmt=lambda v, _: f"{v:.0f}%", tie=0.5)
    _panels(result, best(result), [(m, NAMES[m]) for m in ORDER], lambda r, km: trip_profit(r, fares, km),
            "Profit per trip (game US$)", "What one trip makes",
            where + " Each mode at the headways that make its line the most money.\nA trip's fare minus its km at the "
            f"line's cost a passenger-km (running, upkeep, building over {AMORTIZE_DAYS} game days). Lines leaving the bottom keep falling.",
            os.path.join(out_dir, "profit.png"), zero=True, fmt=lambda v, _: f"{'−' if v < 0 else ''}${abs(v):,.0f}", ylim=(-1000, 300), tie=10)
