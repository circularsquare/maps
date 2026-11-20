"""T-3887: the frontier chart from tools/frontier.py's CSV. One panel per corridor (dense, medium, thin), net profit
a day per person or job near the line against the corridor's length, one curve per mode at its best service, and
the mode that wins at each length named above it.

    python trainworld/tools/frontier.py chart [f.csv] [--png out.png]
"""
import csv
from collections import defaultdict

ORDER = ["bus", "bus_1km", "street_tram", "tram", "metro", "rail", "hsr"]
NAMES = {"bus": "Bus", "bus_1km": "Bus, stops 1 km", "street_tram": "Street tram", "tram": "Tram",
         "metro": "Metro", "rail": "Rail", "hsr": "High speed"}
# the dataviz reference palette's categorical slots 1-7 in order (validated light), a marker shape each as well
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
PANELS = [("dense", "Dense: through the Loop"), ("medium", "Medium: Western Avenue"), ("thin", "Thin: DuPage suburbs")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def best_rows(path):
    """Per (corridor, length, mode): the service with the highest net profit."""
    best = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["corridor"], int(r["length_km"]), r["mode"])
            if k not in best or float(r["net_day"]) > float(best[k]["net_day"]):
                best[k] = r
    return best


def chart(path, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = best_rows(path)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True, facecolor=SURFACE)
    for ax, (cor, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        lengths = sorted({k[1] for k in best if k[0] == cor})
        dens = [int(best[(cor, L, "bus")]["density_km2"]) for L in lengths if (cor, L, "bus") in best]
        for i, mode in enumerate(ORDER):
            pts = [(L, float(best[(cor, L, mode)]["net_day"]) / max(1, int(best[(cor, L, mode)]["people"]) + int(best[(cor, L, mode)]["jobs"])))
                   for L in lengths if (cor, L, mode) in best]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=COLOURS[i], lw=2, marker=MARKERS[i], ms=7, mec=SURFACE, mew=1.5, label=NAMES[mode], zorder=3)
        # the frontier: the winner at each length, ringed and named
        for L in lengths:
            cand = [(float(best[(cor, L, m)]["net_day"]) / max(1, int(best[(cor, L, m)]["people"]) + int(best[(cor, L, m)]["jobs"])), m)
                    for m in ORDER if (cor, L, m) in best]
            if not cand:
                continue
            y, m = max(cand)
            pays = y > 0
            ax.plot([L], [y], marker="o", ms=16, mfc="none", mec=INK if pays else INK2, mew=1.5, ls="none", zorder=4)
            ax.annotate(NAMES[m] if pays else "None pays", (L, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK if pays else INK2)
        ax.axhline(0, color=INK2, lw=1, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(L) for L in lengths])
        ax.minorticks_off()
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("Line length, km")
        sub = f"{min(dens):,}-{max(dens):,} people and jobs a km² near the line" if dens else ""
        ax.set_title(f"{title}\n{sub}", fontsize=10.5, color=INK, loc="left", pad=22)
    axes[0].set_ylabel("Profit a day after building costs, US$ per person or job within 800 m")
    seen = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    labels = [NAMES[m] for m in ORDER if NAMES[m] in seen]
    handles = [seen[lab] for lab in labels]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9.5)
    fig.suptitle("Which mode makes most money on one straight line in Chicago, by length (today's rules)", x=0.01, ha="left",
                 fontsize=12.5, color=INK)
    fig.text(0.01, 0.925, "Each mode at its best service, from a train every minute to one an hour. Building costs spread over 60 game days. "
             "Circled and named: the winner at each length. Game dollars (x100 economy).", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    print(png)


def share_means(path):
    """(mode, spacing, trip km) -> mean shares walking, by train and by car over the starts."""
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("offset_m", "0") != "0" or r["mode"] == "rail":
                continue  # the same trains as the metro: at one stop spacing the same trip, only the building differs
            a = acc[(r["mode"], int(r["spacing_m"]), float(r["trip_km"]))]
            a[0] += float(r["p_walk"]); a[1] += float(r["p_train"]); a[2] += float(r["p_drive"]); a[3] += 1
    return {k: (w / n, t / n, d / n) for k, (w, t, d, n) in acc.items()}


def share_chart(path, png):
    """Trip length across, stop spacing up: each cell the mode with the biggest share of the trips when it alone runs
    along the line (against the car and walking), its share written in, the colour paler the smaller its lead over
    the next mode. Beside it two slices: each mode's share against trip length at 400 m and at 1,500 m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    sm = share_means(path)
    names = dict(NAMES, metro="Metro or rail")
    modes = [m for m in ORDER if any(k[0] == m for k in sm)]
    lengths = sorted({k[2] for k in sm})
    spacings = sorted({k[1] for k in sm if k[0] != "none"})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(15, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.45, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(SURFACE)
    surf = to_rgb(SURFACE)
    for j, sp in enumerate(spacings):
        for i, L in enumerate(lengths):
            sh = sorted(((sm[(m, sp, L)][1], m) for m in modes if (m, sp, L) in sm), reverse=True)
            if not sh:
                continue
            top, m = sh[0]
            lead = top - (sh[1][0] if len(sh) > 1 else 0)
            k = min(1.0, 0.25 + lead / 0.15 * 0.75) if top >= 0.005 else 0.0
            c = to_rgb(COLOURS[ORDER.index(m)])
            ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=tuple(surf[q] + (c[q] - surf[q]) * k for q in range(3)), lw=0))
            ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, ec=SURFACE, lw=2))
            label = f"{names[m]}\n{top * 100:.0f}%" if top >= 0.005 else "none"
            ax.text(i, j, label, ha="center", va="center", fontsize=8, color=INK if k < 0.75 or m in ("tram", "street_tram", "metro") else "#ffffff")
    ax.set_xlim(-0.5, len(lengths) - 0.5)
    ax.set_ylim(-0.5, len(spacings) - 0.5)
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([f"{L:g}" for L in lengths])
    ax.set_yticks(range(len(spacings)))
    ax.set_yticklabels([f"{sp:,} m" for sp in spacings])
    ax.set_xlabel("Trip length along the line, km")
    ax.set_ylabel("Stop spacing")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("The mode most people take, when it alone runs along the line\nits share of all trips; paler where the next mode is close",
                 fontsize=10.5, color=INK, loc="left")
    for row, sp in enumerate([400, 1500]):
        a2 = fig.add_subplot(gs[row, 1])
        a2.set_facecolor(SURFACE)
        for i, m in enumerate(ORDER):
            pts = [(L, sm[(m, sp, L)][1] * 100) for L in lengths if (m, sp, L) in sm]
            if pts:
                xs, ys = zip(*pts)
                a2.plot(xs, ys, color=COLOURS[i], lw=2, marker=MARKERS[i], ms=6, mec=SURFACE, mew=1.2, label=names[m])
        base = [(L, sm[("none", 0, L)][0] * 100) for L in lengths if ("none", 0, L) in sm]
        if base:
            xs, ys = zip(*base)
            a2.plot(xs, ys, color=INK2, lw=1.5, ls="--", label="Walk, with no line")
        a2.set_xscale("log")
        a2.set_xticks(lengths)
        a2.set_xticklabels([f"{L:g}" for L in lengths])
        a2.minorticks_off()
        a2.grid(True, axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            a2.spines[s].set_visible(False)
        a2.set_ylabel("Share of trips, %")
        a2.set_title(f"Stops {sp:,} m apart" + (" (bus: its own stops)" if sp == 400 else ""), fontsize=10, color=INK, loc="left")
        if row == 1:
            a2.set_xlabel("Trip length along the line, km")
    h, l = fig.axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle("Who takes what: one line along Western Avenue, Chicago, commutes in the morning peak (today's rules)", x=0.01,
                 ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.925, "Each mode built alone, a train or bus every 8 minutes at the peak; shares are of all trips "
             "(the rest walk or drive), averaged over 25 starts along the line. At 400 m the bus has its own stops.", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.84, bottom=0.14)
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    print(png)
