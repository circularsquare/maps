"""T-918: the curves as functions of the cost difference; table and plot."""
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/softcurve'
os.makedirs(OUT, exist_ok=True)
sig = lambda x: 1 / (1 + math.exp(-max(-80, min(80, x))))


def bt(car):
    return 0.10 * math.sqrt(min(1.0, 60.0 / car))


def bounds(r, rail, car, on):
    if not on:
        return r
    g = lambda x: max(0.0, min(1.0, (3 - x) / 2))
    lo, hi = 0.05 * g(rail / car), 0.05 * g(car / rail)
    return lo + (1 - lo - hi) * r


def make(kind, a=1.0, w=0.0, g=0.0, b=False):
    def f(car, rail):
        d, s = car - rail, bt(car) * a
        if kind == 'logit':
            r = sig(s * d)
        elif kind == 'mix':
            r = (1 - w) * sig(s * d) + w * sig(s * 0.25 * d)
        elif kind == 'atan':
            r = 0.5 + math.atan(s * d) / math.pi
        else:
            r = sig(g * math.log(car / rail))
        return bounds(r, rail, car, b)
    return f


CURVES = [
    ('Today (logistic 0.10, 5% bounds)', make('logit', b=True)),
    ('Logistic 0.10, no bounds', make('logit')),
    ('Softer logistic 0.07', make('logit', a=0.7)),
    ('Mix: 90% at 0.109, 10% at a quarter', make('mix', a=1.0942, w=0.1)),
    ('Mix: 80% at 0.121, 20% at a quarter', make('mix', a=1.2088, w=0.2)),
    ('Mix: 70% at 0.135, 30% at a quarter', make('mix', a=1.3507, w=0.3)),
    ('Arctan k 0.089', make('atan', a=0.8875)),
    ('Log ratio, power 3.5', make('ratio', g=3.476)),
    ('Mix: 85% at 0.115, 15% at a quarter', make('mix', a=1.1486, w=0.15)),
    ('Log ratio, power 3', make('ratio', g=3.0)),
    ('Log ratio, power 5.5', make('ratio', g=5.485)),
]

DS = (5, 10, 20, 30, 60, 120)
for car in (30.0, 90.0):
    print(f'\nShare taking the cheaper mode; the car costs {car:.0f}, rail that plus D (perceived minutes)')
    print('| Curve | ' + ' | '.join(f'{d}' for d in DS) + ' |')
    for name, f in CURVES:
        print(f'| {name} | ' + ' | '.join(f'{100 * (1 - f(car, car + d)):.1f}' for d in DS) + ' |')
print('\nratio form, rail 10 worse, by car cost:', [(c, round(100 * (1 - make('ratio', g=3.476)(c, c + 10)), 1)) for c in (10, 20, 30, 60, 120)])

COL = ['#2a78d6', '#888888', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
SHOW = [0, 2, 8, 6, 9]
STY = ['-', '-', '--', '-.', ':']
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, car in zip(axs, (30.0, 90.0)):
    xs = [x * 0.5 for x in range(-240, 241)]
    for k, i in enumerate(SHOW):
        name, f = CURVES[i]
        ys = [100 * f(car, car + x) if car + x > 1 else None for x in xs]
        ax.plot(xs, ys, STY[k], color=COL[k], lw=2, label=name)
    ax.set_title(f'Car costs {car:.0f} perceived minutes', fontsize=11)
    ax.set_xlabel('Rail cost minus car cost (perceived minutes)')
    ax.grid(color='#e6e6e6', lw=0.8)
    ax.set_xlim(-car, 120)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("Rail's share of rail + car (%)")
axs[0].legend(fontsize=8, frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig(f'{OUT}/curves.png', dpi=130)

# the tails on a log scale
fig, ax = plt.subplots(figsize=(7, 4.6))
xs = [x for x in range(0, 181)]
for k, i in enumerate(SHOW):
    name, f = CURVES[i]
    ax.plot(xs, [100 * f(30.0, 30.0 + x) for x in xs], STY[k], color=COL[k], lw=2, label=name)
ax.set_yscale('log')
ax.set_ylim(0.01, 60)
ax.set_xlabel('Rail costs this many perceived minutes more than a 30 minute car trip')
ax.set_ylabel("Rail's share of rail + car (%, log)")
ax.grid(color='#e6e6e6', lw=0.8, which='both')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f'{OUT}/tails.png', dpi=130)
print('plots in', OUT)
