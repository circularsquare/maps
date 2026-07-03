"""Head-to-head on synthetic terrain.

Shows the two findings:
  1. Greedy prune (smallest peak first) does NOT balance -- volume and prominence
     pruning give near-identical, wildly unequal territories (the brief's result).
  2. Cutting the rooted divide tree can't balance either: the global max directly
     parents a large fraction of all basins, so the root territory dominates.
  3. Balanced region-growing on the basin ADJACENCY graph DOES balance: equal-heft
     connected territories, one per major massif.

Renders terrain + the three maps + a heft-bar comparison to compare_synthetic.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage

import ascentshed as A

N = 16
n = 260


def fractal_terrain(n, seed=1):
    rng = np.random.default_rng(seed)
    z = np.zeros((n, n))
    for o in range(6):
        s = 2 ** o
        z += ndimage.gaussian_filter(rng.standard_normal((n, n)), sigma=1.5 * s) * (s ** 1.1)
    z += rng.standard_normal((n, n)) * 1e-4
    return z - z.min()


def compact(lab):
    ids = np.unique(lab)
    remap = {v: i for i, v in enumerate(ids)}
    return np.vectorize(remap.get)(lab)


def main():
    z = fractal_terrain(n)
    base = 0.0
    label, peaks = A.ascent_basins(z)
    tree = A.build_tree(z, label, base=base)
    print(f"{len(peaks)} raw basins")

    lab_prom, _ = A.prune(label, tree, N, by='prominence')
    lab_tree, _ = A.balanced_partition(label, tree, N, base=base)   # divide-tree cut
    lab_reg, seeds = A.balanced_regions(label, tree, N, base=base)  # adjacency graph

    results = []
    for name, lab in [('prune', lab_prom), ('tree-cut', lab_tree), ('regions', lab_reg)]:
        v = np.array(sorted(A.territory_volumes(lab, z, tree, base=base).values(),
                            reverse=True))
        results.append((name, lab, v))
        print(f"{name:10s} k={len(v):2d}  CV={A.cv(v):.3f}  max/min={v.max()/v.min():.1f}")

    rng = np.random.default_rng(3)
    cols = plt.cm.tab20(np.linspace(0, 1, 20))
    rng.shuffle(cols)
    cmap = ListedColormap(cols)
    hill = ndimage.gaussian_filter(np.gradient(z)[0], 1)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10.5))
    ax[0, 0].imshow(z, cmap='terrain'); ax[0, 0].set_title('terrain')
    ax[0, 0].imshow(hill, cmap='gray', alpha=0.25)
    titles = ['greedy prune (ultras)', 'balanced cut of divide TREE',
              'balanced regions on adjacency GRAPH']
    # placement: prune top-mid, tree-cut bottom-left, regions bottom-mid
    placement = [ax[0, 1], ax[1, 0], ax[1, 1]]
    for (name, lab, v), a, title in zip(results, placement, titles):
        a.imshow(compact(lab) % 20, cmap=cmap, interpolation='nearest')
        a.imshow(hill, cmap='gray', alpha=0.22)
        a.set_title(f"{title}\nCV={A.cv(v):.2f}")
    # heft bars
    axb = ax[0, 2]
    for name, _, v in results:
        axb.plot(v / v.mean(), marker='o', ms=3, label=name)
    axb.axhline(1.0, color='k', lw=0.6, ls='--')
    axb.set_title('territory heft / mean (sorted)')
    axb.set_xlabel('territory rank'); axb.legend()
    ax[1, 2].axis('off')
    for a in [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig('compare_synthetic.png', dpi=120)
    print("wrote compare_synthetic.png")


if __name__ == "__main__":
    main()
