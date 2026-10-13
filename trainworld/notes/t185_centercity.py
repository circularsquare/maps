"""T-185: where Center City residents work, T-159's groups, pre-T162 packs against the shipped ones
and LODES OD 2023 (PA and NJ files). Read-only.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t185_centercity.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t159_flows as tf  # noqa: E402
import t162_tables as tt  # noqa: E402


def main():
    L, rac, b, _ = tf.lodes()
    C2G = tf.class_to_groups()
    G = {"LODES": C2G.T @ L @ C2G}
    for lab, p in (("pre-T162", tt.BEFORE), ("now", tt.AFTER)):
        G[lab] = C2G.T @ tt.phl_model(p, b) @ C2G
    P = tf.CIDX["42101"]
    pa = [tf.CIDX[f] for f in tf.CF if f.startswith("42") and f != "42101"]
    nj = [tf.CIDX[f] for f in tf.CF if f.startswith("34")]
    print("| Center City residents work in | " + " | ".join(G) + " |")
    print("|---|" + "---|" * len(G))
    for name, cols in (("Center City", [tf.GCC]), ("Philadelphia (incl. Center City)", [P]),
                       ("PA suburbs (Bucks, Chester, Delaware, Montgomery)", pa), ("NJ counties", nj)):
        cells = []
        for M in G.values():
            tot = M[tf.GCC, :tf.NC].sum()
            cells.append(f"{M[tf.GCC, cols].sum() / tot:.1%}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    for lab, M in G.items():
        print(f"  {lab}: Center City resident commuters {M[tf.GCC, :tf.NC].sum():,.0f}")


if __name__ == "__main__":
    main()
