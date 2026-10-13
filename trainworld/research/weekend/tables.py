# Weekend period shares: the model's weekday shares times NHTS 2017's weekend/weekday ratio per
# period, renormalised. Period order: 6-10, 10-16, 16-20, 20-24, 0-6.
def nhts(p):  # NHTS lists 0-6 first
    return p[1:] + p[:1]

N = {  # (weekday, weekend) NHTS shares, NHTS order 0-6,6-10,10-16,16-20,20-24
    "work_out": ([8.3, 56.2, 28.5, 5.5, 1.4], [8.9, 41.7, 36.4, 10.1, 2.9]),
    "work_ret": ([2.9, 2.9, 24.9, 57.2, 12.0], [6.4, 6.9, 25.6, 38.2, 22.9]),
    "err_out": ([0.5, 15.2, 57.4, 23.6, 3.4], [0.4, 14.6, 64.5, 17.7, 2.7]),
    "err_ret": ([0.4, 6.5, 48.8, 35.7, 8.6], [0.2, 7.1, 58.1, 27.4, 7.1]),
    "soc_out": ([1.8, 15.3, 40.5, 36.5, 5.9], [0.8, 21.1, 49.8, 24.1, 4.2]),
    "soc_ret": ([2.0, 6.9, 25.6, 36.1, 29.4], [2.3, 5.3, 40.2, 32.3, 20.0]),
    "med_out": ([1.2, 35.1, 55.7, 7.6, 0.5], [2.8, 39.2, 48.0, 9.1, 0.8]),
    "med_ret": ([0.8, 10.5, 61.7, 24.9, 2.1], [0.6, 11.8, 56.7, 21.3, 9.6]),
    "long": ([6.6, 25.1, 41.3, 22.8, 4.2], [3.7, 20.7, 45.0, 23.4, 7.3]),
}
MODEL = {
    "work_out": [0.70, 0.12, 0.06, 0.04, 0.08], "work_ret": [0.02, 0.18, 0.58, 0.16, 0.06],
    "err_out": [0.10, 0.55, 0.30, 0.04, 0.01], "err_ret": [0.06, 0.49, 0.36, 0.08, 0.01],
    "soc_out": [0.05, 0.35, 0.42, 0.17, 0.01], "soc_ret": [0.02, 0.25, 0.43, 0.27, 0.03],
    "med_out": [0.30, 0.55, 0.12, 0.02, 0.01], "med_ret": [0.08, 0.60, 0.25, 0.05, 0.02],
    "long": [0.20, 0.35, 0.25, 0.15, 0.05],
}
for k, m in MODEL.items():
    wd, we = (nhts(v) for v in N[k])
    raw = [m[q] * we[q] / wd[q] for q in range(5)]
    s = sum(raw)
    out = [round(v / s, 3) for v in raw]
    out[max(range(5), key=lambda q: out[q])] += round(1 - sum(out), 3)
    print(k, [round(v, 3) for v in out], round(sum(out), 3))
