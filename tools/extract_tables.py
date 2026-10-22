#!/usr/bin/env python3
"""
extract_tables.py -- regenerate every quantitative claim in the dissertation
from the raw measurement files in results-final/.

Nothing in the report is copied from an analysis note.  Every table and every
figure coordinate is produced here, from gpu.csv, cpu.csv and the ERT
roofline.json of each of the four machines, so that any number can be traced to
the run that produced it.

Usage:
    python3 tools/extract_tables.py            # print every table
    python3 tools/extract_tables.py ridge      # print one table by name
"""

import csv
import json
import os
import sys
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "results-final"))

# Ordered by ridge point, which is the order every table in the report uses.
CARDS = [
    ("RTX 3050", "cobra-01_rtx3050_20260827", "cobra-01", "Intel Core i5-9500"),
    ("GTX 1080", "panda-01_gtx1080_20260826", "panda-01", "Intel Core i5-8500"),
    ("V100",     "telos_v100_20260828",       "telos",    "2x Xeon Gold 6252"),
    ("H100",     "demos_h100_20260831",       "demos",    "2x EPYC 9334"),
]

GRIDS = ["128", "256", "320", "384", "512"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def gpu_rows(dirname):
    """Only rows that completed and ran the kernel that was requested."""
    p = os.path.join(ROOT, dirname, "gpu.csv")
    return [r for r in csv.DictReader(open(p))
            if r["status"] == "ok" and r["honoured"] == "yes"]


def cpu_rows(dirname):
    p = os.path.join(ROOT, dirname, "cpu.csv")
    return [r for r in csv.DictReader(open(p)) if r.get("status", "ok") == "ok"]


def ert(dirname, prec="fp64"):
    p = os.path.join(ROOT, dirname, "01_roofline_ert", prec, "roofline.json")
    e = json.load(open(p))["empirical"]
    out = {"compute": e["gflops"]["data"][0][1]}
    for name, val in e["gbytes"]["data"]:
        out[name] = val
    out["ridge"] = out["compute"] / out["DRAM"]
    return out


def best(rows, algo, col, N, prec):
    """Fastest time for one algorithm in one direction, over all lane counts.

    A row is only admitted if the kernel that actually ran in that direction
    was the one the row claims, which is what makes a fallback impossible to
    mistake for a result.
    """
    d = col[0]  # 'x', 'y' or 'z'
    v = [(float(r[col]), r["lanes"]) for r in rows
         if r["algo"] == algo and r["N"] == N and r["precision"] == prec
         and r[d + "_kernel"] == algo and float(r[col]) > 0]
    return min(v) if v else None


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def t_ridge():
    print("== ridge points, ERT-measured (report Table: tab:ridge) ==")
    print(f"{'GPU':10s} {'FP64 compute':>13s} {'DRAM':>9s} {'L1':>9s} "
          f"{'ridge':>7s}  regime")
    for name, d, _, _ in CARDS:
        e = ert(d)
        regime = "compute-bound" if e["ridge"] < 2.6 else "memory-bound"
        print(f"{name:10s} {e['compute']:10.1f} GF {e['DRAM']:8.1f} "
              f"{e.get('L1', 0):8.1f} {e['ridge']:7.2f}  {regime}")
    print("  kernel arithmetic intensity: 2.6 to 3.2 FLOP/byte")


def t_hybrid_vs_transpose():
    print("\n== hybrid over Global-Transpose, x direction (tab:fp64x) ==")
    for prec in ("double", "float"):
        print(f"  -- {prec}")
        print(f"  {'GPU':10s}" + "".join(f"{('N='+g):>16s}" for g in GRIDS))
        for name, d, _, _ in CARDS:
            rows = gpu_rows(d)
            cells = []
            for g in GRIDS:
                tr = best(rows, "transpose", "x_ms", g, prec)
                hy = best(rows, "thomas-pcr", "x_ms", g, prec)
                cells.append(f"{tr[0]/hy[0]:5.2f}x L{hy[1]:<2s}     "[:16]
                             if tr and hy else f"{'--':>16s}")
            print(f"  {name:10s}" + "".join(cells))


def t_hybrid_vs_naive_yz():
    print("\n== hybrid over naive, y and z, FP64 (tab:yz) ==")
    print(f"  {'GPU':10s}" + "".join(f"{(d+g):>9s}"
          for g in GRIDS for d in ("y", "z")))
    for name, dn, _, _ in CARDS:
        rows = gpu_rows(dn)
        cells = []
        for g in GRIDS:
            for col in ("y_ms", "z_ms"):
                nv = best(rows, "naive", col, g, "double")
                hy = best(rows, "thomas-pcr", col, g, "double")
                cells.append(f"{nv[0]/hy[0]:8.2f}" if nv and hy else f"{'--':>9s}")
        print(f"  {name:10s}" + "".join(cells))


def t_per_element():
    print("\n== hybrid FP64 x, ns per element at L=32, against M = N/32 "
          "(tab:nselem) ==")
    print(f"  {'GPU':10s}" + "".join(f"{('M='+str(int(g)//32)):>10s}"
          for g in GRIDS))
    for name, dn, _, _ in CARDS:
        rows = [r for r in gpu_rows(dn) if r["precision"] == "double"
                and r["algo"] == "thomas-pcr" and r["lanes"] == "32"]
        cells = []
        for g in GRIDS:
            r = [x for x in rows if x["N"] == g]
            if r:
                N = int(g)
                cells.append(f"{float(r[0]['x_ms'])*1e6/N**3:10.4f}")
            else:
                cells.append(f"{'--':>10s}")
        print(f"  {name:10s}" + "".join(cells))


def t_precision():
    print("\n== FP64 : FP32, best general-purpose per-direction sum (tab:prec) ==")
    print(f"  {'GPU':10s} {'N':>5s} {'FP64':>9s} {'FP32':>9s} {'ratio':>7s}")
    for name, dn, _, _ in CARDS:
        rows = gpu_rows(dn)
        for g in GRIDS:
            tot = {}
            for prec in ("double", "float"):
                s = 0.0
                ok = True
                for col in ("x_ms", "y_ms", "z_ms"):
                    cand = [best(rows, a, col, g, prec)
                            for a in ("naive", "transpose", "thomas-pcr")]
                    cand = [c for c in cand if c]
                    if not cand:
                        ok = False
                        break
                    s += min(cand)[0]
                tot[prec] = s if ok else None
            if tot.get("double") and tot.get("float"):
                print(f"  {name:10s} {g:>5s} {tot['double']:9.2f} "
                      f"{tot['float']:9.2f} {tot['double']/tot['float']:6.2f}x")


def t_cpu_scaling():
    print("\n== CPU thread scan at 256^3 (fig:cpuscaling, tab:cpuscaling) ==")
    for name, dn, host, cpu in CARDS:
        rows = [r for r in cpu_rows(dn) if r["N"] == "256"]
        for prec in ("double", "float"):
            sub = sorted([r for r in rows if r["precision"] == prec],
                         key=lambda r: int(r["threads"]))
            if len(sub) < 3:
                continue
            t1 = float(sub[0]["total_ms"])
            b = min(sub, key=lambda r: float(r["total_ms"]))
            pts = " ".join(f"{r['threads']}:{float(r['total_ms']):.1f}"
                           for r in sub)
            print(f"  {host:9s} {cpu:20s} {prec:6s} {pts}")
            print(f"  {'':9s} {'':20s} {'':6s} best {b['threads']}t = "
                  f"{float(b['total_ms']):.2f} ms, {t1/float(b['total_ms']):.2f}x "
                  f"from 1 thread; z share "
                  f"{100*float(b['z_ms'])/float(b['total_ms']):.0f}%")


def t_speedup():
    print("\n== best GPU against best CPU on the same host (tab:speedup) ==")
    print(f"  {'GPU':10s} {'N':>5s} {'prec':6s} {'CPU ms':>9s} {'thr':>4s} "
          f"{'gen ms':>8s} {'x':>7s} {'ADI ms':>8s} {'x':>7s}")
    for name, dn, host, _ in CARDS:
        g_rows, c_rows = gpu_rows(dn), cpu_rows(dn)
        for g in GRIDS:
            for prec in ("double", "float"):
                cpu = [r for r in c_rows if r["N"] == g and r["precision"] == prec]
                if not cpu:
                    continue
                cb = min(cpu, key=lambda r: float(r["total_ms"]))
                gen = 0.0
                adi = 0.0
                ok = True
                for col in ("x_ms", "y_ms", "z_ms"):
                    cand = [best(g_rows, a, col, g, prec)
                            for a in ("naive", "transpose", "thomas-pcr")]
                    cand = [c for c in cand if c]
                    sf = best(g_rows, "shared-fact", col, g, prec)
                    if not cand or not sf:
                        ok = False
                        break
                    gen += min(cand)[0]
                    adi += sf[0]
                if not ok:
                    continue
                c = float(cb["total_ms"])
                print(f"  {name:10s} {g:>5s} {prec:6s} {c:9.2f} "
                      f"{cb['threads']:>4s} {gen:8.3f} {c/gen:6.1f}x "
                      f"{adi:8.3f} {c/adi:6.1f}x")


def t_winners():
    print("\n== fastest general-purpose kernel per direction (tab:winners) ==")
    for name, dn, _, _ in CARDS:
        rows = gpu_rows(dn)
        for prec in ("double", "float"):
            for g in GRIDS:
                out = []
                for col in ("x_ms", "y_ms", "z_ms"):
                    cand = []
                    for a in ("naive", "transpose", "thomas-pcr"):
                        b = best(rows, a, col, g, prec)
                        if b:
                            cand.append((b[0], a, b[1]))
                    if not cand:
                        out.append("--")
                        continue
                    t, a, L = min(cand)
                    out.append(f"{a}{'' if L == '-' else ' L'+L} {t:.3f}")
                if out[0] != "--":
                    print(f"  {name:10s} {prec:6s} N={g:<4s} "
                          f"x: {out[0]:22s} y: {out[1]:22s} z: {out[2]}")


def t_naive_baseline():
    print("\n== Algorithm 1 (naive) per direction (tab:naivebaseline) ==")
    print(f"  {'GPU':10s} {'prec':6s} {'N':>5s} {'x':>9s} {'y':>8s} {'z':>8s}")
    for name, dn, _, _ in CARDS:
        rows = gpu_rows(dn)
        for prec in ("double", "float"):
            for g in GRIDS:
                r = [x for x in rows if x["algo"] == "naive" and x["N"] == g
                     and x["precision"] == prec]
                if r:
                    print(f"  {name:10s} {prec:6s} {g:>5s} "
                          f"{float(r[0]['x_ms']):9.3f} {float(r[0]['y_ms']):8.3f} "
                          f"{float(r[0]['z_ms']):8.3f}")


def t_provenance():
    print("\n== provenance ==")
    for name, dn, host, _ in CARDS:
        env = os.path.join(ROOT, dn, "00_environment.txt")
        commit = date = "?"
        for line in open(os.path.join(ROOT, dn, "SUMMARY.txt")):
            if line.startswith("git commit"):
                commit = line.split(":", 1)[1].strip()
            if line.startswith("date"):
                date = line.split(":", 1)[1].strip()
        n = len(gpu_rows(dn))
        print(f"  {name:10s} {dn:32s} commit {commit:10s} {date[:10]}  "
              f"{n} honoured GPU rows")


def t_per_direction():
    """The numbers behind Figure fig:raw256 (256^3) and Figure fig:gridall (all N).

    A dash means the kernel does not service that direction and falls back, so
    no time is attributable to it. Algorithm 2 is x-only by design.
    """
    algos = ["naive", "transpose", "thomas-pcr", "shared-fact"]
    for grids, title in ((["256"], "per direction at 256^3, FP64 (fig:raw256)"),
                         (GRIDS, "per direction at every grid, FP64 (fig:gridall)")):
        print(f"\n== {title} ==")
        print(f"  {'machine':10s} {'algorithm':13s} {'N':>4s} "
              f"{'x_ms':>10s} {'y_ms':>10s} {'z_ms':>10s}")
        for name, dn, _h, _c in CARDS:
            rows = gpu_rows(dn)
            for a in algos:
                for g in grids:
                    cells = []
                    for col in ("x_ms", "y_ms", "z_ms"):
                        b = best(rows, a, col, g, "double")
                        cells.append(f"{b[0]:10.3f}" if b else f"{'--':>10s}")
                    if any("--" not in c for c in cells):
                        print(f"  {name:10s} {a:13s} {g:>4s} " + " ".join(cells))



TABLES = {
    "ridge": t_ridge, "xratio": t_hybrid_vs_transpose,
    "yz": t_hybrid_vs_naive_yz, "nselem": t_per_element,
    "precision": t_precision, "cpu": t_cpu_scaling, "speedup": t_speedup,
    "winners": t_winners, "perdir": t_per_direction, "naive": t_naive_baseline, "prov": t_provenance,
}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TABLES)
    for k in want:
        if k not in TABLES:
            print(f"unknown table '{k}'; known: {', '.join(TABLES)}")
            sys.exit(1)
        TABLES[k]()
