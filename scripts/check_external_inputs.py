#!/usr/bin/env python3
"""
check_external_inputs.py -- validate USER-PROVIDED external inputs in data/
before running the experiments.

This script reads only local files you supply. It never downloads, embeds
or reconstructs PJM or Borg data.

Checks:
  * PJM day-ahead LMP file: parsed with the simulator's own loader
    (sim.load_pjm) over the 31-day evaluation window, so a file that passes
    here is read exactly as the experiments read it. Reports the price
    column found, the first timestamp and whether the file had to be tiled.
  * PJM generation-by-fuel file: parsed with sim.load_pjm_genmix (hourly
    water-intensity factor); unmapped fuel types are reported.
  * Borg-derived demand curves demand_curve_cell_<a-h>.csv: required
    columns and the 8929-row, 0-744 h layout written by
    scripts/prepare_borg_cell.py.
  * SHA-256 of every file against the files used in the reported
    evaluation. A mismatch is not an error -- it means results will not be
    bit-identical to the reported ones (e.g. a different download date,
    pnode or export format).

Usage (from the repository root):
  python scripts/check_external_inputs.py
  python scripts/check_external_inputs.py --price-file my_lmp.csv \
      --genmix-file my_genmix.csv
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import sim as S  # noqa: E402

DATA = os.path.join(ROOT, "data")
CELLS = "abcdefgh"
# SHA-256 of the input files used in the reported evaluation (hashes only;
# no data). Files with other hashes still run, but are not the same inputs.
REFERENCE_SHA256 = {
    "pjm_dom_34885183.csv":
        "f6ed8b3f79c46a9e736f17d04b464a602c27bf19bbc9a6cfb69ccdd6f825be48",
    "load_pjm_genmix.csv":
        "ad85df7373400db215a588a71189d460000466dc586128d39831a78b526ab95b",
    "demand_curve_cell_a.csv":
        "853ddd95092bfb2f59721922814ccb620f0dbe622365d33efe94a2e23376f5f4",
    "demand_curve_cell_b.csv":
        "cb1b8dbecf25b76492cf51123a6e65ac32f5a40c3bc092012f893685f800ab0b",
    "demand_curve_cell_c.csv":
        "ddc2dd4c90f25d2d658907763caa46da931693e98a164755cf3cbaaba7ba4a9f",
    "demand_curve_cell_d.csv":
        "4583de40f8834d254310c31b49f8a26f2af860618f6b6f5f4376388c821b21da",
    "demand_curve_cell_e.csv":
        "4253d11bed6abd13dcde1bf3b8a92dd74fdcd13f6a108ab73361dfed7d13c4f3",
    "demand_curve_cell_f.csv":
        "579eba7dfc4ce68afdbce65ac62e0f5f33ca77893759245675fd1a698953fd89",
    "demand_curve_cell_g.csv":
        "737dbe1450039402948c865eb4a024e6ddb868b09f90264de8e31106766d41a6",
    "demand_curve_cell_h.csv":
        "4f6b9cb3014e0f68dc84427def1aa994f12c18e6cea3045b5ddb065ddd570b95",
}
DEMAND_COLS = ("window_5min", "hours", "total_cpu", "rigid_cpu", "defer_low",
               "defer_high", "frac_defer")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_note(name, path):
    ref = REFERENCE_SHA256.get(name)
    if ref is None:
        return "no reference hash"
    return ("identical to the evaluation input" if sha256(path) == ref
            else "DIFFERS from the evaluation input (runs, but results "
                 "will not be bit-identical)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-file", default="pjm_dom_34885183.csv")
    ap.add_argument("--genmix-file", default="load_pjm_genmix.csv")
    a = ap.parse_args()
    os.chdir(DATA)
    ok = True
    # the 31-day evaluation window at the simulator's 60 s step
    t_grid = np.arange(0.0, 744.0 * 3600.0, 60.0)

    print("== PJM day-ahead LMP")
    if not os.path.exists(a.price_file):
        print(f"  MISSING: data/{a.price_file} (see data/README.md)")
        ok = False
    else:
        head = pd.read_csv(a.price_file, nrows=1)
        tcol = next((c for c in head.columns
                     if c.lower().startswith("datetime_beginning_ept")), None)
        if tcol:
            first = pd.read_csv(a.price_file, usecols=[tcol], nrows=1)[tcol][0]
            print(f"  first hour ({tcol}): {first} -- the evaluation reads "
                  f"744 h from the first row (January 2025 in the reported "
                  f"runs)")
        try:
            _, thresh, tiled = S.load_pjm(a.price_file, t_grid)
            print(f"  parsed by sim.load_pjm: cheap threshold (P25 over the "
                  f"window) = {thresh:.2f} $/MWh"
                  + ("; WARNING: file shorter than the window, tiled"
                     if tiled else ""))
        except SystemExit as e:
            print(f"  REJECTED by sim.load_pjm: {e}")
            ok = False
        print(f"  {hash_note(a.price_file, a.price_file)}")

    print("\n== PJM generation by fuel type")
    if not os.path.exists(a.genmix_file):
        print(f"  MISSING: data/{a.genmix_file} -- grid-embedded water will "
              f"be reported as n/a; all other metrics are unaffected")
    else:
        ewif = S.load_pjm_genmix(a.genmix_file, t_grid)
        if ewif is None:
            print("  REJECTED by sim.load_pjm_genmix (see message above)")
            ok = False
        else:
            print(f"  parsed by sim.load_pjm_genmix: {len(ewif)} steps")
        print(f"  {hash_note(a.genmix_file, a.genmix_file)}")

    print("\n== Borg-derived demand curves")
    for c in CELLS:
        name = f"demand_curve_cell_{c}.csv"
        if not os.path.exists(name):
            print(f"  MISSING: data/{name} (see data/README.md)")
            ok = False
            continue
        d = pd.read_csv(name)
        miss = [k for k in DEMAND_COLS if k not in d.columns]
        if miss:
            print(f"  {name}: missing columns {miss}")
            ok = False
            continue
        shape = (len(d) == 8929 and d.hours.iloc[0] == 0.0
                 and d.hours.iloc[-1] == 744.0)
        print(f"  {name}: {len(d)} rows, hours {d.hours.iloc[0]:g}-"
              f"{d.hours.iloc[-1]:g}"
              + ("" if shape else " (expected 8929 rows, 0-744 h)")
              + f"; {hash_note(name, name)}")
        ok &= shape

    print("\nALL REQUIRED INPUTS PRESENT AND READABLE" if ok else
          "\nSOME INPUTS ARE MISSING OR INVALID (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
