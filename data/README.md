# External input data

`data/` is the working directory for all experiments. It holds the external
inputs listed below, and runs write their outputs to
`data/controller_runs/` and `data/analysis/`. All of these are excluded from
version control by `.gitignore`.

**No external data is distributed with this repository.** You need to obtain
the inputs yourself, under the terms of their providers, and place them here
under the exact file names below.

After placing them, check them with:

```bash
python scripts/check_external_inputs.py
```

The checker reads only your local files. It parses them with the
simulator's own loaders and compares each file's SHA-256 with the file used
in the reported evaluation. A different hash is allowed, for example from a
different download date or export format, but the results will then not be
bit-identical to the reported ones.

| File | Source | Used by | Required for |
|---|---|---|---|
| `pjm_dom_34885183.csv` | PJM Data Miner: day-ahead hourly LMPs | `sim.load_pjm` | grid price signal (cheap-price threshold); every experiment that uses the grid |
| `load_pjm_genmix.csv` | PJM Data Miner: generation by fuel type | `sim.load_pjm_genmix` | grid-embedded water accounting only (reported as n/a if absent) |
| `demand_curve_cell_<a-h>.csv` | derived from Google Borg 2019 traces by `scripts/prepare_borg_cell.py` | `sim.load_borg` | every workload-driven experiment and most tests |
| `borg_cell_<a-h>.csv` | your extraction from the Google Borg 2019 traces | `scripts/prepare_borg_cell.py` | regenerating the demand curves |
| `cell<a-h>_metadata.csv` | your extraction from the Google Borg 2019 traces | `scripts/build_year.py` | optional synthetic-year inputs only |

## PJM inputs (not distributed)

PJM Data Miner data are subject to PJM's terms of use. Read them before
downloading. This repository does not include, embed, sample or reconstruct
PJM data, and the scripts never download it.

### 1. Day-ahead hourly LMP — `pjm_dom_34885183.csv`

- **Source:** PJM Data Miner, *Day-Ahead Hourly LMPs*.
- **Node:** pnode 34885183 (pnode name `ACCA`, type `LOAD`, zone `DOM`).
- **Window used in the reported evaluation:** hourly rows from
  2025-01-01 00:00 EPT to 2025-12-31 23:00 EPT (8,760 rows). Every monthly
  experiment reads only the **first 744 hours of the file**, i.e. January
  2025, because the workload traces are 31 days long. A file that starts on
  2025-01-01 00:00 EPT and covers at least 745 hours therefore drives the
  same window.
- **Schema:** the standard Data Miner CSV export. The loader looks for a
  price column containing one of `total_lmp_da`, `total_lmp_rt`, `lmp`,
  `price`, `system_energy_price_da` or `total_lmp`, in that order; the
  reported runs used `total_lmp_da` ($/MWh). For time-of-day diagnostics it
  uses `datetime_beginning_ept` (or `datetime_beginning_utc`).

  Columns of the reported file: `datetime_beginning_utc,
  datetime_beginning_ept, pnode_id, pnode_name, voltage, equipment, type,
  zone, system_energy_price_da, total_lmp_da, congestion_price_da,
  marginal_loss_price_da, row_is_current, version_nbr`.
- **Use in the pipeline:**
  - The hourly price is interpolated to the one-minute grid.
  - A step counts as *cheap* when its price is at or below the 25th
    percentile of the window. Cheap steps are the only times the controllers
    may import grid power for price reasons.
  - Grid expenditure is integrated from the same price series.
- **Sanity checks:** the loader rejects files with fewer than 24 rows, files
  whose values lie in [−2, 2] (a normalized regulation signal, not a price),
  and files whose median is outside 1–500 $/MWh. A file shorter than the
  window is tiled, and this is reported on screen.

### 2. Generation by fuel type — `load_pjm_genmix.csv`

- **Source:** PJM Data Miner, *Generation by Fuel Type*, hourly, the same
  window as the price file (2025, EPT).
- **Schema:** `datetime_beginning_utc, datetime_beginning_ept, fuel_type,
  mw, fuel_percentage_of_total, is_renewable`, one row per hour and fuel
  type. The loader needs a fuel column, an `mw` column and a datetime
  column.
- **Fuel types** must be keys of `sim.FUEL_WATER_GAL_MWH`: Coal, Gas, Hydro,
  Multiple Fuels, Nuclear, Oil, Other Renewables, Solar, Storage, Wind.
  Unmapped types are counted as 0 gal/MWh, and a warning is printed.
- **Use in the pipeline:** an hourly, generation-weighted water-intensity
  factor (gal/MWh, converted to L/MWh) for grid-embedded water only. If the
  file is absent, grid water is reported as n/a and everything else is
  unchanged.
- **Gap filling:** the reported runs used a 2025 file in which 30 missing
  hourly (fuel, hour) values had been filled by linear interpolation in UTC.
  All of these lie in July and November 2025, outside the January window the
  monthly experiments read, so they do not affect the reported results. The
  gap-filling step and its log are not distributed.

### Supplying your own PJM data

1. Download the two Data Miner exports for the window above (or another
   window; see below) and save them in `data/` under the names above. You
   can also pass other names with `--price-file` / `--genmix-file` to
   `src/sim.py` and to `scripts/check_external_inputs.py`.
2. Run `python scripts/check_external_inputs.py`.
3. **Other nodes or months:** the code accepts any hourly PJM LMP export with
   the schema above, but the results are then a different experiment. The
   experiment scripts in `experiments/` also check that their inputs are
   byte-identical to the inputs recorded in your own main-suite manifest, so
   use the same files throughout one reproduction.

## Google Borg 2019 workload (not distributed)

- **Source:** the *Google Borg cluster traces, 2019* (ClusterData 2019), cells
  `a`–`h`, published by Google at
  <https://github.com/google/cluster-data>. That page gives access
  instructions, documentation and the license terms; see also the
  attribution section of the top-level README. The traces are large and are
  not duplicated here.
- **What the experiments read:** eight 31-day demand curves,
  `demand_curve_cell_<x>.csv` (about 1 MB each). Each has 8,929 rows of
  five-minute windows covering hours 0–744, with these columns:

| column | meaning |
|---|---|
| `window_5min` | window index, 1–8929 |
| `hours` | window start, hours from the trace start |
| `total_cpu` | total CPU usage of the cell in the window (normalized trace units) |
| `rigid_cpu` | usage of priority tiers `4_production` + `5_monitoring` |
| `defer_low` | usage of `1_free` + `2_best_effort_batch` (deferrable batch) |
| `defer_high` | `defer_low` + `3_mid` (batch + mid-tier) |
| `frac_defer` | `defer_low / total_cpu` |

  The simulator scales each trace so that its peak `total_cpu` corresponds
  to 85 % utilisation of a 4,000-node facility. It uses
  `fb = defer_low / total_cpu` as the batch fraction and
  `fm = (defer_high − defer_low) / total_cpu` as the mid-tier fraction. The
  remainder is rigid work.
- **Producing the demand curves:** `scripts/prepare_borg_cell.py --cell <x>`
  converts an extraction file `borg_cell_<x>.csv` into
  `demand_curve_cell_<x>.csv`, and writes four diagnostic plots. The
  extraction file is a per-window aggregate of the trace's instance usage,
  with one row per (window, priority tier, scheduling class):

| column | meaning |
|---|---|
| `window_5min` | 1–8929 (consecutive five-minute windows) |
| `priority_tier` | `1_free`, `2_best_effort_batch`, `3_mid`, `4_production`, `5_monitoring` |
| `scheduling_class` | 0–3 |
| `total_cpu`, `total_mem`, `peak_cpu` | summed usage in the window |
| `n_instances` | instances contributing to the row |

  The tier names follow the priority bands defined in the trace
  documentation. **The query that produced `borg_cell_<x>.csv` from the
  public trace is not part of this repository** (see the limitations in the
  top-level README); the schema above is what `prepare_borg_cell.py`
  requires.
- **Cell metadata (optional):** `cell<x>_metadata.csv` has columns
  `cell, machines, total_cpu_capacity, total_memory_capacity, collections`.
  It is used only by `scripts/build_year.py`, which assembles synthetic
  365-day workloads from the eight 31-day curves. Those synthetic years are
  supplementary and not part of the reported evaluation.

## Outputs written here (never commit them)

- `controller_runs/`: per-run CSVs, trip logs, decision episodes and
  write-once manifests.
- `analysis/`: read-only analyses of completed runs.
- PNG previews from `prepare_borg_cell.py` and `validate_deadtime.py`.

See `../results/README.md` for how the reported results are regenerated.
