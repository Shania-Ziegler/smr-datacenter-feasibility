# Results

No result files are included in this repository.

The reported results are simulation outputs driven by PJM price and
generation-mix data, which cannot be redistributed, and by Borg-derived
workload curves, which are not duplicated here. The raw per-run outputs are
also large (the main suite alone is about 2,100 rows plus trip logs, decision
episodes and manifests). They are therefore regenerated locally rather than
stored.

## Regenerating

The commands are listed under *Reproducing the evaluation* in the
top-level README. Each run writes to `data/controller_runs/`, and
read-only analyses write to `data/analysis/`. Every run:

- writes new files only, and refuses to overwrite an existing output;
- writes a read-only manifest with the SHA-256 of the code (at start and
  end), of every input and of every output, plus the exact command line;
- reports energy-conservation residuals per row, and flags any above 1 Wh.

## Figures

The generated figures are not included either. The scripts in `figures/`
regenerate every reported figure from completed runs into `data/figures/`.
With the evaluation's inputs, the regenerated figures are pixel-identical
to the reported ones; see *Reported numbers and figures* in the top-level
README for the order of steps.

## Checking a reproduction

- **Main suite inputs:** run `python scripts/check_external_inputs.py`.
  When your input files are byte-identical to the ones used in the reported
  evaluation, it reports *identical to the evaluation input* for each file.
- **Reference rows:** `experiments/reproduce_reference_rows.py`
  re-simulates rows of your main suite and requires exact equality in every
  non-timing column.
- **One-factor experiments:** the scripts in `experiments/` re-run their
  configuration at the reference setting (for example G = 0.4) and require
  it to reproduce the matching main-suite rows exactly before writing any
  output.
