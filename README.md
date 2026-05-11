# Thermo-hydro-mechanical settlement screening of energy piles

This repository contains the reproducible benchmark data and code associated with the manuscript:

**Thermo-Hydro-Mechanical Settlement Screening of Energy Piles for Civil Infrastructure Foundations**

The benchmark evaluates simplified thermo-hydro-mechanical settlement responses for civil infrastructure foundation cases, including a building core foundation, a bridge abutment retrofit, and an equipment-supported mat foundation. The archived package also includes a 240-case bridge serviceability matrix, global sensitivity regression, convergence checks, and an external order-of-magnitude validation table.

## Contents

- `code/thm_energy_pile_benchmark.py`: Python script that generates the benchmark calculations, figures, table images, and CSV outputs.
- `code/benchmark_manifest.json`: output manifest from the benchmark run.
- `data/*.csv`: time histories, scenario summaries, one-at-a-time sensitivity results, 240-case parametric matrix, global sensitivity regression, layer parameters, external validation checks, and convergence checks.
- `figures/*.png`: high-resolution manuscript figures generated from the benchmark script.
- `table-images/*.png`: high-resolution table images retained for production-stage delivery if requested.

## Reproducing the outputs

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the benchmark:

```bash
python code/thm_energy_pile_benchmark.py
```

The script writes outputs to `computational/outputs` relative to the working directory from which it is run. The CSV files in `data`, the figures in `figures`, and the table images in `table-images` are the archived outputs used for the submitted manuscript package.

## Notes

This is a reduced-order screening benchmark, not a site-specific design model. It is intended to make the manuscript calculations auditable and to support comparison against future full finite-element or field-calibrated analyses.
