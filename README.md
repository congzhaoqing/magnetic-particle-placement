# Where the magnet sits sets the sign of the scaling law

Scripts, computed tables and figure source data for a quantitative comparison of membrane-anchored and
intracellular magnetic-particle actuation of cells at matched single-cell magnetic moment.

No new experimental data were generated. Every number in the manuscript is produced by the scripts in
`scripts/`; no parameter is fitted to any dataset the model is tested against.

## Supplementary tables cited in the manuscript

| | Content | File |
|---|---|---|
| **Table S1** | Provenance of all 142 physical inputs entering either arm, each classified as measured, inferred, assumed or computed here, with source and locator. | `tables/Table1_parameter_provenance_merged.csv` |
| **Table S2** | Validation ladder: pre-registered pass criteria and every measured quantity for all four levels, with our value, the reported value and the ratio. | `tables/validation_ladder_summary.csv` |
| **Table S3** | The five registered source contradictions (C1-C5), each recorded with the handling adopted rather than resolved. | `tables/F_source_contradictions.csv` |
| **Table S4** | Step-out scaling curves with local log-log slopes, and the thirteen matched-moment configurations. | `tables/F_two_path_stepout_scaling.csv, tables/F_two_path_matched_moment.csv` |
| **Table S6** | Per-case published evidence behind table 2 of the manuscript | `tables/F_table_S6_case_evidence.csv` |
| **Table S5** | Screened archive inventory with tier assignment for all 1262 accessions. | `tables/F_GEO_scan_inventory.csv` |

## Figures

| | Title | File |
|---|---|---|
| Figure 1 | The published record is asymmetric and contradicts itself in five places | `figures/Figure1.png` |
| Figure 2 | The two routes do not start from the same kind of particle | `figures/Figure2.png` |
| Figure 3 | The central result: the two placements have step-out exponents of opposite sign | `figures/Figure3.png` |
| Figure 4 | The membrane arm, treated analytically | `figures/Figure4.png` |
| Figure 5 | Matched-moment comparison | `figures/Figure5.png` |
| Figure 6 | Validation ladder: four published measurements, no fitted parameters | `figures/Figure6.png` |
| Figure 7 | The dose plane, computed twice | `figures/Figure7.png` |
| Figure 8 | The axis this paper is built on has never been archived | `figures/Figure8.png` |
| Figure 9 | What a proper design recovers, in systems that have run one | `figures/Figure9.png` |

## Scripts

| File | What it computes |
|---|---|
| `scripts/F_physics_arms.py` | magnetic moments, blocking diameter, rod and whole-cell drag, step-out scaling, cap tension, dose plane |
| `scripts/berret2016_level2.py`, `scripts/berret2016_level2_fig.py` | validation level 2, intracellular step-out of superparamagnetic wires |
| `scripts/li2026_level3_shear.py` | validation level 3, intralysosomal wall shear stress |
| `scripts/cox2016_level4_tension.py` | validation level 4, PIEZO1 gating tension |
| `scripts/validation_ladder_merge.py` | merges all four validation levels into one table |
| `scripts/F_geo_scan.py` | archive screen of GEO and ArrayExpress under pre-registered inclusion rules |
| `scripts/F_bulk_arms.py`, `scripts/F_sc_GSE276125.py`, `scripts/F_simple_effects.py` | re-analysis of public transcriptomes |

`F_geo_scan.py` expects an NCBI E-utilities API key in the `NCBIEUTILITIESAPIKEY` environment variable.
The transcriptome scripts download their matrices from the GEO FTP site; the raw matrices are not
redistributed here.

## Reproducing

```
pip install numpy scipy pandas matplotlib statsmodels pydeseq2 gseapy
python scripts/F_physics_arms.py
```

## Citation

Xia Y, Yin L, Lin A, Tang J, Cong Z. *Where the magnet sits sets the sign of the scaling law:
membrane-anchored versus intracellular magnetic actuation of cells.* Preprint.

Cite a fixed state of this repository by its release tag rather than the moving `main` branch.

## Licence

Code and tables are released under the MIT licence (`LICENSE`).
