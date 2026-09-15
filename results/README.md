# Results included in the repository

`paper_tables/` contains compact CSV versions of the eight quantitative tables used in the revised manuscript. `source_data/` contains the compact raw CSV/JSON files from which those tables were assembled. Large checkpoints, dense reference arrays, and training histories are intentionally omitted from the public release.

| Manuscript table | File | Source experiment |
|---|---|---|
| Table 1 | `table1_fou_exact_reference.csv` | 1D fOU exact-reference validation |
| Table 2 | `table2_boundary_padding.csv` | exterior-padding diagnostic |
| Table 3 | `table3_score_ablation.csv` | matched-update and matched-GL ablations |
| Table 4 | `table4_periodic_2d.csv` | 2D periodic SBPINN/FEM comparison |
| Table 5 | `table5_dirichlet_2d.csv` | 2D killed-Lévy SBPINN/FEM comparison |
| Table 6 | `table6_score_fpinn_comparison.csv` | SBPINN vs Score-fPINN, three matched seeds |
| Table 7 | `table7_bistable.csv` | bistable expert reconstruction |
| Table 8 | `table8_multiplicative_noise.csv` | transformed-u multiplicative noise, three seeds |

Representative PNG figures are collected under `figures/` for convenient inspection. They are not required to run the code.
