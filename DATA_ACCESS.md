# Source data access

The biological source is the AstraZeneca–Sanger Drug Combination DREAM Challenge dataset, described by [Menden et al. (2019)](https://doi.org/10.1038/s41467-019-09799-2). Obtain original measurements through the study's data-access route and follow the provider's conditions.

This repository includes the processed coefficients and frozen problem definitions used for the computational comparisons. The original source tables are not redistributed. To reconstruct panel selection from source measurements, obtain `OI_combinations_synergy_scores_final.txt` and `Drug_info_release.csv`, place them in `evidence/revision_source/`, and run `python prepare_revision.py`. Their recorded hashes are in `evidence/revision_design.json`. Saved designs support all other analysis and simulation commands without these source tables.
