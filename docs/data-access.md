# Source data access

The biological measurements come from the AstraZeneca–Sanger Drug Combination DREAM Challenge, described by [Menden et al. (2019)](https://doi.org/10.1038/s41467-019-09799-2). Obtain them through the study's data-access route and follow the provider's conditions.

Processed coefficients and frozen designs are included under [reproduction/evidence/](../reproduction/evidence/). Rebuilding the panel selection from original measurements requires:

1. Obtain `OI_combinations_synergy_scores_final.txt` and `Drug_info_release.csv`.
2. Place them in `reproduction/evidence/revision_source/`.
3. From `reproduction/`, run `python prepare_revision.py`.

Recorded source hashes are in [revision_design.json](../reproduction/evidence/revision_design.json).
