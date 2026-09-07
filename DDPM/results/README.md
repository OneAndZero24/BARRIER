# Region-construction ablation — results

Summary (10 lines)
1. `compute_protection_loss` now supports five region constructions of the
   same two-corner primitive `T(l,u)=||dWp@l-dWn@u||^2+||dWp@u-dWn@l||^2`:
   `two_corner` (baseline, 2 boxes), `two_random` (placement control, same box
   shapes with a fixed-seed random side pattern stored in `pca_info`),
   `boxes_4` / `boxes_8` (4/8 predefined boxes: coordinates split into 2/4
   consecutive groups, one box per group per side), and `slabs_2k` (2k
   coordinate slabs whose union is exactly the complement of the forget box
   inside the envelope). `region_mode` is a constructor arg; default
   `two_corner` + `normalize_region=False` reproduces the previous loss
   bitwise (regression test).
2. The m-group family (2m boxes) interpolates between `two_corner` (m=1) and
   `slabs_2k` (m=k); boxes_4 = m=2, boxes_8 = m=4, grouping predefined
   (consecutive), so results are deterministic across runs. Every variant
   normalises its interval part by its number of squared terms so the sweep
   compares penalised-scale, not raw-lambda: A/B divide by 4, boxes_4 by 8,
   boxes_8 by 16, C by `4k` (=128 for k=32; 2 terms per T, 2k slabs — see
   note 6).
3. Grid: 5 variants x lambda in {0.5, 1, 2, 5, 10, 25} x 3 seeds = 90 runs,
   each at the paper's DDPM config (airplane forget, QKV+cemb targets, k=32,
   Adam lr=1e-4, 3000 RL steps, alpha=0 remain loss). One row per run in
   `regions.csv`, per-layer diagnostics in `diagnostics.csv`.
4. Surprise so far: none (table not yet populated — jobs submitted via
   `scripts/slurm_ablation_regions.sh`). Diagnostics are logged per run, so the
   first completed runs should be inspected for `env_asym_ratio` and
   `aniso_*` before reading anything into the headline numbers.
5. Timing is kept separate from quality metrics: setup/preprocessing wall time
   (`setup_wall_s`), a CUDA-synced microbenchmark of `compute_protection_loss`
   (20 warmup + 200 timed calls, `prot_loss_ms` ± std), total training wall
   (`train_wall_s`), and `peak_mem_mb` via `max_memory_allocated`.
6. Analytic cost column uses the counts from the ablation spec (A: 4, B: 4,
   C: 8k [M,k]-matvecs/layer, where a corner pair counts as one matvec). The
   actual implementation issues 2 products per corner (A/B: 8, C: 16k raw
   products) and C adds `4k` squared terms, not `4·2k` — a factor-2 count
   error in the spec; it only rescales C's effective lambda, which the sweep
   brackets anyway.
7. `delta_b` is excluded from the interval terms in all three variants to keep
   the comparison apples-to-apples; InTAct's Eqs. 35-36 include it — that is a
   separate issue, not part of this ablation.
8. `summarize` (`--summarize`) aggregates `regions.csv` into `regions_table.tex`
   (booktabs, style of tab:hyperparameters): per variant the fixed-lambda=5 row
   and the best-lambda row (composite `1.5*UA + TA - FID/3`); cells are
   mean ± std over 3 seeds. Rows are deduplicated by (variant, lambda, seed),
   keeping the last run.
9. Tests: (i) `two_corner` reproduces the pre-ablation loss bitwise for both
   nn.Linear and nn.Conv2d (`tests/test_ablation_regions.py`); (ii) for k in
   3..6 the max |delta_f.z| over the 2k slabs equals the max over all 3^k - 1
   complement cells enumerated by brute force, confirming `slabs_2k` covers the
   complement. Also unit-tested: term counts, two_random validity/determinism,
   `box_drift_max` vs brute-force corner search, and the diagnostics helpers.
10. Regenerate the table after the grid completes:
    `python ablation_regions.py --summarize --results_dir ./results`.
    Diagnostic tags: A=two_corner, B=two_random, D=boxes_4, E=boxes_8,
    C=slabs_2k (columns `frac_remain_in_*`, `aniso_*`,
    `vstar_drift_protected_*` / `vstar_drift_envelope_*`).