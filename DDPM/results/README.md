# Mechanism + design-choice ablations — DDPM (CIFAR-10, forget airplane)

Prediction-first summary; outcomes filled in from ablations.csv / mechanism.csv / kappa.csv (commands below).

- Exp1 mechanism (ablation_mechanism.py): predicted n_j anti-correlates with rho_j and cost(j) (Spearman+log-log Pearson in mechanism.csv, tables/mechanism_correlations_ddpm.tex). Held: TBD. If n_j is flat, the method is plain shrinkage — that negative result replaces the mechanism story.
- Exp1 erasure location: reported as top-8 vs bottom-8 share of ||delta_f||_F^2 in kappa.csv. Held: TBD.
- Exp2 kappa (real retain activations): median/90/95/99 of kappa = max_j|z_j|/w_j in kappa.csv; fraction of points where kappa*sqrt(2T) bounds ||delta_f @ z|| must be 1.0 (reported as measured). Held: TBD.
- Exp3 uniform margin: predicted TA/FID degrade vs data-adaptive bounds (tables/exp3_ddpm.tex). Held: TBD; if no degradation, describe the method as isotropic shrinkage on the forget subspace.
- Exp4 centre vs width: predicted width_only raises UA at comparable TA/FID (tables/exp4_ddpm.tex). Held: TBD.
- Exp5 protected region family: env_box/two_random/slabs_2k vs two_corner (tables/exp5_ddpm.tex; env_box = exact complement in 2 terms). Held: TBD.
- Exp6 sign flip: fraction {0,.25,.5} of U_forget rows flipped before recomputing bounds; spread in UA/TA/FID reported (tables/exp6_ddpm.tex). Held: TBD.
- Exp7 percentile alpha {1,5,10}: control-region size vs margin width; read UA/TA/FID against median kappa (tables/exp7_ddpm.tex + kappa.csv). Held: TBD.
- Exp8 delta_b in interval legs: with/without (tables/exp8_ddpm.tex); note whether best lambda shifts (delta_b then appears in L_mean and the interval terms). Held: TBD.
- Exp9 L_res isolation: component attribution of the current Table 10 (tables/exp9_ddpm.tex, lambda fixed 5). Held: TBD.
- Normalisation: every variant's interval part is divided by its own squared-term count (env_box 2, two_corner/two_random 4, width/centre 2, slabs_2k 4k); legend in ablation_common.py.
- Reproducing tables: python ablation_regions.py --summarize --results_dir /shared/results/common/miksa/intact/DDPM/r (aggregates runs/*/rows.json first).
- Reproducing Exp1-2: python ablation_mechanism.py --results_dir <r> --backbone ddpm (drop --no_remain to collect kappa stats).
- Grid: 180 jobs, expgrid.py "ddpm", scripts/slurm_ablation_experiments.sh (rtx4090_batch, qos=batch).