# Mechanism + design-choice ablations — ResNet-18 (CIFAR-10)

Same protocol as DDPM/results/README.md but backbone B: final-FC target, k=32,
SGD lr=1e-3, 10 epochs, RL objective, classwise (forget class 0) and random-10%
settings.  Fixed lambda: 10 (classwise) / 1 (random); sweep 0.5..25.

- Exp3 uniform margin: predicted RA/TA/MIA degrade; Held: TBD (exp3_resnet18.tex).
- Exp4 centre vs width: predicted UA gain from width_only; Held: TBD (exp4_resnet18.tex).
- Exp5 region family incl. env_box; Held: TBD (exp5_resnet18.tex).
- Exp6 sign flip {0,.25,.5}; Held: TBD (exp6_resnet18.tex).
- Exp7 alpha {1,5,10} vs median kappa; Held: TBD (exp7_resnet18.tex + kappa.csv).
- Exp8 delta_b in interval legs; Held: TBD (exp8_resnet18.tex).
- Exp9 L_res isolation; Held: TBD (exp9_resnet18.tex).
- Metrics: UA=100-forget-acc, RA=retain-acc, TA=test-acc, MIA=SVC confidence.
- Tables: python ablation_cls.py --summarize --results_dir /shared/results/common/miksa/intact/Cls/r
- Exp1-2: python ablation_mechanism.py --results_dir <r> --backbone resnet18
- Grids: 180 jobs each, expgrid.py "resnet18-classwise"/"resnet18-random",
  scripts/slurm_ablation_cls_{classwise,random}.sh (dgx, qos=big).