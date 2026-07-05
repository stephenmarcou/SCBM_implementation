# CLAUDE.md

Context file for Claude Code / Claude working in this repository. This project extends
**Stochastic Concept Bottleneck Models (SCBMs)** with a residual channel, with the core
research goal of making that residual channel **interpretable** — i.e. recovering
task-relevant hidden concepts from it in an **unsupervised** manner.

The foundational reference is the SCBM paper (Vandenhirtz, Laguna, Marcinkevičs, Vogt,
NeurIPS 2024) — see `SCBM.pdf` in the project if present. A synthetic setting with known
ground-truth hidden concepts is used to validate discovery methods before eventual
application to real data (where ground truth is unavailable).

---

## 1. Core Research Goal

Given a Residual SCBM whose residual channel `η_r | x ~ N(μ_r(x), Σ_r(x))` may encode
one or more hidden, task-relevant concepts, **discover those hidden concepts from the
residual activations alone**, without supervision, and validate the recovery against
known ground truth in the synthetic setting. The end goal is a discovery pipeline that
generalizes to real datasets with complex (non-synthetic) dependency structure — not one
that relies on synthetic-only guarantees (e.g. oracle access to generative factors).

---

## 2. Model Background (from SCBM + residual extensions)

- **SCBM**: models concept logits `η | x ~ N(μ(x), Σ(x))` instead of assuming
  independent concepts. Enables correlated multi-concept interventions and a
  confidence-region-based intervention strategy. Trained end-to-end via
  reparameterization + Monte Carlo sampling of `η`, with Graphical-Lasso-style
  regularization on the precision matrix `Σ(x)⁻¹`.
- **Naive Residual SCBM**: augments the concept logit vector with `m` extra latent
  "residual" dimensions `r`, jointly modeled with concepts under the same multivariate
  Gaussian. No explicit regularization on residual behavior.
- **Residual SCBM + `L_int`**: adds an intervention loss (from "In Defense of
  Information Leakage in Concept-based Models") that encourages *benign leakage* —
  sufficiency (residual carries label-relevant info not in concepts) and localization
  (concept-relevant info stays in its own concept, not diffused into the residual).
  Two variants: basic (interventions don't propagate to residuals during training) and
  **extended** (interventions do propagate to residuals, since concepts and residuals
  are jointly Gaussian). Extended `L_int` gives materially better full-intervention task
  accuracy (~95% vs ~80-72% for plain Res-SCBM) while keeping concept AUROC/accuracy
  comparable to standard SCBM.
- Current active work sits on top of this: the residual channel is architecturally
  capable of encoding hidden concepts, and the question is how to *find* them.

---

## 3. Synthetic Multilabel Dataset (current primary testbed)

Generative process (see `multilabel_synthetic_dataset.py`):

1. Low-rank base covariance `Σ_base = WWᵀ`, `W ~ N(0, I)`, rank `L` (if `L=0`, hidden
   vars are approximately independent).
2. Final covariance `Σ` built by scaling blocks: `ρcc` (observed-observed), `ρrr`
   (hidden-hidden), `ρcr` (observed-hidden cross block).
3. Latent vector `η_n = [η_c,n, η_r,n] ~ N(0, Σ)`.
4. Binary observed/hidden concepts via thresholding at 0.
5. Continuous signal strengths `c_signal = |η_c| ⊙ c`, `r_signal = |η_r| ⊙ r`.
6. Input `x_n = f(z_signal_n) + ε_n` via a fixed random MLP `f`.
7. Sparse task weights `w_task_c`, `w_task_r` with a `min_weight_ratio` floor and random
   signs.
8. Top-K hidden concepts (`hid_task_idx`) and top-J observed concepts (`obs_task_idx`)
   selected by weight magnitude.
9. Shared context scores `a_c_n` (observed→hidden) and `a_r_n` (hidden→observed),
   standardized.
10. K hidden subtasks: `s_hid_{n,k} = α·u_hid_{n,k} + β·a_c_n`, rank-based binary label.
11. J observed subtasks: `s_obs_{n,j} = α·u_obs_{n,j} + β·a_r_n`, rank-based binary label.
12. Final multilabel target `y_n = [y_hid_1..K, y_obs_1..J]`.

**Current default configuration**: K=5 hidden tasks, J=2 observed tasks, 20 hidden
concept dimensions, 10 observed concept dimensions, 20 residual dimensions, with
controllable `σ_x` (input noise) and `ρ` (correlation) parameters.

`β` controls how much of the shared cross-channel background term contaminates each
subtask score. **`β` (not `α`) multiplies the shared term** — `shared_obs_score` for
hidden tasks, `shared_hid_score` for observed tasks. (An earlier note had this
inverted — disregard any older material associating background ablation with
`alpha=0`.)

**α and β are one knob, not two.** Labels come from a rank-based median split, which is
invariant to positive rescaling of the score — so only the ratio β/α matters, and with
`standardize=True` (both mix terms unit-variance) the background share of label variance
is exactly β²/(α²+β²) (β=1,α=1 → 50%). Sweep β/α, not both. `α=0` is degenerate: all K
hidden subtask scores collapse to the same `β·shared_obs_score`, so all hidden labels
become identical.

**⚠️ `latent_rank=0` silently disables all ρ parameters.** `_make_covariance` returns
`1.1·I` immediately when `latent_rank == 0` — `rho_cr`/`rho_cc`/`rho_rr` are ignored.
Any correlation experiment must set `data.latent_rank > 0` (e.g. 10), and should report
the *realized* cross-correlation (the nominal ρ is distorted by the rank-L base matrix
and the positive-definiteness repairs).

---

## 4. Terminology / Glossary

| Term | Meaning |
|---|---|
| `res_mu` | Predicted mean of residual logits, `μ_r(x)` |
| `c_mu` | Predicted mean of concept logits, `μ_c(x)` |
| `hid_task_idx` | Indices of the top-K hidden concepts selected as task-relevant |
| `obs_task_idx` | Indices of the top-J observed concepts selected as task-relevant |
| `s_hid` / `s_obs` | Per-subtask scores (hidden / observed) before thresholding |
| `num_hid_tasks` (K) / `num_obs_tasks` (J) | Number of hidden / observed subtasks |
| `L_int` | Intervention loss enforcing benign leakage (sufficiency + localization) |
| distributed probe AUC | AUC of linear probes trained per-dimension/distributed over residual axes to detect task-relevant hidden concepts |
| effective rank | Rank of the residual channel's covariance/SVD spectrum — used to check for representational collapse |
| Hamming accuracy | Per-label accuracy averaged across all multilabel targets |
| exact match accuracy | Fraction of samples where *all* labels are predicted correctly |
| `a_c_n` | Shared observed-context background term contaminating hidden subtasks |
| `ρ_cr` | Cross-covariance strength between observed and hidden latents |

---

## 5. Codebase Components

- `multilabel_synthetic_dataset.py` — generative process described in §3
- `multilabel_analysis.ipynb` — primary analysis notebook (ICA, probes, PCA/SVD,
  covariance diagnostics; since 2026-07-05 also the **task-head readout + ΔAUC gate**
  section — the primary discovery pipeline, inserted after the cleaned-residual ICA)
- `eval_datasets.py`
- `SCBLoss` / `SCBresLoss` — concept and residual loss implementations
- `FCNNEncoder`
- `validate_one_epoch_scbm_residual` — validation loop; currently returns an **average**
  covariance matrix (see known issue in §6)
- `MultilabelMetrics`

Config management via **Hydra**; jobs submitted to a cluster via **`sbatch`**.
Experiment configs live at `/cluster/home/smarcou/work/experiments_scbm/`.

Key saved artifacts: `avg_covariance_matrix.pt`, `c_res_mu.pt`.

---

## 6. Running Training Jobs

Entry point is `train.py`, config via **Hydra** (`configs/config.yaml` + `configs/model/*.yaml`
+ `configs/data/*.yaml`, composed with `+model=...` / `+data=...`). Any leaf value can be
overridden on the command line with dotted keys.

Example (synthetic multilabel, Residual SCBM):

```bash
python train.py +model=SCBM_RES +data=multilabel_synthetic \
  model.j_epochs=200 model.encoder_arch=FCNN \
  train_only=True save_model=True \
  data.save_data=True data.save_concept_and_residual_channel=True \
  data.obs_dim=10 data.hid_dim=20 data.num_concepts=10 data.num_residuals=20 \
  data.num_hid_tasks=5 data.num_obs_tasks=2 data.num_classes=7 \
  data.task_sparsity_obs=0.3 data.task_sparsity_hid=0.25 \
  data.rho_cr=0.0 data.rho_cc=0.0 data.rho_rr=0.0 \
  data.alpha=1.0 data.beta=1.0 data.sigma_x=0.5 data.latent_rank=0 \
  model.multilabel_task=True data.standardize=True
```

Key flags:
- `+model=SCBM_RES` — Residual SCBM (`configs/model/SCBM_RES.yaml`, on top of
  `model_defaults.yaml`). Other options: `SCBM`, `CBM`, `CEM`, `AR`, `target_head`.
- `+data=multilabel_synthetic` — synthetic multilabel dataset (§3), config at
  `configs/data/multilabel_synthetic.yaml`. Note `num_hid_tasks` (K) must be
  `<= floor(task_sparsity_hid * hid_dim)` and `num_obs_tasks` (J) must be
  `<= floor(task_sparsity_obs * obs_dim)`; `num_classes` should equal `K + J`.
  `num_concepts`/`num_residuals` should match `obs_dim`/`hid_dim`.
- `train_only=True` — skip the post-training intervention sweep (`intervene(...)` in
  `train.py`), just train + validate + test. Much faster for iterating.
- `save_model=True` — required to persist anything to disk (checkpoints, logs, saved
  data/covariance/concept-residual artifacts). Without it the run is ephemeral
  (still logs to wandb unless `logging.mode=disabled`, which is the config default).
- `data.save_data=True` — save the generated synthetic dataset itself (so it can be
  reloaded later via `data.data_dir_name=<name>` instead of regenerating).
- `data.save_concept_and_residual_channel=True` — save `c_mu`/`res_mu` (and related
  artifacts, e.g. covariance) for post-hoc notebook analysis.

Output location (only when `save_model=True`): `create_experiment_path()` in `train.py`
builds `experiment_dir / model.model / data.dataset / <auto-named-run>`. For
`multilabel_synthetic`, the run name auto-encodes the key hyperparameters (K, J, alpha,
beta, the three rho's, min_weight_ratio, sigma_x, standardize, num_residuals) plus a
timestamp + short UUID, e.g.:
`experiments/scbm_residual/multilabel_synthetic/K_5_J_2_alpha_1.0_beta_1.0_..._<timestamp>_<uuid>/`.
On the group cluster (`update_config_paths()`, hostname contains `"biomed"`),
`experiment_dir` is forced to `/cluster/home/smarcou/work/experiments_scbm/` regardless of
the config default (`./experiments/`) — so local vs. cluster runs land in different places.
Jobs are submitted to the cluster via `sbatch scripts/train.sh <hydra overrides...>`
(`scripts/hyperparameter_search.sh` for sweeps).

Saved artifacts per run typically include `log.txt`, `model.pth` / `model_best.pth`, and
(if requested) `avg_covariance_matrix.pt`, `c_res_mu.pt`, and the saved dataset itself.

**Local (Mac) training notes** (2026-07-05):
- `train.py` **NaNs on Apple MPS** (`intermediate became non-finite`, models.py
  `check_finite`) with configs that train fine on the cluster — force CPU by patching
  `torch.backends.mps.is_available = lambda: False` before running (wrapper approach).
  CPU training is fast enough: ~13 s/epoch → ~45 min per 200-epoch run.
- Local env is `venv_cluster/` (python 3.11); `plotly` and `pytorch-minimize` were
  missing and have been pip-installed into it.
- The run folder name does **not** encode the seed — read `seed` from `log.txt` or the
  dataset's `info.txt` when aggregating multi-seed results.
- Duplicate-run gotcha: two run folders can contain byte-identical artifacts (e.g. the
  two β=1/σ_x=0.5 seed-0 runs `..._635bc` and `..._c7cfa` — same model re-dumped).
  Dedupe by (β, σ_x, seed) before averaging "across runs".

---

## 7. Inference, Interventions & Hyperparameter Search

### Post-hoc inference / interventions on a saved model (`inference.py`)

`inference.py` reloads a trained checkpoint and re-runs evaluation and/or the
intervention sweep, using the same Hydra config system as `train.py`. The run must have
been trained with `save_model=True` (it loads `<experiment_path>/model.pth`).

```bash
python inference.py +model=SCBM_RES +data=multilabel_synthetic \
  inference.ex_name=<run_folder_name> \
  run_inference=True run_interventions=True \
  <same model./data. overrides as training>
```

Key points:
- `inference.ex_name` — the run folder name under
  `experiment_dir/<model>/<dataset>/` (the auto-generated `K_..._<timestamp>_<uuid>`
  name from §6). Required; the script errors if the path doesn't exist.
- `run_inference=True` — test-set evaluation (writes `inference_log.txt`). For
  `scbm_residual` with `data.save_concept_and_residual_channel=True`, it additionally
  re-runs the val and train sets through shuffle-free "analysis loaders"
  (`make_analysis_loader`) to dump `c_mu`/`res_mu` artifacts into `test/`, `val/`,
  `train/` subfolders of the run directory — this is how notebook analysis inputs are
  (re)generated from an existing checkpoint.
- `run_interventions=True` — intervention curves (writes `intervention_log.txt`). For
  `scbm_residual` this currently uses `intervene_scbm_residual_optimized` (the
  non-optimized `intervene_scbm_residual` is commented out).
- **Config must match training**: model/data overrides are NOT auto-recovered from the
  checkpoint for the multilabel synthetic dataset — pass the same `model.*`/`data.*`
  overrides used at training time (dims, K/J, `model.multilabel_task=True`, etc.), and
  point `data.data_dir_name` at the saved dataset so the same data split is reused
  rather than regenerated. (Auto-recovery from `log.txt` exists only for the
  `incomplete=True` CUB path and `synthetic_res_scbm` via
  `update_pkl_dir_and_num_concepts`.)

### Hyperparameter search

- `hyperparameter_search=True` (in `train.py`) evaluates on the validation set at end
  of training and returns val metrics without inference/interventions; results are
  nested under `experiment_dir/hyperparameter_search/...`.
- `scripts/hyperparameter_search.sh` is an sbatch script looping `train.py` over
  `L_int_loss_weight` values (currently CUB-specific). **Known bug**: the loop is
  written `for weight in 1,5,10,50` (commas, not spaces), so bash iterates once with
  the literal string `1,5,10,50` instead of four separate runs.

---

## 8. Current Focus

Building a **generalizable concept discovery pipeline** for the residual channel.
Status 2026-07-05: **Layer 1 is validated** (see below); the open fronts are the
`ρ_cr > 0` stress test (Layer 2) and a multi-concept-tasks dataset variant that makes
the discovery claim non-trivial.

### Established findings (validated across all 10 local runs, 2026-07-05)

- The residual channel collapses to **~K+1 effective dimensions** (K concepts + 1
  background direction; confirmed via SVD).
- Distributed linear probes recover task-relevant hidden concepts selectively (non-task
  concepts near chance ~0.57). Probe ceiling depends on σ_x: ~0.95 (β=0, σ=0.01),
  ~0.87 (β=1, σ=0.01), ~0.75 (σ=0.5, any β). MLP(x→hidden) info ceiling at σ_x=0.5 is
  only ~0.81, so ~0.75 is near the encoder ceiling.
- **σ_x is confirmed the stronger lever than β**: it both lowers the information
  ceiling and Gaussianizes `μ_r(x)` (posterior-mean shrinkage), which selectively
  destroys FastICA's non-Gaussianity requirement while probes degrade gracefully.
- **The raw-ICA failure at σ_x=0.5 is NOT background-driven**: it occurs at *all* β
  values including β=0 (concept 14 stuck at AUC ~0.54), and oracle removal of the
  scalar `a_c_n` does **not** fix it. The blocker is **observed-concept leakage** into
  the residual channel (extra nuisance sources for ICA).
- **`c_mu` residualization (Layer 1) is validated and strictly better than the
  oracle**: OLS-regressing `res_mu` on the 10-dim `c_mu` removes background *and*
  leakage at once. Cleaned ICA achieves **5/5 discovery in every condition** (mean
  matched test AUC 0.745 at β=1/σ=0.5 vs probe ceiling 0.751). This supersedes the
  earlier "oracle-based confound removal validated" note.
- **Task-head readout** (per hidden task, logistic probe from the `c_mu`-cleaned
  residual to `y_k`; one *labeled* direction per task, no Hungarian matching): 5/5 at
  the probe ceiling in every condition (0.749 at β=1/σ=0.5; 0.819 at β=1/σ=0.01;
  0.948 at β=0/σ=0.01). Raw (uncleaned) readout fails at β=1/σ=0.5 (0/5, 0.660).
- **ΔAUC gate**: AUC of `y_k` from `[c_mu, res_mu]` minus `c_mu`-only cleanly flags
  residual-driven tasks at β=0 (hidden Δ≈+0.19..+0.45, observed Δ≈0). Caveat: at β=1
  observed tasks also gain (~+0.15) because `y_obs` contains the hidden background
  `a_r` — the gate detects *residual-dependent*, not *hidden-concept* per se.
- **Raw-ICA discovery counts are run-to-run unstable**: two independent trainings of
  the identical β=1/σ=0.01 config flip between 3/5 and 4/5. Report mean matched AUC
  (and multi-seed error bars), not single-run discovery counts.
- Standard **SAEs remain inappropriate** — no superposition with 20 residual dims
  encoding ~5 concepts.

### Interpretation guardrail: readout ≠ discovery

In the current dataset each hidden task = one concept + shared background, so after
cleaning, the readout direction for task k is *guaranteed by construction* to align
with concept k — the 1:1 heatmap validates mechanics, not discovery. **The discovery
claim rests on the cleaned ICA result** (no labels used in separation; matches the
supervised bound). Roles: readout = supervised ceiling + task attribution + gate;
cleaned ICA = the actual unsupervised discovery step (and the only one that can find
task-irrelevant hidden structure).

### Two-layer generalizable design — status

- **Layer 1 (validated)**: OLS-regress `res_mu` on `c_mu`; removes ~50% of `res_mu`
  variance at β=1/σ=0.5 with no loss of hidden-concept information (probe AUCs
  unchanged). Safe **only because ρ_cr = 0** — nothing predictable from `c_mu` can
  contain hidden signal.
- **Layer 2 (open)**: per-concept gate (R² of hidden signal explained by `c_mu`) to
  flag when removal is unsafe (`ρ_cr > 0`). Candidate softer policies to compare in the
  ρ_cr sweep: partial cleaning (remove only top-k principal directions of the
  `c_mu`-predicted component; background is ~rank-1), capped INLP, shrinkage.

### Next experiments (priority order)

1. **Multi-concept tasks variant** — each hidden task depends on 2–3 overlapping hidden
   concepts (more task-relevant concepts than tasks). Breaks the 1:1 task↔concept
   construction; the cleaned-ICA heatmap is the verdict plot (readout is *expected* to
   smear). Either outcome is a result: ICA separates → discovery claim survives; ICA
   smears → motivates auxiliary-variable nonlinear ICA (iVAE / Hyvärinen-style with `y`
   or `c_mu` as auxiliary).
2. **ρ_cr sweep** (Layer-2 validation): ρ_cr ∈ {0, 0.25, 0.5, 0.75} × β ∈ {0, 1} at
   σ_x=0.01, **`data.latent_rank=10`** (see §3 trap), ρ_cc=ρ_rr=0, 2–3 seeds. Measure:
   damage curve (per-concept probe AUC raw vs cleaned), gate calibration (R² vs
   damage), cleaning policies (none / full OLS / top-k partial / capped INLP), and the
   conditional baseline (hidden-concept AUC from `c_mu` alone — credit the residual
   only with the increment).
3. **Multi-seed replication** of the β × σ_x grid (seeds 1–3; seed 0 exists). An
   18-run local attempt on 2026-07-05 was stopped; submit via sbatch instead. Seed-1
   datasets are already generated under `datasets/multilabel_synthetic/local_..._seed_1/`.

**Critical design constraint** (unchanged): the safety guarantee (`ρ_cr = 0`) is
specific to the synthetic dataset and will **not** generalize to real-world structure.
The priority is a pipeline that works when that guarantee doesn't hold.

### Active covariance evaluation work

`validate_one_epoch_scbm_residual` currently returns an **average** covariance matrix,
which risks sign cancellation and conflates `E_x[Σ(x)]` with `Cov_x[μ(x)]`. The full
decomposition is needed:

```
Cov(η) = E_x[Σ(x)] + Cov_x[μ(x)]
```

Highest-priority diagnostics: elementwise cancellation checks, and stratified averaging
conditioned on hidden task labels.



---

## 9. Key Principles (methodological)

- **PCA variance concentration is a poor proxy for ICA recoverability** — these can
  point in opposite directions (β=0.5 has *lower* top-5 variance concentration than
  β=1, but *better* ICA recovery). Don't use PCA variance concentration as the headline
  diagnostic for discoverability.
- **Hungarian assignment for concept-to-axis matching must be computed on
  training-set AUCs only**, then applied to held-out test data. Using test-set AUCs for
  both matching and scoring is circular analysis (double-dipping).
- **The shared background problem arises from task-score aggregation downstream, not
  from generative correlations** — setting ρ values to 0 does not eliminate it.
- **SNR asymmetry**: observed tasks underperform hidden tasks because the shared
  background term sums over more weights for hidden tasks than observed tasks, making
  per-concept signal harder to detect for observed tasks.
- Axis 6 mixed loading at β=1 is a **representational rotation artifact**, not genuine
  source entanglement (`a_c_n` is orthogonal to hidden concept signals).
- **Cleaning unmasks, it doesn't retrieve**: the hidden-concept information is already
  in `res_mu` (probes on the raw residual hit the ceiling); `c_mu`-cleaning fixes the
  *geometry* so direction-based methods (ICA, readout) can isolate concepts. Cleaning
  never lifts recovery above the probe ceiling set by σ_x.
- **Report mean matched AUC, not discovery counts**, as the headline metric — discovery
  counts (>0.7 threshold) flip between identical re-trainings of the same config.
- **Verify configs from the run's own `log.txt` / dataset `info.txt`**, never from
  folder names, when comparing across experiments (and dedupe byte-identical run
  folders by (β, σ_x, seed)).

---

## 10. Approach & Patterns

- Iterative experimental design: sweep parameters (β, σ_x) → analyze in Jupyter →
  identify failure modes → fix → re-run.
- **Unsupervised discovery → supervised evaluation separation**: fit ICA on activations
  only; use ground-truth hidden concepts strictly for post-hoc validation.
- **Oracle-first, then generalize**: establish what's achievable with ground-truth
  access before designing methods that don't require it.
- Notebook hygiene: watch for stale-cache bugs (e.g. duplicate/uncommented
  `EXPERIMENT_PATH` assignments silently overwriting results) — check cell execution
  order carefully.

---

## 11. Tools & Libraries

- FastICA (concept discovery), PCA/SVD, Lasso regression, distributed linear probes,
  per-task PC correlation heatmaps, raw axis AUC heatmaps
- `scipy.stats.rankdata`, `sklearn.metrics.roc_auc_score` (macro averaging), Hungarian
  assignment for concept-axis matching
- Hydra for config management, `sbatch` for cluster job submission

---

## 12. Relevant Literature

- Locatello et al., ICML 2019 — identifiability without inductive biases
- Eastwood & Williams, ICLR 2018 — DCI framework
- Yeh et al., NeurIPS 2020 — ConceptSHAP
- Ghorbani et al., NeurIPS 2019 — ACE
- Khemakhem et al., 2020 — iVAE
- Kim & Mnih — FactorVAE
- Abid et al. — contrastive PCA
- Chen et al. — Concept Whitening
- Kriegeskorte et al., 2009 — circular analysis (relevant to the Hungarian-matching
  caveat in §9)
- Ravfogel et al., ACL 2020 — INLP (iterative nullspace projection; principled,
  rank-controlled version of Layer-1 cleaning)
- Elazar et al., TACL 2021 — amnesic probing
- Hewitt et al., EMNLP 2021 — conditional probing (formal framing of the ΔAUC gate:
  usable information beyond a baseline representation)
- Kim et al., ICML 2018 — TCAV (linear probe directions as concept vectors; task-head
  readout is the same object derived from task labels)
- Hyvärinen & Morioka 2016/2017; Hyvärinen, Sasaki, Turner 2019 — nonlinear ICA with
  auxiliary variables (identifiability rationale for why the y-supervised readout works
  where vanilla ICA loses identifiability under noise/Gaussianization)
