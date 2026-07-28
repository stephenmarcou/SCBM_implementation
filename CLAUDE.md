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
- `sparse_atom.ipynb` — supervised sparse-dictionary alternative to the task-head
  readout: `SupervisedSparseAtoms` learns a shared residual dictionary `V` (r_dim×m,
  unit-norm columns) plus per-task weights `B` (atom→task, L1-penalized) and `W_c`
  (concept→task, unpenalized), trained end-to-end on multilabel BCE plus an
  off-diagonal decorrelation penalty on atom activations `z = Vᵀr`. Atom↔hidden-concept
  alignment scored via sign-invariant AUC + Hungarian matching (train-set matching
  only, to avoid double-dipping on test-set AUCs). Also has a **Multi-experiment comparison** section
  (`load_experiment` / `run_full_experiment`) that reruns probes + `c_mu` predictive
  power + sparse-atom fit + test metrics across a list of saved run folders side by
  side (used for the β sweep in §8).
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


- **β sweep at σ_x=0.01, ρ_cr=0** (`sparse_atom.ipynb` multi-experiment comparison,
  2026-07-27; β ∈ {0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0}, 7 runs): increasing β degrades
  **unsupervised** sparse-atom discovery of hidden concepts (mean matched atom AUC
  0.946→0.759) much more steeply than the **supervised** distributed probe on raw
  `res_mu` (task-bit AUC 0.948→0.872, a ~0.08 drop; non-task-bit concepts stay flat at
  chance ~0.58 throughout). Confirms unsupervised separability is more
  background-sensitive than raw recoverability, consistent with the K+1-effective-dim
  finding (β grows the shared background mode that competes with the K genuine
  directions for the sparse code's capacity). Also reproduces the ΔAUC-gate
  miscalibration from `layer2-rho-cr-first-result` **even at ρ_cr=0**: `c_mu`-only AUC
  for hidden tasks rises 0.511→0.778 as β increases (the shared observed-context term
  `a_c_n` leaks into `c_mu`), while `c_mu`-only AUC for observed tasks falls
  0.971→0.764 (the shared hidden-context term `a_r_n` lives in the residual) — so the
  gate's observed-task Δ rises from −0.001 to 0.221 purely from background mixing, not
  from ρ_cr>0.

### Sparse-atom decomposition — why separation isn't clean

`sparse_atom.ipynb`'s `SupervisedSparseAtoms` fits its residual dictionary `V` on the
**raw, standardized `res_mu`** — the concept path (`c @ W_c`) is only an additive term
in the task logits (`forward()`: `z = r @ V`, `logits = c @ W_c + z @ B + b`); `r` is
never residualized on `c_mu` before the dictionary is learned. This likely explains the
steep mean-matched-AUC decline with β (0.946→0.759, above): the shared background mode
that grows with β competes for atom capacity directly in the raw residual — the same
geometry problem that made raw ICA fail before `c_mu`-cleaning was introduced (see
`cmu-cleaning-rescues-ica` in memory). Evidence this is a geometry problem rather than
an information ceiling: the raw supervised probe on the same `res_mu` barely degrades
over the same β range (0.948→0.872) — if the hidden-concept information were simply
harder to extract at high β, both metrics would degrade together, not just the
unsupervised one. **Next experiment**: OLS-residualize `res_mu` on `c_mu` (the
validated Layer-1 cleaning step) before fitting the sparse-atom dictionary, and check
whether `mean_matched_auc` recovers toward the raw-probe ceiling the way cleaned ICA
did.

### Interpretation guardrail: readout ≠ discovery

In the current dataset each hidden task = one concept + shared background, so after
cleaning, the readout direction for task k is *guaranteed by construction* to align
with concept k — the 1:1 heatmap validates mechanics, not discovery. **The discovery
claim rests on the cleaned ICA result** (no labels used in separation; matches the
supervised bound). Roles: readout = supervised ceiling + task attribution + gate;
cleaned ICA = the actual unsupervised discovery step (and the only one that can find
task-irrelevant hidden structure).



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
  caveat in §5: match on train-set AUCs only)
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
