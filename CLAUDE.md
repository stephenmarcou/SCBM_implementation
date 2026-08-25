# CLAUDE.md

Context file for Claude Code / Claude working in this repository. This project extends
**Stochastic Concept Bottleneck Models (SCBMs)** with a residual channel, with the core
research goal of making that residual channel **interpretable** — i.e. recovering
task-relevant hidden concepts from it.

The foundational reference is the SCBM paper (Vandenhirtz, Laguna, Marcinkevičs, Vogt,
NeurIPS 2024) — see `SCBM.pdf` in the project if present. **CUB-200-2011 is the real-data
testbed**: by deleting a known subset of the annotated bird attributes, we create a
dataset where the "hidden concepts" are real, semantically named, and known to us —
without being available to the model.

---

## 1. Core Research Goal

Given a Residual SCBM whose residual channel `η_r | x ~ N(μ_r(x), Σ_r(x))` may encode
one or more hidden, task-relevant concepts, **characterize what those residual
dimensions encode**. On CUB this means: train on an *incomplete* attribute set, then
check whether the removed attributes (or their semantic groups, e.g. `has_wing_color`)
can be recovered from the residual channel.

CUB is the setting where the claim has to hold under real, non-synthetic dependency
structure — correlated attributes, class-conditional attribute majority voting, and no
oracle access to generative factors.

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
  capable of encoding hidden concepts, and the question is what it actually encodes.

---

## 3. CUB Dataset (primary testbed)

Config at `configs/data/CUB.yaml`; loading and preprocessing in `datasets/CUB_dataset.py`.

- **200 classes**, **112 binary concepts** (the CBM-paper attribute subset,
  `ATTRIBUTES_IDX_USED`, selected from the original 312), default **30 residual
  dimensions** (`data.num_residuals`).
- Data lives under `<data_path>/CUB/`:
  - `class_attr_data_10/{train,val,test}.pkl` — the complete-attribute split
    (pre-determined split so photographers don't overlap between train and test).
  - `CUB_200_2011/CUB_200_2011/images/...` — images; `img_path` in the pkls is rewritten
    to an absolute path at load time by `train_test_split_CUB`.
  - `CUB_200_2011/attributes.txt` — attribute names (`has_wing_color::blue` etc.), used
    to map attributes to semantic groups.
  - `CUB_200_2011/concept_names.txt` — used by `get_concept_groups` in `utils/data.py`
    for oracle concept grouping.
  - `incomplete_data/<pkl_file_dir>/` — generated incomplete variants (see below).
- **Transforms** (`get_CUB_transforms`, following the CBM paper): train =
  ColorJitter + RandomResizedCrop(299) + Resize(224) + HFlip + ImageNet normalization;
  test = CenterCrop(299) + Resize(224) + normalization.
- `CUB_DatasetGenerator` caches decoded images in memory (`cache=True` for train/val,
  `False` for test), with attributes bit-packed via `_pack_attributes`.

### Incomplete CUB (how hidden concepts are created)

Two removal modes, both in `datasets/CUB_dataset.py`:

- **`create_random_incomplete_dataset_attr_groups(config_data, num_attribute_groups_remove)`**
  — removes whole semantic groups (from `ATTRIBUTE_PARTS`, 28 groups such as
  `has_bill_shape`, `has_wing_color`, `has_size`). This is the mode that gives clean,
  *nameable* hidden concepts. `num_attribute_groups_remove` must be in `[0, 28]`.
- **`create_random_incomplete_dataset_indiv_attr(config_data, ratio_attributes_remove)`**
  — removes a random fraction of individual attributes across all groups.

Both write a new folder `<data_path>/CUB/<incomplete_dir>/class_attr_data_10_incomplete_{cluster,local}[_indiv_attr]_<n>/`
containing rewritten `{train,val,test}.pkl` plus an **`info.txt`** recording: the mode,
the removed group names / indices, the counts, and the **old→new attribute index
mapping**. That mapping is essential for post-hoc analysis — after removal the surviving
concepts are re-indexed, so any comparison against the original 112-dim space must go
through it.

Standalone generation (without training):

```bash
python create_incomplete_cub.py +data=CUB \
  remove_attribute_groups=True num_attribute_groups_remove=14
```

`train.py` will also create the incomplete data on the fly: `check_CUB_data(config)` runs
when `incomplete=True` and the dataset is in `CUB_FAMILY_DATASETS`. If
`<data_path>/CUB/incomplete_data/<data.pkl_file_dir>` does not exist it generates a new
one and overwrites `data.pkl_file_dir`; if it does exist it reuses it and **rewrites
`data.num_concepts`** from the actual pkl contents. So `data.num_concepts=112` in the
config is only correct for the complete run — never hardcode a reduced value.

### TravelingBirds (background-shifted CUB)

`TravelingBirds` is CUB with the birds pasted onto Places365 backgrounds (Koh et al.'s
`CUB_fixed`): same 200 classes, same 112 attributes, same photographer split, **same label
pkls**. Only the image root differs, so it reuses every CUB code path.

- Config: `configs/data/TravelingBirds.yaml` (`+data=TravelingBirds`); identical to
  `CUB.yaml` apart from `dataset`.
- `datasets/CUB_dataset.py` defines `CUB_FAMILY_DATASETS = ("CUB", "TravelingBirds")` —
  every dataset-name branch in `train.py`, `inference.py`, `eval_datasets.py`,
  `utils/data.py` and `utils/plotting.py` tests membership in it, not equality with
  `"CUB"`. Add new CUB-family variants there rather than to each branch.
- `CUB_LABEL_ROOT = "CUB"`: pkls, `attributes.txt`, `concept_names.txt` and
  `incomplete_data/` always come from `<data_path>/CUB/`, for both datasets. Incomplete
  splits are therefore **shared** — generate one with `+data=CUB` and reuse it in a
  TravelingBirds run via `data.pkl_file_dir=<folder>/`, which makes the two directly
  comparable (same removed attribute groups).
- Images: `<data_path>/TravelingBirds/{train,test}/<class_folder>/<img>.jpg`. The
  train *and* val splits read from `train/` (the val split is carved out of CUB's train
  photographers); `test/` holds the test split. `train_test_split_CUB` rewrites
  `img_path` accordingly.

---

## 4. Terminology / Glossary

| Term | Meaning |
|---|---|
| `res_mu` | Predicted mean of residual logits, `μ_r(x)` |
| `c_mu` | Predicted mean of concept logits, `μ_c(x)` |
| attribute group / semantic group | One of the 28 `ATTRIBUTE_PARTS` (`has_wing_color`, …); the unit of removal in the group mode |
| incomplete run | Training on a CUB variant with attributes deleted — the deleted ones are the "hidden concepts" |
| `pkl_file_dir` | Subfolder of `CUB/incomplete_data/` holding the current incomplete split |
| `L_int` | Intervention loss enforcing benign leakage (sufficiency + localization) |
| distributed probe AUC | AUC of linear probes trained per-dimension/distributed over residual axes to detect a held-out attribute |
| effective rank | Rank of the residual channel's covariance/SVD spectrum — used to check for representational collapse |
| ΔAUC gate | Gain in AUC from adding `res_mu` on top of a `c_mu`-only baseline (conditional probing) |

---

## 5. Codebase Components

- `datasets/CUB_dataset.py` — CUB loading, transforms, split, and the two incomplete-dataset
  generators (§3)
- `create_incomplete_cub.py` — thin Hydra entry point around those two generators
- `utils/data.py` — `get_data` (dispatches to `get_CUB_dataloaders`), `get_concept_groups`
  (oracle CUB grouping from `concept_names.txt`), `get_empirical_covariance`,
  `make_analysis_loader`
- `eval_datasets.py`
- `SCBLoss` / `SCBresLoss` — concept and residual loss implementations
- `validate_one_epoch_scbm_residual` — validation loop; currently returns an **average**
  covariance matrix

Config management via **Hydra**; jobs submitted to a cluster via **`sbatch`**.
On the group cluster, experiments live at `/cluster/home/smarcou/work/experiments_scbm/`
and data at `/cluster/home/smarcou/work/Data/`.

Key saved artifacts: `avg_covariance_matrix.pt`, `c_res_mu.pt`.

---

## 6. Running Training Jobs

Entry point is `train.py`, config via **Hydra** (`configs/config.yaml` + `configs/model/*.yaml`
+ `configs/data/*.yaml`, composed with `+model=...` / `+data=...`). Any leaf value can be
overridden on the command line with dotted keys.

Example (incomplete CUB, Residual SCBM with `L_int`):

```bash
python train.py +model=SCBM_RES +data=CUB \
  model.use_L_int_loss=True model.L_int_loss_weight=5 \
  model.j_epochs=50 \
  data.num_residuals=50 \
  incomplete=True remove_attribute_groups=True num_attribute_groups_remove=14 \
  train_only=True save_model=True
```

Key flags:
- `+model=SCBM_RES` — Residual SCBM (`configs/model/SCBM_RES.yaml`, on top of
  `model_defaults.yaml`). Other options: `SCBM`, `CBM`, `CBM_RES`, `CEM`, `AR`,
  `target_head`.
- `+data=CUB` — `configs/data/CUB.yaml`.
- `incomplete=True` — use/create an incomplete attribute set (§3). Together with
  `remove_attribute_groups` (True → group removal, controlled by
  `num_attribute_groups_remove`; False → individual removal, controlled by
  `ratio_attributes_remove`). To *reuse* an existing incomplete split, pass
  `data.pkl_file_dir=<folder_name>/` and it will be loaded rather than regenerated.
- `train_only=True` — skip the post-training intervention sweep (`intervene(...)` in
  `train.py`), just train + validate + test. Much faster for iterating.
- `save_model=True` — required to persist anything to disk (checkpoints, logs, saved
  covariance / concept-residual artifacts). Without it the run is ephemeral (still logs
  to wandb unless `logging.mode=disabled`, which is the config default).
- `data.save_concept_and_residual_channel=True` — save `c_mu`/`res_mu` (and related
  artifacts, e.g. covariance) for post-hoc notebook analysis.
- `save_name=<prefix>` — optional prefix on the run folder / wandb run name.

Output location (only when `save_model=True`): `create_experiment_path()` in `train.py`
builds `experiment_dir / model.model / data.dataset / <auto-named-run>`. For CUB the run
name is prefixed with (in order of precedence): `save_name`, else
`incomplete_<num_attribute_groups_remove>_`, else
`incomplete_rmv_indiv_concepts_<ratio_attributes_remove>_`, else `complete_`; further
prefixed by `L_int_loss_weight_...` / `L_int_extension_loss_weight_...` /
`block_diagonal_cov_True_` when those are enabled.

On the group cluster (`update_config_paths()`, hostname contains `"biomed"`),
`experiment_dir`, `data.data_path` and `model.model_directory` are all forced to the
`/cluster/home/smarcou/work/...` paths regardless of the config defaults — so local vs.
cluster runs land in different places. Jobs are submitted via
`sbatch scripts/train.sh <hydra overrides...>` (`scripts/hyperparameter_search.sh` for
sweeps).

Saved artifacts per run typically include `log.txt`, `model.pth` / `model_best.pth`, and
(if requested) `avg_covariance_matrix.pt`, `c_res_mu.pt`. The **first line of `log.txt`
is a dict of the full config** — that's how `inference.py` recovers `pkl_file_dir`,
`num_concepts` and `num_residuals`.

---

## 7. Inference, Interventions & Hyperparameter Search

### Post-hoc inference / interventions on a saved model (`inference.py`)

`inference.py` reloads a trained checkpoint and re-runs evaluation and/or the
intervention sweep, using the same Hydra config system as `train.py`. The run must have
been trained with `save_model=True` (it loads `<experiment_path>/model.pth`).

```bash
python inference.py +model=SCBM_RES +data=CUB \
  inference.ex_name=<run_folder_name> \
  incomplete=True \
  run_inference=True run_interventions=True
```

Key points:
- `inference.ex_name` — the run folder name under `experiment_dir/<model>/<dataset>/`.
  Required; the script errors if the path doesn't exist.
- **CUB config is auto-recovered.** With `incomplete=True`,
  `update_pkl_dir_and_num_concepts(config)` reads the run's `log.txt` and restores
  `data.pkl_file_dir`, `data.num_concepts` and (for `scbm_residual`)
  `data.num_residuals` — so you don't have to re-pass them. It then asserts that
  `<data_path>/CUB/incomplete_data/<pkl_file_dir>` still exists and raises otherwise.
  Do **not** delete incomplete-data folders that saved runs point at.
- `run_inference=True` — test-set evaluation (writes `inference_log.txt`). For
  `scbm_residual` with `data.save_concept_and_residual_channel=True`, it additionally
  re-runs the val and train sets through shuffle-free "analysis loaders"
  (`make_analysis_loader`) to dump `c_mu`/`res_mu` artifacts into `test/`, `val/`,
  `train/` subfolders of the run directory — this is how notebook analysis inputs are
  (re)generated from an existing checkpoint.
- `run_interventions=True` — intervention curves (writes `intervention_log.txt`). For
  `scbm_residual` this currently uses `intervene_scbm_residual_optimized` (the
  non-optimized `intervene_scbm_residual` is commented out).
- `inference.tb_all_renders=True` (TravelingBirds only) — the full-render sweep. A normal
  run touches only 11788 of the 23576 rendered images, because `train_test_split_CUB`
  picks the image folder by split *name*: train/val read `TravelingBirds/train/`
  (class-correlated backgrounds), test reads `TravelingBirds/test/` (random backgrounds),
  while each folder holds a rendering of **all** 11788 CUB photos. The sweep
  (`run_tb_render_sweep` in `inference.py`) runs each split against *both* image roots
  with the deterministic test transform — 6 passes, 23576 forward passes, every file on
  disk exactly once. `<split>_bg_train` vs `<split>_bg_test` is the sample-aligned
  background-swap pairing (same bird, same labels, different background) that the analysis
  notebook otherwise has to build by hand.
  It *replaces* the standard single-split evaluation and the default `train/`, `val/`,
  `test_analysis/` dumps. Its main output is the `c_mu`/`res_mu` artifacts in
  `<split>_bg_<root>/`, so `data.save_concept_and_residual_channel=True` is required — the
  run errors out without it. Each folder also gets an **`img_paths.txt`**: the image paths
  in loader order, so line *i* names the image behind row *i* of every saved tensor — the
  way to verify that `<split>_bg_train` and `<split>_bg_test` really are row-aligned (the
  two files differ only in the `train/` vs `test/` path component). A row-count mismatch
  against `y_true.pt` raises rather than saving a silently misaligned pairing.
  Per-combination metrics go to `tb_render_sweep_log.txt` (the sweep's own log —
  `inference_log*.txt` is never touched, since the sweep is not tied to one split), to
  stdout, and to wandb under `tb_render_sweep/<split>_bg_<root>/`. `run_interventions` is
  unaffected.

### Hyperparameter search

- `hyperparameter_search=True` (in `train.py`) evaluates on the validation set at end
  of training and returns val metrics without inference/interventions; results are
  nested under `experiment_dir/hyperparameter_search/...`.
- `scripts/hyperparameter_search.sh` is an sbatch script looping `train.py` over
  `L_int_loss_weight` values on CUB (`+data=CUB`, `incomplete=True`,
  `ratio_attributes_remove=0.75`, `data.pkl_file_dir=Mateo_025/`, `data.num_residuals=50`).
  **Known bug**: the loop is written `for weight in 1,5,10,50` (commas, not spaces), so
  bash iterates once with the literal string `1,5,10,50` instead of four separate runs.

---

## 8. Current Focus

Porting the residual analysis pipeline to CUB. The pipeline shape is:

1. Train a Residual SCBM on an **incomplete** CUB split (a known set of attribute groups
   removed).
2. Dump `c_mu` / `res_mu` for train/val/test via `inference.py`.
3. **Clean**: OLS-residualize `res_mu` on `c_mu` to remove leaked observed-concept
   information.
4. **Validate**: score the residual against the held-out attribute labels recorded in
   the incomplete split's `info.txt`, using the supervised distributed probe on
   `res_mu` as the recoverability ceiling and the ΔAUC gate over a `c_mu`-only baseline.

Open items:
- The CUB analysis notebook still has to be written.
- No CUB results are recorded in this file yet.
- Held-out attribute labels for validation must be pulled from the *original*
  `class_attr_data_10` pkls and aligned through `info.txt`'s old→new index mapping.

### Interpretation guardrail

A supervised readout direction (task head, probe) trained on labels tells you what is
*recoverable* from the residual, not what the residual spontaneously organizes itself
around. It is also blind to residual structure the task head does not use. Report probe
and attribution results as statements about recoverability and task relevance, not as
evidence that a dimension "is" a particular hidden concept.

---

## 11. Tools & Libraries

- Lasso regression, distributed linear probes, raw axis AUC heatmaps
- `scipy.stats.rankdata`, `sklearn.metrics.roc_auc_score` (macro averaging)
- Hydra for config management, `sbatch` for cluster job submission

---

## 12. Relevant Literature

- Koh et al., ICML 2020 — Concept Bottleneck Models (source of the 112-attribute CUB
  setup and the image transforms used here)
- Ravfogel et al., ACL 2020 — INLP (iterative nullspace projection; principled,
  rank-controlled version of the cleaning step)
- Elazar et al., TACL 2021 — amnesic probing
- Hewitt et al., EMNLP 2021 — conditional probing (formal framing of the ΔAUC gate:
  usable information beyond a baseline representation)
- Kim et al., ICML 2018 — TCAV (linear probe directions as concept vectors; task-head
  readout is the same object derived from task labels)
