"""
Multiclass Synthetic Dataset for Residual SCBM Interpretability Experiments
============================================================================

Motivation
----------
The standard binary synthetic dataset collapses the residual channel to a
single dimension because the task label only asks one question: is the hidden
task score above or below the median? No matter how many hidden concepts exist,
the residual only needs one direction to answer it.

This dataset breaks that collapse by constructing a multiclass label from
independent binary questions about both hidden and observed concepts:

    - k hidden concept bits  -> forces the residual to encode k orthogonal
                                 directions (one per hidden concept)
    - j observed concept bits -> forces the label predictor to use c_mu,
                                 keeping the observed concept encoder
                                 task-relevant

Total classes: 2^(k + j), all approximately balanced.

Data-generating process
-----------------------
    eta = [eta_obs, eta_hid] ~ N(0, Sigma)

    c_obs = 1[eta_obs >= 0]           (observed concepts, supervised)
    c_hid = 1[eta_hid >= 0]           (hidden concepts, not supervised)

    x = MLP(concept_signal, residual_signal) + noise

    concept_signal  = |eta_obs| * c_obs   (hard difficulty)
    residual_signal = |eta_hid| * c_hid

    w_obs, w_hid = sparse task weight vectors

    Top-k hidden concepts by |w_hid| -> k bits -> bits 0..k-1
    Top-j observed concepts by |w_obs| -> j bits -> bits k..k+j-1

    y = sum of bit_i * 2^i across all k+j bits

Bit encoding example (k=1, j=1, 4 classes):
    class 0: h_top=0, c_top=0
    class 1: h_top=1, c_top=0
    class 2: h_top=0, c_top=1
    class 3: h_top=1, c_top=1

Classes 0 vs 1 differ only on the hidden concept  -> residual must track it.
Classes 0 vs 2 differ only on the observed concept -> label predictor must use c_mu.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os


class MulticlassSyntheticResidualDataset(Dataset):
    """
    Multiclass synthetic dataset for Residual SCBM interpretability experiments.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate before splitting.
    num_covariates : int
        Dimensionality of the observed input x.
    obs_dim : int
        Number of observed (supervised) concepts.
    hid_dim : int
        Number of hidden (unsupervised) concepts.
    latent_rank : int
        Rank of the low-rank component of the covariance matrix.
        Set to 0 for an identity covariance (independent concepts).
    sigma_x : float
        Standard deviation of the Gaussian noise added to x.
    task_sparsity_obs : float
        Fraction of observed concepts with non-zero task weight.
    task_sparsity_hid : float
        Fraction of hidden concepts with non-zero task weight.
    num_hid_class_bits : int
        Number of top hidden concepts (by |w_hid|) used as class-defining
        bits. Must be >= 1. Recommended: 1 or 2.
    num_obs_class_bits : int
        Number of top observed concepts (by |w_obs|) used as class-defining
        bits. Must be >= 1 to keep the observed concept encoder task-relevant.
        Recommended: 1.
    rho_cc : float
        Scales off-diagonal entries of the observed-observed concept block
        in the covariance matrix. 0 = independent observed concepts.
    rho_rr : float
        Scales off-diagonal entries of the hidden-hidden concept block.
        0 = independent hidden concepts.
    rho_cr : float
        Scales the cross-covariance between observed and hidden concepts.
        0 = no correlation between observed and hidden concepts.
    seed : int
        Random seed for reproducibility.
    indices : array-like or None
        If provided, subsets the generated data to these indices. Used
        internally by the factory function for train/val/test splitting.

    Attributes
    ----------
    x                    : (n, num_covariates) float32 tensor
    concepts             : (n, obs_dim) float32 tensor  — supervised
    residuals            : (n, hid_dim) float32 tensor  — not supervised
    y                    : (n,) int64 tensor             — class labels
    eta_concepts         : (n, obs_dim) float32 tensor  — continuous logits
    eta_residuals        : (n, hid_dim) float32 tensor  — continuous logits
    concept_signal       : (n, obs_dim) float32 tensor  — |eta| * binary
    residual_signal      : (n, hid_dim) float32 tensor  — |eta| * binary
    w_obs                : (obs_dim,) float32 tensor    — sparse task weights
    w_hid                : (hid_dim,) float32 tensor    — sparse task weights
    multiclass_hid_idx   : (num_hid_class_bits,) int64 tensor
                           Indices of hidden concepts defining the class bits,
                           ordered by |w_hid| descending.
    multiclass_obs_idx   : (num_obs_class_bits,) int64 tensor
                           Indices of observed concepts defining the class bits,
                           ordered by |w_obs| descending.
    num_classes          : int — total number of classes (2^(k+j))
    """

    def __init__(
        self,
        n_samples,
        num_covariates,
        obs_dim,
        hid_dim,
        latent_rank,
        sigma_x,
        task_sparsity_obs,
        task_sparsity_hid,
        num_hid_class_bits,
        num_obs_class_bits,
        rho_cc=0.0,
        rho_rr=0.0,
        rho_cr=0.0,
        seed=0,
        indices=None,
    ):
        super().__init__()

        assert num_hid_class_bits >= 1, "num_hid_class_bits must be >= 1"
        assert num_obs_class_bits >= 1, "num_obs_class_bits must be >= 1"

        self.n_samples = n_samples
        self.num_covariates = num_covariates
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.latent_rank = latent_rank
        self.sigma_x = sigma_x
        self.task_sparsity_obs = task_sparsity_obs
        self.task_sparsity_hid = task_sparsity_hid
        self.num_hid_class_bits = num_hid_class_bits
        self.num_obs_class_bits = num_obs_class_bits
        self.rho_cc = rho_cc
        self.rho_rr = rho_rr
        self.rho_cr = rho_cr
        self.seed = seed
        self.num_classes = 2 ** (num_hid_class_bits + num_obs_class_bits)

        rng = np.random.default_rng(seed)

        # ----------------------------------------------------------------
        # 1. Covariance matrix for latent concept logits
        # ----------------------------------------------------------------
        Sigma = _make_covariance(
            obs_dim=obs_dim,
            hid_dim=hid_dim,
            latent_rank=latent_rank,
            rho_cc=rho_cc,
            rho_rr=rho_rr,
            rho_cr=rho_cr,
            rng=rng,
        )

        # ----------------------------------------------------------------
        # 2. Sample latent variables eta = [eta_obs, eta_hid]
        # ----------------------------------------------------------------
        eta = rng.multivariate_normal(
            mean=np.zeros(obs_dim + hid_dim),
            cov=Sigma,
            size=n_samples,
        )

        eta_concepts  = eta[:, :obs_dim]
        eta_residuals = eta[:, obs_dim:]

        concepts  = (eta_concepts  >= 0).astype(np.float32)
        residuals = (eta_residuals >= 0).astype(np.float32)

        concept_signal  = np.abs(eta_concepts)  * concepts
        residual_signal = np.abs(eta_residuals) * residuals

        # ----------------------------------------------------------------
        # 3. Generate input features x  (hard difficulty: signal-based)
        # ----------------------------------------------------------------
        eta_for_x = np.concatenate([concept_signal, residual_signal], axis=1)
        x = _random_mlp_features(eta_for_x, num_covariates, rng, sigma_x)

        # ----------------------------------------------------------------
        # 4. Sparse task weights
        # ----------------------------------------------------------------
        w_obs = _make_sparse_weights(obs_dim, task_sparsity_obs, rng)
        w_hid = _make_sparse_weights(hid_dim, task_sparsity_hid, rng)

        # ----------------------------------------------------------------
        # 5. Multiclass label
        #
        # Bit layout:
        #   bits 0 .. k-1  : top-k hidden concepts by |w_hid|
        #   bits k .. k+j-1: top-j observed concepts by |w_obs|
        #
        # Each bit is the sign of the corresponding eta value (>= 0 -> 1),
        # which is Bernoulli(0.5) by construction, giving balanced classes.
        # ----------------------------------------------------------------

        # Top-k hidden concept indices, ordered by |w_hid| descending
        hid_class_idx = np.argsort(np.abs(w_hid))[::-1][:num_hid_class_bits].copy()

        # Top-j observed concept indices, ordered by |w_obs| descending
        obs_class_idx = np.argsort(np.abs(w_obs))[::-1][:num_obs_class_bits].copy()

        y = np.zeros(n_samples, dtype=np.int64)

        # Hidden bits first
        for bit, idx in enumerate(hid_class_idx):
            y += (eta_residuals[:, idx] >= 0).astype(np.int64) * (2 ** bit)

        # Observed bits after
        for j, idx in enumerate(obs_class_idx):
            bit = num_hid_class_bits + j
            y += (eta_concepts[:, idx] >= 0).astype(np.int64) * (2 ** bit)

        # Sanity check: all classes must be populated
        class_counts = np.bincount(y, minlength=self.num_classes)
        assert np.min(class_counts) > 0, (
            f"Some classes are empty: {class_counts}. "
            "Increase n_samples or reduce num_hid_class_bits / num_obs_class_bits."
        )

        # ----------------------------------------------------------------
        # 6. Store as tensors
        # ----------------------------------------------------------------
        self.x               = torch.tensor(x,               dtype=torch.float32)
        self.concepts        = torch.tensor(concepts,        dtype=torch.float32)
        self.residuals       = torch.tensor(residuals,       dtype=torch.float32)
        self.y               = torch.tensor(y,               dtype=torch.long)
        self.eta_concepts    = torch.tensor(eta_concepts,    dtype=torch.float32)
        self.eta_residuals   = torch.tensor(eta_residuals,   dtype=torch.float32)
        self.concept_signal  = torch.tensor(concept_signal,  dtype=torch.float32)
        self.residual_signal = torch.tensor(residual_signal, dtype=torch.float32)
        self.w_obs           = torch.tensor(w_obs,           dtype=torch.float32)
        self.w_hid           = torch.tensor(w_hid,           dtype=torch.float32)
        self.Sigma           = torch.tensor(Sigma,           dtype=torch.float32)

        # Which concept indices define the class structure (for analysis)
        self.multiclass_hid_idx = torch.tensor(hid_class_idx, dtype=torch.long)
        self.multiclass_obs_idx = torch.tensor(obs_class_idx, dtype=torch.long)

        # ----------------------------------------------------------------
        # 7. Apply split indices if provided
        # ----------------------------------------------------------------
        if indices is not None:
            self.x = self.x[indices]
            self.concepts = self.concepts[indices]
            self.residuals = self.residuals[indices]
            self.y = self.y[indices]
            self.eta_concepts    = self.eta_concepts[indices]
            self.eta_residuals   = self.eta_residuals[indices]
            self.concept_signal  = self.concept_signal[indices]
            self.residual_signal = self.residual_signal[indices]
            self.n_samples       = self.x.shape[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "features": self.x[idx],
            "concepts": self.concepts[idx],
            "labels":   self.y[idx],
            "residuals": self.residuals[idx],
        }

    def class_counts(self):
        """Return the number of samples per class as a numpy array."""
        return np.bincount(self.y.numpy(), minlength=self.num_classes)

    def describe(self):
        """Print a summary of the dataset configuration."""
        print(f"MulticlassSyntheticResidualDataset")
        print(f"  n_samples      : {self.n_samples}")
        print(f"  obs_dim        : {self.obs_dim}")
        print(f"  hid_dim        : {self.hid_dim}")
        print(f"  num_classes    : {self.num_classes}  "
              f"(k={self.num_hid_class_bits} hid bits + j={self.num_obs_class_bits} obs bits)")
        print(f"  class counts   : {self.class_counts()}")
        print(f"  hid_class_idx  : {self.multiclass_hid_idx.tolist()}  "
              f"|w_hid|={self.w_hid.abs()[self.multiclass_hid_idx].numpy().round(3)}")
        print(f"  obs_class_idx  : {self.multiclass_obs_idx.tolist()}  "
              f"|w_obs|={self.w_obs.abs()[self.multiclass_obs_idx].numpy().round(3)}")
        print(f"  sigma_x        : {self.sigma_x}")
        print(f"  rho_cc/rr/cr   : {self.rho_cc} / {self.rho_rr} / {self.rho_cr}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_covariance(obs_dim, hid_dim, latent_rank, rho_cc, rho_rr, rho_cr, rng):
    """
    Build a positive-definite covariance matrix with controllable block structure.

        Sigma = [[Sigma_obs, Sigma_cross],
                 [Sigma_cross^T, Sigma_hid]]

    rho_cc : scales off-diagonal entries in the observed-observed block
    rho_rr : scales off-diagonal entries in the hidden-hidden block
    rho_cr : scales entries in the cross block
    """
    total_dim = obs_dim + hid_dim

    if latent_rank == 0:
        return (np.eye(total_dim) + 0.1 * np.eye(total_dim)).astype(np.float32)

    # Low-rank base correlation matrix
    W = rng.normal(size=(total_dim, latent_rank))
    Sigma_base = W @ W.T
    d = np.sqrt(np.diag(Sigma_base) + 1e-8)
    Sigma_base /= np.outer(d, d)

    Sigma = np.eye(total_dim)
    obs = slice(0, obs_dim)
    hid = slice(obs_dim, total_dim)

    # Observed block
    obs_offdiag = Sigma_base[obs, obs].copy()
    np.fill_diagonal(obs_offdiag, 0.0)
    Sigma[obs, obs] = np.eye(obs_dim) + rho_cc * obs_offdiag

    # Hidden block
    hid_offdiag = Sigma_base[hid, hid].copy()
    np.fill_diagonal(hid_offdiag, 0.0)
    Sigma[hid, hid] = np.eye(obs_dim if obs_dim == hid_dim
                              else hid_dim) + rho_rr * hid_offdiag

    # Cross block
    Sigma[obs, hid] = rho_cr * Sigma_base[obs, hid]
    Sigma[hid, obs] = rho_cr * Sigma_base[hid, obs]

    # Symmetrize and ensure positive definiteness
    Sigma = 0.5 * (Sigma + Sigma.T) + 0.1 * np.eye(total_dim)
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig <= 0:
        Sigma += (abs(min_eig) + 1e-3) * np.eye(total_dim)

    return Sigma.astype(np.float32)


def _make_sparse_weights(dim, sparsity, rng):
    """Sparse task weight vector, normalised to unit norm."""
    w = np.zeros(dim, dtype=np.float32)
    n_active = max(1, int(sparsity * dim))
    active_idx = rng.choice(dim, size=n_active, replace=False)
    w[active_idx] = rng.normal(size=n_active).astype(np.float32)
    w /= np.linalg.norm(w) + 1e-8
    return w


def _random_mlp_features(eta, num_covariates, rng, sigma_x):
    """Fixed random two-layer MLP to generate nonlinear input features."""
    n, latent_dim = eta.shape
    hidden_dim = min(512, max(128, 2 * latent_dim))

    W1 = rng.normal(size=(latent_dim, hidden_dim)) / np.sqrt(latent_dim)
    b1 = rng.normal(size=(hidden_dim,)) * 0.1
    W2 = rng.normal(size=(hidden_dim, num_covariates)) / np.sqrt(hidden_dim)
    b2 = rng.normal(size=(num_covariates,)) * 0.1

    h = np.tanh(eta @ W1 + b1)
    x = h @ W2 + b2 + rng.normal(scale=sigma_x, size=(n, num_covariates))
    return x


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_multiclass_datasets(config, seed=0, log_file=None):
    """
    Create train / val / test splits of MulticlassSyntheticResidualDataset.

    Expected config fields
    ----------------------
    config.data.n_samples
    config.data.num_covariates
    config.data.obs_dim
    config.data.hid_dim
    config.data.latent_rank
    config.data.sigma_x
    config.data.task_sparsity_obs
    config.data.task_sparsity_hid
    config.data.num_hid_class_bits     (int, >= 1)
    config.data.num_obs_class_bits     (int, >= 1, default 1)
    config.data.rho_cc                 (float, default 0.0)
    config.data.rho_rr                 (float, default 0.0)
    config.data.rho_cr                 (float, default 0.0)
    config.data.train_ratio
    config.data.val_ratio

    Returns
    -------
    train_dataset, val_dataset, test_dataset
    """
    idx_all = np.arange(config.data.n_samples)
    idx_train, idx_valtest = train_test_split(
        idx_all, train_size=config.data.train_ratio, random_state=seed
    )
    idx_val, idx_test = train_test_split(
        idx_valtest,
        train_size=config.data.val_ratio / (1.0 - config.data.train_ratio),
        random_state=2 * seed,
    )

    print(
        f"Train: {len(idx_train)}, "
        f"Val: {len(idx_val)}, "
        f"Test: {len(idx_test)}"
    )

    shared = dict(
        n_samples=config.data.n_samples,
        num_covariates=config.data.num_covariates,
        obs_dim=config.data.obs_dim,
        hid_dim=config.data.hid_dim,
        latent_rank=config.data.latent_rank,
        sigma_x=config.data.sigma_x,
        task_sparsity_obs=config.data.task_sparsity_obs,
        task_sparsity_hid=config.data.task_sparsity_hid,
        num_hid_class_bits=config.data.num_hid_class_bits,
        num_obs_class_bits=getattr(config.data, "num_obs_class_bits", 1),
        rho_cc=getattr(config.data, "rho_cc", 0.0),
        rho_rr=getattr(config.data, "rho_rr", 0.0),
        rho_cr=getattr(config.data, "rho_cr", 0.0),
        seed=seed,
    )

    train_ds = MulticlassSyntheticResidualDataset(indices=idx_train, **shared)
    val_ds   = MulticlassSyntheticResidualDataset(indices=idx_val,   **shared)
    test_ds  = MulticlassSyntheticResidualDataset(indices=idx_test,  **shared)

    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"[multiclass dataset]\n")
            f.write(f"  n_samples        : {config.data.n_samples}\n")
            f.write(f"  obs_dim          : {config.data.obs_dim}\n")
            f.write(f"  hid_dim          : {config.data.hid_dim}\n")
            f.write(f"  num_hid_class_bits: {config.data.num_hid_class_bits}\n")
            f.write(f"  num_obs_class_bits: {shared['num_obs_class_bits']}\n")
            f.write(f"  num_classes      : {train_ds.num_classes}\n")
            f.write(f"  sigma_x          : {config.data.sigma_x}\n")
            f.write(f"  rho_cc/rr/cr     : {shared['rho_cc']} / "
                    f"{shared['rho_rr']} / {shared['rho_cr']}\n")
            f.write(f"  hid_class_idx    : {train_ds.multiclass_hid_idx.tolist()}\n")
            f.write(f"  obs_class_idx    : {train_ds.multiclass_obs_idx.tolist()}\n")
            f.write(f"  train/val/test   : {len(train_ds)} / "
                    f"{len(val_ds)} / {len(test_ds)}\n")

    return train_ds, val_ds, test_ds




def save_multiclass_data(config, train, val, test, log_file):
    """
    Save a multiclass dataset split to disk and record the save path in the
    training log file.
 
    Directory structure
    -------------------
    <data_path>/synthetic_multiclass/
        <save_name>/
            train/   x.pt, concepts.pt, residuals.pt, y.pt,
                     eta_concepts.pt, eta_residuals.pt,
                     concept_signal.pt, residual_signal.pt,
                     w_obs.pt, w_hid.pt, Sigma.pt,
                     multiclass_hid_idx.pt, multiclass_obs_idx.pt
            val/     (same files)
            test/    (same files)
            info.txt
 
    The save name encodes the key hyperparameters so experiments can be
    identified from the directory name alone. A version suffix (_v1, _v2, ...)
    is appended automatically if a matching directory already exists.
 
    Parameters
    ----------
    config   : config object with config.data.* and config.seed fields
    train    : MulticlassSyntheticResidualDataset (train split)
    val      : MulticlassSyntheticResidualDataset (val split)
    test     : MulticlassSyntheticResidualDataset (test split)
    log_file : path to the training log file; save path is appended to it
    seed     : random seed used to generate the dataset
    """
    root = os.path.join(config.data.data_path, config.data.dataset)
    os.makedirs(root, exist_ok=True)
 
    # Build a human-readable name encoding the key hyperparameters
    hostname = os.uname()[1]
    prefix = "cluster_" if "biomed" in hostname else "local_"
 
    save_name = (
        prefix
        + f"hid_bits_{train.num_hid_class_bits}"
        + f"_obs_bits_{train.num_obs_class_bits}"
        + f"_classes_{train.num_classes}"
        + f"_rho_cr{train.rho_cr}"
        + f"_rho_cc{train.rho_cc}"
        + f"_rho_rr{train.rho_rr}"
        + f"_r_sparsity_{train.task_sparsity_hid}"
        + f"_c_sparsity_{train.task_sparsity_obs}"
        + f"_sigmax_{train.sigma_x}"
        + f"_hid_dim_{train.hid_dim}"
        + f"_obs_dim_{train.obs_dim}"
        + f"_seed_{config.seed}"
    )
 
    # Append version suffix if a matching directory already exists
    existing = [d for d in os.listdir(root) if d.startswith(save_name)]
    if existing:
        version = 1
        while os.path.exists(os.path.join(root, f"{save_name}_v{version}")):
            version += 1
        save_name = f"{save_name}_v{version}"
 
    save_dir = os.path.join(root, save_name)
    os.makedirs(save_dir, exist_ok=True)
 
    # Tensors to save for each split
    tensor_fields = [
        "x", "concepts", "residuals", "y",
        "eta_concepts", "eta_residuals",
        "concept_signal", "residual_signal",
        "w_obs", "w_hid", "Sigma",
        "multiclass_hid_idx", "multiclass_obs_idx",
    ]
 
    for split_name, dataset in [("train", train), ("val", val), ("test", test)]:
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for field in tensor_fields:
            # Save each tensor as a .pt file
            torch.save(
                getattr(dataset, field),
                os.path.join(split_dir, f"{field}.pt"),
            )
 
    # Human-readable info file alongside the split directories
    info_path = os.path.join(save_dir, "info.txt")
    log_parent = os.path.dirname(log_file)
    with open(info_path, "w") as f:
        f.write("[multiclass synthetic dataset]\n")
        f.write(f"num_hid_class_bits : {train.num_hid_class_bits}\n")
        f.write(f"num_obs_class_bits : {train.num_obs_class_bits}\n")
        f.write(f"num_classes        : {train.num_classes}\n")
        f.write(f"hid_class_idx      : {train.multiclass_hid_idx.tolist()}\n")
        f.write(f"obs_class_idx      : {train.multiclass_obs_idx.tolist()}\n")
        f.write(f"rho_cr             : {train.rho_cr}\n")
        f.write(f"rho_cc             : {train.rho_cc}\n")
        f.write(f"rho_rr             : {train.rho_rr}\n")
        f.write(f"task_sparsity_hid  : {train.task_sparsity_hid}\n")
        f.write(f"task_sparsity_obs  : {train.task_sparsity_obs}\n")
        f.write(f"sigma_x            : {train.sigma_x}\n")
        f.write(f"hid_dim            : {train.hid_dim}\n")
        f.write(f"obs_dim            : {train.obs_dim}\n")
        f.write(f"seed               : {config.seed}\n")
        f.write(f"train / val / test : {len(train)} / {len(val)} / {len(test)}\n")
        f.write(f"data created for model at: {log_parent}\n")
 
    # Append save path to the training log
    with open(log_file, "a") as f:
        f.write(f"data_dir: {save_dir}\n")
 
    print(f"Saved multiclass dataset to: {save_dir}")
    return save_dir





class LoadedMulticlassDataset(Dataset):
    """
    Lightweight wrapper around tensors loaded from disk.
    Matches the __getitem__ interface of MulticlassSyntheticResidualDataset
    so it can be used as a drop-in replacement in training code.
    """
 
    def __init__(self, split_dir):
        self.x               = torch.load(os.path.join(split_dir, "x.pt"))
        self.concepts        = torch.load(os.path.join(split_dir, "concepts.pt"))
        self.residuals       = torch.load(os.path.join(split_dir, "residuals.pt"))
        self.y               = torch.load(os.path.join(split_dir, "y.pt"))
        self.eta_concepts    = torch.load(os.path.join(split_dir, "eta_concepts.pt"))
        self.eta_residuals   = torch.load(os.path.join(split_dir, "eta_residuals.pt"))
        self.concept_signal  = torch.load(os.path.join(split_dir, "concept_signal.pt"))
        self.residual_signal = torch.load(os.path.join(split_dir, "residual_signal.pt"))
        self.w_obs           = torch.load(os.path.join(split_dir, "w_obs.pt"))
        self.w_hid           = torch.load(os.path.join(split_dir, "w_hid.pt"))
        self.Sigma           = torch.load(os.path.join(split_dir, "Sigma.pt"))
        self.multiclass_hid_idx = torch.load(os.path.join(split_dir, "multiclass_hid_idx.pt"))
        self.multiclass_obs_idx = torch.load(os.path.join(split_dir, "multiclass_obs_idx.pt"))
        self.n_samples  = self.x.shape[0]
        self.num_classes = int(self.y.max().item()) + 1
 
    def __len__(self):
        return self.n_samples
 
    def __getitem__(self, idx):
        return {
            "features":  self.x[idx],
            "concepts":  self.concepts[idx],
            "labels":    self.y[idx],
            "residuals": self.residuals[idx],
        }
 
 
def load_saved_multiclass_data(config):
    """
    Load a previously saved multiclass dataset from disk.
 
    Parameters
    ----------
    config : DictConfig
        Configuration object containing data directory information.
 
    Returns
    -------
    train, val, test : LoadedMulticlassDataset
    """
    
    
    
    data_dir_root = os.path.join(config.data.data_path, config.data.dataset)
    full_data_dir_path = os.path.join(data_dir_root, config.data.data_dir_name)
    
    
    train = LoadedMulticlassDataset(os.path.join(full_data_dir_path, "train"))
    val   = LoadedMulticlassDataset(os.path.join(full_data_dir_path, "val"))
    test  = LoadedMulticlassDataset(os.path.join(full_data_dir_path, "test"))
    return train, val, test












def check_synthetic_multiclass_dataset(config):
    """
    Update the number of classes in the config based on the multiclass dataset parameters.
    """
    # calculate number of classes based on the number of bits used for hidden and observed concepts
    if config.data.data_dir_name is None:
        num_hid_class_bits = config.data.num_hid_class_bits
        num_obs_class_bits = config.data.num_obs_class_bits
        num_classes = 2 ** (num_hid_class_bits + num_obs_class_bits)
        config.data.num_classes = num_classes  # store in config for later use
        
        config.data.num_concepts = config.data.obs_dim 
    
    else:
        train_data, _, _ = load_saved_multiclass_data(config)
        config.data.num_concepts = train_data.concepts.shape[1]
        
    
    
