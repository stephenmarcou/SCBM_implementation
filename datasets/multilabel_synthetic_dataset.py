"""
Multi-Label Synthetic Dataset for Residual SCBM Interpretability Experiments
=============================================================================

Motivation
----------
The multiclass dataset (multiclass_synthetic_dataset.py) collapses the residual
channel to exactly k dimensions because the label is a single integer encoding
k binary bits. Although k bits force k orthogonal directions, the structure is
artificial: each bit is a perfectly clean Bernoulli(0.5) variable, and the
label encodes them in a fixed positional scheme.

This dataset replaces the single multiclass integer with K independent binary
subtasks. Each subtask k is a noisy threshold of a continuous hidden score:

    s_hid_k = alpha * (concept_signal @ w_obs) + beta * residual_signal[:, hid_task_idx[k]]
    y_k = 1[s_hid_k >= median(s_hid_k)]

This is more realistic because:
  - Each subtask depends on one hidden concept *and* the full observed concept
    signal, so the residual must encode hidden concepts while cooperating with
    the concept channel rather than being fully independent.
  - The continuous score adds graded signal: how strongly a hidden concept is
    expressed (residual_signal = |eta| * binary) affects the label margin, not
    just the binary presence. The residual channel benefits from encoding
    continuous signal, not just sign(eta).
  - K subtasks force K genuinely independent directions in the residual channel,
    but without artificially constraining each direction to a single coordinate.
    The SVD effective rank will be ~K, emerging from the task structure rather
    than being hard-coded.

Data-generating process
-----------------------
    eta = [eta_obs, eta_hid] ~ N(0, Sigma)

    c_obs = 1[eta_obs >= 0]  (observed, supervised)
    c_hid = 1[eta_hid >= 0]  (hidden, not supervised)
    concept_signal = |eta_obs| * c_obs
    residual_signal = |eta_hid| * c_hid

    x = MLP(concept_signal, residual_signal) + noise

    w_obs, w_hid = sparse task weight vectors (unit norm)

    hid_task_idx = top-K hidden concept indices by |w_hid|
    obs_task_idx = top-J observed concept indices by |w_obs|

    For each hidden subtask k in 0..K-1:
        s_hid_k = alpha * (concept_signal @ w_obs)
                + beta  * residual_signal[:, hid_task_idx[k]] * |w_hid[hid_task_idx[k]]|
        y_hid_k = 1[s_hid_k >= median(s_hid_k)]

    For each observed subtask j in 0..J-1:
        s_obs_j = alpha * concept_signal[:, obs_task_idx[j]] * |w_obs[obs_task_idx[j]]|
                + beta  * (residual_signal @ w_hid)
        y_obs_j = 1[s_obs_j >= median(s_obs_j)]

    y = (y_hid_0, ..., y_hid_{K-1}, y_obs_0, ..., y_obs_{J-1})  shape (n, K+J)

The observed subtasks keep the concept encoder task-relevant: if the model
ignores c_mu the observed subtasks cannot be predicted.

Design choices
--------------
- alpha scales the shared observed-concept contribution to every subtask.
  Setting alpha > 0 means each subtask is not purely about one hidden concept;
  the model must use both channels jointly.
- beta scales the per-hidden-concept contribution. Higher beta -> each hidden
  concept is more individually decisive -> cleaner recovery.
- With alpha=0 the subtasks are purely about individual hidden concepts and
  the residual collapses cleanly to K dimensions (closest to the multiclass
  dataset behaviour).
- With alpha=1, beta=1 the subtasks mix both channels, giving a more realistic
  and harder setting.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os


class MultilabelSyntheticResidualDataset(Dataset):
    """
    Multi-label synthetic dataset for Residual SCBM interpretability experiments.

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
        Rank of the low-rank covariance component. 0 = identity (independent).
    sigma_x : float
        Gaussian noise std added to x.
    task_sparsity_obs : float
        Fraction of observed concepts with nonzero task weight.
    task_sparsity_hid : float
        Fraction of hidden concepts with nonzero task weight.
    num_hid_tasks : int
        Number of hidden-concept subtasks (K). Each uses one hidden concept
        as its primary signal. Must be <= number of active hidden concepts
        implied by task_sparsity_hid * hid_dim.
    num_obs_tasks : int
        Number of observed-concept subtasks (J). Keeps the concept encoder
        task-relevant. Default 1.
    alpha : float
        Weight of the shared concept-signal term in each subtask score.
        0 = subtasks are purely about individual concepts (cleaner, lower rank).
        1 = subtasks mix both channels (more realistic, harder).
    beta : float
        Weight of the per-concept signal term in each subtask score.
    rho_cc : float
        Off-diagonal scaling in the observed-observed covariance block.
    rho_rr : float
        Off-diagonal scaling in the hidden-hidden covariance block.
    rho_cr : float
        Cross-covariance scaling between observed and hidden blocks.
    seed : int
    indices : array-like or None
        Subset indices for train/val/test splitting.

    Attributes
    ----------
    x                  : (n, num_covariates) float32
    concepts           : (n, obs_dim) float32   — supervised binary concepts
    residuals          : (n, hid_dim) float32   — hidden binary concepts
    y                  : (n, K+J) float32       — multi-label targets
    eta_concepts       : (n, obs_dim) float32   — continuous concept logits
    eta_residuals      : (n, hid_dim) float32   — continuous hidden logits
    concept_signal     : (n, obs_dim) float32   — |eta_obs| * c_obs
    residual_signal    : (n, hid_dim) float32   — |eta_hid| * c_hid
    s_hid              : (n, K) float32         — continuous scores for hid subtasks
    s_obs              : (n, J) float32         — continuous scores for obs subtasks
    w_obs              : (obs_dim,) float32     — sparse observed task weights
    w_hid              : (hid_dim,) float32     — sparse hidden task weights
    Sigma              : (obs_dim+hid_dim, obs_dim+hid_dim) float32
    hid_task_idx       : (K,) int64  — hidden concept indices per subtask
    obs_task_idx       : (J,) int64  — observed concept indices per subtask
    num_hid_tasks      : int  K
    num_obs_tasks      : int  J
    num_classes          : int  K + J
    alpha, beta        : float
    """

    def __init__(
        self,
        config,
        indices=None,

    ):
        super().__init__()

 
        self.n_samples = config.data.n_samples
        self.num_covariates = config.data.num_covariates
        self.obs_dim = config.data.obs_dim
        self.hid_dim = config.data.hid_dim
        self.latent_rank = config.data.latent_rank
        self.sigma_x = config.data.sigma_x
        self.task_sparsity_obs = config.data.task_sparsity_obs
        self.task_sparsity_hid = config.data.task_sparsity_hid
        self.num_hid_tasks = config.data.num_hid_tasks
        self.num_obs_tasks = config.data.num_obs_tasks
        self.num_classes = self.num_hid_tasks + self.num_obs_tasks
        self.alpha = config.data.alpha
        self.beta = config.data.beta
        self.rho_cc = config.data.rho_cc
        self.rho_rr = config.data.rho_rr
        self.rho_cr = config.data.rho_cr
        self.seed = config.seed
        self.min_weight_ratio = config.data.min_weight_ratio
        self.standardize = config.data.standardize
        
        assert self.num_hid_tasks >= 1, "num_hid_tasks must be >= 1"
        assert self.num_obs_tasks >= 1, "num_obs_tasks must be >= 1"
        assert self.alpha >= 0 and self.beta >= 0, "alpha or beta must be non-negative"


        rng = np.random.default_rng(self.seed)

        # ----------------------------------------------------------------
        # 1. Covariance matrix
        # ----------------------------------------------------------------
        Sigma = _make_covariance(
            obs_dim=self.obs_dim,
            hid_dim=self.hid_dim,
            latent_rank=self.latent_rank,
            rho_cc=self.rho_cc,
            rho_rr=self.rho_rr,
            rho_cr=self.rho_cr,
            rng=rng,
        )

        # ----------------------------------------------------------------
        # 2. Sample latent variables
        # ----------------------------------------------------------------
        eta = rng.multivariate_normal(
            mean=np.zeros(self.obs_dim + self.hid_dim),
            cov=Sigma,
            size=self.n_samples,
        )

        eta_concepts = eta[:, :self.obs_dim]
        eta_residuals = eta[:, self.obs_dim:]

        concepts = (eta_concepts >= 0).astype(np.float32)
        residuals = (eta_residuals >= 0).astype(np.float32)

        concept_signal = np.abs(eta_concepts) * concepts  # (n, obs_dim)
        residual_signal = np.abs(eta_residuals) * residuals  # (n, hid_dim)

        # ----------------------------------------------------------------
        # 3. Input features x
        # ----------------------------------------------------------------
        eta_for_x = np.concatenate([concept_signal, residual_signal], axis=1)
        x = _random_mlp_features(eta_for_x, self.num_covariates, rng, self.sigma_x)

        # ----------------------------------------------------------------
        # 4. Sparse task weights
        # ----------------------------------------------------------------
        w_obs = _make_sparse_weights(
            self.obs_dim,
            self.task_sparsity_obs,
            rng,
            min_weight_ratio=self.min_weight_ratio,
        )
        w_hid = _make_sparse_weights(
            self.hid_dim,
            self.task_sparsity_hid,
            rng,
            min_weight_ratio=self.min_weight_ratio,
        )

        # Validate enough active concepts exist for the requested tasks
        n_active_hid = max(1, int(self.task_sparsity_hid * self.hid_dim))
        n_active_obs = max(1, int(self.task_sparsity_obs * self.obs_dim))
        assert self.num_hid_tasks <= n_active_hid, (
            f"num_hid_tasks={self.num_hid_tasks} exceeds the number of active hidden "
            f"concepts ({n_active_hid}) implied by task_sparsity_hid={self.task_sparsity_hid}. "
            f"Increase task_sparsity_hid or decrease num_hid_tasks."
        )
        assert self.num_obs_tasks <= n_active_obs, (
            f"num_obs_tasks={self.num_obs_tasks} exceeds the number of active observed "
            f"concepts ({n_active_obs}) implied by task_sparsity_obs={self.task_sparsity_obs}. "
            f"Increase task_sparsity_obs or decrease num_obs_tasks."
        )

        # Top-K hidden / top-J observed indices by absolute task weight
        hid_task_idx = np.argsort(np.abs(w_hid))[::-1][:self.num_hid_tasks].copy()
        obs_task_idx = np.argsort(np.abs(w_obs))[::-1][:self.num_obs_tasks].copy()

        # ----------------------------------------------------------------
        # 5. Multi-label targets
        #
        # Hidden subtask k:
        #   s_hid_k = alpha * (concept_signal @ w_obs)
        #           + beta  * residual_signal[:, hid_task_idx[k]] * |w_hid[hid_task_idx[k]]|
        #   y_hid_k = 1[s_hid_k >= median(s_hid_k)]
        #
        # The |w_hid[idx]| scale keeps the per-concept contribution comparable
        # across tasks regardless of weight magnitude, avoiding one subtask
        # dominating the others.
        #
        # Observed subtask j:
        #   s_obs_j = alpha * concept_signal[:, obs_task_idx[j]] * |w_obs[obs_task_idx[j]]|
        #           + beta  * (residual_signal @ w_hid)
        #   y_obs_j = 1[s_obs_j >= median(s_obs_j)]
        # ----------------------------------------------------------------





        # # Shared terms (computed once)
        # shared_obs_score = concept_signal @ w_obs  # (n,)  alpha term for hid subtasks
        # shared_hid_score = residual_signal @ w_hid  # (n,)  beta term for obs subtasks
        
        # def standardize(z, eps=1e-8):
        #     return (z - z.mean()) / (z.std() + eps)

        # if self.standardize:
        #     pass
        

        # s_hid = np.zeros((n_samples, num_hid_tasks), dtype=np.float32)
        # y_hid = np.zeros((n_samples, num_hid_tasks), dtype=np.float32)

        # for k, idx in enumerate(hid_task_idx):
        #     per_concept = residual_signal[:, idx] * float(np.abs(w_hid[idx]))
        #     s_hid[:, k] = alpha * shared_obs_score + beta * per_concept
        #     y_hid[:, k] = (s_hid[:, k] >= np.median(s_hid[:, k])).astype(np.float32)

        # s_obs = np.zeros((n_samples, num_obs_tasks), dtype=np.float32)
        # y_obs = np.zeros((n_samples, num_obs_tasks), dtype=np.float32)

        # for j, idx in enumerate(obs_task_idx):
        #     per_concept = concept_signal[:, idx] * float(np.abs(w_obs[idx]))
        #     s_obs[:, j] = alpha * per_concept + beta * shared_hid_score
        #     y_obs[:, j] = (s_obs[:, j] >= np.median(s_obs[:, j])).astype(np.float32)

        # # Final label matrix: (n, K+J)
        # y = np.concatenate([y_hid, y_obs], axis=1).astype(np.float32)
        
        
        
        # Shared terms (computed once)
        shared_obs_score = concept_signal @ w_obs      # observed context for hidden subtasks
        shared_hid_score = residual_signal @ w_hid     # hidden context for observed subtasks

        def standardize(z, eps=1e-8):
            return (z - z.mean()) / (z.std() + eps)

        if self.standardize:
            shared_obs_score = standardize(shared_obs_score)
            shared_hid_score = standardize(shared_hid_score)

        s_hid = np.zeros((self.n_samples, self.num_hid_tasks), dtype=np.float32)
        y_hid = np.zeros((self.n_samples, self.num_hid_tasks), dtype=np.float32)

        for k, idx in enumerate(hid_task_idx):
            hid_specific = residual_signal[:, idx] * float(np.abs(w_hid[idx]))

            if self.standardize:
                hid_specific = standardize(hid_specific)

            # alpha = main hidden-specific signal
            # beta  = auxiliary observed context
            s_hid[:, k] = self.alpha * hid_specific + self.beta * shared_obs_score
            y_hid[:, k] = (s_hid[:, k] >= np.median(s_hid[:, k])).astype(np.float32)

        s_obs = np.zeros((self.n_samples, self.num_obs_tasks), dtype=np.float32)
        y_obs = np.zeros((self.n_samples, self.num_obs_tasks), dtype=np.float32)

        for j, idx in enumerate(obs_task_idx):
            obs_specific = concept_signal[:, idx] * float(np.abs(w_obs[idx]))

            if self.standardize:
                obs_specific = standardize(obs_specific)

            # alpha = main observed-specific signal
            # beta  = auxiliary hidden context
            s_obs[:, j] = self.alpha * obs_specific + self.beta * shared_hid_score
            y_obs[:, j] = (s_obs[:, j] >= np.median(s_obs[:, j])).astype(np.float32)

        # Final label matrix: (n, K+J)
        y = np.concatenate([y_hid, y_obs], axis=1).astype(np.float32)
        
        
        
        
        
        
        
        

        # ----------------------------------------------------------------
        # 6. Store as tensors
        # ----------------------------------------------------------------
        self.x = torch.tensor(x, dtype=torch.float32)
        self.concepts = torch.tensor(concepts, dtype=torch.float32)
        self.residuals = torch.tensor(residuals, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.eta_concepts = torch.tensor(eta_concepts, dtype=torch.float32)
        self.eta_residuals = torch.tensor(eta_residuals, dtype=torch.float32)
        self.concept_signal = torch.tensor(concept_signal, dtype=torch.float32)
        self.residual_signal = torch.tensor(residual_signal, dtype=torch.float32)
        self.s_hid = torch.tensor(s_hid, dtype=torch.float32)
        self.s_obs = torch.tensor(s_obs, dtype=torch.float32)
        self.w_obs = torch.tensor(w_obs, dtype=torch.float32)
        self.w_hid = torch.tensor(w_hid, dtype=torch.float32)
        self.Sigma = torch.tensor(Sigma, dtype=torch.float32)
        self.hid_task_idx = torch.tensor(hid_task_idx, dtype=torch.long)
        self.obs_task_idx = torch.tensor(obs_task_idx, dtype=torch.long)

        # ----------------------------------------------------------------
        # 7. Apply split indices
        # ----------------------------------------------------------------
        if indices is not None:
            self.x = self.x[indices]
            self.concepts = self.concepts[indices]
            self.residuals = self.residuals[indices]
            self.y = self.y[indices]
            self.eta_concepts = self.eta_concepts[indices]
            self.eta_residuals = self.eta_residuals[indices]
            self.concept_signal = self.concept_signal[indices]
            self.residual_signal = self.residual_signal[indices]
            self.s_hid = self.s_hid[indices]
            self.s_obs = self.s_obs[indices]
            self.n_samples = self.x.shape[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "features": self.x[idx],
            "concepts": self.concepts[idx],
            "labels": self.y[idx],  # (K+J,) float32 multi-label vector
            "residuals": self.residuals[idx],
        }

    def label_counts(self):
        """Return per-task positive counts as a numpy array of shape (K+J,)."""
        return self.y.numpy().sum(axis=0)

    def describe(self):
        """Print a summary of the dataset configuration."""
        print("MultilabelSyntheticResidualDataset")
        print(f"  n_samples      : {self.n_samples}")
        print(f"  obs_dim        : {self.obs_dim}")
        print(f"  hid_dim        : {self.hid_dim}")
        print(f"  num_hid_tasks  : {self.num_hid_tasks}  (K)")
        print(f"  num_obs_tasks  : {self.num_obs_tasks}  (J)")
        print(f"  num_tasks/num_classes      : {self.num_classes}  (K+J, = output dim of y)")
        print(f"  alpha / beta   : {self.alpha} / {self.beta}")
        print(f"  hid_task_idx   : {self.hid_task_idx.tolist()}  "
              f"|w_hid|={self.w_hid.abs()[self.hid_task_idx].numpy().round(3)}")
        print(f"  obs_task_idx   : {self.obs_task_idx.tolist()}  "
              f"|w_obs|={self.w_obs.abs()[self.obs_task_idx].numpy().round(3)}")
        print(f"  label counts   : {self.label_counts().round(0).astype(int)}")
        print(f"  sigma_x        : {self.sigma_x}")
        print(f"  rho_cc/rr/cr   : {self.rho_cc} / {self.rho_rr} / {self.rho_cr}")


# ---------------------------------------------------------------------------
# Private helpers  (identical to multiclass_synthetic_dataset.py)
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

    W = rng.normal(size=(total_dim, latent_rank))
    Sigma_base = W @ W.T
    d = np.sqrt(np.diag(Sigma_base) + 1e-8)
    Sigma_base /= np.outer(d, d)

    Sigma = np.eye(total_dim)
    obs = slice(0, obs_dim)
    hid = slice(obs_dim, total_dim)

    obs_offdiag = Sigma_base[obs, obs].copy()
    np.fill_diagonal(obs_offdiag, 0.0)
    Sigma[obs, obs] = np.eye(obs_dim) + rho_cc * obs_offdiag

    hid_offdiag = Sigma_base[hid, hid].copy()
    np.fill_diagonal(hid_offdiag, 0.0)
    Sigma[hid, hid] = np.eye(obs_dim if obs_dim == hid_dim else hid_dim) + rho_rr * hid_offdiag

    Sigma[obs, hid] = rho_cr * Sigma_base[obs, hid]
    Sigma[hid, obs] = rho_cr * Sigma_base[hid, obs]

    Sigma = 0.5 * (Sigma + Sigma.T) + 0.1 * np.eye(total_dim)
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig <= 0:
        Sigma += (abs(min_eig) + 1e-3) * np.eye(total_dim)

    return Sigma.astype(np.float32)

# This version created weights that were too small
# def _make_sparse_weights(dim, sparsity, rng):
#     """Sparse task weight vector, normalised to unit norm."""
#     w = np.zeros(dim, dtype=np.float32)
#     n_active = max(1, int(sparsity * dim))
#     active_idx = rng.choice(dim, size=n_active, replace=False)
#     w[active_idx] = rng.normal(size=n_active).astype(np.float32)
#     w /= np.linalg.norm(w) + 1e-8
#     return w

def _make_sparse_weights(dim, sparsity, rng, min_weight_ratio=0.4):
    w = np.zeros(dim, dtype=np.float32)
    n_active = max(1, int(sparsity * dim))
    active_idx = rng.choice(dim, size=n_active, replace=False)

    # Sample raw magnitudes
    magnitudes = rng.uniform(low=0.5, high=1.0, size=n_active)

    # Enforce minimum ratio relative to the maximum
    max_mag = magnitudes.max()
    magnitudes = np.maximum(magnitudes, min_weight_ratio * max_mag)

    signs = rng.choice([-1.0, 1.0], size=n_active)
    w[active_idx] = (magnitudes * signs).astype(np.float32)
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

def get_multilabel_datasets(config, seed, log_file=None):
    """
    Create train / val / test splits of MultilabelSyntheticResidualDataset.

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
    config.data.num_hid_tasks        (int, >= 1)
    config.data.num_obs_tasks        (int, >= 1, default 1)
    config.data.alpha                (float, default 1.0)
    config.data.beta                 (float, default 1.0)
    config.data.rho_cc               (float, default 0.0)
    config.data.rho_rr               (float, default 0.0)
    config.data.rho_cr               (float, default 0.0)
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

    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")


    train_ds = MultilabelSyntheticResidualDataset(config, indices=idx_train)
    val_ds = MultilabelSyntheticResidualDataset(config, indices=idx_val)
    test_ds = MultilabelSyntheticResidualDataset(config, indices=idx_test)

    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"[multilabel synthetic dataset]\n")
            f.write(f"  n_samples        : {config.data.n_samples}\n")
            f.write(f"  obs_dim          : {config.data.obs_dim}\n")
            f.write(f"  hid_dim          : {config.data.hid_dim}\n")
            f.write(f"  num_hid_tasks    : {config.data.num_hid_tasks}\n")
            f.write(f"  num_obs_tasks    : {config.data.num_obs_tasks}\n")
            f.write(f"  num_classes       : {train_ds.num_classes}\n")
            f.write(f"  alpha / beta     : {config.data.alpha} / {config.data.beta}\n")
            f.write(f"  sigma_x          : {config.data.sigma_x}\n")
            f.write(
                f"  rho_cc/rr/cr     : {config.data.rho_cc} / "
                f"{config.data.rho_rr} / {config.data.rho_cr}\n"
            )
            f.write(f"  hid_task_idx     : {train_ds.hid_task_idx.tolist()}\n")
            f.write(f"  obs_task_idx     : {train_ds.obs_task_idx.tolist()}\n")
            f.write(
                f"  train/val/test   : {len(train_ds)} / "
                f"{len(val_ds)} / {len(test_ds)}\n"
            )

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_multilabel_data(config, train, val, test, log_file):
    """
    Save a multi-label dataset split to disk.

    Directory structure
    -------------------
    <data_path>/<dataset>/
        <save_name>/
            train/   x.pt, concepts.pt, residuals.pt, y.pt,
                     eta_concepts.pt, eta_residuals.pt,
                     concept_signal.pt, residual_signal.pt,
                     s_hid.pt, s_obs.pt,
                     w_obs.pt, w_hid.pt, Sigma.pt,
                     hid_task_idx.pt, obs_task_idx.pt
            val/     (same)
            test/    (same)
            info.txt
    """
    root = os.path.join(config.data.data_path, config.data.dataset)
    os.makedirs(root, exist_ok=True)

    hostname = os.uname()[1]
    prefix = "cluster_" if "biomed" in hostname else "local_"

    save_name = (
        prefix
        + f"hid_tasks_{train.num_hid_tasks}"
        + f"_obs_tasks_{train.num_obs_tasks}"
        + f"_alpha_{train.alpha}"
        + f"_beta_{train.beta}"
        + f"_rho_cr{train.rho_cr}"
        + f"_rho_cc{train.rho_cc}"
        + f"_rho_rr{train.rho_rr}"
        + f"_r_sparsity_{train.task_sparsity_hid}"
        + f"_c_sparsity_{train.task_sparsity_obs}"
        + f"_sigmax_{train.sigma_x}"
        + f"_hid_dim_{train.hid_dim}"
        + f"_obs_dim_{train.obs_dim}"
        + f"_w_ratio_{train.min_weight_ratio}"
        + f"_standardize_{train.standardize}"
        + f"_seed_{config.seed}"
    )

    existing = [d for d in os.listdir(root) if d.startswith(save_name)]
    if existing:
        version = 1
        while os.path.exists(os.path.join(root, f"{save_name}_v{version}")):
            version += 1
        save_name = f"{save_name}_v{version}"

    save_dir = os.path.join(root, save_name)
    os.makedirs(save_dir, exist_ok=True)

    tensor_fields = [
        "x", "concepts", "residuals", "y",
        "eta_concepts", "eta_residuals",
        "concept_signal", "residual_signal",
        "s_hid", "s_obs",
        "w_obs", "w_hid", "Sigma",
        "hid_task_idx", "obs_task_idx",
    ]

    for split_name, dataset in [("train", train), ("val", val), ("test", test)]:
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for field in tensor_fields:
            torch.save(
                getattr(dataset, field),
                os.path.join(split_dir, f"{field}.pt"),
            )

    info_path = os.path.join(save_dir, "info.txt")
    log_parent = os.path.dirname(log_file)
    with open(info_path, "w") as f:
        f.write("[multilabel synthetic dataset]\n")
        f.write(f"num_hid_tasks : {train.num_hid_tasks}\n")
        f.write(f"num_obs_tasks : {train.num_obs_tasks}\n")
        f.write(f"num_tasks/num_classes : {train.num_classes}\n")
        f.write(f"alpha : {train.alpha}\n")
        f.write(f"beta : {train.beta}\n")
        f.write(f"hid_task_idx : {train.hid_task_idx.tolist()}\n")
        f.write(f"obs_task_idx : {train.obs_task_idx.tolist()}\n")
        f.write(f"rho_cr : {train.rho_cr}\n")
        f.write(f"rho_cc : {train.rho_cc}\n")
        f.write(f"rho_rr : {train.rho_rr}\n")
        f.write(f"task_sparsity_hid : {train.task_sparsity_hid}\n")
        f.write(f"task_sparsity_obs : {train.task_sparsity_obs}\n")
        f.write(f"sigma_x : {train.sigma_x}\n")
        f.write(f"hid_dim : {train.hid_dim}\n")
        f.write(f"obs_dim : {train.obs_dim}\n")
        f.write(f"seed : {config.seed}\n")
        f.write(f"train / val / test : {len(train)} / {len(val)} / {len(test)}\n")
        f.write(f"min_weight_ratio : {train.min_weight_ratio}\n")
        f.write(f"data created for model at: {log_parent}\n")
        f.write(f"standardize : {train.standardize}\n")

    with open(log_file, "a") as f:
        f.write(f"data_dir: {save_dir}\n")

    print(f"Saved multilabel dataset to: {save_dir}")
    return save_dir


class LoadedMultilabelDataset(Dataset):
    """
    Lightweight wrapper around tensors loaded from disk.
    Drop-in replacement for MultilabelSyntheticResidualDataset in training code.
    """

    def __init__(self, split_dir):
        def _load(name):
            return torch.load(os.path.join(split_dir, f"{name}.pt"))

        self.x = _load("x")
        self.concepts = _load("concepts")
        self.residuals = _load("residuals")
        self.y = _load("y")
        self.eta_concepts = _load("eta_concepts")
        self.eta_residuals = _load("eta_residuals")
        self.concept_signal = _load("concept_signal")
        self.residual_signal = _load("residual_signal")
        self.s_hid = _load("s_hid")
        self.s_obs = _load("s_obs")
        self.w_obs = _load("w_obs")
        self.w_hid = _load("w_hid")
        self.Sigma = _load("Sigma")
        self.hid_task_idx = _load("hid_task_idx")
        self.obs_task_idx = _load("obs_task_idx")
        self.n_samples = self.x.shape[0]
        self.num_classes = self.y.shape[1]
        self.num_hid_tasks = self.hid_task_idx.shape[0]
        self.num_obs_tasks = self.obs_task_idx.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "features": self.x[idx],
            "concepts": self.concepts[idx],
            "labels": self.y[idx],
            "residuals": self.residuals[idx],
        }


def load_saved_multilabel_data(config):
    """
    Load a previously saved multilabel dataset from disk.

    Parameters
    ----------
    config : DictConfig with config.data.data_path, config.data.dataset,
             config.data.data_dir_name

    Returns
    -------
    train, val, test : LoadedMultilabelDataset
    """
    data_dir_root = os.path.join(config.data.data_path, config.data.dataset)
    full_data_dir_path = os.path.join(data_dir_root, config.data.data_dir_name)

    train = LoadedMultilabelDataset(os.path.join(full_data_dir_path, "train"))
    val = LoadedMultilabelDataset(os.path.join(full_data_dir_path, "val"))
    test = LoadedMultilabelDataset(os.path.join(full_data_dir_path, "test"))
    return train, val, test


def check_multilabel_dataset(config):
    """
    Populate config fields derived from the dataset structure.
    Call this after loading config, before building the model.

    Sets:
        config.data.num_concepts  — number of supervised concept dimensions
        config.data.num_classes     — number of output heads (K + J)
    """
    if config.data.data_dir_name is None:
        num_hid_tasks = config.data.num_hid_tasks
        num_obs_tasks = getattr(config.data, "num_obs_tasks", 1)
        config.data.num_classes = num_hid_tasks + num_obs_tasks
        config.data.num_concepts = config.data.obs_dim
    else:
        train_data, _, _ = load_saved_multilabel_data(config)
        config.data.num_concepts = train_data.concepts.shape[1]
        config.data.num_classes = train_data.y.shape[1]
