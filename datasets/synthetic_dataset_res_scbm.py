import itertools

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import random
import os
from collections import defaultdict

class SyntheticResidualSCBMDataset(Dataset):
    """
    Synthetic dataset for testing Residual SCBM behaviour.

    Each sample contains:
        x        : observed input features
        c_obs    : observed concept labels given to the model
        y        : binary task label
        c_hid    : hidden concepts, not used as concept supervision
        r_true   : true residual factors, useful only for analysis
        s        : continuous task score before thresholding

    The data-generation process is:

        eta = [eta_obs, eta_hid] ~ N(0, Sigma)

        c_obs = 1[eta_obs >= 0]
        c_hid = 1[eta_hid >= 0]

        x = h(eta) + noise

        s = alpha * w_obs^T c_obs + beta * w_hid^T c_hid
            + gamma * c_obs^T A c_hid

        y = 1[s >= median(s)]
    """

    def __init__(
        self,
        n_samples,
        num_covariates,
        obs_dim,
        hid_dim,
        latent_rank,
        alpha,
        beta,
        rho_cr,
        rho_cc,
        rho_rr,
        sigma_x,
        task_sparsity_obs,
        task_sparsity_hid,
        seed,
        dataset_difficulty,
        indices=None
    ):
        super().__init__()

        self.n_samples = n_samples
        self.num_covariates = num_covariates
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.total_dim = self.obs_dim + self.hid_dim
        self.alpha = alpha
        self.beta = beta
        self.rho_cr = rho_cr
        self.rho_cc = rho_cc
        self.rho_rr = rho_rr
        self.sigma_x = sigma_x
        self.seed = seed
        
        self.task_sparsity_obs = task_sparsity_obs
        self.task_sparsity_hid = task_sparsity_hid
        self.latent_rank = latent_rank
        
        self.indices = indices

        # Random number generator for reproducibility
        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------
        # 1. Build covariance matrix for latent concept logits
        # ------------------------------------------------------------
        Sigma = self._create_covariance(
            c_obs_dim=self.obs_dim,
            c_hid_dim=self.hid_dim,
            latent_rank=latent_rank,
            rho_cr=rho_cr,
            rho_cc=rho_cc,
            rho_rr=rho_rr,
            rng=rng,
        )

        # ------------------------------------------------------------
        # 2. Sample latent variables eta = [eta_obs, eta_hid]
        # ------------------------------------------------------------
        eta = rng.multivariate_normal(
            mean=np.zeros(self.total_dim),
            cov=Sigma,
            size=n_samples,
        )

        eta_concepts = eta[:, :self.obs_dim]
        eta_residuals = eta[:, self.obs_dim:]
        
        # Binary concepts
        concepts = (eta_concepts >= 0).astype(np.float32)
        residuals = (eta_residuals >= 0).astype(np.float32)

        # Continuous concept intensities / saliencies
        concept_strengths = np.abs(eta_concepts)
        residual_strengths = np.abs(eta_residuals)

        # Effective concept values
        concept_signal = concepts * concept_strengths 
        residual_signal = residuals * residual_strengths

        # x is generated from concept presence + strength
        if dataset_difficulty == "easy":
            eta_for_x = np.concatenate([concepts, residuals], axis=1)
        elif dataset_difficulty == "hard" or dataset_difficulty == "medium":
            eta_for_x = np.concatenate([concept_signal, residual_signal], axis=1)

        x = self._random_mlp_features(
            eta=eta_for_x,
            num_covariates=num_covariates,
            rng=rng,
            sigma_x=sigma_x,
        )

        # Global sparse task weights
        w_obs = self._make_sparse_weights(self.obs_dim, task_sparsity_obs, rng)
        w_hid = self._make_sparse_weights(self.hid_dim, task_sparsity_hid, rng)

        # Label depends on expressed concept strength of concepts that are present
        if dataset_difficulty == "easy" or dataset_difficulty == "medium":
            s_concepts = concepts @ w_obs
            s_residuals = residuals @ w_hid
        elif dataset_difficulty == "hard":
            s_concepts = concept_signal @ w_obs
            s_residuals = residual_signal @ w_hid

        s = alpha * s_concepts + beta * s_residuals

        # ------------------------------------------------------------
        # 6. Convert continuous score to balanced binary label
        # ------------------------------------------------------------
        threshold = np.median(s)
        y = (s >= threshold).astype(np.int64)

        # ------------------------------------------------------------
        # Store everything
        # ------------------------------------------------------------ 
        self.x = torch.tensor(x, dtype=torch.float32)
        self.concepts = torch.tensor(concepts, dtype=torch.float32)
        self.residuals = torch.tensor(residuals, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.s = torch.tensor(s, dtype=torch.float32)
        
        # Useful metadata
        self.Sigma = torch.tensor(Sigma, dtype=torch.float32)
        self.w_obs = torch.tensor(w_obs, dtype=torch.float32)
        self.w_hid = torch.tensor(w_hid, dtype=torch.float32)
        self.threshold = float(threshold)
        self.eta_concepts = torch.tensor(eta_concepts, dtype=torch.float32)
        self.eta_residuals = torch.tensor(eta_residuals, dtype=torch.float32)
        self.concept_signal = torch.tensor(concept_signal, dtype=torch.float32)
        self.residual_signal = torch.tensor(residual_signal, dtype=torch.float32)
        
        
        
        if indices is not None:
            self.x = self.x[indices]
            self.concepts = self.concepts[indices]
            self.residuals = self.residuals[indices]
            self.y = self.y[indices]
            self.s = self.s[indices]
            self.eta_concepts = self.eta_concepts[indices]
            self.eta_residuals = self.eta_residuals[indices]
            self.concept_signal = self.concept_signal[indices]
            self.residual_signal = self.residual_signal[indices]
            self.w_obs = self.w_obs
            self.w_hid = self.w_hid
            self.n_samples = self.x.shape[0]

            
            


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Currently only features, concepts and labels used
        return {
            "features": self.x[idx],
            "concepts": self.concepts[idx],
            "labels": self.y[idx],
            "residuals": self.residuals[idx],
            "score": self.s[idx],
        }



    @staticmethod
    def _make_sparse_weights(dim, sparsity, rng):
        """
        Creates global sparse task weights.

        Returns:
            w: shape (dim,)

        sparsity = fraction of dimensions that are globally task-relevant.
        """
        w = np.zeros(dim, dtype=np.float32)

        n_active = max(1, int(sparsity * dim))
        active_idx = rng.choice(dim, size=n_active, replace=False)

        w[active_idx] = rng.normal(size=n_active)

        # Normalize so score magnitudes are stable across dimensions
        w = w / (np.linalg.norm(w) + 1e-8)

        return w.astype(np.float32)

    @staticmethod
    def _create_covariance(
        c_obs_dim,
        c_hid_dim,
        latent_rank,
        rho_cr,
        rho_cc,
        rho_rr,
        rng,
    ):
        """
        Creates a positive definite covariance matrix:

            Sigma = [[Sigma_obs_obs, Sigma_obs_hid],
                    [Sigma_hid_obs, Sigma_hid_hid]]

        where:
            rho_cc controls observed-observed / concept-concept covariance
            rho_rr controls hidden-hidden / residual-residual covariance
            rho_cr controls observed-hidden / concept-residual covariance
        """

        total_dim = c_obs_dim + c_hid_dim
        
        if latent_rank == 0:
            Sigma = np.eye(total_dim)

            # Add jitter for consistency with the rest of the function
            Sigma = Sigma + 0.1 * np.eye(total_dim)

            return Sigma.astype(np.float32)
        
        
        

        # Low-rank structure creates a base correlation matrix
        W = rng.normal(size=(total_dim, latent_rank))
        Sigma_base = W @ W.T

        # Normalize to correlation-like scale
        d = np.sqrt(np.diag(Sigma_base) + 1e-8)
        Sigma_base = Sigma_base / np.outer(d, d)
        
        # np.outer creates a matrix where entry (i,j) is d[i] * d[j]

        # Start with identity so every variable has variance 1
        Sigma = np.eye(total_dim)

        obs = slice(0, c_obs_dim)
        hid = slice(c_obs_dim, total_dim)
        

            

        # --------------------------------------------------
        # Observed concept block: concept-concept covariance
        # --------------------------------------------------
        Sigma_obs = Sigma_base[obs, obs].copy()

        # Keep diagonal equal to 1, scale only off-diagonal entries
        Sigma_obs_offdiag = Sigma_obs.copy()
        np.fill_diagonal(Sigma_obs_offdiag, 0.0)

        Sigma[obs, obs] = np.eye(c_obs_dim) + rho_cc * Sigma_obs_offdiag


        # --------------------------------------------------
        # Hidden residual block: residual-residual covariance
        # --------------------------------------------------
        Sigma_hid = Sigma_base[hid, hid].copy()

        # Keep diagonal equal to 1, scale only off-diagonal entries
        Sigma_hid_offdiag = Sigma_hid.copy()
        np.fill_diagonal(Sigma_hid_offdiag, 0.0)

        Sigma[hid, hid] = np.eye(c_hid_dim) + rho_rr * Sigma_hid_offdiag

        # --------------------------------------------------
        # Cross block: concept-residual covariance
        # --------------------------------------------------
        Sigma[obs, hid] = rho_cr * Sigma_base[obs, hid]
        Sigma[hid, obs] = rho_cr * Sigma_base[hid, obs]

        # Symmetrize to remove tiny numerical asymmetries, it should be symmetrical
        # However because of floating point operations, it might not be perfectly symmetrical
        Sigma = 0.5 * (Sigma + Sigma.T)

        # Add diagonal jitter for numerical stability
        Sigma = Sigma + 0.1 * np.eye(total_dim)

        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        if min_eig <= 0:
            Sigma = Sigma + (abs(min_eig) + 1e-3) * np.eye(total_dim)

        return Sigma.astype(np.float32)

    @staticmethod
    def _random_mlp_features(eta, num_covariates, rng, sigma_x):
        """
        Fixed random MLP h(eta) used to generate observed inputs x.

        This makes x a nonlinear function of the true latent factors.
        """

        n_samples, latent_dim = eta.shape
        hidden_dim = min(512, max(128, 2 * latent_dim))

        W1 = rng.normal(size=(latent_dim, hidden_dim)) / np.sqrt(latent_dim)
        b1 = rng.normal(size=(hidden_dim,)) * 0.1

        W2 = rng.normal(size=(hidden_dim, num_covariates)) / np.sqrt(hidden_dim)
        b2 = rng.normal(size=(num_covariates,)) * 0.1

        h = np.tanh(eta @ W1 + b1)
        x = h @ W2 + b2
        x = x + rng.normal(scale=sigma_x, size=x.shape)

        return x
    
    
    
    
    
    

        



class LoadedSyntheticResidualSCBMDataset(Dataset):
    def __init__(self, x, concepts, residuals, y, s):
        self.x = x
        self.concepts = concepts
        self.residuals = residuals
        self.y = y
        self.s = s
        self.n_samples = x.shape[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "features": self.x[idx],
            "concepts": self.concepts[idx],
            "labels": self.y[idx],
            "residuals": self.residuals[idx],
            "score": self.s[idx],
        }


def _load_split_dataset(split_dir):
    x = torch.load(os.path.join(split_dir, "x.pt"))
    concepts = torch.load(os.path.join(split_dir, "concepts.pt"))
    residuals = torch.load(os.path.join(split_dir, "residuals.pt"))
    y = torch.load(os.path.join(split_dir, "y.pt"))
    s = torch.load(os.path.join(split_dir, "s.pt"))
    return LoadedSyntheticResidualSCBMDataset(x, concepts, residuals, y, s)


def load_saved_synthetic_data(config):
    data_dir_root = os.path.join(config.data.data_path, "synthetic_res_scbm")
    full_data_dir_path = os.path.join(data_dir_root, config.data.data_dir_name)
    train = _load_split_dataset(os.path.join(full_data_dir_path, "train"))
    val = _load_split_dataset(os.path.join(full_data_dir_path, "val"))
    test = _load_split_dataset(os.path.join(full_data_dir_path, "test"))
    return train, val, test


    
    
    
    
    
    
    
def get_synthetic_datasets_res_scbm(config, seed=0, log_file=None):
    # Train-validation-test split
    indices_train, indices_valtest = train_test_split(
        np.arange(0, config.data.n_samples), train_size=config.data.train_ratio, random_state=seed
    )
    indices_val, indices_test = train_test_split(
        indices_valtest,
        train_size=config.data.val_ratio / (1.0 - config.data.train_ratio),
        random_state=2 * seed,
    )
    
    print(f"Train samples: {len(indices_train)}, Val samples: {len(indices_val)}, Test samples: {len(indices_test)}")
    print(f"Train ratio: {len(indices_train)/config.data.n_samples:.2f}, Val ratio: {len(indices_val)/config.data.n_samples:.2f}, Test ratio: {len(indices_test)/config.data.n_samples:.2f}")
    
    
    
    train_dataset = SyntheticResidualSCBMDataset(
        n_samples=config.data.n_samples,
        indices = indices_train,
        num_covariates=config.data.num_covariates,
        obs_dim=config.data.obs_dim,
        hid_dim=config.data.hid_dim,
        latent_rank=config.data.latent_rank,
        alpha=config.data.alpha,
        beta=config.data.beta,
        rho_cr=config.data.rho_cr,
        rho_cc=config.data.rho_cc,
        rho_rr=config.data.rho_rr,
        sigma_x=config.data.sigma_x,
        task_sparsity_obs=config.data.task_sparsity_obs,
        task_sparsity_hid=config.data.task_sparsity_hid,
        dataset_difficulty=config.data.experiment_type,
        seed=seed,
    )
    
    valid_dataset = SyntheticResidualSCBMDataset(
        n_samples=config.data.n_samples,
        indices = indices_val,
        num_covariates=config.data.num_covariates,
        obs_dim=config.data.obs_dim,
        hid_dim=config.data.hid_dim,
        latent_rank=config.data.latent_rank,    
        alpha=config.data.alpha,
        beta=config.data.beta,
        rho_cr=config.data.rho_cr,
        rho_cc=config.data.rho_cc,
        rho_rr=config.data.rho_rr,
        sigma_x=config.data.sigma_x,
        task_sparsity_obs=config.data.task_sparsity_obs,
        task_sparsity_hid=config.data.task_sparsity_hid,
        dataset_difficulty=config.data.experiment_type,
        seed=seed,  
    )
    
    test_dataset = SyntheticResidualSCBMDataset(
        n_samples=config.data.n_samples,
        indices = indices_test,
        num_covariates=config.data.num_covariates,
        obs_dim=config.data.obs_dim,
        hid_dim=config.data.hid_dim,
        latent_rank=config.data.latent_rank,
        alpha=config.data.alpha,
        beta=config.data.beta,
        rho_cr=config.data.rho_cr,
        rho_cc=config.data.rho_cc,
        rho_rr=config.data.rho_rr,
        sigma_x=config.data.sigma_x,
        task_sparsity_obs=config.data.task_sparsity_obs,
        task_sparsity_hid=config.data.task_sparsity_hid,
        dataset_difficulty=config.data.experiment_type,
        seed=seed, 
    )
    
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"rho_cr (correlation between linked concepts and residuals): {train_dataset.rho_cr}\n")
            f.write(f"rho_cc (within-block correlation for observed concepts): {train_dataset.rho_cc}\n")
            f.write(f"rho_rr (within-block correlation for hidden concepts): {train_dataset.rho_rr}\n")
            f.write(f"alpha: {train_dataset.alpha}, beta: {train_dataset.beta}\n")
            f.write(f"Task sparsity for observed concepts: {train_dataset.task_sparsity_obs}, Task sparsity for hidden concepts: {train_dataset.task_sparsity_hid}\n")
            f.write(f"Sigma_x (noise level in x): {train_dataset.sigma_x}\n")
            f.write(f"num_concepts: {train_dataset.concepts.shape[1]}, num_residuals: {train_dataset.residuals.shape[1]}\n")
        

    return train_dataset, valid_dataset, test_dataset
        



class LoadedSyntheticResidualSCBMDataset(Dataset):
    def __init__(self, x, concepts, residuals, y, s):
        self.x = x
        self.concepts = concepts
        self.residuals = residuals
        self.y = y
        self.s = s
        self.n_samples = x.shape[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "features": self.x[idx],
            "concepts": self.concepts[idx],
            "labels": self.y[idx],
            "residuals": self.residuals[idx],
            "score": self.s[idx],
        }


def load_split_dataset(split_dir):
    x = torch.load(os.path.join(split_dir, "x.pt"))
    concepts = torch.load(os.path.join(split_dir, "concepts.pt"))
    residuals = torch.load(os.path.join(split_dir, "residuals.pt"))
    y = torch.load(os.path.join(split_dir, "y.pt"))
    s = torch.load(os.path.join(split_dir, "s.pt"))
    return LoadedSyntheticResidualSCBMDataset(x, concepts, residuals, y, s)


def load_saved_synthetic_data(config):
    data_dir_root = os.path.join(config.data.data_path, "synthetic_res_scbm")
    #full_data_dir_path = os.path.join(data_dir_root, config.data.data_dir_name)
    full_data_dir_path = os.path.join(data_dir_root, config.data.experiment_type, config.data.data_dir_name)
    train = load_split_dataset(os.path.join(full_data_dir_path, "train"))
    val = load_split_dataset(os.path.join(full_data_dir_path, "val"))
    test = load_split_dataset(os.path.join(full_data_dir_path, "test"))
    return train, val, test