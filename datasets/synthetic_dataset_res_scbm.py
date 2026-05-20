import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import random
import os

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
        n_samples=50_000,
        indices=None,
        num_covariates=1500,
        obs_dim=75,
        hid_dim=25,
        latent_rank=10,
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
        rho_cr=0.3,
        sigma_x=1.0,
        task_sparsity_concepts=0.3,
        task_sparsity_residuals=0.3,
        seed=0,
    ):
        super().__init__()

        self.n_samples = n_samples
        self.num_covariates = num_covariates
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.total_dim = self.obs_dim + self.hid_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho_cr = rho_cr
        self.sigma_x = sigma_x
        self.seed = seed
        
        self.task_sparsity_concepts = task_sparsity_concepts
        self.task_sparsity_residuals = task_sparsity_residuals
        self.latent_rank = latent_rank
        
        
        self.indices = indices

        # Random number generator for reproducibility
        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------
        # 1. Build covariance matrix for latent concept logits
        # ------------------------------------------------------------
        # Sigma = self._make_block_covariance(
        #     c_obs_dim=self.obs_dim,
        #     c_hid_dim=self.hid_dim,
        #     latent_rank=latent_rank,
        #     rho_cr=rho_cr,
        #     rng=rng,
        # )

        # ------------------------------------------------------------
        # 2. Sample latent variables eta = [eta_obs, eta_hid]
        # ------------------------------------------------------------
        # eta = rng.multivariate_normal(
        #     mean=np.zeros(self.total_dim),
        #     cov=Sigma,
        #     size=n_samples,
        # )

        # eta_concepts = eta[:, :self.obs_dim]
        # eta_residuals = eta[:, self.obs_dim:]
        
        
        # ------------------------------------------------------------
        # 2.1 Alternative sampling method with clearer dependency structure between concepts and residuals 
        # ------------------------------------------------------------
        # eta_residuals = rho * eta_concepts + sqrt(1 - rho^2) * epsilon, where epsilon is independent noise
        eta_concepts = rng.normal(size=(n_samples, self.obs_dim))
        epsilon = rng.normal(size=(n_samples, self.hid_dim))

        eta_residuals = np.zeros((n_samples, self.hid_dim))

        # residual j is linked to concept i
        num_pairs = int(min(self.obs_dim, self.hid_dim)/2)
        concept_indices = rng.choice(self.obs_dim, size=num_pairs, replace=False)
        residual_indices = rng.choice(self.hid_dim, size=num_pairs, replace=False)

        concept_to_residual_pairs = list(zip(concept_indices.tolist(), residual_indices.tolist()))

        for i, j in concept_to_residual_pairs:
            eta_residuals[:, j] = (
                self.rho_cr * eta_concepts[:, i]
                + np.sqrt(1 - self.rho_cr**2) * epsilon[:, j]
            )
            
        eta = np.concatenate([eta_concepts, eta_residuals], axis=1)
        # ------------------------------------------------------------
        # 3. Threshold latent logits into binary concepts
        # ------------------------------------------------------`-`-----
        concepts = (eta_concepts >= 0).astype(np.float32)
        residuals = (eta_residuals >= 0).astype(np.float32)

        # ------------------------------------------------------------
        # 4. Generate observed input x from all latent factors
        # ------------------------------------------------------------
        x = self._random_mlp_features(
            eta=eta,
            num_covariates=num_covariates,
            rng=rng,
        )
        # Add noise to x to make the task more challenging and prevent perfect fitting
        x = x + sigma_x * rng.normal(size=x.shape)
        x = x.astype(np.float32)

        # Standardize x for easier training
        x = (x - x.mean(axis=0, keepdims=True)) / (
            x.std(axis=0, keepdims=True) + 1e-8
        )

        # ------------------------------------------------------------
        # 5. Generate task score s
        # ------------------------------------------------------------
        
        # Create sparse random task weights for observed and hidden concepts
        w_concepts = self._make_sparse_weights(
            dim=self.obs_dim,
            sparsity=task_sparsity_concepts,
            rng=rng,
        )

        w_residuals = self._make_sparse_weights(
            dim=self.hid_dim,
            sparsity=task_sparsity_residuals,
            rng=rng,
        )
        

        
        
        # Compute score components from observed and hidden concepts (residual factors)
        s_concepts = concepts @ w_concepts
        s_residuals = residuals @ w_residuals

        # Compute interaction term if gamma > 0
        if gamma != 0.0:
            A = rng.normal(size=(self.obs_dim, self.hid_dim))
            A = A / np.sqrt(self.obs_dim * self.hid_dim)
            s_interaction = np.sum((concepts @ A) * residuals, axis=1)
        else:
            A = np.zeros((self.obs_dim, self.hid_dim))
            s_interaction = np.zeros(n_samples)

        s = alpha * s_concepts + beta * s_residuals + gamma * s_interaction

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
        #self.Sigma = torch.tensor(Sigma, dtype=torch.float32)
        self.w_obs = torch.tensor(w_concepts, dtype=torch.float32)
        self.w_hid = torch.tensor(w_residuals, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.threshold = float(threshold)
        self.concepts_linked_to_residuals = concept_to_residual_pairs
        
        
        
        if indices is not None:
            self.x = self.x[indices]
            self.concepts = self.concepts[indices]
            self.residuals = self.residuals[indices]
            self.y = self.y[indices]
            self.s = self.s[indices]
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
        Creates sparse task weights.

        sparsity = fraction of dimensions that are task-relevant.
        """
        w = np.zeros(dim, dtype=np.float32)

        n_active = max(1, int(sparsity * dim))
        # list of indices to be active
        active_idx = rng.choice(dim, size=n_active, replace=False)

        w[active_idx] = rng.normal(size=n_active)

        # Normalize so score magnitudes are stable across dimensions
        w = w / (np.linalg.norm(w) + 1e-8)

        return w.astype(np.float32)

    @staticmethod
    def _make_block_covariance(
        c_obs_dim,
        c_hid_dim,
        latent_rank,
        rho_cr,
        rng,
    ):
        """
        Creates a positive definite covariance matrix:

            Sigma = [[Sigma_obs_obs, Sigma_obs_hid],
                     [Sigma_hid_obs, Sigma_hid_hid]]

        rho_cr controls the strength of observed-hidden correlation.
        """

        total_dim = c_obs_dim + c_hid_dim

        # Low-rank structure creates correlated concepts
        # Latent rank
        # W = [total_dim, latent_rank] with latent_rank << total_dim 
        W = rng.normal(size=(total_dim, latent_rank))
        Sigma_base = W @ W.T

        # Normalize to correlation-like scale
        d = np.sqrt(np.diag(Sigma_base) + 1e-8)
        Sigma_base = Sigma_base / np.outer(d, d)

        # Separate blocks
        Sigma = np.eye(total_dim)

        obs = slice(0, c_obs_dim)
        hid = slice(c_obs_dim, total_dim)

        # Within-block correlations
        Sigma[obs, obs] = Sigma_base[obs, obs]
        Sigma[hid, hid] = Sigma_base[hid, hid]

        # Cross-block correlations controlled by rho_cr
        Sigma[obs, hid] = rho_cr * Sigma_base[obs, hid]
        Sigma[hid, obs] = rho_cr * Sigma_base[hid, obs]

        # Add diagonal jitter for numerical stability
        Sigma = Sigma + 0.1 * np.eye(total_dim)

        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        if min_eig <= 0:
            Sigma = Sigma + (abs(min_eig) + 1e-3) * np.eye(total_dim)

        return Sigma.astype(np.float32)

    @staticmethod
    def _random_mlp_features(eta, num_covariates, rng):
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

        return x
    
    
    
    
    
    
    
    
def get_synthetic_datasets_res_scbm(config, seed=0, log_file=None):
    # This is the original one

    
    
    
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
        gamma=config.data.gamma,
        rho_cr=config.data.rho_cr,
        sigma_x=config.data.sigma_x,
        task_sparsity_concepts=config.data.task_sparsity_concepts,
        task_sparsity_residuals=config.data.task_sparsity_residuals,
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
        gamma=config.data.gamma,
        rho_cr=config.data.rho_cr,
        sigma_x=config.data.sigma_x,
        task_sparsity_concepts=config.data.task_sparsity_concepts,
        task_sparsity_residuals=config.data.task_sparsity_residuals,
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
        gamma=config.data.gamma,
        rho_cr=config.data.rho_cr,
        sigma_x=config.data.sigma_x,
        task_sparsity_concepts=config.data.task_sparsity_concepts,
        task_sparsity_residuals=config.data.task_sparsity_residuals,
        seed=seed,  
    )
    
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"Concepts linked to residuals (concept index, residual index): {train_dataset.concepts_linked_to_residuals}\n")
            f.write(f"Task weight s for concepts: {train_dataset.w_obs}\n")
            f.write(f"Task weight s for residuals: {train_dataset.w_hid}\n")
            f.write(f"rho_cr (correlation between linked concepts and residuals): {train_dataset.rho_cr}\n")
            f.write(f"alpha: {train_dataset.alpha}, beta: {train_dataset.beta}, gamma: {train_dataset.gamma}\n")
            
    
    # Save the generated dataset for reproducibility and analysis
    #data_dir_name = save_synthetic_data(config, train_dataset, valid_dataset, test_dataset, log_file=log_file)
    
    
    
    #return train_dataset, valid_dataset, test_dataset, data_dir_name
    return train_dataset, valid_dataset, test_dataset
        
"""
def save_synthetic_data(config, train, val, test):
    synthetic_data_dir = os.path.join(config.data.data_path, "synthetic_res_scbm")
    os.makedirs(synthetic_data_dir, exist_ok=True)
    num = 0
    for dir_name in os.listdir(synthetic_data_dir):
        if dir_name.startswith(f"synthetic_data_seed_{config.seed}"):
            num = max(num, int(dir_name.split("_")[-1].split(".")[0]) + 1)
    
    # Save directory to save train val test data
    data_dir_name = f"synthetic_data_seed_{config.seed}_{num}"
    os.makedirs(os.path.join(synthetic_data_dir, data_dir_name), exist_ok=True)
    
    
    torch.save(train, os.path.join(synthetic_data_dir, data_dir_name, "train.pt"))
    torch.save(val, os.path.join(synthetic_data_dir, data_dir_name, "val.pt"))
    torch.save(test, os.path.join(synthetic_data_dir, data_dir_name, "test.pt"))
    
    
    # Create info file with dataset parameters
    info_file_path = os.path.join(synthetic_data_dir, data_dir_name, "info.txt")
    with open(info_file_path, "w") as f:
        f.write(f"num_samples: {config.data.n_samples}\n")
        f.write(f"num_covariates: {config.data.num_covariates}\n")
        f.write(f"obs_dim: {config.data.obs_dim}\n")
        f.write(f"hid_dim: {config.data.hid_dim}\n")
        f.write(f"latent_rank: {config.data.latent_rank}\n")
        f.write(f"alpha: {config.data.alpha}\n")
        f.write(f"beta: {config.data.beta}\n")
        f.write(f"gamma: {config.data.gamma}\n")
        f.write(f"rho_cr: {config.data.rho_cr}\n")
        f.write(f"sigma_x: {config.data.sigma_x}\n")
        f.write(f"task_sparsity_concepts: {config.data.task_sparsity_concepts}\n")
        f.write(f"task_sparsity_residuals: {config.data.task_sparsity_residuals}\n")
        f.write(f"seed: {config.seed}\n")
        f.write(f"concepts_linked_to_residuals: {train.concepts_linked_to_residuals}\n")
        
        

    return data_dir_name


def create_synthetic_datasets_res_scbm(config, seed=0):
    print("Creating new synthetic dataset")
    train_dataset, valid_dataset, test_dataset = get_synthetic_datasets_res_scbm(config, seed=seed)
    data_dir_name = save_synthetic_data(config, train_dataset, valid_dataset, test_dataset)
    config.data.data_dir_name = data_dir_name


def update_config_data_properties_from_dataset(config, train, val, test, data_dir_name):
    config.data.obs_dim = train.obs_dim
    config.data.hid_dim = train.hid_dim
    config.data.num_covariates = train.num_covariates
    config.data.n_samples = train.n_samples + val.n_samples + test.n_samples
    config.data.train_ratio = train.n_samples / config.data.n_samples
    config.data.val_ratio = val.n_samples / config.data.n_samples
    config.data.alpha = train.alpha
    config.data.beta = train.beta
    config.data.gamma = train.gamma
    config.data.rho_cr = train.rho_cr
    config.data.sigma_x = train.sigma_x
    config.data.task_sparsity_concepts = train.task_sparsity_concepts
    config.data.task_sparsity_residuals = train.task_sparsity_residuals
    config.data.latent_rank = train.latent_rank
    config.data.data_dir_name = data_dir_name







def load_saved_synthetic_data(config):

    synthetic_data_dir = os.path.join(config.data.data_path, "synthetic_res_scbm", config.data.data_dir_name)
    
    if config.data.data_dir_name is not None and not os.path.exists(synthetic_data_dir):
        raise FileNotFoundError(f"Data directory {synthetic_data_dir} does not exist.")
    

    print(f"Loading existing dataset from {synthetic_data_dir}...")
    train = torch.load(os.path.join(synthetic_data_dir, "train.pt"))
    val = torch.load(os.path.join(synthetic_data_dir, "val.pt"))
    test = torch.load(os.path.join(synthetic_data_dir, "test.pt"))
    

    return train, val, test
    
"""