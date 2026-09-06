"""
SCBM and baseline models.
"""

import os
import copy
import math
import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, MultivariateNormal
import torch.nn.functional as F
from torchvision import models

from models.networks import FCNNEncoder
from utils.training import freeze_module, unfreeze_module

# Checks if a tensor has become non-finite and prints out diagnostic info if so, then raises an error
def check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        finite = tensor[torch.isfinite(tensor)]
        print(f"\n{name} has NaN/Inf")
        if finite.numel() > 0:
            print(f"{name} finite min:", finite.min().item())
            print(f"{name} finite max:", finite.max().item())
        print(f"{name} num NaN:", torch.isnan(tensor).sum().item())
        print(f"{name} num Inf:", torch.isinf(tensor).sum().item())
        raise ValueError(f"{name} became non-finite")



def create_model(config):
    """
    Parse the configuration file and return a relevant model
    """
    model = None
    if config.model.model not in ["cbm", "cbm_residual", "scbm", "scbm_residual"]:
        print("Could not create model with name ", config.model, "!")
        quit()
    if config.model.model == "cbm":
        print("Using CBM model!")
        model = CBM(config)
    elif config.model.model == "cbm_residual":
        print("Using CBM model with independent residual channel!")
        model = CBMResidual(config)
    elif config.model.model == "scbm":
        print("Using SCBM model!")
        model = SCBM(config)
    elif config.model.model == "scbm_residual":
        print("Using residual model!")
        model = SCBM_residual(config)
    if config.model.compile:
        model = torch.compile(model)
    return model
        


class IntCEMMNISTEncoder(nn.Module):
    """
    IntCEM-style MNIST encoder.

    5 convolutional layers:
        - 16 filters
        - 3x3 kernels
        - BatchNorm
        - LeakyReLU

    Followed by a linear projection to 128 features.
    """

    def __init__(self, in_channels=2, output_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(16 * 28 * 28, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)






class SCBM(nn.Module):
    """
    Stochastic Concept Bottleneck Model (SCBM) with Learned Covariance Matrix.

    This class implements a Stochastic Concept Bottleneck Model (SCBM) that extends concept prediction by incorporating
    a learned covariance matrix. The SCBM aims to capture the uncertainty and dependencies between concepts, providing
    a more robust and interpretable model for concept-based learning tasks.

    Key Features:
    - Predicts concepts along with a learned covariance matrix to model the relationships and uncertainties between concepts.
    - Supports various training modes and intervention strategies to improve model performance and interpretability.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        num_monte_carlo (int): The number of Monte Carlo samples for uncertainty estimation.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
        cov_type (str): The type of covariance matrix ("empirical", "global", or "amortized", where "empirical is fixed at start").

    Methods:
        forward(x, epoch, validation=False, c_true=None):
            Perform a forward pass through the model.
        intervene(c_mcmc_probs, c_mcmc_logits):
            Perform an intervention on the model's concept predictions.
    """

    def __init__(self, config):
        super(SCBM, self).__init__()

        # Configuration arguments
        config_model = config.model
        self.num_concepts = config.data.num_concepts
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode
        self.concept_learning = config_model.concept_learning
        self.num_monte_carlo = config_model.num_monte_carlo
        self.straight_through = config_model.straight_through
        self.curr_temp = 1.0
        if self.training_mode == "joint":
            self.num_epochs = config_model.j_epochs
        else:
            self.num_epochs = config_model.t_epochs
        self.cov_type = config_model.cov_type

        self.init_temp = 1.0
        self.final_temp = 0.5
        self.temp_decay_rate = (math.log(self.final_temp) - math.log(self.init_temp)) / float(self.num_epochs)
        
        # Regression or multilabel model
        self.regression_task = config_model.regression_task
        self.multilabel_task = config_model.multilabel_task
        
        
        

        # Architectures
        # Encoder h(.)
        if self.encoder_arch == "FCNN":
            n_features = 256
            self.encoder = FCNNEncoder(
                num_inputs=config.data.num_covariates, num_hidden=n_features, num_deep=2
            )
        elif self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        config_model.model_directory, "resnet/resnet18-5c106cde.pth"
                    ),
                    weights_only=False
                )
            )

            n_features = self.encoder_res.fc.in_features
            # Removed classification head of resnet and replaced with identity to get intermediate features, ouutput dim 512
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(9216, n_features),
                nn.ReLU(),
            )
            
        elif self.encoder_arch == "mnist_encoder":
            n_features = 128
            self.encoder = IntCEMMNISTEncoder(in_channels=config.data.num_covariates, output_dim=n_features)

        else:
            raise NotImplementedError("ERROR: architecture not supported!")

        # Linear network to predict mu of concept distribution
        self.mu_concepts = nn.Linear(n_features, self.num_concepts, bias=True)

        if self.cov_type == "global":
            self.sigma_concepts = nn.Parameter(
                torch.zeros(int(self.num_concepts * (self.num_concepts + 1) / 2))
            )  # Predict lower triangle of concept covariance
        elif self.cov_type == "empirical":
            self.sigma_concepts = torch.zeros(
                int(self.num_concepts * (self.num_concepts + 1) / 2)
            )
        # Linear network to predict sigma of concept distribution (lower triangle of covariance matrix in logit space)
        else:
            self.sigma_concepts = nn.Linear(
                n_features,
                int(self.num_concepts * (self.num_concepts + 1) / 2),
                bias=True,
            )
            self.sigma_concepts.weight.data *= (
                0.01  # To prevent exploding precision matrix at initialization
            )

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        if self.regression_task:
            self.pred_dim = 1
        else:
            if self.num_classes == 2:
                self.pred_dim = 1
            elif self.num_classes > 2:
                self.pred_dim = self.num_classes

        # Final target predictor head 
        if self.head_arch == "linear":
            fc_y = nn.Linear(self.num_concepts, self.pred_dim)
            self.head = nn.Sequential(fc_y)
        else:
            fc1_y = nn.Linear(self.num_concepts, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    def forward(self, x, epoch, validation=False, return_full=False, c_true=None, return_samples=False):
        """
        Perform a forward pass through the Stochastic Concept Bottleneck Model (SCBM).

        This method performs a forward pass through the SCBM, predicting concept probabilities and logits for the target variable.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            return_full (bool, optional): Flag indicating whether to also return mu of concept. Default is False.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.

        Returns:
            tuple: A tuple containing:
                - c_mcmc_prob (torch.Tensor): MCMC samples for predicted concept probabilities. Shape: (batch_size, num_concepts, num_monte_carlo)
                - c_triang_cov (torch.Tensor): Cholesky decomposition of the concept logit covariance matrix. Shape: (batch_size, num_concepts, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, num_classes)
                - c_mu (torch.Tensor, optional): Predicted concept means. Shape: (batch_size, num_concepts). Returned if `return_full` is True.
        Notes:
            - The method first obtains intermediate representations from the encoder.
            - It then predicts the concept means and the Cholesky decomposition of the covariance matrix in the logit space.
            - The method samples from the predicted normal distribution to obtain concept logits and probabilities.
            - Depending on the training mode, it handles different strategies for sampling and backpropagation.
            - Finally, it predicts the target variable logits by averaging over multiple Monte Carlo samples.
        """

        # Get intermediate representations
        intermediate = self.encoder(x)

        # Get mu and cholesky decomposition of covariance
        c_mu = self.mu_concepts(intermediate)
        if self.cov_type == "global":
            c_sigma = self.sigma_concepts.repeat(c_mu.size(0), 1)
        elif self.cov_type == "empirical":
            c_sigma = self.sigma_concepts.unsqueeze(0).repeat(c_mu.size(0), 1, 1)
        else:
            c_sigma = self.sigma_concepts(intermediate)

        if self.cov_type == "empirical":
            c_triang_cov = c_sigma
        else:
            # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
            c_triang_cov = torch.zeros(
                (c_sigma.shape[0], self.num_concepts, self.num_concepts),
                device=c_sigma.device,
            )
            rows, cols = torch.tril_indices(
                row=self.num_concepts, col=self.num_concepts, offset=0
            )
            diag_idx = rows == cols
            c_triang_cov[:, rows, cols] = c_sigma
            c_triang_cov[:, range(self.num_concepts), range(self.num_concepts)] = (
                F.softplus(c_sigma[:, diag_idx]) + 1e-6
            )

        # Sample from predicted normal distribution
        c_dist = MultivariateNormal(c_mu, scale_tril=c_triang_cov)
        c_mcmc_logit = c_dist.rsample([self.num_monte_carlo]).movedim(
            0, -1
        )  # [batch_size,num_concepts,mcmc_size]
        c_mcmc_prob = self.act_c(c_mcmc_logit)

        # For all MCMC samples simultaneously sample from Bernoulli
        if validation or self.training_mode == "sequential":
            # No backpropagation necessary
            c_mcmc = torch.bernoulli(c_mcmc_prob)
        elif self.training_mode == "independent":
            c_mcmc = c_true.unsqueeze(-1).repeat(1, 1, self.num_monte_carlo).float()
        # Joint
        else:
            # Backpropagation necessary
            curr_temp = self.compute_temperature(epoch)
            dist = RelaxedBernoulli(temperature=curr_temp, probs=c_mcmc_prob)

            # Bernoulli relaxation
            mcmc_relaxed = dist.rsample()
            if self.straight_through:
                # Straight-Through Gumbel Softmax
                mcmc_hard = (mcmc_relaxed > 0.5) * 1
                c_mcmc = mcmc_hard - mcmc_relaxed.detach() + mcmc_relaxed
            else:
                c_mcmc = mcmc_relaxed
                
        y_pred_logits = self.compute_y_pred_logits(c_mcmc, c_mcmc_logit)

        if return_samples:
            return c_mcmc_prob, c_mcmc, c_mcmc_logit, c_triang_cov, y_pred_logits, c_mu

        # Return concept mu for interventions
        if return_full:
            return c_mcmc_prob, c_mu, c_triang_cov, y_pred_logits
        else:
            return c_mcmc_prob, c_triang_cov, y_pred_logits

    def compute_y_pred_logits(self, c_mcmc, c_mcmc_logits):
        # Pick the concept tensor: [B, C, M]
        x = c_mcmc if self.concept_learning == "hard" else c_mcmc_logits
        B, C, M = x.shape

        # Run the head over all M samples at once: reshape to [B*M, C]
        x_flat = x.permute(0, 2, 1).reshape(B * M, C)        # [B*M, C]
        y_logits_flat = self.head(x_flat)                     # [B*M, K] or [B*M, 1]

        if self.regression_task:
            # Regression: average predictions
            y_pred_logits = y_logits_flat.view(B, M, self.pred_dim).mean(dim=1)  # [B, 1]
            return y_pred_logits


        if self.multilabel_task:
            # Multilabel: average sigmoid probs independently per task, then back to logits
            # y_logits_flat: [B*M, K+J]
            y_probs = torch.sigmoid(y_logits_flat).view(B, M, self.pred_dim).mean(dim=1)  # [B, K+J]
            y_pred_logits = torch.logit(y_probs, eps=1e-6)                                # [B, K+J]
            return y_pred_logits

        if self.pred_dim == 1:
            # Binary: average Bernoulli probs then convert back to logits
            y_probs = torch.sigmoid(y_logits_flat).view(B, M, 1).mean(dim=1)  # [B, 1]
            y_pred_logits = torch.logit(y_probs, eps=1e-6)                    # [B, 1]
            return y_pred_logits
        else:
            # Multiclass: compute log(mean softmax) in a numerically stable way:
            # log(mean_i p_i) == logsumexp(log p_i) - log(M), where log p_i = log_softmax(logits_i)
            y_log_probs = F.log_softmax(y_logits_flat, dim=-1).view(B, M, self.pred_dim)  # [B, M, K]
            y_pred_log_probs = torch.logsumexp(y_log_probs, dim=1) - math.log(M)          # [B, K]
            return y_pred_log_probs




    def intervene(self, c_mcmc_probs, c_mcmc_logits):
        y_pred_probs_i = 0
        c_hard = torch.bernoulli(c_mcmc_probs)
        # For each monte carlo sample, compute the predicted target probability and average them
        # For regression, we just average the predicted values
        for i in range(self.num_monte_carlo):
            if self.concept_learning == "soft":
                c_i = c_mcmc_logits[:, :, i]
            else:
                c_i = c_hard[:, :, i]

            y_pred_logits_i = self.head(c_i)
            
            
            if self.regression_task:
                y_pred_probs_i += y_pred_logits_i
            else:
                if self.multilabel_task or self.pred_dim == 1:
                    y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
                else:
                    y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)

        y_pred_probs = y_pred_probs_i / self.num_monte_carlo
        
        if self.regression_task:
            return y_pred_probs
        
        
        if self.multilabel_task or self.pred_dim == 1:
            y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
        else:
            # log(1/M * sum_i p_i)
            y_pred_logits = torch.log(y_pred_probs + 1e-6)

        return y_pred_logits
    


    def compute_temperature(self, epoch):
        curr_temp = max(self.init_temp * math.exp(self.temp_decay_rate * epoch), self.final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.mu_concepts.apply(freeze_module)
        if isinstance(self.sigma_concepts, nn.Linear):
            self.sigma_concepts.apply(freeze_module)
        else:
            self.sigma_concepts.requires_grad = False
            
            
            


class SCBM_residual(nn.Module):
    """
    Stochastic Concept Bottleneck Model (SCBM) with Learned Covariance Matrix.

    This class implements a Stochastic Concept Bottleneck Model (SCBM) that extends concept prediction by incorporating
    a learned covariance matrix. The SCBM aims to capture the uncertainty and dependencies between concepts, providing
    a more robust and interpretable model for concept-based learning tasks.

    Key Features:
    - Predicts concepts along with a learned covariance matrix to model the relationships and uncertainties between concepts.
    - Supports various training modes and intervention strategies to improve model performance and interpretability.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        num_monte_carlo (int): The number of Monte Carlo samples for uncertainty estimation.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
        cov_type (str): The type of covariance matrix ("empirical", "global", or "amortized", where "empirical is fixed at start").

    Methods:
        forward(x, epoch, validation=False, c_true=None):
            Perform a forward pass through the model.
        intervene(c_mcmc_probs, c_mcmc_logits):
            Perform an intervention on the model's concept predictions.
    """

    def __init__(self, config):
        super(SCBM_residual, self).__init__()

        # Configuration arguments
        self.block_diagonal_cov = config.model.block_diagonal_cov
        config_model = config.model
        self.num_concepts = config.data.num_concepts
        self.num_residuals = config.data.num_residuals
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode
        self.concept_learning = config_model.concept_learning
        self.num_monte_carlo = config_model.num_monte_carlo
        self.straight_through = config_model.straight_through
        self.curr_temp = 1.0
        if self.training_mode == "joint":
            self.num_epochs = config_model.j_epochs
        else:
            self.num_epochs = config_model.t_epochs
        self.cov_type = config_model.cov_type

        self.init_temp = 1.0
        self.final_temp = 0.5
        self.temp_decay_rate = (math.log(self.final_temp) - math.log(self.init_temp)) / float(self.num_epochs)
        
        
        # Use sparsity loss
        self.residual_sparsity_weight = config_model.residual_sparsity_weight
        
        
        
        # Regression or multilabel model
        self.regression_task = config_model.regression_task
        self.multilabel_task = config_model.multilabel_task

        # Architectures
        # Encoder h(.)
        if self.encoder_arch == "FCNN":
            n_features = 256
            self.encoder = FCNNEncoder(
                num_inputs=config.data.num_covariates, num_hidden=n_features, num_deep=2
            )
        elif self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        config_model.model_directory, "resnet/resnet18-5c106cde.pth"
                    ),
                    weights_only=False
                )
            )

            n_features = self.encoder_res.fc.in_features
            # Removed classification head of resnet and replaced with identity to get intermediate features, ouutput dim 512
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(9216, n_features),
                nn.ReLU(),
            )
            
        elif self.encoder_arch == "mnist_encoder":
            n_features = 128
            self.encoder = IntCEMMNISTEncoder(in_channels=config.data.num_covariates, output_dim=n_features)

        else:
            raise NotImplementedError("ERROR: architecture not supported!")

        # Linear network to predict mu of concept+residual distribution
        self.mu_concepts_residuals = nn.Linear(n_features, self.num_concepts + self.num_residuals, bias=True)

        if self.cov_type == "global":
            self.sigma_concepts_residuals = nn.Parameter(
                torch.zeros(int((self.num_concepts + self.num_residuals) * (self.num_concepts + self.num_residuals + 1) / 2))
            )  # Predict lower triangle of concept+residual covariance
        elif self.cov_type == "empirical":
            self.sigma_concepts_residuals = torch.zeros(
                int((self.num_concepts + self.num_residuals) * (self.num_concepts + self.num_residuals + 1) / 2)
            )
        # Linear network to predict sigma of concept+residual distribution (lower triangle of covariance matrix in logit space)
        else:
            # Have block diagonal covariance matrix with concept and residual blocks, but no off-diagonal blocks
            if self.block_diagonal_cov:
                n_c_tri = self.num_concepts * (self.num_concepts + 1) // 2
                n_r_tri = self.num_residuals * (self.num_residuals + 1) // 2

                self.sigma_concepts = nn.Linear(n_features, n_c_tri, bias=True)
                self.sigma_residuals = nn.Linear(n_features, n_r_tri, bias=True)
                
                self.sigma_concepts.weight.data *= 0.01
                self.sigma_residuals.weight.data *= 0.01

            # Otherwise, have full covariance matrix for concept+residuals
            else:
                self.sigma_concepts_residuals = nn.Linear(
                    n_features,
                    int((self.num_concepts + self.num_residuals) * (self.num_concepts + self.num_residuals + 1) / 2),
                    bias=True,
                )
                self.sigma_concepts_residuals.weight.data *= (
                    0.01  # To prevent exploding precision matrix at initialization
                )

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        if self.regression_task:
            self.pred_dim = 1
        else:
            if self.num_classes == 2:
                self.pred_dim = 1
            elif self.num_classes > 2:
                self.pred_dim = self.num_classes

        # Final target predictor head 
        if self.head_arch == "linear":
            fc_y = nn.Linear(self.num_concepts + self.num_residuals, self.pred_dim)
            self.head = nn.Sequential(fc_y)
        else:
            fc1_y = nn.Linear(self.num_concepts + self.num_residuals, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)


    def forward(
        self,
        x,
        epoch,
        validation=False,
        return_full=False,
        c_true=None,
        return_samples=False,
        return_L_int_extension=False,
    ):
        """
        Perform a forward pass through the Stochastic Concept Bottleneck Model (SCBM).

        This method performs a forward pass through the SCBM, predicting concept probabilities and logits for the target variable.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            return_full (bool, optional): Flag indicating whether to also return mu of concept. Default is False.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.

        Returns:
            tuple: A tuple containing:
                - c_mcmc_prob (torch.Tensor): MCMC samples for predicted concept probabilities. Shape: (batch_size, num_concepts, num_monte_carlo)
                - c_triang_cov (torch.Tensor): Cholesky decomposition of the concept logit covariance matrix. Shape: (batch_size, num_concepts, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, num_classes)
                - c_mu (torch.Tensor, optional): Predicted concept means. Shape: (batch_size, num_concepts). Returned if `return_full` is True.
        Notes:
            - The method first obtains intermediate representations from the encoder.
            - It then predicts the concept means and the Cholesky decomposition of the covariance matrix in the logit space.
            - The method samples from the predicted normal distribution to obtain concept logits and probabilities.
            - Depending on the training mode, it handles different strategies for sampling and backpropagation.
            - Finally, it predicts the target variable logits by averaging over multiple Monte Carlo samples.
        """

        # Get intermediate representations
        intermediate = self.encoder(x)
        check_finite("intermediate", intermediate)





        # Get mu and cholesky decomposition of covariance
        c_res_mu = self.mu_concepts_residuals(intermediate)
        check_finite("c_res_mu", c_res_mu)
        
        
        if self.cov_type == "global":
            c_res_sigma = self.sigma_concepts_residuals.repeat(c_res_mu.size(0), 1)
        elif self.cov_type == "empirical":
            c_res_sigma = self.sigma_concepts_residuals.unsqueeze(0).repeat(c_res_mu.size(0), 1, 1)
        else:
            
            if self.block_diagonal_cov:
                c_sigma = self.sigma_concepts(intermediate)
                r_sigma = self.sigma_residuals(intermediate)
                check_finite("c_sigma", c_sigma)
                check_finite("r_sigma", r_sigma)
            else:
                c_res_sigma = self.sigma_concepts_residuals(intermediate)

                check_finite("c_res_sigma", c_res_sigma)


        if self.cov_type == "empirical":
            c_res_triang_cov = c_res_sigma
        else:
            if self.block_diagonal_cov:
                C, R = self.num_concepts, self.num_residuals
                B = c_sigma.shape[0]
                
                
                
                # Fill L_CC
                rows_c, cols_c = torch.tril_indices(C, C, offset=0, device=intermediate.device)
                L_CC = torch.zeros(B, C, C, device=intermediate.device)
                L_CC[:, rows_c, cols_c] = c_sigma
                L_CC[:, range(C), range(C)] = F.softplus(c_sigma[:, rows_c == cols_c]) + 1e-6
                
                # Fill L_RR
                rows_r, cols_r = torch.tril_indices(R, R, offset=0, device=intermediate.device)
                L_RR = torch.zeros(B, R, R, device=intermediate.device)
                L_RR[:, rows_r, cols_r] = r_sigma
                L_RR[:, range(R), range(R)] = F.softplus(r_sigma[:, rows_r == cols_r]) + 1e-6
                
                # Assemble block-diagonal L
                c_res_triang_cov = torch.zeros(B, C + R, C + R, device=intermediate.device)
                c_res_triang_cov[:, :C, :C] = L_CC
                c_res_triang_cov[:, C:, C:] = L_RR
                
                check_finite("c_res_triang_cov post diag", c_res_triang_cov)
            
            
            else:
                # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
                c_res_triang_cov = torch.zeros(
                    (c_res_sigma.shape[0], self.num_concepts + self.num_residuals, self.num_concepts + self.num_residuals),
                    device=c_res_sigma.device,
                )
                # tril_indices returns the row and column indices of the lower triangular part of a matrix, including the diagonal
                rows, cols = torch.tril_indices(
                    row=self.num_concepts + self.num_residuals, col=self.num_concepts + self.num_residuals, offset=0
                )
                diag_idx = rows == cols
                c_res_triang_cov[:, rows, cols] = c_res_sigma
                
                check_finite("c_res_triang_cov pre diag", c_res_triang_cov)
                
                c_res_triang_cov[:, range(self.num_concepts + self.num_residuals), range(self.num_concepts + self.num_residuals)] = (
                    F.softplus(c_res_sigma[:, diag_idx]) + 1e-6
                )
                check_finite("c_res_triang_cov post diag", c_res_triang_cov)
            
        # Sample from predicted normal distribution
        c_res_dist = MultivariateNormal(c_res_mu, scale_tril=c_res_triang_cov)
        c_res_mcmc_logit = c_res_dist.rsample([self.num_monte_carlo]).movedim(
            0, -1
        )  # [batch_size,num_concepts,mcmc_size]
        c_res_mcmc_prob = self.act_c(c_res_mcmc_logit)

        
        
        # For all MCMC samples simultaneously sample from Bernoulli
        if validation or self.training_mode == "sequential":
            # No backpropagation necessary
            c_res_mcmc = torch.bernoulli(c_res_mcmc_prob)
        elif self.training_mode == "independent":
            raise NotImplementedError("Independent training not implemented for residual model!")
            c_mcmc = c_true.unsqueeze(-1).repeat(1, 1, self.num_monte_carlo).float()
        # Joint
        else:
            # Backpropagation necessary
            curr_temp = self.compute_temperature(epoch)
            dist = RelaxedBernoulli(temperature=curr_temp, probs=c_res_mcmc_prob)

            # Bernoulli relaxation
            mcmc_relaxed = dist.rsample()
            if self.straight_through:
                # Straight-Through Gumbel Softmax
                mcmc_hard = (mcmc_relaxed > 0.5) * 1
                c_res_mcmc = mcmc_hard - mcmc_relaxed.detach() + mcmc_relaxed
            else:
                c_res_mcmc = mcmc_relaxed
                
        y_pred_logits = self.compute_y_pred_logits(c_res_mcmc, c_res_mcmc_logit)

        if return_samples:
            return (
                c_res_mcmc_prob,
                c_res_mcmc,
                c_res_mcmc_logit,
                c_res_triang_cov,
                y_pred_logits,
            )
            
        if return_L_int_extension:
            return (
                c_res_mcmc_prob,
                c_res_mcmc,
                c_res_mcmc_logit,
                c_res_triang_cov,
                y_pred_logits,
                c_res_mu,
            )

        # Return concept mu for interventions
        if return_full:
            return c_res_mcmc_prob, c_res_mu, c_res_triang_cov, y_pred_logits
        else:
            return c_res_mcmc_prob, c_res_triang_cov, y_pred_logits







    


    def compute_y_pred_logits(self, c_res_mcmc, c_res_mcmc_logits):
        """
        c_res_mcmc: [B, C+R, M] is actually the bernoulli sampled concepts+residuals
        """
        if self.residual_sparsity_weight > 0 and self.concept_learning != "hard":
            raise ValueError(
                "residual_sparsity_weight > 0 requires concept_learning='hard'; in soft mode "
                "the head consumes logits, where sigmoid is a bijection and the penalty only "
                "translates them."
            )
        
        
        
        
        # Pick the concept tensor: [B, C, M]
        # if Straight-Through Gumbel Softmax then c_res_mcmc approximately binary
        x = c_res_mcmc if self.concept_learning == "hard" else c_res_mcmc_logits
        B, C, M = x.shape
        
        

        # Run the head over all M samples at once: reshape to [B*M, C]
        x_flat = x.permute(0, 2, 1).reshape(B * M, C)        # [B*M, C]
        y_logits_flat = self.head(x_flat)                     # [B*M, K] or [B*M, 1]


        if self.regression_task:
            # Regression: average predictions
            y_pred_logits = y_logits_flat.view(B, M, self.pred_dim).mean(dim=1)  # [B, 1]
            return y_pred_logits
        
        if self.multilabel_task:
            # Multilabel: average sigmoid probs independently per task, then back to logits
            # y_logits_flat: [B*M, K+J]
            y_probs = torch.sigmoid(y_logits_flat).view(B, M, self.pred_dim).mean(dim=1)  # [B, K+J]
            y_pred_logits = torch.logit(y_probs, eps=1e-6)                                # [B, K+J]
            return y_pred_logits

        if self.pred_dim == 1:
            # Binary: average Bernoulli probs then convert back to logits
            y_probs = torch.sigmoid(y_logits_flat).view(B, M, 1).mean(dim=1)  # [B, 1]
            y_pred_logits = torch.logit(y_probs, eps=1e-6)                    # [B, 1]
            return y_pred_logits
        else:
            # Multiclass: compute log(mean softmax) in a numerically stable way:
            # log(mean_i p_i) == logsumexp(log p_i) - log(M), where log p_i = log_softmax(logits_i)
            y_log_probs = F.log_softmax(y_logits_flat, dim=-1).view(B, M, self.pred_dim)  # [B, M, K]
            y_pred_log_probs = torch.logsumexp(y_log_probs, dim=1) - math.log(M)          # [B, K]
            return y_pred_log_probs

    def intervene(self, c_mcmc_probs, c_mcmc_logits):
        """ 
        
        """
        y_pred_probs_i = 0
        c_hard = torch.bernoulli(c_mcmc_probs)
        for i in range(self.num_monte_carlo):
            if self.concept_learning == "soft":
                c_i = c_mcmc_logits[:, :, i]
            else:
                c_i = c_hard[:, :, i]

            y_pred_logits_i = self.head(c_i)
            
            if self.regression_task:
                y_pred_probs_i += y_pred_logits_i
            else:
                if self.multilabel_task or self.pred_dim == 1:
                    y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
                else:
                    y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)

        y_pred_probs = y_pred_probs_i / self.num_monte_carlo
        
        
        if self.regression_task:
            return y_pred_probs
        
        if self.multilabel_task or self.pred_dim == 1:
            y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
        else:
            # log(1/M * sum_i p_i)
            y_pred_logits = torch.log(y_pred_probs + 1e-6)

        return y_pred_logits
    
    def intervene_straight_through(self, c_mcmc_probs, c_mcmc_logits):
        y_pred_probs_i = 0
        c_hard = torch.bernoulli(c_mcmc_probs)

        c_input = c_hard.detach() - c_mcmc_probs.detach() + c_mcmc_probs
        # Equivalent idea:
        # forward:  c_input == c_hard
        # backward: d c_input / d c_mcmc_probs == 1
        for i in range(self.num_monte_carlo):
            c_i = c_input[:, :, i]

            y_pred_logits_i = self.head(c_i)
            
            
            if self.regression_task:
                y_pred_probs_i += y_pred_logits_i
            else:
                
                if self.multilabel_task or self.pred_dim == 1:
                    y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
                else:
                    y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)

        y_pred_probs = y_pred_probs_i / self.num_monte_carlo
        
        if self.regression_task:
            return y_pred_probs
        
        
        if self.multilabel_task or self.pred_dim == 1:
            y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
        else:
            y_pred_logits = torch.log(y_pred_probs + 1e-6)

        return y_pred_logits
    
    def intervene_straight_through_vectorized(self, c_mcmc_probs, c_mcmc_logits):
        B, C, M = c_mcmc_probs.shape

        c_hard = torch.bernoulli(c_mcmc_probs)

        # forward: hard samples
        # backward: gradients flow through c_mcmc_probs
        c_input = c_hard.detach() - c_mcmc_probs.detach() + c_mcmc_probs

        # [B, C, M] -> [B * M, C]
        c_flat = c_input.permute(0, 2, 1).reshape(B * M, C)

        y_logits_flat = self.head(c_flat)
        
        if self.regression_task:
            y_pred_logits = y_logits_flat.view(B, M, self.pred_dim).mean(dim=1)  # [B, 1]
            return y_pred_logits
        
        if self.multilabel_task:
            # Average sigmoid probs independently per task, then back to logits
            y_probs = torch.sigmoid(y_logits_flat).view(B, M, self.pred_dim).mean(dim=1)  # (B, K+J)
            return torch.logit(y_probs, eps=1e-6)

        if self.pred_dim == 1:
            y_probs = torch.sigmoid(y_logits_flat).view(B, M, 1).mean(dim=1)
            return torch.logit(y_probs, eps=1e-6)

        else:
            y_log_probs = F.log_softmax(y_logits_flat, dim=-1).view(B, M, self.pred_dim)
            y_pred_log_probs = torch.logsumexp(y_log_probs, dim=1) - math.log(M)
            return y_pred_log_probs
        






    def compute_temperature(self, epoch):
        curr_temp = max(self.init_temp * math.exp(self.temp_decay_rate * epoch), self.final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.mu_concepts_residuals.apply(freeze_module)  # FIX 1

        # FIX 2: branch on cov_type and block_diagonal_cov
        if self.cov_type == "global":
            self.sigma_concepts_residuals.requires_grad = False
        elif self.cov_type == "amortized":
            if self.block_diagonal_cov:
                self.sigma_concepts.apply(freeze_module)
                self.sigma_residuals.apply(freeze_module)
            else:
                self.sigma_concepts_residuals.apply(freeze_module)
        # empirical has no parameters to freeze


class CBM(nn.Module):
    """
    Model class encompassing all baselines: Hard & Soft Concept Bottleneck Model (CBM),
                                            Concept Embedding Model (CEM), and Autoregressive CBM (AR).

    This class implements the baselines. Depending on the choice of modely a small part of the full code is used.
    Check the if statements in the forward method to see which part of the code is used for which model.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        concept_learning (str): The concept learning method ("hard", "soft", "embedding", or "autoregressive").
                                This determines the type of method to use
        num_monte_carlo (int): The number of Monte Carlo samples for sampling Gumbel Softmax in AR.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
    """

    def __init__(self, config):
        super(CBM, self).__init__()

        # Configuration arguments
        config_model = config.model
        self.num_concepts = config.data.num_concepts
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode
        self.concept_learning = config_model.concept_learning
        if self.concept_learning in ("hard", "autoregressive"):
            self.num_monte_carlo = config_model.num_monte_carlo
            self.straight_through = config_model.straight_through
            self.curr_temp = 1.0
            if self.training_mode == "joint":
                self.num_epochs = config_model.j_epochs
            else:
                self.num_epochs = config_model.t_epochs
        elif self.concept_learning == "embedding":
            self.CEM_embedding = config_model.embedding_size

        # Architectures
        # Encoder h(.)
        if self.encoder_arch == "FCNN":
            n_features = 256
            self.encoder = FCNNEncoder(
                num_inputs=config.data.num_covariates, num_hidden=n_features, num_deep=2
            )
        elif self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        config_model.model_directory, "resnet/resnet18-5c106cde.pth"
                    ),
                    weights_only=False
                )
            )
            n_features = self.encoder_res.fc.in_features
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(9216, n_features),
                nn.ReLU(),
            )
            
        elif self.encoder_arch == "mnist_encoder":
            n_features = 128
            self.encoder = IntCEMMNISTEncoder(in_channels=config.data.num_covariates, output_dim=n_features)

        else:
            raise NotImplementedError("ERROR: architecture not supported!")
        if self.concept_learning == "embedding":
            print(
                "Please be aware that our implementation of CEMs is without training on interventions! This is because we would deem this an unfair comparison to our method that is also not trained on interventions. Still, be careful when using this CEM code for derivative works"
            )
            self.positive_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(n_features, self.CEM_embedding, bias=True),
                        nn.LeakyReLU(),
                    )
                    for _ in range(self.num_concepts)
                ]
            )
            self.negative_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(n_features, self.CEM_embedding, bias=True),
                        nn.LeakyReLU(),
                    )
                    for _ in range(self.num_concepts)
                ]
            )
            self.scoring_function = nn.Sequential(
                nn.Linear(self.CEM_embedding * 2, 1, bias=True), nn.Sigmoid()
            )
            self.concept_dim = self.CEM_embedding * self.num_concepts
        else:
            if self.concept_learning == "autoregressive":
                self.concept_predictor = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(n_features + i, 50, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(50, 1, bias=True),
                        )
                        for i in range(self.num_concepts)
                    ]
                )

            else:
                self.concept_predictor = nn.Linear(
                    n_features, self.num_concepts, bias=True
                )
            self.concept_dim = self.num_concepts

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes

        if self.head_arch == "linear":
            fc_y = nn.Linear(self.concept_dim, self.pred_dim)
            self.head = nn.Sequential(fc_y)
        else:
            fc1_y = nn.Linear(self.concept_dim, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    def forward(
        self,
        x,
        epoch,
        c_true=None,
        validation=False,
        concepts_train_ar=False,
    ):
        """
        Perform a forward pass through one of the baselines.

        This method performs a forward pass predicting concept probabilities and logits for the target variable.
        It handles different concept learning strategies and training modes, including hard, soft, autoregressive, and embedding-based concepts.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            concepts_train_ar (torch.Tensor, optional): Ground-truth concept values for autoregressive training. Default is False.

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, label_dim)
                - c (torch.Tensor): Predicted hard concept values (if method permits, otherwise the concept representation). Shape: (batch_size, num_concepts, num_monte_carlo) for MCMC sampling or (batch_size, num_concepts) otherwise.
        """

        # Get intermediate representations
        intermediate = self.encoder(x)

        # Get concept predictions
        if self.concept_learning in ("hard", "soft"):
            # CBM
            c_logit = self.concept_predictor(intermediate)
            c_prob = self.act_c(c_logit)

            if self.concept_learning in ("hard"):
                # Hard CBM
                if self.training_mode == "sequential" or validation:
                    # Sample from Bernoulli M times, as we don't need to backprop
                    c_prob_mcmc = c_prob.unsqueeze(-1).expand(
                        -1, -1, self.num_monte_carlo
                    )
                    c = torch.bernoulli(c_prob_mcmc)

                # Relax bernoulli sampling with Gumbel Softmax to allow for backpropagation
                elif self.training_mode == "joint":
                    curr_temp = self.compute_temperature(epoch, device=c_prob.device)
                    dist = RelaxedBernoulli(temperature=curr_temp, probs=c_prob)
                    c_relaxed = dist.rsample([self.num_monte_carlo]).movedim(0, -1)
                    if self.straight_through:
                        # Straight-Through Gumbel Softmax
                        c_hard = (c_relaxed > 0.5) * 1
                        c = c_hard - c_relaxed.detach() + c_relaxed
                    else:
                        # Reparametrization trick.
                        c = c_relaxed

                else:
                    raise NotImplementedError

        elif self.concept_learning == "autoregressive":
            # AR
            if validation:
                c_prob, c_hard = [], []
                for predictor in self.concept_predictor:
                    if c_prob:
                        concept = []
                        for i in range(
                            self.num_monte_carlo
                        ):  # MCMC samples for evaluation and interventions, but not for training
                            concept_input_i = torch.cat(
                                [intermediate, torch.cat(c_hard, dim=1)[..., i]], dim=1
                            )
                            concept.append(self.act_c(predictor(concept_input_i)))
                        concept = torch.cat(concept, dim=-1)
                        c_relaxed = torch.bernoulli(concept)[:, None, :]
                        concept = concept[:, None, :]
                        concept_hard = c_relaxed

                    else:
                        concept_input = intermediate
                        concept = self.act_c(predictor(concept_input))
                        concept = concept.unsqueeze(-1).expand(
                            -1, -1, self.num_monte_carlo
                        )
                        c_relaxed = torch.bernoulli(concept)
                        concept_hard = c_relaxed
                    c_prob.append(concept)
                    c_hard.append(concept_hard)
                c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
                c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)

            elif self.training_mode == "independent":
                # Training
                if c_true is None and concepts_train_ar is not False:
                    c_prob, c_hard = [], []
                    for c_idx, predictor in enumerate(self.concept_predictor):
                        if c_hard:
                            concept_input = torch.cat(
                                [intermediate, concepts_train_ar[:, :c_idx]], dim=1
                            )
                        else:
                            concept_input = intermediate
                        concept = self.act_c(predictor(concept_input))

                        # No Gumbel softmax because backprop can happen through the input connection
                        c_relaxed = torch.bernoulli(concept)
                        concept_hard = c_relaxed

                        # NOTE that the following train-time variables are overly good because they are taking ground truth as input. At validation time, we sample
                        c_prob.append(concept)
                        c_hard.append(concept_hard)
                    c_prob = torch.cat(
                        [c_prob[i] for i in range(self.num_concepts)], dim=1
                    )
                    c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)

                else:  # Training the head with the GT concepts as input
                    c_prob = c_true.float()
                    c = c_true.float()

            else:
                raise NotImplementedError

        elif self.concept_learning == "embedding":
            # CEM
            if self.training_mode == "joint":
                # Obtaining concept embeddings
                c_p = [p(intermediate) for p in self.positive_embeddings]
                c_n = [n(intermediate) for n in self.negative_embeddings]

                # Concept probabilities from scoring function
                c_prob = [
                    self.scoring_function(torch.cat((c_p[i], c_n[i]), dim=1))
                    for i in range(self.num_concepts)
                ]

                # Final concept embedding
                z_prob = [
                    c_prob[i] * c_p[i] + (1 - c_prob[i]) * c_n[i]
                    for i in range(self.num_concepts)
                ]
                z_prob = torch.cat([z_prob[i] for i in range(self.num_concepts)], dim=1)
                c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
                c = z_prob
            else:
                raise Exception("CEMs are trained jointly, change training mode")

        # Get predicted targets
        if self.concept_learning == "hard" or (
            self.concept_learning == "autoregressive" and validation
        ):
            # Hard CBM or validation of AR. Takes MCMC samples.
            # MCMC loop for predicting label
            y_pred_probs_i = 0
            for i in range(self.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                if self.pred_dim == 1:
                    y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
                else:
                    y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)
            y_pred_probs = y_pred_probs_i / self.num_monte_carlo

            if self.pred_dim == 1:
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_logits = torch.log(y_pred_probs + 1e-6)

        elif self.concept_learning == "soft":
            # Soft CBM
            y_pred_logits = self.head(
                c_logit
            )  # NOTE that we're passing logits not probs in soft case as is also done by Koh et al.
            c = torch.empty_like(c_prob)

        elif self.concept_learning == "embedding" or (
            self.concept_learning == "autoregressive" and not validation
        ):
            # CEM or training of AR. Takes ground truth concepts.
            # If CEM: c are predicte embeddings, if AR: c are ground truth concepts
            y_pred_logits = self.head(c)

        return c_prob, y_pred_logits, c

    def intervene(
        self,
        concepts_interv_probs,
        concepts_mask,
        input_features,
        concepts_pred_probs,
    ):
        if self.concept_learning == "soft":
            # Soft CBM
            c_logit = torch.logit(concepts_interv_probs, eps=1e-6)
            y_pred_logits = self.head(c_logit)

        elif self.concept_learning in ("hard", "autoregressive"):
            # Hard CBM or AR
            y_pred_probs_i = 0

            if self.concept_learning == "hard":
                c_prob_mcmc = concepts_interv_probs.unsqueeze(-1).expand(
                    -1, -1, self.num_monte_carlo
                )
                c = torch.bernoulli(c_prob_mcmc)

                # Fix intervened-on concepts to ground truth
                c[concepts_mask == 1] = (
                    concepts_interv_probs[concepts_mask == 1]
                    .unsqueeze(-1)
                    .expand(-1, self.num_monte_carlo)
                )
                weight = torch.ones((c.shape[0], self.num_monte_carlo), device=c.device)

            elif self.concept_learning == "autoregressive":
                # Note: Here, concepts_interv_probs are already the hard, MCMC sampled concepts as determined by the intervene_ar function
                id = torch.nonzero(
                    concepts_interv_probs * concepts_mask == 1, as_tuple=False
                )
                weight_k = torch.log(
                    1 - concepts_pred_probs + 1e-6
                )  # If intervened-on concepts have value 0
                weight_k.index_put_(
                    list(id.t()),
                    torch.log(concepts_pred_probs + 1e-6)[id[:, 0], id[:, 1], id[:, 2]],
                    accumulate=False,
                )  # If intervened-on concepts have value 1
                weight_k = (
                    weight_k * concepts_mask
                )  # Only compute weight for intervened-on concepts
                weight = torch.sum(weight_k, dim=(1))  # Sum over concepts
                weight = torch.softmax(
                    weight, dim=-1
                )  # Replicating their implementation (from log to prob space)
                c = concepts_interv_probs

            for i in range(self.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                if self.pred_dim == 1:
                    y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.sigmoid(
                        y_pred_logits_i
                    )
                else:
                    y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.softmax(
                        y_pred_logits_i, dim=1
                    )
            y_pred_probs = y_pred_probs_i / torch.sum(weight, dim=1).unsqueeze(1)
            if self.pred_dim == 1:
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_logits = torch.log(y_pred_probs + 1e-6)

        elif self.concept_learning == "embedding":
            # CEM
            # Get intermediate representations
            intermediate = self.encoder(input_features)
            # Obtaining concept embeddings
            c_p = [p(intermediate) for p in self.positive_embeddings]
            c_n = [n(intermediate) for n in self.negative_embeddings]
            # Final concept embedding
            z_prob = [
                concepts_interv_probs[:, i].unsqueeze(1) * c_p[i]
                + (1 - concepts_interv_probs[:, i].unsqueeze(1)) * c_n[i]
                for i in range(self.num_concepts)
            ]
            z_prob = torch.cat([z_prob[i] for i in range(self.num_concepts)], dim=1)
            y_pred_logits = self.head(z_prob)

        return y_pred_logits

    def intervene_ar(self, concepts_true, concepts_mask, input_features):
        """
        Perform an intervention on the Autoregressive CBM.

        This method performs an intervention on the Autoregressive CBM by fixing the intervened-on concepts
        to their ground-truth values and MCMC sampling the remaining concepts.
        The predicted probabilities of the intervened-on concepts are stored nevertheless to compute the reweighting.
        The reweighting is computed afterwards using the intervene function.

        Args:
            concepts_true (torch.Tensor): The ground-truth concept values. Shape: (batch_size, num_concepts, num_monte_carlo)
            concepts_mask (torch.Tensor): A mask indicating which concepts are intervened. Shape: (batch_size, num_concepts, num_monte_carlo)
            input_features (torch.Tensor): The input features for the encoder. Shape: (batch_size, input_dims)

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts, num_monte_carlo)
                - c (torch.Tensor): Hard predicted concept values with interventions applied. Shape: (batch_size, num_concepts, num_monte_carlo)
        """
        # Concept predictions for autoregressive model. Intervened-on concepts are fixed to ground truth
        intermediate = self.encoder(input_features)
        c_prob, c_hard = [], []
        for j, (predictor) in enumerate(self.concept_predictor):
            if c_prob:
                concept = []
                for i in range(
                    self.num_monte_carlo
                ):  # MCMC samples for evaluation and interventions, but not for joint training
                    concept_input_i = torch.cat(
                        [intermediate, torch.cat(c_hard, dim=1)[..., i]], dim=1
                    )
                    concept.append(self.act_c(predictor(concept_input_i)))
                concept = torch.cat(concept, dim=-1)
                concept_hard = torch.bernoulli(concept)[:, None, :]
                concept = concept[:, None, :]
            else:
                concept_input = intermediate
                concept = self.act_c(predictor(concept_input))
                concept = concept.unsqueeze(-1).expand(-1, -1, self.num_monte_carlo)
                concept_hard = torch.bernoulli(concept)

            concept_hard = (
                concept_hard * (1 - concepts_mask[:, j, :])[:, None, :]
                + concepts_mask[:, j, :][:, None, :]
                * concepts_true[:, j, :][:, None, :]
            )  # Only update if it is not an intervened on
            concept = (
                concept * (1 - concepts_mask[:, j, :][:, None, :])
                + concepts_mask[:, j, :][:, None, :]
                * concepts_true[:, j, :][:, None, :]
            )

            c_prob.append(concept)
            c_hard.append(concept_hard)
        c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
        c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)
        return c_prob, c

    def compute_temperature(self, epoch, device):
        final_temp = torch.tensor([0.5], device=device)
        init_temp = torch.tensor([1.0], device=device)
        rate = (math.log(final_temp) - math.log(init_temp)) / float(self.num_epochs)
        curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.concept_predictor.apply(freeze_module)








class CBMResidual(nn.Module):
    """
    Hard Concept Bottleneck Model with an independent residual channel.

    Concept pathway:
        x -> concept_encoder -> concept_predictor -> concepts -> concept_head

    Residual pathway:
        x -> residual_encoder -> residual_predictor -> residual_head

    Final task prediction:
        y_logits = concept_y_logits + residual_y_logits

    The residual channel:
        - does not receive concepts as input;
        - is not sampled jointly with concepts;
        - has no concept-residual covariance;
        - is trained only through the task loss.
    """

    def __init__(self, config):
        super().__init__()

        config_model = config.model

        self.num_concepts = config.data.num_concepts
        self.num_residuals = config.data.num_residuals
        self.num_classes = config.data.num_classes

        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode

        self.num_monte_carlo = config_model.num_monte_carlo
        self.straight_through = config_model.straight_through
        self.curr_temp = 1.0

        self.multilabel_task = config_model.multilabel_task
        self.regression_task = config_model.regression_task

        if self.training_mode == "joint":
            self.num_epochs = config_model.j_epochs
        elif self.training_mode == "sequential":
            self.num_epochs = config_model.t_epochs
        else:
            raise ValueError(
                "CBMResidual only supports 'joint' and 'sequential' training."
            )

        # ---------------------------------------------------------
        # Concept encoder
        # ---------------------------------------------------------
        self.encoder, n_features = self._build_encoder(config)

        # ---------------------------------------------------------
        # Independent residual encoder
        # ---------------------------------------------------------
        # deepcopy gives the residual pathway independent parameters,
        # while retaining the same initial encoder architecture/weights.
        self.residual_encoder = copy.deepcopy(self.encoder)

        # ---------------------------------------------------------
        # Concept and residual representations
        # ---------------------------------------------------------
        self.concept_predictor = nn.Linear(
            n_features,
            self.num_concepts,
            bias=True,
        )

        self.residual_predictor = nn.Linear(
            n_features,
            self.num_residuals,
            bias=True,
        )

        self.act_c = nn.Sigmoid()

        # ---------------------------------------------------------
        # Task heads
        # ---------------------------------------------------------
        self.pred_dim = 1 if self.num_classes == 2 else self.num_classes

        if self.head_arch == "linear":
            self.concept_head = nn.Linear(
                self.num_concepts,
                self.pred_dim,
            )
        else:
            self.concept_head = nn.Sequential(
                nn.Linear(self.num_concepts, 256),
                nn.ReLU(),
                nn.Linear(256, self.pred_dim),
            )

        # Keep the residual-to-output mapping deliberately simple.
        self.residual_head = nn.Linear(
            self.num_residuals,
            self.pred_dim,
        )

    def _build_encoder(self, config):
        """Construct the encoder used by one pathway."""

        config_model = config.model

        if self.encoder_arch == "FCNN":
            n_features = 256

            encoder = FCNNEncoder(
                num_inputs=config.data.num_covariates,
                num_hidden=n_features,
                num_deep=2,
            )

        elif self.encoder_arch == "resnet18":
            encoder_res = models.resnet18(weights=None)

            encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        config_model.model_directory,
                        "resnet/resnet18-5c106cde.pth",
                    ),
                    weights_only=False,
                )
            )

            n_features = encoder_res.fc.in_features
            encoder_res.fc = Identity()
            encoder = nn.Sequential(encoder_res)

        elif self.encoder_arch == "simple_CNN":
            n_features = 256

            encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(9216, n_features),
                nn.ReLU(),
            )
            
        elif self.encoder_arch == "mnist_encoder":
            n_features = 128
            encoder = IntCEMMNISTEncoder(
                in_channels=config.data.num_covariates,
                output_dim=n_features,
            )

        else:
            raise NotImplementedError(
                f"Encoder architecture '{self.encoder_arch}' is not supported."
            )

        return encoder, n_features

    def forward(
        self,
        x,
        epoch,
        validation=False,
        return_residual=False,
    ):
        """
        Run the hard CBM and independent residual pathways.

        Returns:
            c_prob:
                Predicted concept probabilities, shape (B, C).

            y_pred_logits:
                Final task logits, shape (B, K).

            c:
                Sampled hard concepts, shape (B, C, M).

            residual, optional:
                Continuous residual representation, shape (B, R).
        """

        # ---------------------------------------------------------
        # Concept pathway
        # ---------------------------------------------------------
        concept_features = self.encoder(x)

        c_logit = self.concept_predictor(concept_features)
        c_prob = self.act_c(c_logit)

        c_prob_mcmc = c_prob.unsqueeze(-1).expand(
            -1,
            -1,
            self.num_monte_carlo,
        )

        if self.training_mode == "sequential" or validation:
            # No task-loss gradient through the sampled concepts.
            c = torch.bernoulli(c_prob_mcmc)

        elif self.training_mode == "joint":
            temperature = self.compute_temperature(
                epoch,
                device=c_prob.device,
            )

            distribution = RelaxedBernoulli(
                temperature=temperature,
                probs=c_prob,
            )

            c_relaxed = distribution.rsample(
                [self.num_monte_carlo]
            ).movedim(0, -1)

            if self.straight_through:
                c_hard = (c_relaxed > 0.5).float()
                c = c_hard - c_relaxed.detach() + c_relaxed
            else:
                c = c_relaxed

        else:
            raise ValueError(
                f"Unsupported training mode: {self.training_mode}"
            )

        # ---------------------------------------------------------
        # Independent residual pathway
        # ---------------------------------------------------------
        residual_features = self.residual_encoder(x)
        residual = self.residual_predictor(residual_features)
        residual_y_logits = self.residual_head(residual)

        # ---------------------------------------------------------
        # Combine the two pathways at the output-logit level
        # ---------------------------------------------------------
        y_pred_probs_sum = 0

        for sample_idx in range(self.num_monte_carlo):
            c_i = c[:, :, sample_idx]

            concept_y_logits_i = self.concept_head(c_i)

            # The residual pathway is an additive correction to the
            # concept-based task logits.
            y_pred_logits_i = (
                concept_y_logits_i + residual_y_logits
            )

            if self.regression_task:
                y_pred_probs_sum += y_pred_logits_i
            elif self.multilabel_task or self.pred_dim == 1:
                # Independent per-task Bernoulli probabilities
                y_pred_probs_sum += torch.sigmoid(
                    y_pred_logits_i
                )
            else:
                y_pred_probs_sum += torch.softmax(
                    y_pred_logits_i,
                    dim=1,
                )

        y_pred_probs = (
            y_pred_probs_sum / self.num_monte_carlo
        )

        if self.regression_task:
            y_pred_logits = y_pred_probs
        elif self.multilabel_task or self.pred_dim == 1:
            y_pred_logits = torch.logit(
                y_pred_probs,
                eps=1e-6,
            )
        else:
            y_pred_logits = torch.log(
                y_pred_probs + 1e-6
            )

        if return_residual:
            return c_prob, y_pred_logits, c, residual

        return c_prob, y_pred_logits, c

    def intervene(
        self,
        concepts_interv_probs,
        concepts_mask,
        input_features,
        concepts_pred_probs=None,
    ):
        """
        Perform interventions on the concept channel.

        `concepts_pred_probs` is accepted for signature compatibility with
        `intervene_cbm` and is unused: the hard concept channel is fully
        described by `concepts_interv_probs`.

        The residual channel is recomputed directly from the input and
        remains unaffected by the concept intervention.
        """

        # ---------------------------------------------------------
        # Sample the concept representation
        # ---------------------------------------------------------
        c_prob_mcmc = concepts_interv_probs.unsqueeze(-1).expand(
            -1,
            -1,
            self.num_monte_carlo,
        )

        c = torch.bernoulli(c_prob_mcmc)

        # Fix intervened concepts to their intervention values.
        intervened = concepts_mask == 1

        c[intervened] = (
            concepts_interv_probs[intervened]
            .unsqueeze(-1)
            .expand(-1, self.num_monte_carlo)
        )

        # ---------------------------------------------------------
        # Recompute the independent residual pathway
        # ---------------------------------------------------------
        residual_features = self.residual_encoder(
            input_features
        )

        residual = self.residual_predictor(
            residual_features
        )

        residual_y_logits = self.residual_head(
            residual
        )

        # ---------------------------------------------------------
        # Predict using intervened concepts and unchanged residuals
        # ---------------------------------------------------------
        y_pred_probs_sum = 0

        for sample_idx in range(self.num_monte_carlo):
            c_i = c[:, :, sample_idx]

            concept_y_logits_i = self.concept_head(c_i)

            y_pred_logits_i = (
                concept_y_logits_i + residual_y_logits
            )

            if self.regression_task:
                y_pred_probs_sum += y_pred_logits_i
            elif self.multilabel_task or self.pred_dim == 1:
                y_pred_probs_sum += torch.sigmoid(
                    y_pred_logits_i
                )
            else:
                y_pred_probs_sum += torch.softmax(
                    y_pred_logits_i,
                    dim=1,
                )

        y_pred_probs = (
            y_pred_probs_sum / self.num_monte_carlo
        )

        if self.regression_task:
            y_pred_logits = y_pred_probs
        elif self.multilabel_task or self.pred_dim == 1:
            y_pred_logits = torch.logit(
                y_pred_probs,
                eps=1e-6,
            )
        else:
            y_pred_logits = torch.log(
                y_pred_probs + 1e-6
            )

        return y_pred_logits

    def compute_temperature(self, epoch, device):
        """Anneal the RelaxedBernoulli temperature from 1.0 to 0.5."""

        final_temp = torch.tensor(0.5, device=device)
        init_temp = torch.tensor(1.0, device=device)

        rate = (
            torch.log(final_temp) - torch.log(init_temp)
        ) / float(self.num_epochs)

        curr_temp = torch.maximum(
            init_temp * torch.exp(rate * epoch),
            final_temp,
        )

        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self):
        """
        Sequential stage 1: train only the concept predictor.

        The residual pathway and both task heads are frozen.
        """

        self.encoder.apply(unfreeze_module)
        self.concept_predictor.apply(unfreeze_module)

        self.concept_head.apply(freeze_module)

        self.residual_encoder.apply(freeze_module)
        self.residual_predictor.apply(freeze_module)
        self.residual_head.apply(freeze_module)

    def freeze_t(self):
        """
        Sequential stage 2: freeze the concept predictor and train the
        concept task head together with the independent residual pathway.
        """

        self.encoder.apply(freeze_module)
        self.concept_predictor.apply(freeze_module)

        self.concept_head.apply(unfreeze_module)

        self.residual_encoder.apply(unfreeze_module)
        self.residual_predictor.apply(unfreeze_module)
        self.residual_head.apply(unfreeze_module)




















class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
