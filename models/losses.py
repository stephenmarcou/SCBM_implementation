"""
Utility methods for constructing loss functions
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from utils.utils import numerical_stability_check


def create_loss(config):
    """
    Create and return a loss function based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        nn.Module: The loss function.
    """
    if config.model.model == "cbm":
        return CBLoss(
            num_classes=config.data.num_classes,
            reduction="mean",
            alpha=config.model.alpha,
            config=config.model,
        )
    elif config.model.model == "scbm":
        return SCBLoss(
            num_classes=config.data.num_classes,
            alpha=config.model.alpha,
            config=config.model,
        )
    
    elif config.model.model == "scbm_residual":
        return SCBresLoss(
            num_classes=config.data.num_classes,
            alpha=config.model.alpha,
            config=config.model,
            num_concepts=config.data.num_concepts,
            num_residuals=config.data.num_residuals,
        )
    else:
        raise NotImplementedError




class CBLoss(nn.Module):
    """
    Loss function for the Concept Bottleneck Model (CBM).
    """

    def __init__(
        self,
        num_classes: Optional[int] = 2,
        reduction: str = "mean",
        alpha: float = 1,
        config: dict = {},
    ) -> None:
        """
        Initialize the CBLoss.

        Args:
            num_classes (int, optional): Number of target classes.
            reduction (str, optional): Reduction method for the loss.
            alpha (float, optional): Weight in joint training.
            config (dict, optional): Configuration dictionary.
        """
        super(CBLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha if config.training_mode == "joint" else 1.0
        self.reduction = reduction

    def forward(
        self,
        concepts_pred_probs: Tensor,
        concepts_true: Tensor,
        target_pred_logits: Tensor,
        target_true: Tensor,
    ) -> Tensor:
        """
        Compute the loss.

        Args:
            concepts_pred_probs (Tensor): Predicted concept probabilities.
            concepts_true (Tensor): Ground-truth concept values.
            target_pred_logits (Tensor): Predicted target logits.
            target_true (Tensor): Ground-truth target values.

        Returns:
            Tensor: Target loss, concept loss, and total loss.
        """
        assert torch.all((concepts_true == 0) | (concepts_true == 1))
        concepts_loss = self.compute_concept_loss(concepts_true, concepts_pred_probs)

        if self.num_classes == 2:
            # Logits to probs
            target_pred_probs = nn.Sigmoid()(target_pred_logits.squeeze(1))
            target_loss = F.binary_cross_entropy(
                target_pred_probs, target_true.float(), reduction=self.reduction
            )
        else:
            target_loss = F.cross_entropy(
                target_pred_logits, target_true.long(), reduction=self.reduction
            )

        total_loss = target_loss + concepts_loss

        return target_loss, concepts_loss, total_loss


    def compute_concept_loss(self, concepts_true, concepts_pred_probs):
        concepts_true = concepts_true.float() # [B, C]    
        concepts_loss = F.binary_cross_entropy(
            concepts_pred_probs, concepts_true , reduction='none'
        ) #  [B, C]
    
        if self.reduction == 'mean':
            concepts_loss = concepts_loss.mean(dim=0).sum()
        elif self.reduction == 'sum':
            concepts_loss = concepts_loss.sum(dim=0).sum()
    
        return (self.alpha * concepts_loss)

class SCBLoss(nn.Module):
    """
    Loss function for the Stochastic Concept Bottleneck Model (SCBM).
    """

    def __init__(
        self, num_classes: Optional[int] = 2, alpha: float = 1, config: dict = {}
    ) -> None:
        """
        Initialize the SCBLoss.

        Args:
            num_classes (int, optional): Number of target classes.
            alpha (float, optional): Weight for joint training.
            config (dict, optional): Configuration dictionary.
        """
        super(SCBLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_precision = config.reg_precision
        self.reg_weight = config.reg_weight

        self.log_num_mc = math.log(config.num_monte_carlo)
        self.alpha = alpha if config.training_mode == "joint" else 1.0

    def forward(
        self,
        concepts_mcmc_probs: Tensor,
        concepts_true: Tensor,
        target_pred_logits: Tensor,
        target_true: Tensor,
        c_triang_cov: Tensor,
        cov_not_triang=False,
    ) -> Tensor:
        """
        Compute the loss.

        Args:
            concepts_mcmc_probs (Tensor): MCMC matrix of predicted concept probabilities.
            concepts_true (Tensor): Ground-truth concept values.
            target_pred_logits (Tensor): Predicted target logits.
            target_true (Tensor): Ground-truth target values.
            c_triang_cov (Tensor): Cholesky decomposition of the concept covariance matrix.
            cov_not_triang (bool, optional): Flag indicating if the covariance is in its cholesky form or already the covariance.

        Returns:
            Tensor: Target loss, concept loss, precision loss, and total loss.
        """
        concepts_loss = self.compute_concept_loss(concepts_mcmc_probs, concepts_true)

        if self.num_classes == 2:
            # Logits to probs
            target_pred_probs = nn.Sigmoid()(target_pred_logits.squeeze(1))
            target_loss = F.binary_cross_entropy(
                target_pred_probs, target_true.float(), reduction="mean"
            )
        else:
            target_loss = F.cross_entropy(
                target_pred_logits, target_true.long(), reduction="mean"
            )

        # Add precision loss
        if self.reg_precision == "l1":
            if cov_not_triang:
                prec_matrix = torch.inverse(c_triang_cov)
            else:
                c_triang_inv = torch.inverse(c_triang_cov)
                prec_matrix = torch.matmul(
                    torch.transpose(c_triang_inv, dim0=1, dim1=2), c_triang_inv
                )
            prec_loss = prec_matrix.abs().sum(dim=(1, 2)) - prec_matrix.diagonal(
                offset=0, dim1=1, dim2=2
            ).abs().sum(-1)
            if prec_matrix.size(1) > 1:
                prec_loss = prec_loss / (
                    prec_matrix.size(1) * (prec_matrix.size(1) - 1)
                )
            else:  # Univariate case, can happen when intervening
                prec_loss = prec_loss
            prec_loss = self.reg_weight * prec_loss.mean(-1)
        else:
            prec_loss = torch.zeros_like(concepts_loss)

        total_loss = target_loss + concepts_loss + prec_loss

        return target_loss, concepts_loss, prec_loss, total_loss
    
    
    
    def compute_concept_loss(self, concepts_mcmc_probs, concepts_true):
        assert torch.all((concepts_true == 0) | (concepts_true == 1))
        concepts_true_expanded = concepts_true.unsqueeze(-1).expand_as(
            concepts_mcmc_probs
        )

        bce_loss = F.binary_cross_entropy(
            concepts_mcmc_probs, concepts_true_expanded.float(), reduction="none"
        )  # [B,C,MCMC]
        intermediate_concepts_loss = -torch.sum(bce_loss, dim=1)  # [B,MCMC]
        mcmc_loss = -torch.logsumexp(
            intermediate_concepts_loss, dim=1
        )  # [B], logsumexp for numerical stability due to shift invariance
        # The concept loss computation is bounded by - log_num_mc adding log_num_mc moves
        # bound to 0. Preventing negative losses.
        return self.alpha * (torch.mean(mcmc_loss) + self.log_num_mc)
    
    

class SCBresLoss(nn.Module):
    """
    Loss function for the Stochastic Concept Bottleneck Model (SCBM).
    """

    def __init__(
        self, num_classes: Optional[int] = 2, alpha: float = 1, config: dict = {}, num_concepts: int = 0, num_residuals: int = 0,
    ) -> None:
        """
        Initialize the SCBLoss.

        Args:
            num_classes (int, optional): Number of target classes.
            alpha (float, optional): Weight for joint training.
            config (dict, optional): Configuration dictionary.
        """
        super(SCBresLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_precision = config.reg_precision
        self.reg_weight = config.reg_weight
        self.num_concepts = num_concepts
        self.num_residuals = num_residuals
        self.L_int_loss_weight = config.L_int_loss_weight
        self.use_L_int_loss = config.use_L_int_loss
        self.concept_learning = config.concept_learning

        self.log_num_mc = math.log(config.num_monte_carlo)
        self.alpha = alpha if config.training_mode == "joint" else 1.0

    def forward(
        self,
        concepts_mcmc_probs: Tensor,
        concepts_true: Tensor,
        target_pred_logits: Tensor,
        target_true: Tensor,
        c_triang_cov: Tensor,
        cov_not_triang=False,
    ) -> Tensor:
        """
        Compute the loss.

        Args:
            concepts_mcmc_probs (Tensor): MCMC matrix of predicted concept probabilities. Shape: [B, C, MCMC]
            concepts_true (Tensor): Ground-truth concept values.
            target_pred_logits (Tensor): Predicted target logits.
            target_true (Tensor): Ground-truth target values.
            c_triang_cov (Tensor): Cholesky decomposition of the concept covariance matrix.
            cov_not_triang (bool, optional): Flag indicating if the covariance is in its cholesky form or already the covariance.

        Returns:
            Tensor: Target loss, concept loss, precision loss, and total loss.
        """
        #concepts_mcmc_probs = concepts_residuals_mcmc_probs[:, :self.num_concepts, :]
        concepts_loss = self.compute_concept_loss(concepts_mcmc_probs, concepts_true)
        


        if self.num_classes == 2:
            # Logits to probs
            target_pred_probs = nn.Sigmoid()(target_pred_logits.squeeze(1))
            target_loss = F.binary_cross_entropy(
                target_pred_probs, target_true.float(), reduction="mean"
            )
        else:
            target_loss = F.cross_entropy(
                target_pred_logits, target_true.long(), reduction="mean"
            )

        # Add precision loss
        if self.reg_precision == "l1":
            if cov_not_triang:
                prec_matrix = torch.inverse(c_triang_cov)
            else:
                c_triang_inv = torch.inverse(c_triang_cov)
                prec_matrix = torch.matmul(
                    torch.transpose(c_triang_inv, dim0=1, dim1=2), c_triang_inv
                )
            prec_loss = prec_matrix.abs().sum(dim=(1, 2)) - prec_matrix.diagonal(
                offset=0, dim1=1, dim2=2
            ).abs().sum(-1)
            if prec_matrix.size(1) > 1:
                prec_loss = prec_loss / (
                    prec_matrix.size(1) * (prec_matrix.size(1) - 1)
                )
            else:  # Univariate case, can happen when intervening
                prec_loss = prec_loss
            prec_loss = self.reg_weight * prec_loss.mean(-1)
        else:
            prec_loss = torch.zeros_like(concepts_loss)

        total_loss = target_loss + concepts_loss + prec_loss

        return target_loss, concepts_loss, prec_loss, total_loss
        

    def compute_L_int_loss(
        self,
        model,
        c_res_mcmc,
        c_res_mcmc_logits,
        c_true,
        y_true,
    ):
        """
        Computes L_int: task loss under full concept intervention.

        Full intervention:
            [c_pred, z_pred] -> [c_true, z_pred]

        Shapes:
            c_res_mcmc:        [B, C + R, M]
            c_res_mcmc_logits: [B, C + R, M]
            c_true:            [B, C]
            y_true:            [B] for multiclass, [B] or [B, 1] for binary
        """

        B, CR, M = c_res_mcmc.shape
        C = self.num_concepts

        assert CR == self.num_concepts + self.num_residuals
        assert c_true.shape[1] == C

        # [B, C, M]
        c_true_mcmc = c_true.float().unsqueeze(-1).expand(-1, -1, M)

        if self.concept_learning == "hard":
            # compute_y_pred_logits will use the first argument: c_res_mcmc

            # Keep residual samples/probs unchanged so gradients flow through residual path
            z = c_res_mcmc[:, C:, :]

            # Replace known concepts with ground truth
            c_res_intervened = torch.cat(
                [c_true_mcmc, z],
                dim=1,
            )

            # Logits are ignored in hard mode, but pass original tensor for shape consistency
            y_int_logits = model.compute_y_pred_logits(
                c_res_intervened,
                c_res_mcmc_logits,
            )

        # Have to check this when concept learning is not hard!
        else:
            # compute_y_pred_logits will use the second argument: c_res_mcmc_logits

            # Convert binary c_true into finite logits
            eps = 1e-4
            c_true_logits = torch.logit(
                c_true_mcmc.clamp(min=eps, max=1.0 - eps)
            )

            # Keep residual logits unchanged so gradients flow through residual path
            z_logits = c_res_mcmc_logits[:, C:, :]

            # Replace known concept logits with ground-truth concept logits
            c_res_intervened_logits = torch.cat(
                [c_true_logits, z_logits],
                dim=1,
            )

            # Probs are ignored in soft/logit mode, but pass original tensor
            y_int_logits = model.compute_y_pred_logits(
                c_res_mcmc,
                c_res_intervened_logits,
            )

        if self.num_classes == 2:
            y_true = y_true.float().view(-1, 1)
            return F.binary_cross_entropy_with_logits(
                y_int_logits,
                y_true,
            )

        else:
            # y_int_logits are the log probabilites
            # view(-1) to ensure shape [B] for cross_entropy, works for both [B] and [B, 1]
            y_true = y_true.long().view(-1)
            return F.nll_loss(
                y_int_logits,
                y_true,
            )
            

    def compute_L_int_extension_loss(
        self,
        model, 
        triang_cov,
        c_res_mu,
        target_true,
        concepts_true,
        device,
        intervention_strategy,
    ):
        
        # Keep float32 for GPU/MPS - float64 is too slow on GPU
        # If CPU needs numerical stability, convert only for CPU
        if device.type == "cpu":
            triang_cov = triang_cov.to(torch.float64)
            c_res_mu = c_res_mu.to(torch.float64)
        else:
            # GPU and MPS use float32 for performance
            triang_cov = triang_cov.to(torch.float32)
            c_res_mu = c_res_mu.to(torch.float32)
        
        # Retrieve original (pre-intervention) covariance matrix through Cholesky decomposition
        c_res_cov = torch.matmul(
            triang_cov,
            torch.transpose(triang_cov, dim0=1, dim1=2),
        )
        c_res_cov = numerical_stability_check(c_res_cov, device=device)
        
        # Intervene on all concepts
        concepts_mask = torch.ones_like(concepts_true) # [B, num_concepts]
        
        
        (
            c_res_interv_mu,
            c_res_interv_cov,
            c_res_mcmc_probs,
            c_res_mcmc_logits,
        ) = intervention_strategy.compute_intervention(
            c_res_mu,
            c_res_cov,
            concepts_true,
            concepts_mask,
        )
        
        #target_pred_logits = model.intervene_straight_through(c_res_mcmc_probs, c_res_mcmc_logits)
        target_pred_logits = model.intervene_straight_through_vectorized(c_res_mcmc_probs, c_res_mcmc_logits)
        L_int_extension_loss = F.nll_loss(target_pred_logits, target_true)
    
        return L_int_extension_loss





    def compute_concept_loss(self, concepts_mcmc_probs, concepts_true):
        assert torch.all((concepts_true == 0) | (concepts_true == 1))
        # Shape of concepts_true: [B, C], concepts_true_expanded: [B, C, MCMC], concepts_mcmc_probs: [B, C, MCMC]
        concepts_true_expanded = concepts_true.unsqueeze(-1).expand_as(
            concepts_mcmc_probs
        )

        bce_loss = F.binary_cross_entropy(
            concepts_mcmc_probs, concepts_true_expanded.float(), reduction="none"
        )  # [B,C,MCMC]
        intermediate_concepts_loss = -torch.sum(bce_loss, dim=1)  # [B,MCMC]
        mcmc_loss = -torch.logsumexp(
            intermediate_concepts_loss, dim=1
        )  # [B], logsumexp for numerical stability due to shift invariance
        # The concept loss computation is bounded by - log_num_mc adding log_num_mc moves
        # bound to 0. Preventing negative losses.
        return self.alpha * (torch.mean(mcmc_loss) + self.log_num_mc)
