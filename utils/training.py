"""
Utility functions for training.
"""

import numpy as np
from sklearn.metrics import jaccard_score
import torch
from torch import nn
from tqdm import tqdm
from torchmetrics import Metric
import wandb
import os

from utils.metrics import calc_target_metrics, calc_concept_metrics, calc_regression_target_metrics
from utils.plotting import compute_and_plot_heatmap
from utils.utils import numerical_stability_check

from utils.intervention import define_strategy
from pathlib import Path






def train_one_epoch_scbm_residual(
    train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device, log_file = None
):
    """
    Train the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function trains the SCBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    if (
        config.model.training_mode == "sequential"
        or config.model.training_mode == "independent"
    ):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    # Define intervention strategy for L_int_extension_loss if needed
    if config.model.use_L_int_extension_loss == True:
        #strategy = "conf_interval_optimal"
        strategy = config.model.inter_strategy #"emp_perc"
        intervention_strategy = define_strategy(
                strategy, train_loader, model, device, config
            )


    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        (
            concepts_residuals_mcmc_probs,
            concepts_residuals_mcmc,
            concepts_residuals_mcmc_logits,
            triang_cov,
            target_pred_logits,
            c_res_mu,
        ) = model(batch_features, epoch, c_true=concepts_true, return_L_int_extension=True)
        
        concepts_mcmc_probs = concepts_residuals_mcmc_probs[:, :config.data.num_concepts, :]
        residual_mcmc_probs = concepts_residuals_mcmc_probs[:, config.data.num_concepts:, :]

        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()


        # target_pred_logits for multiclass classification are logprobabilities
        # while for binary classification they are logits
        # However we can still use cross_entropy for multiclass classification since it is identical
        # to NLL when the input is log_probabilities.
        # For regression tasks, target_pred_logits are the average predicted values across monte carlo samples and we use MSE loss.
    
                

        # Compute the loss
        target_loss, concepts_loss, prec_loss, sparsity_loss, total_loss = loss_fn(
            concepts_mcmc_probs,
            concepts_true,
            target_pred_logits,
            target_true,
            triang_cov,
            residual_mcmc_probs=residual_mcmc_probs
        )
        
        # Intervene on concepts to get L_int loss
        if config.model.use_L_int_loss == True:
            if k == 0:
                print("Using L_int_loss with weight: ", config.model.L_int_loss_weight)
            L_int_loss = loss_fn.compute_L_int_loss(
                model,
                concepts_residuals_mcmc,
                concepts_residuals_mcmc_logits,
                concepts_true,
                target_true,
            )
            total_loss = total_loss + config.model.L_int_loss_weight * L_int_loss
            
        # Intervene on concepts and propagate effect to residuals to get L_int_extension loss
        elif config.model.use_L_int_extension_loss == True:
            if k == 0:
                print("Using L_int_extension_loss with weight: ", config.model.L_int_extension_loss_weight)

            L_int_extension_loss = loss_fn.compute_L_int_extension_loss(
                 model, triang_cov, c_res_mu, target_true, concepts_true, device, intervention_strategy, half_intervention=config.model.half_intervention_l_int_extension_loss
            )
    
            total_loss = total_loss + config.model.L_int_extension_loss_weight * L_int_extension_loss
            
        if config.model.use_L_int_loss == False:
            L_int_loss = None
        if config.model.use_L_int_extension_loss == False:
            L_int_extension_loss = None



        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            (concepts_loss + prec_loss).backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        concepts_pred_probs = concepts_mcmc_probs.mean(-1)
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits,
            concepts_true,
            concepts_pred_probs,
            prec_loss=prec_loss,
            L_int_loss = L_int_loss,
            L_int_extension_loss = L_int_extension_loss,
            
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(prints + "\n")
    metrics.reset()
    return









def train_one_epoch_scbm(
    train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device, log_file = None
):
    """
    Train the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function trains the SCBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    if (
        config.model.training_mode == "sequential"
        or config.model.training_mode == "independent"
    ):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        concepts_mcmc_probs, triang_cov, target_pred_logits = model(
            batch_features, epoch, c_true=concepts_true
        )

        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()

        # Compute the loss
        target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
            concepts_mcmc_probs,
            concepts_true,
            target_pred_logits,
            target_true,
            triang_cov,
        )

        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            (concepts_loss + prec_loss).backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        concepts_pred_probs = concepts_mcmc_probs.mean(-1)
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits,
            concepts_true,
            concepts_pred_probs,
            prec_loss=prec_loss,
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(prints + "\n")
    metrics.reset()
    return


def train_one_epoch_cbm(
    train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device, log_file=None
):
    """
    Train a baseline method for one epoch.

    This function trains the CEM/AR/CBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    if config.model.training_mode in ("sequential", "independent"):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        if config.model.training_mode == "independent" and mode == "t":
            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_features, epoch, concepts_true
            )
        elif config.model.concept_learning == "autoregressive" and mode == "c":
            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_features, epoch, concepts_train_ar=concepts_true
            )
        else:
            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_features, epoch
            )
        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()
        # Compute the loss
        target_loss, concepts_loss, total_loss = loss_fn(
            concepts_pred_probs, concepts_true, target_pred_logits, target_true
        )

        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            concepts_loss.backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits,
            concepts_true,
            concepts_pred_probs,
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(prints + "\n")
    metrics.reset()
    return




def validate_one_epoch_scbm_residual(
    loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    test=False,
    concept_names_graph=None,
    log_file=None,
    save_residual_meta_data_folder=None,
    metrics_only_for_saving=False,
):
    """
    Validate the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function evaluates the SCBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process. It also generates
    and plots a heatmap of the learned concept correlation matrix on the final test set.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The SCBM model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.
        concept_names_graph (list, optional): List of concept names for plotting the heatmap.
                                              Default is None for which range(n_concepts) is used.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
        - During testing, the function generates and plots a heatmap of the concept correlation matrix.
    """

    
    
    model.eval()
    
    # Metadata lists to store concept residuals and predictions for saving
    concepts_residual_probs_mean_list = []
    concepts_residual_prob_std_list = []
    concepts_residual_mean_list = []
    concepts_residual_std_list = []
    concepts_residuals_logits_mean_list = []
    concepts_residuals_logits_std_list = []
    c_res_mu_list = []
    
    y_pred_list = []
    y_true_list = []
    
    cov_matrix_sum = None
    cov_matrix_count = 0
    
    # Define intervention strategy for L_int_extension_loss if needed
    if config.model.use_L_int_extension_loss == True:
        strategy = config.model.inter_strategy #"emp_perc"
        intervention_strategy = define_strategy(
                strategy, loader, model, device, config
            )
    
    
    with torch.no_grad():

        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)

            
            # Forward pass
            (
                concepts_residuals_mcmc_probs,
                concepts_residuals_mcmc,
                concepts_residuals_mcmc_logits,
                triang_cov,
                target_pred_logits,
                c_res_mu,
            ) = model(batch_features, epoch, validation=True, c_true=concepts_true, return_L_int_extension=True)
                
            concepts_mcmc_probs = concepts_residuals_mcmc_probs[:, :config.data.num_concepts, :]
            residual_mcmc_probs = concepts_residuals_mcmc_probs[:, config.data.num_concepts:, :]
            
             # Compute covariance matrix of concepts
            cov = torch.matmul(triang_cov, torch.transpose(triang_cov, dim0=1, dim1=2))
            
            # Save the concept-residual channel 
            if config.data.save_concept_and_residual_channel and save_residual_meta_data_folder:
                concepts_residuals_mcmc_probs_detached = concepts_residuals_mcmc_probs.detach()

                concepts_residuals_mcmc_detached = concepts_residuals_mcmc.detach()
                
                concepts_residuals_mcmc_logits_detached = concepts_residuals_mcmc_logits.detach()
                
                c_res_mu_detached = c_res_mu.detach()
                
                c_res_mu_list.append(c_res_mu_detached.cpu())

                concepts_residuals_pred_probs = concepts_residuals_mcmc_probs_detached.mean(dim=-1)
                concepts_residuals_prob_std = concepts_residuals_mcmc_probs_detached.std(dim=-1, unbiased=False)
                
                
                concepts_residuals_sample_mean = concepts_residuals_mcmc_detached.float().mean(dim=-1)
                concepts_residuals_sample_std = concepts_residuals_mcmc_detached.float().std(dim=-1, unbiased=False)
                
                concepts_residuals_logits_mean = concepts_residuals_mcmc_logits_detached.float().mean(dim=-1)
                concepts_residuals_logits_std = concepts_residuals_mcmc_logits_detached.float().std(dim=-1, unbiased=False)
                
                
                concepts_residual_probs_mean_list.append(concepts_residuals_pred_probs.cpu())
                concepts_residual_prob_std_list.append(concepts_residuals_prob_std.cpu())
                
                concepts_residual_mean_list.append(concepts_residuals_sample_mean.cpu())
                concepts_residual_std_list.append(concepts_residuals_sample_std.cpu())
                
                concepts_residuals_logits_mean_list.append(concepts_residuals_logits_mean.cpu())
                concepts_residuals_logits_std_list.append(concepts_residuals_logits_std.cpu())
                
                y_pred_list.append(target_pred_logits.cpu())
                y_true_list.append(target_true.cpu())
                
                sigma_mean_batch = cov.mean(dim=0).cpu() #[C+R, C+R]
                if cov_matrix_sum is None:
                    cov_matrix_sum = sigma_mean_batch
                else:
                    cov_matrix_sum += sigma_mean_batch
                cov_matrix_count += 1
                                
                
                
                
                
                
                
            
            target_loss, concepts_loss, prec_loss, sparsity_loss, total_loss = loss_fn(
                concepts_mcmc_probs,
                concepts_true,
                target_pred_logits,
                target_true,
                triang_cov,
                residual_mcmc_probs=residual_mcmc_probs
            )
            
            # Intervene on concepts to get L_int loss
            if config.model.use_L_int_loss == True:
                if k == 0:
                    print("Using L_int_loss with weight: ", config.model.L_int_loss_weight)
                L_int_loss = loss_fn.compute_L_int_loss(
                    model,
                    concepts_residuals_mcmc,
                    concepts_residuals_mcmc_logits,
                    concepts_true,
                    target_true,
                )
                total_loss = total_loss + config.model.L_int_loss_weight * L_int_loss
                
            # Intervene on concepts and propagate effect to residuals to get L_int_extension loss
            elif config.model.use_L_int_extension_loss == True:
                if k == 0:
                    print("Using L_int_extension_loss with weight: ", config.model.L_int_extension_loss_weight)

                L_int_extension_loss = loss_fn.compute_L_int_extension_loss(
                    model, triang_cov, c_res_mu, target_true, concepts_true, device, intervention_strategy
                )
        
                total_loss = total_loss + config.model.L_int_extension_loss_weight * L_int_extension_loss
            
            if config.model.use_L_int_loss == False:
                L_int_loss = None
            if config.model.use_L_int_extension_loss == False:
                L_int_extension_loss = None

                

            # Store predictions
            concepts_pred_probs = concepts_mcmc_probs.mean(-1)
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs,
                prec_loss=prec_loss,
                L_int_loss = L_int_loss,
                L_int_extension_loss = L_int_extension_loss,  
            )

    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)

    # When metrics_only_for_saving is True, we only want to save the metadata for concept discovery analysis but not log metrics
    # This is only true for training and validation sets, not for the final test set. 
    if not metrics_only_for_saving:
        if not test:
            wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
            prints = f"Epoch {epoch}, Validation: "
        else:
            wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
            prints = f"Test: "
        for key, value in metrics_dict.items():
            prints += f"{key}: {value:.3f} "
        
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(prints + "\n")
                
        print(prints)
        print()




    if config.data.save_concept_and_residual_channel and save_residual_meta_data_folder:
        concepts_residuals_mean_tensor = torch.cat(concepts_residual_mean_list, dim=0)
        concepts_residuals_std_tensor = torch.cat(concepts_residual_std_list, dim=0)
        concepts_residuals_probs_mean_tensor = torch.cat(concepts_residual_probs_mean_list, dim=0)
        concepts_residuals_probs_std_tensor = torch.cat(concepts_residual_prob_std_list, dim=0)
        concepts_residuals_logits_mean_tensor = torch.cat(concepts_residuals_logits_mean_list, dim=0)
        concepts_residuals_logits_std_tensor = torch.cat(concepts_residuals_logits_std_list, dim=0)
        c_res_mu_tensor = torch.cat(c_res_mu_list, dim=0)
        y_pred_tensor = torch.cat(y_pred_list, dim=0)
        y_true_tensor = torch.cat(y_true_list, dim=0)

        parent_dir_path = os.path.dirname(log_file)
        full_path = os.path.join(parent_dir_path, save_residual_meta_data_folder)
        Path(full_path).mkdir(parents=True, exist_ok=True)




        save_path_concepts_residual_mean = os.path.join(full_path, "concepts_residuals_sample_mean.pt")
        save_path_concepts_residual_std = os.path.join(full_path, "concepts_residuals_sample_std.pt")
        save_path_concepts_residual_probs_mean = os.path.join(full_path, "concepts_residuals_pred_probs_mean.pt")
        save_path_concepts_residual_probs_std = os.path.join(full_path, "concepts_residuals_pred_probs_std.pt")
        save_path_concepts_residuals_logits_mean = os.path.join(full_path, "concepts_residuals_logits_mean.pt")
        save_path_concepts_residuals_logits_std = os.path.join(full_path, "concepts_residuals_logits_std.pt")
        save_path_c_res_mu = os.path.join(full_path, "c_res_mu.pt")
        save_path_y_pred = os.path.join(full_path, "y_pred.pt")
        save_path_y_true = os.path.join(full_path, "y_true.pt")
        
        cov_matrix_avg = cov_matrix_sum / cov_matrix_count  # [C+R, C+R]
        save_path_cov = os.path.join(full_path, "avg_covariance_matrix.pt")
        
        
        
        torch.save(concepts_residuals_mean_tensor, save_path_concepts_residual_mean)
        torch.save(concepts_residuals_std_tensor, save_path_concepts_residual_std)
        torch.save(concepts_residuals_probs_mean_tensor, save_path_concepts_residual_probs_mean)
        torch.save(concepts_residuals_probs_std_tensor, save_path_concepts_residual_probs_std)
        torch.save(concepts_residuals_logits_mean_tensor, save_path_concepts_residuals_logits_mean)
        torch.save(concepts_residuals_logits_std_tensor, save_path_concepts_residuals_logits_std)
        torch.save(c_res_mu_tensor, save_path_c_res_mu)
        torch.save(y_pred_tensor, save_path_y_pred)
        torch.save(y_true_tensor, save_path_y_true)
        torch.save(cov_matrix_avg, save_path_cov)

        print(f"Saved concepts residuals means to {save_path_concepts_residual_mean}")
        print(f"Saved concepts residuals stds to {save_path_concepts_residual_std}")
        print(f"Saved concepts residuals predicted probabilities means to {save_path_concepts_residual_probs_mean}")
        print(f"Saved concepts residuals predicted probabilities stds to {save_path_concepts_residual_probs_std}")
        print(f"Saved concepts residuals logits means to {save_path_concepts_residuals_logits_mean}")
        print(f"Saved concepts residuals logits stds to {save_path_concepts_residuals_logits_std}")
        
        print(f"Saved c_res_mu to {save_path_c_res_mu}")
        print(f"Saved y_pred to {save_path_y_pred}")
        print(f"Saved y_true to {save_path_y_true}")
        print(f"Saved average covariance matrix to {save_path_cov}")
        
    
    
    
    
    
    metrics.reset()
    return metrics_dict










def validate_one_epoch_scbm(
    loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    test=False,
    concept_names_graph=None,
    log_file=None,
    save_concept_meta_data_folder=None,
    metrics_only_for_saving=False,
    **kwargs
):
    """
    Validate the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function evaluates the SCBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process. It also generates
    and plots a heatmap of the learned concept correlation matrix on the final test set.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The SCBM model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.
        concept_names_graph (list, optional): List of concept names for plotting the heatmap.
                                              Default is None for which range(n_concepts) is used.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
        - During testing, the function generates and plots a heatmap of the concept correlation matrix.
    """
    model.eval()

    # Metadata lists to store concept predictions for saving
    concepts_pred_probs_mean_list = []
    concepts_pred_probs_std_list = []
    concepts_sample_mean_list = []
    concepts_sample_std_list = []
    concepts_logits_mean_list = []
    concepts_logits_std_list = []
    c_mu_list = []

    y_pred_list = []
    y_true_list = []

    cov_matrix_sum = None
    cov_matrix_count = 0

    with torch.no_grad():

        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)
            (
                concepts_mcmc_probs,
                concepts_mcmc,
                concepts_mcmc_logits,
                triang_cov,
                target_pred_logits,
                c_mu,
            ) = model(
                batch_features, epoch, validation=True, c_true=concepts_true, return_samples=True
            )
            # Compute covariance matrix of concepts
            cov = torch.matmul(triang_cov, torch.transpose(triang_cov, dim0=1, dim1=2))



            if test and k % (len(loader) // 10) == 0:
                try:
                    corr = (cov[0] / cov[0].diag().sqrt()).transpose(
                        dim0=0, dim1=1
                    ) / cov[0].diag().sqrt()
                    matrix = corr.cpu().numpy()

                    compute_and_plot_heatmap(
                        matrix, concepts_true, concept_names_graph, config
                    )

                except:
                    pass

            # Save the concept channel
            if config.data.save_concept_and_residual_channel and save_concept_meta_data_folder:
                concepts_mcmc_probs_detached = concepts_mcmc_probs.detach()
                concepts_mcmc_detached = concepts_mcmc.detach()
                concepts_mcmc_logits_detached = concepts_mcmc_logits.detach()
                c_mu_detached = c_mu.detach()

                c_mu_list.append(c_mu_detached.cpu())

                concepts_pred_probs_mean = concepts_mcmc_probs_detached.mean(dim=-1)
                concepts_pred_probs_std = concepts_mcmc_probs_detached.std(dim=-1, unbiased=False)

                concepts_sample_mean = concepts_mcmc_detached.float().mean(dim=-1)
                concepts_sample_std = concepts_mcmc_detached.float().std(dim=-1, unbiased=False)

                concepts_logits_mean = concepts_mcmc_logits_detached.float().mean(dim=-1)
                concepts_logits_std = concepts_mcmc_logits_detached.float().std(dim=-1, unbiased=False)

                concepts_pred_probs_mean_list.append(concepts_pred_probs_mean.cpu())
                concepts_pred_probs_std_list.append(concepts_pred_probs_std.cpu())

                concepts_sample_mean_list.append(concepts_sample_mean.cpu())
                concepts_sample_std_list.append(concepts_sample_std.cpu())

                concepts_logits_mean_list.append(concepts_logits_mean.cpu())
                concepts_logits_std_list.append(concepts_logits_std.cpu())

                y_pred_list.append(target_pred_logits.cpu())
                y_true_list.append(target_true.cpu())

                sigma_mean_batch = cov.mean(dim=0).cpu()  # [C, C]
                if cov_matrix_sum is None:
                    cov_matrix_sum = sigma_mean_batch
                else:
                    cov_matrix_sum += sigma_mean_batch
                cov_matrix_count += 1

            target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
                concepts_mcmc_probs,
                concepts_true,
                target_pred_logits,
                target_true,
                triang_cov,
            )

            # Store predictions
            concepts_pred_probs = concepts_mcmc_probs.mean(-1)
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs,
                prec_loss=prec_loss,
            )

    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)

    if not metrics_only_for_saving:
        if not test:
            wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
            prints = f"Epoch {epoch}, Validation: "
        else:
            wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
            prints = f"Test: "
        for key, value in metrics_dict.items():
            prints += f"{key}: {value:.3f} "

        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(prints + "\n")

        print(prints)
        print()

    if config.data.save_concept_and_residual_channel and save_concept_meta_data_folder:
        concepts_sample_mean_tensor = torch.cat(concepts_sample_mean_list, dim=0)
        concepts_sample_std_tensor = torch.cat(concepts_sample_std_list, dim=0)
        concepts_pred_probs_mean_tensor = torch.cat(concepts_pred_probs_mean_list, dim=0)
        concepts_pred_probs_std_tensor = torch.cat(concepts_pred_probs_std_list, dim=0)
        concepts_logits_mean_tensor = torch.cat(concepts_logits_mean_list, dim=0)
        concepts_logits_std_tensor = torch.cat(concepts_logits_std_list, dim=0)
        c_mu_tensor = torch.cat(c_mu_list, dim=0)
        y_pred_tensor = torch.cat(y_pred_list, dim=0)
        y_true_tensor = torch.cat(y_true_list, dim=0)

        parent_dir_path = os.path.dirname(log_file)
        full_path = os.path.join(parent_dir_path, save_concept_meta_data_folder)
        Path(full_path).mkdir(parents=True, exist_ok=True)

        save_path_concepts_sample_mean = os.path.join(full_path, "concepts_sample_mean.pt")
        save_path_concepts_sample_std = os.path.join(full_path, "concepts_sample_std.pt")
        save_path_concepts_pred_probs_mean = os.path.join(full_path, "concepts_pred_probs_mean.pt")
        save_path_concepts_pred_probs_std = os.path.join(full_path, "concepts_pred_probs_std.pt")
        save_path_concepts_logits_mean = os.path.join(full_path, "concepts_logits_mean.pt")
        save_path_concepts_logits_std = os.path.join(full_path, "concepts_logits_std.pt")
        save_path_c_mu = os.path.join(full_path, "c_mu.pt")
        save_path_y_pred = os.path.join(full_path, "y_pred.pt")
        save_path_y_true = os.path.join(full_path, "y_true.pt")

        cov_matrix_avg = cov_matrix_sum / cov_matrix_count  # [C, C]
        save_path_cov = os.path.join(full_path, "avg_covariance_matrix.pt")

        torch.save(concepts_sample_mean_tensor, save_path_concepts_sample_mean)
        torch.save(concepts_sample_std_tensor, save_path_concepts_sample_std)
        torch.save(concepts_pred_probs_mean_tensor, save_path_concepts_pred_probs_mean)
        torch.save(concepts_pred_probs_std_tensor, save_path_concepts_pred_probs_std)
        torch.save(concepts_logits_mean_tensor, save_path_concepts_logits_mean)
        torch.save(concepts_logits_std_tensor, save_path_concepts_logits_std)
        torch.save(c_mu_tensor, save_path_c_mu)
        torch.save(y_pred_tensor, save_path_y_pred)
        torch.save(y_true_tensor, save_path_y_true)
        torch.save(cov_matrix_avg, save_path_cov)

        print(f"Saved concepts sample means to {save_path_concepts_sample_mean}")
        print(f"Saved concepts sample stds to {save_path_concepts_sample_std}")
        print(f"Saved concepts predicted probabilities means to {save_path_concepts_pred_probs_mean}")
        print(f"Saved concepts predicted probabilities stds to {save_path_concepts_pred_probs_std}")
        print(f"Saved concepts logits means to {save_path_concepts_logits_mean}")
        print(f"Saved concepts logits stds to {save_path_concepts_logits_std}")
        print(f"Saved c_mu to {save_path_c_mu}")
        print(f"Saved y_pred to {save_path_y_pred}")
        print(f"Saved y_true to {save_path_y_true}")
        print(f"Saved average covariance matrix to {save_path_cov}")

    metrics.reset()
    return metrics_dict


def validate_one_epoch_cbm(
    loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    test=False,
    concept_names_graph=None,
    log_file=None,
    save_concept_meta_data_folder=None,
    metrics_only_for_saving=False,
    **kwargs
):
    """
    Validate a baseline method for one epoch.

    This function evaluates the CEM/AR/CBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
    """
    model.eval()

    # Metadata lists to store concept predictions for saving
    concepts_pred_probs_list = []
    concepts_logits_list = []
    concepts_sample_mean_list = []
    concepts_sample_std_list = []
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)

            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_features, epoch, validation=True
            )
            if config.model.concept_learning == "autoregressive":
                concepts_input = concepts_hard
            elif config.model.concept_learning == "hard":
                concepts_input = concepts_hard
            else:
                concepts_input = concepts_pred_probs
            if config.model.concept_learning == "autoregressive":
                concepts_pred_probs = torch.mean(
                    concepts_pred_probs, dim=-1
                )  # Calculating the metrics on the average probabilities from MCMC

            # Save the concept channel (deterministic probs/logits, plus MCMC sample
            # statistics for hard/AR concept learning where concepts_hard carries a
            # Monte Carlo dimension)
            if config.data.save_concept_and_residual_channel and save_concept_meta_data_folder:
                concepts_pred_probs_detached = concepts_pred_probs.detach()
                concepts_logits = torch.logit(concepts_pred_probs_detached, eps=1e-6)

                concepts_pred_probs_list.append(concepts_pred_probs_detached.cpu())
                concepts_logits_list.append(concepts_logits.cpu())

                if config.model.concept_learning in ("hard", "autoregressive"):
                    concepts_hard_detached = concepts_hard.detach().float()
                    concepts_sample_mean_list.append(
                        concepts_hard_detached.mean(dim=-1).cpu()
                    )
                    concepts_sample_std_list.append(
                        concepts_hard_detached.std(dim=-1, unbiased=False).cpu()
                    )

                y_pred_list.append(target_pred_logits.detach().cpu())
                y_true_list.append(target_true.cpu())

            target_loss, concepts_loss, total_loss = loss_fn(
                concepts_pred_probs, concepts_true, target_pred_logits, target_true
            )

            # Store predictions
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs,
            )

    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)

    if not metrics_only_for_saving:
        if not test:
            wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
            prints = f"Epoch {epoch}, Validation: "
        else:
            wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
            prints = f"Test: "
        for key, value in metrics_dict.items():
            prints += f"{key}: {value:.3f} "

        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(prints + "\n")

        print(prints)
        print()

    if config.data.save_concept_and_residual_channel and save_concept_meta_data_folder:
        concepts_pred_probs_tensor = torch.cat(concepts_pred_probs_list, dim=0)
        concepts_logits_tensor = torch.cat(concepts_logits_list, dim=0)
        y_pred_tensor = torch.cat(y_pred_list, dim=0)
        y_true_tensor = torch.cat(y_true_list, dim=0)

        parent_dir_path = os.path.dirname(log_file)
        full_path = os.path.join(parent_dir_path, save_concept_meta_data_folder)
        Path(full_path).mkdir(parents=True, exist_ok=True)

        save_path_concepts_pred_probs = os.path.join(full_path, "concepts_pred_probs.pt")
        save_path_concepts_logits = os.path.join(full_path, "concepts_logits.pt")
        save_path_y_pred = os.path.join(full_path, "y_pred.pt")
        save_path_y_true = os.path.join(full_path, "y_true.pt")

        torch.save(concepts_pred_probs_tensor, save_path_concepts_pred_probs)
        torch.save(concepts_logits_tensor, save_path_concepts_logits)
        torch.save(y_pred_tensor, save_path_y_pred)
        torch.save(y_true_tensor, save_path_y_true)

        print(f"Saved concepts predicted probabilities to {save_path_concepts_pred_probs}")
        print(f"Saved concepts logits to {save_path_concepts_logits}")
        print(f"Saved y_pred to {save_path_y_pred}")
        print(f"Saved y_true to {save_path_y_true}")

        if concepts_sample_mean_list:
            concepts_sample_mean_tensor = torch.cat(concepts_sample_mean_list, dim=0)
            concepts_sample_std_tensor = torch.cat(concepts_sample_std_list, dim=0)
            save_path_concepts_sample_mean = os.path.join(full_path, "concepts_sample_mean.pt")
            save_path_concepts_sample_std = os.path.join(full_path, "concepts_sample_std.pt")
            torch.save(concepts_sample_mean_tensor, save_path_concepts_sample_mean)
            torch.save(concepts_sample_std_tensor, save_path_concepts_sample_std)
            print(f"Saved concepts sample means to {save_path_concepts_sample_mean}")
            print(f"Saved concepts sample stds to {save_path_concepts_sample_std}")

    metrics.reset()
    return metrics_dict


def create_optimizer(config, model):
    """
    Parse the configuration file and return a optimizer object to update the model parameters.
    """
    assert config.optimizer in [
        "sgd",
        "adam",
    ], "Only SGD and Adam optimizers are available!"

    optim_params = [
        {
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    ]

    if config.optimizer == "sgd":
        return torch.optim.SGD(optim_params)
    elif config.optimizer == "adam":
        return torch.optim.Adam(optim_params)


class Custom_Metrics(Metric):
    """
    Custom metrics class for tracking and computing various metrics during training and validation.

    This class extends the PyTorch Metric class and provides methods to update and compute metrics such as
    target loss, concept loss, total loss, accuracy, and Jaccard index for both target and concepts.
    It is being updated for each batch. At the end of each epoch, the compute function is called to compute
    the final metrics and return them as a dictionary.

    Args:
        n_concepts (int): The number of concepts in the model.
        device (torch.device): The device to run the computations on.

    Attributes:
        n_concepts (int): The number of concepts in the model.
        target_loss (torch.Tensor): The accumulated target loss.
        concepts_loss (torch.Tensor): The accumulated concepts loss.
        total_loss (torch.Tensor): The accumulated total loss.
        y_true (list): List of true target labels.
        y_pred_logits (list): List of predicted target logits.
        c_true (list): List of true concept labels.
        c_pred_probs (list): List of predicted concept probabilities.
        batch_features (list): List of batch features.
        cov_norm (torch.Tensor): The accumulated covariance norm.
        n_samples (torch.Tensor): The number of samples processed.
        prec_loss (torch.Tensor): The accumulated precision loss.
    """

    def __init__(self, n_concepts, device, config):
        super().__init__()
        self.n_concepts = n_concepts
        self.config = config
        self.add_state("target_loss", default=torch.tensor(0.0, device=device))
        self.add_state("concepts_loss", default=torch.tensor(0.0, device=device))
        self.add_state("total_loss", default=torch.tensor(0.0, device=device))
        self.add_state("y_true", default=[])
        self.add_state("y_pred_logits", default=[])
        self.add_state("c_true", default=[])
        (
            self.add_state("c_pred_probs", default=[]),
            self.add_state("concepts_input", default=[]),
        ),
        self.add_state("batch_features", default=[])
        self.add_state("cov_norm", default=torch.tensor(0.0, device=device))
        self.add_state(
            "n_samples", default=torch.tensor(0, dtype=torch.int, device=device)
        )
        self.add_state("prec_loss", default=torch.tensor(0.0, device=device))
        self.add_state("l_int_loss", default=torch.tensor(0.0, device=device))
        self.add_state("l_int_extension_loss", default=torch.tensor(0.0, device=device))



    def update(
        self,
        target_loss: torch.Tensor,
        concepts_loss: torch.Tensor,
        total_loss: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_logits: torch.Tensor,
        c_true: torch.Tensor,
        c_pred_probs: torch.Tensor,
        cov_norm: torch.Tensor = None,
        prec_loss: torch.Tensor = None,
        L_int_loss: torch.Tensor = None,
        L_int_extension_loss: torch.Tensor = None,
    ):
        assert c_true.shape == c_pred_probs.shape, f"Shape of true concepts {c_true.shape} and predicted concept probabilities {c_pred_probs.shape} must be the same."

        n_samples = y_true.size(0)
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.n_samples += n_samples
        self.target_loss += target_loss * n_samples
        self.concepts_loss += concepts_loss * n_samples
        self.total_loss += total_loss * n_samples
        self.y_true.append(y_true)
        self.y_pred_logits.append(y_pred_logits.detach())
        self.c_true.append(c_true)
        self.c_pred_probs.append(c_pred_probs.detach())
        if cov_norm:
            self.cov_norm += cov_norm * n_samples
        if prec_loss:
            self.prec_loss += prec_loss * n_samples
        if L_int_loss:
            self.l_int_loss += L_int_loss * n_samples
        if L_int_extension_loss:
            self.l_int_extension_loss += L_int_extension_loss * n_samples
    
    def compute(self, validation=False, config=None):
        y_true = torch.cat(self.y_true, dim=0).cpu()
        c_true = torch.cat(self.c_true, dim=0).cpu()
        c_pred_probs = torch.cat(self.c_pred_probs, dim=0).cpu()
        y_pred_logits = torch.cat(self.y_pred_logits, dim=0).cpu()
        # c_pred_probs = c_pred_probs.numpy()
        c_pred = c_pred_probs > 0.5
        
        
        if self.config.model.multilabel_task:
            # Here y_pred_logits are actual logits
            y_pred_probs = torch.sigmoid(y_pred_logits)   # (n, K+J) independent per task
            y_pred = (y_pred_probs > 0.5).float()
            
        elif y_pred_logits.size(1) == 1:
            y_pred_probs = nn.Sigmoid()(y_pred_logits.squeeze())
            y_pred = y_pred_probs > 0.5
            
        else:
            # dim=1 to normalize across classes for multi-class classification for each sample
            # Even with log_probabilities, we can use softmax to get probabilities for each class, check one note
            y_pred_probs = nn.Softmax(dim=1)(y_pred_logits)
            y_pred = y_pred_logits.argmax(dim=-1)




        if self.config.model.multilabel_task:
            # Hamming: fraction of correct (sample, task) pairs
            target_acc = (y_true == y_pred).sum() / (self.n_samples * y_true.size(1))
            target_acc_key = "y_hamming_accuracy"
            # Samples-averaged Jaccard across tasks
            target_jaccard = jaccard_score(y_true.numpy(), y_pred.numpy(), average="samples")
        else:
            target_acc = (y_true == y_pred).sum() / self.n_samples
            target_acc_key = "y_accuracy"
            target_jaccard = jaccard_score(y_true, y_pred, average="micro")
            
        #target_acc = (y_true == y_pred).sum() / self.n_samples
        concept_acc = (c_true == c_pred).sum() / (self.n_samples * self.n_concepts)
        complete_concept_acc = (
            (c_true == c_pred).sum(1) == self.n_concepts
        ).sum() / self.n_samples
        #target_jaccard = jaccard_score(y_true, y_pred, average="micro")
        concept_jaccard = jaccard_score(c_true, c_pred, average="micro")
        if self.l_int_extension_loss != 0:
            metrics = dict(
                {
                    "target_loss": self.target_loss / self.n_samples,
                    "concepts_loss": self.concepts_loss / self.n_samples,
                    "l_int_extension_loss": self.l_int_extension_loss / self.n_samples,
                    "total_loss": self.total_loss / self.n_samples,
                    target_acc_key: target_acc,
                    "c_accuracy": concept_acc,
                    "complete_c_accuracy": complete_concept_acc,
                    "target_jaccard": target_jaccard,
                    "concept_jaccard": concept_jaccard,
                }
            )
        elif self.l_int_loss != 0:  
            metrics = dict(
                {
                    "target_loss": self.target_loss / self.n_samples,
                    "concepts_loss": self.concepts_loss / self.n_samples,
                    "l_int_loss": self.l_int_loss / self.n_samples,
                    "total_loss": self.total_loss / self.n_samples,
                    target_acc_key: target_acc,
                    "c_accuracy": concept_acc,
                    "complete_c_accuracy": complete_concept_acc,
                    "target_jaccard": target_jaccard,
                    "concept_jaccard": concept_jaccard,
                }
            )
        else:
            metrics = dict(
                {
                    "target_loss": self.target_loss / self.n_samples,
                    "prec_loss": self.prec_loss / self.n_samples,
                    "concepts_loss": self.concepts_loss / self.n_samples,
                    "total_loss": self.total_loss / self.n_samples,
                    target_acc_key: target_acc,
                    "c_accuracy": concept_acc,
                    "complete_c_accuracy": complete_concept_acc,
                    "target_jaccard": target_jaccard,
                    "concept_jaccard": concept_jaccard,
                }
            )

        if self.cov_norm != 0:
            metrics = metrics | {"covariance_norm": self.cov_norm / self.n_samples}

        if validation is True:
            c_pred_probs_list = []
            for j in range(self.n_concepts):
                c_pred_probs_list.append(
                    np.hstack(
                        (
                            np.expand_dims(1 - c_pred_probs[:, j], 1),
                            np.expand_dims(c_pred_probs[:, j], 1),
                        )
                    )
                )

            y_metrics = calc_target_metrics(
                y_true.numpy(), y_pred_probs.numpy(), config
            )
            c_metrics, _ = calc_concept_metrics(
                c_true.numpy(), c_pred_probs_list, config.data
            )
            metrics = (
                metrics
                | {f"y_{k}": v for k, v in y_metrics.items()}
                | {f"c_{k}": v for k, v in c_metrics.items()}
            )  # | c_metrics_per_concept # Update dict

        return metrics




class Custom_Regression_Metrics(Metric):
    """
    Metrics class for regression target prediction with binary concept prediction.

    Tracks:
    - target regression loss
    - concept loss
    - total loss
    - MSE, RMSE, MAE, R2 for target regression
    - concept accuracy, complete concept accuracy, concept Jaccard
    """

    def __init__(self, n_concepts, device):
        super().__init__()

        self.n_concepts = n_concepts

        self.add_state("target_loss", default=torch.tensor(0.0, device=device))
        self.add_state("concepts_loss", default=torch.tensor(0.0, device=device))
        self.add_state("total_loss", default=torch.tensor(0.0, device=device))

        self.add_state("y_true", default=[])
        self.add_state("y_pred", default=[])

        self.add_state("c_true", default=[])
        self.add_state("c_pred_probs", default=[])

        self.add_state("cov_norm", default=torch.tensor(0.0, device=device))
        self.add_state(
            "n_samples",
            default=torch.tensor(0, dtype=torch.int, device=device),
        )

        self.add_state("prec_loss", default=torch.tensor(0.0, device=device))
        self.add_state("l_int_loss", default=torch.tensor(0.0, device=device))
        self.add_state("l_int_extension_loss", default=torch.tensor(0.0, device=device))

    def update(
        self,
        target_loss: torch.Tensor,
        concepts_loss: torch.Tensor,
        total_loss: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        c_true: torch.Tensor,
        c_pred_probs: torch.Tensor,
        cov_norm: torch.Tensor = None,
        prec_loss: torch.Tensor = None,
        L_int_loss: torch.Tensor = None,
        L_int_extension_loss: torch.Tensor = None,
    ):
        assert c_true.shape == c_pred_probs.shape, (
            f"Shape of true concepts {c_true.shape} and predicted concept "
            f"probabilities {c_pred_probs.shape} must be the same."
        )

        n_samples = y_true.size(0)
        self.n_samples += n_samples

        self.target_loss += target_loss.detach() * n_samples
        self.concepts_loss += concepts_loss.detach() * n_samples
        self.total_loss += total_loss.detach() * n_samples

        self.y_true.append(y_true.detach())
        self.y_pred.append(y_pred.detach())

        self.c_true.append(c_true.detach())
        self.c_pred_probs.append(c_pred_probs.detach())

        if cov_norm is not None:
            self.cov_norm += cov_norm.detach() * n_samples

        if prec_loss is not None:
            self.prec_loss += prec_loss.detach() * n_samples

        if L_int_loss is not None:
            self.l_int_loss += L_int_loss.detach() * n_samples

        if L_int_extension_loss is not None:
            self.l_int_extension_loss += L_int_extension_loss.detach() * n_samples

    def compute(self, validation=False, config=None):
        y_true = torch.cat(self.y_true, dim=0).cpu().float()
        y_pred = torch.cat(self.y_pred, dim=0).cpu().float()

        c_true = torch.cat(self.c_true, dim=0).cpu()
        c_pred_probs = torch.cat(self.c_pred_probs, dim=0).cpu()

        # Make sure y_true and y_pred have matching shape.
        # For example, both [N] or both [N, 1].
        y_true = y_true.view(y_pred.shape)

        # -------------------------
        # Regression target metrics
        # -------------------------
        errors = y_pred - y_true

        mse = torch.mean(errors ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(errors))

        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        # Optional: Pearson correlation
        y_true_centered = y_true - y_true.mean()
        y_pred_centered = y_pred - y_pred.mean()

        pearson = torch.sum(y_true_centered * y_pred_centered) / (
            torch.sqrt(torch.sum(y_true_centered ** 2))
            * torch.sqrt(torch.sum(y_pred_centered ** 2))
            + 1e-8
        )

        # -------------------------
        # Binary concept metrics
        # -------------------------
        c_pred = c_pred_probs > 0.5

        concept_acc = (c_true == c_pred).sum() / (
            self.n_samples * self.n_concepts
        )

        complete_concept_acc = (
            (c_true == c_pred).sum(dim=1) == self.n_concepts
        ).sum() / self.n_samples

        concept_jaccard = jaccard_score(
            c_true.numpy(),
            c_pred.numpy(),
            average="micro",
        )

        # -------------------------
        # Base metrics dictionary
        # -------------------------
        metrics = {
            "target_loss": self.target_loss / self.n_samples,
            "concepts_loss": self.concepts_loss / self.n_samples,
            "total_loss": self.total_loss / self.n_samples,

            "y_mse": mse,
            "y_rmse": rmse,
            "y_mae": mae,
            "y_r2": r2,
            "y_pearson": pearson,

            "c_accuracy": concept_acc,
            "complete_c_accuracy": complete_concept_acc,
            "concept_jaccard": concept_jaccard,
        }

        if self.prec_loss != 0:
            metrics["prec_loss"] = self.prec_loss / self.n_samples

        if self.l_int_loss != 0:
            metrics["l_int_loss"] = self.l_int_loss / self.n_samples

        if self.l_int_extension_loss != 0:
            metrics["l_int_extension_loss"] = (
                self.l_int_extension_loss / self.n_samples
            )

        if self.cov_norm != 0:
            metrics["covariance_norm"] = self.cov_norm / self.n_samples
            
        if validation is True:
            c_pred_probs_list = []
            for j in range(self.n_concepts):
                c_pred_probs_list.append(
                    np.hstack(
                        (
                            np.expand_dims(1 - c_pred_probs[:, j], 1),
                            np.expand_dims(c_pred_probs[:, j], 1),
                        )
                    )
                )

            y_metrics = calc_regression_target_metrics(
                y_true.numpy(),
                y_pred.numpy(),
            )

            c_metrics, _ = calc_concept_metrics(
                c_true.numpy(),
                c_pred_probs_list,
                config.data,
            )

            metrics = (
                metrics
                | {f"y_{k}": v for k, v in y_metrics.items()}
                | {f"c_{k}": v for k, v in c_metrics.items()}
            )
                        

        return metrics













def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True
