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

from utils.metrics import calc_target_metrics, calc_concept_metrics
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
    #save_concept_target_pred=False
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

    # classwise_covariances = {}
    # classwise_counts = {}
    # classwise_mu = {}
    # concept_residuals_probabilities_batches = []
    
    residual_probs_mean = []
    residual_prob_std = []
    residual_mean = []
    residual_std = []
    
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
            # concepts_residuals_mcmc_probs, triang_cov, target_pred_logits = model(
            #     batch_features, epoch, validation=True, c_true=concepts_true
            # )
            
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
            
            # Save the residual channel 
            if config.data.save_residual_channel and test:
                residuals_mcmc_probs = concepts_residuals_mcmc_probs[
                    :, config.data.num_concepts:, :
                ].detach()

                residuals_mcmc = concepts_residuals_mcmc[
                    :, config.data.num_concepts:, :
                ].detach()

                residuals_pred_probs = residuals_mcmc_probs.mean(dim=-1)
                # unbiased=False ensures that we do not get nan if only one monte carlo sample used 
                residuals_prob_std = residuals_mcmc_probs.std(dim=-1, unbiased=False)

                residuals_sample_mean = residuals_mcmc.float().mean(dim=-1)
                residuals_sample_std = residuals_mcmc.float().std(dim=-1, unbiased=False)

                residual_probs_mean.append(residuals_pred_probs.cpu())
                residual_prob_std.append(residuals_prob_std.cpu())

                residual_mean.append(residuals_sample_mean.cpu())
                residual_std.append(residuals_sample_std.cpu())

            
            
            # This can be deleted
            # ------------------------------------------------------------------
            # if config.data.dataset == "synthetic_res_scbm" and config.data.save_predicted_concepts_residuals and test:
            #     #residuals_probs = concepts_residuals_mcmc_probs[:, config.data.num_concepts:, :].detach().cpu()
            #     concept_residuals_probabilities_batches.append(concepts_residuals_mcmc_probs)

            # # Compute covariance matrix of concepts and residuals
            # cov = torch.matmul(triang_cov, torch.transpose(triang_cov, dim0=1, dim1=2))
            # #print(f"Covariance matrix shape: {cov.shape}")

            # batch_class_ids = target_true.detach().cpu().tolist()
            # for sample_idx, class_id in enumerate(batch_class_ids):
            #     if class_id not in classwise_covariances:
            #         classwise_covariances[class_id] = cov[sample_idx].detach().cpu().clone()
            #         classwise_mu[class_id] = c_res_mu[sample_idx].detach().cpu().clone()
            #         classwise_counts[class_id] = 1
            #     else:
            #         classwise_covariances[class_id] += cov[sample_idx].detach().cpu()
            #         classwise_mu[class_id] += c_res_mu[sample_idx].detach().cpu()
            #         classwise_counts[class_id] += 1
            # ------------------------------------------------------------------


            # if test and k % (len(loader) // 10) == 0:
            #     try:
            #         corr = (cov[0] / cov[0].diag().sqrt()).transpose(
            #             dim0=0, dim1=1
            #         ) / cov[0].diag().sqrt()
            #         matrix = corr.cpu().numpy()

            #         compute_and_plot_heatmap(
            #             matrix, concepts_true, concept_names_graph, config
            #         )

            #     except:
            #         pass
            
            
            
            target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
                concepts_mcmc_probs,
                concepts_true,
                target_pred_logits,
                target_true,
                triang_cov,
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

    if config.data.save_residual_channel and test:
        residuals_mean_tensor = torch.cat(residual_mean, dim=0)
        residuals_std_tensor = torch.cat(residual_std, dim=0)
        residual_probs_mean_tensor = torch.cat(residual_probs_mean, dim=0)
        residual_probs_std_tensor = torch.cat(residual_prob_std, dim=0)

        full_path = os.path.dirname(log_file)
        save_path_residual_mean = os.path.join(full_path, "residuals_sample_mean.pt")
        save_path_residual_std = os.path.join(full_path, "residuals_sample_std.pt")
        save_path_residual_probs_mean = os.path.join(full_path, "residuals_pred_probs_mean.pt")
        save_path_residual_probs_std = os.path.join(full_path, "residuals_pred_probs_std.pt")

        torch.save(residuals_mean_tensor, save_path_residual_mean)
        torch.save(residuals_std_tensor, save_path_residual_std)
        torch.save(residual_probs_mean_tensor, save_path_residual_probs_mean)
        torch.save(residual_probs_std_tensor, save_path_residual_probs_std)

        print(f"Saved residual means to {save_path_residual_mean}")
        print(f"Saved residual stds to {save_path_residual_std}")
        print(f"Saved residual predicted probabilities means to {save_path_residual_probs_mean}")
        print(f"Saved residual predicted probabilities stds to {save_path_residual_probs_std}")






    # Saving checks already done earlier as concept_residuals_probabilities_batches would be empty if
    # not saving concepts or not synthetic dataset
    # This can be deleted
    # --------------------------------------------------------------------------
    # if concept_residuals_probabilities_batches and log_file is not None:
    #     log_file_parent = os.path.dirname(log_file)
    #     concept_residuals_save_path = os.path.join(log_file_parent, "pred_concepts_residuals_probs.pt")
    #     residuals_probs = torch.cat(concept_residuals_probabilities_batches, dim=0)
    #     torch.save(residuals_probs, concept_residuals_save_path)
    #     print(f"Saved predicted concept residual probabilities to {concept_residuals_save_path}")

    # if test:
    #     averaged_classwise_covariances = {
    #         class_id: classwise_covariances[class_id] / classwise_counts[class_id]
    #         for class_id in classwise_covariances
    #         if classwise_counts.get(class_id, 0) > 0
    #     }
        
    #     averaged_classwise_mu = {
    #         class_id: classwise_mu[class_id] / classwise_counts[class_id]   
    #         for class_id in classwise_mu
    #         if classwise_counts.get(class_id, 0) > 0
    #     }
        
        
        
    #     if averaged_classwise_covariances and log_file is not None:
    #         full_path = os.path.dirname(log_file)
    #         save_path_covariance = os.path.join(full_path, "classwise_covariances.pt")
    #         torch.save(averaged_classwise_covariances, save_path_covariance)
    #         print(f"Saved classwise covariances to {save_path_covariance}")
            
    #     if averaged_classwise_mu and log_file is not None:
    #         full_path = os.path.dirname(log_file)
    #         save_path_mu = os.path.join(full_path, "classwise_mu.pt")
    #         torch.save(averaged_classwise_mu, save_path_mu)
    #         print(f"Saved classwise means to {save_path_mu}")
    # --------------------------------------------------------------------------

    print(prints)
    print()
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
    log_file=None
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

    # Compute classwise covariance matrices
    classwise_covariances = {}
    classwise_counts = {}
    with torch.no_grad():

        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)
            concepts_mcmc_probs, triang_cov, target_pred_logits = model(
                batch_features, epoch, validation=True, c_true=concepts_true
            )
            # Compute covariance matrix of concepts
            cov = torch.matmul(triang_cov, torch.transpose(triang_cov, dim0=1, dim1=2))

            batch_class_ids = target_true.detach().cpu().tolist()
            for sample_idx, class_id in enumerate(batch_class_ids):
                if class_id not in classwise_covariances:
                    classwise_covariances[class_id] = cov[sample_idx].detach().cpu().clone()
                    classwise_counts[class_id] = 1
                else:
                    classwise_covariances[class_id] += cov[sample_idx].detach().cpu()
                    classwise_counts[class_id] += 1

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

    if test:
        averaged_classwise_covariances = {
            class_id: classwise_covariances[class_id] / classwise_counts[class_id]
            for class_id in classwise_covariances
            if classwise_counts.get(class_id, 0) > 0
        }
        if averaged_classwise_covariances and log_file is not None:
            save_path = Path(log_file).parent / "classwise_covariances.pt"
            torch.save(averaged_classwise_covariances, save_path)
            print(f"Saved classwise covariances to {save_path}")
            with open(log_file, "a") as f:
                f.write(f"Saved classwise covariances to {save_path}\n")
                
    
    
    print(prints)
    print()
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
    log_file=None
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

    def __init__(self, n_concepts, device):
        super().__init__()
        self.n_concepts = n_concepts
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
        if y_pred_logits.size(1) == 1:
            y_pred_probs = nn.Sigmoid()(y_pred_logits.squeeze())
            y_pred = y_pred_probs > 0.5
        else:
            y_pred_probs = nn.Softmax(dim=1)(y_pred_logits)
            y_pred = y_pred_logits.argmax(dim=-1)

        target_acc = (y_true == y_pred).sum() / self.n_samples
        concept_acc = (c_true == c_pred).sum() / (self.n_samples * self.n_concepts)
        complete_concept_acc = (
            (c_true == c_pred).sum(1) == self.n_concepts
        ).sum() / self.n_samples
        target_jaccard = jaccard_score(y_true, y_pred, average="micro")
        concept_jaccard = jaccard_score(c_true, c_pred, average="micro")
        if self.l_int_extension_loss != 0:
            metrics = dict(
                {
                    "target_loss": self.target_loss / self.n_samples,
                    "concepts_loss": self.concepts_loss / self.n_samples,
                    "l_int_extension_loss": self.l_int_extension_loss / self.n_samples,
                    "total_loss": self.total_loss / self.n_samples,
                    "y_accuracy": target_acc,
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
                    "y_accuracy": target_acc,
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
                    "y_accuracy": target_acc,
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
                y_true.numpy(), y_pred_probs.numpy(), config.data
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


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True
