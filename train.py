"""
Run this file to train models using a Hydra configuration, e.g.:
    python train.py +model=SCBM +data=CUB
"""

import os
from os.path import join
from pathlib import Path
import time
import uuid

import pickle
import torch
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from datasets.multilabel_synthetic_dataset import check_multilabel_dataset
from models.losses import create_loss
from models.models import create_model



from utils.data import get_data, get_empirical_covariance, get_concept_groups
from utils.intervention import intervene_cbm, intervene_scbm
from utils.training import (
    Custom_Regression_Metrics,
    freeze_module,
    unfreeze_module,
    create_optimizer,
    train_one_epoch_cbm,
    train_one_epoch_scbm,
    validate_one_epoch_cbm,
    validate_one_epoch_scbm,
    train_one_epoch_scbm_residual,
    validate_one_epoch_scbm_residual,
    Custom_Metrics,
)
from utils.utils import reset_random_seeds
from datasets.CUB_dataset import create_random_incomplete_dataset_attr_groups, create_random_incomplete_dataset_indiv_attr
from datasets.synthetic_dataset_res_scbm import load_saved_synthetic_data

from utils.data import make_analysis_loader
from datasets.multiclass_synthetic_dataset import check_synthetic_multiclass_dataset



def maybe_save_best_model(
    config,
    model,
    metrics_dict,
    best_val_metric,
    best_model_path,
    epochs_without_improvement,
    log_file,
):
    patience = config.model.early_stopping_patience

    if config.model.regression_task:
        metric_name = "y_rmse"
        metric_mode = "min"
    elif config.model.multilabel_task:
        metric_name = "y_macro_auroc"
        metric_mode = "max"
    else:
        metric_name = "y_accuracy"
        metric_mode = "max"

    y_val_metric = float(metrics_dict[metric_name])

    if metric_mode == "min":
        improved = y_val_metric < best_val_metric
    else:
        improved = y_val_metric > best_val_metric

    if improved:
        best_val_metric = y_val_metric
        epochs_without_improvement = 0

        if config.save_model:
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best validation {metric_name}: {best_val_metric:.4f}. "
                f"Saved checkpoint to {best_model_path}",
                flush=True,
            )

    else:
        epochs_without_improvement += 1

    should_stop = epochs_without_improvement >= patience

    if should_stop:
        message = (
            f"Early stopping triggered: no improvement for {patience} epochs. "
            f"Best val {metric_name}: {best_val_metric:.4f}"
        )
        print(message, flush=True)
        if config.save_model:
            with open(log_file, "a") as f:
                f.write(message + "\n")

    return best_val_metric, epochs_without_improvement, should_stop





def create_experiment_path(config):
    # Set paths
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    
    
    
    if config.model.get("use_L_int_loss"):
        ex_name = "L_int_loss_weight_" + str(config.model.L_int_loss_weight) + "_" + ex_name
    if config.model.get("use_L_int_extension_loss"):
        ex_name = "L_int_extension_loss_weight_" + str(config.model.L_int_extension_loss_weight) + "_" + ex_name

    if config.model.model == "scbm_residual" and config.model.block_diagonal_cov:
        ex_name = "block_diagonal_cov_True_" + ex_name


    
    if config.data.dataset == "CUB":    
        if config.save_name is not None:
            ex_name = config.save_name + "_" + ex_name
        elif not config.save_name and config.incomplete and config.remove_attribute_groups:
            ex_name = "incomplete_" + str(config.num_attribute_groups_remove) + "_" + ex_name
        elif not config.save_name and config.incomplete and not config.remove_attribute_groups:
            ex_name = "incomplete_rmv_indiv_concepts_" + str(config.ratio_attributes_remove) + "_" + ex_name
        else:
            ex_name = "complete_" + ex_name
    

        
    elif config.data.dataset == "multilabel_synthetic":
        if config.save_name is not None:
            ex_name = config.save_name + "_" + ex_name
        if config.model.model == "scbm_residual":
            ex_name = f"num_res_{config.data.num_residuals}_" + ex_name
        if config.data.get("cov_structure", "lowrank") == "paired":
            ex_name = "paired_cov_" + ex_name
        if config.data.get("concepts_per_hid_task") is not None:
            ex_name = f"c_per_hid_task_{config.data.concepts_per_hid_task}_" + ex_name
        ex_name = f"K_{config.data.num_hid_tasks}_J_{config.data.num_obs_tasks}_alpha_{config.data.alpha}_beta_{config.data.beta}_rho_cr_{config.data.rho_cr}_rho_cc_{config.data.rho_cc}_rho_rr_{config.data.rho_rr}_w_ratio_{config.data.min_weight_ratio}_sigma_x_{config.data.sigma_x}_standardize_{config.data.standardize}_" + ex_name
        
        
    if config.hyperparameter_search:
        config.experiment_dir = join(config.experiment_dir, "hyperparameter_search")
    
    if config.data.dataset != "synthetic_res_scbm":
        experiment_path = (
            Path(config.experiment_dir) / config.model.model / config.data.dataset / ex_name
        )
    else:
        experiment_path = (
            Path(config.experiment_dir) / config.model.model / config.data.dataset / config.data.experiment_type / ex_name
        )
    
    return experiment_path

















def train(config):
    """
    Run the experiments for SCBMs or baselines as defined in the config setting. This method will set up the device, the correct
    experimental paths, initialize Wandb for tracking, generate the dataset, train the model, evaluate the test set performance, and
    finally it will evaluate the intervention performance based on the policies and strategies defined in the config.
    All final results and validations will be stored in Wandb, while the most important ones will be also printed out in the terminal.
    If specified, the model can also be saved for further exploration.

    Parameters
    ----------
    configs: dict
        The config settings for training and validating as defined in configs or in the command line.
    """
    # ---------------------------------
    #       Setup
    # ---------------------------------

    # Reproducibility
    gen = reset_random_seeds(config.seed)

    # Setting device on GPU if available, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    

    # Additional info when using cuda
    if device.type == "cuda":
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")

    
    
    
    experiment_path = create_experiment_path(config)
 
    
    # I changed
    if config.save_model:
        experiment_path.mkdir(parents=True)
        config.experiment_dir = str(experiment_path)
        print("Experiment path: ", experiment_path)
        
        log_file = join(experiment_path, "log.txt")
        with open(log_file, "w") as f:
            f.write(str(config) + "\n\n")  # Log the config at the beginning of the log file
    else:
        log_file = None

    # Wandb
    os.environ["WANDB_CACHE_DIR"] = os.path.join(
        Path(__file__).absolute().parent, "wandb", ".cache", "wandb"
    )  # S.t. on slurm, artifacts are logged to the right place
    print("Cache dir:", os.environ["WANDB_CACHE_DIR"])
    wandb.init(
        project=config.logging.project,
        reinit=True,
        entity=config.logging.entity,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.logging.mode,
        tags=[config.model.tag],
    )
    if config.logging.mode in ["online", "disabled"]:
        wandb.run.name = wandb.run.name.split("-")[-1] + "-" + config.experiment_name
    elif config.logging.mode == "offline":
        wandb.run.name = config.experiment_name
    else:
        raise ValueError("wandb needs to be set to online, offline or disabled.")

    # ---------------------------------
    #       Prepare data and model
    # ---------------------------------
    train_loader, val_loader, test_loader = get_data(
        config,
        config.data,
        gen,
        log_file=log_file
    )

    # Get concept names for plotting
    concept_names_graph = get_concept_groups(config.data)

    # Numbers of training epochs
    if config.model.training_mode == "joint":
        t_epochs = config.model.j_epochs
    elif config.model.training_mode in ("sequential", "independent"):
        c_epochs = config.model.c_epochs
        t_epochs = config.model.t_epochs
    if config.model.get("p_epochs") is not None:
        p_epochs = config.model.p_epochs

    # Initialize model and training objects
    model = create_model(config)
    
    # This need to be deleted as we do not intialize the model with empirical covariance
    # in addition model.sigma_concepts is model.sigma_concepts_residuals in scbm_redidual
    # --------------------------------
    # Initialize covariance with empirical covariance
    if config.model.get("cov_type") == "empirical":
        model.sigma_concepts = get_empirical_covariance(train_loader).to(device)
    elif config.model.get("cov_type") == "global":
        lower_triangle = get_empirical_covariance(train_loader).to(device)
        rows, cols = torch.tril_indices(
            row=config.data.num_concepts, col=config.data.num_concepts, offset=0
        )
        model.sigma_concepts = torch.nn.Parameter(lower_triangle[rows, cols])
        # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
        diag_idx = rows == cols
        with torch.no_grad():
            model.sigma_concepts[diag_idx] = (
                lower_triangle[rows, cols][diag_idx].expm1().clamp_min(1e-6).log()
            )  # softplus inverse of diag
    # --------------------------------
    model.to(device)
    loss_fn = create_loss(config)

    if config.model.regression_task:
        metrics = Custom_Regression_Metrics(config.data.num_concepts, device).to(device)
    else:
        metrics = Custom_Metrics(config.data.num_concepts, device, config).to(device)
    
    if config.model.regression_task:
        best_val_metric = float("inf")      # because lower y_rmse is better
    else:
        best_val_metric = float("-inf")     # because higher y_accuracy is better
    best_model_path = join(experiment_path, "model_best.pth")
    patience = config.model.early_stopping_patience 
    epochs_without_improvement = 0


    # ---------------------------------
    #            Training
    # ---------------------------------
    if config.model.model == "cbm":
        validate_one_epoch = validate_one_epoch_cbm
        train_one_epoch = train_one_epoch_cbm
        intervene = intervene_cbm
    elif config.model.model == "scbm":
        validate_one_epoch = validate_one_epoch_scbm
        train_one_epoch = train_one_epoch_scbm
        intervene = intervene_scbm
    else:
        validate_one_epoch = validate_one_epoch_scbm_residual
        train_one_epoch = train_one_epoch_scbm_residual
        

    print(
        "TRAINING "
        + str(config.model.model)
        + ": "
        + str(config.model.concept_learning + "\n")
    )

    # Pretraining autoregressive concept structure for AR baseline
    if (
        config.model.get("pretrain_concepts")
        and config.model.concept_learning == "autoregressive"
    ):
        print("\nStarting concepts pre-training!\n")
        mode = "c"

        # Freeze the target prediction part
        model.freeze_c()
        model.encoder.apply(freeze_module)  # Freezing the encoder

        c_optimizer = create_optimizer(config.model, model)
        lr_scheduler = optim.lr_scheduler.StepLR(
            c_optimizer,
            step_size=config.model.decrease_every,
            gamma=1 / config.model.lr_divisor,
        )
        epochs_without_improvement = 0
        for epoch in range(p_epochs):
            # Validate the model periodically
            if epoch % config.model.validate_per_epoch == 0:
                #print("\nEVALUATION ON THE VALIDATION SET:\n")
                metrics_dict = validate_one_epoch(
                    val_loader, model, metrics, epoch, config, loss_fn, device
                )
                best_val_metric, epochs_without_improvement, should_stop = maybe_save_best_model(config=config, model=model, metrics_dict=metrics_dict,
                                                                                              best_val_metric=best_val_metric, best_model_path=best_model_path, 
                                                                                              epochs_without_improvement=epochs_without_improvement, log_file=log_file)
                if should_stop:
                    break
                
            train_one_epoch(
                train_loader,
                model,
                c_optimizer,
                mode,
                metrics,
                epoch,
                config,
                loss_fn,
                device,
            )
            lr_scheduler.step()

        model.encoder.apply(unfreeze_module)  # Unfreezing the encoder

    # For sequential & independent training: first stage is training of concept encoder
    if config.model.training_mode in ("sequential", "independent"):
        print("\nStarting concepts training!\n")
        mode = "c"

        # Freeze the target prediction part
        model.freeze_c()

        c_optimizer = create_optimizer(config.model, model)
        lr_scheduler = optim.lr_scheduler.StepLR(
            c_optimizer,
            step_size=config.model.decrease_every,
            gamma=1 / config.model.lr_divisor,
        )
        epochs_without_improvement = 0
        for epoch in range(c_epochs):
            # Validate the model periodically
            if epoch % config.model.validate_per_epoch == 0:
                #print("\nEVALUATION ON THE VALIDATION SET:\n")
                metrics_dict = validate_one_epoch(
                    val_loader, model, metrics, epoch, config, loss_fn, device
                )
                best_val_metric, epochs_without_improvement, should_stop = maybe_save_best_model(config=config, model=model, metrics_dict=metrics_dict, 
                                                                                              best_val_metric=best_val_metric, best_model_path=best_model_path, 
                                                                                              epochs_without_improvement=epochs_without_improvement, log_file=log_file)
                if should_stop:
                    break
            train_one_epoch(
                train_loader,
                model,
                c_optimizer,
                mode,
                metrics,
                epoch,
                config,
                loss_fn,
                device,
                log_file=log_file
            )
            lr_scheduler.step()

        # Prepare parameters for target training by unfreezing the target prediction part and freezing the concept encoder
        model.freeze_t()

    # Sequential vs. joint optimisation
    if config.model.training_mode in ("sequential", "independent"):
        print("\nStarting target training!\n")
        mode = "t"
    else:
        print("\nStarting joint training!\n")
        mode = "j"

    optimizer = create_optimizer(config.model, model)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.model.decrease_every,
        gamma=1 / config.model.lr_divisor,
    )

    # If sequential & independent training: second stage is training of target predictor
    # If joint training: training of both concept encoder and target predictor
    epochs_without_improvement = 0
    for epoch in range(0, t_epochs):
        #if epoch % config.model.validate_per_epoch == 0:
            #print("\nEVALUATION ON THE VALIDATION SET:\n")
        metrics_dict = validate_one_epoch(
            val_loader, model, metrics, epoch, config, loss_fn, device, log_file=log_file
        )
        
        best_val_metric, epochs_without_improvement, should_stop = maybe_save_best_model(
            config=config, model=model, metrics_dict=metrics_dict, best_val_metric=best_val_metric, 
            best_model_path=best_model_path, epochs_without_improvement=epochs_without_improvement, log_file=log_file)
        if should_stop:
            break
        
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            mode,
            metrics,
            epoch,
            config,
            loss_fn,
            device,
            log_file=log_file
        )
        lr_scheduler.step()

    model.apply(freeze_module)
    if config.save_model:
        if config.model.regression_task:
            metric_improved = best_val_metric < float("inf")
        else:
            metric_improved = best_val_metric > float("-inf")

        if metric_improved and Path(best_model_path).exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best validation checkpoint from {best_model_path}\n", flush=True)
            with open(log_file, "a") as f:
                f.write(f"Loaded best validation checkpoint from {best_model_path}\n")
        torch.save(model.state_dict(), join(experiment_path, "model.pth"))
        print("\nTRAINING FINISHED, MODEL SAVED!", flush=True)
    else:
        print("\nTRAINING FINISHED", flush=True)
        
        
    if config.hyperparameter_search:
        print("\nHYPERPARAMETER SEARCH - SKIPPING TEST EVALUATION\n")
        with open(log_file, "a") as f:
            f.write("\nHYPERPARAMETER SEARCH - SKIPPING TEST EVALUATION\n")
            
        print("\nEVALUATION ON THE VALIDATION SET:\n")
        with open(log_file, "a") as f:
            f.write("\nEVALUATION ON THE VALIDATION SET:\n")
        metrics_dict = validate_one_epoch(
            val_loader, model, metrics, epoch, config, loss_fn, device, log_file=log_file
        )
        
        print("Done with this hyperparameter setting. Moving to the next one...\n\n")
        return None
    
    
    
    
    
    
    print("\nFINAL EVALUATION ON THE TEST SET:\n")
    validate_one_epoch(
        test_loader,
        model,
        metrics,
        t_epochs,
        config,
        loss_fn,
        device,
        test=True,
        concept_names_graph=concept_names_graph,
        log_file=log_file,
        save_residual_meta_data_folder="test"
    )
    
    
    # ---------------------------------------------------------
    # Save residual meta data for analysis of concept discovery
    # ---------------------------------------------------------
    if config.model.model == "scbm_residual" and config.data.save_concept_and_residual_channel:
        train_analysis_loader = make_analysis_loader(
            train_loader,
            batch_size=config.model.val_batch_size,
            num_workers=config.workers,
        )
        val_analysis_loader = make_analysis_loader(
            val_loader,
            batch_size=config.model.val_batch_size,
            num_workers=config.workers,
        )

        validate_one_epoch(
            val_analysis_loader,
            model,
            metrics,
            t_epochs,
            config,
            loss_fn,
            device,
            test=False,
            concept_names_graph=concept_names_graph,
            log_file=log_file,
            save_residual_meta_data_folder="val",
            metrics_only_for_saving=True,
        )

        validate_one_epoch(
            train_analysis_loader,
            model,
            metrics,
            t_epochs,
            config,
            loss_fn,
            device,
            test=False,
            concept_names_graph=concept_names_graph,
            log_file=log_file,
            save_residual_meta_data_folder="train",
            metrics_only_for_saving=True,
        )
        

       
    

    if config.train_only:
        wandb.finish(quiet=True)
        return None

    # Intervention curves
    print("\nPERFORMING INTERVENTIONS:\n")
    intervene(
        train_loader, test_loader, model, metrics, t_epochs, config, loss_fn, device
    )

    wandb.finish(quiet=True)
    return None



def check_CUB_data(config):
    full_path_pkl_dir = os.path.join(config.data.data_path, "CUB", "incomplete_data", config.data.pkl_file_dir)


    if not os.path.isdir(full_path_pkl_dir):
        if config.remove_attribute_groups:
            print("Creating new incomplete dataset by removing attribute groups...")
            new_pkl_dir, num_attributes_remaining = create_random_incomplete_dataset_attr_groups(config.data, config.num_attribute_groups_remove)
        else:
            print("Creating new incomplete dataset by removing individual attributes...")
            new_pkl_dir, num_attributes_remaining = create_random_incomplete_dataset_indiv_attr(config.data, config.ratio_attributes_remove)
        config.data.pkl_file_dir = new_pkl_dir
        config.data.num_concepts = num_attributes_remaining
    else:
        print("Incomplete dataset already exists, using existing pkl file directory: ", config.data.pkl_file_dir)
        train_path = os.path.join(full_path_pkl_dir, "train.pkl")
        train_data = pickle.load(open(train_path, "rb"))
        # In case we are using incomplete dataset, we need to update the number of concepts in the config based on the dataset we are loading
        config.data.num_concepts = len(train_data[0]["attribute_label"])
        
        
def check_synthetic_res_scbm_data(config):
    if config.data.data_dir_name is not None:
        train_data, _, _ = load_saved_synthetic_data(config)
        config.data.num_concepts = train_data.concepts.shape[1]
        
        
    if config.data.data_dir_name is None and config.data.num_concepts != config.data.obs_dim:
        config.data.num_concepts = config.data.obs_dim
    
        


    
        
    
def check_cluster():
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print("GPU count:", torch.cuda.device_count())
    else:
        print("Using CPU")


def update_config_paths(config):
    hostname = os.uname()[1]
    # Update paths based on the dataset
    if "biomed" in hostname:
        # Remote Datafolder for our group cluster
        config.data.data_path = "/cluster/home/smarcou/work/Data/"
        config.experiment_dir = "/cluster/home/smarcou/work/experiments_scbm/"
        config.model.model_directory = "/cluster/home/smarcou/work/pretrained_networks/"
    elif "data_path" not in config.data:
        # Local Datafolder if not already specified in yaml
        config.data.data_path = "../datasets/"
    elif config.data.data_path is None:
        config.data.data_path = "../datasets/"
    else:
        pass




@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    #print("Config:", config)
    
    check_cluster()
    update_config_paths(config)
    if config.incomplete and config.data.dataset == "CUB":
        print("Incomplete CUB run")
        check_CUB_data(config)
    if config.data.dataset == "synthetic_res_scbm":
        check_synthetic_res_scbm_data(config)
    
    if config.data.dataset == "multiclass_synthetic":
        check_synthetic_multiclass_dataset(config)
    
    if config.data.dataset == "multilabel_synthetic":
        check_multilabel_dataset(config)
        
    # if config.data.dataset == "synthetic_res_scbm":
    #     check_synthetic_res_scbm_data(config)
 
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    train(config)


if __name__ == "__main__":
    main()
