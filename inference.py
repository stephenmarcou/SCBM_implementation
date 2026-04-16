



import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import wandb

from models.losses import create_loss
from utils.data import get_concept_groups, get_data
from utils.intervention import intervene_cbm, intervene_scbm
from utils.training import Custom_Metrics, train_one_epoch_cbm, train_one_epoch_scbm, validate_one_epoch_cbm, validate_one_epoch_scbm
from utils.utils import reset_random_seeds
import torch
from models.models import create_model

def inference(config):
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
        
    experiment_path = (
        Path(config.experiment_dir) / config.model.model / config.data.dataset / config.inference.ex_name
    )
    
    if not experiment_path.exists():
        raise ValueError(f"Experiment path {experiment_path} does not exist.")
    
    
    # Set up logging
    log_file = experiment_path / "inference_log.txt"
    with open(log_file, "w") as f:
        pass  # Just create an empty log file to start with

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
    )
    
    # Get concept names for plotting
    concept_names_graph = get_concept_groups(config.data)
    
    model = create_model(config)
    saved_model_path = experiment_path / "model.pth"
    state_dict = torch.load(saved_model_path, map_location=device)
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()





    metrics = Custom_Metrics(config.data.num_concepts, device).to(device)
    loss_fn = create_loss(config)
    
    
    
    # ---------------------------------
    #       Inference
    # ---------------------------------
    if config.run_inference == True:
        if config.model.model == "cbm":
            validate_one_epoch = validate_one_epoch_cbm
        else:
            validate_one_epoch = validate_one_epoch_scbm
        t_epochs = None

        print("\nEVALUATION ON THE TEST SET:\n")
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
            log_file=log_file
        )
        

    # ---------------------------------
    #       Interventions
    # ---------------------------------

    if config.run_interventions == True:
        if config.model.model == "cbm":
            intervene = intervene_cbm
        else:
            intervene = intervene_scbm
        t_epochs = None
        # Intervention curves
        print("\nPERFORMING INTERVENTIONS:\n")
        intervene(
            train_loader, test_loader, model, metrics, t_epochs, config, loss_fn, device
        )

    wandb.finish(quiet=True)
    return None


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    inference(config)


if __name__ == "__main__":
    main()
