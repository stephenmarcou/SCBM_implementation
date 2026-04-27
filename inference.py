



import ast
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import wandb

from models.losses import create_loss
from utils.data import get_concept_groups, get_data
from utils.intervention import intervene_cbm, intervene_scbm, intervene_scbm_residual
from utils.training import Custom_Metrics, train_one_epoch_cbm, train_one_epoch_scbm, validate_one_epoch_cbm, validate_one_epoch_scbm
from utils.utils import reset_random_seeds
import torch
from models.models import create_model

def run(config):
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
    
    # Epochs setup, need to check this later
    if config.model.training_mode == "joint":
        t_epochs = config.model.j_epochs
    elif config.model.training_mode in ("sequential", "independent"):
        c_epochs = config.model.c_epochs
        t_epochs = config.model.t_epochs
    if config.model.get("p_epochs") is not None:
        p_epochs = config.model.p_epochs
    
    
    
    
    
    # Set up logging
    if config.run_inference == True:
        log_file_inference = experiment_path / "inference_log.txt"
        with open(log_file_inference, "w") as f:
            f.write(f"Inference log for experiment: {experiment_path}\n")


    if config.run_interventions == True:
        log_file = experiment_path / "intervention_log.txt"
        with open(log_file, "w") as f:
            f.write(f"Intervention log for experiment: {experiment_path}\n")



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
    
    print(config.data.num_concepts)
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
            log_file=log_file_inference
        )
        

    # ---------------------------------
    #       Interventions
    # ---------------------------------

    if config.run_interventions == True:
        if config.model.model == "cbm":
            intervene = intervene_cbm
        elif config.model.model == "scbm":
            intervene = intervene_scbm
        else:
            intervene = intervene_scbm_residual
        # Intervention curves
        print("\nPERFORMING INTERVENTIONS:\n")
        intervene(
            train_loader, test_loader, model, metrics, t_epochs, config, loss_fn, device, log_file=log_file
        )

    wandb.finish(quiet=True)
    return None


def update_pkl_dir_and_num_concepts(config):
    experiment_path = (
        Path(config.experiment_dir) / config.model.model / config.data.dataset / config.inference.ex_name
    )
    with open(os.path.join(experiment_path, "log.txt"), "r") as f:
        lines = f.readlines()
        info_line = lines[0]
        info_line_dict = ast.literal_eval(info_line)
        pkl_file_dir = info_line_dict["data"]["pkl_file_dir"]
        pkl_file_dir = pkl_file_dir.strip("/")
        config.data.pkl_file_dir = pkl_file_dir
        config.data.num_concepts = info_line_dict["data"]["num_concepts"]
        if config.model.model == "scbm_residual":
            config.data.num_residuals = info_line_dict["data"]["num_residuals"]
    
    full_path_pkl_dir = os.path.join(config.data.data_path, "CUB", "incomplete_data", config.data.pkl_file_dir)
    if not os.path.exists(full_path_pkl_dir):
        raise ValueError(f"Pickle directory {full_path_pkl_dir} does not exist.")

        
        
    
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
    check_cluster()
    update_config_paths(config)
    if config.incomplete:
        print("Incomplete run")
        update_pkl_dir_and_num_concepts(config)
    
    
    
    
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    run(config)


if __name__ == "__main__":
    main()
