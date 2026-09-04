



import ast
import os
from os.path import join

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import wandb



from models.losses import create_loss
from utils.data import get_concept_groups, get_data, make_analysis_loader
from utils.intervention import intervene_cbm, intervene_scbm, intervene_scbm_residual, intervene_scbm_residual_optimized
from utils.training import Custom_Metrics, Custom_Regression_Metrics, train_one_epoch_cbm, train_one_epoch_scbm, validate_one_epoch_cbm, validate_one_epoch_scbm, validate_one_epoch_scbm_residual
from utils.utils import reset_random_seeds
import torch
from torch.utils.data import DataLoader
from models.models import create_model
from datasets.CUB_dataset import CUB_CONCEPT_DATASETS, CUB_LABEL_ROOT, CUB_DatasetGenerator, get_CUB_transforms
from datasets.Waterbirds_dataset import NUM_CUB_SPECIES, resolve_waterbirds_target

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
        
    
    experiment_path = get_experiment_path(config)   
    
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
    
    
    
    
    
    # Which split to evaluate / intervene on. 'test' keeps the historical file names, any other
    # split gets its own log so a val run does not overwrite the test results of the same model.
    eval_split = config.inference.get("eval_split", "test")
    if eval_split not in ("test", "val", "train"):
        raise ValueError(
            f"inference.eval_split must be one of ['test', 'val', 'train'], got {eval_split}."
        )
    # TravelingBirds full-render sweep: artifact-only, no evaluation log (see run_tb_render_sweep).
    tb_all_renders = config.inference.get("tb_all_renders", False)

    # Override the image root the eval split is read from, so the same photos can be evaluated on
    # both backgrounds (see build_tb_retargeted_loader). None = the historical behaviour, where
    # train_test_split_CUB picks the folder by split name.
    tb_image_root = config.inference.get("tb_image_root", None)
    if tb_image_root is not None:
        # Validate the image root is test or train (backgrounds)
        if tb_image_root not in TB_IMAGE_ROOTS:
            raise ValueError(
                f"inference.tb_image_root must be one of {list(TB_IMAGE_ROOTS)}, got {tb_image_root}."
            )
        if config.data.dataset != "TravelingBirds":
            raise ValueError(
                "inference.tb_image_root only applies to the TravelingBirds dataset, "
                f"got data.dataset={config.data.dataset}."
            )
        if tb_all_renders:
            raise ValueError(
                "inference.tb_image_root and inference.tb_all_renders are mutually exclusive: "
                "the sweep already covers every split against both image roots."
            )


    # --------------------------------
    # Folder naming to save artifacts and logs
    # -------------------------------
    # 'test' with no root override keeps the historical file names; anything else gets its own log
    # so a val run (or the other background) does not overwrite the test results of the same model.
    if tb_image_root is not None:
        # Same naming as the sweep, so both routes produce comparable folders.
        split_suffix = f"_{eval_split}_bg_{tb_image_root}"
    elif eval_split == "test":
        split_suffix = ""
    else:
        split_suffix = f"_{eval_split}"
    # Folder the c_mu/res_mu artifacts are dumped to; matches the split's own folder by default.
    eval_save_folder = f"{eval_split}_bg_{tb_image_root}" if tb_image_root is not None else eval_split
    # Line recorded in both logs so a curve can be traced back to the background it was measured on.
    split_header = f"{eval_split}"
    if tb_image_root is not None:
        background = "class-correlated" if tb_image_root == "train" else "random"
        split_header += f" (images from TravelingBirds/{tb_image_root}/, {background} background)"

    #-------------------------------
    #   Logs
    #-------------------------------
    if config.run_inference == True:
        if tb_all_renders:
            # The sweep evaluates every split, so it gets its own log rather than an
            # inference_log named after a single split (which is left untouched).
            log_file_inference = experiment_path / "tb_render_sweep_log.txt"
            with open(log_file_inference, "w") as f:
                f.write(f"TravelingBirds render sweep log for experiment: {experiment_path}\n")
                f.write(
                    "Every split evaluated against both image roots: "
                    "train/ = class-correlated background, test/ = random background.\n"
                )
        else:
            log_file_inference = experiment_path / f"inference_log{split_suffix}.txt"
            with open(log_file_inference, "w") as f:
                f.write(f"Inference log for experiment: {experiment_path}\n")
                f.write(f"Evaluation split: {split_header}\n")


    if config.run_interventions == True:
        if tb_image_root is not None:
            # Keep the curve next to the c_mu/res_mu artifacts it was measured on, so a
            # <split>_bg_<root>/ folder is self-contained. The folder normally already exists
            # from an earlier inference run, but interventions can be run on their own.
            log_dir = experiment_path / eval_save_folder
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "intervention_log.txt"
        else:
            log_file = experiment_path / f"intervention_log{split_suffix}.txt"
        with open(log_file, "w") as f:
            f.write(f"Intervention log for experiment: {experiment_path}\n")
            f.write(f"Intervention split: {split_header}\n")

        # Per-sample predictions behind every point of the curve, next to the log they belong
        # to. The logged curve is an average over the split; this is what lets it be re-read
        # within groups afterwards (on Waterbirds, the four bird x background cells). Off by
        # default: a 200-class sweep dumps ~50MB per run and the standard curve needs none of it.
        intervention_preds_dir = None
        if config.inference.get("save_intervention_predictions", False):
            base_dir = log_dir if tb_image_root is not None else experiment_path
            intervention_preds_dir = base_dir / f"intervention_preds{split_suffix}"
            print(f"Saving per-intervention predictions to {intervention_preds_dir}")



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
    log_file=log_file_inference if config.run_inference else log_file
    )

    split_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[eval_split]

    # --------------------------------
    # TravelingBirds background override, so we use the specified background
    # --------------------------------
    if tb_image_root is not None:
        # Same records (same birds, same labels), read from the other rendering.
        eval_loader = build_tb_retargeted_loader(config, split_loader, tb_image_root)
        eval_records = eval_loader.dataset.data
    else:
        # The train/val loaders are shuffled, so wrap the chosen split in an analysis loader to get a
        # deterministic, complete pass (matters because the saved c_mu/res_mu artifacts are order-dependent).
        # For 'test' this is a no-op: the test loader is already unshuffled with drop_last=False.
        eval_loader = make_analysis_loader(
            split_loader,
            batch_size=config.model.val_batch_size,
            num_workers=config.workers,
        )
        eval_records = None


    
    # Get concept names for plotting
    concept_names_graph = get_concept_groups(config.data)
   
    
    print(config.data.num_concepts)
    model = create_model(config)
    saved_model_path = experiment_path / "model.pth"
    state_dict = torch.load(saved_model_path, map_location=device)
    print(f"Loaded model state dict from {saved_model_path}")
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if config.model.regression_task:
        metrics = Custom_Regression_Metrics(config.data.num_concepts, device).to(device)
    else:
        metrics = Custom_Metrics(config.data.num_concepts, device, config).to(device)
    loss_fn = create_loss(config)
    
    
    
    # ---------------------------------
    #       Inference
    # ---------------------------------
    if config.run_inference == True:
        if config.model.model in ("cbm", "cbm_residual"):
            validate_one_epoch = validate_one_epoch_cbm
            test_save_kwargs = {"save_concept_meta_data_folder": eval_save_folder}
        elif config.model.model == "scbm":
            validate_one_epoch = validate_one_epoch_scbm
            test_save_kwargs = {"save_concept_meta_data_folder": eval_save_folder}
        elif config.model.model == "scbm_residual":
            validate_one_epoch = validate_one_epoch_scbm_residual
            test_save_kwargs = {"save_residual_meta_data_folder": eval_save_folder}

        #save_concept_target_pred = config.inference.save_concept_target_pred

        # Artifact-only mode: the sweep already covers every split against both image roots,
        # so the single-split evaluation and the default dumps below are skipped.
        if tb_all_renders:
            run_tb_render_sweep(
                config,
                {"train": train_loader, "val": val_loader, "test": test_loader},
                validate_one_epoch,
                model,
                metrics,
                t_epochs,
                loss_fn,
                device,
                concept_names_graph,
                log_file_inference,
            )

    # Still saves concept/residual information given config.data.save_concept_and_residual_channel=true
    if config.run_inference == True and not tb_all_renders:
        print(f"\nEVALUATION ON THE {split_header.upper()} SET:\n")
        validate_one_epoch(
            eval_loader,
            model,
            metrics,
            t_epochs,
            config,
            loss_fn,
            device,
            test=True,
            concept_names_graph=concept_names_graph,
            log_file=log_file_inference,
            **test_save_kwargs,
        )

        # eval_records = eval_loader.dataset.data if tb_image_root is not None else None
        if eval_records is not None and config.data.save_concept_and_residual_channel:
            # Lets the analysis verify row alignment between the two backgrounds: the two
            # img_paths.txt differ only in the train/ vs test/ path component.
            save_render_image_paths(
                eval_records, os.path.join(experiment_path, eval_save_folder)
            )

        # ---------------------------------------------------------
        # Save concept / residual meta data for analysis of concept discovery
        # ---------------------------------------------------------
        # Did inference and saved the c_mu/res_mu artifacts for eval_loader
        # Here we run a full pass over the other splits to save their c_mu/res_mu artifacts too
        # eval_loader is popped from the analysis_loaders dict so it is not re-run
        # We do not do this if tb_image_root is set, as then we only want to do it for a very specific 
        # split background combination
        if config.data.save_concept_and_residual_channel and tb_image_root is None:
            save_kwarg = (
                "save_residual_meta_data_folder"
                if config.model.model in ("scbm_residual", "cbm_residual")
                else "save_concept_meta_data_folder"
            )

            # eval_split was already dumped above by the evaluation pass, through an analysis
            # loader with the same deterministic settings, so re-running it here would only
            # duplicate those artifacts at the cost of a full pass.
            analysis_loaders = {
                "val": val_loader,
                "train": train_loader,
                "test": test_loader,
            }
            analysis_loaders.pop(eval_split, None)

            # Same train images, but with the deterministic (test-time) transform instead
            # of the training-time augmentation, so train-vs-test logit comparisons aren't
            # confounded by ColorJitter/RandomResizedCrop/RandomHorizontalFlip.
            if config.data.dataset in CUB_CONCEPT_DATASETS:
                _, test_transform = get_CUB_transforms()
                analysis_loaders["train_with_test_transform"] = DataLoader(
                    CUB_DatasetGenerator(
                        train_loader.dataset.data, transform=test_transform, cache=False
                    ),
                    batch_size=config.model.val_batch_size,
                )

            for folder, base_loader in analysis_loaders.items():
                validate_one_epoch(
                    make_analysis_loader(
                        base_loader,
                        batch_size=config.model.val_batch_size,
                        num_workers=config.workers,
                    ),
                    model,
                    metrics,
                    t_epochs,
                    config,
                    loss_fn,
                    device,
                    test=False,
                    concept_names_graph=concept_names_graph,
                    log_file=log_file_inference,
                    metrics_only_for_saving=True,
                    **{save_kwarg: folder},
                )

    # ---------------------------------
    #       Interventions
    # ---------------------------------

    if config.run_interventions == True:
        if config.model.model in ("cbm", "cbm_residual"):
            intervene = intervene_cbm
        elif config.model.model == "scbm":
            intervene = intervene_scbm
        else:
            #intervene = intervene_scbm_residual
            # CHANGE AFTERWARDS
            intervene = intervene_scbm_residual_optimized
        # Intervention curves
        print(f"\nPERFORMING INTERVENTIONS ON THE {split_header.upper()} SET:\n")
        # train_loader is used for the intervention strategy, for example for empirical percentile
        intervene(
            train_loader, eval_loader, model, metrics, t_epochs, config, loss_fn, device, log_file=log_file,
            save_predictions_dir=intervention_preds_dir,
        )

    wandb.finish(quiet=True)
    return None


# TravelingBirds ships two renderings of every CUB photo: <data_path>/TravelingBirds/train/
# (class-correlated background) and <data_path>/TravelingBirds/test/ (random background),
# 11788 files each. train_test_split_CUB picks the folder by split *name*, so a normal run
# only ever sees one rendering per photo - half of the 23576 images on disk.
TB_IMAGE_ROOTS = ("train", "test")


def retarget_tb_records(records, data_path, image_root):
    """Copy split records with img_path repointed at TravelingBirds/<image_root>/.

    The labels are untouched: both renderings show the same bird, only the background differs.
    """
    retargeted = []
    for record in records:
        class_dir, file_name = record["img_path"].split("/")[-2:]
        # Keep everything else, just change the image path to the TravelingBirds renderings
        # The original attributes etc are preserved
        retargeted.append(
            {
                **record,
                "img_path": os.path.join(
                    data_path, "TravelingBirds", image_root, class_dir, file_name
                ),
            }
        )
    return retargeted


def build_tb_retargeted_loader(config, loader, image_root):
    """Deterministic loader over a split's records, read from TravelingBirds/<image_root>/.

    shuffle=False / drop_last=False plus the test-time transform, so the passes over the two
    image roots are sample-aligned: row i is the same bird with the same labels, only the
    background differs. That pairing is the point of the override - comparing a split against
    itself on both backgrounds isolates the background, whereas val-vs-test also swaps the
    photographers.
    """
    records = retarget_tb_records(loader.dataset.data, config.data.data_path, image_root)
    _, test_transform = get_CUB_transforms()
    return DataLoader(
        CUB_DatasetGenerator(records, transform=test_transform, cache=False),
        batch_size=config.model.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
    )


def save_render_image_paths(records, folder_path):
    """Write the image paths of one sweep pass, in loader order, next to its artifacts.

    The render loaders are shuffle=False / drop_last=False, so row i of c_res_mu.pt (and of
    every other saved tensor) is the image on line i of img_paths.txt. Row i of
    <split>_bg_train and of <split>_bg_test is therefore the same photo on the two
    backgrounds, which is exactly what the paths file lets you verify.
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    paths_file = os.path.join(folder_path, "img_paths.txt")
    with open(paths_file, "w") as f:
        f.write("\n".join(record["img_path"] for record in records) + "\n")

    # Cross-check against an artifact every model type saves: a length mismatch means the
    # pass did not produce one row per record, and the pairing would be silently misaligned.
    y_true_path = os.path.join(folder_path, "y_true.pt")
    if os.path.exists(y_true_path):
        num_rows = len(torch.load(y_true_path, map_location="cpu"))
        if num_rows != len(records):
            raise RuntimeError(
                f"{folder_path}: {num_rows} saved rows but {len(records)} image paths."
            )
    print(f"Saved {len(records)} image paths to {paths_file}")


def run_tb_render_sweep(
    config,
    loaders,
    validate_one_epoch,
    model,
    metrics,
    t_epochs,
    loss_fn,
    device,
    concept_names_graph,
    log_file,
):
    """Forward pass over every split x every TravelingBirds image root (all 23576 renders).

    Each pass uses the deterministic test-time transform, so the six combinations are directly
    comparable, and dumps its concept/residual artifacts to <split>_bg_<root>/ (only when
    data.save_concept_and_residual_channel is set). log_file is the sweep's own log
    (tb_render_sweep_log.txt); it also determines the run directory the artifacts land in,
    since validate_one_epoch derives that from the log file's parent.
    """
    if config.data.dataset != "TravelingBirds":
        raise ValueError(
            "inference.tb_all_renders only applies to the TravelingBirds dataset, "
            f"got data.dataset={config.data.dataset}."
        )
    if not config.data.save_concept_and_residual_channel:
        # The artifacts are the only output of this mode, so 23576 forward passes would
        # otherwise be thrown away.
        raise ValueError(
            "inference.tb_all_renders needs data.save_concept_and_residual_channel=True, "
            "otherwise the sweep saves nothing."
        )

    # Residual models save under the residual-flavoured kwarg; validate_one_epoch_cbm accepts both.
    save_kwarg = (
        "save_residual_meta_data_folder"
        if config.model.model in ("scbm_residual", "cbm_residual")
        else "save_concept_meta_data_folder"
    )
    for split, loader in loaders.items():
        for image_root in TB_IMAGE_ROOTS:
            # Same records, read from this rendering (see build_tb_retargeted_loader).
            # The loader is shuffle=False / drop_last=False, also get test_transform
            render_loader = build_tb_retargeted_loader(config, loader, image_root)
            records = render_loader.dataset.data

            folder = f"{split}_bg_{image_root}"
            header = (
                f"\nsplit: {split}, image root: {image_root}/ "
                f"({len(records)} images) -> {folder}/"
            )
            print(header)

            # metrics_only_for_saving keeps validate_one_epoch from writing its own
            # "Validation:" line and from overwriting the same wandb keys six times over;
            # the metrics are recorded below under a name identifying the combination.
            metrics_dict = validate_one_epoch(
                render_loader,
                model,
                metrics,
                t_epochs,
                config,
                loss_fn,
                device,
                test=False,
                concept_names_graph=concept_names_graph,
                log_file=log_file,
                metrics_only_for_saving=True,
                **{save_kwarg: folder},
            )

            save_render_image_paths(records, os.path.join(os.path.dirname(log_file), folder))

            summary = " ".join(f"{k}: {v:.3f}" for k, v in metrics_dict.items())
            print(summary + "\n")
            with open(log_file, "a") as f:
                f.write(header + "\n" + summary + "\n")
            wandb.log(
                {f"tb_render_sweep/{folder}/{k}": v for k, v in metrics_dict.items()}
            )



def get_data_dir_name_synthetic_data(experiment_path):

    """
    Get the data directory name from the log.txt file in the experiment path. 
    This is so we can load the correct synthetic dataset for inference and interventions
    """
    data_dir_in_line = False
    loaded_data_dir_in_line = False
    
    with open(os.path.join(experiment_path, "log.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("data_dir:"):
                data_dir_line = line
                data_dir_in_line = True
                break
            # If model was trained on an existing synthetic dataset
            elif line.startswith("Loading existing synthetic dataset from"):
                data_dir_line = line
                loaded_data_dir_in_line = True
                break
        if data_dir_in_line:
            data_dir = data_dir_line.split("data_dir:")[1].strip()
        elif loaded_data_dir_in_line:
            data_dir = data_dir_line.split("Loading existing synthetic dataset from")[1].strip()
        else: 
            raise ValueError("data_dir not found in log.txt")
        
        
        data_dir = "/".join(data_dir.split("/")[-1:])       
    return data_dir



def get_experiment_path(config):
    if config.data.dataset != "synthetic_res_scbm":
        experiment_path = (
            Path(config.experiment_dir) / config.model.model / config.data.dataset / config.inference.ex_name
        )
    
    else:
        if config.data.experiment_type is None:
            experiment_path = (
                Path(config.experiment_dir) / config.model.model / config.data.dataset / config.inference.ex_name
            )
        else:
            experiment_path = (
                Path(config.experiment_dir) / config.model.model / config.data.dataset / config.data.experiment_type / config.inference.ex_name
            )
    return experiment_path




def read_run_config(config):
    """Config dict the run was trained with, stored on the first line of its log.txt."""
    experiment_path = get_experiment_path(config)
    with open(os.path.join(experiment_path, "log.txt"), "r") as f:
        return ast.literal_eval(f.readlines()[0])


def restore_waterbirds_target(config, info_line_dict):
    """Restore which label the Waterbirds run was trained against.

    data.binary_target changes the head width (200 -> 1 logit), so a mismatch here means
    load_state_dict fails on a shape mismatch. Recovering it from the run's own log.txt
    keeps the flag off the inference command line, the same way pkl_file_dir and
    num_residuals already are.
    """
    if config.data.dataset != "Waterbirds":
        return
    trained_binary = bool(info_line_dict["data"].get("binary_target", False))
    if bool(config.data.get("binary_target", False)) != trained_binary:
        print(
            f"Waterbirds: restoring data.binary_target={trained_binary} from log.txt "
            f"(config said {bool(config.data.get('binary_target', False))})."
        )
    config.data.binary_target = trained_binary
    # num_classes has to follow the flag before create_model reads it. Reset it first so the
    # resolver's binary_target=False branch does not trip on a stale 2 from the yaml.
    config.data.num_classes = NUM_CUB_SPECIES
    resolve_waterbirds_target(config.data)


def update_num_concepts_and_residuals(config, info_line_dict):
    """Restore the channel dimensions the checkpoint was trained with.

    Without this the model is built from the yaml defaults, which only match the
    checkpoint for a complete run trained with the default num_residuals.
    """
    config.data.num_concepts = info_line_dict["data"]["num_concepts"]
    if config.model.model in ("scbm_residual", "cbm_residual"):
        if "num_residuals" in info_line_dict["data"]:
            config.data.num_residuals = info_line_dict["data"]["num_residuals"]
        else:
            print(
                f"Warning: no num_residuals in log.txt, keeping config value "
                f"{config.data.num_residuals}"
            )
    # The encoder architecture is part of the checkpoint's shape (e.g. CIFAR-10 runs use
    # 'simple_CNN' against a 'resnet18' yaml default) and isn't otherwise recovered, so a
    # mismatched default here fails create_model's load_state_dict with missing/unexpected
    # keys and shape mismatches rather than a clear error.
    if "encoder_arch" in info_line_dict["model"]:
        config.model.encoder_arch = info_line_dict["model"]["encoder_arch"]
    else:
        print(
            f"Warning: no encoder_arch in log.txt, keeping config value "
            f"{config.model.encoder_arch}"
        )


# Need to change this function
def update_pkl_dir_and_num_concepts(config):

    experiment_path = get_experiment_path(config)




    with open(os.path.join(experiment_path, "log.txt"), "r") as f:
        lines = f.readlines()
        info_line = lines[0]
        info_line_dict = ast.literal_eval(info_line)

        if config.data.dataset != "synthetic_res_scbm":
            pkl_file_dir = info_line_dict["data"]["pkl_file_dir"]
            pkl_file_dir = pkl_file_dir.strip("/")
            config.data.pkl_file_dir = pkl_file_dir
            
            
        if config.data.dataset == "synthetic_res_scbm":
            data_dir_name = get_data_dir_name_synthetic_data(experiment_path)
            config.data.data_dir_name = data_dir_name
        
        # Update num concepts and num residuals to create right model
        print(f"info_line_dict: {info_line_dict}")
        update_num_concepts_and_residuals(config, info_line_dict)

        # For synthetic dataset
        if config.data.dataset == "synthetic_res_scbm":
            config.data.save_data = True
    
    # Ensure that the pkl directory exists
    if config.data.dataset in CUB_CONCEPT_DATASETS:
        full_path_pkl_dir = os.path.join(config.data.data_path, CUB_LABEL_ROOT, "incomplete_data", config.data.pkl_file_dir)
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
    # Need to change this because it is not incomplete for non-CUB datasets
    if config.incomplete:
        print("Incomplete run")
        update_pkl_dir_and_num_concepts(config)
    else:
        # Complete run: pkl_file_dir is the config default, but the channel
        # dimensions still have to come from the run itself
        update_num_concepts_and_residuals(config, read_run_config(config))

    # Independent of `incomplete`: which label (200 or 2 classes) the run targeted also sets the head width.
    restore_waterbirds_target(config, read_run_config(config))

    
    
    
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    run(config)


if __name__ == "__main__":
    main()
