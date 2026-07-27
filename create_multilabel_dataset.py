import os
import hydra
from omegaconf import DictConfig
from datasets.multilabel_synthetic_dataset import get_multilabel_datasets
import torch

def save_multilabel_data(config, train, val, test):
    """
    Save a multi-label dataset split to disk.

    Directory structure
    -------------------
    <data_path>/<dataset>/
        <save_name>/
            train/   x.pt, concepts.pt, residuals.pt, y.pt,
                     eta_concepts.pt, eta_residuals.pt,
                     concept_signal.pt, residual_signal.pt,
                     s_hid.pt, s_obs.pt,
                     w_obs.pt, w_hid.pt, Sigma.pt,
                     hid_task_idx.pt, obs_task_idx.pt
            val/     (same)
            test/    (same)
            info.txt
    """
    root = os.path.join(config.data.data_path, config.data.dataset)
    os.makedirs(root, exist_ok=True)

    hostname = os.uname()[1]
    prefix = "cluster_" if "biomed" in hostname else "local_"


    save_name = (
        "testing_"
        + prefix
        + f"hid_tasks_{train.num_hid_tasks}"
        + f"_obs_tasks_{train.num_obs_tasks}"
        + f"_alpha_{train.alpha}"
        + f"_beta_{train.beta}"
        + f"_rho_cr{train.rho_cr}"
        + f"_rho_cc{train.rho_cc}"
        + f"_rho_rr{train.rho_rr}"
        + f"_r_sparsity_{train.task_sparsity_hid}"
        + f"_c_sparsity_{train.task_sparsity_obs}"
        + f"_sigmax_{train.sigma_x}"
        + f"_hid_dim_{train.hid_dim}"
        + f"_obs_dim_{train.obs_dim}"
        + f"_w_ratio_{train.min_weight_ratio}"
        + f"_standardize_{train.standardize}"
        + (f"_cpt_{train.concepts_per_hid_task}" if train.concepts_per_hid_task > 1 else "")
        + ("_paired" if train.cov_structure == "paired" else "")
        + (f"_grouped{train.cross_group_size}" if train.cov_structure == "grouped" else "")
        + f"_seed_{config.seed}"
    )

    existing = [d for d in os.listdir(root) if d.startswith(save_name)]
    if existing:
        version = 1
        while os.path.exists(os.path.join(root, f"{save_name}_v{version}")):
            version += 1
        save_name = f"{save_name}_v{version}"

    save_dir = os.path.join(root, save_name)
    os.makedirs(save_dir, exist_ok=True)

    tensor_fields = [
        "x", "concepts", "residuals", "y",
        "eta_concepts", "eta_residuals",
        "concept_signal", "residual_signal",
        "s_hid", "s_obs",
        "w_obs", "w_hid", "Sigma",
        "hid_task_idx", "obs_task_idx",
        "w_task_hid",
    ]

    for split_name, dataset in [("train", train), ("val", val), ("test", test)]:
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for field in tensor_fields:
            torch.save(
                getattr(dataset, field),
                os.path.join(split_dir, f"{field}.pt"),
            )

    info_path = os.path.join(save_dir, "info.txt")
    with open(info_path, "w") as f:
        f.write("[multilabel synthetic dataset]\n")
        f.write(f"num_hid_tasks : {train.num_hid_tasks}\n")
        f.write(f"num_obs_tasks : {train.num_obs_tasks}\n")
        f.write(f"num_tasks/num_classes : {train.num_classes}\n")
        f.write(f"alpha : {train.alpha}\n")
        f.write(f"beta : {train.beta}\n")
        f.write(f"hid_task_idx : {train.hid_task_idx.tolist()}\n")
        f.write(f"obs_task_idx : {train.obs_task_idx.tolist()}\n")
        f.write(f"rho_cr : {train.rho_cr}\n")
        f.write(f"rho_cc : {train.rho_cc}\n")
        f.write(f"rho_rr : {train.rho_rr}\n")
        f.write(f"task_sparsity_hid : {train.task_sparsity_hid}\n")
        f.write(f"task_sparsity_obs : {train.task_sparsity_obs}\n")
        f.write(f"sigma_x : {train.sigma_x}\n")
        f.write(f"hid_dim : {train.hid_dim}\n")
        f.write(f"obs_dim : {train.obs_dim}\n")
        f.write(f"seed : {config.seed}\n")
        f.write(f"train / val / test : {len(train)} / {len(val)} / {len(test)}\n")
        f.write(f"min_weight_ratio : {train.min_weight_ratio}\n")
        f.write(f"standardize : {train.standardize}\n")
        f.write(f"latent_rank : {train.latent_rank}\n")
        f.write(f"concepts_per_hid_task : {train.concepts_per_hid_task}\n")
        f.write(f"cov_structure : {train.cov_structure}\n")
        f.write(f"cross_group_size : {train.cross_group_size}\n")


    return save_dir


        





@hydra.main(version_base=None, config_path="configs", config_name="create_multilabel_data")
def main(config: DictConfig):
    trainset, validset, testset = get_multilabel_datasets(config, config.seed)
    save_multilabel_data(config, trainset, validset, testset)
    




if __name__ == "__main__":
    main()