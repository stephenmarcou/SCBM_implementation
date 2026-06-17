import os
import hydra
from omegaconf import DictConfig
from datasets.synthetic_dataset_res_scbm import get_synthetic_datasets_res_scbm
import torch

def save_synthetic_data(config, train, val, test):

    synthetic_data_dir = os.path.join(config.data.data_path, "synthetic_res_scbm", "test_data_sets", config.data.experiment_type)
    os.makedirs(synthetic_data_dir, exist_ok=True)

    # Save dir name
    hostname = os.uname()[1]
    if "biomed" in hostname:
        save_name = "cluster_"
    else:
        save_name = "local_"
    save_name += f"a_{config.data.alpha}_b_{config.data.beta}_rho_cr{config.data.rho_cr}_rho_cc{config.data.rho_cc}_rho_rr{config.data.rho_rr}_r_sparsity_{config.data.task_sparsity_hid}_c_sparsity_{config.data.task_sparsity_obs}_sigmax_{config.data.sigma_x}_seed_{config.seed}"
    
 
 
    for dir_name in os.listdir(synthetic_data_dir):
        if dir_name.startswith(save_name):
            save_name += "_v"
            version = 1
            while True:
                if not os.path.exists(os.path.join(synthetic_data_dir, save_name + str(version))):
                    save_name += str(version)
                    break
                version += 1
    save_dir = os.path.join(synthetic_data_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    datasets = {"train": train, "val": val, "test": test}
    for split, dataset in datasets.items():
        dataset_dir = os.path.join(save_dir, split)
        os.makedirs(dataset_dir, exist_ok=True)
        torch.save(dataset.x, os.path.join(dataset_dir, "x.pt"))
        torch.save(dataset.concepts, os.path.join(dataset_dir, "concepts.pt"))
        torch.save(dataset.residuals, os.path.join(dataset_dir, "residuals.pt"))
        torch.save(dataset.y, os.path.join(dataset_dir, "y.pt"))
        torch.save(dataset.s, os.path.join(dataset_dir, "s.pt"))
        torch.save(dataset.w_obs, os.path.join(dataset_dir, "w_obs.pt"))
        torch.save(dataset.w_hid, os.path.join(dataset_dir, "w_hid.pt"))
        torch.save(dataset.eta_concepts, os.path.join(dataset_dir, "eta_concepts.pt"))
        torch.save(dataset.eta_residuals, os.path.join(dataset_dir, "eta_residuals.pt"))
        torch.save(dataset.concept_signal, os.path.join(dataset_dir, "concept_signal.pt"))
        torch.save(dataset.residual_signal, os.path.join(dataset_dir, "residual_signal.pt"))
        torch.save(dataset.Sigma, os.path.join(dataset_dir, "Sigma.pt"))
        
    # Info file
    info_file = os.path.join(save_dir, "info.txt")
    # info file for dataset
    with open(info_file, "w") as f:
        f.write(f"rho_cr (correlation between linked concepts and residuals): {train.rho_cr}\n")
        f.write(f"rho_cc (within-block correlation for observed concepts): {train.rho_cc}\n")
        f.write(f"rho_rr (within-block correlation for hidden concepts): {train.rho_rr}\n")
        f.write(f"alpha: {train.alpha}, beta: {train.beta}\n")
        f.write(f"Task sparsity for observed concepts: {train.task_sparsity_obs}, Task sparsity for hidden concepts: {train.task_sparsity_hid}\n")
        f.write(f"Sigma_x (noise level in x): {train.sigma_x}\n")
        f.write(f"num_concepts: {train.concepts.shape[1]}, num_residuals: {train.residuals.shape[1]}\n")

        

        





@hydra.main(version_base=None, config_path="configs", config_name="create_synthetic_data")
def main(config: DictConfig):
    trainset, validset, testset = get_synthetic_datasets_res_scbm(config)
    save_synthetic_data(config, trainset, validset, testset)
    




if __name__ == "__main__":
    main()