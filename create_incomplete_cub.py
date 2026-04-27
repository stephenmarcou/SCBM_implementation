from train import check_cluster, update_config_paths
import hydra
from omegaconf import DictConfig
from datasets.CUB_dataset import create_random_incomplete_dataset_attr_groups, create_random_incomplete_dataset_indiv_attr


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    check_cluster()
    update_config_paths(config)
    if config.remove_attribute_groups:
        print("Creating new incomplete dataset by removing attribute groups...")
        new_pkl_dir, num_attributes_remaining = create_random_incomplete_dataset_attr_groups(config.data, config.num_attribute_groups_remove)
    else:
        print("Creating new incomplete dataset by removing individual attributes...")
        new_pkl_dir, num_attributes_remaining = create_random_incomplete_dataset_indiv_attr(config.data, config.ratio_attributes_remove)




if __name__ == "__main__":
    main()