"""
CUB dataset loader with concept labels. 

This module provides a custom DataLoader for the CUB dataset, including concept labels for training, validation, and testing.
The dataset is preprocessed with transformations.

Classes:
    CUB_DatasetGenerator: Custom DataLoader for the CUB dataset.

Functions:
    train_test_split_CUB: Perform train-validation-test split for the CUB dataset according the predefined photographer-specific partitions.
    get_CUB_dataloaders: Get DataLoaders for the CUB dataset.
"""

"""
CIFAR-100 dataset loader Relies on create_dataset_cifar.py to have generated concept labels.

This module provides a custom DataLoader for the CIFAR-100 dataset, including concept labels for training, validation, and testing.
The dataset is preprocessed with transformations.

Classes:
    CIFAR100_CBM_dataloader: Custom DataLoader for CIFAR-100 with concept labels.

Functions:
    get_CIFAR100_CBM_dataloader: Returns DataLoaders for training, validation, and testing splits.
"""

import ctypes
import os
import pickle
import random
from PIL import Image
import numpy as np
import multiprocessing as mp

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CUB_DatasetGenerator(Dataset): 
    """CUB Dataset object with caching"""
 
    def __init__(self, data_pkl, transform=None, cache=False): 
        """ 
        Arguments: 
        data_pkl: list of data dictionaries containing img_path, class_label, attribute_label
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing 
        cache: Whether to cache the dataset in shared system RAM.
        """ 
        self.data = data_pkl 
        self.transform = transform 
        self.cache = cache
        self.cache_hits = 0
        self.cache_misses = 0

        num_samples = len(data_pkl) 
        
        # Maximum possible dimensions from CUB 
        max_height = 500 
        max_width = 500 
        data_dims = (3, max_height, max_width) 
        dimension = int(np.prod(data_dims)) 
        
        if self.cache:
            # Create shared array for image data (padded to max size)
            shared_array_base = mp.Array(
                ctypes.c_uint8, num_samples * dimension 
            ) 
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj()) 
            shared_array = shared_array.reshape(num_samples, *data_dims) 
            self.image_cache = torch.from_numpy(shared_array)
            
            # Create shared array for image dimensions and validity
            # Format: [height, width] per image, initialized to [-1, -1] (invalid)
            dims_array_base = mp.Array(
                ctypes.c_int, num_samples * 2  # 2 values per image: height, width
            )
            dims_array = np.ctypeslib.as_array(dims_array_base.get_obj())
            dims_array = dims_array.reshape(num_samples, 2)
            self.dims_cache = torch.from_numpy(dims_array)
            self.dims_cache.fill_(-1)  # Initialize to -1 to indicate not cached
            
            # CUB has 112 binary attributes - we need 112 bits = 14 bytes per sample
            # We'll use 15 bytes (120 bits) for easier alignment and future expansion
            attr_size = len(data_pkl[0]["attribute_label"])
            self.num_attributes = attr_size
            bytes_per_sample = (attr_size + 7) // 8  # Round up to nearest byte

            attr_array_base = mp.Array(
                ctypes.c_uint8, num_samples * bytes_per_sample
            )
            attr_array = np.ctypeslib.as_array(attr_array_base.get_obj())
            attr_array = attr_array.reshape(num_samples, bytes_per_sample)
            self.attr_cache = torch.from_numpy(attr_array)
            self.attr_cache.fill_(0)  # Initialize to 0
            
            # Create shared array for class labels
            label_array_base = mp.Array(
                ctypes.c_int, num_samples
            )
            label_array = np.ctypeslib.as_array(label_array_base.get_obj())
            self.label_cache = torch.from_numpy(label_array)
            self.label_cache.fill_(-1)  # Initialize to -1 to indicate not cached

    def _pack_attributes(self, attributes):
        """
        Pack binary attributes into bytes.
        
        Args:
            attributes: numpy array of binary values (0 or 1)
        
        Returns:
            Packed byte array
        """
        # Ensure attributes are binary
        attributes = np.array(attributes, dtype=np.uint8)
        attributes = np.clip(attributes, 0, 1)  # Ensure binary
        
        # Calculate number of bytes needed
        n_bytes = (len(attributes) + 7) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        
        # Pack bits into bytes
        for i, attr in enumerate(attributes):
            if attr:
                byte_idx = i // 8
                bit_idx = i % 8
                packed[byte_idx] |= (1 << bit_idx)
        
        return packed
    
    def _unpack_attributes(self, packed_bytes):
        """
        Unpack bytes into binary attributes.
        
        Args:
            packed_bytes: byte array
        
        Returns:
            numpy array of binary values (0 or 1)
        """
        attributes = np.zeros(self.num_attributes, dtype=np.float64)
        
        for i in range(self.num_attributes):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(packed_bytes):
                bit_value = (packed_bytes[byte_idx] >> bit_idx) & 1
                # Store as float64 to match original
                attributes[i] = float(bit_value)
        
        # Return as float32 for consistency with model expectations
        return attributes.astype(np.float32)

    def _is_cached(self, index):
        """Check if an image is already cached by looking at dimensions"""
        if self.cache:
            return self.dims_cache[index][0] != -1 and self.dims_cache[index][1] != -1
        return False

    def _cache_image(self, index, image_pil, image_attr, image_label):
        """Cache an image and its metadata in the shared arrays"""
        # Convert PIL image to numpy array
        img_array = np.array(image_pil)
        h, w = img_array.shape[:2]
        
        # Store dimensions
        self.dims_cache[index] = torch.tensor([h, w])
        
        # Pad image to maximum size if necessary
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
        
        # Convert to CHW format and pad
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
        
        # Pad to maximum size
        padded_img = torch.zeros((3, 500, 500), dtype=torch.uint8)
        padded_img[:, :h, :w] = img_tensor
        
        # Store in cache
        self.image_cache[index] = padded_img
        
        # Pack and store attributes
        packed_attrs = self._pack_attributes(image_attr)
        self.attr_cache[index] = torch.from_numpy(packed_attrs)
        
        self.label_cache[index] = image_label

    def _get_cached_image(self, index):
        """Retrieve a cached image with its original dimensions"""
        h, w = self.dims_cache[index]
        h, w = int(h), int(w)
        
        # Extract the image without padding
        img_tensor = self.image_cache[index][:, :h, :w]  # CHW format
        
        # Convert back to PIL Image (HWC format)
        img_array = img_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        image_pil = Image.fromarray(img_array)
        
        # Unpack attributes
        packed_attrs = self.attr_cache[index].numpy()
        image_attr = self._unpack_attributes(packed_attrs)
        
        image_label = int(self.label_cache[index])
        
        return image_pil, image_attr, image_label

    def __getitem__(self, index): 
        # Check if already cached
        if self._is_cached(index):
            self.cache_hits += 1
            if self.cache_hits % 2000 == 0:
                print(f"[CACHE HIT] {self.cache_hits} hits")
            image_data, image_attr, image_label = self._get_cached_image(index)
        else: 
            self.cache_misses += 1
            if self.cache_misses % 2000 == 0:
                print(f"[CACHE MISS] {self.cache_misses} misses")
            img_data = self.data[index] 
            img_path = img_data["img_path"] 
            image_data = Image.open(img_path).convert("RGB") 
            image_label = img_data["class_label"] 
            image_attr = np.array(img_data["attribute_label"])
            
            if self.cache:
                self._cache_image(index, image_data, image_attr, image_label)
        
        # I changed
        image_attr = image_attr.astype(np.float32)  # Ensure attributes are float32 for model compatibility
        
        if self.transform is not None: 
            image_data = self.transform(image_data) 
            


        
        
        # Return a tuple of images, labels, and protected attributes 
        return { 
            "img_code": index, 
            "labels": image_label, 
            "features": image_data, 
            "concepts": image_attr,  # This is now float32 array as expected
        }
 
    def __len__(self): 
        return len(self.data)


def train_test_split_CUB(config, incomplete):
    """Performs train-validation-test split for the CUB dataset"""

    # Using pre-determined split as to have different photographers in train & test
    data_train = []
    data_val = []
    data_test = []
    
    if not incomplete:
        full_train_pkl_path = os.path.join(config.data_path, "CUB", "class_attr_data_10", "train.pkl")
        full_val_pkl_path = os.path.join(config.data_path, "CUB", "class_attr_data_10", "val.pkl")
        full_test_pkl_path = os.path.join(config.data_path, "CUB", "class_attr_data_10", "test.pkl")
    else:
        full_train_pkl_path = os.path.join(config.data_path, "CUB", "incomplete_data", config.pkl_file_dir, "train.pkl")
        full_val_pkl_path = os.path.join(config.data_path, "CUB", "incomplete_data", config.pkl_file_dir, "val.pkl")
        full_test_pkl_path = os.path.join(config.data_path, "CUB", "incomplete_data", config.pkl_file_dir, "test.pkl")
        print(f"Using incomplete dataset with pkl files from {config.pkl_file_dir}")

    data_train.extend(
        pickle.load(
            open(
                os.path.join(
                    full_train_pkl_path
                ),
                "rb",
            )
        )
    )
    data_val.extend(
        pickle.load(
            open(
                os.path.join(full_val_pkl_path),
                "rb",
            )
        )
    )
    data_test.extend(
        pickle.load(
            open(
                os.path.join(full_test_pkl_path),
                "rb",
            )
        )
    )
    for dataset in [data_train, data_val, data_test]:
        for i in range(len(dataset)):
            parts = dataset[i]["img_path"].split("/")
            index = parts.index("images")
            end_path = "/".join(parts[index:])

            dataset[i]["img_path"] = os.path.join(
                config.data_path, "CUB/CUB_200_2011/CUB_200_2011/", end_path
            )

    return data_train, data_val, data_test


def get_CUB_dataloaders(config, incomplete):
    """Returns a dictionary of data loaders for the CUB dataset, for the training, validation, and test sets."""
    train_imgs, val_imgs, test_imgs = train_test_split_CUB(
        config, incomplete
    )

    # Following the transformations from CBM paper
    resol = 299
    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(resol),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    image_datasets = {
        "train": CUB_DatasetGenerator(train_imgs, transform=train_transform, cache=True),
        "val": CUB_DatasetGenerator(val_imgs, transform=test_transform, cache=True),
        "test": CUB_DatasetGenerator(test_imgs, transform=test_transform, cache=False),
    }

    return (
        image_datasets["train"],
        image_datasets["val"],
        image_datasets["test"],
    )
    
    
# Fom CEM repo
ATTRIBUTES_IDX_USED = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

# Fom CEM repo
ATTRIBUTE_PARTS = [
 'has_bill_shape',
 'has_wing_color',
 'has_upperparts_color',
 'has_underparts_color',
 'has_breast_pattern',
 'has_back_color',
 'has_tail_shape',
 'has_upper_tail_color',
 'has_head_pattern',
 'has_breast_color',
 'has_throat_color',
 'has_eye_color',
 'has_bill_length',
 'has_forehead_color',
 'has_under_tail_color',
 'has_nape_color',
 'has_belly_color',
 'has_wing_shape',
 'has_size',
 'has_shape',
 'has_back_pattern',
 'has_tail_pattern',
 'has_belly_pattern',
 'has_primary_color',
 'has_leg_color',
 'has_bill_color',
 'has_crown_color',
 'has_wing_pattern',
]



def get_attribute_parts_to_indices(config_data):
    """
    Maps attribute idx to attribute parts (e.g. bill, wing, etc.)
    """
    
    
    # Attribute idx in the original 312 attribute space to attribute idx in the new 112 attribute space
    old_idx_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(ATTRIBUTES_IDX_USED)}
    #print(old_idx_to_new_idx)
    

    path_attrubute_file = os.path.join(config_data.data_path, "CUB/CUB_200_2011/attributes.txt")



    with open(path_attrubute_file, "r") as f:
        lines = f.readlines()
        semantic_groups = {}
        for line in lines:
            idx, attr_name = line.strip().split(" ")
            idx = int(idx) - 1 # Convert to 0-based index
            if idx not in ATTRIBUTES_IDX_USED:
                continue
                
            new_idx = old_idx_to_new_idx[idx]
            semantic_group = attr_name.split("::")[0]
            if semantic_group not in semantic_groups:
                semantic_groups[semantic_group] = []
            semantic_groups[semantic_group].append(new_idx)

        return semantic_groups
 
def create_random_incomplete_dataset_attr_groups(config_data, num_attribute_groups_remove=1):
    # mapping between attribute parts and attribute indices in the new 112 attribute space
    attribute_parts_indices_map = get_attribute_parts_to_indices(config_data)
    
    
    # Randomly select attribute groups to remove, and get the corresponding attribute indices to remove
    remove_attribute_parts = random.sample(ATTRIBUTE_PARTS, num_attribute_groups_remove)
    remove_attribute_indices = []
    for part in remove_attribute_parts:
        remove_attribute_indices.extend(attribute_parts_indices_map[part])  
        
    print(f"Removing attribute groups: {remove_attribute_parts} with attribute indices: {remove_attribute_indices}")
    
    num_attributes_remaining = config_data.num_concepts - len(remove_attribute_indices)
    
    path_to_incomplete_data_folder = os.path.join(config_data.data_path, "CUB", config_data.incomplete_dir)
    os.makedirs(path_to_incomplete_data_folder, exist_ok=True)
    
    
    # Create new pkl folder
    largest_digit = 0
    hostname = os.uname()[1]
    for folder_name in os.listdir(path_to_incomplete_data_folder):
        if "class_attr_data_10_incomplete_" in folder_name:
            last_digit = folder_name.split("_")[-1]
            if last_digit.isdigit():
                largest_digit = max(largest_digit, int(last_digit))
                
                
    if "biomed" in hostname:
        new_pkl_dir = f"class_attr_data_10_incomplete_cluster_{largest_digit + 1}/"
    else: 
        new_pkl_dir = f"class_attr_data_10_incomplete_local_{largest_digit + 1}/"
    new_folder_path = os.path.join(path_to_incomplete_data_folder,
                                   new_pkl_dir)
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Create mapping from old attribute indices to new attribute indices after removal, for info.txt
    old_to_new_attr_idx = {}
    new_idx = 0
    for old_idx in range(config_data.num_concepts):
        if old_idx not in remove_attribute_indices:
            old_to_new_attr_idx[old_idx] = new_idx
            new_idx += 1
   
    # Save info about the removed attribute groups and indices in a txt file in the new folder
    info_txt_path = os.path.join(new_folder_path, "info.txt")
    with open(info_txt_path, "w") as f:
        f.write(f"Mode: remove attribute groups\n")
        f.write(f"Removed attribute groups: {remove_attribute_parts}\n")
        f.write(f"Removed attribute indices: {sorted(remove_attribute_indices)}\n")
        f.write(f"Number of attribute groups removed: {num_attribute_groups_remove}\n")
        f.write(f"Total number of attributes removed: {len(remove_attribute_indices)}\n")
        # This is only used if you want to intervene on specific attribute groups and need to know which attribute indices correspond to which groups after the random removal
        f.write(f"New attribute indices mapping: {old_to_new_attr_idx}\n")
    
    # Modify pkl files and save to new folder
    pkl_files = ["train.pkl", "val.pkl", "test.pkl"]
    old_folder_path = os.path.join(config_data.data_path, "CUB", "class_attr_data_10")
    for pkl_file in pkl_files:
        pkl_path = os.path.join(old_folder_path, pkl_file)
        data = pickle.load(open(pkl_path, "rb"))
        
        for sample in data:
            sample["attribute_label"] = [
                v for i, v in enumerate(sample["attribute_label"]) if i not in remove_attribute_indices
            ]
        new_pkl_path = os.path.join(new_folder_path, pkl_file)
        with open(new_pkl_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved modified {pkl_file} to {new_pkl_path}")

    return new_pkl_dir, num_attributes_remaining


def create_random_incomplete_dataset_indiv_attr(config_data, ratio_attributes_remove=0.5):
    # Randomly select attribute indices to remove
    num_attributes_remove = int(ratio_attributes_remove * config_data.num_concepts)
    remove_attribute_indices = random.sample(range(config_data.num_concepts), num_attributes_remove)

    print(f"Removing individual attributes with indices: {remove_attribute_indices}")
    
    num_attributes_remaining = config_data.num_concepts - len(remove_attribute_indices)
    
    path_to_incomplete_data_folder = os.path.join(config_data.data_path, "CUB", config_data.incomplete_dir)
    os.makedirs(path_to_incomplete_data_folder, exist_ok=True)
    
    # Create new pkl folder
    largest_digit = 0
    hostname = os.uname()[1]
    for folder_name in os.listdir(path_to_incomplete_data_folder):
        if "class_attr_data_10_incomplete_" in folder_name:
            last_digit = folder_name.split("_")[-1]
            if last_digit.isdigit():
                largest_digit = max(largest_digit, int(last_digit))
    
                
    if "biomed" in hostname:
        new_pkl_dir = f"class_attr_data_10_incomplete_cluster_indiv_attr_{largest_digit + 1}/"
    else: 
        new_pkl_dir = f"class_attr_data_10_incomplete_local_indiv_attr_{largest_digit + 1}/"
    new_folder_path = os.path.join(path_to_incomplete_data_folder,
                                   new_pkl_dir)
    os.makedirs(new_folder_path, exist_ok=True)
    
    
    
    # Stat on number of attributes removed per semantic group
    attribute_parts_indices_map = get_attribute_parts_to_indices(config_data)
    num_concepts_removed_semantic_group = {}
    for semantic_group, indices in attribute_parts_indices_map.items():
        for idx in indices:
            if idx in remove_attribute_indices:
                if semantic_group not in num_concepts_removed_semantic_group:
                    num_concepts_removed_semantic_group[semantic_group] = 0
                num_concepts_removed_semantic_group[semantic_group] += 1
    
    # Create mapping from old attribute indices to new attribute indices after removal, for info.txt
    old_to_new_attr_idx = {}
    new_idx = 0
    for old_idx in range(config_data.num_concepts):
        if old_idx not in remove_attribute_indices:
            old_to_new_attr_idx[old_idx] = new_idx
            new_idx += 1
    
    # Save info about the removed attribute groups and indices in a txt file in the new folder
    info_txt_path = os.path.join(new_folder_path, "info.txt")
    with open(info_txt_path, "w") as f:
        f.write(f"Mode: remove individual attributes\n")
        f.write(f"Removed attribute indices: {sorted(remove_attribute_indices)}\n")
        f.write(f"Ratio of attributes removed: {ratio_attributes_remove}\n")
        f.write(f"Number of attributes removed: {num_attributes_remove}\n")
        f.write(f"Number of attributes removed per semantic group: {num_concepts_removed_semantic_group}\n")
        # This is only used if you want to intervene on specific attribute groups and need to know which attribute indices correspond to which groups after the random removal
        f.write(f"Mapping from old to new attribute indices: {old_to_new_attr_idx}\n")
    
    
    # Modify pkl files and save to new folder
    pkl_files = ["train.pkl", "val.pkl", "test.pkl"]
    old_folder_path = os.path.join(config_data.data_path, "CUB", "class_attr_data_10")
    for pkl_file in pkl_files:
        pkl_path = os.path.join(old_folder_path, pkl_file)
        data = pickle.load(open(pkl_path, "rb"))
        
        for sample in data:
            sample["attribute_label"] = [
                v for i, v in enumerate(sample["attribute_label"]) if i not in remove_attribute_indices
            ]
        new_pkl_path = os.path.join(new_folder_path, pkl_file)
        with open(new_pkl_path, "wb") as f:
            pickle.dump(data, f)    
        print(f"Saved modified {pkl_file} to {new_pkl_path}")
        
    return new_pkl_dir, num_attributes_remaining