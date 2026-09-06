"""
AwA2 dataset loader with concept labels, designed to match the interface of the
CUB loader used in this codebase while preserving the AwA2 preprocessing used
in the CEM/ECBM data loader.

Expected raw AwA2 structure
---------------------------
<data_path>/AwA2/
    classes.txt
    predicate-matrix-binary.txt
    JPEGImages/
        antelope/
        grizzly+bear/
        ...

CEM-compatible behaviour
------------------------
1. 50-way supervised image classification.
2. 85 binary, class-level attributes from predicate-matrix-binary.txt.
3. Random image-level train/val/test split: 60% / 20% / 20%, seed 42.
4. CEM AwA2 image preprocessing:
   - no augmentation: Resize(256/224 * image_size) -> CenterCrop(image_size)
   - augmentation: RandomResizedCrop + RandomHorizontalFlip
   - ImageNet normalization in both cases.

Incomplete concept datasets
---------------------------
Incomplete datasets are stored under:

<data_path>/AwA2/incomplete_data/<dataset_name>/

Only the concept matrix is reduced. The image split is intentionally kept
unchanged, so full and incomplete runs use exactly the same images and class
labels.
"""

import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


N_CLASSES = 50
N_CONCEPTS = 85
AWA2_FOLDER = "AwA2"
DEFAULT_SEED = 42


# Same 85 concepts, in the same order, as predicate-matrix-binary.txt in AwA2
# and the CEM/ECBM AwA2 processing code.
CONCEPT_SEMANTICS = [
    "black", "white", "blue", "brown", "gray", "orange", "red", "yellow",
    "patches", "spots", "stripes", "furry", "hairless", "toughskin",
    "big", "small", "bulbous", "lean", "flippers", "hands", "hooves",
    "pads", "paws", "longleg", "longneck", "tail", "chewteeth",
    "meatteeth", "buckteeth", "strainteeth", "horns", "claws", "tusks",
    "smelly", "flys", "hops", "swims", "tunnels", "walks", "fast",
    "slow", "strong", "weak", "muscle", "bipedal", "quadrapedal",
    "active", "inactive", "nocturnal", "hibernate", "agility", "fish",
    "meat", "plankton", "vegetation", "insects", "forager", "grazer",
    "hunter", "scavenger", "skimmer", "stalker", "newworld", "oldworld",
    "arctic", "coastal", "desert", "bush", "plains", "forest", "fields",
    "jungle", "mountains", "ocean", "ground", "water", "tree", "cave",
    "fierce", "timid", "smart", "group", "solitary", "nestspot",
    "domestic",
]


# Semantic grouping from the CEM/ECBM AwA2 processing code.
# OrderedDict is used so CEM-style random group sampling is reproducible with
# the same group order.
_CONCEPT_GROUP_NAMES = OrderedDict([
    ("color", ["black", "white", "blue", "brown", "gray", "orange", "red", "yellow"]),
    ("fur_pattern", ["patches", "spots", "stripes", "furry", "hairless", "toughskin"]),
    ("size", ["big", "small", "bulbous", "lean"]),
    ("limb_shape", ["flippers", "hands", "hooves", "pads", "paws", "longleg", "longneck"]),
    ("tail", ["tail"]),
    ("teeth_type", ["chewteeth", "meatteeth", "buckteeth", "strainteeth"]),
    ("horns", ["horns"]),
    ("claws", ["claws"]),
    ("tusks", ["tusks"]),
    ("smelly", ["smelly"]),
    ("transport_mechanism", ["flys", "hops", "swims", "tunnels", "walks"]),
    ("speed", ["fast", "slow"]),
    ("strength", ["strong", "weak"]),
    ("muscle", ["muscle"]),
    ("movement_move", ["bipedal", "quadrapedal"]),
    ("active", ["active", "inactive"]),
    ("nocturnal", ["nocturnal"]),
    ("hibernate", ["hibernate"]),
    ("agility", ["agility"]),
    ("diet", ["fish", "meat", "plankton", "vegetation", "insects"]),
    ("feeding_type", ["forager", "grazer", "hunter", "scavenger", "skimmer", "stalker"]),
    ("general_location", ["newworld", "oldworld", "arctic"]),
    ("biome", ["coastal", "desert", "bush", "plains", "forest", "fields", "jungle", "mountains", "ocean", "ground", "water", "tree", "cave"]),
    ("fierceness", ["fierce", "timid"]),
    ("smart", ["smart"]),
    ("social_mode", ["group", "solitary"]),
    ("nestspot", ["nestspot"]),
    ("domestic", ["domestic"]),
])

CONCEPT_GROUPS = OrderedDict(
    (
        group_name,
        [CONCEPT_SEMANTICS.index(concept_name) for concept_name in concept_names],
    )
    for group_name, concept_names in _CONCEPT_GROUP_NAMES.items()
)


# -----------------------------------------------------------------------------
# Path / metadata helpers
# -----------------------------------------------------------------------------

def _get_awa2_root(config_data):
    """Return <config.data_path>/AwA2."""
    return os.path.join(config_data.data_path, AWA2_FOLDER)


def _get_incomplete_root(config_data):
    """
    Return the folder that stores incomplete AwA2 concept sets.

    By default this is AwA2/incomplete_data. If config.incomplete_dir is set,
    it is used so this matches the convention in the CUB loader.
    """
    incomplete_dir = getattr(config_data, "incomplete_dir", "incomplete_data")
    return os.path.join(_get_awa2_root(config_data), incomplete_dir)


def _get_incomplete_dataset_name(config_data):
    """
    Get the configured incomplete dataset folder name.

    `pkl_file_dir` is accepted deliberately because the existing CUB training
    pipeline already uses that config field. The name is a misnomer here - an
    incomplete AwA2 set is a reduced predicate matrix, not a set of pkls - but every
    writer in the pipeline uses it (check_AwA2_data assigns it, train.py logs it,
    inference.py reads it back out of log.txt), so a second read-side-only alias would
    only produce runs that train once and cannot be re-analysed.
    """
    if hasattr(config_data, "pkl_file_dir") and config_data.pkl_file_dir:
        return str(config_data.pkl_file_dir).strip("/")
    raise ValueError(
        "For incomplete AwA2, set data.pkl_file_dir to the folder of an existing "
        "concept set (same config field as CUB). Sets are created explicitly, via "
        "create_custom_incomplete_dataset or the create_random_incomplete_dataset_* "
        "helpers in this module."
    )


def get_incomplete_concept_set_path(config_data):
    """Absolute path of the configured incomplete AwA2 concept set.

    Public counterpart of the private path helpers above. train.py and inference.py have
    to test for and read back an incomplete set, and this keeps the
    AwA2/<incomplete_dir>/<folder> layout in one place rather than re-derived at each call
    site - the role CUB_LABEL_ROOT plays for the CUB family.
    """
    return os.path.join(
        _get_incomplete_root(config_data),
        _get_incomplete_dataset_name(config_data),
    )


def get_num_concepts_incomplete(config_data):
    """Number of concepts retained by the configured incomplete concept set.

    Read off the reduced predicate matrix rather than info.json, so the value is the one
    the loader will actually hand the model (same reason check_CUB_data reads the pkl
    contents instead of info.txt).
    """
    return int(_load_predicate_matrix(config_data, incomplete=True).shape[1])


def _load_class_to_index(root_dir):
    """Match CEM: class indices follow the line order in classes.txt."""
    class_to_index = {}
    classes_path = os.path.join(root_dir, "classes.txt")
    with open(classes_path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"Unexpected line in classes.txt: {line!r}")
            class_name = parts[1].strip()
            class_to_index[class_name] = len(class_to_index)

    if len(class_to_index) != N_CLASSES:
        raise ValueError(
            f"Expected {N_CLASSES} classes in {classes_path}, found {len(class_to_index)}."
        )
    return class_to_index


def _load_predicate_matrix(config_data, incomplete=False):
    """Load the full or reduced class x concept binary matrix."""
    awa2_root = _get_awa2_root(config_data)

    if incomplete:
        dataset_name = _get_incomplete_dataset_name(config_data)
        matrix_path = os.path.join(
            _get_incomplete_root(config_data),
            dataset_name,
            "predicate-matrix-binary.txt",
        )
        print(f"Using incomplete AwA2 concept matrix from {dataset_name}")
    else:
        matrix_path = os.path.join(awa2_root, "predicate-matrix-binary.txt")

    matrix = np.asarray(np.genfromtxt(matrix_path, dtype=int))

    if matrix.ndim != 2 or matrix.shape[0] != N_CLASSES:
        raise ValueError(
            f"Expected an AwA2 concept matrix with {N_CLASSES} rows, got {matrix.shape} "
            f"from {matrix_path}."
        )

    return matrix


def _resolve_saved_image_path(saved_path, root_dir):
    """
    Resolve paths stored in CEM-style split .npz files.

    CEM stores absolute paths. If the dataset/split file was copied from local
    to the cluster, the old absolute path may no longer exist. In that case we
    keep the same split membership and re-root the suffix after JPEGImages/.
    """
    saved_path = str(saved_path)
    if os.path.exists(saved_path):
        return saved_path

    normalized = saved_path.replace("\\", "/")
    marker = "/JPEGImages/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        return os.path.join(root_dir, "JPEGImages", *suffix.split("/"))

    if normalized.startswith("JPEGImages/"):
        return os.path.join(root_dir, *normalized.split("/"))

    return saved_path


# -----------------------------------------------------------------------------
# Dataset and CEM-compatible splits
# -----------------------------------------------------------------------------

class AWA2_DatasetGenerator(Dataset):
    """
    AwA2 Dataset object with the same sample dictionary interface as the CUB
    loader in this codebase.

    Each image gets the class-level binary attribute vector corresponding to
    its class in predicate-matrix-binary.txt.
    """

    def __init__(self, data, predicate_binary_mat, transform=None):
        self.data = data
        self.predicate_binary_mat = np.asarray(predicate_binary_mat, dtype=np.float32)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.data[index]
        img_path = sample["img_path"]
        image_label = int(sample["class_label"])

        image_data = Image.open(img_path).convert("RGB")
        image_attr = self.predicate_binary_mat[image_label, :].astype(np.float32)

        if self.transform is not None:
            image_data = self.transform(image_data)

        # Same dictionary keys used by CUB_DatasetGenerator.
        return {
            "img_code": index,
            "labels": image_label,
            "features": image_data,
            "concepts": image_attr,
        }

    def __len__(self):
        return len(self.data)


def _generate_cem_splits(root_dir, seed=DEFAULT_SEED, train_size=0.6, val_size=0.2):
    """
    Generate the same random image-level 60/20/20 split used by the CEM AwA2
    processing script and save train_split.npz / val_split.npz / test_split.npz.
    """
    class_to_index = _load_class_to_index(root_dir)

    image_paths = []
    image_classes = []
    img_dir = os.path.join(root_dir, "JPEGImages")

    for walk_root, _, files in os.walk(img_dir):
        for filename in files:
            if filename.lower().endswith(".jpg"):
                img_path = os.path.abspath(os.path.join(walk_root, filename))
                parent_dir = os.path.basename(os.path.dirname(img_path))
                image_paths.append(img_path)
                image_classes.append(class_to_index[parent_dir])

    if not image_paths:
        raise ValueError(f"No .jpg images found under {img_dir}")

    # Deliberately match the original CEM implementation.
    np.random.seed(seed)
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)

    train_end = int(train_size * len(image_paths))
    val_end = train_end + int(val_size * len(image_paths))

    image_paths = np.asarray(image_paths)
    image_classes = np.asarray(image_classes, dtype=np.int64)

    split_specs = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    for split_name, split_indices in split_specs.items():
        np.savez(
            os.path.join(root_dir, f"{split_name}_split.npz"),
            paths=image_paths[split_indices],
            labels=image_classes[split_indices],
        )

    print(
        "Generated CEM-style AwA2 splits: "
        f"train={len(split_specs['train'])}, "
        f"val={len(split_specs['val'])}, "
        f"test={len(split_specs['test'])}"
    )


def train_val_test_split_AWA2(config_data, seed=DEFAULT_SEED):
    """
    Load (or create) the CEM-style AwA2 60/20/20 image-level split.

    Returns three lists of dictionaries with keys:
        img_path, class_label
    """
    root_dir = _get_awa2_root(config_data)

    for split_name in ("train", "val", "test"):
        split_path = os.path.join(root_dir, f"{split_name}_split.npz")
        if not os.path.exists(split_path):
            print(
                "AwA2 split files not found. Generating CEM-style train/val/test "
                f"split with seed {seed}."
            )
            _generate_cem_splits(root_dir, seed=seed)
            break

    output = {}
    for split_name in ("train", "val", "test"):
        split_path = os.path.join(root_dir, f"{split_name}_split.npz")
        split_info = np.load(split_path)
        paths = split_info["paths"]
        labels = split_info["labels"]

        split_data = []
        for img_path, class_label in zip(paths, labels):
            resolved_path = _resolve_saved_image_path(img_path, root_dir)
            split_data.append(
                {
                    "img_path": resolved_path,
                    "class_label": int(class_label),
                }
            )
        output[split_name] = split_data

    print(
        "AwA2 samples: "
        f"train={len(output['train'])}, "
        f"val={len(output['val'])}, "
        f"test={len(output['test'])}"
    )
    return output["train"], output["val"], output["test"]


# -----------------------------------------------------------------------------
# CEM-compatible image transforms
# -----------------------------------------------------------------------------

def get_AWA2_transforms(image_size=224, augment_data=False):
    """
    Return train/test transforms matching the CEM AwA2 processing.

    Important: in the original CEM code, if augment_data=False then the train
    split uses the same deterministic Resize + CenterCrop path as validation
    and test.
    """
    scale = 256.0 / 224.0

    test_transform = transforms.Compose([
        transforms.Resize((int(image_size * scale), int(image_size * scale))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if augment_data:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        train_transform = test_transform

    return train_transform, test_transform


def get_AWA2_dataloaders(config, incomplete=False):
    """
    Return train/val/test Dataset objects, matching the interface of
    get_CUB_dataloaders in this codebase.

    Despite the historical function name, this returns Dataset objects rather
    than torch DataLoader objects, just like the attached CUB implementation.
    """
    seed = getattr(config, "seed", DEFAULT_SEED)
    image_size = getattr(config, "image_size", 224)
    augment_data = getattr(config, "augment_data", False)

    train_imgs, val_imgs, test_imgs = train_val_test_split_AWA2(config, seed=seed)
    predicate_binary_mat = _load_predicate_matrix(config, incomplete=incomplete)
    train_transform, test_transform = get_AWA2_transforms(
        image_size=image_size,
        augment_data=augment_data,
    )

    image_datasets = {
        "train": AWA2_DatasetGenerator(
            train_imgs,
            predicate_binary_mat,
            transform=train_transform,
        ),
        "val": AWA2_DatasetGenerator(
            val_imgs,
            predicate_binary_mat,
            transform=test_transform,
        ),
        "test": AWA2_DatasetGenerator(
            test_imgs,
            predicate_binary_mat,
            transform=test_transform,
        ),
    }

    return (
        image_datasets["train"],
        image_datasets["val"],
        image_datasets["test"],
    )


# -----------------------------------------------------------------------------
# Concept-group utilities
# -----------------------------------------------------------------------------

def get_attribute_parts_to_indices(config_data=None):
    """Return the CEM AwA2 semantic concept groups as 0-based indices."""
    return OrderedDict((k, list(v)) for k, v in CONCEPT_GROUPS.items())


def _next_incomplete_dataset_dir(config_data, mode_name):
    """Create a new numbered incomplete-data folder, similar to the CUB code."""
    root = _get_incomplete_root(config_data)
    os.makedirs(root, exist_ok=True)

    hostname = os.uname()[1] if hasattr(os, "uname") else "local"
    environment = "cluster" if "biomed" in hostname else "local"
    prefix = f"awa2_incomplete_{environment}_{mode_name}_"

    largest_digit = 0
    for folder_name in os.listdir(root):
        if folder_name.startswith(prefix):
            suffix = folder_name[len(prefix):]
            if suffix.isdigit():
                largest_digit = max(largest_digit, int(suffix))

    new_dir_name = f"{prefix}{largest_digit + 1}"
    new_folder_path = os.path.join(root, new_dir_name)
    os.makedirs(new_folder_path, exist_ok=False)
    return new_dir_name, new_folder_path


def _remap_concept_groups(selected_concepts):
    """Remap original 0..84 concept indices into a reduced concept space."""
    selected_concepts = [int(x) for x in selected_concepts]
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_concepts)}
    selected_set = set(selected_concepts)

    new_groups = OrderedDict()
    for group_name, original_indices in CONCEPT_GROUPS.items():
        kept = [remap[idx] for idx in original_indices if idx in selected_set]
        if kept:
            new_groups[group_name] = kept

    return remap, new_groups


def _save_incomplete_concept_set(
    config_data,
    selected_concepts,
    mode_name,
    metadata,
):
    """
    Save a persistent reduced AwA2 predicate matrix and metadata.

    The original image split files remain untouched.
    """
    selected_concepts = sorted(int(x) for x in selected_concepts)
    if not selected_concepts:
        raise ValueError("An incomplete dataset must retain at least one concept.")
    if min(selected_concepts) < 0 or max(selected_concepts) >= N_CONCEPTS:
        raise ValueError("selected_concepts contains an index outside 0..84.")

    full_matrix_path = os.path.join(
        _get_awa2_root(config_data),
        "predicate-matrix-binary.txt",
    )
    full_matrix = np.asarray(np.genfromtxt(full_matrix_path, dtype=int))
    if full_matrix.shape != (N_CLASSES, N_CONCEPTS):
        raise ValueError(
            f"Expected full AwA2 matrix shape {(N_CLASSES, N_CONCEPTS)}, "
            f"got {full_matrix.shape}."
        )

    reduced_matrix = full_matrix[:, selected_concepts]
    removed_concepts = sorted(set(range(N_CONCEPTS)) - set(selected_concepts))
    old_to_new, new_groups = _remap_concept_groups(selected_concepts)

    new_dir_name, new_folder_path = _next_incomplete_dataset_dir(
        config_data,
        mode_name=mode_name,
    )

    np.savetxt(
        os.path.join(new_folder_path, "predicate-matrix-binary.txt"),
        reduced_matrix,
        fmt="%d",
    )
    np.save(
        os.path.join(new_folder_path, "selected_concepts.npy"),
        np.asarray(selected_concepts, dtype=np.int64),
    )
    np.save(
        os.path.join(new_folder_path, "removed_concepts.npy"),
        np.asarray(removed_concepts, dtype=np.int64),
    )

    selected_names = [CONCEPT_SEMANTICS[idx] for idx in selected_concepts]
    removed_names = [CONCEPT_SEMANTICS[idx] for idx in removed_concepts]

    with open(os.path.join(new_folder_path, "concept_names.txt"), "w") as f:
        for new_idx, (old_idx, name) in enumerate(zip(selected_concepts, selected_names)):
            f.write(f"{new_idx}\t{old_idx}\t{name}\n")

    with open(os.path.join(new_folder_path, "concept_groups.json"), "w") as f:
        json.dump(new_groups, f, indent=2)

    all_metadata = {
        "mode": metadata.get("mode", mode_name),
        "num_original_concepts": N_CONCEPTS,
        "num_remaining_concepts": len(selected_concepts),
        "num_removed_concepts": len(removed_concepts),
        "selected_concept_indices_original_space": selected_concepts,
        "selected_concept_names": selected_names,
        "removed_concept_indices_original_space": removed_concepts,
        "removed_concept_names": removed_names,
        "old_to_new_concept_index": old_to_new,
        **metadata,
    }

    with open(os.path.join(new_folder_path, "info.json"), "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Human-readable equivalent, convenient when inspecting experiment folders.
    with open(os.path.join(new_folder_path, "info.txt"), "w") as f:
        for key, value in all_metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"Saved incomplete AwA2 concept set to {new_folder_path}")
    print(
        f"Concepts: {N_CONCEPTS} -> {len(selected_concepts)} "
        f"(removed {len(removed_concepts)})"
    )

    # Keep the same return pattern as your CUB incomplete-data functions.
    return new_dir_name + "/", len(selected_concepts)



# -----------------------------------------------------------------------------
# Incomplete AwA2 creation: Specify concepts to keep
# -----------------------------------------------------------------------------
def create_custom_incomplete_dataset(
    config_data,
    selected_concept_names,
):
    """
    Create an incomplete AwA2 dataset using a manually specified
    set of concepts to KEEP.

    Parameters
    ----------
    config_data :
        Dataset configuration.

    selected_concept_names : list[str]
        Concept names from CONCEPT_SEMANTICS that should be retained.

    Returns
    -------
    new_dir_name : str
        Name of the new incomplete dataset folder.

    num_concepts : int
        Number of concepts retained.
    """

    if not selected_concept_names:
        raise ValueError("You must select at least one concept.")

    # Check that every requested concept actually exists
    invalid_concepts = [
        concept
        for concept in selected_concept_names
        if concept not in CONCEPT_SEMANTICS
    ]

    if invalid_concepts:
        raise ValueError(
            f"Unknown AwA2 concepts: {invalid_concepts}"
        )

    # Convert names -> original 0-based AwA2 indices
    selected_concepts = [
        CONCEPT_SEMANTICS.index(concept)
        for concept in selected_concept_names
    ]

    # Remove duplicates and restore original AwA2 concept ordering
    selected_concepts = sorted(set(selected_concepts))

    print("Keeping AwA2 concepts:")
    for idx in selected_concepts:
        print(f"  {idx}: {CONCEPT_SEMANTICS[idx]}")

    return _save_incomplete_concept_set(
        config_data,
        selected_concepts=selected_concepts,
        mode_name="custom",
        metadata={
            "mode": "manually selected concepts",
            "requested_concept_names": selected_concept_names,
        },
    )




















# -----------------------------------------------------------------------------
# Incomplete AwA2 creation: CUB-style removal API
# -----------------------------------------------------------------------------

def create_random_incomplete_dataset_attr_groups(
    config_data,
    num_attribute_groups_remove=1,
    seed=None,
):
    """
    Remove complete AwA2 semantic attribute groups, analogous to the CUB
    group-removal helper.
    """
    group_names = list(CONCEPT_GROUPS.keys())
    if not 0 <= num_attribute_groups_remove < len(group_names):
        raise ValueError(
            f"num_attribute_groups_remove must be in [0, {len(group_names) - 1}]"
        )

    rng = random.Random(seed) if seed is not None else random
    remove_attribute_groups = rng.sample(group_names, num_attribute_groups_remove)

    remove_attribute_indices = []
    for group_name in remove_attribute_groups:
        remove_attribute_indices.extend(CONCEPT_GROUPS[group_name])
    remove_attribute_indices = sorted(set(remove_attribute_indices))

    selected_concepts = [
        idx for idx in range(N_CONCEPTS) if idx not in set(remove_attribute_indices)
    ]

    print(
        f"Removing AwA2 attribute groups: {remove_attribute_groups}\n"
        f"Removing original concept indices: {remove_attribute_indices}"
    )

    return _save_incomplete_concept_set(
        config_data,
        selected_concepts=selected_concepts,
        mode_name="groups",
        metadata={
            "mode": "remove attribute groups",
            "removed_attribute_groups": remove_attribute_groups,
            "num_attribute_groups_removed": num_attribute_groups_remove,
            "seed": seed,
        },
    )


def create_random_incomplete_dataset_indiv_attr(
    config_data,
    ratio_attributes_remove=0.5,
    seed=None,
):
    """
    Remove a random fraction of individual AwA2 attributes, analogous to the
    CUB individual-attribute removal helper.
    """
    if not 0 <= ratio_attributes_remove < 1:
        raise ValueError("ratio_attributes_remove must be in [0, 1).")

    num_attributes_remove = int(ratio_attributes_remove * N_CONCEPTS)
    rng = random.Random(seed) if seed is not None else random
    remove_attribute_indices = sorted(
        rng.sample(range(N_CONCEPTS), num_attributes_remove)
    )
    remove_set = set(remove_attribute_indices)
    selected_concepts = [idx for idx in range(N_CONCEPTS) if idx not in remove_set]

    removed_per_group = {}
    for group_name, indices in CONCEPT_GROUPS.items():
        count = sum(idx in remove_set for idx in indices)
        if count:
            removed_per_group[group_name] = count

    print(f"Removing individual AwA2 attributes: {remove_attribute_indices}")

    return _save_incomplete_concept_set(
        config_data,
        selected_concepts=selected_concepts,
        mode_name="individual",
        metadata={
            "mode": "remove individual attributes",
            "ratio_attributes_removed": ratio_attributes_remove,
            "number_attributes_removed": num_attributes_remove,
            "number_removed_per_semantic_group": removed_per_group,
            "seed": seed,
        },
    )


# -----------------------------------------------------------------------------
# Incomplete AwA2 creation: exact CEM-style sampling API
# -----------------------------------------------------------------------------

def create_cem_incomplete_dataset(
    config_data,
    sampling_percent=0.5,
    sampling_groups=False,
    seed=DEFAULT_SEED,
):
    """
    Persist the same concept subsampling logic used by the CEM AwA2 loader.

    Parameters
    ----------
    sampling_percent : float
        FRACTION OF CONCEPTS/GROUPS TO KEEP, not the fraction to remove.
        Example: sampling_percent=0.5 keeps approximately 50%.

    sampling_groups : bool
        False -> sample individual concepts.
        True  -> sample entire semantic groups and keep all concepts belonging
                 to the selected groups.

    seed : int
        Seed used for the NumPy permutation. Default 42.
    """
    if not 0 < sampling_percent <= 1:
        raise ValueError("sampling_percent must be in (0, 1].")

    rng = np.random.RandomState(seed)

    if sampling_groups:
        group_names = list(CONCEPT_GROUPS.keys())
        new_n_groups = int(np.ceil(len(group_names) * sampling_percent))
        selected_group_indices = sorted(
            rng.permutation(len(group_names))[:new_n_groups].tolist()
        )
        selected_groups = [group_names[i] for i in selected_group_indices]

        selected_concepts = []
        for group_name in selected_groups:
            selected_concepts.extend(CONCEPT_GROUPS[group_name])
        selected_concepts = sorted(set(selected_concepts))

        metadata = {
            "mode": "CEM-style group sampling",
            "sampling_percent_kept": sampling_percent,
            "sampling_groups": True,
            "selected_group_indices": selected_group_indices,
            "selected_groups": selected_groups,
            "seed": seed,
        }
        mode_name = "cem_groups"
    else:
        new_n_concepts = int(np.ceil(N_CONCEPTS * sampling_percent))
        selected_concepts = sorted(
            rng.permutation(N_CONCEPTS)[:new_n_concepts].tolist()
        )
        metadata = {
            "mode": "CEM-style individual concept sampling",
            "sampling_percent_kept": sampling_percent,
            "sampling_groups": False,
            "seed": seed,
        }
        mode_name = "cem_individual"

    print(f"Selected original AwA2 concept indices: {selected_concepts}")

    return _save_incomplete_concept_set(
        config_data,
        selected_concepts=selected_concepts,
        mode_name=mode_name,
        metadata=metadata,
    )
