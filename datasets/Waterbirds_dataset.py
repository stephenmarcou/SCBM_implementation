"""
Waterbirds dataset loader with CUB concept labels.

Waterbirds (Sagawa et al., ICLR 2020) renders every CUB-200-2011 bird onto a Places
background. The benchmark collapses the 200 species into two classes (0 = landbird,
1 = waterbird), with the background (`place`: 0 = land, 1 = water) as a spurious
attribute ~95% correlated with that binary label in train and exactly balanced within
each class in val and test.

**Which of the two labels is the target is set by `data.binary_target`.** It defaults to
False: the target is CUB's 200-way species label, which makes Waterbirds runs directly
comparable with the CUB and TravelingBirds runs (same task, same concepts, same
transforms) while still inheriting the spurious background. 2 classes is a far coarser
task for a 112-concept bottleneck - the concepts alone can nearly saturate it, leaving
little for the residual channel to carry - so prefer the 200-way target unless the binary
benchmark label is the point of the experiment.

    python train.py +model=SCBM_RES +data=Waterbirds                        # 200-way
    python train.py +model=SCBM_RES +data=Waterbirds data.binary_target=True  # 2-way

`data.num_classes` follows the flag automatically (`resolve_waterbirds_target` forces it to
2 when binary_target is set, from check_Waterbirds_data in train.py before the config is
written to log.txt), so it never has to be passed alongside. Both labels are
kept on every record either way - `species_label` and `waterbird_label` - so whichever is
not the target stays available as a post-hoc probe target on the saved c_mu / res_mu, via
`get_waterbirds_labels` / `get_waterbirds_species`.

Only the images and the split change - the concepts and the target are still CUB's - so
this module reuses `CUB_DatasetGenerator` and `get_CUB_transforms` unchanged and replaces
only the split function.

Waterbirds is deliberately *not* a member of `CUB_FAMILY_DATASETS`. That constant means
"same label pkls, same photographer split, same 200 classes, only the image root differs",
which holds for TravelingBirds but not here: Waterbirds cuts CUB's 5994 train photos into
4795 train / 1199 val, crossing the pkl boundary (953 images of `class_attr_data_10/
train.pkl` are Waterbirds val, and 952 of the pkl val are Waterbirds train). It *is* a
member of `CUB_CONCEPT_DATASETS`, which governs the concept-side code paths it does share:
concept names, concept grouping, and the incomplete-split machinery.

Classes:
    (none - reuses CUB_DatasetGenerator)

Functions:
    resolve_waterbirds_target: Reconcile data.num_classes with data.binary_target.
    train_test_split_Waterbirds: Train-validation-test split following metadata.csv.
    get_Waterbirds_dataloaders: Datasets for the Waterbirds train/val/test splits.
    get_waterbirds_places: Background label per sample, in dataset order.
    get_waterbirds_labels: Binary landbird/waterbird label per sample, in dataset order.
    get_waterbirds_species: CUB 200-way species label per sample, in dataset order.
"""

import csv
import os
import pickle

import torch

from datasets.CUB_dataset import (
    CUB_DatasetGenerator,
    CUB_LABEL_ROOT,
    get_CUB_transforms,
)


# Folder under data_path holding the rendered images and metadata.csv.
WATERBIRDS_ROOT = "Waterbirds"
WATERBIRDS_METADATA = "metadata.csv"

# metadata.csv encodes the split as an integer column.
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
# ... and the background as a binary `place` column.
PLACE_NAMES = {0: "land", 1: "water"}
# The benchmark's binary label (metadata.csv column `y`). Kept alongside the CUB species
# label rather than used as the target - see the module docstring.
WATERBIRD_CLASS_NAMES = {0: "landbird", 1: "waterbird"}
# CUB's species count, i.e. data.num_classes for the default (non-binary) target.
NUM_CUB_SPECIES = 200


def _load_cub_annotations(config, incomplete):
    """Map CUB image id -> {class_label, attribute_label, rel_path}, pooled over the pkls.

    Both the 200-way species label and the concept vector come from here, so the target and
    the concepts are guaranteed to be the same join.

    The pkls partition the 11788 CUB images differently from Waterbirds, so the lookup has
    to be built across all three splits before being re-split by metadata.csv. `id` is
    CUB's own image id (1..11788, matching images.txt), and the incomplete generators only
    rewrite `attribute_label` in place, so the same join works for incomplete splits.

    `rel_path` is `<species_folder>/<file>.jpg`, carved off the tail of the pkl's absolute
    `img_path` (which still points at the original authors' machine). It is not used to
    load anything - Waterbirds reads its own rendered copies - only to verify the id join
    against metadata.csv's `img_filename`, which uses the same two-component form.
    """
    if not incomplete:
        pkl_dir = os.path.join(config.data_path, CUB_LABEL_ROOT, "class_attr_data_10")
    else:
        pkl_dir = os.path.join(
            config.data_path, CUB_LABEL_ROOT, "incomplete_data", config.pkl_file_dir
        )
        print(f"Using incomplete dataset with pkl files from {config.pkl_file_dir}")
    
    
    # ID to (species label, attribute vector, relative image path) mapping
    annotations = {}
    for pkl_file in ("train.pkl", "val.pkl", "test.pkl"):
        with open(os.path.join(pkl_dir, pkl_file), "rb") as f:
            for sample in pickle.load(f):
                annotations[sample["id"]] = {
                    "class_label": sample["class_label"],
                    "attribute_label": sample["attribute_label"],
                    "rel_path": "/".join(sample["img_path"].split("/")[-2:]),
                }
    return annotations


def resolve_waterbirds_target(config_data):
    """Make `num_classes` agree with `binary_target`, and return which target is in force.

    Single source of truth for the two-target switch, called from three places: train.py's
    check_Waterbirds_data (before the config is logged), inference.py (after the flag is
    recovered from log.txt), and the split function itself (so any entry point that reaches
    the loader directly still gets a consistent config rather than a silent mismatch).

    Idempotent, and silent when it has nothing to change - so the second and third calls in
    a run print nothing.

    Args:
        config_data: the `data` config node (mutated in place).

    Returns:
        bool: True if the binary landbird/waterbird label is the target.
    """
    binary_target = bool(config_data.get("binary_target", False))
    if binary_target:
        if config_data.num_classes != 2:
            print(
                f"Waterbirds: binary_target=True, overriding data.num_classes "
                f"{config_data.num_classes} -> 2 (landbird / waterbird)."
            )
        config_data.num_classes = 2
    elif config_data.num_classes != NUM_CUB_SPECIES:
        raise ValueError(
            f"Waterbirds with binary_target=False targets CUB's {NUM_CUB_SPECIES} species, "
            f"but data.num_classes is {config_data.num_classes}. Pass "
            "data.binary_target=True for the 2-class experiment rather than setting "
            "num_classes by hand."
        )
    return binary_target


def train_test_split_Waterbirds(config, incomplete):
    """Performs train-validation-test split for the Waterbirds dataset.

    Uses the `split` column of metadata.csv, i.e. the partition the benchmark is defined
    on: train is spuriously correlated, val and test are background-balanced within each
    class. Train and val come from CUB's official train photographers and test from CUB's
    official test photographers, so the photographer separation is preserved.

    `class_label` is the key CUB_DatasetGenerator emits as `labels`, i.e. the training
    target. Which label lands there is set by `config.binary_target`: the benchmark's
    binary landbird/waterbird label when True, CUB's 200-way species label otherwise. Both
    are always kept on the record as `waterbird_label` and `species_label`, so the one that
    is not the target stays available for post-hoc analysis.

    `config.num_classes` is kept in step with the flag by resolve_waterbirds_target. In a
    train.py run check_Waterbirds_data has already applied it before the config was logged;
    the call here is the backstop for entry points that reach the loader directly.
    """
    binary_target = resolve_waterbirds_target(config)

    images_root = os.path.join(config.data_path, WATERBIRDS_ROOT)
    metadata_path = os.path.join(images_root, WATERBIRDS_METADATA)
    if not os.path.isfile(metadata_path):
        raise ValueError(f"Waterbirds metadata not found at {metadata_path}.")

    # Concepts always come from the CUB folder: Waterbirds ships no attribute annotations
    # of its own, and the birds are the CUB birds.
    annotations = _load_cub_annotations(config, incomplete)

    split_datasets = {"train": [], "val": [], "test": []}
    with open(metadata_path, "r") as f:
        for row in csv.DictReader(f):
            img_id = int(row["img_id"])
            if img_id not in annotations:
                raise ValueError(
                    f"Waterbirds image id {img_id} has no CUB annotation. "
                    "The metadata.csv and the label pkls are out of sync."
                )
            # CUB image id -> {class_label, attribute_label, rel_path}
            annotation = annotations[img_id]

            # The labels are joined by CUB image id, but both files independently name the
            # image: metadata.csv as `img_filename` and the pkl as the tail of `img_path`.
            # Requiring the two to agree pins the join per image, so an out-of-sync
            # metadata.csv or a re-indexed pkl fails here instead of passing silently with
            # a wrong target and a wrong concept vector on every sample.
            if annotation["rel_path"] != row["img_filename"]:
                raise ValueError(
                    f"Waterbirds image id {img_id} is named {row['img_filename']} in "
                    f"metadata.csv but {annotation['rel_path']} in the CUB pkl. The "
                    "metadata and the label pkls are out of sync."
                )

            split_datasets[SPLIT_NAMES[int(row["split"])]].append(
                {
                    "id": img_id,
                    "img_path": os.path.join(images_root, row["img_filename"]),
                    # The training target: this is the only key CUB_DatasetGenerator reads,
                    # and it emits it as `labels`. data.binary_target picks which of the two
                    # labels below fills it.
                    "class_label": (
                        int(row["y"]) if binary_target else annotation["class_label"]
                    ),
                    "attribute_label": annotation["attribute_label"],
                    # Both labels, kept unconditionally alongside the target, plus the
                    # spurious background attribute. None of these three is consumed by
                    # CUB_DatasetGenerator (so the model never sees them); they are here so
                    # post-hoc analysis can recover them via get_waterbirds_species /
                    # get_waterbirds_labels / get_waterbirds_places, and so switching the
                    # target needs no re-render.
                    "species_label": annotation["class_label"],
                    "waterbird_label": int(row["y"]),
                    "place": int(row["place"]),
                }
            )

    for split_name, dataset in split_datasets.items():
        n = len(dataset)
        # Spurious correlation is defined against the *binary* label, not the species one.
        matched = sum(s["waterbird_label"] == s["place"] for s in dataset)
        n_species = len({s["species_label"] for s in dataset})
        target_desc = "2-way landbird/waterbird" if binary_target else "200-way species"
        print(
            f"Waterbirds {split_name}: {n} samples, {n_species} species, "
            f"{100 * matched / n:.1f}% on their majority background, "
            f"target = {target_desc}"
        )

    return split_datasets["train"], split_datasets["val"], split_datasets["test"]


def get_Waterbirds_dataloaders(config, incomplete):
    """Returns a dictionary of datasets for Waterbirds, for the training, validation, and test sets."""
    train_imgs, val_imgs, test_imgs = train_test_split_Waterbirds(config, incomplete)

    # Same transforms as CUB, so that Waterbirds runs stay comparable with the CUB and
    # TravelingBirds runs (rather than with published Waterbirds numbers, which use a
    # different crop).
    train_transform, test_transform = get_CUB_transforms(resol=299)

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


def get_waterbirds_places(dataset):
    """Background label (0 = land, 1 = water) per sample, in dataset order.

    Analysis loaders are shuffle-free with drop_last=False, so row i of this tensor lines
    up with row i of the saved c_mu / res_mu artifacts.
    """
    return torch.tensor([sample["place"] for sample in dataset.data], dtype=torch.long)


def get_waterbirds_labels(dataset):
    """Binary landbird/waterbird label (0 = landbird, 1 = waterbird) per sample.

    The benchmark's own target. Under data.binary_target=True this is what the loader
    emitted as `labels`; otherwise the model was trained on the species label and this is a
    probe target on the saved c_mu / res_mu. Row-aligned with the analysis-loader artifacts
    on the same terms as get_waterbirds_places.
    """
    return torch.tensor(
        [sample["waterbird_label"] for sample in dataset.data], dtype=torch.long
    )


def get_waterbirds_species(dataset):
    """CUB's 200-way species label (0..199) per sample.

    The mirror of get_waterbirds_labels: the loader's `labels` under the default
    data.binary_target=False, and a probe target on the saved c_mu / res_mu under
    binary_target=True. Row-aligned on the same terms as get_waterbirds_places.
    """
    return torch.tensor(
        [sample["species_label"] for sample in dataset.data], dtype=torch.long
    )
