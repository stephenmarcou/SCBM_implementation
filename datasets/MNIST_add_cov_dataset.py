"""
Controlled two-digit MNIST dataset for residual/concept covariance experiments.

Input:
    Two MNIST digit images stacked as channels -> tensor [2, 28, 28].

Observed supervised concepts:
    A1 = 1[d1 is odd]
    H1 = 1[d1 >= 6]
    H2 = 1[d2 >= 6]

Hidden/oracle variables (NEVER included in `concepts`):
    A2 = 1[d2 is odd]
    X  = planted_function(A1, A2)

Target:
    M = H1 OR H2
    y = 4*A1 + 2*M + X        # integer in {0, ..., 7}

The function get_MNIST_add_cov_datasets(...) returns Dataset objects rather
than DataLoaders, matching the way the project's utils/data.py wraps datasets
in a common DataLoader afterwards.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import functional as TF


OBSERVED_CONCEPT_NAMES = [
    "A1::digit1_is_odd",
    "H1::digit1_ge_6",
    "H2::digit2_ge_6",
]

ORACLE_NAMES = [
    "A2::digit2_is_odd",
    "X::planted_hidden_function",
]


def _cfg_get(config, name: str, default):
    """Works with dicts, OmegaConf/DictConfig, and simple config objects."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    if hasattr(config, name):
        return getattr(config, name)
    try:
        return config.get(name, default)
    except Exception:
        return default


def _fingerprint_arrays(*arrays) -> str:
    """Order-sensitive content hash of a list of arrays (dtype + shape + bytes)."""
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _planted_x(a1: int, a2: int, function: str) -> int:
    function = function.lower()
    if function == "xor":
        return int(bool(a1) ^ bool(a2))
    if function == "and":
        return int(bool(a1) and bool(a2))
    if function == "or":
        return int(bool(a1) or bool(a2))
    if function in {"a2", "independent"}:
        # Independent of A1 under the balanced generator.
        return int(a2)
    raise ValueError(
        f"Unknown planted_function={function!r}. "
        "Choose one of: 'xor', 'and', 'or', 'a2'."
    )


def _build_class_pools(targets: torch.Tensor) -> Dict[int, np.ndarray]:
    targets_np = np.asarray(targets)
    return {
        digit: np.flatnonzero(targets_np == digit).astype(np.int64)
        for digit in range(10)
    }


def _split_train_val_pools(
    targets: torch.Tensor,
    val_percent: float,
    seed: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Stratified split of original MNIST train images into disjoint pools."""
    rng = np.random.default_rng(seed)
    full = _build_class_pools(targets)

    train_pools = {}
    val_pools = {}

    for digit, idxs in full.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)

        n_val = int(round(len(idxs) * val_percent))
        n_val = max(1, min(n_val, len(idxs) - 1))

        val_pools[digit] = idxs[:n_val]
        train_pools[digit] = idxs[n_val:]

    return train_pools, val_pools


def _balanced_digit_pairs(n_samples: int, seed: int) -> np.ndarray:
    """
    Generate nearly/exactly uniform ordered digit pairs.

    For every complete block of 100 samples, every ordered pair (d1, d2)
    occurs exactly once. Thus if n_samples is divisible by 100, the pair
    distribution and both digit marginals are exactly uniform.
    """
    rng = np.random.default_rng(seed)
    all_pairs = np.array(
        [(d1, d2) for d1 in range(10) for d2 in range(10)],
        dtype=np.int64,
    )

    n_full = n_samples // 100
    remainder = n_samples % 100

    blocks = []
    for _ in range(n_full):
        block = all_pairs.copy()
        rng.shuffle(block)
        blocks.append(block)

    if remainder:
        block = all_pairs.copy()
        rng.shuffle(block)
        blocks.append(block[:remainder])

    pairs = np.concatenate(blocks, axis=0)
    rng.shuffle(pairs)
    return pairs


def _sample_source_indices(
    digit_pairs: np.ndarray,
    class_pools: Dict[int, np.ndarray],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose an actual MNIST image for each desired digit identity."""
    rng = np.random.default_rng(seed)
    idx1 = np.empty(len(digit_pairs), dtype=np.int64)
    idx2 = np.empty(len(digit_pairs), dtype=np.int64)

    for i, (d1, d2) in enumerate(digit_pairs):
        idx1[i] = rng.choice(class_pools[int(d1)])
        idx2[i] = rng.choice(class_pools[int(d2)])

    return idx1, idx2


def _apply_corruption(
    image: torch.Tensor,
    corruption: str,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Corrupt ONLY the first digit image. image shape: [1, 28, 28]."""
    corruption = corruption.lower()

    if corruption in {"none", "", "clean"} or strength <= 0:
        return image

    if corruption in {"gaussian", "gaussian_noise", "noise"}:
        noise = torch.randn(
            image.shape,
            generator=generator,
            dtype=image.dtype,
            device=image.device,
        )
        return torch.clamp(image + strength * noise, 0.0, 1.0)

    if corruption == "blur":
        sigma = max(0.1, 0.5 + 1.5 * float(strength))
        kernel = int(2 * round(2 * sigma) + 1)
        kernel = max(3, min(kernel, 13))
        if kernel % 2 == 0:
            kernel += 1
        return TF.gaussian_blur(
            image,
            kernel_size=[kernel, kernel],
            sigma=[sigma, sigma],
        )

    if corruption == "occlusion":
        frac = float(np.clip(strength, 0.0, 1.0))
        side = max(1, int(round(28 * frac)))

        max_top = 28 - side
        max_left = 28 - side

        top = (
            int(torch.randint(0, max_top + 1, (1,), generator=generator).item())
            if max_top > 0
            else 0
        )
        left = (
            int(torch.randint(0, max_left + 1, (1,), generator=generator).item())
            if max_left > 0
            else 0
        )

        out = image.clone()
        out[:, top:top + side, left:left + side] = 0.0
        return out

    raise ValueError(
        f"Unknown corruption={corruption!r}. "
        "Choose one of: 'none', 'gaussian', 'blur', 'occlusion'."
    )


class MNISTAddCovDataset(Dataset):
    """
    Controlled two-digit MNIST dataset.

    Main keys match the existing CUB dataset interface:
        img_code
        labels
        features
        concepts

    Extra keys are oracle-only metadata for post-hoc analysis:
        hidden_concepts -> [A2, X]
        A2
        X
        digit_labels    -> [d1, d2]
        is_corrupted
    """

    def __init__(
        self,
        mnist_dataset: MNIST,
        class_pools: Optional[Dict[int, np.ndarray]],
        dataset_size: int,
        seed: int,
        planted_function: str = "xor",
        corruption: str = "none",
        corruption_strength: float = 0.0,
        corruption_probability: float = 1.0,
        manifest: Optional[Dict[str, np.ndarray]] = None,
        removed_concepts: Optional[List[int]] = None,
    ):
        super().__init__()

        self.mnist_dataset = mnist_dataset
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.planted_function = planted_function.lower()

        self.corruption = corruption.lower()
        self.corruption_strength = float(corruption_strength)
        self.corruption_probability = float(corruption_probability)

        if not (0.0 <= self.corruption_probability <= 1.0):
            raise ValueError("corruption_probability must be in [0,1].")

        # Pixels of the corrupted first digit, when they were materialised to
        # disk alongside the manifest. They are the one part of the sample that
        # a seed alone does not pin across environments (torch RNG), so a saved
        # split stores them rather than regenerating them.
        self.corrupted_first_digit: Optional[np.ndarray] = None

        # Observed concept columns dropped from the bottleneck (§ incomplete
        # variants). The split itself is untouched -- same samples, same labels,
        # fewer supervised concepts -- so an incomplete run stays sample-aligned
        # with the complete one.
        self.removed_concept_idx = sorted(int(i) for i in (removed_concepts or []))
        self.kept_concept_idx = [
            i for i in range(len(OBSERVED_CONCEPT_NAMES))
            if i not in self.removed_concept_idx
        ]

        if manifest is not None:
            self._load_from_manifest(manifest)
            return

        # 1) Choose digit identities with an exactly balanced pair distribution.
        self.digit_pairs = _balanced_digit_pairs(self.dataset_size, seed=self.seed)

        # 2) Choose actual MNIST images conditional on those identities.
        self.idx1, self.idx2 = _sample_source_indices(
            self.digit_pairs,
            class_pools,
            seed=self.seed + 1,
        )

        # Decide once which samples are corrupted, making access deterministic.
        rng = np.random.default_rng(self.seed + 2)
        self.corruption_mask = (
            rng.random(self.dataset_size) < self.corruption_probability
        )

        # Precompute all symbolic labels.
        self.observed_concepts = np.zeros((self.dataset_size, 3), dtype=np.float32)
        self.hidden_concepts = np.zeros((self.dataset_size, 2), dtype=np.float32)
        self.task_labels = np.zeros(self.dataset_size, dtype=np.int64)

        for i, (d1, d2) in enumerate(self.digit_pairs):
            a1 = int(d1 % 2 == 1)
            a2 = int(d2 % 2 == 1)
            h1 = int(d1 >= 6)
            h2 = int(d2 >= 6)

            x_hidden = _planted_x(a1, a2, self.planted_function)
            m = int(h1 or h2)

            # Only these 3 values are supervised bottleneck concepts.
            self.observed_concepts[i] = np.array([a1, h1, h2], dtype=np.float32)

            # Retained only for evaluation/validation after training.
            self.hidden_concepts[i] = np.array([a2, x_hidden], dtype=np.float32)

            # 8-class target: three binary bits encoded as an integer.
            self.task_labels[i] = 4 * a1 + 2 * m + x_hidden

    def _load_from_manifest(self, manifest: Dict[str, np.ndarray]) -> None:
        """Adopt a split that was generated once and written to disk."""
        self.digit_pairs = manifest["digit_pairs"].astype(np.int64)
        self.idx1 = manifest["idx1"].astype(np.int64)
        self.idx2 = manifest["idx2"].astype(np.int64)
        self.corruption_mask = manifest["corruption_mask"].astype(bool)
        self.observed_concepts = manifest["observed_concepts"].astype(np.float32)
        self.hidden_concepts = manifest["hidden_concepts"].astype(np.float32)
        self.task_labels = manifest["task_labels"].astype(np.int64)

        if "corrupted_first_digit" in manifest:
            self.corrupted_first_digit = manifest["corrupted_first_digit"].astype(
                np.float32
            )

        self.dataset_size = len(self.digit_pairs)

        for name, arr in (
            ("idx1", self.idx1),
            ("idx2", self.idx2),
            ("corruption_mask", self.corruption_mask),
            ("observed_concepts", self.observed_concepts),
            ("hidden_concepts", self.hidden_concepts),
            ("task_labels", self.task_labels),
        ):
            if len(arr) != self.dataset_size:
                raise ValueError(
                    f"Corrupt split manifest: {name} has {len(arr)} rows, "
                    f"expected {self.dataset_size}."
                )

    def concept_names(self) -> List[str]:
        """Names of the concepts actually exposed in the bottleneck."""
        return [OBSERVED_CONCEPT_NAMES[i] for i in self.kept_concept_idx]

    def removed_concept_names(self) -> List[str]:
        """Names of the observed concepts withheld from the bottleneck."""
        return [OBSERVED_CONCEPT_NAMES[i] for i in self.removed_concept_idx]

    def fingerprint(self) -> str:
        """
        Content hash of this split.

        Covers everything that determines the data: the generator settings, the
        digit identities, *which* MNIST images back them, the derived concept and
        task labels, the corruption mask, and the actual pixels of a fixed probe
        subset (so a differing torch RNG for the corruption noise is caught too).

        Deliberately independent of which concepts are exposed in the
        bottleneck, so a complete run and an incomplete run over the same split
        print the same fingerprint.

        Two runs printing the same fingerprint hold literally the same samples in
        the same order, whatever machine they ran on.
        """
        spec = "|".join(
            str(v)
            for v in (
                self.dataset_size,
                self.seed,
                self.planted_function,
                self.corruption,
                self.corruption_strength,
                self.corruption_probability,
            )
        )

        n_probe = min(self.dataset_size, 16)
        probe_idx = np.unique(
            np.linspace(0, self.dataset_size - 1, n_probe).astype(np.int64)
        )
        probe_pixels = torch.stack(
            [self[int(i)]["features"] for i in probe_idx]
        ).numpy()

        return _fingerprint_arrays(
            np.frombuffer(spec.encode(), dtype=np.uint8),
            self.digit_pairs,
            self.idx1,
            self.idx2,
            self.corruption_mask,
            self.observed_concepts,
            self.hidden_concepts,
            self.task_labels,
            probe_pixels,
        )

    def __len__(self) -> int:
        return self.dataset_size

    def _load_mnist_tensor(self, source_index: int) -> torch.Tensor:
        img = self.mnist_dataset.data[int(source_index)].float() / 255.0
        return img.unsqueeze(0)  # [1, 28, 28]

    def __getitem__(self, index: int):
        d1, d2 = self.digit_pairs[index]

        img1 = self._load_mnist_tensor(self.idx1[index])
        img2 = self._load_mnist_tensor(self.idx2[index])

        is_corrupted = bool(
            self.corruption_mask[index]
            and self.corruption not in {"none", "", "clean"}
            and self.corruption_strength > 0
        )

        if is_corrupted:
            if self.corrupted_first_digit is not None:
                # Materialised split: read the pixels back rather than
                # re-drawing them from a torch RNG.
                img1 = torch.from_numpy(self.corrupted_first_digit[index]).float()
            else:
                # Deterministic corruption for each sample.
                g = torch.Generator()
                g.manual_seed(self.seed * 1_000_003 + int(index))
                img1 = _apply_corruption(
                    img1,
                    corruption=self.corruption,
                    strength=self.corruption_strength,
                    generator=g,
                )

        # Single-backbone input, the two digits as channels: [2, 28, 28].
        # Channel stacking rather than horizontal concatenation, to match
        # data.num_covariates=2 and IntCEMMNISTEncoder, whose first conv takes
        # num_covariates channels and whose projection assumes a 28x28 map.
        features = torch.cat([img1, img2], dim=0)

        all_observed = torch.from_numpy(self.observed_concepts[index]).float()
        concepts = all_observed[self.kept_concept_idx]
        removed = all_observed[self.removed_concept_idx]
        hidden = torch.from_numpy(self.hidden_concepts[index]).float()
        label = torch.tensor(self.task_labels[index], dtype=torch.long)

        return {
            "img_code": int(index),
            "labels": label,
            "features": features,
            "concepts": concepts,

            # Oracle-only analysis metadata; NOT part of `concepts`.
            "hidden_concepts": hidden,      # [A2, X]
            "removed_concepts": removed,    # observed concepts held out, if any
            "A2": hidden[0],
            "X": hidden[1],
            "digit_labels": torch.tensor([int(d1), int(d2)], dtype=torch.long),
            "is_corrupted": torch.tensor(is_corrupted, dtype=torch.bool),
        }


def get_MNIST_add_cov_datasets(
    config,
    incomplete: Optional[bool] = None,
    seed: int = 42,
    log_file: Optional[str] = None,
):
    """
    Return train/validation/test Dataset objects.

    This is intentionally compatible with your existing utils/data.py pattern:
    utils/data.py can wrap these returned datasets using its common DataLoader code.

    Optional config fields
    ----------------------
    data_path:                       default './data'
    train_dataset_size:              default 30000
    val_dataset_size:                default int(train_dataset_size * val_percent)
    test_dataset_size:               default 10000
    val_percent:                     default 0.2

    planted_function:                'xor' (default), 'and', 'or', 'a2'

    data_seed:                       default None -> use the run seed. Set it to
                                     pin the split independently of `config.seed`,
                                     so several training seeds share one dataset.

    corruption:                      'none', 'gaussian', 'blur', 'occlusion'
    corruption_strength:             default 0.0
    corruption_probability:          default 1.0

    test_corruption:                 defaults to corruption
    test_corruption_strength:        defaults to corruption_strength
    test_corruption_probability:     defaults to corruption_probability

    `incomplete` is accepted only to match the style of the CUB loader.
    This dataset is incomplete by construction because A2 and X are withheld
    from the `concepts` tensor.
    """
    # Optional: decouple the data split from the run seed. With data_seed set,
    # every run reuses the exact same train/val/test samples regardless of
    # config.seed; left at None the split follows the run seed as before.
    data_seed = _cfg_get(config, "data_seed", None)
    if data_seed is not None:
        seed = int(data_seed)

    data_path = _cfg_get(config, "data_path", "./data")
    root = os.path.join(data_path, "MNIST_ADD_COV")
    os.makedirs(root, exist_ok=True)

    train_dataset_size = int(_cfg_get(config, "train_dataset_size", 30000))
    val_percent = float(_cfg_get(config, "val_percent", 0.2))
    val_dataset_size = int(
        _cfg_get(config, "val_dataset_size", int(train_dataset_size * val_percent))
    )
    test_dataset_size = int(_cfg_get(config, "test_dataset_size", 10000))

    planted_function = str(_cfg_get(config, "planted_function", "xor"))
    removed_concepts = resolve_removed_concepts(
        _cfg_get(config, "removed_concepts", None)
    )

    corruption = str(_cfg_get(config, "corruption", "none"))
    corruption_strength = float(_cfg_get(config, "corruption_strength", 0.0))
    corruption_probability = float(_cfg_get(config, "corruption_probability", 1.0))

    test_corruption = str(_cfg_get(config, "test_corruption", corruption))
    test_corruption_strength = float(
        _cfg_get(config, "test_corruption_strength", corruption_strength)
    )
    test_corruption_probability = float(
        _cfg_get(config, "test_corruption_probability", corruption_probability)
    )

    mnist_train = MNIST(root=root, train=True, download=True)
    mnist_test = MNIST(root=root, train=False, download=True)

    # Underlying MNIST images used for train and val are disjoint.
    train_pools, val_pools = _split_train_val_pools(
        mnist_train.targets,
        val_percent=val_percent,
        seed=seed,
    )
    test_pools = _build_class_pools(mnist_test.targets)

    trainset = MNISTAddCovDataset(
        mnist_dataset=mnist_train,
        class_pools=train_pools,
        dataset_size=train_dataset_size,
        seed=seed + 10,
        planted_function=planted_function,
        removed_concepts=removed_concepts,
        corruption=corruption,
        corruption_strength=corruption_strength,
        corruption_probability=corruption_probability,
    )

    valset = MNISTAddCovDataset(
        mnist_dataset=mnist_train,
        class_pools=val_pools,
        dataset_size=val_dataset_size,
        seed=seed + 20,
        planted_function=planted_function,
        removed_concepts=removed_concepts,
        corruption=corruption,
        corruption_strength=corruption_strength,
        corruption_probability=corruption_probability,
    )

    testset = MNISTAddCovDataset(
        mnist_dataset=mnist_test,
        class_pools=test_pools,
        dataset_size=test_dataset_size,
        seed=seed + 30,
        planted_function=planted_function,
        removed_concepts=removed_concepts,
        corruption=test_corruption,
        corruption_strength=test_corruption_strength,
        corruption_probability=test_corruption_probability,
    )

    sync_num_concepts(config, trainset, log_file=log_file)

    log_split_fingerprints(
        {"train": trainset, "val": valset, "test": testset},
        seed=seed,
        planted_function=planted_function,
        log_file=log_file,
    )

    return trainset, valset, testset


SPLIT_ROOT = "splits"
MANIFEST_KEYS = (
    "digit_pairs",
    "idx1",
    "idx2",
    "corruption_mask",
    "observed_concepts",
    "hidden_concepts",
    "task_labels",
)

# Which underlying MNIST partition each split draws its images from.
SPLIT_MNIST_SOURCE = {"train": "train", "val": "train", "test": "test"}


def resolve_removed_concepts(removed) -> List[int]:
    """
    Turn a config entry into observed-concept column indices.

    Accepts indices (0..2), full names ("A1::digit1_is_odd") or short names
    ("A1"), so the config can read either way.
    """
    if removed is None:
        return []
    if isinstance(removed, (int, str)):
        removed = [removed]

    short_names = [name.split("::")[0] for name in OBSERVED_CONCEPT_NAMES]

    indices = []
    for entry in removed:
        if isinstance(entry, bool):
            raise ValueError(f"Invalid removed_concepts entry: {entry!r}")
        if isinstance(entry, int):
            index = entry
        elif entry in OBSERVED_CONCEPT_NAMES:
            index = OBSERVED_CONCEPT_NAMES.index(entry)
        elif entry in short_names:
            index = short_names.index(entry)
        else:
            raise ValueError(
                f"Unknown concept {entry!r}. Use an index in "
                f"[0, {len(OBSERVED_CONCEPT_NAMES) - 1}] or one of "
                f"{OBSERVED_CONCEPT_NAMES} / {short_names}."
            )
        if not 0 <= index < len(OBSERVED_CONCEPT_NAMES):
            raise ValueError(
                f"removed_concepts index {index} out of range for "
                f"{len(OBSERVED_CONCEPT_NAMES)} observed concepts."
            )
        indices.append(index)

    if len(set(indices)) == len(OBSERVED_CONCEPT_NAMES):
        raise ValueError("Cannot remove every observed concept.")

    return sorted(set(indices))


def sync_num_concepts(config, dataset: "MNISTAddCovDataset", log_file=None) -> None:
    """
    Keep `data.num_concepts` in step with the concepts actually exposed.

    Mirrors the CUB path, where removing attributes rewrites num_concepts from
    the data rather than trusting the config value.
    """
    num_concepts = len(dataset.kept_concept_idx)
    configured = _cfg_get(config, "num_concepts", num_concepts)

    if configured != num_concepts:
        message = (
            f"MNIST-Add-Cov: removing {dataset.removed_concept_names()} -> "
            f"num_concepts {configured} -> {num_concepts}"
        )
        print(message)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(message + "\n")

    try:
        config.num_concepts = num_concepts
    except Exception:  # plain config objects used by the smoke test
        pass


def _split_dir(config, data_dir_name: str, split: str) -> str:
    data_path = _cfg_get(config, "data_path", "./data")
    return os.path.join(
        data_path, "MNIST_ADD_COV", SPLIT_ROOT, data_dir_name, split
    )


def save_MNIST_add_cov_data(config, train, val, test, log_file=None) -> str:
    """
    Materialise the generated splits to disk, one folder per split.

    Writes the *manifest* of each split -- which MNIST images it uses, in what
    order, with which digit identities, concept and task labels -- rather than
    the images themselves, since the underlying MNIST files are already
    identical everywhere. The one exception is corrupted pixels, which depend on
    the torch RNG and are therefore stored outright.

    Generate once, copy the folder to the cluster (or point both at a shared
    path) and every run loads byte-identical train/val/test regardless of
    numpy/torch versions or the run seed.
    """
    data_path = _cfg_get(config, "data_path", "./data")
    root = os.path.join(data_path, "MNIST_ADD_COV", SPLIT_ROOT)
    os.makedirs(root, exist_ok=True)

    save_name = _cfg_get(config, "save_data_name", None) or (
        f"seed_{train.seed - 10}"
        f"_n_{len(train)}_{len(val)}_{len(test)}"
        f"_{train.planted_function}"
        f"_{train.corruption}_{train.corruption_strength}"
    )

    version = 1
    unique_name = save_name
    while os.path.exists(os.path.join(root, unique_name)):
        unique_name = f"{save_name}_v{version}"
        version += 1
    save_dir = os.path.join(root, unique_name)

    splits = {"train": train, "val": val, "test": test}
    for split, dataset in splits.items():
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        arrays = {key: getattr(dataset, key) for key in MANIFEST_KEYS}

        if dataset.corruption not in {"none", "", "clean"} and (
            dataset.corruption_strength > 0
        ):
            # Corrupted pixels are the only part not derivable from the MNIST
            # files plus the manifest, so store them explicitly.
            arrays["corrupted_first_digit"] = np.stack(
                [
                    dataset[i]["features"][0:1].numpy()
                    for i in range(len(dataset))
                ]
            ).astype(np.float32)

        np.savez_compressed(os.path.join(split_dir, "manifest.npz"), **arrays)

        meta = {
            "split": split,
            "mnist_source": SPLIT_MNIST_SOURCE[split],
            "dataset_size": len(dataset),
            "seed": dataset.seed,
            "planted_function": dataset.planted_function,
            "corruption": dataset.corruption,
            "corruption_strength": dataset.corruption_strength,
            "corruption_probability": dataset.corruption_probability,
            "fingerprint": dataset.fingerprint(),
        }
        with open(os.path.join(split_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    with open(os.path.join(save_dir, "info.txt"), "w") as f:
        f.write("MNIST-Add-Cov materialised split\n")
        f.write(f"generator seed: {train.seed - 10}\n")
        f.write(f"planted_function: {train.planted_function}\n")
        f.write(
            f"corruption: {train.corruption} "
            f"(strength={train.corruption_strength}, "
            f"p={train.corruption_probability})\n"
        )
        f.write(
            f"test corruption: {test.corruption} "
            f"(strength={test.corruption_strength}, "
            f"p={test.corruption_probability})\n"
        )
        f.write(f"sizes: train={len(train)}, val={len(val)}, test={len(test)}\n")
        f.write(f"observed concepts (full): {OBSERVED_CONCEPT_NAMES}\n")
        f.write(f"oracle variables: {ORACLE_NAMES}\n")
        f.write(
            "concept removal is applied at load time from data.removed_concepts, "
            "so incomplete runs reuse this exact split\n"
        )
        for split, dataset in splits.items():
            f.write(f"{split} fingerprint: {dataset.fingerprint()}\n")

    message = f"Saved MNIST-Add-Cov splits to {save_dir}"
    print(message)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(message + "\n")
            f.write(f"data_dir: {save_dir}\n")

    return save_dir


def load_saved_MNIST_add_cov_data(config, log_file=None):
    """
    Load the train/val/test folders written by `save_MNIST_add_cov_data`.

    The split is taken verbatim from disk -- no RNG is touched, so the run seed
    no longer influences which samples land where. `data.removed_concepts` is
    applied on top, which is what makes an incomplete run the *same* split with
    fewer supervised concepts.
    """
    data_dir_name = _cfg_get(config, "data_dir_name", None)
    if data_dir_name is None:
        raise ValueError(
            "load_saved_MNIST_add_cov_data requires data.data_dir_name."
        )

    data_path = _cfg_get(config, "data_path", "./data")
    mnist_root = os.path.join(data_path, "MNIST_ADD_COV")
    removed_concepts = resolve_removed_concepts(
        _cfg_get(config, "removed_concepts", None)
    )

    mnist_by_source = {
        "train": MNIST(root=mnist_root, train=True, download=True),
        "test": MNIST(root=mnist_root, train=False, download=True),
    }

    datasets = []
    for split in ("train", "val", "test"):
        split_dir = _split_dir(config, data_dir_name, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Missing split folder {split_dir}. Generate it once with "
                f"data.save_data=True, then point data.data_dir_name at it."
            )

        with open(os.path.join(split_dir, "meta.json")) as f:
            meta = json.load(f)
        with np.load(os.path.join(split_dir, "manifest.npz")) as npz:
            manifest = {key: npz[key] for key in npz.files}

        dataset = MNISTAddCovDataset(
            mnist_dataset=mnist_by_source[meta["mnist_source"]],
            class_pools=None,
            dataset_size=meta["dataset_size"],
            seed=meta["seed"],
            planted_function=meta["planted_function"],
            corruption=meta["corruption"],
            corruption_strength=meta["corruption_strength"],
            corruption_probability=meta["corruption_probability"],
            manifest=manifest,
            removed_concepts=removed_concepts,
        )
        datasets.append(dataset)

    trainset, valset, testset = datasets
    sync_num_concepts(config, trainset, log_file=log_file)

    message = f"Loaded MNIST-Add-Cov splits from {data_dir_name}"
    if removed_concepts:
        message += f" (removed concepts: {trainset.removed_concept_names()})"
    print(message)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(message + "\n")

    log_split_fingerprints(
        {"train": trainset, "val": valset, "test": testset},
        seed=trainset.seed - 10,
        planted_function=trainset.planted_function,
        log_file=log_file,
    )

    return trainset, valset, testset


def log_split_fingerprints(
    splits: Dict[str, "MNISTAddCovDataset"],
    seed: int,
    planted_function: str,
    log_file: Optional[str] = None,
) -> Dict[str, str]:
    """
    Print (and optionally log) a content hash per split.

    This is the explicit check that two runs — local and cluster — are training
    on the same data: compare the printed fingerprints, they match if and only
    if the samples match.
    """
    fingerprints = {name: ds.fingerprint() for name, ds in splits.items()}

    lines = [
        f"MNIST-Add-Cov split fingerprints "
        f"(data seed={seed}, planted_function={planted_function}):"
    ]
    lines += [
        f"  {name:<5} n={len(splits[name]):<6} sha256[:16]={fp}"
        for name, fp in fingerprints.items()
    ]
    reference = next(iter(splits.values()))
    lines.append(f"  concepts exposed: {reference.concept_names()}")
    if reference.removed_concept_idx:
        lines.append(f"  concepts removed: {reference.removed_concept_names()}")
    lines.append(
        "  Matching fingerprints => identical train/val/test samples across "
        "machines (independent of which concepts are exposed)."
    )

    message = "\n".join(lines)
    print(message)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(message + "\n")

    return fingerprints


def get_mnist_add_cov_concept_names() -> List[str]:
    return OBSERVED_CONCEPT_NAMES.copy()


def get_mnist_add_cov_oracle_names() -> List[str]:
    return ORACLE_NAMES.copy()


def summarize_dataset(dataset: MNISTAddCovDataset) -> None:
    """Diagnostic checks for balance and leakage."""
    concepts = dataset.observed_concepts
    hidden = dataset.hidden_concepts
    labels = dataset.task_labels

    print(f"N={len(dataset)}")
    print(f"Observed concept means [A1,H1,H2]: {concepts.mean(axis=0)}")
    print(f"Oracle means [A2,X]: {hidden.mean(axis=0)}")
    print(f"Task class counts: {np.bincount(labels, minlength=8)}")

    a2 = hidden[:, 0]
    h2 = concepts[:, 2]
    for h2_value in [0, 1]:
        mask = h2 == h2_value
        print(
            f"P(A2=1 | H2={h2_value}) = "
            f"{a2[mask].mean():.4f}  (target 0.5)"
        )


if __name__ == "__main__":
    # Small standalone smoke test.
    class Config:
        data_path = "./data"
        train_dataset_size = 1000
        val_dataset_size = 200
        test_dataset_size = 1000
        val_percent = 0.2

        planted_function = "xor"

        corruption = "gaussian"
        corruption_strength = 0.25
        corruption_probability = 1.0

    trainset, valset, testset = get_MNIST_add_cov_datasets(Config(), seed=42)
    summarize_dataset(trainset)

    sample = trainset[0]
    print("features:", sample["features"].shape)
    print("concepts:", sample["concepts"])
    print("hidden_concepts [A2,X]:", sample["hidden_concepts"])
    print("label:", sample["labels"])
