"""
MNIST addition experiment with exact digit identity as the missing information.

Purpose
-------
Each sample contains two MNIST digits d1 and d2, stacked as channels:
    features.shape == [2, 28, 28]

The supervised concept bottleneck contains three interpretable binary concepts
for EACH digit:
    high(d)  = 1[d >= threshold]      (threshold defaults to 5)
    prime(d) = 1[d in {2, 3, 5, 7}]
    even(d)  = 1[d is even]

So the exposed concept vector is:
    [d1_high, d1_prime, d1_even, d2_high, d2_prime, d2_even]

The exact digit identities are deliberately NOT supervised concepts. They are
kept only as oracle metadata for post-hoc analysis.

The task is standard two-digit MNIST addition:
    y = d1 + d2                      # 19 classes: 0,...,18

Why this is concept-incomplete
------------------------------
For threshold=5 the observed concepts do not uniquely identify all digits:
    {0,4}, {5,7}, {6,8}
share the same concept code within each pair. Therefore the concept bottleneck
alone cannot always recover the sum. With binary residual variables, two
residual bits are the minimum worst-case capacity needed to resolve every sum
conditional on the exposed concepts (some concept-pair cells admit four sums).

The intended covariance analysis groups concepts by digit:
    digit 1 group = concept indices [0,1,2]
    digit 2 group = concept indices [3,4,5]
Then test whether each residual dimension's concept-residual covariance is
preferentially associated with one of these interpretable groups.

Compatibility
-------------
The main functions intentionally keep the old MNIST-Add-Cov names so this file
can be wired into the same utils/data.py path with minimal changes:
    get_MNIST_add_cov_datasets
    save_MNIST_add_cov_data
    load_saved_MNIST_add_cov_data

This file contains only the exact-digit-sum experiment; there is no experiment
selector or planted hidden variable.
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


PRIMES = {2, 3, 5, 7}
NUM_DIGITS = 2
NUM_CLASSES = 19
SPLIT_ROOT = "splits"
DATA_ROOT_NAME = "MNIST_ADD_EXACT_DIGIT"

# These are fixed semantic groupings for the covariance analysis.
CONCEPT_GROUPS = {
    "digit1": [0, 1, 2],
    "digit2": [3, 4, 5],
}
ORACLE_NAMES = ["digit1_exact", "digit2_exact"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg_get(config, name: str, default):
    """Read a field from dicts, OmegaConf/DictConfig, or simple objects."""
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


def _concept_names(threshold: int) -> List[str]:
    return [
        f"D1_HIGH::digit1_ge_{threshold}",
        "D1_PRIME::digit1_is_prime",
        "D1_EVEN::digit1_is_even",
        f"D2_HIGH::digit2_ge_{threshold}",
        "D2_PRIME::digit2_is_prime",
        "D2_EVEN::digit2_is_even",
    ]


def get_mnist_add_cov_concept_names(
    experiment: str = "exact_digit_sum", threshold: int = 5
) -> List[str]:
    """Back-compatible helper used by some analysis code."""
    del experiment
    return _concept_names(int(threshold))


def get_mnist_add_cov_oracle_names(
    experiment: str = "exact_digit_sum",
) -> List[str]:
    del experiment
    return list(ORACLE_NAMES)


def get_concept_groups() -> Dict[str, List[int]]:
    """Return concept indices belonging to digit 1 and digit 2."""
    return {name: list(indices) for name, indices in CONCEPT_GROUPS.items()}


# ---------------------------------------------------------------------------
# Symbolic task definition
# ---------------------------------------------------------------------------

def _digit_concepts(digit: int, threshold: int) -> List[int]:
    digit = int(digit)
    return [
        int(digit >= threshold),
        int(digit in PRIMES),
        int(digit % 2 == 0),
    ]


def _labels_for_digits(d1: int, d2: int, threshold: int):
    concepts = _digit_concepts(d1, threshold) + _digit_concepts(d2, threshold)
    exact_digits = [int(d1), int(d2)]
    task = int(d1) + int(d2)
    return concepts, exact_digits, task


# ---------------------------------------------------------------------------
# MNIST sampling helpers
# ---------------------------------------------------------------------------

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
    """Create disjoint, digit-stratified MNIST image pools for train and val."""
    rng = np.random.default_rng(seed)
    full = _build_class_pools(targets)

    train_pools: Dict[int, np.ndarray] = {}
    val_pools: Dict[int, np.ndarray] = {}

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

    Every full block of 100 contains each ordered pair (d1,d2) exactly once.
    The default train/val/test sizes are all divisible by 100, so the pair
    distribution is exactly uniform in each split.
    """
    rng = np.random.default_rng(seed)
    all_pairs = np.asarray(
        [(d1, d2) for d1 in range(10) for d2 in range(10)],
        dtype=np.int64,
    )

    blocks = []
    n_full, remainder = divmod(int(n_samples), 100)
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
) -> np.ndarray:
    """Choose an actual MNIST image for each desired digit identity."""
    rng = np.random.default_rng(seed)
    out = np.empty((len(digit_pairs), 2), dtype=np.int64)

    for i, (d1, d2) in enumerate(digit_pairs):
        out[i, 0] = rng.choice(class_pools[int(d1)])
        out[i, 1] = rng.choice(class_pools[int(d2)])

    return out


# ---------------------------------------------------------------------------
# Optional image corruption
# ---------------------------------------------------------------------------

def _resolve_corruption_channels(channels) -> Tuple[int, ...]:
    if channels is None:
        return (0, 1)  # treat the two semantic groups symmetrically

    if isinstance(channels, str):
        key = channels.strip().lower()
        if key == "all":
            return (0, 1)
        if key == "first":
            return (0,)
        if key == "second":
            return (1,)
        raise ValueError(
            f"Unknown corruption_channels={channels!r}. "
            "Use 'all', 'first', 'second', an index, or a list of indices."
        )

    if isinstance(channels, int):
        channels = [channels]

    resolved = tuple(sorted({int(c) for c in channels}))
    for c in resolved:
        if c not in (0, 1):
            raise ValueError(f"corruption channel {c} is invalid for two digits.")
    return resolved


def _apply_corruption(
    image: torch.Tensor,
    corruption: str,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    corruption = str(corruption).lower()

    if corruption in {"none", "", "clean"} or strength <= 0:
        return image

    if corruption in {"gaussian", "gaussian_noise", "noise"}:
        noise = torch.randn(
            image.shape,
            generator=generator,
            dtype=image.dtype,
            device=image.device,
        )
        return torch.clamp(image + float(strength) * noise, 0.0, 1.0)

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
        out[:, top : top + side, left : left + side] = 0.0
        return out

    raise ValueError(
        f"Unknown corruption={corruption!r}. "
        "Choose one of: none, gaussian, blur, occlusion."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MNISTAddExactDigitDataset(Dataset):
    """Two-digit MNIST addition with exact digit identity withheld."""

    def __init__(
        self,
        mnist_dataset: MNIST,
        class_pools: Optional[Dict[int, np.ndarray]],
        dataset_size: int,
        seed: int,
        digit_threshold: int = 5,
        corruption: str = "none",
        corruption_strength: float = 0.0,
        corruption_probability: float = 1.0,
        corruption_channels=None,
        manifest: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__()

        self.mnist_dataset = mnist_dataset
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.digit_threshold = int(digit_threshold)
        if not 0 <= self.digit_threshold <= 9:
            raise ValueError("digit_threshold must be between 0 and 9.")

        self.experiment = "exact_digit_sum"
        self.num_digits = NUM_DIGITS
        self.required_digits = NUM_DIGITS
        self.num_classes = NUM_CLASSES
        self.observed_concept_names = _concept_names(self.digit_threshold)
        self.oracle_concept_names = list(ORACLE_NAMES)
        self.kept_concept_idx = list(range(6))
        self.removed_concept_idx: List[int] = []

        self.corruption = str(corruption).lower()
        self.corruption_strength = float(corruption_strength)
        self.corruption_probability = float(corruption_probability)
        if not 0.0 <= self.corruption_probability <= 1.0:
            raise ValueError("corruption_probability must be in [0,1].")
        self.corruption_channels = _resolve_corruption_channels(corruption_channels)

        # If a split was materialised with corruption, these pixels are loaded
        # directly so local/cluster runs are byte-identical.
        self.corrupted_digits: Optional[np.ndarray] = None

        if manifest is not None:
            self._load_from_manifest(manifest)
            return

        if class_pools is None:
            raise ValueError("class_pools is required when generating a new split.")

        self.digit_pairs = _balanced_digit_pairs(self.dataset_size, self.seed)
        self.source_indices = _sample_source_indices(
            self.digit_pairs,
            class_pools,
            seed=self.seed + 1,
        )

        rng = np.random.default_rng(self.seed + 2)
        self.corruption_mask = (
            rng.random(self.dataset_size) < self.corruption_probability
        )

        self.observed_concepts = np.zeros((self.dataset_size, 6), dtype=np.float32)
        self.exact_digits = np.zeros((self.dataset_size, 2), dtype=np.int64)
        self.task_labels = np.zeros(self.dataset_size, dtype=np.int64)

        for i, (d1, d2) in enumerate(self.digit_pairs):
            concepts, exact_digits, task = _labels_for_digits(
                int(d1), int(d2), self.digit_threshold
            )
            self.observed_concepts[i] = np.asarray(concepts, dtype=np.float32)
            self.exact_digits[i] = np.asarray(exact_digits, dtype=np.int64)
            self.task_labels[i] = int(task)

    def _load_from_manifest(self, manifest: Dict[str, np.ndarray]) -> None:
        self.digit_pairs = np.asarray(manifest["digit_pairs"], dtype=np.int64)
        self.source_indices = np.asarray(manifest["source_indices"], dtype=np.int64)
        self.corruption_mask = np.asarray(manifest["corruption_mask"], dtype=bool)
        self.observed_concepts = np.asarray(
            manifest["observed_concepts"], dtype=np.float32
        )
        self.exact_digits = np.asarray(manifest["exact_digits"], dtype=np.int64)
        self.task_labels = np.asarray(manifest["task_labels"], dtype=np.int64)

        if "corrupted_digits" in manifest:
            self.corrupted_digits = np.asarray(
                manifest["corrupted_digits"], dtype=np.float32
            )

        self.dataset_size = len(self.digit_pairs)

        expected_shapes = {
            "digit_pairs": (self.dataset_size, 2),
            "source_indices": (self.dataset_size, 2),
            "observed_concepts": (self.dataset_size, 6),
            "exact_digits": (self.dataset_size, 2),
        }
        arrays = {
            "digit_pairs": self.digit_pairs,
            "source_indices": self.source_indices,
            "observed_concepts": self.observed_concepts,
            "exact_digits": self.exact_digits,
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(
                    f"Corrupt split manifest: {name}.shape={arrays[name].shape}, "
                    f"expected {shape}."
                )

    def concept_names(self) -> List[str]:
        return list(self.observed_concept_names)

    def removed_concept_names(self) -> List[str]:
        return []

    def _load_mnist_tensor(self, source_index: int) -> torch.Tensor:
        img = self.mnist_dataset.data[int(source_index)].float() / 255.0
        return img.unsqueeze(0)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        images = [
            self._load_mnist_tensor(self.source_indices[index, channel])
            for channel in range(2)
        ]

        is_corrupted = bool(
            self.corruption_mask[index]
            and self.corruption not in {"none", "", "clean"}
            and self.corruption_strength > 0
        )

        if is_corrupted:
            for slot, channel in enumerate(self.corruption_channels):
                if self.corrupted_digits is not None:
                    images[channel] = torch.from_numpy(
                        self.corrupted_digits[index, slot]
                    ).float().unsqueeze(0)
                else:
                    g = torch.Generator()
                    g.manual_seed(
                        self.seed * 1_000_003 + int(index) + channel * 7_000_003
                    )
                    images[channel] = _apply_corruption(
                        images[channel],
                        corruption=self.corruption,
                        strength=self.corruption_strength,
                        generator=g,
                    )

        features = torch.cat(images, dim=0)  # [2, 28, 28]
        concepts = torch.from_numpy(self.observed_concepts[index]).float()
        exact_digits = torch.from_numpy(self.exact_digits[index]).long()
        label = torch.tensor(self.task_labels[index], dtype=torch.long)

        return {
            "img_code": int(index),
            "labels": label,
            "features": features,
            "concepts": concepts,

            # Oracle-only metadata. The model should NOT consume these fields.
            "hidden_concepts": exact_digits.float(),
            "exact_digits": exact_digits,
            "digit_labels": exact_digits,
            "d1": exact_digits[0],
            "d2": exact_digits[1],
            "is_corrupted": torch.tensor(is_corrupted, dtype=torch.bool),
        }

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        meta = (
            self.dataset_size,
            self.seed,
            self.digit_threshold,
            self.corruption,
            self.corruption_strength,
            self.corruption_probability,
            self.corruption_channels,
        )
        h.update(repr(meta).encode())
        for arr in (
            self.digit_pairs,
            self.source_indices,
            self.corruption_mask,
            self.observed_concepts,
            self.exact_digits,
            self.task_labels,
        ):
            arr = np.ascontiguousarray(arr)
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        return h.hexdigest()[:16]


# Old class name as an alias, so code importing it does not have to change.
MNISTAddCovDataset = MNISTAddExactDigitDataset


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _sync_config(config, dataset: MNISTAddExactDigitDataset, log_file=None) -> None:
    desired = {
        "num_concepts": 6,
        "num_classes": NUM_CLASSES,
        "num_covariates": NUM_DIGITS,
    }

    messages = []
    for key, value in desired.items():
        current = _cfg_get(config, key, None)
        if current is not None and int(current) != value:
            messages.append(f"MNIST exact-digit sum: {key} {current} -> {value}")
        try:
            setattr(config, key, value)
        except Exception:
            if isinstance(config, dict):
                config[key] = value

    if messages:
        text = "\n".join(messages)
        print(text)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(text + "\n")


def get_MNIST_add_cov_datasets(
    config,
    incomplete: Optional[bool] = None,
    seed: int = 42,
    log_file: Optional[str] = None,
):
    """
    Return train/validation/test Dataset objects for the exact-digit-sum task.

    `incomplete` is accepted only for interface compatibility. This experiment
    is already concept-incomplete by construction: exact digit identity is never
    included in the supervised concept tensor.
    """
    del incomplete

    removed_concepts = _cfg_get(config, "removed_concepts", [])
    if removed_concepts not in (None, [], ()):
        raise ValueError(
            "This dedicated experiment keeps all six coarse concepts exposed. "
            "Set data.removed_concepts: []."
        )

    configured_covariates = int(_cfg_get(config, "num_covariates", NUM_DIGITS))
    if configured_covariates != NUM_DIGITS:
        raise ValueError(
            f"This experiment requires num_covariates=2, got {configured_covariates}."
        )

    data_seed = _cfg_get(config, "data_seed", None)
    if data_seed is not None:
        seed = int(data_seed)

    digit_threshold = int(_cfg_get(config, "digit_threshold", 5))
    data_path = _cfg_get(config, "data_path", "./data")
    root = os.path.join(data_path, DATA_ROOT_NAME)
    os.makedirs(root, exist_ok=True)

    train_size = int(_cfg_get(config, "train_dataset_size", 12000))
    val_percent = float(_cfg_get(config, "val_percent", 0.2))
    val_size = int(_cfg_get(config, "val_dataset_size", 2400))
    test_size = int(_cfg_get(config, "test_dataset_size", 10000))

    corruption = str(_cfg_get(config, "corruption", "none"))
    corruption_strength = float(_cfg_get(config, "corruption_strength", 0.0))
    corruption_probability = float(_cfg_get(config, "corruption_probability", 1.0))
    corruption_channels = _cfg_get(config, "corruption_channels", "all")

    test_corruption = str(_cfg_get(config, "test_corruption", corruption))
    test_corruption_strength = float(
        _cfg_get(config, "test_corruption_strength", corruption_strength)
    )
    test_corruption_probability = float(
        _cfg_get(config, "test_corruption_probability", corruption_probability)
    )

    mnist_train = MNIST(root=root, train=True, download=True)
    mnist_test = MNIST(root=root, train=False, download=True)

    train_pools, val_pools = _split_train_val_pools(
        mnist_train.targets,
        val_percent=val_percent,
        seed=seed,
    )
    test_pools = _build_class_pools(mnist_test.targets)

    common = dict(
        digit_threshold=digit_threshold,
        corruption_channels=corruption_channels,
    )

    trainset = MNISTAddExactDigitDataset(
        mnist_dataset=mnist_train,
        class_pools=train_pools,
        dataset_size=train_size,
        seed=seed + 10,
        corruption=corruption,
        corruption_strength=corruption_strength,
        corruption_probability=corruption_probability,
        **common,
    )
    valset = MNISTAddExactDigitDataset(
        mnist_dataset=mnist_train,
        class_pools=val_pools,
        dataset_size=val_size,
        seed=seed + 20,
        corruption=corruption,
        corruption_strength=corruption_strength,
        corruption_probability=corruption_probability,
        **common,
    )
    testset = MNISTAddExactDigitDataset(
        mnist_dataset=mnist_test,
        class_pools=test_pools,
        dataset_size=test_size,
        seed=seed + 30,
        corruption=test_corruption,
        corruption_strength=test_corruption_strength,
        corruption_probability=test_corruption_probability,
        **common,
    )

    _sync_config(config, trainset, log_file=log_file)
    log_split_fingerprints(
        {"train": trainset, "val": valset, "test": testset},
        seed=seed,
        log_file=log_file,
    )
    return trainset, valset, testset


# ---------------------------------------------------------------------------
# Materialised splits
# ---------------------------------------------------------------------------

MANIFEST_KEYS = (
    "digit_pairs",
    "source_indices",
    "corruption_mask",
    "observed_concepts",
    "exact_digits",
    "task_labels",
)
SPLIT_MNIST_SOURCE = {"train": "train", "val": "train", "test": "test"}


def _split_dir(config, data_dir_name: str, split: str) -> str:
    data_path = _cfg_get(config, "data_path", "./data")
    return os.path.join(data_path, DATA_ROOT_NAME, SPLIT_ROOT, data_dir_name, split)


def save_MNIST_add_cov_data(config, train, val, test, log_file=None) -> str:
    """Materialise the exact train/val/test manifests for reproducible runs."""
    data_path = _cfg_get(config, "data_path", "./data")
    root = os.path.join(data_path, DATA_ROOT_NAME, SPLIT_ROOT)
    os.makedirs(root, exist_ok=True)

    save_name = _cfg_get(config, "save_data_name", None) or (
        f"seed_{train.seed - 10}"
        f"_thr_{train.digit_threshold}"
        f"_n_{len(train)}_{len(val)}_{len(test)}"
        f"_{train.corruption}_{train.corruption_strength}"
    )

    unique_name = save_name
    version = 1
    while os.path.exists(os.path.join(root, unique_name)):
        unique_name = f"{save_name}_v{version}"
        version += 1
    save_dir = os.path.join(root, unique_name)

    for split, dataset in {"train": train, "val": val, "test": test}.items():
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        arrays = {key: getattr(dataset, key) for key in MANIFEST_KEYS}
        if dataset.corruption not in {"none", "", "clean"} and dataset.corruption_strength > 0:
            channels = list(dataset.corruption_channels)
            arrays["corrupted_digits"] = np.stack(
                [
                    dataset[i]["features"][channels].numpy()
                    for i in range(len(dataset))
                ]
            ).astype(np.float32)

        np.savez_compressed(os.path.join(split_dir, "manifest.npz"), **arrays)

        meta = {
            "split": split,
            "mnist_source": SPLIT_MNIST_SOURCE[split],
            "dataset_size": len(dataset),
            "seed": dataset.seed,
            "experiment": dataset.experiment,
            "digit_threshold": dataset.digit_threshold,
            "corruption_channels": list(dataset.corruption_channels),
            "corruption": dataset.corruption,
            "corruption_strength": dataset.corruption_strength,
            "corruption_probability": dataset.corruption_probability,
            "fingerprint": dataset.fingerprint(),
        }
        with open(os.path.join(split_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    with open(os.path.join(save_dir, "info.txt"), "w") as f:
        f.write("MNIST exact-digit addition split\n")
        f.write(f"digit threshold: >= {train.digit_threshold}\n")
        f.write(f"concepts: {train.concept_names()}\n")
        f.write("target: y = d1 + d2 (19 classes)\n")
        f.write("oracle only: exact digit identities [d1,d2]\n")
        f.write(f"concept groups: {CONCEPT_GROUPS}\n")
        f.write(f"train fingerprint: {train.fingerprint()}\n")
        f.write(f"val fingerprint: {val.fingerprint()}\n")
        f.write(f"test fingerprint: {test.fingerprint()}\n")

    message = f"Saved MNIST exact-digit splits to {save_dir}"
    print(message)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(message + "\n")
    return save_dir


def load_saved_MNIST_add_cov_data(config, log_file=None):
    """Load splits created by save_MNIST_add_cov_data."""
    data_dir_name = _cfg_get(config, "data_dir_name", None)
    if data_dir_name is None:
        raise ValueError("data.data_dir_name must be set to load a saved split.")

    removed_concepts = _cfg_get(config, "removed_concepts", [])
    if removed_concepts not in (None, [], ()):
        raise ValueError("This experiment requires removed_concepts: [].")

    requested_threshold = int(_cfg_get(config, "digit_threshold", 5))
    data_path = _cfg_get(config, "data_path", "./data")
    mnist_root = os.path.join(data_path, DATA_ROOT_NAME)
    mnist_by_source = {
        "train": MNIST(root=mnist_root, train=True, download=True),
        "test": MNIST(root=mnist_root, train=False, download=True),
    }

    datasets = []
    for split in ("train", "val", "test"):
        split_dir = _split_dir(config, data_dir_name, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        with open(os.path.join(split_dir, "meta.json")) as f:
            meta = json.load(f)
        with np.load(os.path.join(split_dir, "manifest.npz")) as npz:
            manifest = {key: npz[key] for key in npz.files}

        saved_threshold = int(meta["digit_threshold"])
        if saved_threshold != requested_threshold:
            raise ValueError(
                f"Saved split uses digit_threshold={saved_threshold}, but config "
                f"requests {requested_threshold}."
            )

        dataset = MNISTAddExactDigitDataset(
            mnist_dataset=mnist_by_source[meta["mnist_source"]],
            class_pools=None,
            dataset_size=int(meta["dataset_size"]),
            seed=int(meta["seed"]),
            digit_threshold=saved_threshold,
            corruption=meta["corruption"],
            corruption_strength=float(meta["corruption_strength"]),
            corruption_probability=float(meta["corruption_probability"]),
            corruption_channels=meta.get("corruption_channels", "all"),
            manifest=manifest,
        )
        datasets.append(dataset)

    trainset, valset, testset = datasets
    _sync_config(config, trainset, log_file=log_file)
    log_split_fingerprints(
        {"train": trainset, "val": valset, "test": testset},
        seed=trainset.seed - 10,
        log_file=log_file,
    )
    return trainset, valset, testset


def log_split_fingerprints(
    splits: Dict[str, MNISTAddExactDigitDataset],
    seed: int,
    log_file: Optional[str] = None,
) -> Dict[str, str]:
    fingerprints = {name: ds.fingerprint() for name, ds in splits.items()}
    reference = next(iter(splits.values()))
    lines = [
        f"MNIST exact-digit sum fingerprints (data seed={seed}, "
        f"threshold={reference.digit_threshold}):"
    ]
    lines += [
        f"  {name:<5} n={len(splits[name]):<6} sha256[:16]={fp}"
        for name, fp in fingerprints.items()
    ]
    lines.append(f"  concepts: {reference.concept_names()}")
    lines.append(f"  groups: {CONCEPT_GROUPS}")
    message = "\n".join(lines)
    print(message)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(message + "\n")
    return fingerprints


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def summarize_dataset(dataset: MNISTAddExactDigitDataset) -> None:
    concepts = dataset.observed_concepts
    digits = dataset.exact_digits
    labels = dataset.task_labels

    print(f"N={len(dataset)}")
    print(f"threshold: >= {dataset.digit_threshold}")
    print(f"concepts: {dataset.concept_names()}")
    print(f"concept means: {concepts.mean(axis=0)}")
    print(f"task class counts: {np.bincount(labels, minlength=NUM_CLASSES)}")

    # Show which exact digits remain ambiguous under the supervised concept code.
    print("Digit concept-code equivalence classes:")
    codes: Dict[Tuple[int, int, int], List[int]] = {}
    for d in range(10):
        code = tuple(_digit_concepts(d, dataset.digit_threshold))
        codes.setdefault(code, []).append(d)
    for code, members in sorted(codes.items()):
        suffix = "  <- ambiguous" if len(members) > 1 else ""
        print(f"  {code}: {members}{suffix}")

    # Worst-case number of task labels available after observing all six concepts.
    max_sums = 0
    max_examples = []
    for code1, d1s in codes.items():
        for code2, d2s in codes.items():
            sums = sorted({d1 + d2 for d1 in d1s for d2 in d2s})
            if len(sums) > max_sums:
                max_sums = len(sums)
                max_examples = [(code1, d1s, code2, d2s, sums)]
            elif len(sums) == max_sums:
                max_examples.append((code1, d1s, code2, d2s, sums))

    bits_needed = int(np.ceil(np.log2(max_sums))) if max_sums > 1 else 0
    print(f"max sums within one observed-concept cell: {max_sums}")
    print(f"minimum binary residual bits in the worst case: {bits_needed}")
    if max_examples:
        code1, d1s, code2, d2s, sums = max_examples[0]
        print(
            "example hardest cell: "
            f"digit1 {d1s} code={code1}, digit2 {d2s} code={code2} -> sums {sums}"
        )

    # The generated exact digits should agree with the digit pairs by construction.
    assert np.array_equal(dataset.digit_pairs, digits)


if __name__ == "__main__":
    # Symbolic smoke test that does not need the training pipeline.
    threshold = 5
    print("Concept codes for threshold >= 5:")
    for digit in range(10):
        print(digit, _digit_concepts(digit, threshold))

    # Verify the key design claim: threshold 5 leaves ambiguity and two binary
    # residual bits are sufficient/necessary in the hardest concept cell.
    codes: Dict[Tuple[int, int, int], List[int]] = {}
    for digit in range(10):
        codes.setdefault(tuple(_digit_concepts(digit, threshold)), []).append(digit)
    assert any(len(v) > 1 for v in codes.values())

    max_sums = max(
        len({d1 + d2 for d1 in ds1 for d2 in ds2})
        for ds1 in codes.values()
        for ds2 in codes.values()
    )
    assert max_sums == 4
    assert int(np.ceil(np.log2(max_sums))) == 2
    print("Symbolic checks passed: hardest concept cell has 4 sums -> 2 residual bits.")
