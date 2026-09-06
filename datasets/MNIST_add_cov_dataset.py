"""
Controlled two-digit MNIST dataset for residual/concept covariance experiments.

Input:
    Two MNIST digit images concatenated horizontally -> tensor [1, 28, 56].

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
        class_pools: Dict[int, np.ndarray],
        dataset_size: int,
        seed: int,
        planted_function: str = "xor",
        corruption: str = "none",
        corruption_strength: float = 0.0,
        corruption_probability: float = 1.0,
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
            # Deterministic corruption for each sample.
            g = torch.Generator()
            g.manual_seed(self.seed * 1_000_003 + int(index))
            img1 = _apply_corruption(
                img1,
                corruption=self.corruption,
                strength=self.corruption_strength,
                generator=g,
            )

        # Single-backbone input: [1, 28, 56].
        features = torch.cat([img1, img2], dim=2)

        concepts = torch.from_numpy(self.observed_concepts[index]).float()
        hidden = torch.from_numpy(self.hidden_concepts[index]).float()
        label = torch.tensor(self.task_labels[index], dtype=torch.long)

        return {
            "img_code": int(index),
            "labels": label,
            "features": features,
            "concepts": concepts,

            # Oracle-only analysis metadata; NOT part of `concepts`.
            "hidden_concepts": hidden,      # [A2, X]
            "A2": hidden[0],
            "X": hidden[1],
            "digit_labels": torch.tensor([int(d1), int(d2)], dtype=torch.long),
            "is_corrupted": torch.tensor(is_corrupted, dtype=torch.bool),
        }


def get_MNIST_add_cov_datasets(
    config,
    incomplete: Optional[bool] = None,
    seed: int = 42,
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
        corruption=test_corruption,
        corruption_strength=test_corruption_strength,
        corruption_probability=test_corruption_probability,
    )

    return trainset, valset, testset


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
