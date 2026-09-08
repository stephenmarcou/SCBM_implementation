"""
MNIST-Planted-Cov: does the learned concept-residual covariance identify what
the residual encoded?

One experiment family, three variants, one strength knob.

    input       five MNIST digits stacked as channels -> [5, 28, 28]

    concepts    A  = 1[d1 >= 5]        supervised
    (observed)  B  = 1[d2 >= 5]        supervised
                D1 = 1[d3 >= 5]        supervised, distractor
                D2 = 1[d4 >= 5]        supervised, distractor

    hidden      X  = 1[d5 >= 5]        NEVER supervised

    target      y  = 4*A + 2*B + X     8 classes

d1..d4 are drawn independently and exactly balanced, so A, B, D1 and D2 are
mutually uncorrelated. X is then drawn conditional on a chosen pair of those
concepts, and d5 is drawn uniformly from the digits consistent with X. So the
dependence between X and the concepts is planted by the generator, exactly, and
everything else is independent by construction.

Variants (`data.experiment`)
----------------------------
planted_parents   P(X=1 | A,B)   = 0.5 + kappa/2 * (A + B - 1)
                      corr(A,X) = corr(B,X) = +kappa/2
                      corr(D1,X) = corr(D2,X) = 0

planted_swap      P(X=1 | D1,D2) = 0.5 + kappa/2 * (D1 + D2 - 1)
                      the mirror image: distractors become the parents, and
                      A, B become the null channels. Rules out "the cross-block
                      just tracks whichever concepts drive the target", since
                      y is still 4A + 2B + X.

planted_diff      P(X=1 | A,B)   = 0.5 + kappa/2 * (A - B)
                      corr(A,X) = +kappa/2, corr(B,X) = -kappa/2
                      tests whether the sign is recovered, not just membership.

`data.kappa` in [0, 1] sets the strength. kappa = 0 is the null: X is
independent of every concept, so every cross-block entry should sit at the
noise floor. Sweeping kappa in {0, 0.3, 0.6, 0.9} turns a single ratio into a
dose-response curve, which is the point of the design.

Exact consequences (verified by `summarize_dataset`)
----------------------------------------------------
    P(X = 1)                     = 0.5           for every kappa
    corr(parent, X)              = kappa / 2
    corr(non-parent, X)          = 0
    concept-only task ceiling    = 0.5 + kappa/4
    residual headroom            = 0.5 - kappa/4

X is never a deterministic function of the concepts for kappa < 1, so no head
can shortcut it and the residual is strictly required.

Corruption
----------
Two independent settings, and both matter for different reasons.

`corruption_strength` corrupts channel 5 only. X must NOT be perfectly readable
from d5: if it were, the concepts would add nothing on top of the image and the
optimal cross-block would be zero. Sigma carries information exactly where the
concepts know something the residual does not. Target a probe accuracy of
X from the residual around 0.80-0.85.

`concept_corruption_strength` corrupts channels 1-4, all four on the same
terms so parents and distractors stay comparable. Mild is right here (aim for
~0.93-0.95 concept accuracy): some concept-side uncertainty keeps the
coherent-sampling mechanism alive, but too much reopens the memorisation and
copy-a-concept failure modes. Default 0.0.

Corruption is drawn deterministically per (sample, channel) from `seed`, so a
split is reproducible across machines; `fingerprint()` hashes the pixels of a
probe subset to prove it.
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


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# Only the parent pair changes between variants, plus whether the second parent
# enters with a positive or negative sign. Everything else -- the concepts, the
# target, the channel layout -- is shared.

CONCEPT_NAMES: Tuple[str, ...] = (
    "A::digit1_ge_5",
    "B::digit2_ge_5",
    "D1::digit3_ge_5",
    "D2::digit4_ge_5",
)

# hidden_concepts keeps width 2 so downstream analysis code can rely on index 1
# being X. Index 0 is an inert slot, retained only for that shape stability; no
# part of the experiment reads it.
ORACLE_NAMES: Tuple[str, ...] = ("A2::unused", "X::digit5_ge_5")

NUM_DIGITS = 5
NUM_CLASSES = 8
HIDDEN_CHANNEL = 4                      # d5, the channel X is read from
CONCEPT_CHANNELS: Tuple[int, ...] = (0, 1, 2, 3)

# The arrays that fully determine a split -- everything __getitem__ needs
# beyond the scalar settings in meta.json. Corruption itself is not part of
# this: it is redrawn deterministically from `seed` at __getitem__ time (see
# module docstring), so it never needs to be persisted.
MANIFEST_KEYS: Tuple[str, ...] = (
    "digit_labels_all",
    "source_indices",
    "observed_concepts",
    "hidden_concepts",
    "task_labels",
)

EXPERIMENT_SPECS: Dict[str, Dict] = {
    # parents: (concept index, sign) pairs feeding P(X=1 | .)
    "planted_parents": {
        "parents": ((0, +1), (1, +1)),
        "description": "X depends on A and B, both positively",
    },
    "planted_swap": {
        "parents": ((2, +1), (3, +1)),
        "description": "X depends on D1 and D2 instead; A and B become null",
    },
    "planted_diff": {
        "parents": ((0, +1), (1, -1)),
        "description": "X depends on A positively and B negatively",
    },
}

DEFAULT_EXPERIMENT = "planted_parents"


def _normalize_experiment(experiment: Optional[str]) -> str:
    name = str(experiment or DEFAULT_EXPERIMENT).strip().lower()
    aliases = {
        "parents": "planted_parents",
        "swap": "planted_swap",
        "diff": "planted_diff",
        "null": "planted_parents",      # the null is kappa=0, not a variant
    }
    name = aliases.get(name, name)
    if name not in EXPERIMENT_SPECS:
        raise ValueError(
            f"Unknown experiment {experiment!r}. "
            f"Choose one of {sorted(EXPERIMENT_SPECS)} "
            f"(the null control is kappa=0, not a separate experiment)."
        )
    return name


def get_experiment_spec(experiment: str) -> Dict:
    return EXPERIMENT_SPECS[_normalize_experiment(experiment)]


def experiment_parent_names(experiment: str) -> List[str]:
    """Short names of the concepts X was planted to depend on."""
    spec = get_experiment_spec(experiment)
    return [CONCEPT_NAMES[i].split("::")[0] for i, _ in spec["parents"]]


def experiment_concept_names(experiment: str = DEFAULT_EXPERIMENT) -> List[str]:
    return list(CONCEPT_NAMES)


def experiment_num_digits(experiment: str = DEFAULT_EXPERIMENT) -> int:
    return NUM_DIGITS


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _cfg_get(config, name: str, default):
    """Read `name` off an OmegaConf node, a namespace or a dict."""
    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            value = config.get(name, default)
        except TypeError:
            value = getattr(config, name, default)
    else:
        value = getattr(config, name, default)
    return default if value is None else value


def resolve_num_digits(config, experiment: str = DEFAULT_EXPERIMENT) -> int:
    """
    Channels to stack. Defaults to 5; a larger value appends pure distractor
    channels that no concept, oracle or label depends on. Smaller is an error.
    """
    requested = _cfg_get(config, "num_covariates", NUM_DIGITS)
    requested = int(requested)
    if requested < NUM_DIGITS:
        raise ValueError(
            f"data.num_covariates={requested} but experiment={experiment!r} "
            f"reads {NUM_DIGITS} digits (d1..d4 for the concepts, d5 for X)."
        )
    return requested


def _check_kappa(kappa) -> float:
    kappa = float(kappa)
    if not (0.0 <= kappa <= 1.0):
        raise ValueError(f"data.kappa must be in [0, 1], got {kappa}.")
    return kappa


# ---------------------------------------------------------------------------
# Digit generation
# ---------------------------------------------------------------------------


def _balanced_digit_pairs(n_samples: int, seed: int) -> np.ndarray:
    """
    Ordered digit pairs, exactly uniform over the 100 combinations in every
    complete block of 100 samples. Both marginals are then exactly uniform, so
    the concepts derived from them are exactly balanced and uncorrelated.
    """
    rng = np.random.default_rng(seed)
    all_pairs = np.array(
        [(d1, d2) for d1 in range(10) for d2 in range(10)], dtype=np.int64
    )

    blocks = []
    for _ in range(n_samples // 100):
        block = all_pairs.copy()
        rng.shuffle(block)
        blocks.append(block)

    remainder = n_samples % 100
    if remainder:
        block = all_pairs.copy()
        rng.shuffle(block)
        blocks.append(block[:remainder])

    pairs = np.concatenate(blocks, axis=0)
    rng.shuffle(pairs)
    return pairs


def _balanced_concept_digits(n_samples: int, seed: int) -> np.ndarray:
    """
    d1..d4, shape [n, 4]. Two independent balanced pair blocks, so (d1,d2) and
    (d3,d4) are exactly uniform and independent of each other. That independence
    is what makes D1 and D2 exactly uncorrelated with an X built from A and B,
    and vice versa under planted_swap.
    """
    return np.concatenate(
        [
            _balanced_digit_pairs(n_samples, seed=seed),
            _balanced_digit_pairs(n_samples, seed=seed + 5_000_011),
        ],
        axis=1,
    )


def _padding_digits(n_samples: int, n_extra: int, seed: int) -> np.ndarray:
    """Extra channels beyond d5, when num_covariates > 5. Pure distractors."""
    if n_extra <= 0:
        return np.empty((n_samples, 0), dtype=np.int64)
    blocks = [
        _balanced_digit_pairs(n_samples, seed=seed + k * 7_000_003)
        for k in range((n_extra + 1) // 2)
    ]
    return np.concatenate(blocks, axis=1)[:, :n_extra]


def _sample_hidden_x(
    observed: np.ndarray,
    experiment: str,
    kappa: float,
    seed: int,
) -> np.ndarray:
    """
    Draw X conditional on the variant's parent concepts.

        P(X = 1 | parents) = 0.5 + (kappa / 2) * sum_j sign_j * (p_j - 0.5) * 2
                           = 0.5 + (kappa / 2) * sum_j sign_j * (2 * p_j - 1) / 2 * 2

    In the two-parent case this reduces to the forms quoted in the docstring:
        both +1 : 0.5 + kappa/2 * (P1 + P2 - 1)
        +1, -1  : 0.5 + kappa/2 * (P1 - P2)

    Both give corr(parent, X) = +/- kappa/2 exactly, and P(X=1) = 0.5.
    """
    spec = get_experiment_spec(experiment)
    rng = np.random.default_rng(seed)

    contribution = np.zeros(len(observed), dtype=np.float64)
    for index, sign in spec["parents"]:
        # (concept - 0.5) is +/-0.5, so two parents with the same sign span
        # [-1, +1] and the probability stays inside [0, 1] for kappa <= 1.
        contribution += sign * (observed[:, index] - 0.5)

    p = 0.5 + (kappa / 2.0) * contribution
    p = np.clip(p, 0.0, 1.0)
    return (rng.random(len(observed)) < p).astype(np.int64)


def _digit_for_x(x: np.ndarray, seed: int) -> np.ndarray:
    """d5 uniform over {5..9} when X=1, over {0..4} when X=0, so X = 1[d5>=5]."""
    rng = np.random.default_rng(seed)
    low = rng.integers(0, 5, size=len(x))
    high = rng.integers(5, 10, size=len(x))
    return np.where(x == 1, high, low).astype(np.int64)


def _build_class_pools(targets: torch.Tensor) -> Dict[int, np.ndarray]:
    """MNIST indices grouped by digit label."""
    labels = targets.numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    return {digit: np.flatnonzero(labels == digit) for digit in range(10)}


def _split_train_val_pools(
    class_pools: Dict[int, np.ndarray],
    val_percent: float,
    seed: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Disjoint per-digit image pools, so no MNIST image appears in both."""
    rng = np.random.default_rng(seed)
    train_pools, val_pools = {}, {}
    for digit, pool in class_pools.items():
        pool = pool.copy()
        rng.shuffle(pool)
        cut = max(1, int(round(len(pool) * val_percent)))
        val_pools[digit] = pool[:cut]
        train_pools[digit] = pool[cut:]
    return train_pools, val_pools


def _sample_source_indices(
    digits: np.ndarray,
    class_pools: Dict[int, np.ndarray],
    seed: int,
) -> np.ndarray:
    """Pick an actual MNIST image for each desired digit identity."""
    rng = np.random.default_rng(seed)
    n_samples, n_digits = digits.shape
    out = np.empty((n_samples, n_digits), dtype=np.int64)
    for i in range(n_samples):
        for j in range(n_digits):
            out[i, j] = rng.choice(class_pools[int(digits[i, j])])
    return out


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------


def _apply_corruption(
    image: torch.Tensor,
    corruption: str,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Corrupt a single digit image, shape [1, 28, 28]."""
    corruption = (corruption or "none").lower()

    if corruption in {"none", "", "clean"} or strength <= 0:
        return image

    if corruption in {"gaussian", "gaussian_noise", "noise"}:
        noise = torch.randn(
            image.shape, generator=generator, dtype=image.dtype, device=image.device
        )
        return torch.clamp(image + strength * noise, 0.0, 1.0)

    if corruption == "blur":
        sigma = max(0.1, 0.5 + 1.5 * float(strength))
        kernel = max(3, min(int(2 * round(2 * sigma) + 1), 13))
        return TF.gaussian_blur(image, kernel_size=[kernel, kernel], sigma=[sigma, sigma])

    if corruption == "occlusion":
        side = int(round(28 * float(strength)))
        if side <= 0:
            return image
        side = min(side, 28)
        top = int(torch.randint(0, 28 - side + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, 28 - side + 1, (1,), generator=generator).item())
        out = image.clone()
        out[:, top : top + side, left : left + side] = 0.0
        return out

    raise ValueError(
        f"Unknown corruption={corruption!r}. "
        "Use 'none', 'gaussian', 'blur' or 'occlusion'."
    )


def _fingerprint_arrays(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MNISTPlantedCovDataset(Dataset):
    """
    Each sample:

        features          [num_digits, 28, 28]   one digit per channel
        concepts          [4]                    A, B, D1, D2 (minus removed)
        labels            scalar                 y = 4A + 2B + X
        hidden_concepts   [2]                    [unused, X] -- oracle only
        digit_labels      [num_digits]           the true digit identities
    """

    def __init__(
        self,
        mnist_dataset: MNIST,
        class_pools: Optional[Dict[int, np.ndarray]],
        dataset_size: int,
        seed: int,
        experiment: str = DEFAULT_EXPERIMENT,
        kappa: float = 0.9,
        corruption: str = "gaussian",
        corruption_strength: float = 0.0,
        concept_corruption: Optional[str] = None,
        concept_corruption_strength: float = 0.0,
        num_covariates: Optional[int] = None,
        removed_concepts: Optional[List[int]] = None,
        manifest: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        `manifest`, when given, is the dict written by `save_MNIST_add_cov_data`
        (one array per `MANIFEST_KEYS`): it replaces steps 1-4 below verbatim, so
        `class_pools` is unused and may be None. This is what
        `load_saved_MNIST_add_cov_data` uses to reconstruct a split byte-for-byte
        from disk instead of re-drawing it from `seed`.
        """
        super().__init__()

        self.mnist_dataset = mnist_dataset
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.experiment = _normalize_experiment(experiment)
        self.kappa = _check_kappa(kappa)

        self.num_digits = NUM_DIGITS if num_covariates is None else int(num_covariates)
        if self.num_digits < NUM_DIGITS:
            raise ValueError(
                f"num_covariates={self.num_digits} < {NUM_DIGITS} required digits."
            )

        self.observed_concept_names = list(CONCEPT_NAMES)
        self.oracle_concept_names = list(ORACLE_NAMES)
        self.num_classes = NUM_CLASSES
        self.parent_idx = [i for i, _ in get_experiment_spec(self.experiment)["parents"]]
        self.nonparent_idx = [
            i for i in range(len(CONCEPT_NAMES)) if i not in self.parent_idx
        ]

        self.corruption = (corruption or "none").lower()
        self.corruption_strength = float(corruption_strength)
        self.concept_corruption = (concept_corruption or self.corruption).lower()
        self.concept_corruption_strength = float(concept_corruption_strength)

        self.removed_concept_idx = sorted(int(i) for i in (removed_concepts or []))
        self.kept_concept_idx = [
            i for i in range(len(CONCEPT_NAMES)) if i not in self.removed_concept_idx
        ]

        if manifest is not None:
            # Verbatim from disk: skip generation entirely so the split does
            # not depend on the run seed or on numpy/torch RNG versions.
            for key in MANIFEST_KEYS:
                setattr(self, key, np.asarray(manifest[key]))
        else:
            # 1) d1..d4: exactly balanced, mutually independent.
            concept_digits = _balanced_concept_digits(self.dataset_size, seed=self.seed)

            observed = (concept_digits >= 5).astype(np.float32)     # [N, 4]

            # 2) X conditional on the variant's parents, then d5 consistent with X.
            x = _sample_hidden_x(observed, self.experiment, self.kappa, seed=self.seed + 3)
            hidden_digit = _digit_for_x(x, seed=self.seed + 4)

            # 3) Any channels beyond d5 are pure padding.
            padding = _padding_digits(
                self.dataset_size, self.num_digits - NUM_DIGITS, seed=self.seed + 5
            )

            self.digit_labels_all = np.concatenate(
                [concept_digits, hidden_digit[:, None], padding], axis=1
            )

            self.observed_concepts = observed
            self.hidden_concepts = np.stack(
                [np.zeros_like(x, dtype=np.float32), x.astype(np.float32)], axis=1
            )
            self.task_labels = (
                4 * observed[:, 0] + 2 * observed[:, 1] + x
            ).astype(np.int64)

            # 4) MNIST images backing those identities.
            self.source_indices = _sample_source_indices(
                self.digit_labels_all, class_pools, seed=self.seed + 1
            )

    # -- accessors ----------------------------------------------------------

    def concept_names(self) -> List[str]:
        return [self.observed_concept_names[i] for i in self.kept_concept_idx]

    def removed_concept_names(self) -> List[str]:
        return [self.observed_concept_names[i] for i in self.removed_concept_idx]

    def parent_names(self) -> List[str]:
        return [CONCEPT_NAMES[i].split("::")[0] for i in self.parent_idx]

    def concept_only_ceiling(self) -> float:
        """
        Best task accuracy achievable from perfect concepts and no residual.
        Computed from the split rather than the 0.5 + kappa/4 formula, so it
        stays right if the generator ever changes.
        """
        obs = self.observed_concepts
        x = self.hidden_concepts[:, 1]
        a, b = obs[:, 0] > 0.5, obs[:, 1] > 0.5

        correct = 0.0
        for av in (0, 1):
            for bv in (0, 1):
                cell = (a == bool(av)) & (b == bool(bv))
                if not cell.any():
                    continue
                p = x[cell].mean()
                correct += cell.sum() * max(p, 1.0 - p)
        return float(correct / len(x))

    def planted_correlations(self) -> Dict[str, float]:
        """Realised corr(concept, X) on this split."""
        x = self.hidden_concepts[:, 1]
        return {
            CONCEPT_NAMES[j].split("::")[0]: float(
                np.corrcoef(self.observed_concepts[:, j], x)[0, 1]
            )
            for j in range(len(CONCEPT_NAMES))
        }

    def fingerprint(self) -> str:
        """
        Content hash of the split: generator settings, digit identities, which
        MNIST images back them, the derived labels, and the pixels of a fixed
        probe subset (so a differing torch RNG for the corruption is caught).
        Independent of which concepts are exposed.
        """
        spec = "|".join(
            str(v)
            for v in (
                self.dataset_size,
                self.seed,
                self.experiment,
                self.kappa,
                self.num_digits,
                self.corruption,
                self.corruption_strength,
                self.concept_corruption,
                self.concept_corruption_strength,
            )
        )
        n_probe = min(self.dataset_size, 16)
        probe_idx = np.unique(np.linspace(0, self.dataset_size - 1, n_probe).astype(int))
        probe_pixels = torch.stack([self[int(i)]["features"] for i in probe_idx]).numpy()

        return _fingerprint_arrays(
            np.frombuffer(spec.encode(), dtype=np.uint8),
            self.digit_labels_all,
            self.source_indices,
            self.observed_concepts,
            self.hidden_concepts,
            self.task_labels,
            probe_pixels,
        )

    # -- sample -------------------------------------------------------------

    def _load_mnist_tensor(self, source_index: int) -> torch.Tensor:
        img = self.mnist_dataset.data[int(source_index)].float() / 255.0
        return img.unsqueeze(0)                                   # [1, 28, 28]

    def _channel_corruption(self, channel: int) -> Tuple[str, float]:
        """Channel 5 gets its own setting; the concept channels share another."""
        if channel == HIDDEN_CHANNEL:
            return self.corruption, self.corruption_strength
        if channel in CONCEPT_CHANNELS:
            return self.concept_corruption, self.concept_corruption_strength
        return "none", 0.0                                        # padding channels

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        images = []
        for channel in range(self.num_digits):
            img = self._load_mnist_tensor(self.source_indices[index, channel])
            corruption, strength = self._channel_corruption(channel)
            if corruption not in {"none", "", "clean"} and strength > 0:
                # Deterministic per (sample, channel), so the split is
                # reproducible and fingerprint() is meaningful.
                g = torch.Generator()
                g.manual_seed(self.seed * 1_000_003 + int(index) + channel * 7_000_003)
                img = _apply_corruption(img, corruption, strength, generator=g)
            images.append(img)

        features = torch.cat(images, dim=0)                       # [num_digits, 28, 28]

        all_observed = torch.from_numpy(self.observed_concepts[index]).float()
        hidden = torch.from_numpy(self.hidden_concepts[index]).float()

        return {
            "img_code": int(index),
            "labels": torch.tensor(self.task_labels[index], dtype=torch.long),
            "features": features,
            "concepts": all_observed[self.kept_concept_idx],
            # Oracle-only analysis metadata; NOT part of `concepts`.
            "hidden_concepts": hidden,                            # [unused, X]
            "removed_concepts": all_observed[self.removed_concept_idx],
            "X": hidden[1],
            "digit_labels": torch.from_numpy(self.digit_labels_all[index]).long(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_MNIST_planted_cov_datasets(
    config,
    incomplete: Optional[bool] = None,
    seed: int = 42,
    log_file: Optional[str] = None,
):
    """
    Return train / val / test Dataset objects.

    Config fields (all optional except where noted)
    ----------------------------------------------
    data_path                      default './data'
    train_dataset_size             default 12000
    val_dataset_size               default int(train * val_percent)
    test_dataset_size              default 10000
    val_percent                    default 0.2
    data_seed                      overrides the `seed` argument

    experiment                     planted_parents | planted_swap | planted_diff
    kappa                          [0, 1], default 0.9   (0 = null control)
    num_covariates                 >= 5, default 5

    corruption                     none | gaussian | blur | occlusion
    corruption_strength            channel 5 only; tune so the residual reads X
                                   at roughly 0.80-0.85
    concept_corruption             defaults to `corruption`
    concept_corruption_strength    channels 1-4; mild (~0.93-0.95 concept acc)

    test_corruption_strength       defaults to corruption_strength
    test_concept_corruption_strength
                                   defaults to concept_corruption_strength

    removed_concepts               indices to withhold from the bottleneck
    """
    data_path = _cfg_get(config, "data_path", "./data")
    seed = int(_cfg_get(config, "data_seed", seed))

    experiment = _normalize_experiment(_cfg_get(config, "experiment", DEFAULT_EXPERIMENT))
    kappa = _check_kappa(_cfg_get(config, "kappa", 0.9))
    num_covariates = resolve_num_digits(config, experiment)

    train_size = int(_cfg_get(config, "train_dataset_size", 12000))
    val_percent = float(_cfg_get(config, "val_percent", 0.2))
    val_size = int(_cfg_get(config, "val_dataset_size", int(train_size * val_percent)))
    test_size = int(_cfg_get(config, "test_dataset_size", 10000))

    corruption = str(_cfg_get(config, "corruption", "gaussian"))
    corruption_strength = float(_cfg_get(config, "corruption_strength", 0.0))
    concept_corruption = str(_cfg_get(config, "concept_corruption", corruption))
    concept_strength = float(_cfg_get(config, "concept_corruption_strength", 0.0))

    test_corruption_strength = float(
        _cfg_get(config, "test_corruption_strength", corruption_strength)
    )
    test_concept_strength = float(
        _cfg_get(config, "test_concept_corruption_strength", concept_strength)
    )

    removed = list(_cfg_get(config, "removed_concepts", []) or [])

    mnist_root = os.path.join(data_path, "MNIST_ADD_COV")
    os.makedirs(mnist_root, exist_ok=True)
    mnist_train = MNIST(root=mnist_root, train=True, download=True)
    mnist_test = MNIST(root=mnist_root, train=False, download=True)

    pools = _build_class_pools(mnist_train.targets)
    train_pools, val_pools = _split_train_val_pools(pools, val_percent, seed=seed + 7)
    test_pools = _build_class_pools(mnist_test.targets)

    shared = dict(
        experiment=experiment,
        kappa=kappa,
        corruption=corruption,
        concept_corruption=concept_corruption,
        num_covariates=num_covariates,
        removed_concepts=removed,
    )

    trainset = MNISTPlantedCovDataset(
        mnist_train, train_pools, train_size, seed=seed,
        corruption_strength=corruption_strength,
        concept_corruption_strength=concept_strength, **shared,
    )
    valset = MNISTPlantedCovDataset(
        mnist_train, val_pools, val_size, seed=seed + 100_003,
        corruption_strength=corruption_strength,
        concept_corruption_strength=concept_strength, **shared,
    )
    testset = MNISTPlantedCovDataset(
        mnist_test, test_pools, test_size, seed=seed + 200_003,
        corruption_strength=test_corruption_strength,
        concept_corruption_strength=test_concept_strength, **shared,
    )

    log_split_fingerprints(
        {"train": trainset, "val": valset, "test": testset},
        seed=seed, experiment=experiment, kappa=kappa, log_file=log_file,
    )
    return trainset, valset, testset


# Alias, so call sites written against the old module keep working.
get_MNIST_add_cov_datasets = get_MNIST_planted_cov_datasets


# ---------------------------------------------------------------------------
# Materialised splits (save/load to disk)
# ---------------------------------------------------------------------------
# Generate once, copy the folder to the cluster (or point both at a shared
# path) and every run loads byte-identical train/val/test regardless of the
# run seed -- pixels aside, which are redrawn deterministically from `seed` at
# __getitem__ time (see module docstring) rather than stored.

SPLIT_ROOT = "splits"
SPLIT_MNIST_SOURCE: Dict[str, str] = {"train": "train", "val": "train", "test": "test"}


def _split_dir(config, data_dir_name: str, split: str) -> str:
    data_path = _cfg_get(config, "data_path", "./data")
    return os.path.join(data_path, "MNIST_ADD_COV", SPLIT_ROOT, data_dir_name, split)


def save_MNIST_add_cov_data(config, train, val, test, log_file=None) -> str:
    """Write the manifest (§MANIFEST_KEYS) of each split to its own folder."""
    data_path = _cfg_get(config, "data_path", "./data")
    root = os.path.join(data_path, "MNIST_ADD_COV", SPLIT_ROOT)
    os.makedirs(root, exist_ok=True)

    save_name = _cfg_get(config, "save_data_name", None) or (
        f"seed_{train.seed}_n_{len(train)}_{len(val)}_{len(test)}"
        f"_{train.experiment}_kappa_{train.kappa}"
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
        np.savez_compressed(os.path.join(split_dir, "manifest.npz"), **arrays)

        meta = {
            "split": split,
            "mnist_source": SPLIT_MNIST_SOURCE[split],
            "dataset_size": len(dataset),
            "seed": dataset.seed,
            "experiment": dataset.experiment,
            "kappa": dataset.kappa,
            "num_digits": dataset.num_digits,
            "corruption": dataset.corruption,
            "corruption_strength": dataset.corruption_strength,
            "concept_corruption": dataset.concept_corruption,
            "concept_corruption_strength": dataset.concept_corruption_strength,
            "fingerprint": dataset.fingerprint(),
        }
        with open(os.path.join(split_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    with open(os.path.join(save_dir, "info.txt"), "w") as f:
        f.write("MNIST-Add-Cov materialised split\n")
        f.write(f"generator seed: {train.seed}\n")
        f.write(f"experiment: {train.experiment}   kappa: {train.kappa}\n")
        f.write(
            f"corruption: {train.corruption} (strength={train.corruption_strength})\n"
        )
        f.write(
            f"concept corruption: {train.concept_corruption} "
            f"(strength={train.concept_corruption_strength})\n"
        )
        f.write(
            f"test corruption: {test.corruption} (strength={test.corruption_strength})\n"
        )
        f.write(f"sizes: train={len(train)}, val={len(val)}, test={len(test)}\n")
        f.write(f"num digit channels: {train.num_digits}\n")
        f.write(f"observed concepts (full): {train.observed_concept_names}\n")
        f.write(f"oracle variables: {train.oracle_concept_names}\n")
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

    The split is taken verbatim from disk -- no RNG is touched, so the run
    seed no longer influences which samples land where. `data.removed_concepts`
    is applied on top, which is what makes an incomplete run the *same* split
    with fewer supervised concepts.
    """
    data_dir_name = _cfg_get(config, "data_dir_name", None)
    if data_dir_name is None:
        raise ValueError("load_saved_MNIST_add_cov_data requires data.data_dir_name.")

    data_path = _cfg_get(config, "data_path", "./data")
    requested_experiment = _normalize_experiment(
        _cfg_get(config, "experiment", DEFAULT_EXPERIMENT)
    )
    removed_concepts = resolve_removed_concepts(
        _cfg_get(config, "removed_concepts", None), experiment=requested_experiment
    )
    num_covariates = resolve_num_digits(config, requested_experiment)

    # Same root as get_MNIST_planted_cov_datasets, so a saved split reuses the
    # exact same on-disk MNIST cache as a plain generate-on-the-fly run.
    mnist_root = os.path.join(data_path, "MNIST_ADD_COV")
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
            manifest = {key: npz[key] for key in MANIFEST_KEYS}

        saved_experiment = _normalize_experiment(meta["experiment"])
        if saved_experiment != requested_experiment:
            raise ValueError(
                f"Saved split experiment={saved_experiment!r} but YAML requests "
                f"experiment={requested_experiment!r}. Use the matching split."
            )

        saved_num_digits = int(meta["num_digits"])
        if saved_num_digits != num_covariates:
            raise ValueError(
                f"Saved split {data_dir_name!r} stacks {saved_num_digits} digit "
                f"channels but data.num_covariates={num_covariates}. Set "
                f"num_covariates={saved_num_digits}, or regenerate the split "
                f"with data.save_data=True."
            )

        dataset = MNISTPlantedCovDataset(
            mnist_dataset=mnist_by_source[meta["mnist_source"]],
            class_pools=None,
            dataset_size=meta["dataset_size"],
            seed=meta["seed"],
            experiment=saved_experiment,
            kappa=meta["kappa"],
            corruption=meta["corruption"],
            corruption_strength=meta["corruption_strength"],
            concept_corruption=meta["concept_corruption"],
            concept_corruption_strength=meta["concept_corruption_strength"],
            num_covariates=saved_num_digits,
            removed_concepts=removed_concepts,
            manifest=manifest,
        )

        loaded_fingerprint = dataset.fingerprint()
        if loaded_fingerprint != meta["fingerprint"]:
            print(
                f"WARNING: {split} fingerprint on load ({loaded_fingerprint}) "
                f"differs from the saved fingerprint ({meta['fingerprint']}); "
                "corrupted pixels may not reproduce identically on this machine."
            )

        datasets.append(dataset)

    trainset, valset, testset = datasets
    sync_num_concepts(config, trainset, log_file=log_file)
    sync_num_classes(config, trainset, log_file=log_file)
    sync_num_covariates(config, trainset, log_file=log_file)

    message = f"Loaded MNIST-Add-Cov splits from {data_dir_name}"
    if removed_concepts:
        message += f" (removed concepts: {trainset.removed_concept_names()})"
    print(message)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(message + "\n")

    log_split_fingerprints(
        {"train": trainset, "val": valset, "test": testset},
        seed=trainset.seed, experiment=trainset.experiment, kappa=trainset.kappa,
        log_file=log_file,
    )
    return trainset, valset, testset


# ---------------------------------------------------------------------------
# Config sync helpers (mirrors of the CUB path)
# ---------------------------------------------------------------------------


def _write_back(config, name: str, value, label: str, log_file=None) -> None:
    configured = _cfg_get(config, name, None)
    if configured is not None and int(configured) != int(value):
        message = f"MNIST-Planted-Cov {label}: {name} {configured} -> {value}"
        print(message)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(message + "\n")
    try:
        setattr(config, name, int(value))
    except Exception:
        pass


def sync_num_concepts(config, dataset: MNISTPlantedCovDataset, log_file=None) -> None:
    _write_back(config, "num_concepts", len(dataset.kept_concept_idx),
                f"experiment={dataset.experiment}", log_file)


def sync_num_classes(config, dataset: MNISTPlantedCovDataset, log_file=None) -> None:
    _write_back(config, "num_classes", dataset.num_classes,
                f"experiment={dataset.experiment}", log_file)


def sync_num_covariates(config, dataset: MNISTPlantedCovDataset, log_file=None) -> None:
    _write_back(config, "num_covariates", dataset.num_digits,
                f"experiment={dataset.experiment}", log_file)


def resolve_removed_concepts(removed, experiment: str = DEFAULT_EXPERIMENT) -> List[int]:
    """No automatic removals in this experiment; concepts are all supervised."""
    return sorted(int(i) for i in (removed or []))


def get_mnist_add_cov_concept_names(experiment: str = DEFAULT_EXPERIMENT) -> List[str]:
    return list(CONCEPT_NAMES)


def get_mnist_add_cov_oracle_names(experiment: str = DEFAULT_EXPERIMENT) -> List[str]:
    return list(ORACLE_NAMES)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def log_split_fingerprints(
    splits: Dict[str, MNISTPlantedCovDataset],
    seed: int,
    experiment: str,
    kappa: float,
    log_file: Optional[str] = None,
) -> Dict[str, str]:
    """
    Print a content hash per split. Two runs printing the same fingerprints hold
    literally the same samples in the same order, on any machine.
    """
    fingerprints = {name: ds.fingerprint() for name, ds in splits.items()}
    reference = next(iter(splits.values()))

    lines = [
        f"MNIST-Planted-Cov split fingerprints "
        f"(data seed={seed}, experiment={experiment}, kappa={kappa}):"
    ]
    lines += [
        f"  {name:<5} n={len(splits[name]):<6} sha256[:16]={fp}"
        for name, fp in fingerprints.items()
    ]
    lines.append(f"  concepts exposed: {reference.concept_names()}")
    lines.append(f"  planted parents : {reference.parent_names()}")
    if reference.removed_concept_idx:
        lines.append(f"  concepts removed: {reference.removed_concept_names()}")
    lines.append(
        "  Matching fingerprints => identical train/val/test samples across machines."
    )

    message = "\n".join(lines)
    print(message)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(message + "\n")
    return fingerprints


def summarize_dataset(dataset: MNISTPlantedCovDataset) -> None:
    """
    The pre-training sanity check. Prints the marginal concept-X correlations
    the whole covariance argument rests on, the task ceiling the residual has to
    beat, and confirms X is not determined by the concepts (if it were, the head
    would compute X directly and the residual would stay empty).
    """
    obs = dataset.observed_concepts
    x = dataset.hidden_concepts[:, 1]
    names = [n.split("::")[0] for n in dataset.observed_concept_names]
    parents = set(dataset.parent_names())

    print(f"N={len(dataset)}   experiment={dataset.experiment}   kappa={dataset.kappa}")
    print(f"  {get_experiment_spec(dataset.experiment)['description']}")
    print(f"Channels: {dataset.num_digits} "
          f"({NUM_DIGITS} used, {dataset.num_digits - NUM_DIGITS} padding)")
    print(f"Concepts exposed: {dataset.concept_names()}")
    print(f"Concept means {names}: {np.round(obs.mean(0), 4)}")
    print(f"P(X = 1) = {x.mean():.4f}   (0.5 by construction, any kappa)")
    print(f"Task class counts: {np.bincount(dataset.task_labels, minlength=NUM_CLASSES)}")

    print("\nMarginal concept-X correlations (what the cross-block should recover):")
    expected = {}
    for index, sign in get_experiment_spec(dataset.experiment)["parents"]:
        expected[names[index]] = sign * dataset.kappa / 2.0
    for name, value in dataset.planted_correlations().items():
        target = expected.get(name, 0.0)
        tag = "  <- parent" if name in parents else "  (null channel)"
        print(f"  Corr({name}, X) = {value:+.4f}   expected {target:+.4f}{tag}")

    print("\nIs X determined by the exposed concepts?")
    a, b = obs[:, 0] > 0.5, obs[:, 1] > 0.5
    for av in (0, 1):
        for bv in (0, 1):
            cell = (a == bool(av)) & (b == bool(bv))
            if not cell.any():
                continue
            p = x[cell].mean()
            flag = "  <- DETERMINED (residual not required here)" if p in (0.0, 1.0) else ""
            print(f"  A={av}, B={bv}: n={cell.sum():<6} P(X=1)={p:.4f}{flag}")

    ceiling = dataset.concept_only_ceiling()
    # The ceiling uses A and B, which are the target's own bits. Under
    # planted_swap the parents are D1/D2, so A and B say nothing about X and
    # the ceiling stays at 0.5 whatever kappa is.
    formula = 0.5 + dataset.kappa / 4 if 0 in dataset.parent_idx else 0.5
    print(f"\nConcept-only task ceiling : {ceiling:.4f}   (expected {formula:.4f})")
    print(f"Residual headroom         : {1.0 - ceiling:.4f}")
    print("The residual-only probe of X must clear the ceiling before the")
    print("cross-block in Sigma means anything.")


if __name__ == "__main__":
    # Smoke test: verify the planted structure over the kappa sweep without
    # touching MNIST pixels.
    class _Cfg(dict):
        def get(self, name, default=None):
            return dict.get(self, name, default)

    for experiment in ("planted_parents", "planted_swap", "planted_diff"):
        for kappa in (0.0, 0.3, 0.6, 0.9):
            n = 20000
            digits = _balanced_concept_digits(n, seed=0)
            observed = (digits >= 5).astype(np.float32)
            x = _sample_hidden_x(observed, experiment, kappa, seed=3)
            corr = [float(np.corrcoef(observed[:, j], x)[0, 1]) for j in range(4)]
            ceiling = 0.5 + kappa / 4
            print(f"{experiment:<16} kappa={kappa:<4} "
                  f"corr(A,B,D1,D2 ; X) = {np.round(corr, 3)}   "
                  f"ceiling={ceiling:.3f}")
