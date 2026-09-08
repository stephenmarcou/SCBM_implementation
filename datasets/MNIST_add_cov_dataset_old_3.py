"""
Controlled two-digit MNIST dataset for residual/concept covariance experiments.

Select the experiment from YAML with `data.experiment`.

Supported experiments
---------------------
1) original
   Input:
       Two MNIST digit images stacked as channels -> [2, 28, 28]

   Exposed concepts:
       A1 = 1[d1 is odd]
       H1 = 1[d1 >= 6]
       H2 = 1[d2 >= 6]

   Hidden/oracle variables:
       A2 = 1[d2 is odd]
       X  = planted_function(A1, A2)

   Target:
       M = H1 OR H2
       y = 4*A1 + 2*M + X        # 8 classes

2) hidden_xor
   Input:
       Same two-digit MNIST input.

   Exposed concepts:
       H1 = 1[d1 >= 6]
       H2 = 1[d2 >= 6]

   Hidden/oracle variable:
       X = H1 XOR H2

   Target:
       y = X                     # binary task

   A1 is automatically withheld from the bottleneck in this mode. X is NEVER
   included in the supervised `concepts` tensor; it is retained only as oracle
   metadata for post-hoc residual recovery analysis.

3) hidden_carry            <-- covariance-identifies-the-residual experiment
   Input:
       FOUR MNIST digit images stacked as channels -> [4, 28, 28]
       d1, d2 are the addends; d3, d4 back the distractor concepts.

   Exposed concepts:
       A  = 1[d1 >= 5]
       B  = 1[d2 >= 5]
       D1 = 1[d3 >= 5]      distractor, independent of X
       D2 = 1[d4 >= 5]      distractor, independent of X

   Hidden/oracle variable:
       X = 1[d1 + d2 >= 10]          the addition carry

   Target:
       y = 4*A + 2*B + X             # 8 encodings, 6 reachable

   X is NOT a function of (A, B): when A != B the carry genuinely depends on
   the digit values, so a linear head cannot shortcut it and must route X
   through the residual channel. But X is strongly monotone in both parents:

       P(X)=0.45, P(X|A=1)=0.70  ->  Cov(X,A)=0.125,  corr = +0.50
       symmetric in B;  Cov(X,D1) = Cov(X,D2) = 0 exactly

   So the learnt concept<->residual cross-block should light up on exactly
   A and B and stay flat on D1 and D2. That contrast is the result.

4) hidden_diff             <-- signed variant of the above
       X = 1[d1 - d2 >= 2]
       y = 4*A + 2*B + X
   Same construction, but corr(X,A) = +0.50 and corr(X,B) = -0.50, so the
   cross-block has to recover *direction* and not merely membership. Here X is
   never determined by (A,B), for any value of (A,B).

5) hidden_carry_swap       <-- specificity control
       X = 1[d3 + d4 >= 10]          carry over the DISTRACTOR digits
       y = 4*A + 2*B + X
   The residual is still required, but now it should correlate with D1/D2 and
   not with A/B. Confirms the cross-block tracks the residual's actual content
   rather than whichever concepts happen to drive the target.

6) hidden_carry_null       <-- calibration run
       y = 2*A + B                   # 4 classes; X never enters the target
   X is still recorded as an oracle. The residual has no job, so every
   cross-block entry should go flat. This is what calibrates "how big is big".

In every hidden_* experiment X is NEVER part of the supervised `concepts`
tensor; it is oracle metadata only.

Input channels
--------------
The number of digit images stacked as channels is `data.num_covariates`, so the
input is always [num_covariates, 28, 28] and matches the encoder's first conv.
Each experiment declares how many digits it *needs* (2 for original/hidden_xor,
4 for the carry family) to build its concepts, its hidden X and its label; those
are always the leading channels. Leaving num_covariates unset falls back to that
requirement. Setting it higher appends independently drawn digit channels that
nothing in the experiment reads -- distractor input for varying the input
dimensionality while holding the causal structure fixed. Setting it lower is an
error.

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


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# Everything that differs between experiments lives here: how many digits the
# input stacks, which observed concepts exist, how the hidden X and the task
# label are built, and how many task classes result. Adding an experiment means
# adding an entry plus a label function -- no changes anywhere else.
#
# Every experiment keeps the oracle bank at a fixed width [A2, X] so that
# `hidden_concepts`, sample["A2"] and sample["X"] have the same shape for all
# experiments and downstream collation is unaffected.


def _labels_original(digits, planted_function):
    d1, d2 = int(digits[0]), int(digits[1])
    a1 = int(d1 % 2 == 1)
    a2 = int(d2 % 2 == 1)
    h1 = int(d1 >= 6)
    h2 = int(d2 >= 6)

    x = _planted_x(a1, a2, planted_function)
    m = int(h1 or h2)
    return [a1, h1, h2], [a2, x], 4 * a1 + 2 * m + x


def _labels_hidden_xor(digits, planted_function):
    d1, d2 = int(digits[0]), int(digits[1])
    a1 = int(d1 % 2 == 1)
    a2 = int(d2 % 2 == 1)
    h1 = int(d1 >= 6)
    h2 = int(d2 >= 6)

    # X is a nonlinear function of the TWO KNOWN concepts H1 and H2.
    x = int(bool(h1) ^ bool(h2))
    return [a1, h1, h2], [a2, x], x


def _carry_concepts(digits):
    """Shared [A, B, D1, D2] bank for the four-digit experiments."""
    d1, d2, d3, d4 = (int(d) for d in digits[:4])
    return [int(d1 >= 5), int(d2 >= 5), int(d3 >= 5), int(d4 >= 5)]


def _labels_hidden_carry(digits, planted_function):
    d1, d2 = int(digits[0]), int(digits[1])
    concepts = _carry_concepts(digits)
    a2 = int(d2 % 2 == 1)

    x = int(d1 + d2 >= 10)  # the addition carry
    task = 4 * concepts[0] + 2 * concepts[1] + x
    return concepts, [a2, x], task


def _labels_hidden_diff(digits, planted_function):
    d1, d2 = int(digits[0]), int(digits[1])
    concepts = _carry_concepts(digits)
    a2 = int(d2 % 2 == 1)

    x = int(d1 - d2 >= 2)  # corr(X,A) = +0.5, corr(X,B) = -0.5
    task = 4 * concepts[0] + 2 * concepts[1] + x
    return concepts, [a2, x], task


def _labels_hidden_carry_swap(digits, planted_function):
    d2, d3, d4 = int(digits[1]), int(digits[2]), int(digits[3])
    concepts = _carry_concepts(digits)
    a2 = int(d2 % 2 == 1)

    x = int(d3 + d4 >= 10)  # carry over the DISTRACTOR digits
    task = 4 * concepts[0] + 2 * concepts[1] + x
    return concepts, [a2, x], task


def _labels_hidden_carry_null(digits, planted_function):
    d1, d2 = int(digits[0]), int(digits[1])
    concepts = _carry_concepts(digits)
    a2 = int(d2 % 2 == 1)

    x = int(d1 + d2 >= 10)  # recorded as oracle, but absent from the target
    task = 2 * concepts[0] + concepts[1]
    return concepts, [a2, x], task


_CARRY_CONCEPT_NAMES = [
    "A::digit1_ge_5",
    "B::digit2_ge_5",
    "D1::digit3_ge_5",
    "D2::digit4_ge_5",
]

EXPERIMENT_SPECS: Dict[str, Dict] = {
    "original": {
        "num_digits": 2,
        "concept_names": [
            "A1::digit1_is_odd",
            "H1::digit1_ge_6",
            "H2::digit2_ge_6",
        ],
        "oracle_names": ["A2::digit2_is_odd", "X::planted_hidden_function"],
        "num_classes": 8,
        "label_fn": _labels_original,
        "auto_removed": [],       # concepts withheld just by picking this mode
        "protected": [],          # concepts that must stay in the bottleneck
        "corrupt_channels": (0,), # which digit channels corruption touches
        "uses_planted_function": True,
    },
    "hidden_xor": {
        "num_digits": 2,
        "concept_names": [
            "A1::digit1_is_odd",
            "H1::digit1_ge_6",
            "H2::digit2_ge_6",
        ],
        "oracle_names": ["A2::digit2_is_odd", "X::h1_xor_h2"],
        "num_classes": 2,
        "label_fn": _labels_hidden_xor,
        "auto_removed": [0],      # A1 hidden automatically
        "protected": [1, 2],      # H1, H2 must remain exposed
        "corrupt_channels": (0,),
        "uses_planted_function": False,
    },
    "hidden_carry": {
        "num_digits": 4,
        "concept_names": list(_CARRY_CONCEPT_NAMES),
        "oracle_names": ["A2::digit2_is_odd", "X::carry_d1_plus_d2_ge_10"],
        "num_classes": 8,
        "label_fn": _labels_hidden_carry,
        "auto_removed": [],
        "protected": [0, 1],      # A and B are the parents of X
        "corrupt_channels": "all",
        "uses_planted_function": False,
    },
    "hidden_diff": {
        "num_digits": 4,
        "concept_names": list(_CARRY_CONCEPT_NAMES),
        "oracle_names": ["A2::digit2_is_odd", "X::d1_minus_d2_ge_2"],
        "num_classes": 8,
        "label_fn": _labels_hidden_diff,
        "auto_removed": [],
        "protected": [0, 1],
        "corrupt_channels": "all",
        "uses_planted_function": False,
    },
    "hidden_carry_swap": {
        "num_digits": 4,
        "concept_names": list(_CARRY_CONCEPT_NAMES),
        "oracle_names": ["A2::digit2_is_odd", "X::carry_d3_plus_d4_ge_10"],
        "num_classes": 8,
        "label_fn": _labels_hidden_carry_swap,
        "auto_removed": [],
        "protected": [0, 1, 2, 3],
        "corrupt_channels": "all",
        "uses_planted_function": False,
    },
    "hidden_carry_null": {
        "num_digits": 4,
        "concept_names": list(_CARRY_CONCEPT_NAMES),
        "oracle_names": ["A2::digit2_is_odd", "X::carry_unused_by_target"],
        "num_classes": 4,
        "label_fn": _labels_hidden_carry_null,
        "auto_removed": [],
        "protected": [0, 1],
        "corrupt_channels": "all",
        "uses_planted_function": False,
    },
}

# Backwards-compatible module-level names: these describe experiment
# 'original', which is what they always described.
OBSERVED_CONCEPT_NAMES = EXPERIMENT_SPECS["original"]["concept_names"]
ORACLE_NAMES = EXPERIMENT_SPECS["original"]["oracle_names"]


def get_experiment_spec(experiment: str) -> Dict:
    return EXPERIMENT_SPECS[_normalize_experiment(experiment)]


def experiment_concept_names(experiment: str) -> List[str]:
    return list(get_experiment_spec(experiment)["concept_names"])


def experiment_num_digits(experiment: str) -> int:
    """Digit channels the experiment *needs* to build its concepts, X and label."""
    return int(get_experiment_spec(experiment)["num_digits"])


def _check_num_digits(num_digits: int, experiment: str) -> int:
    """Reject a channel count that cannot back the experiment's variables."""
    required = experiment_num_digits(experiment)
    num_digits = int(num_digits)
    if num_digits < required:
        raise ValueError(
            f"num_covariates={num_digits} gives fewer digit channels than "
            f"experiment={experiment!r} needs: its concepts, its hidden X and "
            f"its task label are all built from the first {required} digits. "
            f"Set data.num_covariates >= {required}."
        )
    return num_digits


def resolve_num_digits(config, experiment: str) -> int:
    """
    How many digit channels the input stacks -- read off `data.num_covariates`.

    The experiment fixes how many digits it *needs* (2 for original/hidden_xor,
    4 for the carry family); num_covariates fixes how many the image actually
    has. Setting it above the requirement appends extra digit channels that no
    concept, no oracle and no label reads -- pure distractor input, drawn
    independently of the digits that matter -- which is how you vary input
    dimensionality without changing the experiment's causal structure.

    Left unset it falls back to the experiment's requirement, which is what
    every run did before the channel count became configurable.
    """
    num_covariates = _cfg_get(config, "num_covariates", None)
    if num_covariates is None:
        return experiment_num_digits(experiment)
    return _check_num_digits(num_covariates, experiment)


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


def _normalize_experiment(experiment: str) -> str:
    """Normalize and validate the YAML experiment selector."""
    experiment = str(experiment).strip().lower().replace("-", "_")
    aliases = {
        "default": "original",
        "mnist_add_cov": "original",
        "xor": "hidden_xor",
        "known_concept_xor": "hidden_xor",
        "carry": "hidden_carry",
        "hidden_add_carry": "hidden_carry",
        "diff": "hidden_diff",
        "hidden_difference": "hidden_diff",
        "carry_swap": "hidden_carry_swap",
        "swap": "hidden_carry_swap",
        "carry_null": "hidden_carry_null",
        "null": "hidden_carry_null",
    }
    experiment = aliases.get(experiment, experiment)

    if experiment not in EXPERIMENT_SPECS:
        raise ValueError(
            f"Unknown MNIST-Add-Cov experiment={experiment!r}. "
            f"Choose one of: {sorted(EXPERIMENT_SPECS)}."
        )
    return experiment


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


def _balanced_digit_tuples(n_samples: int, num_digits: int, seed: int) -> np.ndarray:
    """
    Generate digit identities for `num_digits` channels, shape [n_samples, num_digits].

    num_digits == 2 delegates to _balanced_digit_pairs unchanged, so splits for
    'original' and 'hidden_xor' are bit-identical to before this file grew the
    four-digit experiments.

    Beyond that the digits are drawn in independent balanced pairs: (d1, d2) is
    exactly uniform over the 100 ordered pairs, (d3, d4) likewise, and the
    blocks are independent of each other. That independence is what makes the
    distractor concepts D1/D2 exactly uncorrelated with a hidden X built from
    d1 and d2 (and vice versa for hidden_carry_swap), and it extends to the
    padding channels a larger `data.num_covariates` asks for: block k is seeded
    from `seed` the same way whatever the total channel count, so raising
    num_covariates appends channels without disturbing the digits the
    experiment is defined on.

    An odd count drops the trailing column of the last block; its first column
    is still an exactly balanced marginal.
    """
    if num_digits == 2:
        return _balanced_digit_pairs(n_samples, seed=seed)

    if num_digits < 1:
        raise ValueError(f"num_digits must be at least 1, got {num_digits}.")

    blocks = [
        _balanced_digit_pairs(n_samples, seed=seed + block * 5_000_011)
        for block in range((num_digits + 1) // 2)
    ]
    return np.concatenate(blocks, axis=1)[:, :num_digits]


def _sample_source_indices(
    digit_tuples: np.ndarray,
    class_pools: Dict[int, np.ndarray],
    seed: int,
) -> np.ndarray:
    """
    Choose an actual MNIST image for each desired digit identity.

    Returns [n_samples, num_digits]. The RNG is consumed in the same order as
    the original two-digit implementation (all digits of sample 0, then all
    digits of sample 1, ...), so two-digit splits are unchanged.
    """
    rng = np.random.default_rng(seed)
    digit_tuples = np.atleast_2d(digit_tuples)
    n_samples, num_digits = digit_tuples.shape

    source_indices = np.empty((n_samples, num_digits), dtype=np.int64)
    for i in range(n_samples):
        for j in range(num_digits):
            source_indices[i, j] = rng.choice(class_pools[int(digit_tuples[i, j])])

    return source_indices


def _resolve_corruption_channels(channels, num_digits: int) -> Tuple[int, ...]:
    """
    Turn a corruption-channel setting into concrete channel indices.

    Accepts 'first', 'all', a single index, or an explicit list of indices.

    Which channels are corrupted matters more than it looks. Sigma is a function
    of the image (the model predicts it from the encoder features), so it
    measures how concept uncertainty co-varies with residual uncertainty *given
    that image*. A concept read off a clean, unambiguous digit has essentially
    no uncertainty, so its Sigma diagonal collapses toward zero and its
    cross-block entry is a 0/0 ratio carrying no information. For the four-digit
    experiments the distractors therefore have to be corrupted on the same terms
    as the parents, otherwise "A and B light up, D1 and D2 stay flat" is an
    artifact of who got corrupted rather than a fact about the residual.
    """
    if isinstance(channels, str):
        key = channels.strip().lower()
        if key == "first":
            return (0,)
        if key == "all":
            return tuple(range(num_digits))
        raise ValueError(
            f"Unknown corruption_channels={channels!r}. "
            "Use 'first', 'all', an index, or a list of indices."
        )

    if isinstance(channels, int):
        channels = [channels]

    resolved = sorted({int(c) for c in channels})
    for c in resolved:
        if not 0 <= c < num_digits:
            raise ValueError(
                f"corruption_channels index {c} out of range for "
                f"{num_digits} digit channels."
            )
    return tuple(resolved)


def _apply_corruption(
    image: torch.Tensor,
    corruption: str,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Corrupt a single digit image. image shape: [1, 28, 28]."""
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
        experiment: str = "original",
        planted_function: str = "xor",
        corruption: str = "none",
        corruption_strength: float = 0.0,
        corruption_probability: float = 1.0,
        manifest: Optional[Dict[str, np.ndarray]] = None,
        removed_concepts: Optional[List[int]] = None,
        corruption_channels=None,
        num_covariates: Optional[int] = None,
    ):
        super().__init__()

        self.mnist_dataset = mnist_dataset
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.experiment = _normalize_experiment(experiment)
        self.planted_function = planted_function.lower()

        # Everything experiment-specific comes from the registry.
        self.spec = EXPERIMENT_SPECS[self.experiment]

        # `required_digits` is what the experiment reads; `num_digits` is how
        # many channels the image stacks, and follows data.num_covariates. Any
        # channel beyond the requirement is a distractor: independent of the
        # concepts, of X and of the label.
        self.required_digits = int(self.spec["num_digits"])
        self.num_digits = (
            self.required_digits
            if num_covariates is None
            else _check_num_digits(num_covariates, self.experiment)
        )
        self.observed_concept_names = list(self.spec["concept_names"])
        self.oracle_concept_names = list(self.spec["oracle_names"])
        self.num_classes = int(self.spec["num_classes"])
        self._label_fn = self.spec["label_fn"]

        self.corruption_channels = _resolve_corruption_channels(
            corruption_channels
            if corruption_channels is not None
            else self.spec["corrupt_channels"],
            self.num_digits,
        )

        self.corruption = corruption.lower()
        self.corruption_strength = float(corruption_strength)
        self.corruption_probability = float(corruption_probability)

        if not (0.0 <= self.corruption_probability <= 1.0):
            raise ValueError("corruption_probability must be in [0,1].")

        # Pixels of the corrupted digit channels, when they were materialised to
        # disk alongside the manifest. They are the one part of the sample that
        # a seed alone does not pin across environments (torch RNG), so a saved
        # split stores them rather than regenerating them.
        # Shape: [N, len(corruption_channels), 28, 28].
        self.corrupted_digits: Optional[np.ndarray] = None

        # Observed concept columns dropped from the bottleneck (§ incomplete
        # variants). The split itself is untouched -- same samples, same labels,
        # fewer supervised concepts -- so an incomplete run stays sample-aligned
        # with the complete one.
        self.removed_concept_idx = sorted(int(i) for i in (removed_concepts or []))
        self.kept_concept_idx = [
            i for i in range(len(self.observed_concept_names))
            if i not in self.removed_concept_idx
        ]

        if manifest is not None:
            self._load_from_manifest(manifest)
            return

        # 1) Choose digit identities with an exactly balanced pair distribution.
        self.digit_pairs = _balanced_digit_tuples(
            self.dataset_size,
            num_digits=self.num_digits,
            seed=self.seed,
        )

        # 2) Choose actual MNIST images conditional on those identities.
        self.source_indices = _sample_source_indices(
            self.digit_pairs,
            class_pools,
            seed=self.seed + 1,
        )

        # Decide once which samples are corrupted, making access deterministic.
        rng = np.random.default_rng(self.seed + 2)
        self.corruption_mask = (
            rng.random(self.dataset_size) < self.corruption_probability
        )

        # Precompute all symbolic labels via the experiment's label function.
        n_observed = len(self.observed_concept_names)
        n_oracle = len(self.oracle_concept_names)

        self.observed_concepts = np.zeros(
            (self.dataset_size, n_observed), dtype=np.float32
        )
        self.hidden_concepts = np.zeros((self.dataset_size, n_oracle), dtype=np.float32)
        self.task_labels = np.zeros(self.dataset_size, dtype=np.int64)

        for i, digits in enumerate(self.digit_pairs):
            observed, oracle, task_label = self._label_fn(digits, self.planted_function)

            # Store the full observed-concept bank. Which columns are exposed is
            # determined by kept_concept_idx / removed_concept_idx.
            self.observed_concepts[i] = np.asarray(observed, dtype=np.float32)

            # Retained only for evaluation/validation after training.
            self.hidden_concepts[i] = np.asarray(oracle, dtype=np.float32)

            self.task_labels[i] = int(task_label)

    def _load_from_manifest(self, manifest: Dict[str, np.ndarray]) -> None:
        """Adopt a split that was generated once and written to disk."""
        self.digit_pairs = np.atleast_2d(manifest["digit_pairs"].astype(np.int64))

        if "source_indices" in manifest:
            self.source_indices = np.atleast_2d(
                manifest["source_indices"].astype(np.int64)
            )
        else:
            # Splits written before the four-digit experiments existed.
            self.source_indices = np.column_stack(
                [manifest["idx1"].astype(np.int64), manifest["idx2"].astype(np.int64)]
            )

        self.corruption_mask = manifest["corruption_mask"].astype(bool)
        self.observed_concepts = manifest["observed_concepts"].astype(np.float32)
        self.hidden_concepts = manifest["hidden_concepts"].astype(np.float32)
        self.task_labels = manifest["task_labels"].astype(np.int64)

        if "corrupted_digits" in manifest:
            self.corrupted_digits = manifest["corrupted_digits"].astype(np.float32)
        elif "corrupted_first_digit" in manifest:
            # Old key: [N, 1, 28, 28], channel 0 only.
            self.corrupted_digits = manifest["corrupted_first_digit"].astype(np.float32)

        self.dataset_size = len(self.digit_pairs)

        if self.digit_pairs.shape[1] != self.num_digits:
            raise ValueError(
                f"Split manifest holds {self.digit_pairs.shape[1]} digit columns "
                f"but this run asks for {self.num_digits} channels "
                f"(experiment={self.experiment!r}, "
                f"data.num_covariates={self.num_digits}). Set num_covariates to "
                f"{self.digit_pairs.shape[1]}, or regenerate the split."
            )
        if self.observed_concepts.shape[1] != len(self.observed_concept_names):
            raise ValueError(
                f"Split manifest holds {self.observed_concepts.shape[1]} observed "
                f"concepts but experiment={self.experiment!r} defines "
                f"{len(self.observed_concept_names)}."
            )

        for name, arr in (
            ("source_indices", self.source_indices),
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
        return [self.observed_concept_names[i] for i in self.kept_concept_idx]

    def removed_concept_names(self) -> List[str]:
        """Names of the observed concepts withheld from the bottleneck."""
        return [self.observed_concept_names[i] for i in self.removed_concept_idx]

    @property
    def idx1(self) -> np.ndarray:
        """Back-compat accessor: MNIST source index of the first digit."""
        return self.source_indices[:, 0]

    @property
    def idx2(self) -> np.ndarray:
        """Back-compat accessor: MNIST source index of the second digit."""
        return self.source_indices[:, 1]

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
                self.experiment,
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

        # Hash the source-index columns separately rather than the [N, D] block,
        # so a two-digit split hashes exactly as it did when these were stored
        # as separate idx1/idx2 arrays.
        source_columns = [
            np.ascontiguousarray(self.source_indices[:, j])
            for j in range(self.source_indices.shape[1])
        ]

        return _fingerprint_arrays(
            np.frombuffer(spec.encode(), dtype=np.uint8),
            self.digit_pairs,
            *source_columns,
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
        digits = self.digit_pairs[index]

        images = [
            self._load_mnist_tensor(self.source_indices[index, j])
            for j in range(self.num_digits)
        ]

        is_corrupted = bool(
            self.corruption_mask[index]
            and self.corruption not in {"none", "", "clean"}
            and self.corruption_strength > 0
        )

        if is_corrupted:
            for slot, channel in enumerate(self.corruption_channels):
                if self.corrupted_digits is not None:
                    # Materialised split: read the pixels back rather than
                    # re-drawing them from a torch RNG.
                    images[channel] = torch.from_numpy(
                        self.corrupted_digits[index, slot]
                    ).float().unsqueeze(0)
                else:
                    # Deterministic corruption per (sample, channel). The
                    # channel offset is chosen so channel 0 keeps the seed the
                    # two-digit experiments always used.
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

        # Single-backbone input, the digits as channels: [num_digits, 28, 28],
        # and num_digits is data.num_covariates. Channel stacking rather than
        # horizontal concatenation, to match IntCEMMNISTEncoder, whose first
        # conv takes num_covariates channels and whose projection assumes a
        # 28x28 map.
        features = torch.cat(images, dim=0)

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
            "hidden_concepts": hidden,      # [A2, X] for every experiment
            "removed_concepts": removed,    # observed concepts held out, if any
            "A2": hidden[0],
            "X": hidden[1],
            "digit_labels": torch.tensor(
                [int(d) for d in digits], dtype=torch.long
            ),
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

    experiment:                      'original' (default) or 'hidden_xor'
                                     hidden_xor automatically exposes only
                                     H1/H2 and uses X=H1 XOR H2, y=X.

    num_covariates:                  digit images stacked as channels, i.e. the
                                     input is [num_covariates, 28, 28]. Defaults
                                     to the experiment's own digit count (2 or
                                     4); a larger value appends distractor
                                     channels that no concept, oracle or label
                                     depends on. Smaller than the experiment
                                     needs is an error.

    planted_function:                'xor' (default), 'and', 'or', 'a2'
                                     Used by the original experiment only.

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
    A2 and X are always withheld from the `concepts` tensor. In hidden_xor
    mode A1 is also automatically withheld so the bottleneck is exactly [H1,H2].
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

    experiment = _normalize_experiment(_cfg_get(config, "experiment", "original"))
    planted_function = str(_cfg_get(config, "planted_function", "xor"))

    # How many digit images get stacked as channels. Follows data.num_covariates.
    num_covariates = resolve_num_digits(config, experiment)

    removed_concepts = _apply_experiment_concept_rules(
        resolve_removed_concepts(
            _cfg_get(config, "removed_concepts", None), experiment=experiment
        ),
        experiment,
    )

    corruption_channels = _cfg_get(config, "corruption_channels", None)
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
        experiment=experiment,
        planted_function=planted_function,
        removed_concepts=removed_concepts,
        corruption=corruption,
        corruption_strength=corruption_strength,
        corruption_probability=corruption_probability,
        corruption_channels=corruption_channels,
        num_covariates=num_covariates,
    )

    valset = MNISTAddCovDataset(
        mnist_dataset=mnist_train,
        class_pools=val_pools,
        dataset_size=val_dataset_size,
        seed=seed + 20,
        experiment=experiment,
        planted_function=planted_function,
        removed_concepts=removed_concepts,
        corruption=corruption,
        corruption_strength=corruption_strength,
        corruption_probability=corruption_probability,
        corruption_channels=corruption_channels,
        num_covariates=num_covariates,
    )

    testset = MNISTAddCovDataset(
        mnist_dataset=mnist_test,
        class_pools=test_pools,
        dataset_size=test_dataset_size,
        seed=seed + 30,
        experiment=experiment,
        planted_function=planted_function,
        removed_concepts=removed_concepts,
        corruption=test_corruption,
        corruption_strength=test_corruption_strength,
        corruption_probability=test_corruption_probability,
        corruption_channels=corruption_channels,
        num_covariates=num_covariates,
    )

    sync_num_concepts(config, trainset, log_file=log_file)
    sync_num_classes(config, trainset, log_file=log_file)
    sync_num_covariates(config, trainset, log_file=log_file)

    log_split_fingerprints(
        {"train": trainset, "val": valset, "test": testset},
        seed=seed,
        experiment=experiment,
        planted_function=planted_function,
        log_file=log_file,
    )

    return trainset, valset, testset


SPLIT_ROOT = "splits"
MANIFEST_KEYS = (
    "digit_pairs",
    "source_indices",
    "corruption_mask",
    "observed_concepts",
    "hidden_concepts",
    "task_labels",
)

# Which underlying MNIST partition each split draws its images from.
SPLIT_MNIST_SOURCE = {"train": "train", "val": "train", "test": "test"}


def _apply_experiment_concept_rules(removed_concepts: List[int], experiment: str) -> List[int]:
    """
    Fold the experiment's own bottleneck rules into the removal list.

    Each spec declares `auto_removed` (concepts hidden simply by selecting the
    experiment, e.g. A1 in hidden_xor) and `protected` (concepts the experiment
    is defined around and that therefore must stay exposed).
    """
    spec = EXPERIMENT_SPECS[_normalize_experiment(experiment)]
    names = spec["concept_names"]

    clashes = [i for i in spec["protected"] if i in removed_concepts]
    if clashes:
        raise ValueError(
            f"experiment={experiment!r} requires "
            f"{[names[i].split('::')[0] for i in spec['protected']]} to remain "
            f"exposed, but data.removed_concepts removes "
            f"{[names[i].split('::')[0] for i in clashes]}."
        )

    return sorted(set(removed_concepts) | set(spec["auto_removed"]))


def resolve_removed_concepts(removed, experiment: str = "original") -> List[int]:
    """
    Turn a config entry into observed-concept column indices.

    Accepts indices, full names ("A1::digit1_is_odd") or short names ("A1"),
    so the config can read either way. The valid names depend on the selected
    experiment, hence the `experiment` argument (defaulting to 'original', which
    is what this function always assumed).
    """
    observed_names = experiment_concept_names(experiment)

    if removed is None:
        return []
    if isinstance(removed, (int, str)):
        removed = [removed]

    short_names = [name.split("::")[0] for name in observed_names]

    indices = []
    for entry in removed:
        if isinstance(entry, bool):
            raise ValueError(f"Invalid removed_concepts entry: {entry!r}")
        if isinstance(entry, int):
            index = entry
        elif entry in observed_names:
            index = observed_names.index(entry)
        elif entry in short_names:
            index = short_names.index(entry)
        else:
            raise ValueError(
                f"Unknown concept {entry!r} for experiment={experiment!r}. "
                f"Use an index in [0, {len(observed_names) - 1}] or one of "
                f"{observed_names} / {short_names}."
            )
        if not 0 <= index < len(observed_names):
            raise ValueError(
                f"removed_concepts index {index} out of range for "
                f"{len(observed_names)} observed concepts in "
                f"experiment={experiment!r}."
            )
        indices.append(index)

    if len(set(indices)) == len(observed_names):
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


def sync_num_classes(config, dataset: "MNISTAddCovDataset", log_file=None) -> None:
    """Keep data.num_classes aligned with the selected experiment."""
    num_classes = dataset.num_classes
    configured = _cfg_get(config, "num_classes", num_classes)

    if configured != num_classes:
        message = (
            f"MNIST-Add-Cov experiment={dataset.experiment}: "
            f"num_classes {configured} -> {num_classes}"
        )
        print(message)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(message + "\n")

    try:
        config.num_classes = num_classes
    except Exception:
        pass


def sync_num_covariates(config, dataset: "MNISTAddCovDataset", log_file=None) -> None:
    """
    Write back the channel count the split was actually built with.

    The direction here is config -> data: `data.num_covariates` decides how many
    digit channels get stacked, and the dataset already honoured it. This only
    fills the key in when it was left unset (falling back to the experiment's
    own digit count), so IntCEMMNISTEncoder -- whose first conv is built with
    in_channels=data.num_covariates -- always matches the tensors it is fed.
    """
    num_covariates = dataset.num_digits
    configured = _cfg_get(config, "num_covariates", None)

    if configured is not None and int(configured) != num_covariates:
        message = (
            f"MNIST-Add-Cov experiment={dataset.experiment}: "
            f"num_covariates {configured} -> {num_covariates}"
        )
        print(message)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(message + "\n")

    try:
        config.num_covariates = num_covariates
    except Exception:
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
        f"_{train.experiment}"
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
            # files plus the manifest, so store them explicitly -- one plane per
            # corrupted channel, in the order of dataset.corruption_channels.
            channels = list(dataset.corruption_channels)
            arrays["corrupted_digits"] = np.stack(
                [
                    dataset[i]["features"][channels].numpy()
                    for i in range(len(dataset))
                ]
            ).astype(np.float32)
            arrays["corruption_channels"] = np.asarray(channels, dtype=np.int64)

        np.savez_compressed(os.path.join(split_dir, "manifest.npz"), **arrays)

        meta = {
            "split": split,
            "mnist_source": SPLIT_MNIST_SOURCE[split],
            "dataset_size": len(dataset),
            "seed": dataset.seed,
            "experiment": dataset.experiment,
            "planted_function": dataset.planted_function,
            "num_digits": dataset.num_digits,
            "required_digits": dataset.required_digits,
            "corruption_channels": list(dataset.corruption_channels),
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
        f.write(f"experiment: {train.experiment}\n")
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
        f.write(
            f"num digit channels: {train.num_digits} "
            f"(experiment uses the first {train.required_digits})\n"
        )
        f.write(f"observed concepts (full): {train.observed_concept_names}\n")
        f.write(f"oracle variables: {train.oracle_concept_names}\n")
        f.write(f"corrupted channels: {list(train.corruption_channels)}\n")
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
    requested_experiment = _normalize_experiment(
        _cfg_get(config, "experiment", "original")
    )
    removed_concepts = _apply_experiment_concept_rules(
        resolve_removed_concepts(
            _cfg_get(config, "removed_concepts", None),
            experiment=requested_experiment,
        ),
        requested_experiment,
    )
    num_covariates = resolve_num_digits(config, requested_experiment)

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

        saved_experiment = _normalize_experiment(meta.get("experiment", "original"))
        if saved_experiment != requested_experiment:
            raise ValueError(
                f"Saved split experiment={saved_experiment!r} but YAML requests "
                f"experiment={requested_experiment!r}. Use the matching split."
            )

        # A materialised split fixes its own channel count, so num_covariates
        # has to agree with it rather than reshape it.
        saved_num_digits = int(
            meta.get("num_digits", np.atleast_2d(manifest["digit_pairs"]).shape[1])
        )
        if saved_num_digits != num_covariates:
            raise ValueError(
                f"Saved split {data_dir_name!r} stacks {saved_num_digits} digit "
                f"channels but data.num_covariates={num_covariates}. Set "
                f"num_covariates={saved_num_digits}, or regenerate the split "
                f"with data.save_data=True."
            )

        dataset = MNISTAddCovDataset(
            num_covariates=num_covariates,
            mnist_dataset=mnist_by_source[meta["mnist_source"]],
            class_pools=None,
            dataset_size=meta["dataset_size"],
            seed=meta["seed"],
            experiment=saved_experiment,
            planted_function=meta["planted_function"],
            corruption=meta["corruption"],
            corruption_strength=meta["corruption_strength"],
            corruption_probability=meta["corruption_probability"],
            manifest=manifest,
            removed_concepts=removed_concepts,
            corruption_channels=meta.get("corruption_channels"),
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
        seed=trainset.seed - 10,
        experiment=trainset.experiment,
        planted_function=trainset.planted_function,
        log_file=log_file,
    )

    return trainset, valset, testset


def log_split_fingerprints(
    splits: Dict[str, "MNISTAddCovDataset"],
    seed: int,
    experiment: str,
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
        f"(data seed={seed}, experiment={experiment}, "
        f"planted_function={planted_function}):"
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


def get_mnist_add_cov_concept_names(experiment: str = "original") -> List[str]:
    return experiment_concept_names(experiment)


def get_mnist_add_cov_oracle_names(experiment: str = "original") -> List[str]:
    return list(get_experiment_spec(experiment)["oracle_names"])


def summarize_dataset(dataset: MNISTAddCovDataset) -> None:
    """
    Diagnostic checks for the selected experiment.

    For the four-digit experiments this is the sanity check to run *before*
    training: it prints the marginal concept-X correlations the whole covariance
    argument rests on, and confirms X is not determined by the exposed concepts
    (if it were, a linear head would compute X directly and the residual would
    stay empty).
    """
    concepts = dataset.observed_concepts
    hidden = dataset.hidden_concepts
    labels = dataset.task_labels
    names = [n.split("::")[0] for n in dataset.observed_concept_names]

    print(f"N={len(dataset)}")
    print(f"Experiment: {dataset.experiment}")
    print(
        f"Digit channels: {dataset.num_digits} "
        f"({dataset.required_digits} used by the experiment, "
        f"{dataset.num_digits - dataset.required_digits} distractor)"
    )
    print(f"Concepts exposed: {dataset.concept_names()}")
    print(f"Observed concept means {names}: {concepts.mean(axis=0)}")
    print(f"Oracle means [A2,X]: {hidden.mean(axis=0)}")
    print(
        f"Task class counts: "
        f"{np.bincount(labels, minlength=dataset.num_classes)}"
    )

    x = hidden[:, 1]

    if dataset.experiment == "hidden_xor":
        h1, h2 = concepts[:, 1], concepts[:, 2]
        print(f"Corr(H1, X): {np.corrcoef(h1, x)[0, 1]:+.4f}")
        print(f"Corr(H2, X): {np.corrcoef(h2, x)[0, 1]:+.4f}")
        print(
            "  ^ both near zero BY CONSTRUCTION: parity is pairwise "
            "independent of its parents, so the marginal cross-block carries "
            "no signal here. Analyse this one conditionally instead."
        )
        for h1_value in (0, 1):
            for h2_value in (0, 1):
                mask = (h1 == h1_value) & (h2 == h2_value)
                print(
                    f"H1={h1_value}, H2={h2_value}: "
                    f"n={mask.sum()}, P(X=1)={x[mask].mean():.4f}"
                )
        return

    if dataset.required_digits == 4:
        print("Marginal concept-X correlations (the covariance target):")
        for j, name in enumerate(names):
            print(f"  Corr({name}, X) = {np.corrcoef(concepts[:, j], x)[0, 1]:+.4f}")

        print("Is X determined by the exposed concepts?")
        determined = 0
        for a_value in (0, 1):
            for b_value in (0, 1):
                mask = (concepts[:, 0] == a_value) & (concepts[:, 1] == b_value)
                if mask.sum() == 0:
                    continue
                p = x[mask].mean()
                flag = "  <- determined" if p in (0.0, 1.0) else ""
                determined += mask.sum() if p in (0.0, 1.0) else 0
                print(
                    f"  A={a_value}, B={b_value}: n={mask.sum()}, "
                    f"P(X=1)={p:.4f}{flag}"
                )
        frac = determined / len(dataset)
        print(
            f"  X is pinned by (A,B) on {frac:.1%} of samples; the residual is "
            f"strictly required on the remaining {1 - frac:.1%}."
        )
        return

    a2, h2 = hidden[:, 0], concepts[:, 2]
    for h2_value in (0, 1):
        mask = h2 == h2_value
        print(f"P(A2=1 | H2={h2_value}) = {a2[mask].mean():.4f}  (target 0.5)")


if __name__ == "__main__":
    # Small standalone smoke test over every registered experiment.
    import sys

    class Config:
        data_path = "./data"
        train_dataset_size = 2000
        val_dataset_size = 400
        test_dataset_size = 2000
        val_percent = 0.2

        experiment = "hidden_carry"
        planted_function = "xor"

        corruption = "none"
        corruption_strength = 0.0
        corruption_probability = 1.0

    requested = sys.argv[1:] or sorted(EXPERIMENT_SPECS)

    for name in requested:
        config = Config()
        config.experiment = name

        print("\n" + "=" * 70)
        trainset, valset, testset = get_MNIST_add_cov_datasets(config, seed=42)
        summarize_dataset(trainset)

        sample = trainset[0]
        print("features:", tuple(sample["features"].shape))
        print("concepts:", sample["concepts"].tolist())
        print("hidden_concepts [A2,X]:", sample["hidden_concepts"].tolist())
        print("digit_labels:", sample["digit_labels"].tolist())
        print("label:", int(sample["labels"]))
        print(
            f"synced config: num_concepts={config.num_concepts}, "
            f"num_classes={config.num_classes}, "
            f"num_covariates={config.num_covariates}"
        )
