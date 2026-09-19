"""Input corruptions for post-hoc robustness checks (inference.noise).

Nothing here touches training: a saved checkpoint is evaluated / intervened on against a
corrupted copy of one split, so the curve on the clean split and the curve on the noisy one
differ only in the pixels the model sees. The noise is applied in [0, 1] pixel space, after
the geometric test-time transform (CenterCrop + Resize + ToTensor) and *before* the ImageNet
normalization, so that "salt" really is a white pixel and "pepper" a black one rather than a
value in normalized units.

The noise is deterministic per (seed, sample index): the same command reproduces the same
corrupted images regardless of the number of workers, and two model checkpoints swept with
the same seed see the same corrupted pixels.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Amounts are interpreted per type:
#   salt_pepper: fraction of pixels replaced (half by white, half by black), shared across
#                the RGB channels so a corrupted pixel is fully white or fully black.
#   gaussian:    standard deviation of additive N(0, amount^2) noise in [0, 1] pixel units,
#                clipped back to [0, 1].
NOISE_TYPES = ("salt_pepper", "gaussian")


def apply_image_noise(img, noise_type, amount, generator):
    """Corrupt one image tensor (C, H, W) with values in [0, 1]. Returns a new tensor."""
    if noise_type == "salt_pepper":
        # One mask per pixel, broadcast over channels.
        corrupt = torch.rand(img.shape[-2:], generator=generator) < amount
        salt = torch.rand(img.shape[-2:], generator=generator) < 0.5
        img = torch.where(corrupt & salt, torch.ones_like(img), img)
        img = torch.where(corrupt & ~salt, torch.zeros_like(img), img)
        return img
    if noise_type == "gaussian":
        noise = torch.randn(img.shape, generator=generator) * amount
        return (img + noise).clamp_(0.0, 1.0)
    raise ValueError(f"Unknown noise type {noise_type!r}, expected one of {list(NOISE_TYPES)}.")


def split_normalization(transform):
    """Split a Compose into (everything before Normalize, the Normalize step).

    Lets a noisy loader reuse the dataset's own test transform verbatim, so the geometric
    preprocessing and the normalization constants stay in sync with the clean evaluation
    instead of being copied here.
    """
    steps = list(transform.transforms)
    normalize_steps = [t for t in steps if isinstance(t, transforms.Normalize)]
    if len(normalize_steps) != 1:
        raise ValueError(
            f"Expected exactly one Normalize step in the test transform, found {len(normalize_steps)}."
        )
    unnormalized = transforms.Compose([t for t in steps if not isinstance(t, transforms.Normalize)])
    return unnormalized, normalize_steps[0]


class NoisyImageDataset(Dataset):
    """Wraps a dataset whose 'features' are un-normalized [0, 1] images: adds noise, then normalizes.

    Every other key of the item (labels, concepts, img_code) is passed through untouched, so
    the wrapped loader is a drop-in replacement for the clean one in evaluation and in the
    intervention sweep.
    """

    def __init__(self, dataset, noise_type, amount, seed, normalize):
        if noise_type not in NOISE_TYPES:
            raise ValueError(
                f"inference.noise.type must be one of {list(NOISE_TYPES)}, got {noise_type!r}."
            )
        if amount <= 0:
            raise ValueError(f"inference.noise.amount must be > 0, got {amount}.")
        if noise_type == "salt_pepper" and amount > 1:
            raise ValueError(
                f"inference.noise.amount is a pixel fraction for salt_pepper, got {amount}."
            )
        self.dataset = dataset
        self.noise_type = noise_type
        self.amount = float(amount)
        self.seed = int(seed)
        self.normalize = normalize

    def __getitem__(self, index):
        item = self.dataset[index]
        # Disjoint stream per (seed, index): a different seed never reuses another seed's
        # per-sample streams, and the corruption of sample i does not depend on batch order.
        generator = torch.Generator().manual_seed(self.seed * len(self.dataset) + index)
        noisy = apply_image_noise(item["features"], self.noise_type, self.amount, generator)
        item["features"] = self.normalize(noisy)
        return item

    def __len__(self):
        return len(self.dataset)

    @property
    def data(self):
        """The wrapped split's records, so callers that read loader.dataset.data still work."""
        return self.dataset.data
