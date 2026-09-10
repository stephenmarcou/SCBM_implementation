"""
Create CIFAR-10/CIFAR-100 concept labels using CLIP and median thresholding.

This version follows the CIFAR-10 construction used by Espinosa Zarlenga et al.
(MixCEM / "Avoiding Leakage Poisoning"). For each image and textual concept,
we compute a CLIP cosine-similarity score. Each concept is then binarised using
its 50th-percentile score estimated over the entire dataset (train + test).

Expected input:
    <DATA_ROOT>/<cifar>_filtered.txt
        One textual concept per line (143 concepts for the CIFAR-10 setup).

Outputs:
    <DATA_ROOT>/<cifar>_train_concept_labels.pt
    <DATA_ROOT>/<cifar>_test_concept_labels.pt
    <DATA_ROOT>/<cifar>_concept_thresholds.pt

The saved train/test concept-label tensors have dtype torch.bool and shape
[num_samples, num_concepts].
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from transformers import CLIPModel, CLIPProcessor


cifar = "cifar10"  # Set to "cifar10" or "cifar100"
batch_size = 128
num_workers = 0  # Safe default on macOS; increase on Linux if desired.

DATA_ROOT = Path(__file__).resolve().parent / cifar

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

pin_memory = device.type == "cuda"


class CLIPImageTransform:
    """Convert a PIL image to the CLIP pixel tensor expected by CLIPModel."""

    def __init__(self, processor: CLIPProcessor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)


def load_concepts():
    concept_file = DATA_ROOT / f"{cifar}_filtered.txt"
    with open(concept_file, "r") as file:
        concepts = [line.strip() for line in file if line.strip()]

    print(f"Loaded {len(concepts)} concepts from {concept_file}")
    if cifar == "cifar10" and len(concepts) != 143:
        print(
            "WARNING: The MixCEM/SCBM CIFAR-10 setup uses 143 concepts, "
            f"but this file contains {len(concepts)}."
        )
    return concepts


def get_dataset(train: bool, transform):
    if cifar == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=str(DATA_ROOT),
            train=train,
            transform=transform,
            download=True,
        )
    if cifar == "cifar100":
        return torchvision.datasets.CIFAR100(
            root=str(DATA_ROOT),
            train=train,
            transform=transform,
            download=True,
        )
    raise ValueError(f"Unsupported dataset: {cifar}")


def compute_text_embeddings(model, processor, concepts):
    """Compute one L2-normalised CLIP embedding for each textual concept."""
    text_inputs = processor(
        text=concepts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

    return text_embeddings


def compute_concept_scores(model, dataset, text_embeddings, split_name):
    """Return [num_images, num_concepts] CLIP cosine-similarity scores."""
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    score_batches = []

    with torch.no_grad():
        for batch_idx, (pixel_values, _) in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(
                    f"{split_name}: batch {batch_idx}/{len(data_loader)} "
                    f"({100.0 * batch_idx / len(data_loader):.1f}%)"
                )

            pixel_values = pixel_values.to(device, non_blocking=pin_memory)

            image_embeddings = model.get_image_features(pixel_values=pixel_values)
            image_embeddings = F.normalize(image_embeddings, dim=-1)

            # Normalised dot product = cosine similarity.
            similarities = image_embeddings @ text_embeddings.T
            score_batches.append(similarities.cpu())

    scores = torch.cat(score_batches, dim=0)
    print(f"{split_name} score tensor: {tuple(scores.shape)}")
    return scores


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    concepts = load_concepts()

    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()

    # Unlike the original SCBM construction, we use only the positive concept
    # descriptions. There are no "not <concept>" prompts in this version.
    text_embeddings = compute_text_embeddings(model, processor, concepts)

    transform = CLIPImageTransform(processor)
    train_dataset = get_dataset(train=True, transform=transform)
    test_dataset = get_dataset(train=False, transform=transform)

    # First compute continuous CLIP scores for both splits.
    train_scores = compute_concept_scores(
        model, train_dataset, text_embeddings, split_name="train"
    )
    test_scores = compute_concept_scores(
        model, test_dataset, text_embeddings, split_name="test"
    )

    # MixCEM construction: estimate each concept's 50th-percentile threshold
    # from the ENTIRE CIFAR dataset (train + test), then use the same threshold
    # to binarise both splits.
    all_scores = torch.cat([train_scores, test_scores], dim=0)
    thresholds = torch.quantile(all_scores, q=0.5, dim=0)

    train_concept_labels = train_scores > thresholds
    test_concept_labels = test_scores > thresholds

    train_path = DATA_ROOT / f"{cifar}_train_concept_labels.pt"
    test_path = DATA_ROOT / f"{cifar}_test_concept_labels.pt"
    threshold_path = DATA_ROOT / f"{cifar}_concept_thresholds.pt"

    torch.save(train_concept_labels, train_path)
    torch.save(test_concept_labels, test_path)
    torch.save(thresholds, threshold_path)

    # Sanity checks. Across train + test, concept prevalence should be close
    # to 0.5 for every concept (subject to possible ties at the median).
    all_labels = torch.cat([train_concept_labels, test_concept_labels], dim=0)
    prevalence = all_labels.float().mean(dim=0)

    print("\nSaved:")
    print(f"  {train_path}  shape={tuple(train_concept_labels.shape)}")
    print(f"  {test_path}  shape={tuple(test_concept_labels.shape)}")
    print(f"  {threshold_path}  shape={tuple(thresholds.shape)}")
    print(
        "Overall concept prevalence: "
        f"min={prevalence.min().item():.4f}, "
        f"mean={prevalence.mean().item():.4f}, "
        f"max={prevalence.max().item():.4f}"
    )


if __name__ == "__main__":
    main()
