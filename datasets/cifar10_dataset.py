import os
import numpy as np
import torch
from torchvision import datasets, transforms

TRAIN_TF = transforms.Compose([
    transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TF = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CIFAR10_CBM_dataloader(datasets.CIFAR10):
    def __init__(self, root, split, concept_file, indices=None, download=False):
        super().__init__(root=root, train=(split != "test"),
                         transform=TRAIN_TF if split == "train" else EVAL_TF,
                         download=download)
        concepts = torch.load(os.path.join(root, concept_file), map_location="cpu").float()
        if indices is not None:
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]
            concepts = concepts[indices]
        assert len(concepts) == len(self.data), (len(concepts), len(self.data))
        self.concepts = concepts

    def __getitem__(self, idx):
        X, target = super().__getitem__(idx)
        return {"img_code": idx, "labels": target,
                "features": X, "concepts": self.concepts[idx]}


def get_CIFAR10_CBM_dataloader(datapath):
    # The folder is 'cifar10' on disk on both the laptop and the cluster; the capitalised
    # spelling only ever resolved because macOS is case-insensitive, so it failed on Linux.
    root = os.path.join(datapath, "cifar10")
    if not os.path.isdir(root):
        root = os.path.join(datapath, "CIFAR10")
    train_idxs = np.load(os.path.join(root, "train_idxs.npy"))
    val_idxs = np.load(os.path.join(root, "val_idxs.npy"))
    train_file = "c_train_percentile_threshold_bool.pt"
    test_file = "c_test_percentile_threshold_bool.pt"

    return (
        CIFAR10_CBM_dataloader(root, "train", train_file, indices=train_idxs),
        CIFAR10_CBM_dataloader(root, "val", train_file, indices=val_idxs),
        CIFAR10_CBM_dataloader(root, "test", test_file),
    )