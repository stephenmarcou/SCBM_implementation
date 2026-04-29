"""
CIFAR-100 dataset loader with concept labels. Relies on create_dataset_cifar.py to have generated concept labels.

This module provides a custom DataLoader for the CIFAR-100 dataset, including concept labels for training, validation, and testing.
The dataset is preprocessed with transformations.

Classes:
    CIFAR100_CBM_dataloader: Custom DataLoader for CIFAR-100 with concept labels.

Functions:
    get_CIFAR100_CBM_dataloader: Returns DataLoaders for training, validation, and testing splits.
"""

import os
import pickle

import ctypes
from PIL import Image
import numpy as np
import multiprocessing as mp

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.nn.functional as F


def get_CIFAR100_CBM_dataloader(datapath, gen, val_ratio=0.1, use_full_train_after_tuning=False):
    datapath = datapath + "cifar100/"
    print("Loading CIFAR-100 dataset from: " + datapath)
    print("Caching enabled - preprocessed images will be cached in memory")
    # Load full training set
    train_full = CIFAR100_CBM_dataloader(
        root=datapath,
        train=True,
        download=False,
        cache=True,
    )

    if use_full_train_after_tuning:
        # Final training phase: train on 100% of the official training set.
        # Validation metrics are then computed on the same split and should not be used for model selection.
        train_data = train_full
        val_data = train_full
    else:
        # Hyperparameter tuning phase: hold out part of training data for validation.
        train_size = int((1 - val_ratio) * len(train_full))
        val_size = len(train_full) - train_size
        train_data, val_data = random_split(
            train_full,
            [train_size, val_size],
            generator=gen,
        )
    
    # Test set stays as-is (not cached - only evaluated once)
    test_data = CIFAR100_CBM_dataloader(
        root=datapath,
        train=False,
        download=False,
        cache=False,
    )

    return train_data, val_data, test_data


class CIFAR100_CBM_dataloader(datasets.CIFAR100):

    def __init__(self, cache=False, *args, **kwargs):
        super(CIFAR100_CBM_dataloader, self).__init__(*args, **kwargs)

        self.cache = cache
        self.cache_hits = 0
        self.cache_misses = 0

        if kwargs["train"]:
            self.transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # implicitly divides by 255
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),  # implicitly divides by 255
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Load concepts from CIFAR-100 dataset which correspond to 20 coarse labels
        split_name = "train" if kwargs["train"] else "test"
        with open(os.path.join(kwargs["root"], "cifar-100-python", split_name), "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            coarse_labels = torch.as_tensor(entry["coarse_labels"], dtype=torch.long)

            self.num_concepts = len(set(entry["coarse_labels"]))
            # Concepts shape [num_samples, 20] with one-hot encoding
            self.concepts = F.one_hot(coarse_labels, num_classes=self.num_concepts).float()

        # Setup shared-memory caches if requested (store raw uint8 images)
        if self.cache:
            num_samples = len(self.data)

            # CIFAR images are 32x32 RGB
            max_height = 32
            max_width = 32
            data_dims = (3, max_height, max_width)
            dimension = int(np.prod(data_dims))

            # Create shared array for image data (padded to max size)
            shared_array_base = mp.Array(
                ctypes.c_uint8, num_samples * dimension
            )
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            shared_array = shared_array.reshape(num_samples, *data_dims)
            self.image_cache = torch.from_numpy(shared_array)

            # Create shared array for image dimensions and validity
            # Format: [height, width] per image, initialized to [-1, -1] (invalid)
            dims_array_base = mp.Array(
                ctypes.c_int, num_samples * 2
            )
            dims_array = np.ctypeslib.as_array(dims_array_base.get_obj())
            dims_array = dims_array.reshape(num_samples, 2)
            self.dims_cache = torch.from_numpy(dims_array)
            self.dims_cache.fill_(-1)

            # Create shared array for class labels
            label_array_base = mp.Array(
                ctypes.c_int, num_samples
            )
            label_array = np.ctypeslib.as_array(label_array_base.get_obj())
            self.label_cache = torch.from_numpy(label_array)
            self.label_cache.fill_(-1)

    def _is_cached(self, index):
        if not self.cache:
            return False
        return int(self.dims_cache[index][0]) != -1 and int(self.dims_cache[index][1]) != -1

    def _cache_image(self, index, image_pil, label):
        img_array = np.array(image_pil)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        h, w = img_array.shape[:2]

        # store dims
        self.dims_cache[index] = torch.tensor([h, w])

        # convert to CHW uint8
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(torch.uint8)

        # pad to 32x32
        padded = torch.zeros((3, 32, 32), dtype=torch.uint8)
        padded[:, :h, :w] = img_tensor[:, :h, :w]
        self.image_cache[index] = padded

        # label
        self.label_cache[index] = int(label)

    def _get_cached_image(self, index):
        h, w = self.dims_cache[index]
        h, w = int(h), int(w)
        img_tensor = self.image_cache[index][:, :h, :w]
        img_array = img_tensor.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray(img_array)
        label = int(self.label_cache[index])
        return image_pil, label


    def __getitem__(self, idx):
        # Diagnostic prints
        if self.cache_hits != 0 and self.cache_hits % 5000 == 0:
            print(f"Cache hits: {self.cache_hits}")
        if self.cache_misses != 0 and self.cache_misses % 5000 == 0:
            print(f"Cache misses: {self.cache_misses}")

        if self.cache:
            if self._is_cached(idx):
                self.cache_hits += 1
                image_pil, label = self._get_cached_image(idx)
                features = self.transform(image_pil) if self.transform is not None else image_pil
                return {
                    "img_code": idx,
                    "labels": label,
                    "features": features,
                    "concepts": self.concepts[idx],
                }
            else:
                self.cache_misses += 1
                img_array = self.data[idx]
                image_pil = Image.fromarray(img_array)
                label = self.targets[idx]
                self._cache_image(idx, image_pil, label)
                features = self.transform(image_pil) if self.transform is not None else image_pil
                return {
                    "img_code": idx,
                    "labels": label,
                    "features": features,
                    "concepts": self.concepts[idx],
                }

        X, target = super().__getitem__(idx)
        return {
            "img_code": idx,
            "labels": target,
            "features": X,
            "concepts": self.concepts[idx],
        }
