import random
from functools import partial
from pathlib import Path
from typing import *

import timm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from pytorch_lightning import seed_everything
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_dataset(split: str, perc: float = 1, val_perc: float = 0.15):

    seed_everything(42)
    assert 0 < perc <= 1
    assert 0 < val_perc < 1
    dataset = load_dataset(dataset_name)[split]

    # If the split is 'train', further split into train and validation sets
    if split == "train":
        # Shuffle and select a random subset for the entire training set
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: int(len(indices) * perc)]

        # Calculate lengths and indices for train and validation subsets
        val_len = int(len(indices) * val_perc)
        train_len = len(indices) - val_len

        train_indices = indices[:train_len]
        val_indices = indices[train_len:]

        # Create train and validation subsets
        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)

        return train_dataset, val_dataset

    else:
        # For other splits, you can apply the original random subset logic
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: int(len(indices) * perc)]
        dataset = dataset.select(indices)

        return dataset


def relative_projection(x, anchors, phi=None):
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)
    if phi is None:
        return torch.einsum("bm, am -> ba", x, anchors)
    else:
        return torch.einsum("bm, mm, am -> ba", x, phi, anchors)


def load_transformer(transformer_name):
    transformer = timm.create_model(transformer_name, pretrained=True, num_classes=0)
    return transformer.requires_grad_(False).eval()


@torch.no_grad()
def call_transformer(batch, transformer):
    sample_encodings = transformer(batch["encoding"].to(device))
    return {"hidden": sample_encodings}


def get_latents(
    dataloader, split: str, transformer
) -> Dict[str, torch.Tensor]:
    absolute_latents: List = []
    labels: List = []

    transformer = transformer.to(device)
    for batch in tqdm(dataloader, desc=f"[{split}] Computing latents"):
        with torch.no_grad():
            transformer_out = call_transformer(batch=batch, transformer=transformer)

            absolute_latents.append(transformer_out["hidden"].cpu())

            labels.append(batch["label"].cpu())

    absolute_latents: torch.Tensor = torch.cat(absolute_latents, dim=0).cpu()
    labels: torch.Tensor = torch.cat(labels, dim=0).cpu()

    transformer = transformer.cpu()
    return {
        "absolute": absolute_latents,
        "labels": labels,
    }


def collate_fn(batch, feature_extractor, transform):
    return {
        "encoding": torch.stack(
            [transform(sample["img"].convert("RGB")) for sample in batch], dim=0
        ),
        "label": torch.tensor([sample["label"] for sample in batch]),
    }


def encode_latents(transformer_names: Sequence[str], dataset, transformer_name2latents, split: str):
    for transformer_name in transformer_names:
        # Load the transformer model
        transformer = load_transformer(transformer_name=transformer_name)

        # Create a transform for the data based on the transformer's requirements
        config = resolve_data_config({}, model=transformer)
        transform = create_transform(**config)

        # Process the main dataset
        dataset_latents_output = get_latents(
            dataloader=DataLoader(
                dataset,
                num_workers=0,
                pin_memory=True,
                collate_fn=partial(
                    collate_fn, feature_extractor=None, transform=transform
                ),
                batch_size=32,
            ),
            split=f"{split}/{transformer_name}",
            transformer=transformer,
        )

        # Store the latents and labels
        transformer_name2latents[transformer_name] = {
            **dataset_latents_output,
        }

        # Save latents and labels if caching is enabled
        if CACHE_LATENTS:
            print(f"Saving latents and labels for {transformer_name}...")
            transformer_path = (
                LATENTS_DIR / split / f"{transformer_name.replace('/', '-')}.pt"
            )
            transformer_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(transformer_name2latents[transformer_name], transformer_path)


def load_latents(split: str, transformer_names: Sequence[str]):
    transformer2latents = {}

    for transformer_name in transformer_names:
        transformer_path = (
            LATENTS_DIR / split / f"{transformer_name.replace('/', '-')}.pt"
        )
        if transformer_path.exists():
            transformer2latents[transformer_name] = torch.load(transformer_path)

    return transformer2latents


####################################################################################################
####################################################################################################


if __name__ == "__main__":
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "cifar10"
    transformer_names = list(
        {
            
            #"vit_small_patch16_224",
            #"vit_base_patch16_224",
        }
    )


    CACHE_LATENTS: bool = True

    # Data
    LATENTS_DIR = Path(f"./data/latents/{dataset_name}")
    LATENTS_DIR.mkdir(exist_ok=True, parents=True)

    train_dataset, val_dataset = get_dataset(
        split="train"
    )  

    print(len(train_dataset), len(val_dataset))

    test_dataset = get_dataset(split="test")
    print(len(test_dataset))

    seed_everything(42)


    # Compute train latents
    FORCE_RECOMPUTE: bool = False
    CACHE_LATENTS: bool = True

    transformer2train_latents: Dict[str, Mapping[str, torch.Tensor]] = load_latents(
        split="train", transformer_names=transformer_names
    )
    missing_transformers = (
        transformer_names
        if FORCE_RECOMPUTE
        else [
            t_name
            for t_name in transformer_names
            if t_name not in transformer2train_latents
        ]
    )
    encode_latents(
        transformer_names=missing_transformers,
        dataset=train_dataset,
        transformer_name2latents=transformer2train_latents,
        split="train",
    )

    # Compute val latents

    FORCE_RECOMPUTE: bool = False
    CACHE_LATENTS: bool = True

    transformer2val_latents: Dict[str, Mapping[str, torch.Tensor]] = load_latents(
        split="val", transformer_names=transformer_names
    )
    missing_transformers = (
        transformer_names
        if FORCE_RECOMPUTE
        else [
            t_name
            for t_name in transformer_names
            if t_name not in transformer2val_latents
        ]
    )
    encode_latents(
        transformer_names=missing_transformers,
        dataset=val_dataset,
        transformer_name2latents=transformer2val_latents,
        split="val",
    )

    # Compute test latents
    FORCE_RECOMPUTE: bool = False
    CACHE_LATENTS: bool = True

    transformer2test_latents: Dict[str, Mapping[str, torch.Tensor]] = load_latents(
        split="test", transformer_names=transformer_names
    )
    missing_transformers = (
        transformer_names
        if FORCE_RECOMPUTE
        else [
            t_name
            for t_name in transformer_names
            if t_name not in transformer2test_latents
        ]
    )
    encode_latents(
        transformer_names=missing_transformers,
        dataset=test_dataset,
        transformer_name2latents=transformer2test_latents,
        split="test",
    )
