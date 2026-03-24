import json
import os
import random
from typing import Dict, List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient, models


def sum_compose(
    img_emb: torch.Tensor, text_emb: torch.Tensor, alpha: float = 1.0, beta: float = 0.5
):
    composed = alpha * img_emb + beta * text_emb
    composed = F.normalize(composed, dim=-1)

    return composed


def haversine(gps1: list | tuple | np.ndarray, gps2: list | tuple | np.ndarray):
    if not isinstance(gps1, np.ndarray):
        gps1 = np.array(gps1)
    if not isinstance(gps2, np.ndarray):
        gps2 = np.array(gps2)

    gps1 = np.atleast_2d(gps1)
    gps2 = np.atleast_2d(gps2)

    lat1, lon1 = np.radians(gps1).T
    lat2, lon2 = np.radians(gps2).T

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    dist = 6371 * c
    if dist.size == 1:
        return dist.item()
    return dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )


def create_collection(
    client: QdrantClient,
    name: str,
    vector_size: int = 768,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    try:
        client.get_collection(name)
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )


def clip_collate_fn(processor, batch):
    images = [b["image"] for b in batch]
    inputs = processor(images=images, return_tensors="pt")
    return inputs


def build_index(d: int = 768):
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    return index


def add_record_to_index(index: faiss.IndexHNSWFlat, embeddings: np.ndarray):
    faiss.normalize_L2(embeddings)
    index.add(embeddings)


def save_index(index: faiss.IndexHNSWFlat, metadata: List[Dict], target_dir: str):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(target_dir, "index.index"))
    with open(os.path.join(target_dir, "metadata.json"), "w") as f:
        json.dump({"metadata": metadata}, f)


def read_index(target_dir: str):
    index = faiss.read_index(os.path.join(target_dir, "index.index"))
    with open(os.path.join(target_dir, "metadata.json"), "r") as f:
        metadata = json.loads(f.read())

    return index, metadata
