"""UniIR CLIP-ScoreFusion (M-BEIR fine-tuned ViT-L/14) as a plain image/text encoder.

The "score fusion" is only an ``img_emb + txt_emb`` sum for multimodal queries; for
the image-only baseline we use the two towers like plain CLIP. UniIR's own
``CLIPScoreFusion`` class is imported straight from the clone -- point at it with
``$UNIIR_ROOT`` or ``uniir_root=``; checkpoint defaults to
``<uniir_root>/src/checkpoint/clip_sf_large.pth``.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import torch

_DEFAULT_UNIIR_ROOT = os.path.expanduser(os.environ.get("UNIIR_ROOT", "~/project/UniIR"))
CLIP_SF_MODEL_NAME = "ViT-L/14"
EMBED_DIM = 768


def _clip_sf_dir(uniir_root: str) -> str:
    return os.path.join(uniir_root, "src", "models", "uniir_clip", "clip_scorefusion")


@lru_cache(maxsize=1)
def _import_clip_score_fusion(uniir_root: str):
    clip_sf_dir = _clip_sf_dir(uniir_root)
    if not os.path.isfile(os.path.join(clip_sf_dir, "clip_sf.py")):
        raise FileNotFoundError(
            f"UniIR CLIP-ScoreFusion source not found under {clip_sf_dir!r}. "
            "Clone https://github.com/TIGER-AI-Lab/UniIR and set $UNIIR_ROOT."
        )
    for p in (clip_sf_dir, os.path.join(uniir_root, "src", "models")):
        if p not in sys.path:
            sys.path.append(p)
    from clip_sf import CLIPScoreFusion  # type: ignore

    return CLIPScoreFusion


class _UniIRImageProcessor:
    def __init__(self, preprocess_fn):
        self.preprocess_fn = preprocess_fn

    def __call__(self, images, return_tensors="pt", **_):
        batch = images if isinstance(images, (list, tuple)) else [images]
        return {"pixel_values": torch.stack([self.preprocess_fn(im) for im in batch])}


class _UniIRTokenizer:
    def __init__(self, tokenize_fn):
        self.tokenize_fn = tokenize_fn

    def __call__(self, text, max_length: int = 77, return_tensors="pt", **_):
        if isinstance(text, str):
            text = [text]
        input_ids = self.tokenize_fn(text)
        return {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}


class UniIRProcessor:
    """HF-ish adapter over OpenAI-CLIP preprocessing, for ``GeoTIRDataset`` /
    ``clip_collate_fn`` (needs ``.image_processor`` and ``.tokenizer``)."""

    def __init__(self, preprocess_fn, tokenize_fn):
        self.image_processor = _UniIRImageProcessor(preprocess_fn)
        self.tokenizer = _UniIRTokenizer(tokenize_fn)

    def __call__(self, images=None, text=None, return_tensors="pt", **_):
        out = {}
        if images is not None:
            out.update(self.image_processor(images=images))
        if text is not None:
            out.update(self.tokenizer(text))
        return out


def load_uniir_clip_sf(device, ckpt_path: str | None = None, uniir_root: str | None = None):
    uniir_root = os.path.expanduser(uniir_root or _DEFAULT_UNIIR_ROOT)
    ckpt_path = ckpt_path or os.path.join(uniir_root, "src", "checkpoint", "clip_sf_large.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"UniIR checkpoint not found: {ckpt_path!r}. Download clip_sf_large.pth from "
            "https://huggingface.co/TIGER-Lab/UniIR or pass --ckpt-path."
        )

    CLIPScoreFusion = _import_clip_score_fusion(uniir_root)
    model = CLIPScoreFusion(model_name=CLIP_SF_MODEL_NAME, device="cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[uniir] load_state_dict: {len(missing)} missing / {len(unexpected)} unexpected")

    model = model.float().to(device).eval()
    processor = UniIRProcessor(model.get_img_preprocess_fn(), model.get_tokenizer())
    return model, processor
