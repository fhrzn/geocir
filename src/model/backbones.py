"""Shared backbone loading + encoders for indexing (`src/index.py`) and
evaluation (`src/evaluate.py`).

Both entry points need the same thing: instantiate a retrieval backbone (CLIP /
G3 / GeoCLIP / GeoTIR / UniIR / SigLIP / BLIP / BLIP-2) with its weights, then
encode either images (to build the FAISS index) or query text (to search it).
This module owns that once.

    bb = load_backbone("geotir", device, ckpt_path=...)
    img_np   = bb.encode_image(batch)          # (B, bb.image_dim) float32 numpy
    txt_cpu  = bb.encode_text(texts, batch_sz) # (N, bb.image_dim) cpu tensor
    gps_np   = bb.encode_gps(batch)            # only if bb.supports_two_step
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from src.model.g3 import G3
from src.model.geoclip import GeoCLIP
from src.model.geotir.model import GeoTIRModel

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
BLIP_MODEL_NAME = "Salesforce/blip-itm-base-coco"
BLIP2_MODEL_NAME = "Salesforce/blip2-itm-vit-g"
MODEL_NAMES = ("clip", "g3", "geoclip", "geotir", "uniir", "siglip", "blip", "blip2")

ImgEncoder = Callable[[dict], np.ndarray]
TextEncoder = Callable[[list[str], int], torch.Tensor]


@dataclass
class Backbone:
    name: str
    model: object
    processor: object          # image processor (indexing); may also hold a tokenizer
    image_dim: int             # width of the image / text retrieval vector
    encode_image: ImgEncoder   # batch dict -> (B, image_dim) float32 numpy
    encode_text: TextEncoder   # (texts, batch_size) -> (N, image_dim) cpu tensor
    supports_two_step: bool = False
    gps_dim: int | None = None
    encode_gps: ImgEncoder | None = None
    # query text -> gps/location-aligned embedding (for two-step retrieval search);
    # None means the backbone has no text->location path (e.g. G3).
    encode_text_gps: TextEncoder | None = None

    def __post_init__(self):
        if self.supports_two_step and (self.encode_gps is None or self.gps_dim is None):
            raise ValueError("two-step backbone must provide encode_gps and gps_dim")


def load_backbone(name: str, device, ckpt_path: str | None = None) -> Backbone:
    if name == "clip":
        return _clip(device)
    if name == "g3":
        return _g3(device)
    if name == "geoclip":
        return _geoclip(device)
    if name == "geotir":
        return _geotir(device, ckpt_path)
    if name == "uniir":
        return _uniir(device, ckpt_path)
    if name == "siglip":
        return _siglip(device)
    if name == "blip":
        return _blip(device)
    if name == "blip2":
        return _blip2(device)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def _text_batches(texts: list[str], batch_size: int):
    for i in tqdm(range(0, len(texts), batch_size), desc="encode text", leave=False):
        yield texts[i : i + batch_size]


# --------------------------------------------------------------------------- #
# CLIP                                                                        #
# --------------------------------------------------------------------------- #
def _clip(device) -> Backbone:
    model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    model = torch.compile(model)
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)

    def encode_image(batch):
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            out = model.get_image_features(batch["pixel_values"].to(device))
        return F.normalize(out.pooler_output.float(), dim=-1).cpu().numpy()

    def encode_text(texts, batch_size):
        embs = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for chunk in _text_batches(texts, batch_size):
                inp = processor(
                    text=chunk, return_tensors="pt",
                    padding=True, truncation=True, max_length=77,
                ).to(device)
                feats = model.get_text_features(**inp).pooler_output
                embs.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(embs, dim=0)

    return Backbone("clip", model, processor, 768, encode_image, encode_text)


# --------------------------------------------------------------------------- #
# G3 (two-step)                                                               #
# --------------------------------------------------------------------------- #
def _g3(device) -> Backbone:
    model = G3.from_pretrained().to(device).eval()
    processor = model._processor

    def encode_image(batch):
        with torch.no_grad():
            img_emb = model.vision_proj(
                model.vision_model(batch["pixel_values"].to(device)).pooler_output
            )
            img_emb_n = F.normalize(img_emb, dim=-1)
            img2txt_n = F.normalize(model.img2txt_proj(img_emb), dim=-1)
            img2loc_n = F.normalize(model.img2loc_proj(img_emb), dim=-1)
            out = F.normalize(torch.cat([img_emb_n, img2txt_n, img2loc_n], dim=1), dim=-1)
        return out.cpu().numpy()

    def encode_gps(batch):
        with torch.no_grad():
            gps = model.location_encoder(batch["latlon"].to(device))
            gps = model.loc2img_proj(gps.reshape(gps.shape[0], -1))
        return F.normalize(gps, dim=-1).cpu().numpy()

    def encode_text(texts, batch_size):
        embs = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for chunk in _text_batches(texts, batch_size):
                inp = {k: v.to(device) for k, v in model.preprocess_text(chunk).items()}
                text_emb = model.text_proj(model.text_model(**inp)[1])
                txt2img = F.normalize(model.txt2img_proj(text_emb).float(), dim=-1)
                text_n = F.normalize(text_emb.float(), dim=-1)
                loc_pad = torch.zeros_like(text_n)
                out = F.normalize(torch.cat([txt2img, text_n, loc_pad], dim=1), dim=-1)
                embs.append(out.cpu())
        return torch.cat(embs, dim=0)

    return Backbone(
        "g3", model, processor, 2304, encode_image, encode_text,
        supports_two_step=True, gps_dim=768, encode_gps=encode_gps,
    )


# --------------------------------------------------------------------------- #
# GeoCLIP (two-step)                                                          #
# --------------------------------------------------------------------------- #
def _geoclip(device) -> Backbone:
    model = GeoCLIP().to(device).eval()
    processor = model.image_encoder.image_processor

    def encode_image(batch):
        with torch.no_grad():
            out = model.image_encoder(batch["pixel_values"].to(device))
        return F.normalize(out, dim=-1).cpu().numpy()

    def encode_gps(batch):
        with torch.no_grad():
            out = model.location_encoder(batch["latlon"].to(device))
        return F.normalize(out, dim=-1).cpu().numpy()

    def encode_text(texts, batch_size):
        embs = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for chunk in _text_batches(texts, batch_size):
                inp = model.text_encoder.preprocess_text(chunk)
                inp = {k: v.to(device) for k, v in inp.items()}
                feats = model.text_encoder(**inp)
                embs.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(embs, dim=0)

    # GeoCLIP's text encoder shares the projection MLP with the image encoder, so
    # text embeddings already live in the image/location-aligned space -> the same
    # text embedding can be searched against the GPS index directly.
    return Backbone(
        "geoclip", model, processor, 512, encode_image, encode_text,
        supports_two_step=True, gps_dim=512, encode_gps=encode_gps,
        encode_text_gps=encode_text,
    )


# --------------------------------------------------------------------------- #
# GeoTIR (LoRA-finetuned CLIP; requires a checkpoint)                         #
# --------------------------------------------------------------------------- #
def _geotir(device, ckpt_path: str | None) -> Backbone:
    if not ckpt_path:
        raise ValueError("--ckpt-path is required for --model geotir")
    model = GeoTIRModel(clip_model_name=CLIP_MODEL_NAME).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = torch.compile(model.eval())
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)

    def encode_image(batch):
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            out = model.encode_images(pixel_values=batch["pixel_values"].to(device))
        return out.cpu().float().numpy()

    def encode_text(texts, batch_size):
        embs = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for chunk in _text_batches(texts, batch_size):
                inp = processor(
                    text=chunk, return_tensors="pt",
                    padding=True, truncation=True, max_length=77,
                ).to(device)
                out = model.encode_texts(inp["input_ids"], inp["attention_mask"])
                embs.append(out.cpu().float())
        return torch.cat(embs, dim=0)

    return Backbone("geotir", model, processor, 768, encode_image, encode_text)


# --------------------------------------------------------------------------- #
# UniIR CLIP-ScoreFusion (M-BEIR fine-tuned ViT-L/14; used image-only)         #
# --------------------------------------------------------------------------- #
def _uniir(device, ckpt_path: str | None) -> Backbone:
    from src.model.uniir.model import EMBED_DIM, load_uniir_clip_sf

    model, processor = load_uniir_clip_sf(device, ckpt_path)
    tokenize = processor.tokenizer.tokenize_fn

    def encode_image(batch):
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            out = model.encode_image(batch["pixel_values"].to(device))
        return F.normalize(out.float(), dim=-1).cpu().numpy()

    def encode_text(texts, batch_size):
        embs = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for chunk in _text_batches(texts, batch_size):
                tok = tokenize(list(chunk)).to(device)
                feats = model.encode_text(tok)
                embs.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(embs, dim=0)

    return Backbone("uniir", model, processor, EMBED_DIM, encode_image, encode_text)


class _MaskTokenizerShim:
    """Wrap a tokenizer so it always yields an ``attention_mask`` (SigLIP omits it,
    which breaks ``GeoTIRDataset``; SigLIP pads to a fixed length with full attention)."""

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, k):
        return getattr(self._tok, k)

    def __call__(self, *args, **kwargs):
        out = self._tok(*args, **kwargs)
        if "input_ids" in out and "attention_mask" not in out:
            ids = out["input_ids"]
            out["attention_mask"] = (
                torch.ones_like(ids)
                if torch.is_tensor(ids)
                else [[1] * len(x) for x in ids]
            )
        return out


class _ProcessorShim:
    def __init__(self, proc):
        self._proc = proc
        self.image_processor = proc.image_processor
        self.tokenizer = _MaskTokenizerShim(proc.tokenizer)

    def __getattr__(self, k):
        return getattr(self._proc, k)

    def __call__(self, *args, **kwargs):
        return self._proc(*args, **kwargs)


# --------------------------------------------------------------------------- #
# SigLIP (so400m/14 @ 384; image-only, dual-encoder like CLIP)                 #
# --------------------------------------------------------------------------- #
def _siglip(device) -> Backbone:
    from transformers import SiglipModel

    model = SiglipModel.from_pretrained(SIGLIP_MODEL_NAME).to(device).eval()
    processor = _ProcessorShim(AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME))

    def encode_image(batch):
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            out = model.get_image_features(pixel_values=batch["pixel_values"].to(device))
        return F.normalize(out.pooler_output.float(), dim=-1).cpu().numpy()

    def encode_text(texts, batch_size):
        embs = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for chunk in _text_batches(texts, batch_size):
                inp = processor(
                    text=list(chunk), return_tensors="pt",
                    padding="max_length", truncation=True,
                ).to(device)
                feats = model.get_text_features(**inp).pooler_output
                embs.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(embs, dim=0)

    return Backbone("siglip", model, processor, 1152, encode_image, encode_text)


# --------------------------------------------------------------------------- #
# BLIP (ITM base, COCO; ITC projection head -> 256-d)                          #
# --------------------------------------------------------------------------- #
def _blip(device) -> Backbone:
    from transformers import BlipForImageTextRetrieval

    model = BlipForImageTextRetrieval.from_pretrained(BLIP_MODEL_NAME).to(device).eval()
    processor = AutoProcessor.from_pretrained(BLIP_MODEL_NAME)

    def encode_image(batch):
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            vis = model.vision_model(batch["pixel_values"].to(device))[0]
            emb = model.vision_proj(vis[:, 0, :])
        return F.normalize(emb.float(), dim=-1).cpu().numpy()

    def encode_text(texts, batch_size):
        embs = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for chunk in _text_batches(texts, batch_size):
                inp = processor(
                    text=list(chunk), return_tensors="pt",
                    padding=True, truncation=True, max_length=77,
                ).to(device)
                txt = model.text_encoder(
                    input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
                )[0]
                emb = model.text_proj(txt[:, 0, :])
                embs.append(F.normalize(emb.float(), dim=-1).cpu())
        return torch.cat(embs, dim=0)

    return Backbone("blip", model, processor, 256, encode_image, encode_text)


# --------------------------------------------------------------------------- #
# BLIP-2 (ITM ViT-g; Q-Former ITC head -> 256-d, image query tokens mean-pooled)#
# --------------------------------------------------------------------------- #
def _blip2(device) -> Backbone:
    from types import SimpleNamespace

    from transformers import (
        Blip2TextModelWithProjection,
        Blip2VisionModelWithProjection,
    )

    vision = Blip2VisionModelWithProjection.from_pretrained(BLIP2_MODEL_NAME).to(device).eval()
    text = Blip2TextModelWithProjection.from_pretrained(BLIP2_MODEL_NAME).to(device).eval()
    processor = AutoProcessor.from_pretrained(BLIP2_MODEL_NAME)

    def encode_image(batch):
        with torch.no_grad():
            emb = vision(pixel_values=batch["pixel_values"].to(device)).image_embeds
        return F.normalize(emb.mean(dim=1).float(), dim=-1).cpu().numpy()

    def encode_text(texts, batch_size):
        embs = []
        with torch.no_grad():
            for chunk in _text_batches(texts, batch_size):
                # Blip2Processor(text=...) mishandles max_length; tokenize directly.
                inp = processor.tokenizer(
                    list(chunk), return_tensors="pt",
                    padding=True, truncation=True, max_length=64,
                ).to(device)
                emb = text(
                    input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
                ).text_embeds[:, 0, :]
                embs.append(F.normalize(emb.float(), dim=-1).cpu())
        return torch.cat(embs, dim=0)

    model = SimpleNamespace(vision=vision, text=text)
    return Backbone("blip2", model, processor, 256, encode_image, encode_text)
