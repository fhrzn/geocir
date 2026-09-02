from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch.nn import functional as F
from torch import nn
from src.model.geoclip import GeoCLIP
from src.model.g3 import G3
from src.data.data import GeoTIRDataset, warm_image_cache
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
from torch.utils.data import DataLoader
from __future__ import annotations


_BACKBONES: dict[str, type[BaseBackboneRetriever]] = {}
def register_backbone(cls):
    _BACKBONES[cls.name] = cls
    return cls

class BaseBackboneRetriever(nn.Module, ABC):
    name: str

    @abstractmethod
    def encode_gps(self, gps: torch.Tensor): ...

    @abstractmethod
    def encode_image(self, image: torch.Tensor): ...

    @abstractmethod
    def encode_text(self, text: torch.Tensor): ...

    @property
    def processor(self):
        return self.processor

    @property
    def tokenizer(self):
        return self.tokenizer

@register_backbone
class GeoCLIPBackboneRetriever(BaseBackboneRetriever):
    name = "geoclip"

    def __init__(self, **kwargs):
        super().__init__()

        self.model = GeoCLIP(**kwargs).eval()
        self.processor = self.model.image_encoder.image_processor
        self.tokenizer = self.model.text_encoder.tokenizer
        self.index_d = 512

    @torch.no_grad()
    def encode_gps(self, gps: torch.Tensor):
        return F.normalize(self.model.location_encoder(gps), dim=-1)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        return F.normalize(self.model.image_encoder(image), dim=-1)

    @torch.no_grad(9)
    def encode_text(self, text: torch.Tensor):
        pass

@register_backbone
class G3BackboneRetriever(BaseBackboneRetriever):
    name = "g3"

    def __init__(self, **kwargs):
        super().__init__()

        self.model = G3.from_pretrained().eval()
        self.processor = self.model.vision_processor
        self.tokenizer = self.model.text_processor
        self.index_d = 768

    @torch.no_grad()
    def encode_gps(self, gps: torch.Tensor):
        gps_embed = self.model.location_encoder(gps)
        gps_embed = self.model.loc2img_proj(gps_embed.reshape(gps_embed.shape[0], -1))
        return F.normalize(gps_embed, dim=-1)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        img_embed = self.model.vision_model(image).pooler_output
        img_embed = self.model.vision_proj(img_embed)
        img_embed = self.model.img2txt_proj(img_embed)
        return F.normalize(img_embed, dim=-1)

    @torch.no_grad(9)
    def encode_text(self, text: torch.Tensor):
        pass


class TwoStepRetriever(nn.Module):
    def __init__(self, backbone: Literal["clip", "geoclip", "g3"] = "geoclip", backbone_kwargs = None):
        super().__init__()

        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; available {sorted(_BACKBONES)}")

        self.backbone = _BACKBONES[backbone](**(backbone_kwargs or {}))

    def forward(self):
        pass
