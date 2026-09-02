import torch
import torch.nn as nn
from pyproj import Proj, Transformer
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer
import os

from .rff.layers import GaussianEncoding


DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoint", "g3.pth")

# The released `g3.pth` checkpoint follows the naming used in the original G3
# repository. This module renamed a few projection heads, so remap the prefixes
# before calling `load_state_dict`. Shapes are identical, only the names differ.
_CHECKPOINT_KEY_MAP = {
    "vision_projection_else_1.": "img2txt_proj.",
    "text_projection_else.": "txt2img_proj.",
    "vision_projection_else_2.": "img2loc_proj.",
    "location_projection_else.": "loc2img_proj.",
    "vision_projection.": "vision_proj.",
    "text_projection.": "text_proj.",
}


def _remap_checkpoint_keys(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        for old_prefix, new_prefix in _CHECKPOINT_KEY_MAP.items():
            if key.startswith(old_prefix):
                key = new_prefix + key[len(old_prefix) :]
                break
        remapped[key] = value
    return remapped


class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class CustomLocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8]):
        super(CustomLocationEncoder, self).__init__()

        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module("LocEnc" + str(i), LocationEncoderCapsule(sigma=s))

        proj_wgs84 = Proj("epsg:4326")
        proj_mercator = Proj("epsg:3857")
        self.transformer = Transformer.from_proj(
            proj_wgs84, proj_mercator, always_xy=True
        )

    def forward(self, input):
        lat = input[:, 0].float().detach().cpu().numpy()
        lon = input[:, 1].float().detach().cpu().numpy()
        projected_lon_lat = self.transformer.transform(lon, lat)
        location = []
        for coord in zip(*projected_lon_lat):
            location.append([coord[1], coord[0]])
        location = torch.Tensor(location).to("cuda")
        location = location / 20037508.3427892

        location_features = torch.zeros(location.shape[0], 512).to("cuda")

        for i in range(self.n):
            location_features += self._modules["LocEnc" + str(i)](location)

        return location_features


class G3(torch.nn.Module):
    def __init__(
        self,
    ):
        super(G3, self).__init__()
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.text_processor = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_model = clip_model.vision_model
        self.text_model = clip_model.text_model
        self.vision_proj = clip_model.visual_projection
        self.text_proj = clip_model.text_projection

        self.location_encoder = CustomLocationEncoder()  # output batch_size, 3, 512
        # self.location_encoder = LocationEncoder(sigma=[2**0, 2**4, 2**8])
        self.img2txt_proj = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
        )
        self.txt2img_proj = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
        )

        self.img2loc_proj = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
        )
        self.loc2img_proj = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 768)
        )

        self.logit_scale1 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale2 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale3 = nn.Parameter(torch.tensor(3.99))

        # freeze CLIP
        self.vision_model.requires_grad_(False)
        self.vision_proj.requires_grad_(False)
        self.text_model.requires_grad_(False)
        self.text_proj.requires_grad_(False)

    def preprocess_image(self, image):
        return self.vision_processor(images=image, return_tensors="pt")["pixel_values"]

    def preprocess_text(self, text):
        return self.text_processor(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

    def load_checkpoint(self, checkpoint_path=DEFAULT_CHECKPOINT, strict=True):
        """Load a released G3 checkpoint into this model.

        Handles the parameter-name differences between the original G3 repo
        (used by `g3.pth`) and this module. Returns the `(missing_keys,
        unexpected_keys)` reported by `load_state_dict`.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _remap_checkpoint_keys(state_dict)
        return self.load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_pretrained(
        cls, checkpoint_path=DEFAULT_CHECKPOINT, device="cuda", strict=True
    ):
        """Build a G3 model and load `checkpoint_path` onto `device`."""
        model = cls()
        model.load_checkpoint(checkpoint_path, strict=strict)
        return model.to(device)

    def forward(self, images, texts, longitude, latitude, return_loss=True):

        vision_output = self.vision_model(images)[1]
        text_output = self.text_model(**texts)[1]
        image_embeds = self.vision_proj(vision_output)
        text_embeds = self.text_proj(text_output)  # batch_size, 512
        this_batch_locations = torch.stack((latitude, longitude), dim=1)
        location_embeds = self.location_encoder(this_batch_locations)

        # phase _1
        image_embeds_1 = self.img2txt_proj(image_embeds)
        text_embeds_1 = self.txt2img_proj(text_embeds.reshape(text_embeds.shape[0], -1))

        # normalized features
        image_embeds_1 = image_embeds_1 / image_embeds_1.norm(p=2, dim=-1, keepdim=True)
        text_embeds_1 = text_embeds_1 / text_embeds_1.norm(p=2, dim=-1, keepdim=True)

        # image with texts
        logit_scale = self.logit_scale1.exp()
        logits_per_texts_with_images = (
            torch.matmul(text_embeds_1, image_embeds_1.t()) * logit_scale
        )
        logits_per_images_with_texts = logits_per_texts_with_images.t()
        if return_loss:
            loss1 = self.clip_loss(logits_per_texts_with_images)

        loss_phase_1 = None
        if return_loss:
            loss_phase_1 = loss1

        # phase _2
        image_embeds_2 = self.img2loc_proj(image_embeds)
        location_embeds_2 = self.loc2img_proj(
            location_embeds.reshape(location_embeds.shape[0], -1)
        )

        # normalized features
        image_embeds_2 = image_embeds_2 / image_embeds_2.norm(p=2, dim=-1, keepdim=True)
        location_embeds_2 = location_embeds_2 / location_embeds_2.norm(
            p=2, dim=-1, keepdim=True
        )

        # image with location
        logit_scale = self.logit_scale2.exp()
        logits_per_locations_with_images = (
            torch.matmul(location_embeds_2, image_embeds_2.t()) * logit_scale
        )
        logits_per_images_with_locations = logits_per_locations_with_images.t()
        loss_phase_2 = None
        if return_loss:
            loss_phase_2 = self.clip_loss(logits_per_locations_with_images)

        loss = loss_phase_1 + loss_phase_2

        return {
            "logits_per_texts_with_images": logits_per_texts_with_images,
            "logits_per_images_with_texts": logits_per_images_with_texts,
            "logits_per_locations_with_images": logits_per_locations_with_images,
            "logits_per_images_with_locations": logits_per_images_with_locations,
            "logits_per_locations_with_texts": None,
            "logits_per_texts_with_locations": None,
            "loss": loss,
            "vision_output": vision_output,
            "text_output": text_output,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
        }

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
