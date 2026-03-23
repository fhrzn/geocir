import warnings

import torch.nn as nn
from transformers import AutoProcessor, CLIPModel

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class ImageEncoder(nn.Module):
    def __init__(self, CLIP: CLIPModel):
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIP
        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.CLIP.get_image_features(pixel_values=x).pooler_output
        x = self.mlp(x)
        return x
