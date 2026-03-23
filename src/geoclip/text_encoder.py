import warnings

import torch.nn as nn
from transformers import AutoTokenizer, CLIPModel

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class TextEncoder(nn.Module):
    def __init__(self, CLIP: CLIPModel):
        super(TextEncoder, self).__init__()
        self.CLIP = CLIP
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

    def preprocess_text(self, text):
        x = self.tokenizer(text, return_tensors="pt")
        return x

    def forward(self, **kwargs):
        x = self.CLIP.get_text_features(**kwargs).pooler_output
        x = self.mlp(x)
        return x
