import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel


def build_positive_mask(cell_ids: torch.Tensor) -> torch.Tensor:
    """(N, N) bool: True where two samples share a (category, country) cell id."""
    mask = cell_ids.unsqueeze(0) == cell_ids.unsqueeze(1)
    mask.fill_diagonal_(False)
    return mask


def multi_positive_infonce_loss(
    image_embeds: torch.Tensor,   # (N, D) normalized
    text_embeds: torch.Tensor,    # (N, D) normalized
    positive_mask: torch.Tensor,  # (N, N) bool
    temperature: float,
    weights: torch.Tensor | None = None,  # (N,) per-anchor loss weights
) -> torch.Tensor:
    """
    Symmetric multi-positive InfoNCE loss.

    For each anchor i, positives are all j where positive_mask[i, j] = True.
    All other j (including diagonal) are treated as negatives.
    Loss is averaged over both image->text and text->image directions.

    `weights` (optional): a per-anchor weight (e.g. class-balanced inverse
    frequency of the anchor's cell). Anchors with no in-batch positive are
    excluded from both the sum and the weight normaliser.
    """
    logits_i2t = (image_embeds @ text_embeds.T) / temperature
    logits_t2i = (text_embeds @ image_embeds.T) / temperature

    positive_mask = positive_mask.to(image_embeds.device)
    if weights is not None:
        weights = weights.to(image_embeds.device)

    def _loss_one_direction(logits: torch.Tensor) -> torch.Tensor:
        has_positive = positive_mask.any(dim=1)
        if not has_positive.any():
            return logits.sum() * 0.0  # keep the graph connected, contribute nothing
        log_denom = torch.logsumexp(logits, dim=1)
        log_probs = logits - log_denom.unsqueeze(1)
        pos_log_probs = (log_probs * positive_mask.float()).sum(dim=1)
        num_positives = positive_mask.float().sum(dim=1).clamp(min=1)
        per_anchor_loss = (-pos_log_probs / num_positives)[has_positive]
        if weights is None:
            return per_anchor_loss.mean()
        w = weights[has_positive]
        return (per_anchor_loss * w).sum() / w.sum().clamp(min=1e-8)

    loss_i2t = _loss_one_direction(logits_i2t)
    loss_t2i = _loss_one_direction(logits_t2i)
    return (loss_i2t + loss_t2i) / 2


class GeoTIRModel(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        init_temperature: float = 0.07,
    ):
        super().__init__()

        base_model = CLIPModel.from_pretrained(clip_model_name)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.clip = get_peft_model(base_model, lora_config)
        self.clip.print_trainable_parameters()

        # Learnable temperature in log-scale for numerical stability
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=0.5)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeds = self.clip.get_image_features(pixel_values=pixel_values).pooler_output
        return F.normalize(embeds, dim=-1)

    def encode_texts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        return F.normalize(embeds, dim=-1)

    def forward(self, batch: dict) -> dict:
        image_embeds = self.encode_images(batch["pixel_values"])
        text_embeds = self.encode_texts(batch["input_ids"], batch["attention_mask"])

        positive_mask = build_positive_mask(batch["cell_id"].to(image_embeds.device))

        loss = multi_positive_infonce_loss(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            positive_mask=positive_mask,
            temperature=self.temperature,
            weights=batch.get("weight"),
        )

        return {
            "loss": loss,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "temperature": self.temperature.item(),
        }
