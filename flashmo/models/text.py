"""Frozen CLIP and optional DistilBERT text encoders for MotionSiT."""

import torch
from torch import nn


class FrozenTextEncoder(nn.Module):
    def __init__(self, kind="clip", model_name=None, local_files_only=False):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel, CLIPTextModelWithProjection

        self.kind = kind
        if kind == "clip":
            name = model_name or "openai/clip-vit-base-patch32"
            self.text_model = CLIPTextModelWithProjection.from_pretrained(
                name, local_files_only=local_files_only)
            self.output_dim = self.text_model.config.projection_dim
            self.max_length = self.text_model.config.max_position_embeddings
        elif kind == "bert":
            name = model_name or "distilbert/distilbert-base-uncased"
            self.text_model = AutoModel.from_pretrained(name, local_files_only=local_files_only)
            self.output_dim = self.text_model.config.hidden_size
            self.max_length = self.text_model.config.max_position_embeddings
        else:
            raise ValueError("kind must be 'clip' or 'bert'")
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Preserve frozen encoder dropout behavior even when the parent trains.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, texts):
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("texts must be a non-empty sequence of strings")
        device = next(self.text_model.parameters()).device
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_length).to(device)
        output = self.text_model(**tokens)
        if self.kind == "clip":
            features = output.text_embeds.float()[:, None]
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
        else:
            features = output.last_hidden_state.float()
            mask = tokens.attention_mask.bool()
        return {"features": features, "mask": mask}
