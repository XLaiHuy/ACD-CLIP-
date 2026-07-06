import torch
import torch.nn as nn

from .tokenizer import tokenize


class SoftPromptLearner(nn.Module):
    def __init__(
            self,
            ctx_len: int = 4,
            text_dim: int = 768,
            init_std: float = 0.02,
            init_phrase: str = "a photo of a",
            clip_model=None,
    ):
        super().__init__()
        self.ctx_len = int(ctx_len)
        self.text_dim = int(text_dim)
        self.init_std = float(init_std)
        self.init_phrase = init_phrase
        self.ctx_normal = nn.Parameter(torch.empty(self.ctx_len, self.text_dim))
        self.ctx_abnormal = nn.Parameter(torch.empty(self.ctx_len, self.text_dim))
        self.reset_parameters(clip_model)

    def reset_parameters(self, clip_model=None):
        nn.init.normal_(self.ctx_normal, std=self.init_std)
        nn.init.normal_(self.ctx_abnormal, std=self.init_std)
        if clip_model is None or self.init_phrase == "random":
            return
        with torch.no_grad():
            tokenized = tokenize(self.init_phrase).to(clip_model.token_embedding.weight.device)
            token_ids = tokenized[0]
            # CLIP EOT is the highest token id; copy only real phrase tokens.
            payload = token_ids[1:]
            eot_offset = int(payload.argmax().item())
            copy_len = min(self.ctx_len, max(eot_offset, 0))
            if copy_len <= 0:
                return
            token_embed = clip_model.token_embedding(tokenized).detach().float()[0, 1:1 + copy_len]
            self.ctx_normal.data[:copy_len].copy_(token_embed.to(self.ctx_normal.device))
            self.ctx_abnormal.data[:copy_len].copy_(token_embed.to(self.ctx_abnormal.device))

    def get_context(self, state_idx: int) -> torch.Tensor:
        if state_idx == 0:
            return self.ctx_normal
        if state_idx == 1:
            return self.ctx_abnormal
        raise ValueError(f"state_idx must be 0 or 1, got {state_idx}")

    def stats(self):
        with torch.no_grad():
            normal = self.ctx_normal.detach().float()
            abnormal = self.ctx_abnormal.detach().float()
            return {
                "ctx_norm_normal": float(normal.norm(dim=-1).mean().item()),
                "ctx_norm_abnormal": float(abnormal.norm(dim=-1).mean().item()),
                "ctx_absmax_normal": float(normal.abs().max().item()),
                "ctx_absmax_abnormal": float(abnormal.abs().max().item()),
            }
