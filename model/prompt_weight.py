import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptWeighting(nn.Module):
    def __init__(
            self,
            normal_count: int = 6,
            abnormal_count: int = 10,
            temperature: float = 2.0,
            eps: float = 1e-8,
    ):
        super().__init__()
        if normal_count <= 0 or abnormal_count <= 0:
            raise ValueError("prompt counts must be positive")
        if temperature <= 0:
            raise ValueError("prompt weighting temperature must be positive")
        self.normal_count = normal_count
        self.abnormal_count = abnormal_count
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.raw_w_normal = nn.Parameter(torch.zeros(normal_count))
        self.raw_w_abnormal = nn.Parameter(torch.zeros(abnormal_count))
        self._runtime_stats = {}

    def _raw_weight(self, state_idx: int) -> torch.Tensor:
        if state_idx == 0:
            return self.raw_w_normal
        if state_idx == 1:
            return self.raw_w_abnormal
        raise ValueError(f"prompt state index must be 0 or 1, got {state_idx}")

    def weights(self, state_idx: int) -> torch.Tensor:
        raw_w = self._raw_weight(state_idx).float()
        return F.softmax(raw_w / self.temperature, dim=0)

    def aggregate(self, prompt_features: torch.Tensor, state_idx: int) -> torch.Tensor:
        expected_count = self.normal_count if state_idx == 0 else self.abnormal_count
        if prompt_features.shape[0] != expected_count:
            raise ValueError(
                f"prompt feature count mismatch for state {state_idx}: "
                f"got {prompt_features.shape[0]}, expected {expected_count}"
            )
        weights = self.weights(state_idx).to(
            device=prompt_features.device,
            dtype=prompt_features.dtype,
        )
        weighted = torch.sum(prompt_features * weights[:, None], dim=0)
        with torch.no_grad():
            raw_w = self._raw_weight(state_idx).detach()
            if torch.allclose(raw_w.float(), torch.zeros_like(raw_w).float()):
                mean_feature = prompt_features.mean(dim=0)
                diff = (weighted.detach().float() - mean_feature.detach().float()).abs().max()
                key = "normal_mean_equiv_max_abs_diff" if state_idx == 0 else "abnormal_mean_equiv_max_abs_diff"
                prev = self._runtime_stats.get(key)
                value = float(diff.item())
                self._runtime_stats[key] = value if prev is None else max(prev, value)
        return weighted

    def kl_loss(self) -> torch.Tensor:
        kl_terms = []
        for state_idx, count in [(0, self.normal_count), (1, self.abnormal_count)]:
            weights = self.weights(state_idx)
            kl_terms.append(torch.sum(weights * torch.log(weights * count + self.eps)))
        return kl_terms[0] + kl_terms[1]

    @staticmethod
    def _entropy(weights: torch.Tensor) -> torch.Tensor:
        return -torch.sum(weights * torch.log(weights + 1e-8))

    def stats(self) -> dict:
        stats = {}
        for state_idx, prefix in [(0, "normal"), (1, "abnormal")]:
            weights = self.weights(state_idx).detach().cpu()
            entropy = self._entropy(weights).item()
            max_value, max_idx = torch.max(weights, dim=0)
            raw = self._raw_weight(state_idx).detach().cpu()
            stats[f"prompt_w_{prefix}"] = weights.tolist()
            stats[f"max_{prefix}_weight"] = float(max_value.item())
            stats[f"max_{prefix}_index"] = int(max_idx.item())
            stats[f"entropy_{prefix}"] = float(entropy)
            stats[f"raw_w_{prefix}"] = raw.tolist()
        kl = self.kl_loss().detach().cpu()
        normal_w = self.weights(0).detach()
        abnormal_w = self.weights(1).detach()
        stats["kl_normal"] = float(torch.sum(normal_w * torch.log(normal_w * self.normal_count + self.eps)).cpu().item())
        stats["kl_abnormal"] = float(torch.sum(abnormal_w * torch.log(abnormal_w * self.abnormal_count + self.eps)).cpu().item())
        stats["prompt_kl"] = float(kl.item())
        stats["temperature"] = self.temperature
        stats.update(self._runtime_stats)
        return stats

    def reset_runtime_stats(self):
        self._runtime_stats = {}
