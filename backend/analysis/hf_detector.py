"""
Pretrained Hugging Face AI-vs-real classifier.

Primary: dima806/ai_vs_human_generated_image_detection (public ViT, no login).
Fallback: Dafilab/ai-image-detector (gated — requires HF_TOKEN + model access).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from PIL import Image

from .compression_robust import build_inference_variants

logger = logging.getLogger(__name__)

# Public model (no authentication)
HF_REPO_PUBLIC = "dima806/ai_vs_human_generated_image_detection"
# Gated fallback
HF_REPO_GATED = "Dafilab/ai-image-detector"
HF_WEIGHTS_GATED = "pytorch_model.pth"
IMG_SIZE_EFFNET = 380


def _tta_crops(pil_img: Image.Image, *, include_recompression: bool = True) -> list[Image.Image]:
    """Spatial crops + optional messenger-style JPEG branches for robustness."""
    if include_recompression:
        return build_inference_variants(pil_img, max_variants=6)
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    crops = [rgb]
    for scale in (0.88, 0.76):
        cw = max(64, int(w * scale))
        ch = max(64, int(h * scale))
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        crops.append(rgb.crop((x0, y0, x0 + cw, y0 + ch)))
    return crops


def _ai_index_from_labels(id2label: dict) -> int:
    """Find logit index for the AI / fake class."""
    for idx, name in id2label.items():
        label = str(name).lower().replace(" ", "_").replace("-", "_")
        if any(k in label for k in ("ai", "fake", "synthetic", "generated")):
            return int(idx)
    return 1


class HfAiDetector:
    _instance: "HfAiDetector | None" = None

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.transform = None
        self.ai_class_index = 1
        self.backend: str = ""
        self.device = "cpu"
        self.load_error: str | None = None

    @classmethod
    def get_instance(cls) -> "HfAiDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_vit_public(self, device: str) -> str:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(HF_REPO_PUBLIC)
        self.model = AutoModelForImageClassification.from_pretrained(HF_REPO_PUBLIC)
        self.model.eval()
        self.model.to(self.device)
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.ai_class_index = _ai_index_from_labels(id2label)
        self.backend = "vit"
        self.load_error = None
        return f"ok ({HF_REPO_PUBLIC} ViT)"

    def _load_effnet_gated(self, device: str) -> str:
        import torch
        import timm
        from huggingface_hub import hf_hub_download
        from torchvision import transforms as T

        self.device = device
        weights_path = hf_hub_download(repo_id=HF_REPO_GATED, filename=HF_WEIGHTS_GATED)
        model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
        state = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        model.to(self.device)
        self.model = model
        self.transform = T.Compose(
            [
                T.Resize((IMG_SIZE_EFFNET, IMG_SIZE_EFFNET)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.ai_class_index = 0
        self.backend = "efficientnet"
        self.load_error = None
        return f"ok ({HF_REPO_GATED} EfficientNet-B4)"

    def load(self, device: str = "cpu") -> str:
        if self.model is not None:
            return f"ok ({self.backend})"

        errors: list[str] = []
        try:
            return self._load_vit_public(device)
        except Exception as e:
            errors.append(f"public ViT: {e}")
            logger.warning("Public HF detector failed: %s", e)

        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            try:
                return self._load_effnet_gated(device)
            except Exception as e:
                errors.append(f"gated EfficientNet: {e}")
                logger.warning("Gated HF detector failed: %s", e)

        self.load_error = "; ".join(errors)
        return f"failed: {self.load_error}"

    def predict_ai_likelihood(
        self,
        pil_img: Image.Image,
        *,
        is_messaging_compressed: bool = False,
    ) -> tuple[float, dict[str, Any]]:
        if self.model is None:
            return 50.0, {"available": False, "error": self.load_error, "backend": self.backend}

        try:
            import torch

            ai_probs: list[float] = []
            crops = _tta_crops(pil_img, include_recompression=is_messaging_compressed)
            for crop in crops:
                if self.backend == "vit" and self.processor is not None:
                    inputs = self.processor(images=crop, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)[0]
                        ai_probs.append(float(probs[self.ai_class_index]))
                elif self.transform is not None:
                    tensor = self.transform(crop).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits = self.model(tensor)
                        probs = torch.softmax(logits, dim=-1)[0]
                        ai_probs.append(float(probs[self.ai_class_index]))
                else:
                    break

            if not ai_probs:
                return 50.0, {"available": False, "error": "no inference backend"}

            # Max over branches: any strong synthetic view counts (messenger-safe)
            ai_prob = float(np.max(ai_probs))
            ai_mean = float(np.mean(ai_probs))
            if is_messaging_compressed and len(ai_probs) >= 3:
                # Blend max with upper quartile to reduce single-branch noise
                sorted_probs = sorted(ai_probs)
                q75 = sorted_probs[int(0.75 * (len(sorted_probs) - 1))]
                ai_prob = max(ai_prob, q75 * 0.92)

            return round(ai_prob * 100.0, 2), {
                "available": True,
                "backend": self.backend,
                "repo": HF_REPO_PUBLIC if self.backend == "vit" else HF_REPO_GATED,
                "ai_prob_max": round(ai_prob, 4),
                "ai_prob_mean": round(ai_mean, 4),
                "tta_crops": len(ai_probs),
                "messenger_tta": is_messaging_compressed,
            }
        except Exception as e:
            return 50.0, {"available": False, "error": str(e), "backend": self.backend}
