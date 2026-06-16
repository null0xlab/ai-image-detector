"""
Singleton registry — load ML models once at application startup.
Graceful fallback: failed models are skipped and reported in load_status.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import cv2
import numpy as np

from .hf_detector import HfAiDetector

logger = logging.getLogger(__name__)

ML_INPUT_SIZE = 256


class ModelRegistry:
    _instance: "ModelRegistry | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.load_status: dict[str, str] = {}
        self.device = "cpu"
        self.cnn_model = None
        self.cnn_preprocess = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.clip_text_features = None
        self.efficientnet_model = None
        self.efficientnet_preprocess = None
        self.face_cascade = None
        self.hf_detector: HfAiDetector | None = None
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_all(self) -> dict[str, str]:
        if self._loaded:
            return self.load_status

        self._load_torch_device()
        self._load_face_detector()
        self._load_cnn_detection()
        self._load_clip_detector()
        self._load_efficientnet_deepfake()
        self._load_hf_ai_detector()
        self._loaded = True
        return self.load_status

    def _load_hf_ai_detector(self) -> None:
        """Dafilab EfficientNet-B4 — compression-robust AI vs real classifier."""
        try:
            det = HfAiDetector.get_instance()
            status = det.load(device=self.device)
            self.hf_detector = det
            self.load_status["hf_ai_detector"] = status
        except Exception as e:
            self.load_status["hf_ai_detector"] = f"failed: {e}"

    def _load_torch_device(self) -> None:
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.load_status["torch"] = f"ok ({self.device})"
        except ImportError:
            self.load_status["torch"] = "unavailable (pip install torch)"

    def _load_face_detector(self) -> None:
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                raise RuntimeError("Haar cascade failed to load")
            self.face_cascade = cascade
            self.load_status["face_detector"] = "ok (opencv haar)"
        except Exception as e:
            self.load_status["face_detector"] = f"failed: {e}"

    def _load_cnn_detection(self) -> None:
        """ResNet50 backbone (CNNDetection-style general AI detector)."""
        try:
            import torch
            import torchvision.transforms as T
            from torchvision import models

            weights = models.ResNet50_Weights.IMAGENET1K_V2
            backbone = models.resnet50(weights=weights)
            backbone.fc = torch.nn.Linear(backbone.fc.in_features, 2)
            backbone.eval()
            backbone.to(self.device)

            self.cnn_model = backbone
            self.cnn_preprocess = T.Compose(
                [
                    T.Resize((ML_INPUT_SIZE, ML_INPUT_SIZE)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.load_status["cnn_detection"] = "ok (resnet50, untrained head — heuristic logits)"
        except Exception as e:
            self.load_status["cnn_detection"] = f"failed: {e}"

    def _load_clip_detector(self) -> None:
        try:
            import torch
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            model.eval()
            model.to(self.device)
            tokenizer = open_clip.get_tokenizer("ViT-L-14")

            # Carefully crafted prompt ensemble covering the visual cues each class exhibits.
            # More diverse prompts reduce sensitivity to exact phrasing and
            # improve zero-shot calibration.
            real_prompts = [
                "a real photograph taken by a camera",
                "a genuine photo captured on a smartphone",
                "an authentic photograph with natural lighting and lens artifacts",
                "a candid real-world photo with natural noise and depth of field",
                "a raw photograph showing natural chromatic aberration and grain",
                "an unedited camera photo with realistic lens blur and sensor noise",
                "a natural snapshot with realistic camera imperfections",
            ]
            ai_prompts = [
                "an AI generated synthetic image",
                "an image created by artificial intelligence like Midjourney, DALL-E, Flux, or Stable Diffusion",
                "a fully computer-generated photorealistic image with no real camera",
                "an AI artwork generated by stable diffusion, Flux, Wan, or a generative model",
                "a synthetically generated image with perfect smooth textures and digital noise",
                "an AI illustration with unrealistically uniform sharpness and saturation",
                "a digitally synthesized scene generated by a diffusion model",
                "a photorealistic AI portrait generated by ChatGPT or Grok",
            ]
            deepfake_prompts = [
                "a deepfake manipulated face photo",
                "a face swap created by AI replacing someone's face in a photo",
                "a digitally altered portrait with AI-replaced facial features",
                "a photo where the face has been swapped using deep learning",
                "a manipulated image with inconsistent face and background lighting",
                "a portrait with a digitally inserted face that does not match the surroundings",
            ]

            all_prompts = real_prompts + ai_prompts + deepfake_prompts
            with torch.no_grad():
                tokens = tokenizer(all_prompts).to(self.device)
                all_features = model.encode_text(tokens)
                all_features /= all_features.norm(dim=-1, keepdim=True)

            # Average within each class to get robust class prototypes
            n_real = len(real_prompts)
            n_ai = len(ai_prompts)
            n_df = len(deepfake_prompts)
            real_proto = all_features[:n_real].mean(dim=0, keepdim=True)
            ai_proto = all_features[n_real:n_real + n_ai].mean(dim=0, keepdim=True)
            df_proto = all_features[n_real + n_ai:].mean(dim=0, keepdim=True)
            # Re-normalize prototypes after averaging
            real_proto /= real_proto.norm(dim=-1, keepdim=True)
            ai_proto /= ai_proto.norm(dim=-1, keepdim=True)
            df_proto /= df_proto.norm(dim=-1, keepdim=True)

            text_features = torch.cat([real_proto, ai_proto, df_proto], dim=0)

            self.clip_model = model
            self.clip_preprocess = preprocess
            self.clip_tokenizer = tokenizer
            self.clip_text_features = text_features
            self.load_status["clip_detector"] = "ok (ViT-L-14 zero-shot, multi-prompt 3-class ensemble)"
        except Exception as e:
            self.load_status["clip_detector"] = f"failed: {e}"

    def _load_efficientnet_deepfake(self) -> None:
        try:
            import timm
            import torch
            import torchvision.transforms as T

            model = timm.create_model(
                "efficientnet_b4", pretrained=True, num_classes=2
            )
            model.eval()
            model.to(self.device)
            self.efficientnet_model = model
            self.efficientnet_preprocess = T.Compose(
                [
                    T.Resize((ML_INPUT_SIZE, ML_INPUT_SIZE)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.load_status["efficientnet_deepfake"] = (
                "ok (efficientnet_b4, ImageNet head — face-region heuristic)"
            )
        except Exception as e:
            self.load_status["efficientnet_deepfake"] = f"failed: {e}"

    def detect_faces(self, cv_img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) face bounding boxes."""
        if self.face_cascade is None:
            return []
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def is_ready(self) -> bool:
        return self._loaded
