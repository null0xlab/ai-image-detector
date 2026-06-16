"""
Layer 1 — Fast triage: EXIF, C2PA, ELA heatmap, double JPEG compression.
Reuses existing metadata analysis where appropriate and adds structured forensic signals.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import cv2
import exifread
import numpy as np
import piexif
from PIL import Image

from .metadata import AI_SIGNATURES, analyze_metadata

AI_SOFTWARE_EXIF_MAP = {
    "adobe firefly": "Adobe Firefly",
    "firefly": "Adobe Firefly",
    "dall-e": "OpenAI DALL-E",
    "dalle": "OpenAI DALL-E",
    "openai": "OpenAI DALL-E",
    "chatgpt": "OpenAI DALL-E",
    "stable diffusion": "Stable Diffusion",
    "midjourney": "Midjourney",
    "canva": "Canva AI",
    "bing image creator": "Microsoft Designer (DALL-E)",
    "comfyui": "Stable Diffusion (ComfyUI)",
    "automatic1111": "Stable Diffusion (A1111)",
    "a1111": "Stable Diffusion (A1111)",
    "novelai": "NovelAI",
    "flux": "Flux",
}


def _detect_software_from_exif(image_bytes: bytes) -> str | None:
    try:
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
        for key in ("Image Software", "Image Model", "Image ImageDescription"):
            if key in tags:
                val = str(tags[key]).lower()
                for sig, label in AI_SOFTWARE_EXIF_MAP.items():
                    if sig in val:
                        return label
        piexif_dict = piexif.load(image_bytes)
        for ifd in ("0th", "Exif", "1st"):
            if not piexif_dict.get(ifd):
                continue
            for tag_val in piexif_dict[ifd].values():
                if isinstance(tag_val, bytes):
                    val = tag_val.decode("utf-8", errors="ignore").lower()
                else:
                    val = str(tag_val).lower()
                for sig, label in AI_SOFTWARE_EXIF_MAP.items():
                    if sig in val:
                        return label
    except Exception:
        pass
    return None


def analyze_exif_forensic(
    image_bytes: bytes,
    metadata_result: dict | None = None,
    is_messaging_compressed: bool = False,
    is_pristine_digital: bool = False,
    is_web_downloaded: bool = False,
) -> dict[str, Any]:
    """
    Enhanced EXIF pass: exif_score, detected_software, anomaly flags.
  """
    if metadata_result is None:
        metadata_result = analyze_metadata(
            image_bytes,
            is_messaging_compressed=is_messaging_compressed,
            is_pristine_digital=is_pristine_digital,
            is_web_downloaded=is_web_downloaded,
        )

    exif_score = float(metadata_result["score"])
    detected_software = _detect_software_from_exif(image_bytes)
    tags = metadata_result.get("tags") or {}
    software_tag = tags.get("software")

    exif_anomaly = exif_score >= 55.0 or detected_software is not None
    if not detected_software and software_tag:
        sw_lower = software_tag.lower()
        for sig, label in AI_SOFTWARE_EXIF_MAP.items():
            if sig in sw_lower:
                detected_software = label
                break

    missing_camera = not any(k in tags for k in ("make", "model", "f_number", "iso"))
    missing_maker_note = True  # piexif rarely exposes MakerNote without camera make

    return {
        "exif_score": round(exif_score, 1),
        "detected_software": detected_software,
        "exif_anomaly": exif_anomaly,
        "missing_camera_model": missing_camera,
        "missing_maker_note": missing_maker_note,
        "details": metadata_result.get("details", ""),
        "tags": tags,
    }


def check_c2pa(image_bytes: bytes) -> dict[str, Any]:
    """Content Credentials / C2PA manifest probe with optional c2pa library."""
    has_c2pa = False
    c2pa_producer: str | None = None
    ai_tool_used: str | None = None
    edit_history: list[str] = []

    # Heuristic byte scan (works without c2pa-python)
    sample = image_bytes[: min(len(image_bytes), 512_000)]
    lower = sample.decode("latin-1", errors="ignore").lower()
    if "c2pa" in lower or "contentcredentials" in lower or "jumbf" in lower:
        has_c2pa = True
        for producer_hint, label in (
            ("adobe", "Adobe"),
            ("firefly", "Adobe Firefly"),
            ("openai", "OpenAI"),
            ("dall-e", "OpenAI DALL-E"),
            ("google", "Google"),
            ("gemini", "Google Gemini"),
            ("imagen", "Google Imagen"),
        ):
            if producer_hint in lower:
                c2pa_producer = label
                ai_tool_used = label
                break

    try:
        import c2pa  # type: ignore

        reader = c2pa.Reader("image/jpeg", image_bytes)
        manifest = reader.json()
        if manifest:
            has_c2pa = True
            if isinstance(manifest, str):
                manifest = json.loads(manifest)
            c2pa_producer = c2pa_producer or _extract_c2pa_field(manifest, "claim_generator")
            ai_tool_used = ai_tool_used or _extract_c2pa_field(manifest, "softwareAgent")
            actions = _extract_c2pa_actions(manifest)
            if actions:
                edit_history = actions
    except ImportError:
        pass
    except Exception:
        pass

    # Try c2pa-rs CLI if installed
    if not has_c2pa:
        try:
            import subprocess
            import tempfile
            import os

            suffix = ".jpg" if image_bytes[:3] == b"\xff\xd8" else ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            try:
                out = subprocess.run(
                    ["c2patool", tmp_path, "-d"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                if out.returncode == 0 and out.stdout.strip():
                    has_c2pa = True
                    if "adobe" in out.stdout.lower():
                        c2pa_producer = c2pa_producer or "Adobe"
                    if "openai" in out.stdout.lower() or "dall" in out.stdout.lower():
                        c2pa_producer = c2pa_producer or "OpenAI"
            finally:
                os.unlink(tmp_path)
        except Exception:
            pass

    return {
        "has_c2pa": has_c2pa,
        "c2pa_producer": c2pa_producer,
        "ai_tool_used": ai_tool_used,
        "edit_history": edit_history,
    }


def _extract_c2pa_field(manifest: dict, key: str) -> str | None:
    try:
        blob = json.dumps(manifest).lower()
        if key.lower() in blob:
            m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', json.dumps(manifest), re.I)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def _extract_c2pa_actions(manifest: dict) -> list[str]:
    actions: list[str] = []
    try:
        text = json.dumps(manifest)
        for action in re.findall(r'"action"\s*:\s*"([^"]+)"', text, re.I):
            if action not in actions:
                actions.append(action)
    except Exception:
        pass
    return actions[:10]


def analyze_ela_forensic(pil_img: Image.Image, amplify: float = 12.0, quality: int = 90) -> dict[str, Any]:
    """ELA at fixed quality with amplified difference heatmap (original resolution)."""
    try:
        ela_rgb = pil_img.convert("RGB")
        out = io.BytesIO()
        ela_rgb.save(out, format="JPEG", quality=quality)
        out.seek(0)
        resaved = Image.open(out)

        original_np = np.array(ela_rgb).astype(np.float32)
        resaved_np = np.array(resaved).astype(np.float32)
        diff = np.abs(original_np - resaved_np)
        mean_diff = float(np.mean(diff))
        # High global diff on noisy camera frames is not a splice indicator
        if mean_diff > 18.0:
            return {
                "ela_score": round(min(mean_diff / 4.0, 22.0), 1),
                "ela_heatmap": "",
                "ela_anomaly_detected": False,
                "ela_ratio": 1.0,
                "mean_diff": round(mean_diff, 3),
                "note": "high_sensor_noise_baseline",
            }

        ela_vis = np.clip(diff * amplify, 0, 255).astype(np.uint8)
        ela_gray = np.mean(ela_vis, axis=2)

        gray = cv2.cvtColor(np.array(ela_rgb), cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        flat_mask = edges < np.percentile(edges, 40)
        edge_mask = edges > np.percentile(edges, 80)

        ela_flat = float(np.mean(ela_gray[flat_mask])) if np.any(flat_mask) else 0.0
        ela_edge = float(np.mean(ela_gray[edge_mask])) if np.any(edge_mask) else 1e-8
        ratio = ela_edge / (ela_flat + 1e-8)

        # High ELA in uniform (flat) areas suggests manipulation
        flat_ela_high = ela_flat > 8.0 and ratio < 2.5
        ela_anomaly = flat_ela_high or ratio < 2.2 or mean_diff > 6.0

        ela_score = 0.0
        if ratio < 2.5:
            ela_score += min((2.5 - ratio) * 35, 55)
        if flat_ela_high:
            ela_score += min(ela_flat * 2.5, 45)
        if mean_diff > 4.0:
            ela_score += min((mean_diff - 4.0) * 8, 30)
        ela_score = round(min(max(ela_score, 0.0), 100.0), 1)

        _, buffer = cv2.imencode(".png", cv2.cvtColor(ela_vis, cv2.COLOR_RGB2BGR))
        heatmap_b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return {
            "ela_score": ela_score,
            "ela_heatmap": heatmap_b64,
            "ela_anomaly_detected": ela_anomaly,
            "ela_ratio": round(ratio, 3),
            "mean_diff": round(mean_diff, 3),
        }
    except Exception as e:
        return {
            "ela_score": 50.0,
            "ela_heatmap": "",
            "ela_anomaly_detected": False,
            "error": str(e),
        }


def detect_double_jpeg(cv_img_bgr: np.ndarray) -> dict[str, Any]:
    """Detect double JPEG compression via DCT block periodicity (FFT on AC coefficients)."""
    try:
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        if h8 < 64 or w8 < 64:
            return {
                "double_compression_detected": False,
                "compression_score": 0.0,
            }

        gray8 = cv2.resize(gray[:h8, :w8], (min(512, w8), min(512, h8)), interpolation=cv2.INTER_AREA)
        hh, ww = gray8.shape
        hh8, ww8 = (hh // 8) * 8, (ww // 8) * 8
        gray8 = gray8[:hh8, :ww8].astype(np.float64)

        ac_hist = []
        for r in range(0, hh8, 8):
            for c in range(0, ww8, 8):
                block = gray8[r : r + 8, c : c + 8]
                dct_block = cv2.dct(block)
                ac = dct_block.flatten()[1:]
                ac_hist.extend(ac.tolist())

        ac_arr = np.array(ac_hist, dtype=np.float64)
        if ac_arr.size < 1000:
            return {"double_compression_detected": False, "compression_score": 0.0}

        # Periodicity in quantized coefficient histogram
        hist, _ = np.histogram(ac_arr, bins=64)
        hist = hist.astype(np.float64)
        hist -= hist.mean()
        spectrum = np.abs(np.fft.rfft(hist))
        if spectrum.size < 4:
            return {"double_compression_detected": False, "compression_score": 0.0}

        peak_ratio = float(np.max(spectrum[2:]) / (np.mean(spectrum[1:]) + 1e-8))
        double_detected = peak_ratio > 3.8
        compression_score = round(min(max((peak_ratio - 2.5) * 22, 0), 100), 1)

        return {
            "double_compression_detected": double_detected,
            "compression_score": compression_score,
            "dct_peak_ratio": round(peak_ratio, 3),
        }
    except Exception as e:
        return {
            "double_compression_detected": False,
            "compression_score": 0.0,
            "error": str(e),
        }


def run_layer1_forensic(
    image_bytes: bytes,
    pil_img: Image.Image,
    cv_img_bgr: np.ndarray,
    metadata_result: dict,
    is_messaging_compressed: bool = False,
    is_pristine_digital: bool = False,
    is_web_downloaded: bool = False,
) -> dict[str, Any]:
    """Run full Layer 1 pipeline."""
    exif = analyze_exif_forensic(
        image_bytes,
        metadata_result=metadata_result,
        is_messaging_compressed=is_messaging_compressed,
        is_pristine_digital=is_pristine_digital,
        is_web_downloaded=is_web_downloaded,
    )
    c2pa = check_c2pa(image_bytes)
    ela = analyze_ela_forensic(pil_img)
    djpeg = detect_double_jpeg(cv_img_bgr)

    metadata_score = float(exif["exif_score"])
    if c2pa.get("has_c2pa") and c2pa.get("ai_tool_used"):
        metadata_score = max(metadata_score, 85.0)

    return {
        "metadata_score": round(metadata_score, 1),
        "exif": exif,
        "c2pa": c2pa,
        "ela": ela,
        "double_jpeg": djpeg,
        "forensic_signals": {
            "exif_anomaly": exif["exif_anomaly"],
            "has_c2pa": c2pa["has_c2pa"],
            "c2pa_producer": c2pa.get("c2pa_producer"),
            "double_compression": djpeg["double_compression_detected"],
            "ela_anomaly_detected": ela.get("ela_anomaly_detected", False),
        },
    }
