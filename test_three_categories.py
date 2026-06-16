"""
Test the updated pipeline against all three categories:
  1. AI-Generated (no metadata, smooth gradient + grid)
  2. Real/Authentic (noisy JPEG with camera EXIF)
  3. Deepfake (face region with different texture from background)

Also tests metadata-stripped scenario (messaging app simulation).
"""
import cv2
import numpy as np
import io
import piexif
from PIL import Image

from backend.analysis import run_v2_pipeline
from backend.analysis.model_registry import ModelRegistry


def make_ai_image():
    """Smooth gradient PNG, no EXIF — AI-generated."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        img[y, :] = [int(y * 0.4 + 50), int(220 - y * 0.3), int(150 + y * 0.2)]
    # Add regular grid lines (GAN/diffusion upsampling artifact)
    for i in range(0, 512, 16):
        img[i, :] = [100, 100, 200]
        img[:, i] = [100, 100, 200]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def make_real_image():
    """Noisy JPEG with Canon EXIF — authentic real photo."""
    np.random.seed(42)
    img = np.random.randint(20, 240, (600, 800, 3), dtype=np.uint8)
    # Add non-uniform noise (camera-like)
    noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: b"Canon",
            piexif.ImageIFD.Model: b"EOS R5",
        }
    }
    exif_bytes = piexif.dump(exif_dict)
    pil.save(buf, format="JPEG", quality=88, exif=exif_bytes)
    return buf.getvalue()


def make_deepfake_image():
    """JPEG with camera EXIF but over-smooth face region (deepfake-like)."""
    np.random.seed(99)
    img = np.random.randint(30, 200, (480, 640, 3), dtype=np.uint8)
    # Heavy noise everywhere (natural background)
    noise = np.random.normal(0, 18, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Over-smooth face ellipse (very different texture from background)
    face_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.ellipse(face_mask, (320, 200), (90, 120), 0, 0, 360, 255, -1)
    blurred_face = cv2.GaussianBlur(img, (31, 31), 0)
    # Fill face area with solid smooth color
    img = np.where(face_mask[:, :, np.newaxis] > 0, blurred_face, img)
    cv2.ellipse(img, (320, 200), (90, 120), 0, 0, 360, (195, 165, 145), -1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: b"Samsung",
            piexif.ImageIFD.Model: b"Galaxy S23",
        }
    }
    exif_bytes = piexif.dump(exif_dict)
    pil.save(buf, format="JPEG", quality=85, exif=exif_bytes)
    return buf.getvalue()


def make_metadata_stripped_ai():
    """AI image sent through messaging app — no EXIF, JPEG recompressed."""
    img = np.zeros((720, 720, 3), dtype=np.uint8)
    for y in range(720):
        img[y, :] = [int(y * 0.25 + 80), int(200 - y * 0.15), int(180)]
    # Uniform sharpness everywhere (AI signature)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=75)  # messenger recompression, no EXIF
    return buf.getvalue()


def main():
    print("Loading models...")
    registry = ModelRegistry.get_instance()
    registry.load_all()
    print("Load status:", registry.load_status)
    print()

    tests = [
        ("AI-Generated (PNG, no EXIF)", make_ai_image(), "test_ai.png"),
        ("Real Photo (JPEG + camera EXIF)", make_real_image(), "test_real.jpg"),
        ("Deepfake-like (JPEG + camera EXIF, smooth face)", make_deepfake_image(), "test_deepfake.jpg"),
        ("Metadata-stripped AI (messenger JPEG, no EXIF)", make_metadata_stripped_ai(), "test_stripped.jpg"),
    ]

    all_pass = True
    for name, img_bytes, fname in tests:
        result = run_v2_pipeline(img_bytes, fname, registry)
        bd = result["breakdown"]
        ctx = result["analysis_context"]
        df_layer = result.get("layer_details", {}).get("deepfake_forensics", {})
        pf_layer = result.get("layer_details", {}).get("pixel_forensics", {})

        print(f"[{name}]")
        print(f"  Verdict: {result['verdict']}  Confidence: {result['confidence']}")
        print(f"  ML={bd['ml_ensemble_score']:.1f} | DeepfakeScore={bd['deepfake_score']:.1f} | PF={bd['pixel_forensics_score']:.1f}")
        print(f"  CLIP AI={bd.get('clip_ai_score', 0):.1f} | CLIP DF={bd.get('clip_deepfake_score', 0):.1f}")
        print(f"  Context: {ctx['analysis_mode']}  MetaStripped: {ctx['metadata_stripped']}")
        print(f"  DF forensics: {df_layer.get('score', 0):.1f}  Faces: {df_layer.get('face_count', 0)}")
        print(f"  PF strong_signals: {pf_layer.get('strong_signal_count', 0)}")
        print(f"  Labels: {result.get('labels', [])}")
        print()

    print("Pipeline test complete — check verdicts above.")


if __name__ == "__main__":
    main()
