import io
import cv2
import numpy as np
from PIL import Image

def load_image(image_bytes: bytes):
    """
    Loads an image from raw bytes.
    Returns:
        pil_img: PIL Image object
        cv_img_bgr: OpenCV-compatible BGR image (numpy array)
    """
    raw_img = Image.open(io.BytesIO(image_bytes))
    
    # Convert PIL Image (RGB) to OpenCV (BGR)
    cv_img_bgr = cv2.cvtColor(np.array(raw_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return raw_img, cv_img_bgr

def estimate_jpeg_quality(pil_img: Image.Image) -> int:
    """
    Estimates the JPEG quality factor of an image.
    If the image has quantization tables, extracts it directly.
    Otherwise, uses ELA (Error Level Analysis) grid sweep to find the quality level
    that minimizes the difference after a second compression (local minimum property).
    Defaults to 85 if estimation fails or is not a JPEG.
    """
    # 1. Try to extract from quantization tables
    try:
        tables = getattr(pil_img, "quantization", None)
        if tables:
            # Simple heuristic mapping quantization tables to quality factor
            # Based on standard JPEG Annex A tables
            q_sum = 0
            count = 0
            for t_id in tables:
                q_sum += sum(tables[t_id])
                count += len(tables[t_id])
            if count > 0:
                avg_q = q_sum / count
                # Lower quantization values = higher quality
                # Formula approximates standard IJG scale
                estimated = int(clip(100 - (avg_q - 5) * 0.6, 10, 100))
                return estimated
    except Exception:
        pass

    # 2. Heuristic check using ELA search
    # Resaving a JPEG at its original quality leads to a local minimum in pixel change
    try:
        img_rgb = pil_img.copy()
        best_q = 85
        min_diff = float("inf")
        
        # Test a subset of qualities to keep it fast
        for q in [50, 60, 70, 80, 90, 95]:
            out = io.BytesIO()
            img_rgb.save(out, format="JPEG", quality=q)
            out.seek(0)
            resaved = Image.open(out)
            
            diff = np.abs(np.array(img_rgb).astype(float) - np.array(resaved).astype(float))
            mean_diff = np.mean(diff)
            
            if mean_diff < min_diff:
                min_diff = mean_diff
                best_q = q
        return best_q
    except Exception:
        return 85

def preprocess_and_denoise(cv_img_bgr: np.ndarray, quality_factor: int) -> np.ndarray:
    """
    Applies quality-adaptive preprocessing to mitigate compression artifacts 
    while preserving genuine texture/frequency patterns.
    If quality is low (heavy compression), applies bilateral filtering to smooth JPEG grid blocks
    without erasing high frequency edges.
    """
    if cv_img_bgr is None:
        return None
        
    # If the quality is very high, keep it raw to avoid destroying signals
    if quality_factor >= 90:
        return cv_img_bgr.copy()
    
    # Adaptive bilateral filter parameters based on compression level
    # Heavy compression (low quality) -> stronger filtering
    d = 5
    sigma_color = int(clip(35 - (quality_factor * 0.3), 5, 25))
    sigma_space = int(clip(15 - (quality_factor * 0.15), 3, 10))
    
    denoised = cv2.bilateralFilter(cv_img_bgr, d, sigma_color, sigma_space)
    return denoised

def clip(val, minimum, maximum):
    return max(minimum, min(val, maximum))


def normalize_compressed_image(cv_img_bgr: np.ndarray, quality_factor: int) -> np.ndarray:
    """
    Normalizes a heavily compressed image by sharpening and enhancing local contrast
    to bring out visual/frequency artifacts before deep learning inference.
    """
    if cv_img_bgr is None:
        return None
    if quality_factor >= 85:
        return cv_img_bgr.copy()

    # Standard sharpening kernel to restore lost high-frequency textures
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(cv_img_bgr, -1, kernel)

    # Adaptive blend based on quality (lower quality -> more sharpening/contrast enhancement)
    alpha = clip(1.0 - (quality_factor / 100.0), 0.25, 0.75)
    enhanced = cv2.addWeighted(cv_img_bgr, 1.0 - alpha, sharpened, alpha, 0)

    return enhanced


def estimate_jpeg_blockiness(gray: np.ndarray) -> float:
    """Higher values suggest heavy JPEG recompression (typical of web downloads)."""
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 32 or w8 < 32:
        return 0.0
    g = gray[:h8, :w8].astype(np.float64)
    boundary_diff = 0.0
    count = 0
    for c in range(8, w8, 8):
        boundary_diff += np.mean(np.abs(g[:, c] - g[:, c - 1]))
        count += 1
    for r in range(8, h8, 8):
        boundary_diff += np.mean(np.abs(g[r, :] - g[r - 1, :]))
        count += 1
    return float(boundary_diff / max(count, 1))


def classify_upload_context(
    cv_img_bgr: np.ndarray,
    ext: str,
    has_exif: bool,
    quality: int,
) -> dict:
    """
    Distinguish AI pristine exports from web-downloaded real photos (Google Images, etc.).
    """
    h, w = cv_img_bgr.shape[:2]
    pixels = h * w
    max_dim = max(h, w)
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    blockiness = estimate_jpeg_blockiness(gray)

    is_lossless_origin = ext in [".png", ".webp"]
    is_jpeg = ext in [".jpg", ".jpeg"]

    # Tiny thumbnails from Google Images — not messenger shares
    is_web_downloaded = (not has_exif) and (
        pixels < 350_000
        or (max_dim < 720 and pixels < 500_000)
        or (blockiness > 4.2 and max_dim < 900)
    )

    # Telegram / WhatsApp / Messenger: EXIF stripped + JPEG recompression
    is_messaging_compressed = (
        (not has_exif)
        and is_jpeg
        and (not is_web_downloaded)
        and (
            quality < 82
            or blockiness > 2.5
            or (max_dim <= 2048 and pixels <= 2_500_000)
        )
    )

    # Large pristine PNG or very high-res direct export — ChatGPT / DALL-E pattern
    is_pristine_digital = (
        (not has_exif)
        and (not is_messaging_compressed)
        and (not is_web_downloaded)
        and (
            (is_lossless_origin and pixels >= 600_000 and max_dim >= 800)
            or (quality >= 93 and pixels >= 1_800_000 and max_dim >= 1600)
        )
    )

    if is_pristine_digital:
        analysis_mode = "pristine_digital"
    elif is_web_downloaded:
        analysis_mode = "web_photo"
    elif is_messaging_compressed:
        analysis_mode = "messaging_compressed"
    else:
        analysis_mode = "standard"

    return {
        "analysis_mode": analysis_mode,
        "is_lossless_origin": is_lossless_origin,
        "is_messaging_compressed": is_messaging_compressed,
        "is_pristine_digital": is_pristine_digital,
        "is_web_downloaded": is_web_downloaded,
        "metadata_stripped": not has_exif,
        "blockiness": round(blockiness, 2),
        "pixels": pixels,
        "max_dim": max_dim,
    }


def to_json_safe(obj):
    """Convert numpy scalars/arrays to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
