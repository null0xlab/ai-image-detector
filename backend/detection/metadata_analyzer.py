from __future__ import annotations

import io
from typing import Any

import exifread
import piexif

from analysis.layer1_forensic import check_c2pa


def _decode_exif_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
        return text or None
    return str(value).strip() or None


def analyze_image_metadata(image_bytes: bytes) -> dict[str, Any]:
    has_camera_exif = False
    software: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    gps_present = False
    date_created: str | None = None
    suspicious = False
    tags: dict[str, str] = {}

    try:
        exif_tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
        for key, val in exif_tags.items():
            tags[key.split(" ")[-1].lower()] = str(val)
        camera_make = tags.get("make")
        camera_model = tags.get("model")
        software = tags.get("software")
        if any(k in exif_tags for k in ("GPS GPSLatitude", "GPS GPSLongitude")):
            gps_present = True
        for date_key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
            if date_key in exif_tags:
                date_created = str(exif_tags[date_key])
                break
        has_camera_exif = bool(camera_make or camera_model)
    except Exception:
        pass

    try:
        piexif_dict = piexif.load(image_bytes)
        zeroth = piexif_dict.get("0th", {})
        exif_ifd = piexif_dict.get("Exif", {})
        if not camera_make:
            camera_make = _decode_exif_value(zeroth.get(piexif.ImageIFD.Make))
        if not camera_model:
            camera_model = _decode_exif_value(zeroth.get(piexif.ImageIFD.Model))
        if not software:
            software = _decode_exif_value(zeroth.get(piexif.ImageIFD.Software))
        if not date_created:
            date_created = _decode_exif_value(exif_ifd.get(piexif.ExifIFD.DateTimeOriginal))
        if camera_make or camera_model:
            has_camera_exif = True
        gps_ifd = piexif_dict.get("GPS")
        if gps_ifd:
            gps_present = True
    except Exception:
        pass

    c2pa_data = check_c2pa(image_bytes)
    c2pa_result = "No C2PA credentials found"
    if c2pa_data.get("has_c2pa"):
        producer = c2pa_data.get("c2pa_producer") or c2pa_data.get("ai_tool_used")
        if producer:
            c2pa_result = f"C2PA manifest: {producer}"
        else:
            c2pa_result = "C2PA manifest present"
        if c2pa_data.get("ai_tool_used"):
            suspicious = True

    if not has_camera_exif and not c2pa_data.get("has_c2pa"):
        suspicious = True

    ai_software_markers = (
        "midjourney",
        "dall-e",
        "dalle",
        "stable diffusion",
        "firefly",
        "openai",
        "flux",
        "comfyui",
    )
    if software:
        lower = software.lower()
        if any(marker in lower for marker in ai_software_markers):
            suspicious = True

    metadata_score = 0.0
    if not has_camera_exif:
        metadata_score += 45.0
    if not gps_present and not has_camera_exif:
        metadata_score += 10.0
    if suspicious:
        metadata_score += 20.0
    metadata_score = min(metadata_score, 100.0)

    return {
        "has_camera_exif": has_camera_exif,
        "camera_make": camera_make,
        "camera_model": camera_model,
        "software": software,
        "gps_present": gps_present,
        "date_created": date_created,
        "c2pa_result": c2pa_result,
        "c2pa_data": c2pa_data,
        "suspicious": suspicious,
        "metadata_score": round(metadata_score, 2),
        "tags": tags,
    }
