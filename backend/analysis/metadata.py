import exifread
import piexif
import io
from PIL import Image

AI_SIGNATURES = [
    "dall-e", "midjourney", "stable diffusion", "adobe firefly", "firefly",
    "bing image creator", "novelai", "sd xl", "sdxl", "playgroundai",
    "civitai", "leonardo.ai", "craiyon", "flux.1", "imagine ai", "foocus"
]

def analyze_metadata(
    image_bytes: bytes,
    is_messaging_compressed: bool = False,
    is_pristine_digital: bool = False,
    is_web_downloaded: bool = False,
) -> dict:
    """
    Analyzes EXIF metadata for AI signatures or suspicious patterns.
    Returns a dict with:
        score: float (0 - 100)
        details: str
        tags: dict (extracted relevant metadata)
    """
    score = 50.0
    details = "No EXIF data found."
    extracted_tags = {}
    
    try:
        # Convert bytes to file-like object for exifread
        f = io.BytesIO(image_bytes)
        tags = exifread.process_file(f, details=False)
        
        # Also parse with piexif to read software tags robustly
        piexif_dict = None
        try:
            piexif_dict = piexif.load(image_bytes)
        except Exception:
            pass
            
        # 1. Look for explicit AI software signatures
        found_signature = None
        
        # List of tags to scan in exifread
        search_keys = [
            "Image Software", "Image Model", "Image Make", 
            "Image ImageDescription", "EXIF UserComment", 
            "Image Artist", "Image Copyright"
        ]
        
        for k in search_keys:
            if k in tags:
                val = str(tags[k]).lower()
                for sig in AI_SIGNATURES:
                    if sig in val:
                        found_signature = f"{k}: '{str(tags[k])}'"
                        break
            if found_signature:
                break
                
        # Scan raw piexif tags if available
        if not found_signature and piexif_dict:
            for ifd in ("0th", "Exif", "GPS", "1st"):
                if not piexif_dict.get(ifd):
                    continue
                for tag_id, tag_val in piexif_dict[ifd].items():
                    if isinstance(tag_val, bytes):
                        try:
                            val_str = tag_val.decode("utf-8", errors="ignore").lower()
                            for sig in AI_SIGNATURES:
                                if sig in val_str:
                                    found_signature = f"EXIF Tag {tag_id}: '{val_str}'"
                                    break
                        except Exception:
                            pass
                    if found_signature:
                        break
                if found_signature:
                    break
                    
        if found_signature:
            return {
                "score": 100.0,
                "details": f"Explicit AI generator signature detected in image metadata ({found_signature}).",
                "tags": {"ai_signature": found_signature}
            }
            
        # 2. Extract standard camera tags
        make = str(tags.get("Image Make", "")).strip()
        model = str(tags.get("Image Model", "")).strip()
        software = str(tags.get("Image Software", "")).strip()
        f_number = str(tags.get("EXIF FNumber", "")).strip()
        exposure = str(tags.get("EXIF ExposureTime", "")).strip()
        iso = str(tags.get("EXIF ISOSpeedRatings", "")).strip()
        datetime = str(tags.get("Image DateTime", "")).strip()
        has_gps = any(k.startswith("GPS") for k in tags.keys())
        
        if make: extracted_tags["make"] = make
        if model: extracted_tags["model"] = model
        if software: extracted_tags["software"] = software
        if f_number: extracted_tags["f_number"] = f_number
        if exposure: extracted_tags["exposure_time"] = exposure
        if iso: extracted_tags["iso"] = iso
        if datetime: extracted_tags["datetime"] = datetime
        if has_gps: extracted_tags["has_gps"] = True
        
        # 3. Analyze Camera Checklist
        has_camera_indicators = bool(make or model or f_number or exposure or iso)
        
        if has_camera_indicators:
            camera_info = []
            if make: camera_info.append(make)
            if model: camera_info.append(model)
            camera_str = " ".join(camera_info) if camera_info else "Camera"
            
            # Camera tags present -> highly likely authentic photo
            score = 5.0
            details = f"Authentic camera metadata found ({camera_str})."
        else:
            # No camera details present
            if len(tags) == 0:
                if is_web_downloaded:
                    score = 28.0
                    details = (
                        "No EXIF metadata — normal for images saved from websites or search engines; "
                        "this alone does not indicate AI generation."
                    )
                elif is_pristine_digital:
                    score = 72.0
                    details = (
                        "Large pristine digital file with zero camera EXIF — can indicate "
                        "ChatGPT/DALL-E exports; corroborated by other forensic signals."
                    )
                elif is_messaging_compressed:
                    score = 18.0
                    details = (
                        "No EXIF metadata — normal for Telegram/WhatsApp/Messenger shares. "
                        "This does not indicate AI; pixel forensics drive the verdict."
                    )
                else:
                    score = 68.0
                    details = "Metadata is completely empty. Suspicious for a high-quality direct upload (usually camera photos retain EXIF)."
            else:
                # Some digital metadata exists (e.g. Photoshop, GIMP, web exports) but no camera hardware info
                if software:
                    score = 60.0
                    details = f"Digital software metadata found ({software}) without camera parameters. Typical for synthetic or edited images."
                else:
                    score = 55.0
                    details = "Image contains basic structural metadata but lacks camera sensor parameters."
                    
    except Exception as e:
        score = 50.0
        details = f"Error scanning image metadata: {str(e)}"
        
    return {
        "score": round(score, 1),
        "details": details,
        "tags": extracted_tags
    }
