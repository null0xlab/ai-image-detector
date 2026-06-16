import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import our custom analysis package
from analysis import (
    ModelRegistry,
    load_image,
    estimate_jpeg_quality,
    preprocess_and_denoise,
    analyze_metadata,
    analyze_frequency,
    analyze_texture,
    analyze_compression,
    analyze_visual_and_heatmap,
    analyze_generative,
    run_pixel_forensics,
    classify_upload_context,
    run_v2_pipeline,
    to_json_safe,
)
from analysis.calibration import synthetic_and_authentic_likelihood
from analysis.evidence_fusion import (
    count_authentic_signals,
    count_strong_signals,
    compute_synthetic_likelihood,
    check_explicit_ai_proof,
    messenger_verdict_from_likelihood,
)
from analysis.v1_adapter import v2_to_v1_response
from detection.pipeline_full import run_full_pipeline

# Thread pool for CPU-heavy v2 inference (keeps event loop responsive)
_v2_executor = ThreadPoolExecutor(max_workers=2)
_v2_jobs: dict[str, dict[str, Any]] = {}

app = FastAPI(
    title="AI Image Detector API",
    description="Multi-signal forensic analysis to detect AI-generated images.",
    version="1.0.0"
)

# Enable CORS for local React Vite development on different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limit upload size to 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Mount assets directory if it exists (for production React build inside Docker)
assets_dir = os.path.join(os.path.dirname(__file__), "static", "assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

ela_dir = os.path.join(os.path.dirname(__file__), "static", "ela")
os.makedirs(ela_dir, exist_ok=True)
app.mount("/static/ela", StaticFiles(directory=ela_dir), name="ela_heatmaps")


@app.on_event("startup")
def startup_load_models():
    """Load ML models once at startup (singleton ModelRegistry)."""
    registry = ModelRegistry.get_instance()
    status = registry.load_all()
    print("ModelRegistry load status:", status)


def _validate_upload(file: UploadFile, contents: bytes) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file upload.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format {ext}. Supported formats: JPG, JPEG, PNG, WEBP.",
        )
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File is too large. Maximum allowed size is 10MB.",
        )
    return ext


def _run_v2_job(job_id: str, contents: bytes, filename: str) -> None:
    try:
        registry = ModelRegistry.get_instance()
        result = run_v2_pipeline(contents, filename, registry=registry)
        _v2_jobs[job_id] = {"status": "completed", "result": result}
    except Exception as e:
        _v2_jobs[job_id] = {"status": "failed", "error": str(e)}

@app.post("/api/v2/analyze")
async def analyze_v2(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_mode: bool = Query(False, description="Run heavy ML inference in background"),
):
    """
    4-layer hybrid detection: forensic triage, ML ensemble, AI attribution, explainability.
    Set async_mode=true to receive a job_id and poll GET /api/v2/jobs/{job_id}.
    """
    contents = await file.read()
    _validate_upload(file, contents)
    filename = file.filename or "upload.jpg"

    if async_mode:
        job_id = str(uuid.uuid4())
        _v2_jobs[job_id] = {"status": "processing"}
        background_tasks.add_task(_run_v2_job, job_id, contents, filename)
        return {"job_id": job_id, "status": "processing", "poll_url": f"/api/v2/jobs/{job_id}"}

    try:
        registry = ModelRegistry.get_instance()
        loop = __import__("asyncio").get_event_loop()
        result = await loop.run_in_executor(
            _v2_executor,
            lambda: run_v2_pipeline(contents, filename, registry=registry),
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"v2 analysis failed: {str(e)}",
        )


@app.get("/api/v2/jobs/{job_id}")
async def get_v2_job(job_id: str):
    job = _v2_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "processing":
        return {"job_id": job_id, "status": "processing"}
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error", "Analysis failed"))
    return {"job_id": job_id, "status": "completed", "result": job["result"]}


@app.post("/analyze/full")
async def analyze_full(file: UploadFile = File(...)):
    """
    Section-based multi-pipeline analysis with ELA heatmap, metadata, C2PA, and ensemble voting.
    """
    contents = await file.read()
    _validate_upload(file, contents)
    filename = file.filename or "upload.jpg"

    try:
        registry = ModelRegistry.get_instance()
        loop = __import__("asyncio").get_event_loop()
        result = await loop.run_in_executor(
            _v2_executor,
            lambda: run_full_pipeline(contents, filename, registry=registry),
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Full section analysis failed: {str(e)}",
        )


@app.post("/api/analyze/full")
async def analyze_full_api(file: UploadFile = File(...)):
    return await analyze_full(file)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Legacy endpoint — delegates to v2 ensemble for consistent accuracy."""
    contents = await file.read()
    _validate_upload(file, contents)
    filename = file.filename or "upload.jpg"

    try:
        registry = ModelRegistry.get_instance()
        loop = __import__("asyncio").get_event_loop()
        v2_result = await loop.run_in_executor(
            _v2_executor,
            lambda: run_v2_pipeline(contents, filename, registry=registry),
        )
        return v2_to_v1_response(v2_result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


@app.post("/api/analyze/legacy")
async def analyze_legacy_heuristics(file: UploadFile = File(...)):
    """Original heuristic-only path (kept for regression comparison)."""
    contents = await file.read()
    _validate_upload(file, contents)
    ext = os.path.splitext(file.filename)[1].lower()

    try:
        # 2. Load PIL and OpenCV Images
        pil_img, cv_img = load_image(contents)
        
        # 3. Estimate JPEG Compression Quality
        quality = estimate_jpeg_quality(pil_img)
        
        # 4. Classify upload context (AI export vs web-downloaded real photo)
        has_exif = hasattr(pil_img, "_getexif") and pil_img._getexif() is not None
        ctx = classify_upload_context(cv_img, ext, has_exif, quality)
        analysis_mode = ctx["analysis_mode"]
        is_lossless_origin = ctx["is_lossless_origin"]
        is_messaging_compressed = ctx["is_messaging_compressed"]
        is_pristine_digital = ctx["is_pristine_digital"]
        is_web_downloaded = ctx["is_web_downloaded"]
        
        # 5. Denoise adaptive pre-processing to handle compression noise
        cv_img_denoised = preprocess_and_denoise(cv_img, quality)
        
        # 6. Generative scan first — may override web-photo classification
        generative_res = analyze_generative(
            cv_img, pil_img, contents, is_messaging_compressed=is_messaging_compressed
        )
        if generative_res["score"] >= 45 and not has_exif:
            is_web_downloaded = False
            is_pristine_digital = not is_messaging_compressed
            analysis_mode = "pristine_digital" if is_pristine_digital else "standard"

        metadata_res = analyze_metadata(
            contents,
            is_messaging_compressed=is_messaging_compressed,
            is_pristine_digital=is_pristine_digital,
            is_web_downloaded=is_web_downloaded,
        )
        frequency_res = analyze_frequency(cv_img_denoised)
        texture_res = analyze_texture(
            cv_img_denoised,
            is_web_downloaded=is_web_downloaded,
            is_messaging_compressed=is_messaging_compressed,
        )
        compression_res = analyze_compression(
            pil_img, cv_img_denoised, is_lossless_origin=is_lossless_origin
        )
        
        visual_res, heatmap_b64 = analyze_visual_and_heatmap(
            cv_img, compression_res["score"], texture_res["score"]
        )

        metadata_stripped = bool(ctx.get("metadata_stripped", not has_exif))
        # Pixel-level forensics — works even when metadata is stripped
        pixel_forensics_res = run_pixel_forensics(
            cv_img,
            is_messaging_compressed=bool(is_messaging_compressed or metadata_stripped),
            n_crops=3 if metadata_stripped else 2,
        )
        pf_score = float(pixel_forensics_res["score"])

        # 7. Dynamic weighting by upload context
        # Pixel forensics weight increases for messaging-compressed (no EXIF) images
        if analysis_mode == "pristine_digital":
            w_meta, w_gen, w_freq, w_texture, w_compression, w_visual, w_pf = (
                0.13, 0.22, 0.13, 0.20, 0.09, 0.14, 0.09
            )
        elif analysis_mode == "web_photo":
            w_meta, w_gen, w_freq, w_texture, w_compression, w_visual, w_pf = (
                0.05, 0.20, 0.17, 0.13, 0.20, 0.16, 0.09
            )
        elif analysis_mode == "messaging_compressed" or (
            metadata_stripped and not is_pristine_digital and not is_web_downloaded
        ):
            # Metadata is stripped — lean heavily on pixel forensics + generative + frequency
            w_meta, w_gen, w_freq, w_texture, w_compression, w_visual, w_pf = (
                0.01, 0.16, 0.24, 0.20, 0.13, 0.07, 0.19
            )
        else:
            w_meta, w_gen, w_freq, w_texture, w_compression, w_visual, w_pf = (
                0.14, 0.17, 0.17, 0.16, 0.13, 0.13, 0.10
            )
            
        weight_sum = w_meta + w_gen + w_freq + w_texture + w_compression + w_visual + w_pf
        
        final_score = (
            metadata_res["score"] * w_meta +
            generative_res["score"] * w_gen +
            frequency_res["score"] * w_freq +
            texture_res["score"] * w_texture +
            compression_res["score"] * w_compression +
            visual_res["score"] * w_visual +
            pf_score * w_pf
        ) / weight_sum if weight_sum > 0 else 50.0
        
        # Fusion boost only when generative + forensic signals agree (avoids web photo false positives)
        forensic_scores = [
            frequency_res["score"],
            texture_res["score"],
            compression_res["score"],
            visual_res["score"],
        ]
        strong_forensic = sum(1 for s in forensic_scores if s >= 50)
        gen_details_lower = generative_res["details"].lower()
        has_explicit_ai_proof = (
            generative_res["score"] >= 95
            or metadata_res["score"] >= 100
            or any(
                k in gen_details_lower
                for k in ("openai", "dall-e", "dalle", "chatgpt", "midjourney", "embedded hint")
            )
        )
        strong_ai = has_explicit_ai_proof or (
            generative_res["score"] >= 50
            and (strong_forensic >= 2 or generative_res["score"] >= 75)
        )

        if has_explicit_ai_proof:
            final_score = max(final_score, 94.0)
        elif generative_res["score"] >= 60:
            final_score = max(final_score, 80.0)
        elif strong_ai and is_pristine_digital:
            final_score = max(final_score, 75.0)
            if strong_forensic >= 2:
                final_score = max(final_score, 85.0)
            if generative_res["score"] >= 70:
                final_score = max(final_score, 90.0)
        elif (
            is_pristine_digital
            and texture_res["score"] >= 65
            and (visual_res["score"] >= 28 or frequency_res["score"] >= 45)
        ):
            final_score = max(final_score, 82.0)
        elif is_pristine_digital and generative_res["score"] < 35:
            final_score = min(final_score, 45.0)

        if is_web_downloaded and not has_explicit_ai_proof:
            if generative_res["score"] < 35 and strong_forensic < 2:
                final_score = min(final_score, 20.0)
            elif generative_res["score"] < 50:
                final_score = min(final_score, 32.0)

        # Messenger/Telegram: EXIF stripped — resolve from pixel forensics, not metadata gap
        messenger_note = None
        if (
            (
                analysis_mode == "messaging_compressed"
                or (metadata_stripped and not is_pristine_digital and not is_web_downloaded)
            )
            and not has_explicit_ai_proof
        ):
            pf_strong = pixel_forensics_res.get("strong_signal_count", 0)
            forensic_scores_with_pf = forensic_scores + [pf_score]
            strong_forensic_pf = sum(1 for s in forensic_scores_with_pf if s >= 45)

            if generative_res["score"] < 35 and strong_forensic_pf < 2 and pf_score < 30:
                final_score = min(final_score, 28.0)
            elif pf_score >= 55 and strong_forensic_pf >= 2:
                # Pixel forensics + other signals agree — confident even without EXIF
                final_score = max(final_score, 65.0 + min(pf_score * 0.15, 12.0))
            elif generative_res["score"] >= 55 and strong_forensic_pf >= 2:
                final_score = max(final_score, 65.0)
            elif pf_score >= 45 and generative_res["score"] >= 45:
                final_score = max(final_score, 56.0)
            elif generative_res["score"] >= 48 and strong_forensic_pf >= 1:
                final_score = max(final_score, 52.0)
            elif pf_strong >= 3:
                final_score = max(final_score, 52.0)
            elif generative_res["score"] < 40 and strong_forensic_pf < 2:
                final_score = min(final_score, 32.0)
            messenger_note = (
                "Image appears shared via a messaging app (metadata removed by Telegram/WhatsApp/Messenger). "
                "Verdict is based on pixel-level forensics, sensor noise analysis, texture, and frequency signals — not missing EXIF."
            )

        is_messenger = bool(
            analysis_mode == "messaging_compressed"
            or (metadata_stripped and not is_pristine_digital and not is_web_downloaded)
        )
        explicit_ai = check_explicit_ai_proof(
            metadata_score=float(metadata_res["score"]),
            generative_score=float(generative_res["score"]),
            generative_details=generative_res.get("details", ""),
        )
        signal_map = {
            "generative": float(generative_res["score"]),
            "frequency": float(frequency_res["score"]),
            "texture": float(texture_res["score"]),
            "compression": float(compression_res["score"]),
            "visual": float(visual_res["score"]),
            "pixel_forensics": pf_score,
        }
        if is_messenger:
            for k in ("frequency", "texture", "compression"):
                signal_map[k] = round(signal_map[k] * 0.75, 1)

        npr_hint = float(pixel_forensics_res.get("signal_scores", {}).get("npr", 0))
        final_score = compute_synthetic_likelihood(
            weighted_score=final_score,
            generative_score=float(generative_res["score"]),
            metadata_score=float(metadata_res["score"]),
            pf_score=pf_score,
            npr_score=npr_hint,
            is_messenger=is_messenger,
            has_explicit_ai=explicit_ai,
            hf_ai_score=0.0,
            clip_ai_score=0.0,
        )
        final_score = min(max(final_score, 0.0), 100.0)

        strong_ai = count_strong_signals(signal_map, ai_threshold=52)
        strong_auth = count_authentic_signals(signal_map, auth_threshold=40)

        # 8. Determine Verdict (evidence-based, conservative for messenger shares)
        if is_messenger and not explicit_ai:
            verdict, is_ai_generated = messenger_verdict_from_likelihood(
                final_score,
                strong_ai_signals=strong_ai,
                strong_auth_signals=strong_auth,
                has_explicit_ai=explicit_ai,
                generative_score=float(generative_res["score"]),
            )
        else:
            is_ai_generated = final_score > 48.0
            if final_score <= 25.0:
                verdict = "Likely Real Photo"
                is_ai_generated = False
            elif final_score <= 45.0 and strong_auth >= 2:
                verdict = "Likely Real Photo"
                is_ai_generated = False
            elif final_score <= 45.0:
                verdict = "Uncertain / Possibly AI"
                is_ai_generated = False
            elif final_score >= 58.0:
                verdict = "Likely AI-Generated"
            else:
                verdict = "Uncertain / Possibly AI"
                is_ai_generated = final_score > 50.0

        likelihood = synthetic_and_authentic_likelihood(final_score)
            
        # 9. Generate Natural Language Explanation
        reasons = []
        if is_ai_generated:
            if metadata_res["score"] == 100.0:
                reasons.append("we discovered explicit AI generator signatures in the EXIF tags")
            elif metadata_res["score"] >= 70.0 and is_pristine_digital and strong_ai:
                reasons.append(
                    "it is a pristine digital export with no camera EXIF combined with generative "
                    "rendering signatures"
                )
            elif metadata_res["score"] > 60.0:
                reasons.append("the image is completely stripped of standard hardware camera metadata tags")
            if generative_res["score"] >= 55.0:
                reasons.append(generative_res["details"].replace("Generative indicators: ", "").rstrip("."))
            if frequency_res["score"] > 55.0:
                reasons.append("its Fourier frequency spectrum contains highly artificial, repeating grid peaks indicating convolutional upsampling artifacts")
            if texture_res["score"] > 55.0:
                reasons.append("the surface microstructures are unnaturally smoothed and lack organic, variable camera sensor noise")
            if compression_res["score"] > 55.0:
                reasons.append("it displays a perfectly uniform compression error level and DCT coefficient irregularities that violate photographic physics")
            if visual_res["score"] > 55.0:
                reasons.append("it exhibits digital visual cues, such as over-polished facial boundaries, synthetic bokeh cutoff, or distorted text stroke contours")
                
            if not reasons:
                reasons.append("the statistical fusion of frequency, noise, and compression grids strongly points to synthetic rendering")
                
            explanation = "This image is classified as likely AI-generated because " + ", ".join(reasons[:-1]) + ", and " + reasons[-1] + "."
        elif verdict == "Likely Real Photo" or final_score <= 48.0:
            if messenger_note:
                explanation = (
                    messenger_note
                    + " No strong synthetic fingerprints were found; this is consistent with a real phone/camera photo "
                    "after chat compression. For highest certainty, upload the original file (not re-forwarded)."
                )
            elif final_score <= 42.0:
                explanation = (
                    "Forensic signals are mixed or weak — nothing strongly indicates AI generation. "
                    "Upload a higher-resolution original file for a stronger verdict."
                )
            else:
                reasons = []
                if frequency_res["score"] < 40.0:
                    reasons.append("frequency spectrum looks consistent with natural capture")
                if texture_res["score"] < 40.0:
                    reasons.append("texture/noise variation matches typical camera photos")
                if pf_score < 45.0:
                    reasons.append("pixel-level sensor noise tests do not show strong synthetic signatures")
                if not reasons:
                    reasons.append("overall forensic evidence favors a real photograph")
                explanation = "This image is classified as likely authentic because " + ", ".join(reasons) + "."
        else:
            if metadata_res["score"] < 10.0:
                reasons.append("it retains original hardware camera manufacturer tags and lens settings")
            if frequency_res["score"] < 30.0:
                reasons.append("its Fast Fourier Transform shows a healthy, organic power-law decay with no digital upsampling checkerboards")
            if texture_res["score"] < 30.0:
                reasons.append("micro-textures reveal complex grain and organic, spatially variable sensor noise")
            if compression_res["score"] < 30.0:
                reasons.append("compression error levels vary naturally matching the local sharpness of details")
            if not reasons:
                reasons.append("all major frequency, texture, and pixel-level indicators appear consistent with organic camera capture")
            explanation = "This image is classified as authentic because " + ", ".join(reasons[:-1]) + ", and " + reasons[-1] + "."
            
        # Compile complete response payload (sanitize numpy types for JSON)
        return to_json_safe({
            "is_ai_generated": bool(is_ai_generated),
            "confidence": round(float(final_score), 1),
            "synthetic_likelihood": likelihood["synthetic_likelihood"],
            "authentic_likelihood": likelihood["authentic_likelihood"],
            "verdict": verdict,
            "signals": {
                "metadata": metadata_res,
                "generative": generative_res,
                "frequency": {
                    "score": frequency_res["score"],
                    "details": frequency_res["details"]
                },
                "texture": texture_res,
                "compression": compression_res,
                "visual": visual_res,
                "pixel_forensics": {
                    "score": round(pf_score, 1),
                    "details": (
                        "Pixel-level forensic signals (PRNU sensor noise, chromatic aberration, DCT statistics, "
                        "wavelet subbands, GAN grid artifacts): " +
                        ("; ".join(pixel_forensics_res.get("all_findings", [])[:4]) or "No strong pixel-level artifacts detected.")
                    ),
                    "tags": {
                        "prnu_score": pixel_forensics_res.get("details", {}).get("prnu", {}).get("score", 0),
                        "ca_score": pixel_forensics_res.get("details", {}).get("chromatic_aberration", {}).get("score", 0),
                        "wavelet_score": pixel_forensics_res.get("details", {}).get("wavelet", {}).get("score", 0),
                        "grid_score": pixel_forensics_res.get("details", {}).get("grid_artifacts", {}).get("score", 0),
                        "strong_signals": pixel_forensics_res.get("strong_signal_count", 0),
                        "works_without_exif": True,
                    }
                },
            },
            "heatmap_base64": heatmap_b64,
            "fft_base64": frequency_res["fft_base64"],
            "compression_details": {
                "estimated_quality": int(quality),
                "analysis_mode": analysis_mode,
                "is_compressed_mode": bool(is_messaging_compressed),
                "is_pristine_digital": bool(is_pristine_digital),
                "is_web_downloaded": bool(is_web_downloaded),
                "metadata_stripped": bool(ctx.get("metadata_stripped", not has_exif)),
                "jpeg_blockiness": ctx["blockiness"],
                "dynamic_weights": {
                    "metadata": round(w_meta, 3),
                    "generative": round(w_gen, 3),
                    "frequency": round(w_freq, 3),
                    "texture": round(w_texture, 3),
                    "compression": round(w_compression, 3),
                    "visual": round(w_visual, 3),
                    "pixel_forensics": round(w_pf, 3),
                }
            },
            "explanation": explanation
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during multi-signal forensic analysis: {str(e)}"
        )

# Serve the HTML client at the root /
@app.get("/api/docs.md")
async def serve_api_docs_markdown():
    docs_path = os.path.join(os.path.dirname(__file__), "API_DOCS.md")
    if not os.path.exists(docs_path):
        raise HTTPException(status_code=404, detail="API_DOCS.md not found.")
    return FileResponse(docs_path, media_type="text/markdown; charset=utf-8")


@app.get("/api/guide", response_class=HTMLResponse)
async def serve_api_guide():
    guide_path = os.path.join(os.path.dirname(__file__), "api_guide.html")
    if os.path.exists(guide_path):
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="api_guide.html not found.")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Mini fallback if index.html is missing
        return """
        <html>
            <body style="background:#090d16;color:#e2e8f0;font-family:sans-serif;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;">
                <h1>AI Image Detector API</h1>
                <p>Backend is running successfully! Upload fallback page 'index.html' is missing in /backend directory.</p>
            </body>
        </html>
        """

if __name__ == "__main__":
    # reload=False avoids orphaned worker processes holding port 8000 on Windows
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
