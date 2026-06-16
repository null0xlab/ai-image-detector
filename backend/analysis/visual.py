import cv2
import numpy as np
import base64

def analyze_visual_and_heatmap(cv_img_bgr: np.ndarray, ela_score: float, texture_score: float) -> tuple[dict, str]:
    """
    Analyzes visual artifacts:
        - Skin smoothness (over-smoothing of faces)
        - Background blur transitions (synthetic bokeh vs lens physics)
        - Hand/contour convexity defects (deformed fingers)
        - High-contrast text contour coherence (distorted text)
    Generates a composite, transparent glowing RGBA heatmap highlighting 
    suspicious regions.
    Returns:
        visual_analysis_dict: dict
        heatmap_base64: str (base64 transparent PNG)
    """
    h, w, _ = cv_img_bgr.shape
    
    # Standardize working size for visual features
    work_h, work_w = 512, 512
    img_resized = cv2.resize(cv_img_bgr, (work_w, work_h))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Initialize heatmap canvas (Grayscale first, will map to transparent RGBA red)
    heatmap_gray = np.zeros((work_h, work_w), dtype=np.float32)
    
    findings = []
    visual_score = 0.0
    
    # --- 1. Skin & Face Smoothness Check ---
    # Detect skin tones in YCrCb color space
    ycrcb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
    # Standard skin color threshold bounds
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Compute high frequency edge details via Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    abs_laplacian = np.abs(laplacian)
    
    skin_pixels = np.sum(skin_mask > 0)
    skin_ratio = skin_pixels / (work_h * work_w)
    
    # If skin is detected, evaluate its smoothness (variance of high frequency details)
    if skin_pixels > 2000:
        mean_skin_detail = np.mean(abs_laplacian[skin_mask > 0])
        # Real skin has pores/texture -> high detail (mean > 4.0)
        # AI skin is polished -> low detail (mean < 2.0)
        if mean_skin_detail < 3.4:
            smoothness_factor = (3.4 - mean_skin_detail) / 3.4
            smoothness_penalty = min(smoothness_factor * 80, 80)
            visual_score += smoothness_penalty * 0.40
            findings.append(f"hyper-smoothed skin texture (micro-detail level: {mean_skin_detail:.2f})")
            
            # Map smooth skin regions to heatmap
            heatmap_gray[skin_mask > 0] += smoothness_factor * 150
            
    # --- 2. Synthetic Bokeh Blur Boundaries ---
    # Detect blurry regions using a threshold on local variance
    # Divide into 16x16 blocks
    block_size = 32
    blur_map = np.zeros_like(gray, dtype=np.float32)
    
    for r in range(0, work_h, block_size):
        for c in range(0, work_w, block_size):
            block = gray[r:r+block_size, c:c+block_size]
            var = np.var(block)
            if var < 40:  # Low variance = highly blurred/flat
                blur_map[r:r+block_size, c:c+block_size] = 1.0
                
    # Detect transitions (edges of the blur map)
    blur_edges = cv2.Canny(blur_map.astype(np.uint8) * 255, 50, 150)
    
    # Check if there are sharp image borders coinciding with the blur boundaries
    # Natural lenses have gradual blur transitions (low gradient). AI bokeh has a sharp border.
    real_edges = cv2.Canny(gray, 30, 100)
    overlapping_edges = cv2.bitwise_and(blur_edges, real_edges)
    
    overlap_count = np.sum(overlapping_edges > 0)
    if overlap_count > 70:
        bokeh_penalty = min(overlap_count * 0.35, 55.0)
        visual_score += bokeh_penalty * 0.25
        findings.append("sharp synthetic depth-of-field boundary (unnatural background bokeh cutoff)")
        
        # Dilate overlap boundaries and add to heatmap
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated_overlap = cv2.dilate(overlapping_edges, kernel)
        heatmap_gray[dilated_overlap > 0] += 120.0
        
    # --- 3. Hand / Finger Contour Convexity Defects ---
    # Hand contours in skin mask can reveal rendering distortions.
    # Convexity defects measure deviation of contour from its convex hull.
    # AI hands are highly deformed, creating extreme, messy convexity defects.
    try:
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_anomalies = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Focus on mid-sized contours that could be hands (relative to 512x512 size)
            if 1500 < area < 15000:
                hull = cv2.convexHull(cnt, returnPoints=False)
                if len(cnt) > 3 and len(hull) > 3:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        # High number of deep defects represents malformed structure/fingers
                        deep_defects = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            # Distance to hull in pixels (d/256.0)
                            dist = d / 256.0
                            if dist > 15.0:  # Significant protrusion / cavity
                                deep_defects += 1
                                
                        if deep_defects > 6:  # Unusually high for natural hands
                            hand_anomalies += 1
                            # Highlight anomalous contour on heatmap
                            cv2.drawContours(heatmap_gray, [cnt], -1, 140.0, -1)
                            
        if hand_anomalies > 0:
            hand_penalty = min(hand_anomalies * 25.0, 50.0)
            visual_score += hand_penalty * 0.20
            findings.append("irregular structural profiles in extremities (suspicious hand/finger morphology)")
    except Exception:
        pass
        
    # --- 4. Text Rendering Distortions ---
    # Detect high-contrast text-like blocks
    # AI text has bumpy, inconsistent contours. Natural text is composed of perfect line segments/circles.
    # Check bumpy edge contours
    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        distorted_text_count = 0
        
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            # Find small text-like letters
            if 5 < w_box < 60 and 5 < h_box < 60:
                # Calculate solidity: contour area / bounding box area
                solidity = cv2.contourArea(cnt) / (w_box * h_box + 1e-8)
                # Check aspect ratio
                aspect = w_box / (h_box + 1e-8)
                
                # Check perimeter-to-area ratio (AI text has complex/bumpy borders)
                perimeter = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)
                
                if area > 10:
                    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
                    # Natural text strokes have highly regular circularity or high-solidity boxes
                    # Bumpy, jagged AI letters show extremely low circularity and fragmented shapes
                    if circularity < 0.15 and aspect > 0.3 and solidity < 0.4:
                        distorted_text_count += 1
                        # Mark distorted letters on heatmap
                        cv2.rectangle(heatmap_gray, (x, y), (x+w_box, y+h_box), 160.0, -1)
                        
        if distorted_text_count > 6:
            text_penalty = min((distorted_text_count - 6) * 4.5, 55.0)
            visual_score += text_penalty * 0.15
            findings.append("deformed high-contrast typographic contours (distorted synthetic text rendering)")
    except Exception:
        pass
        
    # --- 5. Base Heatmap Compilation ---
    # Merge texture anomaly score and ELA score into heatmap
    # Higher ELA/texture anomaly scores boost the heatmap globally in highly active areas.
    if ela_score > 35 or texture_score > 35:
        # Boost ELA active areas on the heatmap
        # Resize gray to match work_w, work_h
        ela_factor = min((ela_score + texture_score) / 200.0, 0.75)
        # Add high frequency details where texture or ELA is anomalous
        heatmap_gray += (abs_laplacian > np.percentile(abs_laplacian, 85)) * ela_factor * 120.0
        
    # Clip and blur heatmap for glowing, smooth visualization
    heatmap_gray = np.clip(heatmap_gray, 0.0, 255.0).astype(np.uint8)
    heatmap_blurred = cv2.GaussianBlur(heatmap_gray, (25, 25), 0)
    
    # Make color RGBA heatmap: Grayscale intensity represents transparency (Alpha)
    # Color is bright red-orange (RGB: 225, 29, 72)
    h_out, w_out, _ = cv_img_bgr.shape
    heatmap_resized = cv2.resize(heatmap_blurred, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
    
    rgba_heatmap = np.zeros((h_out, w_out, 4), dtype=np.uint8)
    rgba_heatmap[:, :, 0] = 225  # Red
    rgba_heatmap[:, :, 1] = 29   # Green
    rgba_heatmap[:, :, 2] = 72   # Blue
    
    # Map intensity to alpha (max alpha = 140/255, so it is transparent)
    alpha = (heatmap_resized.astype(float) / 255.0) * 140.0
    rgba_heatmap[:, :, 3] = alpha.astype(np.uint8)
    
    # Encode RGBA heatmap to Base64 PNG
    _, buffer = cv2.imencode(".png", rgba_heatmap)
    heatmap_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

    # Global polish: low micro-detail variance across entire frame (DALL-E aesthetic)
    global_detail = float(np.std(abs_laplacian))
    if global_detail < 9.5 and skin_ratio > 0.05:
        polish_penalty = min((9.5 - global_detail) * 5, 40)
        visual_score += polish_penalty * 0.20
        findings.append(f"global digital polish (frame micro-detail σ: {global_detail:.1f})")

    if not findings:
        details = "No significant visual anomalies (extreme skin smoothing, synthetic bokeh boundaries, hand morphs, or text distortions) detected."
    else:
        details = "Anomalies detected: " + ", ".join(findings) + "."

    visual_analysis_dict = {
        "score": round(min(visual_score * 1.4, 100.0), 1),
        "details": details
    }
    
    return visual_analysis_dict, heatmap_base64
