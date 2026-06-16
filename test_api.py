import os
import sys
import requests
import numpy as np
from PIL import Image, ImageDraw

def create_mock_synthetic_image(output_path="mock_synthetic.jpg"):
    """
    Generates a mock 'synthetic' image programmatically.
    It has a smooth sterile color gradient and thin vertical stripes (interpolation grids)
    designed to trigger texture smoothness and frequency anomalies in the detector.
    """
    print(f"[*] Generating mock synthetic image at: {output_path}")
    size = 512
    img = Image.new("RGB", (size, size), "#1f2937")
    draw = ImageDraw.Draw(img)
    
    # Draw sterile smooth gradients
    for y in range(size):
        r = int(15 + (y / size) * 45)
        g = int(25 + (y / size) * 55)
        b = int(50 + (y / size) * 80)
        draw.line([(0, y), (size, y)], fill=(r, g, b))
        
    # Draw unnatural high-frequency vertical lines (interpolation artifacts simulation)
    # This leaves strong grid peaks in the FFT power spectrum
    for x in range(0, size, 16):
        draw.line([(x, 0), (x, size)], fill=(20, 32, 60), width=1)
        
    # Add a smooth blurred center (mock face zone)
    face = Image.new("RGB", (120, 120), "#d97706")
    img.paste(face, (196, 196))
    
    img.save(output_path, "JPEG", quality=95)
    return output_path

def test_api(image_path, server_url="http://localhost:8000"):
    """
    Uploads an image to the FastAPI forensic endpoint and prints a detailed diagnostic summary.
    """
    url = f"{server_url}/api/analyze"
    print(f"[*] Connecting to forensics server at: {url}")
    print(f"[*] Uploading and analyzing: {image_path} ({os.path.getsize(image_path) / 1024:.1f} KB)")
    
    if not os.path.exists(image_path):
        print(f"[!] Error: Image path '{image_path}' does not exist.")
        return False
        
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(url, files=files)
            
        if response.status_code != 200:
            print(f"[!] Server returned error code {response.status_code}: {response.text}")
            return False
            
        res = response.json()
        
        # Verify JSON keys are present
        required_keys = ["is_ai_generated", "confidence", "verdict", "signals", "heatmap_base64", "fft_base64", "explanation"]
        missing_keys = [k for k in required_keys if k not in res]
        if missing_keys:
            print(f"[!] Verification Failed: Missing JSON response fields: {missing_keys}")
            return False
            
        print("\n" + "="*50)
        print("          FORENSIC ANALYSIS REPORT")
        print("="*50)
        print(f" Verdict      : {res['verdict'].upper()}")
        print(f" Confidence   : {res['confidence']}%")
        print(f" AI Generated : {res['is_ai_generated']}")
        print(f" Mode         : {res['compression_details']['is_compressed_mode'] and 'COMPRESSED (Adaptive Weights)' or 'STANDARD'}")
        print(f" Est. Quality : {res['compression_details']['estimated_quality']}")
        print("-"*50)
        print(" SIGNAL BREAKDOWN:")
        
        for name, sig in res["signals"].items():
            score = sig["score"]
            details = sig["details"]
            weight = res["compression_details"]["dynamic_weights"][name]
            
            # Map visual slider
            bar_len = int(score / 5)
            bar = "[" + "#"*bar_len + "-"*(20-bar_len) + "]"
            print(f"  - {name.capitalize():11} (Wt: {weight*100:4.1f}%): {score:5.1f}% {bar}")
            print(f"    Detail: {details}")
            
        print("-"*50)
        print(" EXPLANATION:")
        print(f"  {res['explanation']}")
        print("-"*50)
        
        # Verify image formats
        has_heatmap = res['heatmap_base64'].startswith("data:image/png;base64,")
        has_fft = res['fft_base64'].startswith("data:image/png;base64,")
        print(f" Heatmap payload: {has_heatmap and 'VALID (Base64 PNG)' or 'INVALID'}")
        print(f" Fourier payload: {has_fft and 'VALID (Base64 PNG)' or 'INVALID'}")
        print("="*50 + "\n")
        
        print("[+] Test completed successfully!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("[!] Connection Error: Is the FastAPI server running? Run 'python backend/main.py' first.")
        return False
    except Exception as e:
        print(f"[!] Test failed with exception: {str(e)}")
        return False

if __name__ == "__main__":
    server = "http://localhost:8000"
    
    # If a path was passed in via CLI
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Default fallback: generate mock and test
        target = create_mock_synthetic_image()
        
    success = test_api(target, server)
    
    # Clean up mock file if created
    if len(sys.argv) == 1 and os.path.exists("mock_synthetic.jpg"):
        try:
            os.remove("mock_synthetic.jpg")
            print("[*] Cleaned up temporary mock synthetic image.")
        except Exception:
            pass
            
    sys.exit(0 if success else 1)
