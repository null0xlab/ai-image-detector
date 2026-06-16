export const CODE_TABS = [
  { id: 'curl', label: 'cURL' },
  { id: 'js', label: 'JS' },
  { id: 'python', label: 'Python' },
  { id: 'node', label: 'Node.js' },
]

export const CODE_SAMPLES = {
  curl: `curl -X POST "http://localhost:8000/api/v2/analyze" \\
  -H "X-API-Key: your_secure_api_key" \\
  -F "file=@sample.jpg"`,
  js: `async function analyze(fileObject) {
  const formData = new FormData();
  formData.append("file", fileObject);
  const response = await fetch("/api/v2/analyze", {
    method: "POST",
    headers: { "X-API-Key": "your_api_key" },
    body: formData,
  });
  return await response.json();
}`,
  python: `import requests

def scan_image(path, api_key):
    url = "http://localhost:8000/api/v2/analyze"
    headers = {"X-API-Key": api_key}
    with open(path, "rb") as f:
        files = {"file": (path, f, "image/jpeg")}
        r = requests.post(url, headers=headers, files=files, timeout=120)
    return r.json()`,
  node: `const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const form = new FormData();
form.append('file', fs.createReadStream('./image.jpg'));

axios.post('/api/v2/analyze', form, {
  headers: { ...form.getHeaders(), 'X-API-Key': 'your_api_key' },
  timeout: 120000,
}).then(r => console.log(r.data));`,
}

export const RESPONSE_SAMPLE = `{
  "verdict": "AI_GENERATED",
  "confidence": 87.4,
  "synthetic_likelihood": 87.4,
  "authentic_likelihood": 12.6,
  "dual_detection": {
    "ai_generated": { "score": 87.2, "likely_fake": true },
    "deepfake": { "score": 14.1, "likely_fake": false }
  },
  "breakdown": { "hf_ai_score": 82.1, "clip_ai_score": 76.4 },
  "heatmap_base64": "...",
  "explanation": "Strong semantic AI signals with compression-robust fusion."
}`
