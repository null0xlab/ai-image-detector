import React from 'react'
import Accordion from './Accordion'
import Section from './Section'

function StepCard({ step, title, description }) {
  return (
    <div className="section-card p-5 flex gap-4">
      <div className="shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-outfit font-bold text-sm shadow-lg shadow-indigo-500/25">
        {step}
      </div>
      <div>
        <h3 className="text-sm font-bold text-slate-900 dark:text-white">{title}</h3>
        <p className="text-xs text-slate-600 dark:text-slate-400 mt-1.5 leading-relaxed">{description}</p>
      </div>
    </div>
  )
}

function InfoCard({ icon, title, children }) {
  return (
    <div className="section-card p-5 h-full">
      <div className="text-2xl mb-3" aria-hidden>{icon}</div>
      <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-2">{title}</h3>
      <div className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed space-y-2">{children}</div>
    </div>
  )
}

const FAQ_ITEMS = [
  {
    question: 'Is this detection 100% accurate?',
    answer:
      'No. Scores are probabilistic estimates based on multiple forensic signals. Always treat results as risk indicators, not legal proof of authenticity or forgery.',
  },
  {
    question: 'Does it work on images from Telegram or WhatsApp?',
    answer:
      'Yes. The ensemble is tuned for messenger re-compression: when EXIF metadata is stripped, semantic models (HF + CLIP) receive higher weight while pixel forensics are capped to reduce JPEG false positives.',
  },
  {
    question: 'What file types and sizes are supported?',
    answer: 'JPG, PNG, and WEBP up to 10 MB per upload. Larger files can be processed via the async API jobs endpoint documented in the Developer Portal.',
  },
  {
    question: 'Are my images stored permanently?',
    answer:
      'Images are processed in memory for analysis. They are not persisted to long-term storage by default. See the Privacy & Security section for deployment-specific details.',
  },
  {
    question: 'Can I integrate this into my own app?',
    answer:
      'Yes. Use the REST API at POST /api/v2/analyze with multipart form upload. Full schemas, SDK examples, and authentication are in the API Documentation.',
  },
  {
    question: 'What is the difference between Full and AI Image modes?',
    answer:
      'Full Analysis runs the complete ensemble. AI Image mode emphasizes generative-artifact detection (Midjourney, DALL·E, Flux, etc.).',
  },
]

function LandingSections() {
  return (
    <div className="space-y-12 pb-8">
      <Section
        id="how-to-use"
        title="How to Use"
        subtitle="Analyze any image in four simple steps — no account required for the web demo."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <StepCard
            step="1"
            title="Upload your image"
            description="Drag and drop or click to browse. Supported formats: JPG, PNG, WEBP (max 10 MB)."
          />
          <StepCard
            step="2"
            title="Choose analysis mode"
            description="Select Full Analysis for comprehensive results or AI Image for generative detection."
          />
          <StepCard
            step="3"
            title="Run analysis"
            description="Click Analyze Image. The ensemble runs semantic classifiers, pixel forensics, frequency analysis, and optional face forensics."
          />
          <StepCard
            step="4"
            title="Review results"
            description="Inspect the verdict, confidence score, heatmap overlay, FFT spectrum, and per-signal diagnostic cards."
          />
        </div>
      </Section>

      <Section
        id="how-it-works"
        title="How It Works"
        subtitle="A multi-layer pipeline fuses pretrained vision models with classical digital forensics."
      >
        <div className="section-card p-6 sm:p-8">
          <ol className="space-y-4 text-sm text-slate-600 dark:text-slate-400">
            <li className="flex gap-3">
              <span className="shrink-0 font-mono text-[10px] font-bold px-2 py-1 rounded bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">L1</span>
              <span><strong className="text-slate-800 dark:text-slate-200">Context detection</strong> — Estimates whether the image is pristine, web-sourced, or messenger-compressed (metadata stripped).</span>
            </li>
            <li className="flex gap-3">
              <span className="shrink-0 font-mono text-[10px] font-bold px-2 py-1 rounded bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">L2</span>
              <span><strong className="text-slate-800 dark:text-slate-200">Semantic ensemble</strong> — Hugging Face EfficientNet classifier plus CLIP embeddings detect AI-generated content even after recompression.</span>
            </li>
            <li className="flex gap-3">
              <span className="shrink-0 font-mono text-[10px] font-bold px-2 py-1 rounded bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">L3</span>
              <span><strong className="text-slate-800 dark:text-slate-200">Pixel & frequency forensics</strong> — ELA, DCT, FFT, PRNU, wavelet, LBP texture, and SRM noise patterns surface manipulation artifacts.</span>
            </li>
            <li className="flex gap-3">
              <span className="shrink-0 font-mono text-[10px] font-bold px-2 py-1 rounded bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">L4</span>
              <span><strong className="text-slate-800 dark:text-slate-200">Fusion & explanation</strong> — Dual-path scores (AI-generated vs deepfake) merge into a verdict with GradCAM heatmap and natural-language explanation.</span>
            </li>
          </ol>
        </div>
      </Section>

      <Section
        id="accuracy"
        title="Accuracy Information"
        subtitle="Understand what confidence scores mean and where the system performs best."
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <InfoCard icon="📊" title="Estimated accuracy">
            <p>
              On curated benchmarks, the v7.2 ensemble typically achieves <strong>~85–92%</strong> on clearly AI-generated vs authentic photos. Real-world accuracy varies by source, compression, and generator type.
            </p>
          </InfoCard>
          <InfoCard icon="🎯" title="Confidence score">
            <p>
              The <strong>synthetic likelihood</strong> (0–100%) is a fused probability, not a court-grade certainty. Values above ~70% suggest strong AI signals; 40–70% may indicate uncertainty.
            </p>
          </InfoCard>
          <InfoCard icon="⚠️" title="Important caveat">
            <p>
              Scores are advisory. Do not use as sole evidence for moderation, legal, or journalistic decisions without human review and corroborating context.
            </p>
          </InfoCard>
        </div>
      </Section>

      <Section
        id="why-different"
        title="Why This Project Is Different"
        subtitle="Built for production forensic workflows, not single-model demos."
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            ['Dual-path detection', 'Parallel AI-generated and deepfake pipelines with fused verdicts.'],
            ['Messenger-robust', 'Adaptive weighting when Telegram/WhatsApp strips EXIF metadata.'],
            ['Transparent signals', 'Per-layer diagnostics, heatmaps, and FFT — not a black-box label.'],
            ['Pretrained + forensic', 'Combines Dafilab HF classifier with classical PRNU, ELA, and wavelet analysis.'],
            ['Developer-ready API', 'REST v2 with async jobs, OpenAPI docs, and multi-language examples.'],
            ['Self-hostable', 'Run entirely on your infrastructure with Python + optional GPU acceleration.'],
          ].map(([title, desc]) => (
            <div key={title} className="section-card p-5">
              <h3 className="text-sm font-bold text-indigo-600 dark:text-indigo-400">{title}</h3>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-2 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section
        id="open-source"
        title="Open Source Information"
        subtitle="Inspect, extend, and contribute to the detection stack."
      >
        <div className="section-card p-6 space-y-4 text-sm text-slate-600 dark:text-slate-400">
          <p>
            This project is developed by <strong className="text-slate-800 dark:text-slate-200">null0xlab</strong> as an open, research-oriented forensic toolkit. Core detection logic, API server, and web UI source are available for local deployment and customization.
          </p>
          <ul className="list-disc pl-5 space-y-2 text-xs">
            <li><strong>License:</strong> Use responsibly — detection outputs are probabilistic; see repository LICENSE for terms.</li>
            <li><strong>Repository:</strong> Host your fork on GitHub or GitLab; publish the URL in your deployment README.</li>
            <li><strong>Contributions:</strong> Issues and PRs welcome for model updates, new forensic signals, UI improvements, and API enhancements.</li>
            <li><strong>Models:</strong> Uses open weights (e.g. Dafilab/ai-image-detector on Hugging Face) downloaded on first run.</li>
          </ul>
        </div>
      </Section>

      <Section
        id="tech-stack"
        title="Technology Stack"
        subtitle="Modern, battle-tested tools across the full stack."
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            ['Frontend', 'React 18, Vite, Tailwind CSS'],
            ['Backend', 'FastAPI, Python 3.10+, Uvicorn'],
            ['AI / ML', 'PyTorch, Transformers, CLIP, EfficientNet, OpenCV'],
            ['APIs', 'REST v2, multipart upload, async job polling'],
            ['Forensics', 'ELA, FFT, PRNU, wavelet, LBP, face landmarks'],
            ['Database', 'Optional Redis/SQLite for async jobs (deployment-dependent)'],
            ['Infra', 'Docker-ready, CPU or CUDA GPU, self-hosted or cloud VM'],
            ['Docs', 'Static API guide + OpenAPI / Swagger at /docs'],
          ].map(([label, tech]) => (
            <div key={label} className="section-card p-4">
              <p className="text-[10px] font-bold uppercase tracking-wider text-indigo-500">{label}</p>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-1.5">{tech}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section
        id="privacy"
        title="Privacy & Security"
        subtitle="How your data is handled during analysis."
      >
        <div className="section-card p-6 space-y-3 text-sm text-slate-600 dark:text-slate-400">
          <p><strong className="text-slate-800 dark:text-slate-200">Processing:</strong> Images are decoded in server memory for inference and discarded after the response is sent (default self-hosted behavior).</p>
          <p><strong className="text-slate-800 dark:text-slate-200">Storage:</strong> No persistent image gallery unless you add custom logging. Async job IDs may briefly cache results in memory or Redis.</p>
          <p><strong className="text-slate-800 dark:text-slate-200">Transport:</strong> Use HTTPS in production. API keys should be sent via <code className="text-xs bg-slate-100 dark:bg-slate-800 px-1 rounded">X-API-Key</code> header — never embed in client-side public code.</p>
          <p><strong className="text-slate-800 dark:text-slate-200">Compliance:</strong> Operators are responsible for GDPR/local privacy rules if collecting user uploads at scale.</p>
        </div>
      </Section>

      <Section id="faq" title="Frequently Asked Questions">
        <Accordion items={FAQ_ITEMS} />
      </Section>

      <Section
        id="limitations"
        title="Known Limitations"
        subtitle="Situations where accuracy may decrease."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            'Heavy filters, stickers, or collages obscuring original pixels',
            'Very low resolution or extreme cropping',
            'Novel generators not represented in training data',
            'Authentic images with unusual post-processing (HDR, artistic filters)',
            'Screenshots of screens with moiré patterns',
            'Images with no faces when using Deepfake-only interpretation',
          ].map((item) => (
            <div key={item} className="flex gap-3 section-card p-4 text-xs text-slate-600 dark:text-slate-400">
              <span className="text-amber-500 shrink-0">●</span>
              <span>{item}</span>
            </div>
          ))}
        </div>
      </Section>

      <Section
        id="version"
        title="Version & Updates"
        subtitle="Current release and what is planned next."
      >
        <div className="section-card p-6">
          <div className="flex flex-wrap items-center gap-3 mb-4">
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-indigo-500/15 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20">
              v7.2 — Production Ensemble
            </span>
            <span className="text-xs text-slate-500">Released 2026</span>
          </div>
          <div className="grid md:grid-cols-2 gap-6 text-xs text-slate-600 dark:text-slate-400">
            <div>
              <h3 className="text-sm font-bold text-slate-800 dark:text-slate-200 mb-2">Current highlights</h3>
              <ul className="list-disc pl-4 space-y-1">
                <li>HF + CLIP dual semantic detection</li>
                <li>Messenger-aware adaptive fusion</li>
                <li>Interactive heatmap & FFT visualizations</li>
                <li>Unified light/dark theme across app & API docs</li>
              </ul>
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-800 dark:text-slate-200 mb-2">Planned improvements</h3>
              <ul className="list-disc pl-4 space-y-1">
                <li>Video frame batch analysis API</li>
                <li>Expanded generator fingerprint library</li>
                <li>User dashboard with API key management</li>
                <li>Exportable PDF forensic reports</li>
              </ul>
            </div>
          </div>
        </div>
      </Section>
    </div>
  )
}

export default LandingSections
