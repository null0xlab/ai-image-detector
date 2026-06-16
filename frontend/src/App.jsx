import React, { useState, useEffect } from 'react'
import Dropzone from './components/Dropzone'
import ResultDisplay from './components/ResultDisplay'
import HeatmapViewer from './components/HeatmapViewer'
import FftVisualizer from './components/FftVisualizer'
import SignalCard from './components/SignalCard'
import SiteHeader from './components/SiteHeader'
import SiteFooter from './components/SiteFooter'
import LandingSections from './components/LandingSections'
import CostAnalysis from './components/CostAnalysis'
import Toast from './components/Toast'

const THEME_KEY = 'ai-detector-theme'
const MODE_KEY = 'ai-detector-analysis-mode'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [loadingStatus, setLoadingStatus] = useState('')
  const [results, setResults] = useState(null)
  const [toast, setToast] = useState({ message: '', type: 'info' })
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem(THEME_KEY)
    if (saved === 'light') return false
    if (saved === 'dark') return true
    return true
  })
  const [detectionMode, setDetectionMode] = useState(() => {
    const saved = localStorage.getItem(MODE_KEY) || 'full'
    return saved === 'deepfake' ? 'full' : saved
  })
  const [heatmapOpacity, setHeatmapOpacity] = useState(45)

  const showToast = (message, type = 'info') => setToast({ message, type })
  const clearToast = () => setToast({ message: '', type: 'info' })

  const mapV2ToUi = (v2) => {
    const verdict = v2?.verdict
    const is_uncertain = verdict === 'UNCERTAIN' || verdict === 'SOFTWARE_EDITED'
    const is_ai_generated =
      (verdict === 'AI_GENERATED' || verdict === 'DEEPFAKE') && !is_uncertain
    const is_deepfake = verdict === 'DEEPFAKE'
    const uiVerdict =
      verdict === 'AUTHENTIC'
        ? 'Likely Real Photo'
        : verdict === 'DEEPFAKE'
          ? 'Deepfake Detected'
          : is_uncertain
            ? 'Uncertain / Possibly AI'
            : 'Likely AI-Generated'

    const exif = v2?.layer_details?.layer1?.exif || {}
    const legacy = v2?.layer_details?.legacy_signals || {}
    const pf = v2?.layer_details?.pixel_forensics || {}
    const dfForensics = v2?.layer_details?.deepfake_forensics || {}
    const pfSignalScores = pf?.signal_scores || {}
    const pfDetails = (pf?.all_findings || []).slice(0, 5).join('; ')
    const dfFindings = (dfForensics?.all_findings || []).slice(0, 4).join('; ')
    const dualAi = v2?.dual_detection?.ai_generated?.score ?? 0
    const isAiVerdict = verdict === 'AI_GENERATED' || verdict === 'DEEPFAKE'
    let synthetic = v2?.synthetic_likelihood ?? v2?.confidence ?? 0
    if (isAiVerdict) {
      synthetic = Math.max(synthetic, dualAi, verdict === 'AI_GENERATED' ? v2?.confidence ?? 0 : 0)
    } else if (verdict === 'AUTHENTIC') {
      synthetic = Math.min(synthetic, dualAi, 25)
    }
    const authentic = v2?.authentic_likelihood ?? Math.max(0, 100 - synthetic)
    const weights = v2?.layer_details?.layer2?.weights || {}

    return {
      is_ai_generated,
      is_deepfake,
      confidence: synthetic,
      synthetic_likelihood: synthetic,
      authentic_likelihood: authentic,
      verdict: uiVerdict,
      explanation: v2?.explanation ?? '',
      dual_detection: v2?.dual_detection,
      breakdown: v2?.breakdown,
      heatmap_base64: v2?.heatmap_base64 ?? '',
      fft_base64: v2?.fft_base64 ?? '',
      signals: {
        metadata: {
          score: v2?.breakdown?.metadata_score ?? 50,
          details: exif?.details ?? '',
          tags: exif?.tags || {},
        },
        generative: legacy?.generative ?? { score: 50, details: '' },
        frequency: legacy?.frequency ?? { score: 50, details: '' },
        texture: legacy?.texture ?? { score: 50, details: '' },
        compression: legacy?.compression ?? { score: 50, details: '' },
        visual: legacy?.visual ?? { score: 50, details: '' },
        ml_semantic: {
          score: v2?.breakdown?.hf_ai_score ?? 50,
          details: `Pretrained HF: ${v2?.breakdown?.hf_ai_score ?? '—'} · CLIP AI: ${v2?.breakdown?.clip_ai_score ?? '—'}`,
        },
        pixel_forensics: {
          score: v2?.breakdown?.pixel_forensics_score ?? 50,
          details: pfDetails || 'Pixel-level forensics computed.',
          tags: {
            prnu_score: pfSignalScores?.prnu,
            ca_score: pfSignalScores?.chromatic_aberration,
            wavelet_score: pfSignalScores?.wavelet,
            srm_score: pfSignalScores?.srm_noise,
            grid_score: pfSignalScores?.grid_artifacts,
            strong_signals: pf?.strong_signal_count ?? 0,
            works_without_exif: !!v2?.analysis_context?.metadata_stripped,
          },
        },
        deepfake_forensics: {
          score: v2?.breakdown?.deepfake_score ?? 0,
          details: dfFindings || 'Face forensics computed.',
          tags: {
            face_count: dfForensics?.face_count ?? 0,
            clip_deepfake: v2?.breakdown?.clip_deepfake_score ?? 0,
          },
        },
      },
      compression_details: {
        estimated_quality: v2?.analysis_context?.estimated_jpeg_quality ?? 85,
        analysis_mode: v2?.analysis_context?.analysis_mode ?? 'standard',
        is_compressed_mode: !!v2?.analysis_context?.messenger_shared,
        jpeg_blockiness: 0,
        dynamic_weights: weights,
      },
      model_disagreement: !!v2?.model_disagreement,
      weighted_semantic_score: v2?.weighted_semantic_score,
      forensic_signals: v2?.forensic_signals || {},
    }
  }

  useEffect(() => {
    const root = document.documentElement
    root.classList.add('theme-transitioning')
    root.classList.remove('dark')

    if (isDarkMode) {
      root.classList.add('dark')
      root.style.colorScheme = 'dark'
      localStorage.setItem(THEME_KEY, 'dark')
    } else {
      root.style.colorScheme = 'light'
      localStorage.setItem(THEME_KEY, 'light')
    }

    const timer = window.setTimeout(() => {
      root.classList.remove('theme-transitioning')
    }, 220)

    return () => window.clearTimeout(timer)
  }, [isDarkMode])

  useEffect(() => {
    localStorage.setItem(MODE_KEY, detectionMode)
  }, [detectionMode])

  const handleFileSelected = (file) => {
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResults(null)
  }

  const handleClear = () => {
    setSelectedFile(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl('')
    setResults(null)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setIsAnalyzing(true)
    setResults(null)

    const steps = [
      'Loading pretrained AI detector (HF)...',
      'Running CLIP semantic ensemble...',
      'Computing frequency & wavelet forensics...',
      'Analyzing face boundaries (deepfake path)...',
      'Fusing dual detection scores...',
    ]
    let stepIdx = 0
    setLoadingStatus(steps[0])
    const loadingInterval = setInterval(() => {
      if (stepIdx < steps.length - 1) {
        stepIdx++
        setLoadingStatus(steps[stepIdx])
      }
    }, 800)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('/api/v2/analyze', {
        method: 'POST',
        body: formData,
      })

      clearInterval(loadingInterval)

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'API failed during analysis.')
      }

      const data = await response.json()
      if (data?.analysis_context && data?.layer_details) {
        setResults(mapV2ToUi(data))
      } else {
        const syn = data.synthetic_likelihood ?? data.confidence ?? 0
        setResults({
          ...data,
          synthetic_likelihood: syn,
          authentic_likelihood: data.authentic_likelihood ?? Math.max(0, 100 - syn),
        })
      }
      showToast('Analysis complete.', 'success')
    } catch (err) {
      clearInterval(loadingInterval)
      showToast(err.message || 'Forensics failed. Check that the backend is running.', 'error')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const modeBtn = (mode, label) => (
    <button
      type="button"
      onClick={() => setDetectionMode(mode)}
      className={`text-[11px] px-3 py-1.5 rounded-lg border transition-all ${
        detectionMode === mode
          ? 'bg-indigo-600 border-indigo-500 text-white shadow-sm'
          : 'border-slate-300 text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800'
      }`}
    >
      {label}
    </button>
  )

  return (
    <div className="page-shell font-sans">
      <div className="page-bg" aria-hidden="true" />

      <SiteHeader isDarkMode={isDarkMode} onThemeChange={setIsDarkMode} />

      <main className="flex-1 w-full max-w-6xl mx-auto px-4 py-8 sm:py-10 relative z-[1] isolate">
        <section className="detector-panel" aria-labelledby="detector-heading">
        <div id="detector" className="scroll-mt-28 mb-10 text-center max-w-2xl mx-auto">
          <p className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20 mb-4">
            Enterprise-grade forensic analysis
          </p>
          <h2 id="detector-heading" className="text-3xl sm:text-4xl font-outfit font-extrabold text-slate-900 dark:text-white tracking-tight leading-tight">
            Detect AI images & deepfakes with confidence
          </h2>
          <p className="mt-3 text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
            Upload any photo to run our multi-signal ensemble — robust to Telegram, WhatsApp, and social re-compression.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-5 flex flex-col space-y-6">
            <div className="glass p-6 rounded-2xl shadow-xl flex flex-col space-y-6">
              <div>
                <h2 className="text-lg font-outfit font-bold text-slate-900 dark:text-white">
                  Upload Image
                </h2>
                <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 leading-relaxed">
                  Dual-path detection: pretrained HF classifier + CLIP semantics plus pixel forensics,
                  frequency, and deepfake face analysis.
                </p>
              </div>

              <div className="flex flex-wrap gap-2" role="group" aria-label="Analysis mode">
                {modeBtn('full', 'Full Analysis')}
                {modeBtn('ai', 'AI Image')}
              </div>

              <Dropzone
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                onFileSelected={handleFileSelected}
                onClear={handleClear}
              />

              <button
                id="analyzeBtn"
                onClick={handleAnalyze}
                disabled={!selectedFile || isAnalyzing}
                className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 font-semibold text-white shadow-lg shadow-indigo-500/20 hover:from-indigo-600 hover:to-purple-700 hover:shadow-indigo-500/30 transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                {isAnalyzing && (
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24" aria-hidden>
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                )}
                <span>{isAnalyzing ? 'Analyzing...' : 'Analyze Image'}</span>
              </button>
            </div>
          </div>

          <div className="lg:col-span-7 flex flex-col space-y-6">
            {!isAnalyzing && !results && (
              <div className="glass rounded-2xl shadow-xl p-12 sm:p-16 flex flex-col items-center justify-center text-center space-y-4 min-h-[400px]">
                <div className="w-16 h-16 rounded-2xl bg-slate-100 dark:bg-slate-800/80 flex items-center justify-center text-slate-400">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <h3 className="text-base font-outfit font-bold text-slate-700 dark:text-slate-200">
                  No Image Analyzed
                </h3>
                <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
                  Upload a JPG, PNG, or WEBP and run full dual detection to see verdict, heatmap, and signal breakdown.
                </p>
              </div>
            )}

            {isAnalyzing && (
              <div className="glass rounded-2xl shadow-xl p-12 sm:p-16 flex flex-col items-center justify-center text-center space-y-6 min-h-[400px]" aria-live="polite">
                <div className="relative w-20 h-20">
                  <div className="absolute inset-0 rounded-full border-4 border-slate-200 dark:border-slate-900" />
                  <div className="absolute inset-0 rounded-full border-4 border-t-indigo-500 border-r-purple-500 animate-spin" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">Running forensic ensemble</p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">{loadingStatus}</p>
                </div>
              </div>
            )}

            {!isAnalyzing && results && (
              <div className="flex flex-col space-y-6 animate-fade-in">
                <ResultDisplay results={results} detectionMode={detectionMode} />

                <div className="glass p-6 rounded-2xl shadow-xl flex flex-col space-y-4">
                  <div className="flex justify-between items-center flex-wrap gap-2">
                    <h3 className="text-sm font-outfit font-bold text-slate-900 dark:text-white">
                      Interactive Heatmap & Fourier Spectrum
                    </h3>
                    <div className="flex items-center space-x-2">
                      <span className="text-[11px] text-slate-500">Overlay</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={heatmapOpacity}
                        onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                        aria-label="Heatmap overlay opacity"
                        className="w-24 h-1 rounded-lg cursor-pointer accent-indigo-500 bg-slate-200 dark:bg-slate-700"
                      />
                      <span className="text-[11px] font-bold text-indigo-600 dark:text-indigo-400 w-8">
                        {heatmapOpacity}%
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <HeatmapViewer baseImg={previewUrl} heatmapImg={results.heatmap_base64} opacity={heatmapOpacity} />
                    <FftVisualizer fftImg={results.fft_base64} />
                  </div>
                </div>

                <div className="glass p-6 rounded-2xl shadow-xl flex flex-col space-y-4">
                  <h3 className="text-sm font-outfit font-bold text-slate-900 dark:text-white">
                    Multi-Signal Diagnostics
                  </h3>
                  <div className="flex flex-col space-y-3">
                    {results.signals.ml_semantic && (
                      <SignalCard
                        title="A. Semantic AI Detection (HF + CLIP)"
                        id="ml_semantic"
                        data={results.signals.ml_semantic}
                        weight={results.compression_details.dynamic_weights?.hf_ai ?? 0.35}
                      />
                    )}
                    <SignalCard title="B. Metadata Analysis" id="metadata" data={results.signals.metadata} weight={0.05} />
                    <SignalCard title="C. Frequency (FFT)" id="frequency" data={results.signals.frequency} weight={0.1} />
                    <SignalCard title="D. Texture (LBP · GLCM)" id="texture" data={results.signals.texture} weight={0.08} />
                    <SignalCard title="E. Compression (ELA · DCT)" id="compression" data={results.signals.compression} weight={0.08} />
                    <SignalCard title="F. Visual Artifacts" id="visual" data={results.signals.visual} weight={0.08} />
                    {results.signals.pixel_forensics && (
                      <SignalCard
                        title="G. Pixel Forensics (PRNU · SRM · Wavelet)"
                        id="pixel_forensics"
                        data={results.signals.pixel_forensics}
                        weight={results.compression_details.dynamic_weights?.pixel_forensics ?? 0.05}
                      />
                    )}
                    {results.signals.deepfake_forensics && (
                      <SignalCard
                        title="H. Deepfake Forensics"
                        id="deepfake_forensics"
                        data={results.signals.deepfake_forensics}
                        weight={0.12}
                      />
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        </section>

        <div className="content-sections">
          <CostAnalysis />
          <LandingSections />
        </div>
      </main>

      <SiteFooter />

      <Toast message={toast.message} type={toast.type} onClose={clearToast} />
    </div>
  )
}

export default App
