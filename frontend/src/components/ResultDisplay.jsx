import React from 'react'
import DualScoreCards from './DualScoreCards'

function ResultDisplay({ results, detectionMode = 'full' }) {
  const dual = results.dual_detection
  const syntheticScore = results.synthetic_likelihood ?? results.confidence ?? 0
  const authenticScore = results.authentic_likelihood ?? Math.max(0, 100 - syntheticScore)
  const isAi = results.is_ai_generated
  const isDeepfake = results.is_deepfake
  const verdict = results.verdict
  const explanation = results.explanation
  const cDetails = results.compression_details || {}
  const isAuthentic = typeof verdict === 'string' && verdict.toLowerCase().includes('real')
  const isUncertain =
    typeof verdict === 'string' && verdict.toLowerCase().includes('uncertain')
  const score = isAuthentic
    ? authenticScore
    : isUncertain
      ? syntheticScore
      : syntheticScore

  const radius = 42
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (score / 100) * circumference

  let scoreGradient = {
    start: '#10b981',
    end: '#14b8a6',
    textGlow: 'text-glow-green text-emerald-400',
    badgeClass:
      'bg-emerald-55 text-emerald-800 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20',
  }

  if (isDeepfake) {
    scoreGradient = {
      start: '#a855f7',
      end: '#7c3aed',
      textGlow: 'text-glow-purple text-purple-400',
      badgeClass:
        'bg-purple-50 text-purple-800 border-purple-200 dark:bg-purple-500/10 dark:text-purple-400 dark:border-purple-500/20',
    }
  } else if (isAi) {
    scoreGradient = {
      start: '#ef4444',
      end: '#dc2626',
      textGlow: 'text-glow-red text-rose-400',
      badgeClass:
        'bg-rose-50 text-rose-800 border-rose-200 dark:bg-rose-500/10 dark:text-rose-400 dark:border-rose-500/20',
    }
  } else if (isAuthentic) {
    scoreGradient = {
      start: '#10b981',
      end: '#14b8a6',
      textGlow: 'text-glow-green text-emerald-400',
      badgeClass:
        'bg-emerald-50 text-emerald-800 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20',
    }
  } else if (score > 30 && score <= 60) {
    scoreGradient = {
      start: '#f59e0b',
      end: '#d97706',
      textGlow: 'text-glow-amber text-amber-400',
      badgeClass:
        'bg-amber-50 text-amber-800 border-amber-200 dark:bg-amber-500/10 dark:text-amber-400 dark:border-amber-500/20',
    }
  } else if (score > 60) {
    scoreGradient = {
      start: '#ef4444',
      end: '#dc2626',
      textGlow: 'text-glow-red text-rose-400',
      badgeClass:
        'bg-rose-50 text-rose-800 border-rose-200 dark:bg-rose-500/10 dark:text-rose-400 dark:border-rose-500/20',
    }
  }

  const title = isDeepfake
    ? 'Deepfake / Face Manipulation'
    : isAi
      ? 'AI-Generated Image'
      : isAuthentic
        ? 'Likely Real Photo'
        : 'Uncertain Image Structure'

  const confidenceLabel = isAuthentic
    ? score >= 70
      ? 'Likely authentic'
      : score >= 50
        ? 'Probably authentic'
        : 'Weak evidence'
    : score >= 70
      ? 'Likely synthetic'
      : score >= 45
        ? 'Possibly synthetic'
        : 'Weak AI signals'

  const layerScores = dual?.layer_scores || results.breakdown || {}
  const hfScore = Number(
    layerScores.semantic_hf ?? results.breakdown?.hf_ai_score ?? 0
  )
  const clipScore = Number(
    layerScores.semantic_clip ?? results.breakdown?.clip_ai_score ?? 0
  )
  const weightedSem = Number(
    results.weighted_semantic_score ?? layerScores.weighted_semantic ?? 0
  )
  const modelDisagreement = !!results.model_disagreement

  return (
    <div className="glass p-6 rounded-2xl shadow-xl flex flex-col space-y-5">
      {modelDisagreement && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-800 dark:text-amber-200 leading-relaxed">
          <span className="font-bold">Models disagree:</span> the pixel classifier (HF) says real (
          {hfScore.toFixed(0)}%) while semantic or spectral paths show mixed synthetic signals. Results
          may be unreliable for messenger-compressed images.
        </div>
      )}
      {dual && detectionMode !== 'hidden' && (
        <DualScoreCards dual={dual} detectionMode={detectionMode} />
      )}

      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
        <div className="md:col-span-5 flex flex-col items-center justify-center py-2">
          <div className="relative w-40 h-40">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r={radius}
                stroke="currentColor"
                strokeWidth="8"
                fill="transparent"
                className="text-slate-200 dark:text-slate-900"
              />
              <circle
                cx="50"
                cy="50"
                r={radius}
                stroke="url(#gaugeGradientReact)"
                strokeWidth="8"
                fill="transparent"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                className="transition-all duration-1000 ease-out"
              />
              <defs>
                <linearGradient id="gaugeGradientReact" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor={scoreGradient.start} />
                  <stop offset="100%" stopColor={scoreGradient.end} />
                </linearGradient>
              </defs>
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-4xl font-outfit font-extrabold tracking-tight text-slate-900 dark:text-slate-100">
                {score}%
              </span>
              <span className="text-[9px] font-bold text-slate-600 uppercase tracking-wider dark:text-slate-500">
                {confidenceLabel}
              </span>
            </div>
          </div>
        </div>

        <div className="md:col-span-7 flex flex-col space-y-3 text-center md:text-left">
          <span
            className={`inline-flex self-center md:self-start px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider shadow-sm border ${scoreGradient.badgeClass}`}
          >
            {verdict}
          </span>
          <h3 className={`text-xl font-outfit font-extrabold text-center md:text-left ${scoreGradient.textGlow}`}>
            {title}
          </h3>
          <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-400">
            {explanation}
          </p>

          <div className="flex flex-wrap items-center gap-2 pt-2 justify-center md:justify-start">
            <span className="text-[10px] px-2 py-1 rounded border text-slate-600 border-slate-200 bg-slate-100 dark:bg-slate-950 dark:border-slate-800 dark:text-slate-400">
              {cDetails.is_compressed_mode ? 'Messenger Mode (no metadata)' : 'Standard Mode'}
            </span>
            <span className="text-[10px] px-2 py-1 rounded border text-slate-600 border-slate-200 bg-slate-100 dark:bg-slate-950 dark:border-slate-800 dark:text-slate-400">
              Quality: {cDetails.estimated_quality ?? '—'}
            </span>
            {hfScore > 0 && (
              <span className="text-[10px] px-2 py-1 rounded border text-indigo-800 border-indigo-200 bg-indigo-50 dark:text-indigo-300 dark:bg-indigo-500/10 dark:border-indigo-500/30">
                HF model: {hfScore.toFixed(0)}%
              </span>
            )}
            {clipScore > 0 && (
              <span className="text-[10px] px-2 py-1 rounded border text-purple-800 border-purple-200 bg-purple-50 dark:text-purple-300 dark:bg-purple-500/10 dark:border-purple-500/30">
                CLIP AI: {clipScore.toFixed(0)}%
              </span>
            )}
            {weightedSem > 0 && (
              <span className="text-[10px] px-2 py-1 rounded border text-slate-700 border-slate-200 bg-slate-50 dark:text-slate-300 dark:bg-slate-900/50 dark:border-slate-700">
                Weighted semantic: {weightedSem.toFixed(0)}%
              </span>
            )}
            {cDetails.is_compressed_mode && (
              <span className="text-[10px] px-2 py-1 rounded border text-blue-800 border-blue-200 bg-blue-50 dark:text-blue-300 dark:bg-blue-900/30 dark:border-blue-500/20">
                Messenger Mode
              </span>
            )}
            {results.forensic_signals?.generator_watermark && (
              <span className="text-[10px] px-2 py-1 rounded border text-amber-800 border-amber-200 bg-amber-50 dark:text-amber-200 dark:bg-amber-500/10 dark:border-amber-500/30">
                Gemini / AI watermark
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultDisplay
