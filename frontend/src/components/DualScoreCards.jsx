import React from 'react'

function ScoreBar({ label, score, icon, textClass, barClass }) {
  const pct = Math.min(100, Math.max(0, score))
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900/40">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
          {icon} {label}
        </span>
        <span className={`text-lg font-outfit font-bold ${textClass}`}>{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${barClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-[10px] mt-2 text-slate-500">
        {pct >= 52 ? 'Likely fake' : pct >= 40 ? 'Uncertain' : 'Likely authentic'}
      </p>
    </div>
  )
}

function DualScoreCards({ dual, detectionMode }) {
  if (!dual) return null

  const ai = dual.ai_generated || {}
  const df = dual.deepfake || {}
  const showAi = detectionMode === 'full' || detectionMode === 'ai'
  const showDf = detectionMode === 'full' || detectionMode === 'deepfake'

  if (!showAi && !showDf) return null

  return (
    <div className={`grid gap-4 ${showAi && showDf ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
      {showAi && (
        <ScoreBar
          label="AI-Generated"
          score={ai.score ?? 0}
          icon="🤖"
          textClass="text-rose-700 dark:text-rose-400"
          barClass="bg-rose-500"
        />
      )}
      {showDf && (
        <ScoreBar
          label="Deepfake"
          score={df.score ?? 0}
          icon="👤"
          textClass="text-purple-700 dark:text-purple-400"
          barClass="bg-purple-500"
        />
      )}
    </div>
  )
}

export default DualScoreCards
