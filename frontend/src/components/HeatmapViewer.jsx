import React from 'react'

function HeatmapViewer({ baseImg, heatmapImg, opacity }) {
  return (
    <div className="flex flex-col space-y-2">
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wide">Suspicious Regions Heatmap</span>
      <div className="relative border border-slate-900 rounded-xl overflow-hidden bg-slate-950 aspect-square flex items-center justify-center dark:border-slate-900">
        {/* Base Layer: Original Image */}
        <img 
          className="max-h-full max-w-full object-contain absolute" 
          src={baseImg} 
          alt="Base Source" 
        />
        {/* Overlay Layer: Semi-transparent Heatmap */}
        {heatmapImg && (
          <img 
            className="max-h-full max-w-full object-contain absolute pointer-events-none transition-opacity duration-200" 
            style={{ opacity: opacity / 100 }} 
            src={heatmapImg} 
            alt="Heatmap Overlay" 
          />
        )}
      </div>
      <span className="text-[9px] text-slate-500 leading-normal">
        Highlighted glowing zones indicate spatial anomalies: excessive local smoothness (synthetic skin), depth edge cutoffs (synthetic bokeh), hand anomalies, or LBP irregularities.
      </span>
    </div>
  )
}

export default HeatmapViewer
