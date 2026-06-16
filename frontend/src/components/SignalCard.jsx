import React, { useState } from 'react'

function SignalCard({ title, id, data, weight }) {
  const [isOpen, setIsOpen] = useState(false);

  const score = data.score;
  const details = data.details;
  const tags = data.tags;

  // Set colors based on score
  let barColor = "bg-emerald-500";
  let badgeClass = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
  if (score > 30 && score <= 60) {
    barColor = "bg-amber-500";
    badgeClass = "bg-amber-500/10 text-amber-400 border-amber-500/20";
  } else if (score > 60) {
    barColor = "bg-rose-500";
    badgeClass = "bg-rose-500/10 text-rose-400 border-rose-500/20";
  }

  // Choose icon based on ID
  const getIcon = () => {
    switch(id) {
      case "metadata":
        return <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>;
      case "frequency":
        return <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>;
      case "texture":
        return <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm0 8a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1v-2zm0 8a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1v-2z"></path></svg>;
      case "compression":
        return <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>;
      case "visual":
        return <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg>;
      case "pixel_forensics":
        return <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18"></path></svg>;
      case "deepfake_forensics":
        return <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>;
      default:
        return null;
    }
  };

  return (
    <div className={`border rounded-xl overflow-hidden bg-white border-slate-200 dark:border-slate-800 dark:bg-slate-900 ${isOpen ? 'border-indigo-500/40 dark:border-indigo-500/30' : ''}`}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-5 py-4 flex items-center justify-between text-left hover:bg-slate-50 dark:hover:bg-slate-800/80"
      >
        <div className="flex items-center space-x-3">
          <span className="text-slate-400">{getIcon()}</span>
          <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">{title}</span>
        </div>
        <div className="flex items-center space-x-3">
          <span className="text-[10px] font-bold text-slate-500 dark:text-slate-500 uppercase tracking-wider">Weight: {Math.round(weight * 100)}%</span>
          <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${badgeClass}`}>{score}%</span>
          <svg className={`w-4 h-4 text-slate-400 transform transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
          </svg>
        </div>
      </button>
      
      {isOpen && (
        <div className="p-5 flex flex-col space-y-3 border-t border-slate-900 dark:border-slate-900/60">
          <div className="flex items-center space-x-3">
            <span className="text-[10px] font-bold text-slate-400 uppercase w-20 dark:text-slate-500">Anomaly Index:</span>
            <div className="flex-1 bg-slate-950 dark:bg-slate-950/60 rounded-full h-2 overflow-hidden">
              <div className={`${barColor} h-2 rounded-full`} style={{ width: `${score}%` }}></div>
            </div>
          </div>
          
          <p className="text-xs text-slate-600 leading-relaxed dark:text-slate-400">
            <strong>Assessment:</strong> {details}
          </p>

          {id === "pixel_forensics" && tags && (
            <div className="mt-2.5 p-3 rounded-lg bg-slate-950/70 border border-slate-900 grid grid-cols-2 md:grid-cols-3 gap-2 text-[10px] text-slate-400 dark:bg-slate-950/90 dark:border-slate-900">
              {tags.prnu_score !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">PRNU Noise Score</span><span className={`font-medium ${tags.prnu_score > 40 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.prnu_score}%</span></div>}
              {tags.ca_score !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Chromatic Aberration</span><span className={`font-medium ${tags.ca_score > 40 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.ca_score > 40 ? 'Missing (AI)' : 'Present (Real)'}</span></div>}
              {tags.srm_score !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">SRM Noise Residual</span><span className={`font-medium ${tags.srm_score > 40 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.srm_score}%</span></div>}
              {tags.wavelet_score !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Wavelet Uniformity</span><span className={`font-medium ${tags.wavelet_score > 40 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.wavelet_score}%</span></div>}
              {tags.grid_score !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">GAN Grid Artifacts</span><span className={`font-medium ${tags.grid_score > 40 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.grid_score > 40 ? 'Detected' : 'Not Detected'}</span></div>}
              {tags.strong_signals !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Strong Signals</span><span className={`font-medium ${tags.strong_signals >= 3 ? 'text-rose-400' : tags.strong_signals >= 1 ? 'text-amber-400' : 'text-emerald-400'}`}>{tags.strong_signals} / 7</span></div>}
              {tags.works_without_exif && <div className="col-span-full border-t border-slate-900 pt-2"><span className="font-bold text-indigo-400 block">Metadata-Independent Analysis</span><span className="text-slate-400">PRNU, SRM, wavelet, and chromatic aberration signals detect AI/deepfake images even when metadata is stripped by Telegram, Messenger, or WhatsApp.</span></div>}
            </div>
          )}

          {id === "deepfake_forensics" && tags && (
            <div className="mt-2.5 p-3 rounded-lg bg-slate-950/70 border border-purple-900/30 grid grid-cols-2 md:grid-cols-3 gap-2 text-[10px] text-slate-400 dark:bg-slate-950/90">
              {tags.face_count !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Faces Detected</span><span className={`font-medium ${tags.face_count > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>{tags.face_count}</span></div>}
              {tags.blend_boundary !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Blend Boundary</span><span className={`font-medium ${tags.blend_boundary > 30 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.blend_boundary.toFixed(0)}%</span></div>}
              {tags.face_frequency !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Freq. Mismatch</span><span className={`font-medium ${tags.face_frequency > 30 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.face_frequency.toFixed(0)}%</span></div>}
              {tags.texture_inconsistency !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Texture Mismatch</span><span className={`font-medium ${tags.texture_inconsistency > 30 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.texture_inconsistency.toFixed(0)}%</span></div>}
              {tags.eye_artifacts !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Eye Region Artifacts</span><span className={`font-medium ${tags.eye_artifacts > 30 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.eye_artifacts.toFixed(0)}%</span></div>}
              {tags.clip_deepfake !== undefined && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">CLIP Deepfake Score</span><span className={`font-medium ${tags.clip_deepfake > 40 ? 'text-rose-400' : 'text-emerald-400'}`}>{tags.clip_deepfake.toFixed(0)}%</span></div>}
              <div className="col-span-full border-t border-slate-900 pt-2 text-slate-500">Deepfake detection works purely on pixel-level face analysis — no metadata needed. Detects blend boundaries, noise mismatches, eye artifacts, and frequency domain inconsistencies.</div>
            </div>
          )}

          {id === "metadata" && tags && Object.keys(tags).length > 0 && (
            <div className="mt-2.5 p-3 rounded-lg bg-slate-950/70 border border-slate-900 grid grid-cols-2 md:grid-cols-3 gap-2 text-[10px] text-slate-400 dark:bg-slate-950/90 dark:border-slate-900">
              {tags.make && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Manufacturer</span><span className="font-medium text-slate-300 dark:text-slate-400">{tags.make}</span></div>}
              {tags.model && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Device Model</span><span className="font-medium text-slate-300 dark:text-slate-400">{tags.model}</span></div>}
              {tags.software && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Software/Origin</span><span className="font-medium text-slate-300 dark:text-slate-400 truncate block">{tags.software}</span></div>}
              {tags.f_number && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Aperture</span><span className="font-medium text-slate-300 dark:text-slate-400">{tags.f_number}</span></div>}
              {tags.exposure_time && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Exposure</span><span className="font-medium text-slate-300 dark:text-slate-400">{tags.exposure_time}s</span></div>}
              {tags.iso && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">ISO Rating</span><span className="font-medium text-slate-300 dark:text-slate-400">{tags.iso}</span></div>}
              {tags.datetime && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">Capture Date</span><span className="font-medium text-slate-300 dark:text-slate-400 truncate block">{tags.datetime}</span></div>}
              {tags.has_gps && <div><span className="font-bold text-slate-500 dark:text-slate-600 block">GPS Coordinates</span><span className="font-medium text-slate-300 dark:text-slate-400 text-emerald-400">Embedded</span></div>}
              {tags.ai_signature && (
                <div className="col-span-full border-t border-rose-950/40 pt-2">
                  <span className="font-bold text-rose-500 block">AI Software Tool Tag</span>
                  <span className="font-mono text-rose-400 block break-all text-[9px] bg-rose-500/10 p-1.5 rounded mt-1">{tags.ai_signature}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default SignalCard
