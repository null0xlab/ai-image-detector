import React from 'react'

function FftVisualizer({ fftImg }) {
  return (
    <div className="flex flex-col space-y-2">
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wide">Fourier Magnitude Spectrum (FFT)</span>
      <div className="relative border border-slate-900 rounded-xl overflow-hidden bg-slate-950 aspect-square flex items-center justify-center dark:border-slate-900">
        {fftImg ? (
          <img 
            className="max-h-full max-w-full object-contain" 
            src={fftImg} 
            alt="Fourier Spectrum" 
          />
        ) : (
          <div className="text-xs text-slate-500">Fourier spectrum not computed</div>
        )}
      </div>
      <span className="text-[9px] text-slate-500 leading-normal">
        The logarithmic 2D fast Fourier transform. AI-generated images (especially GANs) display symmetrical grid spikes/stars, representing interpolation and resampling anomalies in high frequency bands.
      </span>
    </div>
  )
}

export default FftVisualizer
