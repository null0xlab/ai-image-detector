import React from 'react'

const FOOTER_LINKS = [
  { href: '/api/guide', label: 'API Docs' },
  { href: '#cost-analysis', label: 'Costs' },
  { href: '#privacy', label: 'Privacy' },
  { href: '#version', label: 'Version' },
]

function SiteFooter() {
  return (
    <footer className="border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#030712] relative z-[1]">
      <div className="max-w-6xl mx-auto px-4 py-10">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6 text-center md:text-left">
          <div>
            <p className="font-outfit font-bold text-slate-800 dark:text-slate-200">AI Image Detector</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-md">
              Multi-signal forensic ensemble — HF classifier, CLIP semantics, pixel forensics, and deepfake analysis.
            </p>
          </div>
          <div className="flex flex-wrap justify-center gap-4 text-xs font-semibold text-slate-500 dark:text-slate-400">
            {FOOTER_LINKS.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
              >
                {link.label}
              </a>
            ))}
          </div>
        </div>
        <div className="mt-8 pt-6 border-t border-slate-200 dark:border-slate-800 text-center text-xs text-slate-500 dark:text-slate-400">
          <p>
            Copyrighted <a href="https://github.com/null0xlab" target="_blank" rel="noopener noreferrer" className="text-indigo-650 dark:text-indigo-400 hover:text-indigo-500 dark:hover:text-indigo-300 transition-colors font-medium">null0xlab</a>
          </p>
        </div>
      </div>
    </footer>
  )
}

export default SiteFooter
