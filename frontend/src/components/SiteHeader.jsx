import React, { useState } from 'react'
import ThemeToggle from './ThemeToggle'

const DOCS_PATH = '/api/guide'

const NAV = [
  { href: '#detector', label: 'Detector' },
  { href: '#cost-analysis', label: 'Costs' },
  { href: '#how-to-use', label: 'Guide' },
  { href: '#faq', label: 'FAQ' },
]

const docsLinkClass =
  'text-xs font-semibold px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-900 hover:bg-indigo-500 hover:text-white dark:hover:bg-indigo-600 dark:hover:text-white text-slate-700 dark:text-slate-300 inline-flex items-center space-x-1.5 shadow-sm'

function SiteHeader({ isDarkMode, onThemeChange }) {
  const [menuOpen, setMenuOpen] = useState(false)

  const navLink = (href, label, onClick) => (
    <a
      href={href}
      onClick={onClick}
      className="px-3 py-2 text-xs font-semibold text-slate-600 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
    >
      {label}
    </a>
  )

  return (
    <header className="site-header">
      <div className="max-w-6xl mx-auto px-4 py-4 flex justify-between items-center gap-4">
        <a href="/" className="flex items-center space-x-3 shrink-0 group">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/20 group-hover:scale-105 transition-transform">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-outfit font-extrabold tracking-tight text-slate-900 dark:text-slate-100">
              AI Image Detector
            </h1>
            <p className="text-[10px] font-semibold tracking-wider uppercase text-slate-500">
              null0xlab · Ensemble v7.2
            </p>
          </div>
        </a>

        <nav className="hidden lg:flex items-center gap-0.5" aria-label="Primary">
          {NAV.map((item) => navLink(item.href, item.label))}
        </nav>

        <div className="flex items-center space-x-2 sm:space-x-3">
          <a href={DOCS_PATH} className={`hidden sm:inline-flex ${docsLinkClass}`} title="API Documentation">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            <span>API Docs</span>
          </a>
          <ThemeToggle isDarkMode={isDarkMode} onChange={onThemeChange} />
          <button
            type="button"
            className="lg:hidden p-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300"
            aria-label="Open menu"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>

      {menuOpen && (
        <nav className="lg:hidden border-t border-slate-200 dark:border-slate-800 px-4 py-3 flex flex-col gap-0.5 bg-white dark:bg-[#030712]" aria-label="Mobile">
          {NAV.map((item) => navLink(item.href, item.label, () => setMenuOpen(false)))}
          <a
            href={DOCS_PATH}
            onClick={() => setMenuOpen(false)}
            className={`mt-1 ${docsLinkClass} justify-center`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            <span>API Documentation</span>
          </a>
        </nav>
      )}
    </header>
  )
}

export default SiteHeader
