import React from 'react'

function ThemeToggle({ isDarkMode, onChange }) {
  return (
    <button
      onClick={() => onChange(!isDarkMode)}
      aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
      title={isDarkMode ? 'Light mode' : 'Dark mode'}
      className="p-2 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800 dark:text-slate-200"
    >
      {isDarkMode ? (
        <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
        </svg>
      ) : (
        <svg className="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m12.728 0l-.707-.707M6.343 6.343l-.707-.707M12 8a4 4 0 100 8 4 4 0 000-8z"></path>
        </svg>
      )}
    </button>
  )
}

export default ThemeToggle
