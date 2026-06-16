import React, { useEffect } from 'react'

const STYLES = {
  error: 'border-rose-500/30 bg-rose-50 text-rose-900 dark:bg-rose-950/80 dark:text-rose-100',
  success: 'border-emerald-500/30 bg-emerald-50 text-emerald-900 dark:bg-emerald-950/80 dark:text-emerald-100',
  info: 'border-indigo-500/30 bg-indigo-50 text-indigo-900 dark:bg-indigo-950/80 dark:text-indigo-100',
}

function Toast({ message, type = 'info', onClose }) {
  useEffect(() => {
    if (!message) return undefined
    const t = setTimeout(onClose, 5200)
    return () => clearTimeout(t)
  }, [message, onClose])

  if (!message) return null

  return (
    <div
      role="alert"
      className={`fixed bottom-6 right-6 z-[100] max-w-sm px-4 py-3 rounded-xl border shadow-2xl backdrop-blur-md animate-fade-in-up text-sm font-medium ${STYLES[type] || STYLES.info}`}
    >
      <div className="flex items-start gap-3">
        <p className="flex-1 leading-snug">{message}</p>
        <button
          type="button"
          onClick={onClose}
          aria-label="Dismiss notification"
          className="opacity-60 hover:opacity-100 transition-opacity shrink-0"
        >
          ×
        </button>
      </div>
    </div>
  )
}

export default Toast
