import React, { useState } from 'react'

function CopyableCode({ code, label }) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="relative rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#0f1419] overflow-hidden">
      {label && (
        <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 px-4 pt-3">
          {label}
        </p>
      )}
      <button
        type="button"
        onClick={copy}
        className="absolute top-2 right-2 z-10 text-[10px] font-semibold px-2.5 py-1 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:bg-indigo-600 hover:text-white hover:border-indigo-600 transition-colors"
      >
        {copied ? 'Copied' : 'Copy'}
      </button>
      <pre className="code-block text-[10px] p-4 pt-9 overflow-x-auto font-mono m-0 border-0 bg-transparent max-h-none">
        <code>{code}</code>
      </pre>
    </div>
  )
}

export default CopyableCode
