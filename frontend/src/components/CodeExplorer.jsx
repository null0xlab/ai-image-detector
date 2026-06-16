import React, { useState } from 'react'
import { CODE_TABS, CODE_SAMPLES } from '../data/apiDocSamples'

function CodeExplorer({ compact = false }) {
  const [activeTab, setActiveTab] = useState('curl')
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(CODE_SAMPLES[activeTab] || '')
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      /* ignore */
    }
  }

  return (
    <div className={`code-explorer flex flex-col rounded-2xl border border-slate-200 dark:border-slate-800 overflow-hidden ${compact ? '' : 'lg:sticky lg:top-28'}`}>
      <div className="code-explorer-tabs flex flex-wrap gap-1 p-3 border-b border-slate-200 dark:border-slate-800">
        {CODE_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`text-[10px] font-mono font-semibold uppercase px-2.5 py-1.5 rounded-lg transition-all ${
              activeTab === tab.id
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
            }`}
          >
            {tab.label}
          </button>
        ))}
        <button
          type="button"
          onClick={copy}
          className="ml-auto text-[10px] font-semibold px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-indigo-600 hover:text-white hover:border-indigo-600 transition-all"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="code-explorer-body flex-1 p-4 overflow-x-auto text-[11px] leading-relaxed font-mono m-0">
        <code>{CODE_SAMPLES[activeTab]}</code>
      </pre>
    </div>
  )
}

export default CodeExplorer
