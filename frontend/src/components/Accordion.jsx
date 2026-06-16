import React, { useState } from 'react'

function AccordionItem({ question, answer, isOpen, onToggle }) {
  return (
    <div className="section-card overflow-hidden">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isOpen}
        className="w-full flex items-center justify-between gap-4 px-5 py-4 text-left hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
      >
        <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">{question}</span>
        <span
          className={`shrink-0 w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 transition-transform duration-300 ${isOpen ? 'rotate-45' : ''}`}
        >
          +
        </span>
      </button>
      <div
        className={`grid transition-all duration-300 ease-out ${isOpen ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}
      >
        <div className="overflow-hidden">
          <p className="px-5 pb-4 text-sm text-slate-600 dark:text-slate-400 leading-relaxed">{answer}</p>
        </div>
      </div>
    </div>
  )
}

function Accordion({ items }) {
  const [openIndex, setOpenIndex] = useState(0)

  return (
    <div className="space-y-3">
      {items.map((item, i) => (
        <AccordionItem
          key={item.question}
          question={item.question}
          answer={item.answer}
          isOpen={openIndex === i}
          onToggle={() => setOpenIndex(openIndex === i ? -1 : i)}
        />
      ))}
    </div>
  )
}

export default Accordion
