import React from 'react'

function Section({ id, title, subtitle, children, className = '' }) {
  return (
    <section id={id} className={`scroll-mt-28 ${className}`}>
      <div className="mb-8">
        <h2 className="text-2xl sm:text-3xl font-outfit font-extrabold text-slate-900 dark:text-white tracking-tight">
          {title}
        </h2>
        {subtitle && (
          <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-3xl leading-relaxed">{subtitle}</p>
        )}
      </div>
      {children}
    </section>
  )
}

export default Section
