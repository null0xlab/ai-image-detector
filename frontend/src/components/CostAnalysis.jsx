import React from 'react'
import Section from './Section'
import {
  VOLUME_TIERS,
  MONTHLY_ESTIMATE,
  COST_BREAKDOWN,
  COMPARISONS,
  formatUsd,
} from '../data/costAnalysis'

function BarChart({ items, valueKey = 'pct', labelKey = 'name', formatValue }) {
  const max = Math.max(...items.map((i) => i[valueKey]), 1)
  return (
    <div className="space-y-3" role="img" aria-label="Cost breakdown chart">
      {items.map((item) => (
        <div key={item[labelKey]}>
          <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-slate-700 dark:text-slate-300">{item[labelKey]}</span>
            <span className="text-slate-500 dark:text-slate-400 tabular-nums">
              {formatValue ? formatValue(item) : `${item[valueKey]}%`}
            </span>
          </div>
          <div className="h-2.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-700"
              style={{ width: `${(item[valueKey] / max) * 100}%` }}
            />
          </div>
          {item.note && <p className="text-[10px] text-slate-500 dark:text-slate-500 mt-1">{item.note}</p>}
        </div>
      ))}
    </div>
  )
}

function CostAnalysis() {
  const breakdownTotal = COST_BREAKDOWN.reduce((s, i) => s + i.monthly, 0)

  return (
    <Section
      id="cost-analysis"
      title="Cost Analysis & Economic Efficiency"
      subtitle="Estimated operating costs for self-hosted deployment at moderate scale. Figures are illustrative — adjust for your cloud region, GPU tier, and traffic."
    >
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        {VOLUME_TIERS.map((tier) => (
          <div key={tier.label} className="section-card p-4 text-center">
            <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">{tier.label}</p>
            <p className="text-xl sm:text-2xl font-outfit font-extrabold text-slate-900 dark:text-white mt-2 tabular-nums">
              {formatUsd(tier.cost)}
            </p>
            {tier.count > 1 && (
              <p className="text-[10px] text-slate-500 mt-1">{formatUsd(tier.cost / tier.count)} / req</p>
            )}
          </div>
        ))}
      </div>

      <div className="section-card p-5 mb-8 bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border-indigo-500/20">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Monthly cost estimate</p>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">{MONTHLY_ESTIMATE.label}</p>
          </div>
          <p className="text-3xl font-outfit font-extrabold text-slate-900 dark:text-white tabular-nums">
            {formatUsd(MONTHLY_ESTIMATE.total)}
            <span className="text-sm font-normal text-slate-500 dark:text-slate-400"> / mo</span>
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
        <div className="section-card p-6">
          <h3 className="text-sm font-outfit font-bold text-slate-900 dark:text-white mb-5">Cost breakdown (monthly)</h3>
          <BarChart
            items={COST_BREAKDOWN}
            formatValue={(item) => `${item.pct}% · ${formatUsd(item.monthly)}`}
          />
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-4 pt-4 border-t border-slate-100 dark:border-slate-800">
            Component total: <strong className="text-slate-700 dark:text-slate-300">{formatUsd(breakdownTotal)}</strong> / mo
            at ~25k requests (inference-weighted).
          </p>
        </div>

        <div className="section-card p-6">
          <h3 className="text-sm font-outfit font-bold text-slate-900 dark:text-white mb-5">Share of monthly spend</h3>
          <div className="flex flex-wrap gap-3 justify-center py-4">
            {COST_BREAKDOWN.map((item, i) => {
              const colors = ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#818cf8', '#64748b']
              return (
                <div key={item.name} className="flex flex-col items-center">
                  <div
                    className="w-14 h-14 rounded-full flex items-center justify-center text-white text-[10px] font-bold shadow-lg"
                    style={{ background: colors[i % colors.length] }}
                  >
                    {item.pct}%
                  </div>
                  <span className="text-[10px] text-slate-600 dark:text-slate-400 mt-2 text-center max-w-[72px]">{item.name}</span>
                </div>
              )
            })}
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 text-center">
            AI inference dominates spend — optimize with GPU batching and model warmup caching.
          </p>
        </div>
      </div>

      <h3 className="text-lg font-outfit font-bold text-slate-900 dark:text-white mb-4">Comparison with alternatives</h3>
      <div className="overflow-x-auto rounded-2xl border border-slate-200 dark:border-slate-800">
        <table className="w-full text-left text-xs min-w-[640px]">
          <thead>
            <tr className="bg-slate-50 dark:bg-slate-900/80 text-slate-500 dark:text-slate-400 uppercase tracking-wider text-[10px]">
              <th className="px-4 py-3 font-bold">Solution</th>
              <th className="px-4 py-3 font-bold">Type</th>
              <th className="px-4 py-3 font-bold">Est. monthly</th>
              <th className="px-4 py-3 font-bold">$/request</th>
              <th className="px-4 py-3 font-bold">Performance</th>
              <th className="px-4 py-3 font-bold">Scalability</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            {COMPARISONS.map((row, idx) => (
              <tr
                key={row.name}
                className={idx === 0 ? 'bg-indigo-500/5 dark:bg-indigo-500/10' : 'bg-white dark:bg-slate-900/30'}
              >
                <td className="px-4 py-3 font-semibold text-slate-800 dark:text-slate-200">{row.name}</td>
                <td className="px-4 py-3 text-slate-600 dark:text-slate-400">{row.type}</td>
                <td className="px-4 py-3 tabular-nums font-medium">{formatUsd(row.monthly)}</td>
                <td className="px-4 py-3 tabular-nums">{formatUsd(row.perReq)}</td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 max-w-[80px]">
                      <div
                        className="h-full rounded-full bg-emerald-500"
                        style={{ width: `${row.performance}%` }}
                      />
                    </div>
                    <span className="tabular-nums text-slate-600 dark:text-slate-400">{row.performance}%</span>
                  </div>
                </td>
                <td className="px-4 py-3 text-slate-600 dark:text-slate-400 max-w-[140px]">{row.scalability}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid sm:grid-cols-3 gap-4 mt-8">
        {[
          {
            title: 'Cost vs commercial API',
            value: '~72% lower',
            desc: 'At 25k req/mo vs average SaaS forensic API pricing.',
          },
          {
            title: 'Cost-to-performance',
            value: '0.049 $/pt',
            desc: 'USD per performance point (92 score baseline).',
          },
          {
            title: 'Resource efficiency',
            value: 'Moderate',
            desc: 'GPU optional; CPU viable with longer latency.',
          },
        ].map((card) => (
          <div key={card.title} className="section-card p-5">
            <p className="text-[10px] font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">{card.title}</p>
            <p className="text-2xl font-outfit font-extrabold text-slate-900 dark:text-white mt-2">{card.value}</p>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-2">{card.desc}</p>
          </div>
        ))}
      </div>
    </Section>
  )
}

export default CostAnalysis
