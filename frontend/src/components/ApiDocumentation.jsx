import React, { useState } from 'react'
import Section from './Section'
import CodeExplorer from './CodeExplorer'
import CopyableCode from './CopyableCode'
import Accordion from './Accordion'
import { RESPONSE_SAMPLE } from '../data/apiDocSamples'

const DOC_TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'endpoints', label: 'Endpoints' },
  { id: 'schemas', label: 'Schemas' },
  { id: 'sdk', label: 'SDK' },
  { id: 'faq', label: 'FAQ' },
]

const API_FAQ = [
  {
    question: 'Do I need an API key for the web demo?',
    answer: 'The built-in web UI does not require a key. Programmatic access via REST should send your key in the X-API-Key header.',
  },
  {
    question: 'What is the rate limit?',
    answer: 'Default deployments target 60 requests per minute per API key. Enterprise configurations can raise this limit.',
  },
  {
    question: 'Can I run analysis asynchronously?',
    answer: 'Yes. Pass async_mode=true on POST /api/v2/analyze for large images, then poll GET /api/v2/jobs/{id}.',
  },
  {
    question: 'Which formats are supported?',
    answer: 'JPG, PNG, and WEBP up to 10 MB per request via multipart/form-data.',
  },
]

function MethodBadge({ method }) {
  const isGet = method === 'GET'
  return (
    <span
      className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
        isGet
          ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400'
          : 'bg-indigo-500/15 text-indigo-700 dark:text-indigo-400'
      }`}
    >
      {method}
    </span>
  )
}

function EndpointCard({ method, path, title, description, children }) {
  return (
    <div className="section-card overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800 flex flex-wrap items-center gap-2">
        <MethodBadge method={method} />
        <code className="text-xs font-mono text-slate-800 dark:text-slate-200">{path}</code>
        <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 w-full sm:w-auto sm:ml-auto">{title}</span>
      </div>
      <div className="p-5 space-y-3 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
        <p>{description}</p>
        {children}
      </div>
    </div>
  )
}

function ApiDocumentation() {
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <Section
      id="api-docs"
      title="API Documentation"
      subtitle="Integrate AI image and deepfake detection without leaving the app. REST v2, code samples, and schemas — same theme and navigation as the detector."
    >
      <div className="flex flex-col lg:flex-row gap-4 mb-6 overflow-x-auto pb-1">
        {DOC_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`shrink-0 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
              activeTab === tab.id
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/25'
                : 'section-card text-slate-600 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        <div className="xl:col-span-7 space-y-6">
          {activeTab === 'overview' && (
            <>
              <div className="section-card p-5 space-y-3 text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
                <p>
                  The <strong className="text-slate-800 dark:text-slate-200">AI Image Detector API</strong> exposes
                  enterprise-grade forensic analysis: dual-path AI-generated vs deepfake detection, messenger-aware
                  fusion, and granular signal breakdowns.
                </p>
                <p>
                  Authenticate with <code className="text-xs bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded font-mono">X-API-Key</code>{' '}
                  on all programmatic requests.
                </p>
              </div>
              <div className="grid sm:grid-cols-2 gap-3">
                {[
                  ['Dual-path classification', 'Parallel pipelines for generative AI art and face deepfakes.'],
                  ['Compression robust', 'Adaptive weights when Telegram/WhatsApp strips EXIF.'],
                  ['Rich diagnostics', 'ELA, FFT, PRNU, wavelet, and semantic scores in one response.'],
                  ['Async jobs', 'Background processing for large or high-resolution images.'],
                ].map(([t, d]) => (
                  <div key={t} className="section-card p-4">
                    <p className="text-xs font-bold text-indigo-600 dark:text-indigo-400">{t}</p>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-1.5">{d}</p>
                  </div>
                ))}
              </div>
              <div className="section-card p-5 border-indigo-500/20 bg-indigo-500/5">
                <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-2">Quick start</h3>
                <ol className="list-decimal pl-5 space-y-2 text-xs text-slate-600 dark:text-slate-400">
                  <li>Obtain an API key from your deployment administrator.</li>
                  <li>POST an image to <code className="font-mono">/api/v2/analyze</code> as multipart form data.</li>
                  <li>Parse <code className="font-mono">verdict</code>, <code className="font-mono">confidence</code>, and <code className="font-mono">dual_detection</code> in your app logic.</li>
                </ol>
              </div>
            </>
          )}

          {activeTab === 'endpoints' && (
            <div className="space-y-4">
              <EndpointCard
                method="POST"
                path="/api/v2/analyze"
                title="Full ensemble analysis"
                description="Upload an image file. Returns verdict, confidence, dual_detection, breakdown, heatmap, and forensic signals."
              >
                <p><strong className="text-slate-700 dark:text-slate-300">Body:</strong> multipart — field name <code className="font-mono">file</code></p>
                <p><strong className="text-slate-700 dark:text-slate-300">Errors:</strong> 400 invalid file · 413 over 10MB · 429 rate limit · 5xx server</p>
              </EndpointCard>
              <EndpointCard
                method="GET"
                path="/api/v2/jobs/{job_id}"
                title="Async job status"
                description="Poll when analyze was submitted with async_mode=true."
              />
              <EndpointCard
                method="POST"
                path="/api/analyze"
                title="Legacy v1-shaped response"
                description="Same engine as v2 with backward-compatible JSON structure."
              />
              <div className="section-card p-4 text-xs text-slate-600 dark:text-slate-400">
                <strong className="text-slate-800 dark:text-slate-200">OpenAPI / Swagger:</strong>{' '}
                <a href="/docs" className="text-indigo-600 dark:text-indigo-400 hover:underline font-semibold">
                  /docs
                </a>{' '}
                — interactive explorer on the backend port when running locally.
              </div>
            </div>
          )}

          {activeTab === 'schemas' && (
            <div className="space-y-4">
              <div className="section-card p-5">
                <CopyableCode code={RESPONSE_SAMPLE} label="Success response (v2)" />
              </div>
              <div className="grid sm:grid-cols-2 gap-3 text-xs">
                {[
                  ['verdict', 'AI_GENERATED | AUTHENTIC | DEEPFAKE | UNCERTAIN'],
                  ['confidence', 'Fused synthetic likelihood 0–100'],
                  ['dual_detection', 'Parallel AI vs deepfake scores'],
                  ['breakdown', 'Per-signal HF, CLIP, pixel, deepfake weights'],
                ].map(([k, v]) => (
                  <div key={k} className="section-card p-3">
                    <code className="font-mono text-indigo-600 dark:text-indigo-400">{k}</code>
                    <p className="text-slate-600 dark:text-slate-400 mt-1">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'sdk' && (
            <div className="space-y-4 text-xs text-slate-600 dark:text-slate-400">
              <div className="section-card p-5 space-y-2">
                <p className="font-semibold text-slate-800 dark:text-slate-200">Installation</p>
                <CopyableCode code="pip install requests" />
              </div>
              <div className="section-card p-5 space-y-2">
                <p className="font-semibold text-slate-800 dark:text-slate-200">Best practices</p>
                <ul className="list-disc pl-4 space-y-1">
                  <li>Retry 5xx with exponential backoff; do not retry most 4xx except 429.</li>
                  <li>Set client timeout ≥ 120s for full ensemble on CPU.</li>
                  <li>Never expose API keys in browser-side public JavaScript.</li>
                </ul>
              </div>
            </div>
          )}

          {activeTab === 'faq' && <Accordion items={API_FAQ} />}
        </div>

        <div className="xl:col-span-5">
          <CodeExplorer />
        </div>
      </div>
    </Section>
  )
}

export default ApiDocumentation
