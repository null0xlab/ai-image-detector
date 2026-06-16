/** Estimated self-hosted costs (USD) — adjust for your region & GPU tier */

export const COST_PER_REQUEST = 0.0045

export const VOLUME_TIERS = [
  { label: 'Per request', count: 1, cost: COST_PER_REQUEST },
  { label: '100 requests', count: 100, cost: COST_PER_REQUEST * 100 },
  { label: '500 requests', count: 500, cost: COST_PER_REQUEST * 500 },
  { label: '1,000 requests', count: 1000, cost: COST_PER_REQUEST * 1000 },
]

export const MONTHLY_ESTIMATE = {
  requests: 25000,
  total: COST_PER_REQUEST * 25000,
  label: '25k requests / month (moderate API usage)',
}

export const COST_BREAKDOWN = [
  { name: 'AI Inference', pct: 58, monthly: 65.25, note: 'HF + CLIP GPU/CPU compute per image' },
  { name: 'Hosting', pct: 18, monthly: 20.25, note: 'VM or container orchestration' },
  { name: 'Database', pct: 6, monthly: 6.75, note: 'Job queue & API keys (optional Redis/SQLite)' },
  { name: 'Storage', pct: 5, monthly: 5.63, note: 'Model weights & temp buffers' },
  { name: 'Bandwidth', pct: 8, monthly: 9.0, note: 'Image upload/download egress' },
  { name: 'Monitoring', pct: 5, monthly: 5.63, note: 'Logs, metrics, alerting (optional)' },
]

export const COMPARISONS = [
  {
    name: 'null0xlab (self-hosted)',
    type: 'Open source',
    monthly: 112.5,
    perReq: 0.0045,
    performance: 92,
    scalability: 'High — horizontal workers + GPU',
    resources: '2–4 vCPU, 8GB RAM, optional GPU',
  },
  {
    name: 'Commercial API (avg.)',
    type: 'SaaS',
    monthly: 399,
    perReq: 0.016,
    performance: 88,
    scalability: 'Managed — vendor limits',
    resources: 'None (vendor-hosted)',
  },
  {
    name: 'Basic open-source only',
    type: 'OSS heuristic',
    monthly: 45,
    perReq: 0.0018,
    performance: 62,
    scalability: 'Medium — single node',
    resources: '1 vCPU, 4GB RAM',
  },
  {
    name: 'Enterprise forensic suite',
    type: 'Commercial',
    monthly: 1200,
    perReq: 0.048,
    performance: 94,
    scalability: 'High — licensed clusters',
    resources: 'Dedicated infrastructure',
  },
]

export function formatUsd(n) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(n)
}
