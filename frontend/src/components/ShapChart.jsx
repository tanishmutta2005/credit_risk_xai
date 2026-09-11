/**
 * ShapChart.jsx — Horizontal bar chart of SHAP feature contributions.
 *
 * Red bars  → features that INCREASE default risk (positive SHAP)
 * Green bars → features that DECREASE default risk (negative SHAP)
 *
 * Uses Recharts BarChart.
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from 'recharts'

// ── Helpers ───────────────────────────────────────────────────────────────────

function cleanFeatureName(name) {
  // Remove sklearn transformer prefixes (e.g. "num__dti" → "DTI")
  return name
    .replace(/^(num__|cat__|num_|cat_)/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const isRisk = d.shap_value > 0
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-lg max-w-xs">
      <p className="font-semibold text-slate-200 mb-1">{d.label}</p>
      <p className={isRisk ? 'text-red-400' : 'text-emerald-400'}>
        SHAP: {d.shap_value > 0 ? '+' : ''}{d.shap_value.toFixed(4)}
      </p>
      <p className="text-slate-500 mt-1">
        {isRisk ? '↑ Increases default risk' : '↓ Reduces default risk'}
      </p>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function ShapChart({ riskFactors = [], positiveFactors = [] }) {
  // Merge and sort by absolute SHAP value
  const all = [
    ...riskFactors.map((f) => ({ ...f, label: cleanFeatureName(f.feature) })),
    ...positiveFactors.map((f) => ({ ...f, label: cleanFeatureName(f.feature) })),
  ].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))

  if (!all.length) {
    return (
      <p className="text-xs text-slate-500 text-center py-4">
        No SHAP data available. Run train.py to generate SHAP explanations.
      </p>
    )
  }

  return (
    <div>
      {/* Legend */}
      <div className="flex gap-4 mb-3 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-red-500/70 inline-block" />
          Increases risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-emerald-500/70 inline-block" />
          Reduces risk
        </span>
      </div>

      <ResponsiveContainer width="100%" height={40 * all.length + 20}>
        <BarChart
          data={all}
          layout="vertical"
          margin={{ top: 0, right: 30, left: 0, bottom: 0 }}
        >
          <XAxis
            type="number"
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={145}
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine x={0} stroke="#475569" strokeWidth={1} />
          <Bar dataKey="shap_value" radius={[0, 4, 4, 0]} barSize={18}>
            {all.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.shap_value > 0
                  ? 'rgba(239,68,68,0.75)'
                  : 'rgba(34,197,94,0.75)'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
