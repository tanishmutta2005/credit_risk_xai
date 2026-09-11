/**
 * FairnessPanel.jsx — Fairness audit visualisation.
 *
 * Shows per-group FNR / FPR / Selection Rate via grouped bar chart
 * plus a summary metrics grid (Demographic Parity, Disparate Impact).
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts'

const MetricBadge = ({ label, value, good, bad }) => {
  // Colour code based on simple thresholds
  const v = Number(value)
  let color = 'text-slate-300'
  if (good != null && bad != null) {
    color = Math.abs(v) <= good ? 'text-emerald-400'
          : Math.abs(v) <= bad  ? 'text-amber-400'
          : 'text-red-400'
  }
  return (
    <div className="bg-slate-800/60 rounded-xl p-3 text-center">
      <p className="text-xs text-slate-500 mb-1 leading-tight">{label}</p>
      <p className={`text-base font-bold tabular-nums ${color}`}>
        {value != null ? Number(value).toFixed(3) : '—'}
      </p>
    </div>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="font-semibold text-slate-200 mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: {Number(p.value).toFixed(3)}
        </p>
      ))}
    </div>
  )
}

export default function FairnessPanel({ groups = [], summary = {} }) {
  if (!groups.length) {
    return (
      <p className="text-xs text-slate-500 text-center py-4">
        No fairness data available. Run train.py to generate fairness audit.
      </p>
    )
  }

  // Transform for Recharts
  const chartData = groups.map((g) => ({
    group: g.group,
    FNR: +(g.false_negative_rate).toFixed(3),
    FPR: +(g.false_positive_rate).toFixed(3),
    'Selection Rate': +(g.selection_rate).toFixed(3),
  }))

  return (
    <div className="space-y-5">
      {/* Summary metrics */}
      {summary && Object.keys(summary).length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricBadge
            label="Demographic Parity Diff"
            value={summary.demographic_parity_difference}
            good={0.05}
            bad={0.15}
          />
          <MetricBadge
            label="Demographic Parity Ratio"
            value={summary.demographic_parity_ratio}
          />
          <MetricBadge
            label="Equalized Odds Diff"
            value={summary.equalized_odds_difference}
            good={0.05}
            bad={0.15}
          />
          <MetricBadge
            label="Disparate Impact Ratio"
            value={summary.disparate_impact_ratio}
          />
        </div>
      )}

      {/* Grouped bar chart */}
      <div>
        <p className="text-xs text-slate-500 mb-2">
          Sensitive attribute: <span className="text-slate-400">{summary.sensitive_attribute || 'age_group'}</span>
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="group"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }}
            />
            <Bar dataKey="FNR" name="False Neg. Rate" fill="rgba(239,68,68,0.7)" radius={[3,3,0,0]} barSize={20} />
            <Bar dataKey="FPR" name="False Pos. Rate" fill="rgba(249,115,22,0.7)" radius={[3,3,0,0]} barSize={20} />
            <Bar dataKey="Selection Rate" fill="rgba(99,102,241,0.7)" radius={[3,3,0,0]} barSize={20} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-group table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left py-2 px-3 text-slate-400 font-medium">Group</th>
              <th className="text-center py-2 px-3 text-slate-400 font-medium">N</th>
              <th className="text-center py-2 px-3 text-slate-400 font-medium">FNR</th>
              <th className="text-center py-2 px-3 text-slate-400 font-medium">FPR</th>
              <th className="text-center py-2 px-3 text-slate-400 font-medium">Sel. Rate</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((g, i) => (
              <tr key={i} className={`border-b border-slate-800/60 ${i % 2 === 0 ? 'bg-slate-800/20' : ''}`}>
                <td className="py-2 px-3 font-medium text-slate-300">{g.group}</td>
                <td className="py-2 px-3 text-center text-slate-400">{g.n}</td>
                <td className="py-2 px-3 text-center tabular-nums text-red-400">
                  {g.false_negative_rate.toFixed(3)}
                </td>
                <td className="py-2 px-3 text-center tabular-nums text-orange-400">
                  {g.false_positive_rate.toFixed(3)}
                </td>
                <td className="py-2 px-3 text-center tabular-nums text-indigo-400">
                  {g.selection_rate.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-slate-600 italic">
        ⚖ Fairness metrics use age_group as a proxy sensitive attribute.
        These should be interpreted carefully and not used for real lending decisions.
      </p>
    </div>
  )
}
