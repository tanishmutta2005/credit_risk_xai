/**
 * CounterfactualTable.jsx — Side-by-side comparison table.
 *
 * Shows the minimal feature changes needed to flip the prediction
 * from high-risk (Default) → low-risk (Fully Paid).
 *
 * Change direction:
 *   Positive change (increase) → blue arrow
 *   Negative change (decrease) → green arrow (usually protective)
 */

import { ArrowUp, ArrowDown } from 'lucide-react'

function cleanName(name) {
  return name
    .replace(/^(num__|cat__|num_|cat_)/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function fmt(val) {
  if (val == null) return '—'
  const n = Number(val)
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) < 1 && n !== 0) return n.toFixed(3)
  return n.toFixed(2)
}

export default function CounterfactualTable({ changes = [] }) {
  if (!changes.length) {
    return (
      <p className="text-xs text-slate-500 text-center py-4">
        No counterfactual changes found (applicant may already be low-risk).
      </p>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="text-left py-2 px-3 text-xs text-slate-400 font-medium">Feature</th>
            <th className="text-right py-2 px-3 text-xs text-slate-400 font-medium">Current</th>
            <th className="text-center py-2 px-3 text-xs text-slate-400 font-medium">Change</th>
            <th className="text-right py-2 px-3 text-xs text-slate-400 font-medium">Target</th>
          </tr>
        </thead>
        <tbody>
          {changes.map((row, i) => {
            const delta = row.change
            const isDecrease = delta < 0
            return (
              <tr
                key={i}
                className={`border-b border-slate-800/60 ${i % 2 === 0 ? 'bg-slate-800/20' : ''}`}
              >
                {/* Feature name */}
                <td className="py-2.5 px-3 font-medium text-slate-200">
                  {cleanName(row.feature)}
                </td>

                {/* Original value */}
                <td className="py-2.5 px-3 text-right text-slate-400 tabular-nums">
                  {fmt(row.original)}
                </td>

                {/* Delta */}
                <td className="py-2.5 px-3 text-center">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-mono
                      ${isDecrease
                        ? 'bg-emerald-900/40 text-emerald-400'
                        : 'bg-blue-900/40 text-blue-400'}`}
                  >
                    {isDecrease
                      ? <ArrowDown className="w-3 h-3" />
                      : <ArrowUp className="w-3 h-3" />}
                    {isDecrease ? '' : '+'}{fmt(delta)}
                  </span>
                </td>

                {/* Counterfactual value */}
                <td className="py-2.5 px-3 text-right font-semibold tabular-nums text-slate-100">
                  {fmt(row.counterfactual)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <p className="mt-3 text-xs text-slate-600 italic">
        ℹ Counterfactual values represent the minimal changes that could move this applicant
        to a lower-risk classification. Results are indicative only.
      </p>
    </div>
  )
}
