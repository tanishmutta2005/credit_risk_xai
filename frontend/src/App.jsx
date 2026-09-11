/**
 * App.jsx — Root layout and API integration
 *
 * Layout:
 *   Left panel  : ApplicantForm (collects inputs)
 *   Right panel : Results — RiskGauge, ShapChart, CounterfactualTable, FairnessPanel
 */

import { useState } from 'react'
import ApplicantForm from './components/ApplicantForm'
import RiskGauge from './components/RiskGauge'
import ShapChart from './components/ShapChart'
import CounterfactualTable from './components/CounterfactualTable'
import FairnessPanel from './components/FairnessPanel'
import { AlertTriangle, ShieldCheck, Loader2, Activity } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

// ── Default applicant values (matches sample_input.json) ─────────────────────
const DEFAULT_APPLICANT = {
  age: 35,
  employment_status: 'Employed',
  employment_length: 5,
  annual_income: 60000,
  education: 'Bachelor',
  dependents: 1,
  credit_history_length: 8,
  previous_defaults: 0,
  late_payments: 2,
  credit_utilization: 0.65,
  loan_amount: 15000,
  loan_term: 36,
  interest_rate: 14.5,
  loan_purpose: 'debt_consolidation',
  installment: 520,
  existing_debt: 25000,
  monthly_expenses: 3200,
  savings: 2000,
}

export default function App() {
  const [applicant, setApplicant] = useState(DEFAULT_APPLICANT)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // ── Submit applicant to /predict endpoint ─────────────────────────────────
  const handleSubmit = async (formData) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
      setApplicant(formData)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // ── Risk colour helper ────────────────────────────────────────────────────
  const riskColor = (cat) => ({
    Low: 'text-emerald-400',
    Medium: 'text-amber-400',
    High: 'text-red-400',
  }[cat] || 'text-slate-300')

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-screen-2xl mx-auto px-6 py-4 flex items-center gap-3">
          <Activity className="text-blue-400 w-6 h-6" />
          <span className="text-lg font-semibold tracking-tight">
            Credit Risk <span className="text-blue-400">XAI</span> Dashboard
          </span>
          <span className="ml-auto text-xs text-slate-500">
            Research / Education Only — Not for real lending decisions
          </span>
        </div>
      </header>

      <div className="max-w-screen-2xl mx-auto px-4 md:px-6 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-[420px_1fr] gap-6">

          {/* ── Left: Applicant Form ─────────────────────────────────────── */}
          <aside>
            <div className="glass-card p-6 sticky top-24">
              <h2 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-blue-400" />
                Applicant Profile
              </h2>
              <ApplicantForm
                initialValues={DEFAULT_APPLICANT}
                onSubmit={handleSubmit}
                loading={loading}
              />
            </div>
          </aside>

          {/* ── Right: Results Panel ─────────────────────────────────────── */}
          <main className="space-y-5">

            {/* Loading state */}
            {loading && (
              <div className="glass-card p-10 flex flex-col items-center justify-center gap-3 text-slate-400">
                <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
                <p className="text-sm">Analysing applicant profile…</p>
              </div>
            )}

            {/* Error state */}
            {error && !loading && (
              <div className="glass-card p-6 border-red-700/60 bg-red-950/30 flex gap-3">
                <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
                <div>
                  <p className="font-medium text-red-300">API Error</p>
                  <p className="text-sm text-red-400 mt-1">{error}</p>
                  <p className="text-xs text-slate-500 mt-2">
                    Make sure the FastAPI backend is running: <code>uvicorn api:app --reload</code>
                  </p>
                </div>
              </div>
            )}

            {/* Empty state */}
            {!result && !loading && !error && (
              <div className="glass-card p-12 flex flex-col items-center text-center text-slate-500 gap-3">
                <Activity className="w-10 h-10 text-slate-700" />
                <p className="text-sm">Fill in the applicant form and click <strong className="text-slate-400">Assess Risk</strong> to view results.</p>
              </div>
            )}

            {/* Results */}
            {result && !loading && (
              <>
                {/* ── Risk Summary Card ─────────────────────────────────── */}
                <div className="glass-card p-6">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <p className="text-xs text-slate-500 uppercase tracking-widest mb-1">
                        Default Probability
                      </p>
                      <p className={`text-5xl font-bold tabular-nums ${riskColor(result.risk_category)}`}>
                        {(result.default_probability * 100).toFixed(1)}
                        <span className="text-2xl font-normal ml-1">%</span>
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-slate-500 uppercase tracking-widest mb-1">
                        Risk Category
                      </p>
                      <span className={`inline-block px-4 py-1.5 rounded-full text-sm font-semibold badge-${result.risk_category.toLowerCase()}`}>
                        {result.risk_category} Risk
                      </span>
                    </div>
                  </div>

                  {/* Model metrics row */}
                  {result.model_metrics && (
                    <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {[
                        ['ROC-AUC', result.model_metrics.roc_auc],
                        ['F1-Score', result.model_metrics.f1],
                        ['Precision', result.model_metrics.precision],
                        ['Brier Score', result.model_metrics.brier_score],
                      ].map(([label, val]) => (
                        <div key={label} className="bg-slate-800/60 rounded-xl p-3 text-center">
                          <p className="text-xs text-slate-500 mb-1">{label}</p>
                          <p className="text-base font-semibold text-slate-200">
                            {val != null ? Number(val).toFixed(3) : '—'}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Model name + disclaimer */}
                  <p className="mt-3 text-xs text-slate-600">
                    Model: <span className="text-slate-500">{result.model_name}</span>
                    {' · '}
                    {result.disclaimer}
                  </p>
                </div>

                {/* ── Risk Gauge ─────────────────────────────────────────── */}
                <div className="glass-card p-6">
                  <h3 className="text-sm font-semibold text-slate-300 mb-4">Risk Gauge</h3>
                  <RiskGauge probability={result.default_probability} category={result.risk_category} />
                </div>

                {/* ── SHAP Factors ───────────────────────────────────────── */}
                <div className="glass-card p-6">
                  <h3 className="text-sm font-semibold text-slate-300 mb-4">
                    Feature Impact (SHAP)
                  </h3>
                  <ShapChart
                    riskFactors={result.top_risk_factors}
                    positiveFactors={result.top_positive_factors}
                  />
                </div>

                {/* ── Counterfactual ─────────────────────────────────────── */}
                {result.counterfactual_changes?.length > 0 && (
                  <div className="glass-card p-6">
                    <h3 className="text-sm font-semibold text-slate-300 mb-1">
                      Counterfactual Scenario
                    </h3>
                    <p className="text-xs text-slate-500 mb-4">
                      Minimal changes to flip prediction to low-risk
                      {result.counterfactual_probability != null &&
                        ` (estimated probability after changes: ${(result.counterfactual_probability * 100).toFixed(1)}%)`}
                    </p>
                    <CounterfactualTable changes={result.counterfactual_changes} />
                  </div>
                )}

                {/* ── Fairness Panel ─────────────────────────────────────── */}
                {result.fairness_by_group?.length > 0 && (
                  <div className="glass-card p-6">
                    <h3 className="text-sm font-semibold text-slate-300 mb-4">
                      Fairness Audit
                    </h3>
                    <FairnessPanel
                      groups={result.fairness_by_group}
                      summary={result.fairness_summary}
                    />
                  </div>
                )}
              </>
            )}
          </main>
        </div>
      </div>
    </div>
  )
}
