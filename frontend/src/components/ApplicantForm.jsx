/**
 * ApplicantForm.jsx — Input form for all applicant features.
 *
 * Groups:
 *   1. Demographics & Employment
 *   2. Credit History
 *   3. Loan Details
 *   4. Financial Position
 */

import { useState } from 'react'
import { Loader2, Send } from 'lucide-react'

// ── Shared input components ───────────────────────────────────────────────────

const Label = ({ children }) => (
  <label className="block text-xs font-medium text-slate-400 mb-1">{children}</label>
)

const NumberInput = ({ label, name, value, onChange, min, max, step = 1, prefix, suffix }) => (
  <div>
    <Label>{label}</Label>
    <div className="relative">
      {prefix && (
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 text-xs">{prefix}</span>
      )}
      <input
        type="number"
        name={name}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={onChange}
        className={`w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100
          focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500
          ${prefix ? 'pl-7' : ''} ${suffix ? 'pr-10' : ''}`}
      />
      {suffix && (
        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 text-xs">{suffix}</span>
      )}
    </div>
  </div>
)

const SelectInput = ({ label, name, value, onChange, options }) => (
  <div>
    <Label>{label}</Label>
    <select
      name={name}
      value={value}
      onChange={onChange}
      className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100
        focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>{opt.label}</option>
      ))}
    </select>
  </div>
)

const SliderInput = ({ label, name, value, onChange, min, max, step = 0.01, format }) => {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <Label>{label}</Label>
        <span className="text-xs font-mono text-blue-400">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        name={name}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={onChange}
        className="w-full h-1.5 rounded-full appearance-none bg-slate-700 accent-blue-500"
      />
      <div className="flex justify-between text-xs text-slate-600 mt-0.5">
        <span>{format ? format(min) : min}</span>
        <span>{format ? format(max) : max}</span>
      </div>
    </div>
  )
}

const SectionTitle = ({ children }) => (
  <p className="text-xs font-semibold text-blue-400 uppercase tracking-widest mt-5 mb-3 border-b border-slate-800 pb-1">
    {children}
  </p>
)

// ── Main Component ────────────────────────────────────────────────────────────

export default function ApplicantForm({ initialValues, onSubmit, loading }) {
  const [form, setForm] = useState(initialValues)

  const handleChange = (e) => {
    const { name, value, type } = e.target
    setForm((prev) => ({
      ...prev,
      [name]: type === 'number' || type === 'range' ? Number(value) : value,
    }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit(form)
  }

  const pctFmt = (v) => `${(v * 100).toFixed(0)}%`

  return (
    <form onSubmit={handleSubmit} className="space-y-1 max-h-[76vh] overflow-y-auto pr-1">

      {/* ── Demographics ─────────────────────────────────────────────────── */}
      <SectionTitle>Demographics & Employment</SectionTitle>
      <div className="grid grid-cols-2 gap-3">
        <NumberInput label="Age" name="age" value={form.age} onChange={handleChange} min={18} max={100} />
        <NumberInput label="Dependents" name="dependents" value={form.dependents} onChange={handleChange} min={0} max={20} />
      </div>

      <SelectInput
        label="Employment Status"
        name="employment_status"
        value={form.employment_status}
        onChange={handleChange}
        options={[
          { value: 'Employed', label: 'Employed' },
          { value: 'Self-Employed', label: 'Self-Employed' },
          { value: 'Unemployed', label: 'Unemployed' },
          { value: 'Part-Time', label: 'Part-Time' },
        ]}
      />

      <div className="grid grid-cols-2 gap-3">
        <NumberInput
          label="Employment Length (yrs)"
          name="employment_length"
          value={form.employment_length}
          onChange={handleChange}
          min={0} max={50} step={0.5}
        />
        <SelectInput
          label="Education"
          name="education"
          value={form.education}
          onChange={handleChange}
          options={[
            { value: 'None', label: 'None' },
            { value: 'High School', label: 'High School' },
            { value: 'Bachelor', label: 'Bachelor' },
            { value: 'Master', label: 'Master' },
            { value: 'PhD', label: 'PhD' },
          ]}
        />
      </div>

      <NumberInput
        label="Annual Income"
        name="annual_income"
        value={form.annual_income}
        onChange={handleChange}
        min={0} max={500000} step={1000}
        prefix="$"
      />

      {/* ── Credit History ────────────────────────────────────────────────── */}
      <SectionTitle>Credit History</SectionTitle>
      <div className="grid grid-cols-2 gap-3">
        <NumberInput
          label="Credit History (yrs)"
          name="credit_history_length"
          value={form.credit_history_length}
          onChange={handleChange}
          min={0} max={50} step={0.5}
        />
        <NumberInput
          label="Previous Defaults"
          name="previous_defaults"
          value={form.previous_defaults}
          onChange={handleChange}
          min={0} max={10}
        />
      </div>
      <NumberInput
        label="Late Payments"
        name="late_payments"
        value={form.late_payments}
        onChange={handleChange}
        min={0} max={30}
      />
      <SliderInput
        label="Credit Utilization"
        name="credit_utilization"
        value={form.credit_utilization}
        onChange={handleChange}
        min={0} max={1} step={0.01}
        format={pctFmt}
      />

      {/* ── Loan Details ──────────────────────────────────────────────────── */}
      <SectionTitle>Loan Details</SectionTitle>
      <div className="grid grid-cols-2 gap-3">
        <NumberInput
          label="Loan Amount"
          name="loan_amount"
          value={form.loan_amount}
          onChange={handleChange}
          min={500} max={100000} step={500}
          prefix="$"
        />
        <SelectInput
          label="Loan Term (months)"
          name="loan_term"
          value={form.loan_term}
          onChange={handleChange}
          options={[12, 24, 36, 48, 60].map((t) => ({ value: t, label: `${t} mo` }))}
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <NumberInput
          label="Interest Rate"
          name="interest_rate"
          value={form.interest_rate}
          onChange={handleChange}
          min={1} max={35} step={0.5}
          suffix="%"
        />
        <NumberInput
          label="Monthly Installment"
          name="installment"
          value={form.installment}
          onChange={handleChange}
          min={1} max={10000} step={10}
          prefix="$"
        />
      </div>
      <SelectInput
        label="Loan Purpose"
        name="loan_purpose"
        value={form.loan_purpose}
        onChange={handleChange}
        options={[
          'debt_consolidation', 'credit_card', 'home_improvement',
          'medical', 'car', 'vacation', 'small_business', 'other',
        ].map((p) => ({ value: p, label: p.replace(/_/g, ' ') }))}
      />

      {/* ── Financial Position ────────────────────────────────────────────── */}
      <SectionTitle>Financial Position</SectionTitle>
      <div className="grid grid-cols-2 gap-3">
        <NumberInput
          label="Existing Debt"
          name="existing_debt"
          value={form.existing_debt}
          onChange={handleChange}
          min={0} max={500000} step={1000}
          prefix="$"
        />
        <NumberInput
          label="Savings"
          name="savings"
          value={form.savings}
          onChange={handleChange}
          min={0} max={500000} step={500}
          prefix="$"
        />
      </div>
      <NumberInput
        label="Monthly Expenses"
        name="monthly_expenses"
        value={form.monthly_expenses}
        onChange={handleChange}
        min={0} max={20000} step={100}
        prefix="$"
      />

      {/* ── Submit ────────────────────────────────────────────────────────── */}
      <div className="pt-4">
        <button
          type="submit"
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500
            disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold
            py-2.5 px-4 rounded-xl transition-colors text-sm"
        >
          {loading ? (
            <><Loader2 className="w-4 h-4 animate-spin" /> Assessing…</>
          ) : (
            <><Send className="w-4 h-4" /> Assess Risk</>
          )}
        </button>
      </div>
    </form>
  )
}
