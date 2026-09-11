/**
 * RiskGauge.jsx — SVG arc gauge showing default probability.
 *
 * Colour zones:
 *   0–30%   → Green  (Low risk)
 *   30–60%  → Amber  (Medium risk)
 *   60–100% → Red    (High risk)
 */

const RADIUS = 80
const STROKE = 14
const CENTER = 100

// Convert a probability [0,1] to arc sweep angle [0, π]
function probToAngle(prob) {
  return prob * Math.PI  // 0 → left end (π), 1 → right end (0)
}

// Compute SVG arc path point from angle (measuring from left, clockwise)
function arcPoint(angle, r) {
  // Start from left of semi-circle, go clockwise
  const x = CENTER - r * Math.cos(angle)
  const y = CENTER - r * Math.sin(angle) + STROKE / 2
  return { x, y }
}

function buildArc(from, to, r) {
  const start = arcPoint(Math.PI - from, r)
  const end = arcPoint(Math.PI - to, r)
  const largeArc = (to - from) > Math.PI ? 1 : 0
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`
}

const ZONE_COLORS = {
  Low: '#16a34a',
  Medium: '#d97706',
  High: '#dc2626',
}

export default function RiskGauge({ probability, category }) {
  const pct = Math.min(1, Math.max(0, probability))

  // Needle angle (maps 0→leftmost, 1→rightmost of semi-circle)
  const needleAngle = Math.PI * (1 - pct)  // in full circle terms
  const nx = CENTER + (RADIUS - 4) * Math.cos(needleAngle)
  const ny = CENTER - (RADIUS - 4) * Math.sin(needleAngle) + STROKE / 2

  const color = ZONE_COLORS[category] || '#94a3b8'

  // Zone arcs (Low: 0–0.30, Medium: 0.30–0.60, High: 0.60–1.0)
  const zones = [
    { from: 0, to: 0.30, color: '#16a34a' },
    { from: 0.30, to: 0.60, color: '#d97706' },
    { from: 0.60, to: 1.00, color: '#dc2626' },
  ]

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 20 200 120" className="w-full max-w-xs">
        {/* Background arc (grey) */}
        <path
          d={buildArc(0, 1, RADIUS)}
          fill="none"
          stroke="#1e293b"
          strokeWidth={STROKE}
          strokeLinecap="round"
        />

        {/* Zone arcs */}
        {zones.map((z, i) => (
          <path
            key={i}
            d={buildArc(z.from, z.to, RADIUS)}
            fill="none"
            stroke={z.color}
            strokeWidth={STROKE}
            opacity={0.25}
            strokeLinecap="round"
          />
        ))}

        {/* Filled progress arc */}
        {pct > 0.005 && (
          <path
            d={buildArc(0, pct, RADIUS)}
            fill="none"
            stroke={color}
            strokeWidth={STROKE}
            strokeLinecap="round"
          />
        )}

        {/* Needle */}
        <line
          x1={CENTER}
          y1={CENTER + STROKE / 2}
          x2={nx}
          y2={ny}
          stroke="white"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        <circle cx={CENTER} cy={CENTER + STROKE / 2} r={5} fill="white" />

        {/* Zone labels */}
        <text x="22" y="118" fill="#16a34a" fontSize="9" textAnchor="middle" opacity="0.8">Low</text>
        <text x="100" y="28" fill="#d97706" fontSize="9" textAnchor="middle" opacity="0.8">Medium</text>
        <text x="178" y="118" fill="#dc2626" fontSize="9" textAnchor="middle" opacity="0.8">High</text>

        {/* Centre probability text */}
        <text x="100" y="108" fill={color} fontSize="22" fontWeight="bold" textAnchor="middle">
          {(pct * 100).toFixed(1)}%
        </text>
        <text x="100" y="120" fill="#64748b" fontSize="8" textAnchor="middle">
          Default Probability
        </text>
      </svg>

      {/* Category badge */}
      <span className={`mt-2 px-4 py-1 rounded-full text-xs font-semibold
        badge-${category?.toLowerCase()}`}>
        {category} Risk
      </span>
    </div>
  )
}
