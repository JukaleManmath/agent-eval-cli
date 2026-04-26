const SCORERS = [
  { key: 'task_completion',       label: 'Task Completion' },
  { key: 'instruction_following', label: 'Instruction Following' },
  { key: 'coherence',             label: 'Coherence' },
  { key: 'turn_efficiency',       label: 'Turn Efficiency' },
  { key: 'hallucination_risk',    label: 'Hallucination Risk' },
]

const N = SCORERS.length
const CX = 200
const CY = 200
const R = 140
const LEVELS = 5

function axisAngle(i) {
  return (-Math.PI / 2) + (2 * Math.PI * i) / N
}

function polarToXY(angle, radius) {
  return {
    x: CX + radius * Math.cos(angle),
    y: CY + radius * Math.sin(angle),
  }
}

function polygonPoints(values) {
  return values
    .map((v, i) => {
      const pt = polarToXY(axisAngle(i), v * R)
      return `${pt.x},${pt.y}`
    })
    .join(' ')
}

function scoreColor(v) {
  if (v === null) return '#94a3b8'
  if (v >= 0.7) return '#059669'
  if (v >= 0.5) return '#d97706'
  return '#e11d48'
}

export default function RadarChart({ scores }) {
  if (!scores) return null

  const values = SCORERS.map(({ key }) => {
    const d = scores[key]
    if (d === null || d === undefined) return null
    const v = typeof d === 'object' ? d.score : d
    return v === null || v === undefined ? null : Math.max(0, Math.min(1, v))
  })

  const hasAny = values.some(v => v !== null)
  if (!hasAny) return null

  const filled = values.map(v => v ?? 0)
  const hasNulls = values.some(v => v === null)

  return (
    <div className="bg-white rounded-2xl p-7 shadow-sm ring-1 ring-slate-200/60">
      <div className="flex items-center justify-between mb-6">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Score Profile</p>
        {hasNulls && (
          <p className="text-[10px] text-slate-400">
            Dashed = scorer unavailable — install <code className="font-mono bg-slate-100 px-1 rounded">agenteval[ml]</code>
          </p>
        )}
      </div>

      <div className="flex justify-center px-10 py-4">
        <svg viewBox="0 0 400 400" width="100%" style={{ maxWidth: 480, overflow: 'visible' }}>
          <defs>
            <radialGradient id="radarFill" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#818cf8" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0.08" />
            </radialGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Grid rings */}
          {Array.from({ length: LEVELS }, (_, lvl) => {
            const r = (R * (lvl + 1)) / LEVELS
            const pts = Array.from({ length: N }, (_, i) => {
              const pt = polarToXY(axisAngle(i), r)
              return `${pt.x},${pt.y}`
            }).join(' ')
            const isOuter = lvl === LEVELS - 1
            return (
              <polygon
                key={lvl}
                points={pts}
                fill={isOuter ? 'rgba(99,102,241,0.03)' : 'none'}
                stroke={isOuter ? '#c7d2fe' : '#e2e8f0'}
                strokeWidth={isOuter ? '1.5' : '1'}
                strokeDasharray={isOuter ? 'none' : '4 3'}
              />
            )
          })}

          {/* Axis lines */}
          {SCORERS.map((_, i) => {
            const outer = polarToXY(axisAngle(i), R)
            const isNull = values[i] === null
            return (
              <line
                key={i}
                x1={CX} y1={CY}
                x2={outer.x} y2={outer.y}
                stroke={isNull ? '#e2e8f0' : '#c7d2fe'}
                strokeWidth="1"
                strokeDasharray={isNull ? '3 3' : 'none'}
              />
            )
          })}

          {/* Filled polygon with glow */}
          <polygon
            points={polygonPoints(filled)}
            fill="url(#radarFill)"
            stroke="#6366f1"
            strokeWidth="2.5"
            strokeLinejoin="round"
            filter="url(#glow)"
          />

          {/* Level value labels (0.2, 0.4 …) */}
          {Array.from({ length: LEVELS }, (_, lvl) => {
            const val = ((lvl + 1) / LEVELS).toFixed(1)
            const r = (R * (lvl + 1)) / LEVELS
            return (
              <text
                key={lvl}
                x={CX + 4}
                y={CY - r + 4}
                fontSize="9"
                fill="#cbd5e1"
                fontFamily="Inter, sans-serif"
              >
                {val}
              </text>
            )
          })}

          {/* Axis labels + score badges */}
          {SCORERS.map(({ label }, i) => {
            const angle = axisAngle(i)
            const LABEL_R = R + 32
            const pt = polarToXY(angle, LABEL_R)
            const isNull = values[i] === null
            const score = values[i]

            // nudge anchors so labels don't clip
            let anchor = 'middle'
            const cosA = Math.cos(angle)
            if (cosA > 0.3) anchor = 'start'
            else if (cosA < -0.3) anchor = 'end'

            return (
              <g key={i}>
                <text
                  x={pt.x}
                  y={pt.y - (score !== null ? 8 : 0)}
                  textAnchor={anchor}
                  dominantBaseline="central"
                  fontSize="11"
                  fontFamily="Inter, sans-serif"
                  fontWeight="600"
                  fill={isNull ? '#94a3b8' : '#334155'}
                >
                  {label}
                </text>
                {score !== null && (
                  <text
                    x={pt.x}
                    y={pt.y + 10}
                    textAnchor={anchor}
                    dominantBaseline="central"
                    fontSize="11"
                    fontFamily="Inter, sans-serif"
                    fontWeight="700"
                    fill={scoreColor(score)}
                  >
                    {score.toFixed(2)}
                  </text>
                )}
              </g>
            )
          })}

          {/* Data point dots */}
          {filled.map((v, i) => {
            const pt = polarToXY(axisAngle(i), v * R)
            const isNull = values[i] === null
            if (isNull) return null
            return (
              <circle
                key={i}
                cx={pt.x}
                cy={pt.y}
                r={5}
                fill="#6366f1"
                stroke="white"
                strokeWidth="2.5"
                filter="url(#glow)"
              />
            )
          })}

          {/* Center dot */}
          <circle cx={CX} cy={CY} r={3} fill="#c7d2fe" />
        </svg>
      </div>
    </div>
  )
}
