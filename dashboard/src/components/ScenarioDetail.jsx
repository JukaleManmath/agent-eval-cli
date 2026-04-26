import RadarChart from './RadarChart.jsx'

const SCORER_SHORT = {
  task_completion: 'Task Completion',
  instruction_following: 'Instructions',
  coherence: 'Coherence',
  turn_efficiency: 'Efficiency',
  hallucination_risk: 'Hallucination',
}

function scoreColor(score) {
  if (score === null || score === undefined) return 'text-slate-400'
  if (score >= 0.7) return 'text-emerald-700'
  if (score >= 0.5) return 'text-amber-800'
  return 'text-rose-800'
}

function terminationColor(reason) {
  if (reason === 'goal_achieved') return 'text-emerald-700'
  if (reason === 'max_turns_reached') return 'text-amber-800'
  if (reason === 'agent_error') return 'text-rose-800'
  return 'text-slate-400'
}

function ScorerCard({ name, detail }) {
  const score = detail?.score ?? null
  const passed = score !== null && score >= 0.7

  return (
    <div className="bg-white rounded-xl p-5 shadow-sm ring-1 ring-slate-200/60">
      <div className="text-xs text-slate-400 font-medium uppercase tracking-widest">{name}</div>
      <div className={`text-3xl font-bold tabular-nums leading-none mt-3 ${scoreColor(score)}`}>
        {score !== null ? score.toFixed(3) : '—'}
      </div>
      {score !== null && (
        <div className="mt-3">
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 ring-1 ring-slate-200">
            {passed ? 'Pass' : 'Fail'}
          </span>
        </div>
      )}
    </div>
  )
}

export default function ScenarioDetail({ scenario }) {
  if (!scenario) return (
    <div className="h-48 flex items-center justify-center">
      <p className="text-slate-400">Select a scenario</p>
    </div>
  )

  const hasEvidence = Object.entries(scenario.scores || {}).some(([, d]) => d.evidence?.length > 0)

  return (
    <div className="flex flex-col gap-7">

      {/* Header card */}
      <div className="bg-white rounded-xl p-7 shadow-sm ring-1 ring-slate-200/60">
        <div className="flex items-start justify-between gap-6">
          <div className="min-w-0 flex-1">
            <h2 className="text-lg font-semibold text-slate-900 leading-snug">
              {scenario.scenario_name}
            </h2>
            <div className="mt-3 flex items-center gap-3 flex-wrap">
              <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-slate-100 text-slate-600 uppercase tracking-wide">
                {scenario.outcome_type}
              </span>
              <span className={`text-sm ${terminationColor(scenario.termination_reason)}`}>
                {scenario.termination_reason?.replace(/_/g, ' ')}
              </span>
              <span className="text-slate-300">·</span>
              <span className="text-sm text-slate-500 tabular-nums">{scenario.turns_used} turns</span>
            </div>
          </div>
          {!scenario.errored && (
            <div className="shrink-0 text-right">
              <div className={`text-5xl font-bold tabular-nums leading-none ${scoreColor(scenario.aggregate_score)}`}>
                {scenario.aggregate_score?.toFixed(3)}
              </div>
              <div className="text-xs text-slate-400 mt-2 uppercase tracking-widest">aggregate</div>
            </div>
          )}
        </div>

        {scenario.errored && (
          <div className="mt-5 rounded-lg bg-red-50 ring-1 ring-red-100 px-5 py-4">
            <p className="text-xs text-rose-800 uppercase tracking-widest font-semibold">{scenario.error_type}</p>
            <p className="text-sm text-red-500 mt-1.5 leading-relaxed">{scenario.error}</p>
          </div>
        )}
      </div>

      {/* Radar chart — full width */}
      {!scenario.errored && Object.keys(scenario.scores || {}).length > 0 && (
        <RadarChart scores={scenario.scores} />
      )}

      {/* Scorer cards */}
      {!scenario.errored && Object.keys(scenario.scores || {}).length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">Scorers</p>
          <div className="grid grid-cols-5 gap-3">
            {Object.entries(scenario.scores).map(([key, detail]) => (
              <ScorerCard key={key} name={SCORER_SHORT[key] || key} detail={detail} />
            ))}
          </div>
        </div>
      )}

      {/* Evidence */}
      {!scenario.errored && hasEvidence && (
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">Evidence</p>
          <div className="flex flex-col gap-4">
            {Object.entries(scenario.scores).map(([key, detail]) =>
              detail.evidence?.length > 0 ? (
                <div key={key} className="bg-white rounded-xl p-5 shadow-sm ring-1 ring-slate-200/60">
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">{SCORER_SHORT[key] || key}</p>
                  <div className="flex flex-col gap-2">
                    {detail.evidence.map((e, i) => (
                      <div key={i} className="text-sm text-slate-600 font-mono bg-slate-50 rounded-lg px-4 py-3 leading-relaxed">
                        {e}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null
            )}
          </div>
        </div>
      )}

      {/* Transcript */}
      {scenario.conversation?.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Transcript</p>
            <span className="text-xs text-slate-400">{scenario.conversation.length} turns</span>
          </div>
          <div className="flex flex-col gap-3 max-w-2xl">
            {scenario.conversation.map((turn) => {
              const isUser = turn.role === 'user'
              return (
                <div key={turn.number} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[72%] px-4 py-3 text-sm leading-relaxed rounded-2xl ${
                    isUser
                      ? 'bg-indigo-600 text-white rounded-tr-sm'
                      : 'bg-white text-slate-700 shadow-sm ring-1 ring-slate-200/60 rounded-tl-sm'
                  }`}>
                    {turn.content}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

    </div>
  )
}
