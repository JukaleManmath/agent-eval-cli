function scoreColor(score) {
  if (score === null || score === undefined) return 'text-slate-400'
  if (score >= 0.7) return 'text-emerald-700'
  if (score >= 0.5) return 'text-amber-800'
  return 'text-rose-800'
}

function StatusChip({ scenario }) {
  if (scenario.errored)
    return <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 ring-1 ring-slate-200">Error</span>
  if (scenario.passed)
    return <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 ring-1 ring-slate-200">Pass</span>
  return <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 ring-1 ring-slate-200">Fail</span>
}

export default function ScenarioTable({ scenarios, selectedId, onSelect }) {
  const passed = scenarios.filter((s) => !s.errored && s.passed).length
  const failed = scenarios.filter((s) => !s.errored && !s.passed).length
  const errored = scenarios.filter((s) => s.errored).length

  return (
    <div className="flex flex-col h-full">

      <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between sticky top-0 bg-white z-10">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Scenarios</span>
        <div className="flex items-center gap-1.5">
          {passed > 0 && <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 ring-1 ring-slate-200">{passed} pass</span>}
          {failed > 0 && <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 ring-1 ring-slate-200">{failed} fail</span>}
          {errored > 0 && <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 ring-1 ring-slate-200">{errored} err</span>}
        </div>
      </div>

      <div className="flex-1">
        {scenarios.map((scenario) => {
          const isSelected = selectedId === scenario.scenario_id
          return (
            <div
              key={scenario.scenario_id}
              onClick={() => onSelect(scenario.scenario_id)}
              className={`border-b border-slate-100 cursor-pointer transition-colors px-5 py-4 border-l-2 ${
                isSelected
                  ? 'bg-indigo-50 border-l-indigo-500'
                  : 'border-l-transparent hover:bg-slate-50'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <span className="text-sm font-medium text-slate-800 leading-snug">
                  {scenario.scenario_name}
                </span>
                <span className={`text-sm font-bold tabular-nums shrink-0 ${scoreColor(scenario.aggregate_score)}`}>
                  {scenario.errored ? '—' : scenario.aggregate_score?.toFixed(3)}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <StatusChip scenario={scenario} />
                <span className="text-xs text-slate-400 tabular-nums">{scenario.turns_used} turns</span>
                <span className="text-slate-300">·</span>
                <span className="text-xs text-slate-400">{scenario.termination_reason?.replace(/_/g, ' ')}</span>
              </div>
            </div>
          )
        })}
      </div>

    </div>
  )
}
