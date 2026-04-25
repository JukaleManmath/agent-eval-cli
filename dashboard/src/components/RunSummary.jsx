function scoreColor(score) {
  if (score === null || score === undefined) return 'text-slate-400'
  if (score >= 0.7) return 'text-emerald-700'
  if (score >= 0.5) return 'text-amber-800'
  return 'text-rose-800'
}

function Stat({ label, value, cls = '' }) {
  return (
    <div className="flex flex-col py-3.5 pr-8">
      <span className="text-xs text-slate-400 uppercase tracking-widest font-medium mb-1">{label}</span>
      <span className={`text-sm font-semibold tabular-nums text-slate-800 ${cls}`}>{value}</span>
    </div>
  )
}

export default function RunSummary({ summary, report }) {
  const passRate = summary.total_scenarios > 0
    ? Math.round((summary.passed / summary.total_scenarios) * 100)
    : 0

  const timestamp = report.timestamp.replace(
    /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/,
    '$1-$2-$3  $4:$5:$6'
  )

  return (
    <div className="flex items-center gap-0">

      <div className="flex flex-col py-3.5 pr-8 mr-6 border-r border-slate-200">
        <span className="text-xs text-slate-400 uppercase tracking-widest font-medium mb-1">Run</span>
        <span className="text-xs text-slate-500 tabular-nums font-mono">{timestamp}</span>
        {report.simulator_mode && (
          <span className="text-xs text-slate-400 mt-0.5">{report.simulator_mode}{report.simulator_model ? ` · ${report.simulator_model}` : ''}</span>
        )}
      </div>

      <div className="flex items-stretch divide-x divide-slate-200">
        <Stat label="Scenarios" value={summary.total_scenarios} />
        <div className="pl-8"><Stat label="Passed" value={summary.passed} cls="text-emerald-700" /></div>
        <div className="pl-8"><Stat label="Failed" value={summary.failed} cls={summary.failed > 0 ? 'text-rose-800' : 'text-slate-400'} /></div>
        {summary.errored > 0 && (
          <div className="pl-8"><Stat label="Errored" value={summary.errored} cls="text-amber-800" /></div>
        )}
        <div className="pl-8"><Stat label="Avg Score" value={summary.aggregate_score?.toFixed(3)} cls={scoreColor(summary.aggregate_score)} /></div>
        <div className="pl-8"><Stat label="Pass Rate" value={`${passRate}%`} cls={passRate >= 70 ? 'text-emerald-700' : passRate >= 50 ? 'text-amber-800' : 'text-rose-800'} /></div>
      </div>

      <div className="ml-auto pl-8">
        <span className={`text-xs px-3 py-1.5 rounded-full font-semibold ring-1 ${
          summary.overall_pass
            ? 'bg-slate-900 text-white ring-slate-900'
            : 'bg-slate-100 text-slate-600 ring-slate-200'
        }`}>
          {summary.overall_pass ? 'Overall Pass' : 'Overall Fail'}
        </span>
      </div>

    </div>
  )
}
