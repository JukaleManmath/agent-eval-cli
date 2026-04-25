import { useEffect, useState } from 'react'
import RunSummary from './components/RunSummary.jsx'
import ScenarioDetail from './components/ScenarioDetail.jsx'
import ScenarioTable from './components/ScenarioTable.jsx'

export default function App() {
  const [reports, setReports] = useState([])
  const [selectedReport, setSelectedReport] = useState(null)
  const [selectedScenarioId, setSelectedScenarioId] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/reports/manifest.json')
      .then((res) => res.json())
      .then((manifest) => {
        const filenames = manifest.reports || []
        return Promise.all(filenames.map((f) => fetch(`/reports/${f}`).then((r) => r.json())))
      })
      .then((data) => {
        setReports(data)
        if (data.length > 0) {
          setSelectedReport(data[0])
          if (data[0].scenarios?.length > 0) setSelectedScenarioId(data[0].scenarios[0].scenario_id)
        }
        setLoading(false)
      })
      .catch((err) => { setError(err.message); setLoading(false) })
  }, [])

  const selectedScenario = selectedReport?.scenarios?.find(s => s.scenario_id === selectedScenarioId)

  if (loading) return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <div className="w-6 h-6 rounded-full border-2 border-slate-200 border-t-indigo-500 animate-spin" />
        <span className="text-sm text-slate-500">Loading reports…</span>
      </div>
    </div>
  )

  if (error) return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center">
      <div className="text-center">
        <p className="font-medium text-red-600">Failed to load reports</p>
        <p className="text-sm text-slate-500 mt-1">{error}</p>
      </div>
    </div>
  )

  if (reports.length === 0) return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center">
      <div className="text-center">
        <p className="text-slate-600">No reports found</p>
        <p className="text-sm text-slate-400 mt-2">
          Run <code className="text-indigo-600 bg-white px-1.5 py-0.5 rounded font-mono text-xs border border-slate-200">agenteval run test_cases/</code> to generate a report.
        </p>
      </div>
    </div>
  )

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-slate-50">

      {/* Navbar */}
      <nav className="h-14 bg-white border-b border-slate-200 flex items-center px-8 justify-between shrink-0">
        <div className="flex items-center gap-8">
          <span className="logo-text">AgentEval</span>
          {reports.length > 1 && (
            <select
              value={reports.indexOf(selectedReport)}
              onChange={(e) => {
                const r = reports[Number(e.target.value)]
                setSelectedReport(r)
                setSelectedScenarioId(r.scenarios?.[0]?.scenario_id ?? null)
              }}
              className="text-sm text-slate-600 bg-slate-100 border-0 rounded-lg px-3 py-1.5 cursor-pointer focus:outline-none focus:ring-2 focus:ring-indigo-300"
            >
              {reports.map((r, i) => {
                const label = r.timestamp.replace(
                  /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/,
                  '$1-$2-$3  $4:$5:$6'
                )
                return <option key={i} value={i}>{label}</option>
              })}
            </select>
          )}
        </div>
        <span className="text-xs text-slate-400">v{selectedReport.agenteval_version}</span>
      </nav>

      {/* Summary strip */}
      <div className="bg-white border-b border-slate-200 px-8 shrink-0">
        <RunSummary summary={selectedReport.summary} report={selectedReport} />
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        <aside className="w-72 shrink-0 bg-white border-r border-slate-200 overflow-y-auto">
          <ScenarioTable
            scenarios={selectedReport.scenarios}
            selectedId={selectedScenarioId}
            onSelect={setSelectedScenarioId}
          />
        </aside>
        <main className="flex-1 overflow-y-auto bg-slate-50 px-8 py-7">
          <ScenarioDetail scenario={selectedScenario} />
        </main>
      </div>

    </div>
  )
}
