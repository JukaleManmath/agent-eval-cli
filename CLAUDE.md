# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What this project is

**AgentEval** (`agent-eval-cli` on PyPI, `agenteval` CLI command, `agenteval` Python import) is an open-source CLI tool for multi-turn AI agent simulation and evaluation. It simulates conversations against any HTTP POST agent endpoint and scores the results locally — no cloud dependency required.

Two simulation modes:
- `--mode scripted` — deterministic, reads fixed turns from YAML. Zero API calls. Use for CI/CD.
- `--mode groq` — Llama 3.3 70B via Groq free API plays the user side. Requires `GROQ_API_KEY`.

---

## Commands

```bash
# Install for development (all extras)
pip install -e ".[all,dev]"

# Lint
ruff check .

# Type check
mypy agenteval/

# Run all tests
pytest

# Run a single test file
pytest tests/unit/test_turn_efficiency.py

# Run a single test by name
pytest tests/unit/test_turn_efficiency.py::test_refusal_fast_no_penalty

# Run with coverage
pytest --cov=agenteval --cov-report=term-missing

# Build dashboard (required before agenteval dashboard works locally)
make build-dashboard

# Build package for PyPI
cd dashboard && npm run build && cp -r dist/* ../agenteval/dashboard_dist/
cd .. && python -m build && twine check dist/*
```

Install extras:
```bash
pip install -e ".[groq]"   # Groq simulation mode
pip install -e ".[ml]"     # ML scorers (sentence-transformers + spaCy)
pip install -e ".[dev]"    # pytest, ruff, mypy, respx, pre-commit
```

---

## Architecture

### Package layout

```
agenteval/
├── cli.py                   # Click entrypoint: run, validate, dashboard commands
├── config.py                # Settings / env var loading
├── schema/
│   ├── test_case.py         # Pydantic: TestCase, UserPersona, Evaluation, ConversationConfig
│   └── report.py            # Pydantic: JSON report schema
├── simulation/
│   ├── engine.py            # Async main loop — drives one scenario end-to-end; substitutes ${SESSION_ID} / ${USER_MESSAGE} into request_template
│   ├── session.py           # Turn, Session dataclasses
│   ├── agent_client.py      # httpx.AsyncClient POST; extracts agent reply via jmespath (response_path field)
│   ├── simulator_factory.py # create_simulator(mode, test_case) → BaseSimulator; calls _resolve_goal() from groq_simulator
│   ├── scripted_simulator.py
│   └── groq_simulator.py    # lazy `import groq`; _truncate_history pops user+agent pairs; _resolve_goal() defined here
├── scorers/
│   ├── _embeddings.py       # Shared embed() + cosine_similarity() — lazy-loaded SentenceTransformer
│   ├── base.py
│   ├── task_completion.py   # core + [ml]
│   ├── instruction_following.py  # core only (string matching)
│   ├── coherence.py         # [ml] only — returns None without it
│   ├── turn_efficiency.py   # core only (pure Python)
│   ├── hallucination.py     # [ml] only — returns None without it
│   └── aggregate.py         # re-normalises weights when scorers return None
├── runner.py                # asyncio.gather with semaphore; file discovery (sorted rglob); ScenarioError defined here; summary plain-mean
├── reporter.py              # writes agenteval_report_{timestamp}.json
└── dashboard_dist/          # pre-built React bundle (not git-tracked; built by make build-dashboard)
```

### Data flow

```
YAML file(s)
  → Pydantic TestCase (schema/test_case.py)
    → engine.py: UUID4 session_id, create_simulator(), loop until [GOAL_ACHIEVED] or max_turns
      → agent_client.py: async POST, jmespath extraction
      → simulator (scripted or groq): returns next user message or "[GOAL_ACHIEVED]"
    → Session (turns, termination_reason)
      → scorers run locally on completed Session
        → aggregate.py: re-normalise None scorers, weighted mean
    → ScenarioResult / ScenarioError
  → reporter.py: JSON report file
```

### Key contracts

**`termination_reason`** — three possible values stored on `Session` and in the JSON report:
- `"goal_achieved"` — simulator emitted `[GOAL_ACHIEVED]`
- `"max_turns_reached"` — hit `conversation.max_turns` without goal signal
- `"agent_error"` — agent returned an error on every turn

**Request template substitution** — `engine.py` replaces `${SESSION_ID}` and `${USER_MESSAGE}` in the YAML `request_template` string before each HTTP call. `${AGENT_API_KEY}` (and any other `${VAR}` patterns) are resolved from environment variables via `python-dotenv`.

**Response extraction** — `agent_client.py` uses `jmespath.search(response_path, json_body)` to pull the agent's reply text out of the response. `response_path` uses dot notation (e.g. `"response.text"`).

**`_resolve_goal()`** — defined in `groq_simulator.py`, imported by `simulator_factory.py`. Maps `outcome_type` → the correct intent field (`success_intent`, `refusal_intent`, or `escalation_intent`) from `evaluation` to populate the Groq prompt's `{goal}` variable.

**`--fail-on-threshold`** — `cli.py` sets `sys.exit(1)` when `summary.overall_pass` is False, enabling CI gate behaviour. `overall_pass` is True only when every non-errored scenario's aggregate score meets its per-scenario threshold AND the run-level threshold is met.

**Scorers** return `float | tuple[float, list] | None`.
- `float` — score only (task_completion, coherence, turn_efficiency)
- `tuple[float, list]` — score + evidence items (instruction_following, hallucination_risk)
- `None` — scorer skipped (ML not installed, or no context_facts provided)

`calculate_aggregate` in `aggregate.py` uses `_extract_score(v)` to unpack tuples before arithmetic — never pass raw scorer output directly into arithmetic.

**Simulation termination** — `[GOAL_ACHIEVED]` is a flow-control signal from the simulator only. It is stripped from the visible transcript by `_parse_termination()` in `engine.py`. Scorer 1 (task completion) evaluates quality independently — it never reads this signal.

**Session IDs** — UUID4, generated per scenario at simulation start, injected as `${SESSION_ID}` into the request template. Never shared across scenarios.

**Groq history truncation** — `_truncate_history` in `groq_simulator.py` always pops user+agent message *pairs* (indices 1+2) to avoid orphaned turns that violate the OpenAI message format.

### Dashboard

The React app is pre-built and bundled into `agenteval/dashboard_dist/`. The CLI copies report JSON files into `dashboard_dist/reports/` and writes a `manifest.json` listing them before starting the HTTP server. The React app fetches `/reports/manifest.json` on load, then fetches each listed report file. `dashboard_dist/` is in `.gitignore` except for `.gitkeep`; it is populated by `make build-dashboard`.

### Optional dependencies

| Extra | Unlocks |
|---|---|
| `[groq]` | `--mode groq` simulation |
| `[ml]` | Scorers 1 (soft check), 3, 5; spaCy `en_core_web_sm` installed as wheel |
| `[all]` | Both of the above |
| `[dev]` | pytest, ruff, mypy, respx, pre-commit |

`ML_AVAILABLE` is set via a `try/except ImportError` at the top of each ML scorer and in `_embeddings.py`. Scorers degrade to `None` (excluded from aggregate) when `[ml]` is not installed.

---

## Testing

- `pytest-asyncio` is configured with `asyncio_mode = "auto"` — async test functions work without decorators.
- `respx` is used to mock `httpx` requests in unit tests. Do not mock at the `httpx` transport level directly.
- Integration tests in `tests/integration/` use a real Docker mock agent via the `mock_agent` session-scoped fixture in `tests/conftest.py`. Run with `docker compose up` available.
- `tests/golden/` contains a pinned JSON report fixture — golden tests catch unintended changes to report schema or scorer output.

---

## v1 scope boundary

v1 supports **synchronous JSON over HTTP POST only**. Out of scope: WebSocket/SSE, OAuth/cookie auth, multi-message response payloads, non-JSON payloads.
