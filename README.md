# AgentEval

![CI](https://github.com/JukaleManmath/agent-eval-cli/actions/workflows/ci.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/agent-eval-cli.svg)](https://pypi.org/project/agent-eval-cli/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Open-source multi-turn AI agent simulation and evaluation.**

Define a test scenario in YAML, run it against your agent's HTTP endpoint, and get a structured pass/fail report, **from your terminal, for free, with no mandatory cloud dependency**.

```bash
pip install agent-eval-cli
agenteval run test_cases/ --mode scripted
```



https://github.com/user-attachments/assets/1fc3cd83-04ca-4cc7-8a75-774e8dfc91d5



---

## The gap this fills

Every existing tool evaluates **what your agent already said**. You bring the conversation; they score it.

Nobody simulates the conversation for you — free — against any endpoint.

| Tool | Multi-turn simulation | Free | Open source |
|---|---|---|---|
| LangSmith | ❌ | ❌ | ❌ |
| Braintrust | ❌ | ❌ | ❌ |
| Promptfoo | ❌ | ✅ | ✅ |
| DeepEval | ❌ | ✅ | ✅ |
| Coval | ✅ | ❌ | ❌ |
| **AgentEval** | **✅** | **✅** | **✅** |

---

## Install

```bash
# Core — scripted mode only, no ML scorers
pip install agent-eval-cli

# + Groq simulation mode (free API key, no credit card)
pip install agent-eval-cli[groq]

# + ML scorers (sentence-transformers, ~500MB)
pip install agent-eval-cli[ml]

# Everything
pip install agent-eval-cli[all]
```

**Requires Python 3.9+**

---

## Quick start

**1. Write a test case:**

```yaml
# test_cases/booking_happy_path.yaml
meta:
  scenario_id: "booking-happy-path-001"
  name: "Patient successfully books a doctor's appointment"
  tags: ["healthcare", "booking"]

agent:
  endpoint: "http://localhost:8080/chat"
  method: "POST"
  headers:
    Content-Type: "application/json"
  request_template: |
    {
      "session_id": "${SESSION_ID}",
      "message": "${USER_MESSAGE}"
    }
  response_path: "response"
  timeout_seconds: 30

user_persona:
  name: "Sarah Chen"
  tone: "polite but slightly anxious"
  background: "First-time patient, not tech-savvy"
  opening_message: "Hi, I need to see a doctor about my knee."

outcome_type: success   # success | refusal | escalation

conversation:
  max_turns: 12
  min_turns: 3
  expected_turns: 8

evaluation:
  success_intent: "User receives a confirmed appointment with specific date, time, and doctor name"
  success_keywords:
    - "appointment confirmed"
    - "booked for"
    - "confirmation number"
  forbidden_phrases:
    - "I don't know"
    - "I cannot help"
  context_facts:
    - "Appointments available Monday through Friday"
    - "Dr. Smith specialises in orthopedics"
    - "Clinic hours are 8am to 6pm"
  thresholds:
    task_completion:       0.65
    instruction_following: 0.80
    turn_efficiency:       0.60
    aggregate:             0.70

scripted_turns:
  - "Hi, I need to see a doctor about my knee. It's been hurting for two weeks."
  - "Is Dr. Smith available this week?"
  - "Thursday at 2pm works for me."
  - "My name is Sarah Chen, date of birth March 12 1990."
  - "Yes, that email is correct."
  - "Great, thank you so much."
```

**2. Run:**

```bash
agenteval run test_cases/ --mode scripted
```

**3. View results:**

```bash
agenteval dashboard ./reports/
```

---

## Simulation modes

### `--mode scripted` — deterministic CI mode

Sends the fixed `scripted_turns` from your YAML in order. Zero API calls. Zero variance. After the last scripted turn, automatically signals goal completion if the agent responded without error.

Use this for CI/CD gates — results are reproducible across every run.

```bash
agenteval run test_cases/ --mode scripted --concurrency 4 --fail-on-threshold 0.70
```

![Scripted simulation](https://raw.githubusercontent.com/JukaleManmath/agent-eval-cli/main/demo_images/scripted_simulation.png)

### `--mode groq` — realistic simulation

Llama 3.3 70B (via Groq's free API) plays the user side, guided by the persona and outcome type you defined. Has natural variance — useful for exploratory evaluation and finding edge cases scripted mode misses.

Requires a free [Groq API key](https://console.groq.com) — no credit card.

```bash
export GROQ_API_KEY=your_key_here
agenteval run test_cases/ --mode groq
```

![Groq simulation](https://raw.githubusercontent.com/JukaleManmath/agent-eval-cli/main/demo_images/Groq_simulation.png)

> **Privacy notice:** when using `--mode groq`, your `context_facts` are sent to Groq's API. Use `--mode scripted` for sensitive or proprietary data.

---

## Scorers

All scoring runs locally after the conversation completes. No API calls.

> These are **heuristic signals** designed to catch obvious failures — not authoritative ground truth. Calibrate thresholds against real transcripts before using them as hard CI gates.

| Scorer | What it measures | Requires `[ml]` |
|---|---|---|
| **Task Completion** | Semantic similarity between agent responses and the outcome intent | Yes |
| **Instruction Following** | Did the agent obey forbidden/required phrase rules? | No |
| **Response Coherence** | Are agent responses contextually relevant to what was asked? | Yes |
| **Turn Efficiency** | Did the agent resolve the goal in a reasonable number of turns? | No |
| **Hallucination Risk** | Did the agent state facts not grounded in `context_facts`? | Yes |

Scorers 1, 3, and 5 return `None` when `[ml]` is not installed — they are excluded from the aggregate and weights are re-normalised automatically.

**Default aggregate weights:**

| Scorer | Weight |
|---|---|
| Task Completion | 30% |
| Instruction Following | 25% |
| Response Coherence | 20% |
| Turn Efficiency | 15% |
| Hallucination Risk | 10% |

---

## CLI reference

```bash
# Run test cases (scripted mode — CI-safe, no API key)
agenteval run test_cases/ --mode scripted

# Run with Groq simulation
agenteval run test_cases/ --mode groq

# Filter scenarios by tag
agenteval run test_cases/ --mode scripted --tag customer-support

# Parallel execution (default: 4)
agenteval run test_cases/ --mode scripted --concurrency 8

# Fail with non-zero exit code when aggregate score is below threshold
agenteval run test_cases/ --mode scripted --fail-on-threshold 0.70

# Save reports to a directory (auto-timestamped filenames)
agenteval run test_cases/ --mode scripted --output-dir ./reports/

# Print full conversation transcript on failure
agenteval run test_cases/ --mode scripted --verbose

# Validate YAML test cases without running
agenteval validate test_cases/booking.yaml
agenteval validate test_cases/

# Open dashboard for all reports in a directory
agenteval dashboard ./reports/

# Version
agenteval --version
```

---

## YAML test case schema

### `outcome_type`

Three outcome types are supported:

- **`success`** — the agent should complete the user's request
- **`refusal`** — the agent should correctly decline (e.g. no availability, policy block)
- **`escalation`** — the agent should route the user to a human

**Refusal example:**

```yaml
outcome_type: refusal
policy_reason: "No Sunday appointments available per clinic policy"

evaluation:
  refusal_intent: "Agent clearly communicates no availability without offering to book an unavailable slot"
  refusal_keywords:
    - "unfortunately"
    - "no availability"
    - "fully booked"
  forbidden_phrases:
    - "appointment confirmed"
    - "I can book that"

scripted_turns:
  - "I need to book an appointment for next Sunday."
  - "What about Monday at 9pm?"
  - "Is there anything at all this week?"
```

### Environment variable substitution

The following variables are substituted into `request_template` at runtime:

| Variable | Value |
|---|---|
| `${SESSION_ID}` | UUID4 generated per scenario — prevents state leakage on stateful agents |
| `${USER_MESSAGE}` | The current user turn content |
| `${AGENT_API_KEY}` | Resolved from environment (set in `.env` or CI secrets) |

Any `${VAR}` pattern in the template is resolved from the environment.

### `response_path`

Uses [jmespath](https://jmespath.org) dot notation to extract the agent's reply from the JSON response body.

```yaml
response_path: "response.text"       # {"response": {"text": "..."}}
response_path: "choices[0].message.content"  # OpenAI-style
response_path: "reply"               # {"reply": "..."}
```

---

![CI gate — fail on threshold](https://raw.githubusercontent.com/JukaleManmath/agent-eval-cli/main/demo_images/failed_on_threshold.png)

## CI/CD — GitHub Action

```yaml
# .github/workflows/agenteval.yml
name: AgentEval — AI Agent Tests

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  agent-eval:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}

      - name: Install AgentEval
        run: pip install "agent-eval-cli==0.1.0"

      - name: Start agent under test
        run: docker compose up -d agent   # replace 'agent' with your service name

      - name: Wait for agent readiness
        run: |
          for i in {1..12}; do
            if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
              echo "Agent ready."; exit 0
            fi
            echo "Attempt $i/12 — retrying in 5s..."; sleep 5
          done
          echo "Agent did not become ready in 60s."; exit 1

      - name: Run AgentEval
        env:
          AGENT_API_KEY: ${{ secrets.AGENT_API_KEY }}
        run: |
          agenteval run test_cases/ \
            --mode scripted \
            --concurrency 4 \
            --output-dir ./reports/ \
            --fail-on-threshold 0.70

      - name: Upload reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: agenteval-reports-${{ github.sha }}
          path: ./reports/
          retention-days: 30
```

Reports are uploaded as a build artifact on every run. Use `actions/download-artifact` in a subsequent step to post results as a PR comment.

---

## Demo agent (try it in 2 minutes)

A ready-made customer support agent for **TableEase** (a restaurant booking service) is included so you can run AgentEval end-to-end without building your own agent first.

```bash
# Install dependencies
cd demo_agent
pip install -r requirements.txt

# Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY=your_key_here

# Start the agent
uvicorn main:app --reload
# Agent running at http://localhost:8000/chat
```

Then in a second terminal, run the included test cases:

```bash
agenteval run test_cases/ --mode scripted --output-dir ./reports/
agenteval dashboard ./reports/
```

The demo agent handles table bookings, cancellation policy questions, competitor refusals, and escalations to a human manager — covering all three `outcome_type` values.

---

## Dashboard

```bash
agenteval dashboard ./reports/
# Opens http://localhost:8080 in your browser
```

Features:
- Run summary with pass/fail counts and aggregate score
- Radar chart showing all 5 scorer dimensions per scenario
- Per-scenario score breakdown with pass/fail per scorer
- Full conversation transcript replay (chat bubble view)
- Evidence items flagged by instruction-following and hallucination scorers
- Multi-report dropdown to switch between runs

![AgentEval Dashboard](https://raw.githubusercontent.com/JukaleManmath/agent-eval-cli/main/demo_images/dashboard.png)

![Transcript replay](https://raw.githubusercontent.com/JukaleManmath/agent-eval-cli/main/demo_images/Transcripts.png)

![Evidence scores](https://raw.githubusercontent.com/JukaleManmath/agent-eval-cli/main/demo_images/Evidence_score.png)

---

## v1 scope

v1 supports **synchronous JSON over HTTP POST only**.

Out of scope for v1 (planned for v2):
- WebSocket / SSE / streaming responses
- OAuth and cookie-based authentication
- Multi-message response payloads
- Non-JSON payloads

---

## License

MIT
