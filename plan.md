# AgentEval — Implementation Plan v3
### Target: Coval | Build #4 | Phase 2 — Primary Target
### Reviewed by Codex (v1 gaps) + Claude Code (adversarial review) — all issues resolved

---

## BEFORE YOU WRITE A SINGLE LINE OF CODE

Three things to do right now, today, before starting Day 1:

**1. PyPI name confirmed — `agent-eval-cli` is available**
```bash
# Already verified. pip install agent-eval-cli is free to use.
# GitHub repo name: agent-eval-cli
# PyPI: agent-eval-cli
# CLI command: agenteval   ← this stays the same
# Python import: agenteval ← this stays the same
```

**2. Check Python version**
```bash
python --version   # must be 3.9+
```
spaCy 3.7 requires Python 3.9 minimum. If you're on 3.8, upgrade first.

**3. Install build tools**
```bash
pip install build twine ruff mypy pytest pytest-cov
```
These are needed for Day 6 and Day 9. Install now so they don't block you.

---

## 0. Deployment Model & Docker

**What AgentEval is:** A developer CLI tool. Not a SaaS. Not hosted.

```bash
pip install agent-eval-cli                              # install
agenteval run test_cases/ --mode scripted          # run against your agent
agenteval dashboard --reports-dir ./reports/       # open visual dashboard
```

Everything runs on the developer's laptop or inside their own CI environment.
No cloud hosting. No uptime responsibility. No infrastructure bill.

**Docker:** One place only — the example mock agent.

```
examples/mock_agent/
├── main.py            # FastAPI echo agent — returns structured responses
├── Dockerfile
├── docker-compose.yml
└── README.md          # instructions for testing AgentEval against this
```

Developers run `docker-compose up` to get a real HTTP endpoint before
pointing AgentEval at their actual agent. AgentEval itself never needs Docker.

---

## 1. The Real Gap (Research-Backed)

| Tool | What it does | The gap |
|---|---|---|
| **LangSmith** | Traces LangChain apps, prompt management | LangChain-only, no simulation, $39/user/month |
| **Braintrust** | Eval-driven dev, CI/CD gates, production tracing | No simulation — evaluates existing traces only. SaaS-only. |
| **Promptfoo** | CLI YAML prompt testing, red-teaming | Tests single prompts. No multi-turn conversation simulation. |
| **DeepEval** | 50+ metrics, pytest-native, agent metrics | Evaluates outputs — never simulates the conversation itself |
| **Langfuse** | Open-source observability, self-hostable | Observability only. No evaluation. No simulation. |
| **Coval** | Simulates thousands of conversations + evaluates | Paid platform. No open-source version. Requires demo call. |

**The single gap nobody fills:**

Every existing tool evaluates **what your agent already said**.
You bring the conversation; they score it.

Nobody simulates the conversation for you — free — against any endpoint.

That is what Coval built a company around.
AgentEval is the only open-source tool that does it.

---

## 2. Feature Spec

### Core user story
> I define a test scenario in YAML, run it against my agent's HTTP endpoint,
> and get a structured pass/fail evaluation report — from my terminal,
> with no mandatory cloud dependency, for free.

### v1 Scope Boundary (hard limit — documented in README)

v1 supports **synchronous JSON over HTTP POST only**.

Out of scope for v1 (v2 roadmap):
- WebSocket / SSE / streaming responses
- OAuth flows and cookie-based auth
- Multi-message response payloads
- Non-JSON payloads

### Five core features

**Feature 1 — YAML Test Case Schema**
Version-controlled YAML. Describes user persona, goal, outcome type,
success criteria, forbidden phrases, context facts, and scripted turns.
Supports three outcome types: `success`, `refusal`, `escalation`.
(`policy_block` collapsed into `refusal` with optional `policy_reason` —
explained in Section 4.)

**Feature 2 — Dual-Mode Simulation Engine**
- `--mode groq`: Groq free API (Llama 3.3 70B) plays the user side.
  Realistic. Has variance. Use for exploratory evaluation.
  Requires free Groq API key (no credit card).
- `--mode scripted`: Deterministic. Reads fixed turns from YAML.
  Zero API calls. Zero variance. Use for CI/CD gates.

Each scenario gets a **UUID4 session ID** generated at simulation start,
injected as `${SESSION_ID}` into the request template. Never shared
across scenarios — prevents state leakage on stateful agents.

**Feature 3 — 5-Dimension Local Scoring Engine**
All scorers run locally after the conversation completes. No API calls.
These are heuristic signals — not authoritative ground truth. README says
this explicitly. Use them to catch obvious failures, not to replace
human review.

**Feature 4 — React Dashboard**
Built React app bundled inside the Python package. `agenteval dashboard`
starts a local Python HTTP server, serves the pre-built static files,
and opens `localhost:5173` in the browser. Supports `--reports-dir` to
load all JSON reports from a folder for regression tracking.

**Feature 5 — CI/CD JSON Export + GitHub Action**
Versioned JSON report. Ready-made GitHub Action with proper health-check
readiness loop, pip caching, version-pinned install, and PR comment
with find-or-update logic.

---

## 3. The 5 Scorers — Detailed Specification

**Framing (README states this explicitly):**
These scorers are heuristic signals designed to catch obvious failures.
They are not production-quality evaluation ground truth. Calibrate them
against real transcripts before using them as hard CI gates.

All scorers return `0.0–1.0` plus a human-readable explanation and
flagged evidence items. Scorers return `None` when skipped.

---

### Shared Embeddings Module — `agenteval/scorers/_embeddings.py`

**FIX (3rd review #2): `embed()` and `cosine_similarity()` are called across
Scorers 1, 3, and 5 but never defined. They live in a shared module, lazily
initialised only when [ml] extras are installed.**

```python
# agenteval/scorers/_embeddings.py
from __future__ import annotations
import numpy as np

_model = None   # lazy — not loaded until first call

def _get_model():
    """Load sentence-transformers model on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed(text: str) -> np.ndarray:
    """Return a normalised embedding vector for text."""
    model = _get_model()
    return model.encode(text, normalize_embeddings=True)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalised vectors. Returns float 0–1."""
    # Because both vectors are already normalised, dot product == cosine sim
    return float(np.dot(a, b))

def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Batch embed for efficiency — used in coherence scorer."""
    model = _get_model()
    return list(model.encode(texts, normalize_embeddings=True))
```

All three scorers import from this module:
```python
# at top of task_completion.py, coherence.py, hallucination.py
try:
    from agenteval.scorers._embeddings import embed, cosine_similarity, embed_batch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
```

---

### Scorer 1: Task Completion
**Question:** Did the agent help the user achieve their stated goal?

**Separation of concerns (fixed from v2):**
The simulator's `[GOAL_ACHIEVED]` signal controls **loop termination only**.
Scorer 1 independently measures **how well** the goal was achieved.
These never conflict — one is flow control, one is quality measurement.

**Algorithm:**
```python
def score_task_completion(session, test_case):
    agent_turns = [t for t in session.turns if t.role == "agent"]

    # FIX #11: guard against short conversations
    final_n = min(3, len(agent_turns))
    final_turns = agent_turns[-final_n:]
    final_text = " ".join(t.content for t in final_turns).lower()

    if test_case.outcome_type == "refusal":
        # For refusal scenarios: check the agent correctly declined
        matched = any(kw.lower() in final_text
                      for kw in test_case.evaluation.refusal_keywords)
        hard_check = 1.0 if matched else 0.0
        intent = test_case.evaluation.refusal_intent

    elif test_case.outcome_type == "escalation":
        matched = any(kw.lower() in final_text
                      for kw in test_case.evaluation.escalation_keywords)
        hard_check = 1.0 if matched else 0.0
        intent = test_case.evaluation.escalation_intent

    else:  # success
        matched = any(kw.lower() in final_text
                      for kw in test_case.evaluation.success_keywords)
        hard_check = 1.0 if matched else 0.0
        intent = test_case.evaluation.success_intent

    # Soft check only available with [ml] extras
    if ML_AVAILABLE and intent:
        # Warn if success_intent is vague (< 8 words)
        if len(intent.split()) < 8:
            warn("success_intent is very short — vague intents may "
                 "produce unreliable scores. Be more specific.")
        last_agent = agent_turns[-1].content if agent_turns else ""
        soft = cosine_similarity(embed(last_agent), embed(intent))
        return (0.4 * hard_check) + (0.6 * soft)

    return hard_check  # core-only fallback
```

**Core dep:** string matching (stdlib)
**ML extra:** `sentence-transformers` all-MiniLM-L6-v2 (local)
**Threshold:** Default pass = 0.65

---

### Scorer 2: Instruction Following
**Question:** Did the agent obey its configured rules?

**Algorithm:**
```python
def score_instruction_following(session, test_case):
    agent_turns = [t for t in session.turns if t.role == "agent"]
    total_turns = len(agent_turns)

    forbidden = test_case.evaluation.forbidden_phrases or []
    required  = test_case.evaluation.required_phrases  or []

    forbidden_matches = 0
    evidence = []
    for turn in agent_turns:
        for phrase in forbidden:
            if phrase.lower() in turn.content.lower():
                forbidden_matches += 1
                evidence.append(f"Turn {turn.number}: '{phrase}' found")

    all_text = " ".join(t.content.lower() for t in agent_turns)
    required_found = sum(
        1 for p in required if p.lower() in all_text
    )

    # FIX #12: clamp — multiple violations in few turns must not go negative
    if total_turns > 0:
        forbidden_score = max(0.0, 1.0 - (forbidden_matches / total_turns))
    else:
        forbidden_score = 0.0

    required_score = (required_found / len(required)) if required else 1.0

    return (0.5 * forbidden_score) + (0.5 * required_score), evidence
```

**Limitation (documented):** Catches verbatim violations only — valid
paraphrases of forbidden phrases are invisible to this scorer.

**Core dep:** Python `re` / string matching (stdlib)
**Threshold:** Default pass = 0.80

---

### Scorer 3: Response Coherence
**Question:** Are agent responses contextually relevant to what was asked?

**Algorithm changes from v2:**
- Loop detection threshold lowered from 0.97 → **0.85** (FIX #13)
- Sliding window of last 3 agent turns for loop detection

```python
def score_coherence(session):
    # Requires [ml] extras — returns None if not installed
    if not ML_AVAILABLE:
        return None

    # FIX (3rd review new #3): derive turn lists from session
    user_turns  = [t for t in session.turns if t.role == "user"]
    agent_turns = [t for t in session.turns if t.role == "agent"]

    pairs = list(zip(user_turns, agent_turns))
    relevance_scores = []

    for user_turn, agent_turn in pairs:
        u_emb = embed(user_turn.content)
        a_emb = embed(agent_turn.content)
        sim = cosine_similarity(u_emb, a_emb)
        relevance_scores.append(sim)

    # Loop detection: sliding window of last 3 agent responses
    # FIX #13: threshold 0.85 catches paraphrase loops
    loop_detected = False
    if len(agent_turns) >= 3:
        window = agent_turns[-3:]
        for i in range(len(window)):
            for j in range(i + 1, len(window)):
                sim = cosine_similarity(embed(window[i].content),
                                        embed(window[j].content))
                if sim > 0.85:
                    loop_detected = True

    # Weight later turns more heavily
    weights = [1.0 + (i * 0.1) for i in range(len(relevance_scores))]
    weighted_score = sum(
        s * w for s, w in zip(relevance_scores, weights)
    ) / sum(weights)

    # Hard penalty for detected loops
    if loop_detected:
        weighted_score *= 0.7

    return max(0.0, min(1.0, weighted_score))
```

**ML extra required:** `sentence-transformers`
**Fallback:** `None` (excluded from aggregate, weights re-normalised)
**Threshold:** Default pass = 0.70

---

### Scorer 4: Turn Efficiency
**Question:** Did the agent resolve the goal in a reasonable number of turns?

**Algorithm (FIX #15 — refusal path never penalises fast refusals):**
**FIX (3rd review #5): min_turns check was dead code — the three elif branches
are exhaustive, so the `if actual < min_turns` could never execute.
Restructured so min_turns guard runs first on the success path.**

```python
def score_turn_efficiency(session, test_case):
    actual     = session.turns_used
    expected   = test_case.conversation.expected_turns
    min_turns  = test_case.conversation.min_turns
    terminated = session.termination_reason

    # Refusal/escalation: fewer turns to clean outcome = better.
    # Never apply min_turns penalty here.
    if test_case.outcome_type in ("refusal", "escalation"):
        if terminated == "goal_achieved":
            return 1.0 if actual <= expected else max(
                0.3, 1.0 - ((actual - expected) / expected) * 0.5
            )
        return 0.0  # outcome was never reached

    # Success path — check max_turns first (hard fail)
    if terminated == "max_turns_reached":
        return 0.0

    # FIX (3rd review #5): min_turns guard BEFORE the efficiency branches
    # so it is actually reachable. Suspiciously fast = agent short-circuited.
    if actual < min_turns:
        return 0.5

    # Efficiency scoring
    if actual <= expected:
        return 1.0
    elif actual <= expected * 1.5:
        ratio = (actual - expected) / (expected * 0.5)
        return max(0.5, 1.0 - ratio * 0.5)
    else:
        return 0.3  # rambling agent
```

**Core dep:** Pure Python (zero deps)
**Threshold:** Default pass = 0.60

---

### Scorer 5: Hallucination Risk
**Question:** Did the agent state facts not grounded in its context?

**Algorithm (FIX #14 — replace TF-IDF with sentence-transformers):**

TF-IDF over 4–6 short sentences is statistically unreliable — IDF weights
collapse because common words appear in every short "document". Replaced
with sentence-transformers cosine similarity, which is already available
in [ml] extras and gives consistent results on short corpora.

**FIX (2nd review #4): `extract_factual_sentences()` — fully defined.**

The previous plan called this function without specifying it. Definition:
A sentence is considered "factual-sounding" if it contains at least one
spaCy Named Entity (DATE, TIME, CARDINAL, MONEY, ORG, GPE, PERSON, PRODUCT,
LAW, QUANTITY) OR contains a definitive assertion verb pattern
("is", "are", "will", "costs", "requires", "takes", "opens", "closes")
with a subject. Sentences that are questions or imperatives are excluded.

```python
def extract_factual_sentences(text: str, nlp) -> list[str]:
    """
    Returns sentences that are factual-sounding:
    - Contains at least one Named Entity (DATE, TIME, CARDINAL, MONEY,
      ORG, GPE, PERSON, PRODUCT, LAW, QUANTITY), OR
    - Contains a definitive assertion verb with a subject noun phrase
    - Excludes questions (ends with ?) and imperatives (no subject)
    """
    FACTUAL_ENTITY_TYPES = {
        "DATE", "TIME", "CARDINAL", "MONEY", "ORG",
        "GPE", "PERSON", "PRODUCT", "LAW", "QUANTITY"
    }
    ASSERTION_VERBS = {
        "is", "are", "was", "were", "will", "costs", "cost",
        "requires", "require", "takes", "take", "opens", "closes",
        "opens", "available", "located", "operates"
    }

    doc = nlp(text)
    factual = []

    for sent in doc.sents:
        sent_text = sent.text.strip()

        # Exclude questions
        if sent_text.endswith("?"):
            continue

        # Check for named entities
        has_entity = any(
            ent.label_ in FACTUAL_ENTITY_TYPES for ent in sent.ents
        )

        # Check for assertion verb with subject
        has_assertion = any(
            token.lemma_.lower() in ASSERTION_VERBS
            and any(child.dep_ in ("nsubj", "nsubjpass")
                    for child in token.children)
            for token in sent
        )

        if has_entity or has_assertion:
            factual.append(sent_text)

    return factual
```

```python
def score_hallucination_risk(session, test_case):
    if not test_case.evaluation.context_facts:
        return None
    if not ML_AVAILABLE:
        return None

    # FIX (3rd review #1): nlp loaded at module level in hallucination.py,
    # not inside this function. Loading spaCy inside a per-call function is
    # 200–500ms overhead per turn. Module-level pattern:
    #   import spacy
    #   _nlp = None
    #   def _get_nlp():
    #       global _nlp
    #       if _nlp is None: _nlp = spacy.load("en_core_web_sm")
    #       return _nlp
    # Then call: extract_factual_sentences(turn.content, _get_nlp())
    context_embeddings = [embed(f)
                          for f in test_case.evaluation.context_facts]

    # FIX (3rd review new #3): derive agent_turns from session
    agent_turns = [t for t in session.turns if t.role == "agent"]

    flagged = []
    total_factual = 0

    for turn in agent_turns:
        # FIX (3rd review #1): nlp passed explicitly — never call without it
        factual_sentences = extract_factual_sentences(turn.content, _get_nlp())
        total_factual += len(factual_sentences)

        for sentence in factual_sentences:
            sent_emb = embed(sentence)
            # FIX (2nd review #3): list.index() on numpy arrays raises ValueError
            # because == comparison on arrays returns an array, not a bool.
            # Use np.argmax on explicit similarity list instead.
            sims = [cosine_similarity(sent_emb, cf_emb)
                    for cf_emb in context_embeddings]
            best_idx = int(np.argmax(sims))
            max_sim = sims[best_idx]
            if max_sim < 0.30:
                flagged.append({
                    "turn": turn.number,
                    "claim": sentence,
                    "max_context_similarity": round(max_sim, 3),
                    "closest_fact": test_case.evaluation.context_facts[best_idx]
                })

    if total_factual == 0:
        return 1.0  # no factual claims made — no risk

    return max(0.0, 1.0 - (len(flagged) / total_factual)), flagged
```

**ML extra required:** `sentence-transformers` + `spaCy en_core_web_sm`
**Fallback:** `None` (excluded from aggregate)
**Threshold:** Default pass = 0.75

---

### Aggregate Score + Re-normalisation

When scorers return `None`, weights re-normalise to sum to 1.0.

```python
BASE_WEIGHTS = {
    "task_completion":       0.30,
    "instruction_following": 0.25,
    "coherence":             0.20,
    "turn_efficiency":       0.15,
    "hallucination_risk":    0.10,
}

def calculate_aggregate(scores: dict) -> tuple[float, dict]:
    # FIX (3rd review new #1): scorers 2 and 5 return (score, evidence) tuples.
    # Extract just the float before arithmetic — evidence is for the report only.
    def _extract_score(v) -> float:
        return v[0] if isinstance(v, tuple) else v

    active = {k: v for k, v in scores.items() if v is not None}
    if not active:
        return 0.0, {}
    active_base = {k: BASE_WEIGHTS[k] for k in active}
    total = sum(active_base.values())
    normalised = {k: w / total for k, w in active_base.items()}
    return round(sum(_extract_score(scores[k]) * normalised[k] for k in active), 4), normalised
```

**Report records** which scorers were active and normalised weights used.

---

## 4. YAML Schema v3 — Consolidated Changes

**Key changes from v2:**
- `policy_block` collapsed into `refusal` with optional `policy_reason` (FIX #10)
- `session_id` field removed — generated as UUID4 per scenario at runtime
- `policy_reason` field added to refusal scenarios for documentation
- `--tag` filter now implemented (FIX #25)
- `success_intent` gets a validation warning if < 8 words (FIX #30)
- `response_path` uses jmespath dot notation (FIX #18)

```yaml
# test_cases/customer_support/booking_happy_path.yaml

meta:
  schema_version: "1.0"
  scenario_id: "booking-happy-path-001"
  name: "Patient successfully books a doctor's appointment"
  description: "Standard happy path — first-time patient books with Dr. Smith"
  version: "1.0.0"
  tags: ["healthcare", "booking", "happy-path"]  # filterable with --tag

agent:
  endpoint: "http://localhost:8080/chat"
  method: "POST"
  headers:
    Content-Type: "application/json"
    Authorization: "Bearer ${AGENT_API_KEY}"
  request_template: |
    {
      "session_id": "${SESSION_ID}",
      "message": "${USER_MESSAGE}"
    }
  # jmespath dot notation — FIX #18 (replaces jsonpath-ng)
  response_path: "response.text"
  timeout_seconds: 30
  health_check_path: "/health"

user_persona:
  name: "Sarah Chen"
  tone: "polite but slightly anxious"
  background: "First-time patient, not tech-savvy"
  opening_message: "Hi, I need to see a doctor about my knee."

# outcome_type options: success | refusal | escalation
# policy_block is now refusal + policy_reason (FIX #10)
outcome_type: success

conversation:
  max_turns: 12
  min_turns: 3
  expected_turns: 8

evaluation:
  # success_intent: be specific (>8 words) — vague intents score unreliably
  success_intent: "User receives a confirmed appointment with specific date, time, and doctor name"
  success_keywords:
    - "appointment confirmed"
    - "booked for"
    - "see you on"
    - "confirmation number"

  forbidden_phrases:
    - "I don't know"
    - "I cannot help"
    - "please call us"

  required_phrases:
    - "doctor"
    - "appointment"

  context_facts:
    - "Appointments available Monday through Friday"
    - "Dr. Smith specialises in orthopedics"
    - "New patient appointments require 30 minutes"
    - "Cancellation requires 24 hours notice"
    - "Clinic hours are 8am to 6pm"

  thresholds:
    task_completion:       0.65
    instruction_following: 0.80
    coherence:             0.70
    turn_efficiency:       0.60
    hallucination_risk:    0.75
    aggregate:             0.70

scripted_turns:
  - "Hi, I need to see a doctor about my knee. It's been hurting for two weeks."
  - "Is Dr. Smith available this week?"
  - "Thursday at 2pm works for me."
  - "My name is Sarah Chen, date of birth March 12 1990."
  - "Yes, that email is correct."
  - "Great, thank you so much."
```

**Refusal scenario (with policy_reason — FIX #10):**

```yaml
# test_cases/customer_support/booking_no_availability.yaml

meta:
  schema_version: "1.0"
  scenario_id: "booking-no-availability-001"
  name: "Agent correctly handles no availability"
  tags: ["customer-support", "refusal"]

outcome_type: refusal
# policy_reason documents WHY this is a refusal — for your team
# policy_reason: "No Sunday appointments available per clinic policy"

evaluation:
  refusal_intent: "Agent clearly communicates no availability without
    offering to book an unavailable slot"
  refusal_keywords:
    - "unfortunately"
    - "no availability"
    - "no slots"
    - "fully booked"
  forbidden_phrases:
    - "appointment confirmed"
    - "I can book that"

scripted_turns:
  - "I need to book an appointment for next Sunday."
  - "What about Monday at 9pm?"
  - "Is there anything at all this week?"
```

---

## 5. Simulation Engine — Robust `[GOAL_ACHIEVED]` Handling

### FIX #6 — Robust parsing

LLMs embed `[GOAL_ACHIEVED]` mid-sentence, wrap it in quotes, or
hallucinate it in turn 2. Strict parsing prevents silent truncation.

```python
def _parse_termination(response_text: str) -> tuple[str, str]:
    """
    Returns (cleaned_response_text, termination_signal | None)
    Termination signals: "goal_achieved" | None
    """
    lines = response_text.strip().split('\n')
    last_line = lines[-1].strip()

    # Normalise: strip punctuation, backticks, quotes, whitespace
    normalised = last_line.strip('`"\'.,!? ').strip()

    if normalised == "[GOAL_ACHIEVED]":
        # Remove the signal line from the visible response
        clean = '\n'.join(lines[:-1]).strip()
        return clean, "goal_achieved"

    # Check if it's embedded mid-sentence (still fire, but warn)
    if "[GOAL_ACHIEVED]" in response_text:
        clean = response_text.replace("[GOAL_ACHIEVED]", "").strip()
        return clean, "goal_achieved"

    return response_text, None
```

### FIX #7 — Scripted mode termination

Scripted mode emits `[GOAL_ACHIEVED]` only if the agent's last response
was non-empty and non-error. If the agent returned an error on every turn,
termination_reason = `"max_turns_reached"` — not `"goal_achieved"`.

```python
class ScriptedSimulator:
    def get_next_turn(self, turn_index, last_agent_response):
        if turn_index >= len(self.scripted_turns):
            # Only emit goal_achieved if last agent response was valid
            if last_agent_response and not last_agent_response.startswith("ERROR"):
                return "[GOAL_ACHIEVED]"
            return None  # engine will set termination_reason = max_turns_reached
        return self.scripted_turns[turn_index]
```

### Session + Turn + ScenarioError dataclasses

**FIX (3rd review #6): `Turn` dataclass was referenced by every scorer
but never formally defined.**
**FIX (3rd review #3): `ScenarioError` was instantiated in runner.py
but never defined or specified for the JSON report.**

```python
# agenteval/simulation/session.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Turn:
    number: int                          # 1-indexed turn number
    role: Literal["user", "agent"]
    content: str                         # the message text
    flags: list[str] = field(default_factory=list)
    # flags are set by scorers during evaluation, e.g.:
    # ["forbidden_phrase", "hallucination_risk", "loop_detected"]

@dataclass
class Session:
    scenario_id: str
    session_id: str          # UUID4 generated at simulation start (FIX #8)
    turns: list[Turn]
    turns_used: int
    termination_reason: str  # "goal_achieved" | "max_turns_reached" | "agent_error"

# agenteval/runner.py — defined alongside ScenarioResult
@dataclass
class ScenarioError:
    """
    Recorded when run_single_scenario() raises an unhandled exception.
    Appears in scenarios[] in the JSON report alongside normal results.
    """
    scenario_id: str
    scenario_name: str       # pulled from test_case.meta.name
    passed: bool = False
    aggregate_score: float = 0.0
    error: str = ""          # str(exception)
    error_type: str = ""     # type(exception).__name__
    conversation: list = field(default_factory=list)

    def to_report_dict(self) -> dict:
        return {
            "scenario_id":     self.scenario_id,
            "scenario_name":   self.scenario_name,
            "passed":          False,
            "aggregate_score": 0.0,
            "errored":         True,
            "error":           self.error,
            "error_type":      self.error_type,
            "scores":          {},
            "conversation":    [],
        }
```

### FIX (3rd review #8): `simulator_factory.py` — interface defined

```python
# agenteval/simulation/simulator_factory.py
from __future__ import annotations
from agenteval.simulation.scripted_simulator import ScriptedSimulator
from agenteval.schema.test_case import TestCase

class BaseSimulator:
    """Abstract interface all simulators must implement."""
    def get_opening_message(self) -> str:
        """Return the first user message to send to the agent."""
        raise NotImplementedError

    def get_next_turn(self, turn_index: int, last_agent_response: str) -> str | None:
        """
        Return the next user message, or None to let the engine handle
        termination (max_turns). Return "[GOAL_ACHIEVED]" to terminate
        the conversation as successfully completed.
        """
        raise NotImplementedError

def create_simulator(mode: str, test_case: TestCase) -> BaseSimulator:
    """
    Factory function — returns the correct simulator for the given mode.
    Called once per scenario at the start of simulation.
    """
    if mode == "scripted":
        return ScriptedSimulator(
            scripted_turns=test_case.scripted_turns,
            opening_message=test_case.user_persona.opening_message,
        )
    elif mode == "groq":
        # Lazy import — groq package only needed in this branch
        from agenteval.simulation.groq_simulator import GroqSimulator
        return GroqSimulator(
            persona=test_case.user_persona,
            goal=_resolve_goal(test_case),
            outcome_type=test_case.outcome_type,
        )
    else:
        raise ValueError(f"Unknown simulation mode: '{mode}'. Use 'scripted' or 'groq'.")
```

`engine.py` calls `create_simulator(mode, test_case)` at the start of each
scenario. It never imports simulators directly — always goes through the factory.

`engine.py` is fully async. Runner uses `asyncio.gather` for parallel
scenario execution. Default `--concurrency 4`. Prevents CI timeout
on large test suites.

```python
# runner.py
async def run_scenarios(test_cases, mode, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(tc):
        async with semaphore:
            return await run_single_scenario(tc, mode)

    # FIX (2nd review #2): return_exceptions=True prevents one bad scenario
    # from crashing the entire run. Exceptions are caught and recorded as
    # errored scenarios in the report — run always completes and writes output.
    results = await asyncio.gather(
        *[run_one(tc) for tc in test_cases],
        return_exceptions=True
    )

    # Separate successful results from exceptions
    processed = []
    for tc, result in zip(test_cases, results):
        if isinstance(result, Exception):
            processed.append(ScenarioError(
                scenario_id=tc.meta.scenario_id,
                scenario_name=tc.meta.name,   # FIX (3rd review new #2): was missing — no default, raises TypeError
                error=str(result),
                error_type=type(result).__name__,
            ))
        else:
            processed.append(result)
    return processed
```

**Groq rate limit handling (FIX #16):**

The limit is tokens per minute (TPM), not requests per day. Long
conversation histories inflate prompt size and can hit TPM mid-run.

```python
# groq_simulator.py
MAX_CONTEXT_TOKENS = 4000  # conservative limit per call

def _truncate_history(history: list[dict]) -> list[dict]:
    """
    Keep the most recent turns that fit within MAX_CONTEXT_TOKENS.
    Always keep the system prompt (index 0).
    FIX (2nd review #6): pop in pairs (user + agent together).
    Popping a single message leaves an orphaned agent turn with no
    preceding user message, which violates the Groq/OpenAI message
    format and causes API errors or degraded generation.

    Structure: [system, user1, agent1, user2, agent2, ...]
    Pairs start at index 1. Pop from index 1 (oldest user+agent pair).
    """
    # Rough token estimate: 1 token ≈ 4 chars
    total = sum(len(m["content"]) // 4 for m in history)

    while total > MAX_CONTEXT_TOKENS and len(history) > 3:
        # Remove oldest user+agent pair (indices 1 and 2 after system)
        # Check we have at least a pair to remove
        if len(history) > 3:
            removed_user  = history.pop(1)
            removed_agent = history.pop(1)  # now at index 1 after user removed
            total -= (len(removed_user["content"]) + len(removed_agent["content"])) // 4
        else:
            break  # can't remove further without losing system + last pair

    return history
```

---

## 6. Project Structure

```
agenteval/
│
├── README.md
├── LICENSE                           # MIT
├── pyproject.toml                    # full config — see Section 8
├── ruff.toml                         # linting config (FIX #20)
├── .env.example
├── .pre-commit-config.yaml           # ruff + mypy on commit (FIX #20)
│
├── .github/
│   └── workflows/
│       └── agenteval.yml             # CI template — ships with package
│
├── agenteval/
│   ├── __init__.py
│   ├── cli.py                        # Click: run, dashboard, validate
│   ├── config.py
│   │
│   ├── dashboard_dist/               # pre-built React bundle (FIX #3)
│   │   ├── index.html
│   │   ├── assets/
│   │   └── .gitkeep                  # populated by: cd dashboard && npm run build
│   │
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── test_case.py
│   │   └── report.py
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── engine.py                 # async main loop
│   │   ├── groq_simulator.py         # lazily imports groq (FIX #19)
│   │   ├── scripted_simulator.py
│   │   ├── simulator_factory.py
│   │   ├── agent_client.py           # httpx async, jmespath extraction
│   │   └── session.py
│   │
│   ├── scorers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── _embeddings.py            # FIX (3rd review #2): shared embed() + cosine_similarity()
│   │   ├── task_completion.py
│   │   ├── instruction_following.py
│   │   ├── coherence.py
│   │   ├── turn_efficiency.py
│   │   ├── hallucination.py          # sentence-transformers, no TF-IDF
│   │   └── aggregate.py
│   │
│   ├── runner.py                     # async gather, --concurrency
│   └── reporter.py
│
├── dashboard/                        # React source (not shipped in PyPI)
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── components/
│           ├── RunSelector.jsx
│           ├── ScenarioList.jsx
│           ├── RadarChart.jsx
│           ├── ConversationReplay.jsx
│           ├── ScoreBreakdown.jsx
│           ├── PassFailBadge.jsx
│           └── RegressionChart.jsx
│
├── test_cases/
│   ├── customer_support/
│   │   ├── booking_happy_path.yaml
│   │   ├── booking_no_availability.yaml    # outcome_type: refusal
│   │   ├── angry_user_escalation.yaml      # outcome_type: escalation
│   │   └── policy_refusal.yaml             # refusal + policy_reason
│   ├── healthcare/
│   │   ├── appointment_booking.yaml
│   │   └── prescription_refill_refusal.yaml
│   ├── hr_onboarding/
│   │   └── new_employee_faq.yaml
│   └── sales_qualification/
│       └── discovery_call.yaml
│
├── examples/
│   ├── mock_agent/
│   │   ├── main.py
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── README.md
│   └── github_action_example/
│       └── agenteval.yml
│
└── tests/
    ├── unit/
    │   ├── test_task_completion.py
    │   ├── test_instruction_following.py   # includes negative-clamp test
    │   ├── test_coherence.py               # includes loop detection at 0.85
    │   ├── test_turn_efficiency.py         # includes refusal fast-path test
    │   ├── test_hallucination.py           # sentence-transformers, no TF-IDF
    │   ├── test_aggregate_renormalisation.py
    │   ├── test_yaml_schema_validation.py
    │   ├── test_goal_achieved_parsing.py   # FIX #6 — all edge cases
    │   └── test_session_id_isolation.py    # FIX #8 — UUID per scenario
    ├── integration/
    │   └── test_end_to_end.py
    └── golden/
        ├── fixtures/golden_report_v1.json
        └── test_golden_reports.py
```

---

## 7. Dashboard — Bundling & Serving (FIX #3)

The React dashboard must be pre-built and bundled into the Python package.
`file://` breaks CORS when loading JSON files — a Python HTTP server
is required.

### Build step (runs once before PyPI publish, and in CI)
```bash
cd dashboard
npm install
npm run build
cp -r dist/* ../agenteval/dashboard_dist/
```

### FIX (2nd review #12): dashboard_dist git tracking strategy

Do NOT git-track the built files — every `npm run build` produces a large
diff. Add to `.gitignore`:

```
agenteval/dashboard_dist/*
!agenteval/dashboard_dist/.gitkeep
```

The `.gitkeep` file stays tracked so the directory exists on fresh clone.
The built files are generated locally and during PyPI publish CI only.

Add a `Makefile` target so contributors know how to build:

```makefile
# Makefile
.PHONY: build-dashboard

build-dashboard:
	cd dashboard && npm install && npm run build
	cp -r dashboard/dist/* agenteval/dashboard_dist/
	@echo "Dashboard built and copied to agenteval/dashboard_dist/"
```

README note: "Run `make build-dashboard` before `pip install -e .` if
you want `agenteval dashboard` to work from a local dev install."

### pyproject.toml package_data (required for PyPI to include the files)
```toml
[tool.setuptools.package-data]
agenteval = ["dashboard_dist/**/*"]
```

### CLI serving (cli.py)
```python
@cli.command()
@click.argument("report_file", required=False)
@click.option("--reports-dir", default=".", help="Directory of report JSON files")
@click.option("--port", default=5173)
def dashboard(report_file, reports_dir, port):
    """Serve the AgentEval dashboard locally."""
    import http.server
    import socketserver
    import webbrowser
    import threading
    import shutil   # FIX (3rd review #10): was missing, needed for shutil.copy()
    import json
    from pathlib import Path

    # Resolve the dashboard_dist directory inside the installed package
    dist_dir = Path(__file__).parent / "dashboard_dist"
    if not dist_dir.exists():
        raise click.ClickException(
            "Dashboard not built. Run: cd dashboard && npm run build"
        )

    # Copy report files into dashboard_dist/reports/ so they're serveable
    reports_path = dist_dir / "reports"
    reports_path.mkdir(exist_ok=True)

    if report_file:
        shutil.copy(report_file, reports_path)
    else:
        for f in Path(reports_dir).glob("agenteval_report_*.json"):
            shutil.copy(f, reports_path)

    # FIX (3rd review #4): HTTP static file servers don't expose directory
    # listings in a parseable format. React's RunSelector has no way to
    # discover which report JSON files exist. Solution: write a manifest.
    manifest = {
        "reports": [f.name for f in sorted(reports_path.glob("agenteval_report_*.json"))]
    }
    (reports_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # React app fetches /reports/manifest.json on load, then fetches each
    # listed filename. RunSelector.jsx: fetch('/reports/manifest.json')
    # → fetch('/reports/{name}') for each entry in manifest.reports

    # FIX (2nd review #7): directory= kwarg, no os.chdir()
    class _StaticHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(dist_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # suppress per-request logs — too noisy for a local tool

    with socketserver.TCPServer(("", port), _StaticHandler) as httpd:
        url = f"http://localhost:{port}"
        click.echo(f"Dashboard running at {url}")
        threading.Thread(target=webbrowser.open, args=(url,)).start()
        httpd.serve_forever()
```

### --reports-dir + auto-naming (FIX #29)
All report files are auto-named with timestamps:
```python
# reporter.py
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
default_filename = f"agenteval_report_{timestamp}.json"
```
`agenteval run --output-dir ./reports/` writes reports there automatically.
`agenteval dashboard --reports-dir ./reports/` loads all of them.
Regression chart in the dashboard shows score trends across all loaded reports.

---

## 8. pyproject.toml (complete — all fixes)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
# FIX (2nd review #1): correct backend string — "setuptools.backends.legacy:build" is invalid
build-backend = "setuptools.build_meta"

[project]
name = "agent-eval-cli"  # confirmed available on PyPI — pip install agent-eval-cli
version = "0.1.0"
description = "Open-source multi-turn AI agent simulation and evaluation"
readme = "README.md"
license = {text = "MIT"}
# FIX #5: spaCy 3.7 requires Python 3.9+
requires-python = ">=3.9"

dependencies = [
    # FIX #19: groq removed from core — lazily imported in groq_simulator.py
    # FIX (2nd review #13): httpx[asyncio] extra does not exist in httpx>=0.20
    # Async support (httpx.AsyncClient) is built-in. Remove the extra.
    "httpx>=0.27.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
    # FIX #18: jmespath replaces jsonpath-ng (cleaner dot notation)
    "jmespath>=1.0.1",
]

[project.optional-dependencies]

# FIX #19: groq in its own optional extra
groq = [
    "groq>=0.4.0",
]

# ML scorers — large deps, opt-in
ml = [
    "sentence-transformers>=2.7.0",
    "numpy>=1.26.0",
    "spacy>=3.7.0",
    # FIX #4: spaCy model as direct wheel — no manual python -m spacy download
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
]

# Full install (groq + ml) — FIX (2nd review #5): list explicitly, not
# self-referential — self-referential extras have pip compatibility issues
all = [
    "groq>=0.4.0",
    "sentence-transformers>=2.7.0",
    "numpy>=1.26.0",
    "spacy>=3.7.0",
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
]

# Dev tools — FIX #17
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-asyncio>=0.23",     # async test support
    "respx>=0.21",               # mock httpx requests in tests
    "ruff>=0.4.0",               # FIX #20: linting
    "mypy>=1.10.0",              # FIX #20: type checking
    "pre-commit>=3.7.0",
    "build>=1.2.0",
    "twine>=5.0.0",
]

[project.scripts]
agenteval = "agenteval.cli:cli"

# FIX #3: include pre-built dashboard in package
[tool.setuptools.package-data]
agenteval = ["dashboard_dist/**/*"]

[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.9"
strict = false
ignore_missing_imports = true

# FIX (2nd review #10): pytest-asyncio requires explicit mode config.
# Without this, async test functions are silently skipped in pytest>=0.23.
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

## 9. CLI Interface (complete)

```bash
# ── Install ───────────────────────────────────────────────────────
# PyPI name: agent-eval-cli  |  CLI command: agenteval  |  Import: agenteval
pip install agent-eval-cli              # core (scripted mode only, no ML)
pip install agent-eval-cli[groq]        # + Groq simulation mode
pip install agent-eval-cli[ml]          # + all 5 scorers (ML)
pip install agent-eval-cli[all]         # everything

# ── Run ───────────────────────────────────────────────────────────
# Scripted mode — deterministic, CI-safe, no API key
agenteval run test_cases/ --mode scripted

# Groq mode — realistic simulation (requires GROQ_API_KEY)
agenteval run test_cases/ --mode groq

# Filter by tag — FIX #25
agenteval run test_cases/ --mode scripted --tag healthcare

# Parallel execution — FIX #9
agenteval run test_cases/ --mode scripted --concurrency 4

# Fail process when below threshold (for CI exit code)
agenteval run test_cases/ --mode scripted --fail-on-threshold 0.70

# Save report with auto timestamp name — FIX #29
agenteval run test_cases/ --mode scripted --output-dir ./reports/

# Verbose — print full conversation transcript on failure — FIX #27
agenteval run test_cases/ --mode scripted --verbose

# ── Validate ──────────────────────────────────────────────────────
# FIX #26: validate supports both single file and directory
agenteval validate test_cases/booking.yaml
agenteval validate test_cases/               # validates all *.yaml in dir

# ── Dashboard ─────────────────────────────────────────────────────
# Load a single report
agenteval dashboard report.json

# Load all reports from a directory (for regression chart) — FIX #29
agenteval dashboard --reports-dir ./reports/

# ── Version ───────────────────────────────────────────────────────
agenteval --version
```

---

## 10. Groq Simulator Prompt

```python
GROQ_USER_SIMULATOR_PROMPT = """
You are roleplaying as a real user interacting with an AI agent.

PERSONA:
Name: {persona_name}
Background: {persona_background}
Tone: {persona_tone}

YOUR GOAL:
{goal}

EXPECTED OUTCOME:
{outcome_type_instruction}

STRICT RULES:
1. Stay in character as {persona_name}. Never break character.
2. Keep messages SHORT — 1 to 3 sentences. Real users don't write paragraphs.
3. Pursue your goal naturally across multiple turns.
4. If the agent answers your question, move toward your goal.
5. If the agent says something confusing, ask for clarification in character.
6. If the goal has been achieved, respond naturally to close the conversation.
7. If the agent refuses and that is the expected outcome, accept naturally.
8. If the agent is unhelpful after 3 turns, show mild frustration in character.
9. When the conversation outcome is fully achieved, output on its own line:
   [GOAL_ACHIEVED]

CONVERSATION SO FAR:
{conversation_history}

AGENT'S LAST MESSAGE:
{last_agent_message}

Your response as {persona_name} (short, natural, in character):
"""

# FIX (3rd review #7): {goal} in the prompt has no direct YAML field.
# Derived from evaluation fields based on outcome_type:
#   success    → evaluation.success_intent  ("User receives a confirmed appointment...")
#   refusal    → evaluation.refusal_intent  ("Agent clearly communicates no availability...")
#   escalation → evaluation.escalation_intent ("User is connected to a human agent...")
#
# groq_simulator.py resolves this at runtime:
def _resolve_goal(test_case) -> str:
    ot = test_case.outcome_type
    if ot == "success":
        return test_case.evaluation.success_intent or "Complete the user's request."
    elif ot == "refusal":
        return test_case.evaluation.refusal_intent or "Request something the agent should decline."
    elif ot == "escalation":
        return test_case.evaluation.escalation_intent or "Escalate to a human agent."
    return "Complete the conversation naturally."

OUTCOME_INSTRUCTIONS = {
    "success":    "You are trying to complete a task. Keep going until it's done.",
    "refusal":    "You are making a request the agent should decline. Accept the refusal.",
    "escalation": "You have a complex issue. You want to speak to a human agent.",
}
```

---

## 11. GitHub Action (production-grade — all fixes)

```yaml
# .github/workflows/agenteval.yml
name: AgentEval — AI Agent Tests

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  agent-eval:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      # FIX (2nd review #22 retained): pip cache
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
          restore-keys: ${{ runner.os }}-pip-

      # FIX (2nd review #11): sentence-transformers downloads models at runtime
      # (~80MB all-MiniLM-L6-v2) — not covered by pip cache. Cache separately.
      # NOTE: Only needed when using agenteval[ml]. Core install (scripted mode)
      # does NOT download any models and does NOT need this cache entry.
      # Remove this step if using core install only (recommended for CI).
      - name: Cache HuggingFace models
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-models-minilm-v6

      # FIX #24: version-pinned install — no surprise upgrades
      - name: Install AgentEval
        run: pip install "agent-eval-cli==0.1.0"

      # FIX (3rd review #9): docker-compose (v1 standalone binary) is not
      # available on Docker Desktop >=4.x or modern CI runners. Use
      # docker compose (v2 subcommand) instead.
      - name: Start agent under test
        run: docker compose up -d agent   # CHANGE THIS: replace 'agent' with your service name

      # Proper readiness check — retry loop, not sleep (v2 fix retained)
      - name: Wait for agent readiness
        run: |
          echo "Waiting for agent..."
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

      # FIX #23: find-or-update PR comment (no duplicate blocks per push)
      - name: Post or update PR comment
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const path = require('path');

            // Find latest report file
            const reports = fs.readdirSync('./reports/')
              .filter(f => f.startsWith('agenteval_report_'))
              .sort().reverse();
            if (!reports.length) { console.log('No reports found.'); return; }

            const report = JSON.parse(
              fs.readFileSync(path.join('./reports/', reports[0]))
            );
            const s = report.summary;
            const icon = s.overall_pass ? '✅' : '❌';
            const rows = report.scenarios.map(sc =>
              `| ${sc.scenario_name} | ${sc.passed ? '✅' : '❌'} ` +
              `| ${(sc.aggregate_score * 100).toFixed(0)}% |`
            ).join('\n');
            const body =
              `<!-- agenteval-result -->\n` +
              `## ${icon} AgentEval Results\n\n` +
              `**Score:** ${(s.aggregate_score * 100).toFixed(1)}% ` +
              `| **Passed:** ${s.passed}/${s.total_scenarios} scenarios\n\n` +
              `| Scenario | Pass | Score |\n|---|---|---|\n${rows}\n\n` +
              `> Mode: \`${report.run_config.mode}\` ` +
              `| Scorers: ${report.run_config.scorers_active.join(', ')}`;

            // Find existing comment to update (FIX #23)
            const comments = await github.rest.issues.listComments({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
            });
            const existing = comments.data.find(
              c => c.body.includes('<!-- agenteval-result -->')
            );
            if (existing) {
              await github.rest.issues.updateComment({
                comment_id: existing.id,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body
              });
            } else {
              await github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body
              });
            }
```

---

## 12. Runtime Privacy Warning (FIX #28)

When `--mode groq` and `context_facts` is non-empty, emit a warning
before starting — not just in the README.

```python
# runner.py — before simulation starts
def _check_privacy(test_case, mode):
    if mode == "groq" and test_case.evaluation.context_facts:
        click.echo(
            click.style(
                "⚠  Privacy notice: context_facts will be sent to Groq's API "
                "in groq mode. Use --mode scripted for sensitive or proprietary data.",
                fg="yellow"
            )
        )
```

---

## 13. Test Strategy

```
tests/unit/
├── test_task_completion.py
│   ├── test_short_conversation_final_n_guard      # FIX #11
│   ├── test_success_path
│   ├── test_refusal_path
│   └── test_escalation_path

├── test_instruction_following.py
│   ├── test_forbidden_score_clamp_no_negative     # FIX #12
│   ├── test_multiple_violations
│   └── test_required_phrases_all_found

├── test_coherence.py
│   ├── test_loop_detection_at_085_threshold       # FIX #13
│   ├── test_paraphrase_loop_detected
│   └── test_relevant_responses_pass

├── test_turn_efficiency.py
│   ├── test_refusal_fast_no_penalty               # FIX #15
│   ├── test_success_linear_decay
│   └── test_max_turns_hard_fail

├── test_hallucination.py
│   ├── test_sentence_transformers_not_tfidf       # FIX #14
│   ├── test_no_context_facts_returns_none
│   └── test_valid_claim_passes

├── test_aggregate_renormalisation.py
│   ├── test_one_scorer_skipped
│   ├── test_two_scorers_skipped
│   └── test_all_scorers_active

├── test_yaml_schema_validation.py
│   ├── test_valid_yaml_passes
│   ├── test_missing_required_field_raises
│   └── test_policy_block_rejected               # policy_block removed in v3

├── test_goal_achieved_parsing.py                 # FIX #6
│   ├── test_clean_last_line
│   ├── test_embedded_mid_sentence
│   ├── test_wrapped_in_backticks
│   └── test_never_emitted_returns_none

└── test_session_id_isolation.py                  # FIX #8
    └── test_each_scenario_gets_unique_uuid


tests/integration/
└── test_end_to_end.py
    ├── test_scripted_mode_full_run_with_mock_agent
    └── test_exit_code_nonzero_on_threshold_fail

tests/golden/
├── fixtures/golden_report_v1.json
└── test_golden_reports.py
```

**FIX (2nd review #14): conftest.py — Docker fixture specification.**

Without this file, `test_end_to_end.py` has no `mock_agent` fixture to
import — pytest will error with `fixture 'mock_agent' not found`.

```python
# tests/conftest.py
import subprocess
import time
import httpx
import pytest

MOCK_AGENT_COMPOSE = "examples/mock_agent/docker-compose.yml"
MOCK_AGENT_HEALTH  = "http://localhost:8080/health"

def _wait_for_health(url: str, timeout: int = 30):
    """Poll health endpoint with backoff until ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(
        f"Mock agent at {url} did not become ready in {timeout}s"
    )

@pytest.fixture(scope="session")
def mock_agent():
    """Start the mock agent Docker container for the test session."""
    # FIX (3rd review #9): docker compose (v2) not docker-compose (v1)
    subprocess.run(
        ["docker", "compose", "-f", MOCK_AGENT_COMPOSE, "up", "-d"],
        check=True
    )
    _wait_for_health(MOCK_AGENT_HEALTH)
    yield  # all tests in the session run here
    subprocess.run(
        ["docker", "compose", "-f", MOCK_AGENT_COMPOSE, "down"],
        check=True
    )
```

---

## 14. JSON Report Schema (versioned)

```json
{
  "schema_version": "1.0",
  "agenteval_version": "0.1.0",
  "run_id": "run_20260423_143052",
  "timestamp": "2026-04-23T14:30:52Z",
  "simulator_mode": "scripted",
  "simulator_model": null,
  "run_config": {
    "mode": "scripted",
    "concurrency": 4,
    "fail_on_threshold": 0.70,
    "scorers_active": ["task_completion", "instruction_following", "turn_efficiency"],
    "scorers_skipped": ["coherence", "hallucination_risk"],
    "skip_reason": "agenteval[ml] not installed"
  },
  "summary": {
    "total_scenarios": 3,
    "passed": 2,
    "failed": 1,
    "errored": 0,
    "aggregate_score": 0.79,
    "aggregate_score_method": "plain_mean",
    "min_scenario_score": 0.41,
    "overall_pass": false,
    "_aggregate_note": "aggregate_score is the plain mean of all non-errored scenario scores. A single failing scenario can be masked by high scores elsewhere — always check min_scenario_score for outliers."
  },
  "scenarios": [
    {
      "scenario_id": "booking-happy-path-001",
      "scenario_name": "Patient books a doctor's appointment",
      "outcome_type": "success",
      "session_id": "f3a2b1c0-...",
      "simulator_mode": "scripted",
      "passed": true,
      "aggregate_score": 0.84,
      "aggregate_weights_used": {
        "task_completion": 0.429,
        "instruction_following": 0.357,
        "turn_efficiency": 0.214
      },
      "turns_used": 6,
      "termination_reason": "goal_achieved",
      "scores": { "..." : "..." },
      "conversation": [ "..." ],
      "errors": []
    }
  ]
}
```

---

## 15. Day-by-Day Build Plan

### Day 0 (Before starting)
- PyPI name `agent-eval-cli` already confirmed available — locked in
- If taken, pick fallback name and update all references
- Run `pip install build twine ruff mypy pytest pytest-cov pytest-asyncio respx`
- Follow Brooke Hopkins on Twitter

**Deliverable:** Name locked: `agent-eval-cli` on PyPI. Build tools installed. Day 1 unblocked.

---

### Day 1 — Foundation + Schema
- Set up package structure with complete `pyproject.toml`:
  - FIX (2nd review #1): `build-backend = "setuptools.build_meta"`
  - FIX (2nd review #5): `[all]` extra lists deps explicitly
  - FIX (2nd review #10): `asyncio_mode = "auto"` in pytest config
  - FIX (2nd review #13): `httpx>=0.27.0` (no `[asyncio]` extra)
- Create `ruff.toml`, `.pre-commit-config.yaml`, `Makefile` with `build-dashboard`
- FIX (2nd review #12): add `agenteval/dashboard_dist/*` to `.gitignore`
- Write Pydantic schemas: `TestCase`, `UserPersona`, `Evaluation`
- `outcome_type` enum: `success | refusal | escalation`
- `policy_reason` optional field on refusal scenarios
- Write `session.py` with `session_id: str` and `termination_reason: str`
- Write `agent_client.py` — async httpx POST, jmespath extraction, 30s timeout
- Unit tests for schema validation

**Deliverable:** `agenteval validate test_cases/booking_happy_path.yaml` works

---

### Day 2 — Simulation Engine
- Write `scripted_simulator.py` — conditional `[GOAL_ACHIEVED]` (FIX #7)
- Write `groq_simulator.py`:
  - Lazy `import groq` (FIX #19)
  - FIX (2nd review #6): `_truncate_history` pops user+agent pairs, not singles
  - Token context truncation strategy
- Write `simulator_factory.py`
- Write `engine.py`:
  - Async, robust `[GOAL_ACHIEVED]` parsing with normalisation (FIX #6)
  - UUID4 session ID per scenario (FIX #8)
- Write `runner.py`:
  - FIX (2nd review #2): `asyncio.gather(..., return_exceptions=True)`
  - FIX (2nd review #9): `sorted(Path(dir).rglob("*.yaml"))` for file discovery
  - FIX (2nd review #8): `summary.aggregate_score` = plain mean + `min_scenario_score`
  - `--concurrency N` via semaphore
- Test scripted mode against mock agent

**Deliverable:** Concurrent scenarios run. `termination_reason` populated.
Single scenario failure doesn't crash the run — recorded as errored scenario.

---

### Day 3 — Scorers 2 & 4 (core, no ML)
- Write `instruction_following.py` — `max(0.0, ...)` clamp (FIX #12)
- Write `turn_efficiency.py` — refusal fast-path, no min_turns penalty (FIX #15)
- Write `aggregate.py` — re-normalisation
- Write `base.py`
- Unit tests for all (including edge cases from review)

**Deliverable:** Scorers 2 + 4 running with correct edge case handling

---

### Day 4 — Scorers 1, 3, 5 (ML extras)
- Write `task_completion.py` — `min(3, len(turns))` guard (FIX #11), short intent warning (FIX #30)
- Write `coherence.py` — loop threshold 0.85, sliding window (FIX #13)
- Write `hallucination.py` — sentence-transformers cosine, no TF-IDF (FIX #14)
- All three degrade gracefully without [ml]
- Unit tests including all edge cases from review

**Deliverable:** All 5 scorers running correctly

---

### Day 5 — Reporter + CLI + UX
- Write `reporter.py` — auto-timestamped filenames, `--output-dir` support (FIX #29)
- Write `cli.py`:
  - `run` with `--verbose` (FIX #27), `--tag` filter (FIX #25), `--concurrency`, `--output-dir`
  - `validate` with directory support (FIX #26)
  - `dashboard` with `--reports-dir` (FIX #29)
  - Runtime privacy warning in `run` for groq + context_facts (FIX #28)
- Test full end-to-end: `agenteval run test_cases/ --mode scripted --output-dir ./reports/`

**Deliverable:** Complete CLI with all UX fixes

---

### Day 6 — Tests + Action + Packaging
- Write integration test (`test_end_to_end.py`) with mock agent
- Write golden report test with fixture
- Write all unit tests from Section 13
- Write GitHub Action template with all fixes (#21, #22, #23, #24)
- Run `ruff check .` — zero warnings
- Run `mypy agenteval/` — clean
- Test `pip install -e ".[all]"` on fresh virtualenv
- Test `pip install -e ".[dev]"` — `pytest` runs clean

**Deliverable:** All tests green. Linting clean. Package installs correctly.

---

### Day 7 — React Dashboard + Bundling
- Scaffold React + Vite + TailwindCSS
- Build all components (RunSelector, RadarChart, ConversationReplay, ScoreBreakdown, RegressionChart)
- RegressionChart reads all loaded reports — works with `--reports-dir` auto-named files
- RadarChart shows "N/A" for skipped scorers (not 0)
- `npm run build` → copy `dist/` → `agenteval/dashboard_dist/` (FIX #3)
- Add `package_data` to `pyproject.toml`
- Test `agenteval dashboard --reports-dir ./reports/` serves correctly, CORS works

**Deliverable:** Dashboard bundled in package. `agenteval dashboard` works after `pip install agent-eval-cli`.

---

### Day 8 — Test Cases + Polish + Demo
- Write 8 YAML test cases: 3 outcome types × multiple verticals
- Confirm all scripted_turns match expected agent flows
- Write README: install profiles, two modes, scorer framing (heuristics), privacy note, v1 scope
- Run full `pytest` — all green
- Record 90s demo video: YAML → scripted run → verbose output → dashboard with regression chart

**Deliverable:** Shippable project with demo video

---

### Day 9 — Ship

```bash
# Build and publish — FIX #1 (not 'pip publish')
cd dashboard && npm run build && cp -r dist/* ../agenteval/dashboard_dist/
cd ..
python -m build
twine check dist/*
twine upload dist/*
```

- Push to GitHub with README, screenshots, architecture diagram
- Submit GitHub Action to Marketplace
- Post on Twitter + LinkedIn with demo video
- DM Brooke Hopkins

---

## 16. Comparison Table

| Capability | LangSmith | Braintrust | Promptfoo | DeepEval | **AgentEval** |
|---|---|---|---|---|---|
| **Multi-turn simulation** | ❌ | ❌ | ❌ | ❌ | ✅ |
| Deterministic CI mode | ❌ | ❌ | ✅ | ✅ | ✅ |
| Framework agnostic (HTTP POST) | Partial | ✅ | ✅ | ✅ | ✅ |
| Fully free (no credit card) | ❌ | ❌ | ✅ | ✅ | ✅ |
| YAML test cases (versionable) | ❌ | ❌ | ✅ | ❌ | ✅ |
| Multiple outcome types | ❌ | ❌ | Partial | ❌ | ✅ |
| Lightweight core install | ❌ | ❌ | ✅ | Partial | ✅ |
| Local scoring (no cloud eval) | ❌ | ❌ | Partial | ✅ | ✅ |
| Tag filtering | ❌ | ❌ | ✅ | ❌ | ✅ |
| Verbose failure output | Partial | ✅ | ✅ | Partial | ✅ |
| CI/CD exit code + PR comment | ❌ | ✅ | ✅ | ✅ | ✅ |
| Visual conversation replay | ❌ | Partial | ❌ | ❌ | ✅ |
| Regression chart | ❌ | ✅ | ❌ | ❌ | ✅ |
| Open source | ❌ | ❌ | ✅ | ✅ | ✅ |

---

## 17. Outreach Timing

| Milestone | Action |
|---|---|
| Day 0 | Follow Brooke on Twitter. Like her 3 most recent posts. |
| Day 5 | Teaser tweet: "Building an open-source AI agent testing framework. Scripted + LLM simulation. Who's still testing agents by hand?" |
| Day 8 | Full demo post on Twitter + LinkedIn. `#buildinpublic #LLMOps #AI` |
| 50 GitHub stars | DM Brooke on Twitter AND LinkedIn the same day |
| **Twitter DM** | "Hi Brooke, built AgentEval — open-source agent testing with multi-turn simulation (Groq mode) + deterministic scripted mode for CI. YAML test cases, 5 local scorers, pip installable: `pip install agent-eval-cli`. [GitHub]. Only free tool that simulates conversations. Would love to chat." |
| **LinkedIn DM** | Full message from spreadsheet outreach template |

---

*Total build: 9 days (+ Day 0 pre-check). Total cost: $0.
v1 scope: synchronous JSON POST agents only.
The only open-source multi-turn agent simulator in existence.*