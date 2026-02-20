## Introduction
Panel Monitoring Agent

Flexible monitoring agent built with LangGraph and Vertex AI (Gemini) / OpenAI + LangSmith.
Supports running locally for development or inside Google Cloud for production.

**Architectural Ownership:** I personally designed the LangGraph state machine, defined the GCP infrastructure (Pub/Sub & Cloud Run), and implemented the Firestore security layer.

**Production Standards:** The code follows strict Pydantic data validation and Ruff linting to ensure system reliability and clean-formatted audit logs.

**Security & IP:** To protect sensitive data, specific business fraud rules and private dataset weights are excluded. The system uses GCP IAM service accounts rather than hardcoded API keys.

## Setup

### Python version
Use Python 3.12+ for compatibility with LangGraph and project dependencies.

Check your version:
```
python3 --version
```

### Create an environment and install dependencies

This project uses uv (a fast alternative to pip).

# Create or reset the virtual environment
```
uv venv .venv --clear
```

# Install all dependencies from pyproject.toml / uv.lock
```
uv sync
```

# Activate your virtual environment
```
source .venv/bin/activate
```

### Running notebooks
If you don't have Jupyter set up, follow installation instructions [here](https://jupyter.org/install).
```
$ jupyter notebook
```

### Setting up env variables
Briefly going over how to set up environment variables. You can also 
use a `.env` file with `python-dotenv` library.
#### Mac/Linux/WSL
```
$ export API_ENV_VAR="your-api-key-here"
```

Create a `.env` file in the repo root (auto-loaded), or set environment variables manually:

```
# Google / GCP
GOOGLE_APPLICATION_CREDENTIALS="path/to/creds.json"
GOOGLE_CLOUD_PROJECT="your-gcp-project"
GOOGLE_CLOUD_LOCATION="us-central1"
FIRESTORE_DATABASE_ID="your-firestore-db-id"

# Agent
ENVIRONMENT=local                        # set to "local" for dev credential loading
PANEL_PROJECT_ID="your-panel-project-id" # Firestore project namespace
PANEL_DEFAULT_PROVIDER=vertexai          # vertexai | openai | genai
VERTEX_MODEL=gemini-2.5-flash            # model override for Vertex AI

# OpenAI (if used)
OPENAI_API_KEY="your-key"

# LangSmith (optional — tracing and eval)
LANGSMITH_API_KEY="your-key"
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="your-langsmith-project"
```

Infrastructure & Smoke Checks

Since the agent now uses Asynchronous Firestore, always use the provided smoke scripts to verify connectivity before running the agent.
Auth/connectivity:
```
uv run python -m panel_monitoring.scripts.smoke_auth_check
```
Data Pipeline Check

Minimal write/read path:
```
uv run python -m panel_monitoring.scripts.smoke_datastore
```

### Seeding Firestore with demo data

You can seed Firestore with a sample project and event to verify the client setup.

Run the seeder from the repo root:

```
uv run python -m panel_monitoring.scripts.seed_firestore
```

### Verify Firestore seed

After seeding, you can quickly check the latest event was written:

```
uv run python -m panel_monitoring.scripts.peek_firestore
```

This will print the most recent event document under your project, e.g.:

```
mH2rAYijvDXOhDH6kLji {'type': 'signup', 'source': 'web', ...}
```

### LangSmith (optional — tracing and eval)

Sign up at [smith.langchain.com](https://smith.langchain.com/). Once set up, every agent run is traced automatically. To seed the eval dataset:

```
uv run python testing-examples/datasets/seed_langsmith_dataset.py
uv run python testing-examples/datasets/tag_dataset_version.py
```

### Running the Panel Monitoring Agent

The Panel Monitoring Agent supports three execution modes, depending on your workflow and environment.

#### Run via the unified CLI:

OpenAI
```
uv run python -m panel_monitoring.scripts.panel_agent --provider openai
```

Vertex AI (Gemini)
```
uv run python -m panel_monitoring.scripts.panel_agent --provider vertexai
```

GenAI (Google Generative AI API)
```
uv run python -m panel_monitoring.scripts.panel_agent --provider genai
```

### Set up LangGraph Studio

* LangGraph Studio is a custom IDE for viewing and testing agents.
* Studio can be run locally and opened in your browser on Mac, Windows, and Linux.
* See documentation [here](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#local-development-server) on the local Studio development server and [here](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server). 
* Graphs for LangGraph Studio are in the `module-x/studio/` folders.
* To start the local development server, run the following command in your terminal in the `/studio` directory each module:

```
langgraph dev
```

You should see the following output:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

* To use Studio, you will need to create a .env file with the relevant API keys
* Run this from the command line to create these files for module 1 to 5, as an example:
```
for i in {1..5}; do
  cp module-$i/studio/.env.example module-$i/studio/.env
  echo "OPENAI_API_KEY=\"$OPENAI_API_KEY\"" > module-$i/studio/.env
done
echo "TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> module-4/studio/.env
```

### Production Mode via Google Cloud Event Trigger

* In production, the agent runs automatically in response to real user or system events.

Flow:

* Pub/Sub event is published (e.g., user signup, suspicious behavior, survey action)

* Cloud Function Gen2 receives the event and invokes the agent

* Agent processes the event using Vertex AI

* Final action is logged and pushed to downstream services (notifications, dashboards, etc.)

This mode is best for:

* Production automation

* Scalable event-driven workflows

* Real-time monitoring of panel activity

### Testing

Run the full test suite:
```
uv run pytest tests/
```

Run specific suites:
```
# Golden tests (classification accuracy against hand-labeled production data)
uv run pytest tests/golden_tests/

# Unit tests (prompt spec, injection detection, retry logic)
uv run pytest tests/test_prompt_spec.py tests/test_injection_detection.py tests/test_injection_ml.py tests/test_retry.py
```

Golden tests use hardcoded local prompts (not Firestore) for stability — a prompt change in Firestore will never silently break them.

### Security: Prompt Injection Detection

The agent runs a two-layer injection scan on all untrusted inputs before they reach the LLM:

1. **Regex scan** (`utils.detect_prompt_injection`) — fast pattern matching for known injection techniques (instruction overrides, role hijacking, delimiter escapes, output manipulation)
2. **ML scan** (`injection_detector.detect_injection_ml`) — DeBERTa v3 model (`protectai/deberta-v3-base-prompt-injection-v2`) for freeform text fields

If injection is detected and the LLM still returns `normal_signup`, the result is overridden to `suspicious_signup` with confidence ≥ 0.85.

### RAG: Business Context Ingestion

The agent retrieves similar fraud patterns from Firestore using vector search to ground LLM decisions in real case history.

To ingest or refresh the business context:
```
uv run python -m panel_monitoring.scripts.ingest_business_context
```

This chunks `panel_monitoring/data/business_context.txt` and writes embeddings to the `fraud_patterns` collection in Firestore.

### Prompt Management

Prompts are stored in Firestore as versioned `PromptSpec` documents and are **immutable after creation**. Never edit or delete an existing version — past runs reference it by ID, and changing it would corrupt the audit trail.

#### Push a new prompt version

Edit `panel_monitoring/app/prompts.py`, then run:

```
uv run python -m panel_monitoring.scripts.push_prompt_to_firestore
```

This creates a new document (e.g. `signup_classification_v4`) with `deployment_status = pre_live`. The agent will not use it yet.

#### Promote to live

In the Firestore console:
1. Set the old live version's `deployment_status` → `deactivated`
2. Set the new version's `deployment_status` → `live` (or `canary` first for a gradual rollout)

#### Deployment statuses

| Status | Meaning |
|--------|---------|
| `pre_live` | Created, not yet active |
| `canary` | Receiving a portion of traffic (manual routing) |
| `live` | Active — picked up by the agent |
| `failover` | Used if the live version fails |
| `deactivated` | Retired, kept for audit history |

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `system_prompt` | str | The system-level instructions for the LLM |
| `user_prompt` | str | The user-turn template (must contain `{event}`) |
| `version` | str | Integer string, auto-incremented on each push |
| `deployment_role` | str | Which agent uses this prompt (e.g. `signup_classification`) |
| `model_host` | PromptModelHost | Provider: `vertexai`, `gemini`, `openai`, `anthropic` |
| `model_name` | str | Model override (e.g. `gemini-2.5-flash`) |

### Code Quality: Ruff (lint & format)

This repo uses Ruff for linting and formatting.

Install (adds to your project via uv):
```
uv add ruff
```

Run lint checks:
```
uv run ruff check .
```

Auto-fix what can be fixed safely:
```
uv run ruff check . --fix
```

Format code (Ruff’s formatter):
```
uv run ruff format .
```
