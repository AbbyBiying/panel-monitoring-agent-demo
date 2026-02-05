## Introduction
Panel Monitoring Agent

Flexible monitoring agent built with LangGraph and Vertex AI (Gemini) / OpenAI + LangSmith.
Supports running locally for development or inside Google Cloud for production.

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

Create a .env file in the repo root (auto-loaded), or set environment variables manually:

# Google credentials
```
GOOGLE_APPLICATION_CREDENTIALS="path/to/creds.json"
GOOGLE_CLOUD_PROJECT="your-gcp-project"
GOOGLE_CLOUD_LOCATION="us-central1"
FIRESTORE_DATABASE_ID="(default)"       
```
# OpenAI (if used)
```
OPENAI_API_KEY="your-key"
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

# LangSmith (optional monitoring/tracing)
```
LANGSMITH_API_KEY="your-key"
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="panel-monitoring-agent"
# Tavily (optional web search)
```
TAVILY_API_KEY="your-key"

reference: <!-- https://docs.langchain.com/langsmith/manage-datasets -->
Run 
```
uv run python testing-examples/datasets/seed_langsmith_dataset.py
``` 
to create the **Panel Monitoring Cases** dataset in LangSmith, seeding evaluation examples aligned with GraphState for classification and action testing.

Tag latest as v1:
``` 
uv run python testing-examples/datasets/tag_dataset_version.py
``` 

### Set OpenAI API key
* If you don't have an OpenAI API key, you can sign up [here](https://openai.com/index/openai-api/).
*  Set `OPENAI_API_KEY` in your environment 

### Sign up and Set LangSmith API
* Sign up for LangSmith [here](https://smith.langchain.com/), find out more about LangSmith
* and how to use it within your workflow [here](https://www.langchain.com/langsmith), and relevant library [docs](https://docs.smith.langchain.com/)!
*  Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT="panel-monitoring-agent"` in your environment 

### Running the Panel Monitoring Agent

The Panel Monitoring Agent supports three execution modes, depending on your workflow and environment.
# Run via the unified CLI (with FunctionProvider):
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
