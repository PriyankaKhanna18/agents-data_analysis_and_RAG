# Applied Gen AI: Deep Agents Labs

This repo has two agents built with [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) and LangGraph, both running on Cohere models:

| Agent | Graph id | Entry point | What it does |
| --- | --- | --- | --- |
| **Data Analyst** | `data_analyst` | [deep_agent_from_scratch.py](deep_agent_from_scratch.py) | Writes and runs pandas/matplotlib scripts against CSVs in a local workspace, hands charting to a `visualizer` subagent and shows the charts inline in LangGraph Studio. |
| **Docs RAG Agent** | `rag_agent` | [rag_agent/rag_agent.py](rag_agent/rag_agent.py) | Answers LangChain questions from an embedded index of the LangChain docs, using parallel `chunk-analyst` subagents to read the retrieved chunks. |

There's also a smaller **Sandbox** coding agent ([Sandbox.py](Sandbox.py)) that creates a Python package and runs `pytest` until the tests pass.

For the design decisions behind these agents, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Prerequisites

- **Python 3.11 or later** (tested on 3.14)
- A **Cohere API key**. A free trial key works, and the agents are rate-limited to stay within its limits. Get one at <https://dashboard.cohere.com/api-keys>.
- Optional: a **LangSmith API key** for tracing.

> ⚠️ **No sandboxing.** The Data Analyst and Sandbox agents run shell commands **directly on your machine**. Their file tools are confined to their workspace folders, and their shell gets only `PATH` (no API keys), but shell commands themselves are not isolated. Run them only on a machine you're comfortable experimenting on.

---

## Setup

```bash
# 1. Clone
git clone <your-repo-url>
cd <repo-folder>

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API keys
cp .env.example .env               # Windows: copy .env.example .env
```

Then open `.env` and fill in at least `COHERE_API_KEY`:

```dotenv
COHERE_API_KEY=your-cohere-key
LANGSMITH_API_KEY=            # optional
LANGSMITH_TRACING=false       # set to true once you add a LangSmith key
```

`.env` is git-ignored, so your keys stay on your machine. Every variable is listed and explained in [.env.example](.env.example).

| Variable | Required | Used by |
| --- | --- | --- |
| `COHERE_API_KEY` | ✅ | All agents (chat + embeddings) |
| `LANGSMITH_API_KEY` | optional | Tracing, LangGraph Studio |
| `LANGSMITH_TRACING` | optional | Turns LangSmith tracing on or off |
| `CHART_BASE_URL` | optional | Base URL for chart images in Studio. Defaults to `http://127.0.0.1:2024` |
| `ANTHROPIC_API_KEY`, `GROQ_API_KEY` | no | Only the experimental scratch scripts |

---

## Running the agents

### Option A: LangGraph Studio (recommended)

From the repo root:

```bash
langgraph dev
```

This reads [langgraph.json](langgraph.json), loads `.env`, starts a local API server on `http://127.0.0.1:2024` and opens LangGraph Studio in your browser. Pick a graph from the drop-down:

- **`data_analyst`**: try *"Analyze /Walmart_Sales.csv and chart the top 5 stores by total weekly sales."* Charts appear inline in the chat.
- **`rag_agent`**: try *"How do I stream intermediate tool results from a subagent?"*

> **First start is slow.** On first start, the RAG agent downloads the LangChain docs and embeds them. On a trial key this takes a few minutes because embedding is paced to stay under Cohere's rate limit. The result is cached in `rag_agent/docs_index.json`, so later starts are fast. To rebuild the index, delete that file.

If you change the port (`langgraph dev --port 8123`), set `CHART_BASE_URL=http://127.0.0.1:8123` in `.env` so chart images still load.

### Option B: Command line

Each agent can also run a single built-in example query from the terminal:

```bash
# Data Analyst: analyses a small sample sales.csv, draws a chart and writes report.md
python deep_agent_from_scratch.py

# Docs RAG Agent: answers the example question about subagent streaming
python rag_agent/rag_agent.py

# Sandbox coding agent: creates a small package in agent_workspace/ and runs pytest
python Sandbox.py
```

Output files are written to `analysis_workspace/` (Data Analyst) and `agent_workspace/` (Sandbox).

---

## Project layout

```
.
├── deep_agent_from_scratch.py   # Data Analyst agent (graph: data_analyst)
├── chart_server.py              # Extra HTTP route: serves chart PNGs to Studio
├── rag_agent/
│   └── rag_agent.py             # Docs RAG agent (graph: rag_agent)
├── Sandbox.py                   # Sandbox coding agent (CLI only)
├── skills/
│   └── pandas-patterns/SKILL.md # Skill loaded by the Data Analyst on demand
├── analysis_workspace/
│   └── Walmart_Sales.csv        # Dataset for the Data Analyst (other output is git-ignored)
├── langgraph.json               # Graphs + HTTP app + env file for `langgraph dev`
├── requirements.txt
├── .env.example                 # Template for your .env
├── README.md
└── ARCHITECTURE.md
```

Scratch files from the labs (`Test-agent.py`, `Data Analyis Agent using Langchain.py`, `Data Analysis Agent using Langchain.txt`) are early experiments. They are not part of the runnable project and are not maintained.

### What's not committed

These files are created locally and listed in [.gitignore](.gitignore):

- `.env`: your API keys
- `.langgraph_api/`: local checkpoints and chat history from `langgraph dev`
- `rag_agent/docs_index.json`: the RAG vector index, rebuilt on first run
- `agent_workspace/` and everything in `analysis_workspace/` except `Walmart_Sales.csv`: agent output

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `KeyError` / auth error mentioning Cohere | `COHERE_API_KEY` is missing from `.env`, or `.env` isn't in the repo root. |
| `429 Too Many Requests` from Cohere | Trial limits are 20 chat requests/min and 100k embedding tokens/min. Wait a minute. The built-in limiters keep a single agent under these limits, but running several agents at once can still exceed them. |
| Charts show as broken images in Studio | The dev server isn't running on the URL in `CHART_BASE_URL`. Match it to your `langgraph dev` host and port. |
| `langgraph dev` fails with a missing `.env` | Run `cp .env.example .env`. `langgraph.json` expects the file to exist. |
| RAG answers seem out of date | Delete `rag_agent/docs_index.json` and restart to re-index the docs. |
