# Skills agent

A Deep Agent that loads [skills](https://docs.langchain.com/oss/python/deepagents/skills)
from `./skills/`, using Cohere `command-a-03-2025` like the rest of this repo.

## Sample skills

| Skill | Tests | Supporting file |
|---|---|---|
| `haiku-poet` | Plain SKILL.md instructions plus a fixed output format | none |
| `release-notes` | Loading a template from `assets/` and saving output with `write_file` | `assets/template.md` |
| `sql-review` | Loading a reference doc on demand from `references/` | `references/checklist.md` |

The agent sees only each skill's name and description at first. It reads the
full `SKILL.md`, then any supporting file it points to, when a request
matches (progressive disclosure).

## Run

```bash
# from this folder; needs COHERE_API_KEY in the repo-root .env
python3 -m pytest -q                         # offline checks, no API calls
python3 skills_agent.py                      # 3 sample prompts, FilesystemBackend
python3 skills_agent.py --backend state      # same skills, loaded via invoke(files=...)
python3 skills_agent.py --prompt "Write a haiku about rain"
```

Each run prints the files the agent read and whether it loaded the expected
skill. It's also registered as `skills_agent` in the repo's `langgraph.json`
for `langgraph dev` / Studio.

## Adding a skill

Create `skills/<name>/SKILL.md` with `name` (lowercase and hyphens, the same
as the folder) and `description` (say when to use it) in the YAML
frontmatter. New skills are picked up on the next run.

## Cohere gotcha

Deep Agents builds the system prompt as a list of text blocks. langchain-cohere
passes that list through, and Cohere silently ignores it, so the model never
saw the skills list and answered everything from its own knowledge.
`FlattenSystemPromptMiddleware` turns the prompt back into one string before
each model call.

# Memory agent

`memory_agent.py` follows the Deep Agents
[memory example](https://docs.langchain.com/oss/python/deepagents/memory#full-example-seed-memory-and-invoke):
`/memories/AGENTS.md` and `/skills/` are routed to a LangGraph store, so what
the agent learns in one thread is in its system prompt in the next.

```bash
python3 memory_agent.py          # the doc's demo: thread 1 learns a preference, thread 2 applies it
python3 memory_agent.py --chat   # REPL: /new starts a new thread, /memory prints AGENTS.md, /quit
```

**UI:** run `langgraph dev` from the repo root and pick `memory_agent` in
LangGraph Studio. Tell it a preference, start a new thread (+), and ask
something. The memory lives in the dev server's store under the
`memory-agent/memories` namespace.

Differences from the doc example (deepagents 0.7.21):

- `StateBackend()` / `StoreBackend(namespace=...)` take no `rt` argument.
- `CompositeBackend` strips the route prefix, so `/memories/AGENTS.md` is
  stored under key `/AGENTS.md`. Seeding the full path, as the doc does,
  leaves the agent with no memory. Each route gets its own namespace.
- Seeding happens in `SeedMemoryMiddleware`, because `langgraph dev` provides
  its own store and that store can't be seeded at import time.
