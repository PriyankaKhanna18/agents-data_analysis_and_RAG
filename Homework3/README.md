# Homework 3: Deep Agents

Four agents built with [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
(deepagents 0.7.21), all using Cohere `command-a-03-2025`.

| Agent | Graph name | File | Shows |
|---|---|---|---|
| Skills agent | `skills_agent` | [skills_agent/skills_agent.py](skills_agent/skills_agent.py) | Loading skills from `SKILL.md` files on demand |
| Memory agent | `memory_agent` | [skills_agent/memory_agent.py](skills_agent/memory_agent.py) | Long-term memory that carries across threads |
| Sub-agent | `sub_agent` | [sub_agent.py](sub_agent.py) | Delegating work to specialised sub-agents |
| Grading rubric | `grading_rubric` | [grading_rubric.py](grading_rubric.py) | A grader that checks answers against a rubric and makes the agent revise |

## Setup

From the repo root:

```bash
pip install -r requirements.txt
echo "COHERE_API_KEY=your-key" >> .env
```

## Running

**LangGraph Studio (UI):** all four agents are registered in the repo's
[`langgraph.json`](../langgraph.json).

```bash
# from the repo root
langgraph dev
```

Open the Studio link it prints and pick an agent from the graph dropdown.

**Command line:** each file also runs on its own. See the per-agent sections below.

**Tests (offline, no API calls):**

```bash
cd Homework3/skills_agent && python3 -m pytest -q     # skills + memory
cd Homework3 && python3 -m pytest -q test_grading_rubric.py
```

---

## 1. Skills agent

Based on the [skills docs](https://docs.langchain.com/oss/python/deepagents/skills).
The agent starts out seeing only each skill's name and description. When a
request matches one, it reads that skill's full `SKILL.md`, plus any file the
skill points to.

| Skill | What it tests |
|---|---|
| `haiku-poet` | Plain instructions with a fixed output format |
| `release-notes` | Loading a template from `assets/` and saving output with `write_file` |
| `sql-review` | Loading a reference checklist from `references/` |

```bash
cd Homework3/skills_agent
python3 skills_agent.py                                  # 3 sample prompts
python3 skills_agent.py --prompt "Write a haiku about rain"
```

**Try:**
- `Write a haiku about rain`: should read `haiku-poet/SKILL.md`.
- `Review this SQL: SELECT * FROM users WHERE name = '" + name + "'`: should read `sql-review` and its checklist.
- `Write release notes for v2.0: added dark mode, fixed login bug`: should use the `release-notes` template.

To add a skill, create `skills/<name>/SKILL.md` with `name` and `description`
in the YAML frontmatter. More detail in [skills_agent/README.md](skills_agent/README.md).

## 2. Memory agent

Based on the [memory example](https://docs.langchain.com/oss/python/deepagents/memory#full-example-seed-memory-and-invoke).
`/memories/AGENTS.md` is stored in the LangGraph store, not in the thread, so
anything the agent saves there is in its system prompt in every later thread.

```bash
cd Homework3/skills_agent
python3 memory_agent.py          # demo: thread 1 learns a preference, thread 2 uses it
python3 memory_agent.py --chat   # /new = new thread, /memory = show AGENTS.md, /quit
```

**Try (start a new thread in Studio for each second step):**
1. `I prefer detailed explanations with examples. Please remember that.`
   Then, in a new thread: `Explain how a hash map works.` The answer should be long, with examples.
2. `My name is Priyanka and I mostly code in Python.`
   Then, in a new thread: `What's my name and what language should you use for examples?`
3. In any thread: `Show me the exact contents of your memory file.`

Memory is loaded once at the start of a thread, so a new preference only
takes effect in the next thread.

## 3. Sub-agent

Based on the [subagents docs](https://docs.langchain.com/oss/python/deepagents/subagents).
The main agent delegates through the built-in `task` tool to:

- **research-agent**: searches Wikipedia (no API key needed) and returns a short summary with sources.
- **critique-agent**: has no tools; reviews a draft and lists concrete fixes.

Each sub-agent starts with a fresh context, so raw search results don't fill
up the main agent's conversation.

```bash
cd Homework3
python3 sub_agent.py                                     # sample prompts
python3 sub_agent.py --prompt "Who painted the Mona Lisa?"
```

**Try** (expand the `task` tool calls in Studio to see each delegation):
- `Who designed the Eiffel Tower and when was it completed?`: research-agent; Gustave Eiffel, 1889.
- `Compare Python and Rust: who created each and when?`: research-agent once per language.
- `Give me a reviewed, polished paragraph on what photosynthesis is.`: research-agent, then critique-agent.
- `What is 12 times 9?`: answers 108 with no delegation.

## 4. Grading rubric

Based on the [rubric docs](https://docs.langchain.com/oss/python/deepagents/rubric).
When the agent finishes, a grader checks its answer against a rubric. If any
criterion fails, the grader's feedback goes back to the agent, which revises.
This repeats for up to 3 grading passes. The grader has a `run_test_suite`
tool, so it runs real tests on generated code.

In Studio's chat view, put the rubric at the end of the message after a
`RUBRIC:` line. The answer ends with a **Grader report** listing each pass
and criterion. With no rubric, the agent behaves like a plain agent.

```bash
cd Homework3
python3 grading_rubric.py                                # the doc's examples
python3 grading_rubric.py --prompt "Write a limerick about cats" \
    --rubric "- Five lines\n- AABBA rhyme scheme"
```

**Try:**

Shows a revision almost every time, since the first draft usually fails a test it can't see:
```
Write a Python function find_duplicates(lst) that returns a list of all elements
that appear more than once in the input list, in the order they first appear.

RUBRIC:
- All tests pass in run_test_suite
- The function is named find_duplicates and accepts a single list argument
```

Usually passes on the first try:
```
Write a haiku about spring.

RUBRIC:
- The poem has three lines
- Lines follow a 5-7-5 syllable pattern
- The theme is spring
```

No rubric, so no grading: `What is 12 times 9?`

---

## Notes on Cohere

- Deep Agents sends the system prompt as a list of blocks, which Cohere
  silently ignores. `FlattenSystemPromptMiddleware` (in `skills_agent.py`)
  joins it into one string, and every agent and sub-agent uses it.
- Cohere rejects `tool_choice="any"`, which the rubric grader uses.
  `GraderChatCohere` sends `"REQUIRED"` instead.
- All agents share one rate limiter to stay under the Cohere trial limits.

## Known limitations

- The rubric grader is lenient on borderline cases (for example, it has passed
  an 8-syllable haiku line).
- The sub-agent's final answer sometimes leaves out sources.
- When saving memory, Cohere sometimes makes a bad `edit_file` call before
  retrying successfully.
