"""Deep agent that delegates to sub-agents, following
https://docs.langchain.com/oss/python/deepagents/subagents

The main agent plans the work and hands focused tasks to two sub-agents
through the built-in `task` tool:
- research-agent: looks things up on Wikipedia and returns a short summary
  with sources. Its raw search results stay in its own context, so they don't
  bloat the main agent's conversation.
- critique-agent: reviews a draft answer and lists concrete fixes. It has no
  tools.

Each sub-agent starts with a fresh context ("isolated" mode) and only sees
the task description the main agent writes for it.

Usage:
    python sub_agent.py                         # run the sample prompts
    python sub_agent.py --prompt "Compare Rust and Go for CLI tools"

For a UI, run `langgraph dev` from the repo root and open `sub_agent` in
LangGraph Studio. Expand the `task` tool calls to see each delegation.
"""

import argparse
import sys
from pathlib import Path

import requests
from deepagents import create_deep_agent
from langchain_core.tools import tool

# `langgraph dev` loads this file by path, so sibling folders aren't on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent / "skills_agent"))
from skills_agent import FlattenSystemPromptMiddleware, build_model  # noqa: E402

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Wikipedia rejects requests without a descriptive User-Agent.
HEADERS = {"User-Agent": "deepagents-subagent-homework/1.0 (educational)"}


@tool
def wikipedia_search(query: str, max_results: int = 3) -> str:
	"""Search Wikipedia and return the top article titles with snippets."""
	params = {"action": "query", "list": "search", "srsearch": query, "srlimit": max_results, "format": "json"}
	try:
		hits = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15).json()["query"]["search"]
	except Exception as e:
		return f"Search failed: {e}"
	if not hits:
		return f"No Wikipedia results for {query!r}."
	lines = []
	for h in hits:
		snippet = h["snippet"].replace('<span class="searchmatch">', "").replace("</span>", "")
		lines.append(f"- {h['title']}: {snippet}")
	return "\n".join(lines)


@tool
def wikipedia_summary(title: str) -> str:
	"""Return the lead summary and URL of a Wikipedia article, given its exact title."""
	try:
		resp = requests.get(WIKI_SUMMARY + title.replace(" ", "_"), headers=HEADERS, timeout=15)
		resp.raise_for_status()
		data = resp.json()
	except Exception as e:
		return f"Could not fetch {title!r}: {e}"
	url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
	return f"{data.get('title', title)}\n{url}\n\n{data.get('extract', '')}"


RESEARCH_PROMPT = (
	"You are a research assistant. Use wikipedia_search to find relevant "
	"articles, then wikipedia_summary to read the most relevant ones (at most "
	"3). If a search returns nothing useful, retry with a more specific query "
	"(e.g. add 'programming language' or the full name) before giving up. Reply with a concise summary of the facts that answer the question "
	"(under 200 words), followed by a 'Sources:' list of the article URLs you "
	"used. Do not include raw search output."
)

CRITIQUE_PROMPT = (
	"You are a strict editor. You are given a draft answer and the question it "
	"answers. List up to 5 concrete problems (missing facts, unclear wording, "
	"unsupported claims, structure), each with a suggested fix. If the draft is "
	"good, say so in one line. Do not rewrite the whole answer."
)

MAIN_PROMPT = (
	"You are a lead researcher who coordinates sub-agents. For any question "
	"that needs facts, delegate the lookup to research-agent with the task "
	"tool; give it one focused question per call, and call it several times in "
	"parallel when the question has separate parts (e.g. comparisons). Write "
	"your answer from what it returns and end it with a 'Sources:' list of "
	"every URL the sub-agents returned. When the user asks "
	"for a reviewed or polished answer, send your draft to critique-agent and "
	"apply its feedback before replying. Answer simple chit-chat or arithmetic "
	"yourself without delegating."
)


def build_agent():
	# One model instance, so the main agent and sub-agents share Cohere's rate limiter.
	model = build_model()
	# Sub-agents don't inherit the parent's custom middleware, so each needs
	# its own copy of the Cohere system-prompt fix.
	subagents = [
		{
			"name": "research-agent",
			"description": (
				"Looks up facts on Wikipedia and returns a short sourced summary. "
				"Give it one focused research question per call."
			),
			"system_prompt": RESEARCH_PROMPT,
			"tools": [wikipedia_search, wikipedia_summary],
			"model": model,
			"middleware": [FlattenSystemPromptMiddleware()],
		},
		{
			"name": "critique-agent",
			"description": (
				"Reviews a draft answer and returns a list of concrete improvements. "
				"Pass it the original question and the full draft."
			),
			"system_prompt": CRITIQUE_PROMPT,
			"tools": [],
			"model": model,
			"middleware": [FlattenSystemPromptMiddleware()],
		},
	]
	return create_deep_agent(
		model=model,
		system_prompt=MAIN_PROMPT,
		subagents=subagents,
		middleware=[FlattenSystemPromptMiddleware()],
	)


SAMPLE_PROMPTS = [
	"Who designed the Eiffel Tower and when was it completed?",
	"Compare the Python and Rust programming languages: who created each and when?",
]


def delegations(messages):
	"""(subagent_type, description) for every task() call, in order."""
	return [
		(call["args"].get("subagent_type"), call["args"].get("description", "")[:80])
		for message in messages
		for call in getattr(message, "tool_calls", None) or []
		if call["name"] == "task"
	]


# Module-level graph for `langgraph dev` / Studio.
agent = build_agent()


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--prompt", help="run one prompt instead of the samples")
	args = parser.parse_args()

	for i, prompt in enumerate([args.prompt] if args.prompt else SAMPLE_PROMPTS):
		print(f"\n{'=' * 70}\nUSER: {prompt}\n{'-' * 70}")
		result = agent.invoke(
			{"messages": [{"role": "user", "content": prompt}]},
			config={"configurable": {"thread_id": str(i)}},
		)
		print(result["messages"][-1].text)
		print(f"{'-' * 70}\nDelegations:")
		for name, desc in delegations(result["messages"]) or [("(none)", "")]:
			print(f"  {name}: {desc}")


if __name__ == "__main__":
	main()
