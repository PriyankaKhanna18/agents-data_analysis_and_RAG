"""Deep agent with long-term memory, following
https://docs.langchain.com/oss/python/deepagents/memory#full-example-seed-memory-and-invoke

/memories/AGENTS.md and /skills/ are routed to a LangGraph store (StoreBackend),
so they survive across threads. Everything else lives in per-thread state
(StateBackend). At the start of each thread the agent loads AGENTS.md into its
system prompt; when it learns something worth keeping it updates the file with
edit_file, and the next thread starts with that knowledge.

Usage:
    python memory_agent.py              # the doc's two-thread demo
    python memory_agent.py --chat       # interactive chat (/new, /memory, /quit)

For a UI, run `langgraph dev` from the repo root and open `memory_agent` in
LangGraph Studio. Start a new thread to see memory carry over.
"""

import argparse
import sys
from pathlib import Path

import requests
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langchain_core.utils.uuid import uuid7
from langgraph.store.memory import InMemoryStore

# `langgraph dev` loads this file by path, so its folder isn't on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from skills_agent import FlattenSystemPromptMiddleware, build_model  # noqa: E402

AGENT_NAME = "memory-agent"
MEMORY_PATH = "/memories/AGENTS.md"

# Route prefix -> store namespace. CompositeBackend strips the prefix before
# calling the routed backend, so /memories/AGENTS.md is stored under key
# /AGENTS.md. (The doc example seeds the full path as the key, which this
# deepagents version never finds.) Separate namespaces keep the two routes'
# keys from colliding.
ROUTES = {"/memories/": (AGENT_NAME, "memories"), "/skills/": (AGENT_NAME, "skills")}

# Seed files, written to the store only if they aren't there yet.
SEED_FILES = {
	MEMORY_PATH: """## User Preferences
- Keep responses concise
- Use code examples where possible
""",
	"/skills/langgraph-docs/SKILL.md": """---
name: langgraph-docs
description: Fetch relevant LangGraph documentation to provide accurate guidance. Use only for questions about LangChain, LangGraph, or Deep Agents.
---

# langgraph-docs

Use the fetch_url tool to read https://docs.langchain.com/llms.txt, then fetch relevant pages.
""",
}

SYSTEM_PROMPT = (
	"You are a helpful assistant with long-term memory. Your memory file is "
	f"{MEMORY_PATH}; its contents are shown to you at the start of every "
	"conversation. When the user states a preference, a fact about themselves, "
	f"or asks you to remember something, update {MEMORY_PATH} with edit_file "
	"right away. The file is a markdown list under the heading "
	"'## User Preferences'. If the new item conflicts with an existing one "
	"(e.g. 'detailed' vs 'concise'), replace that existing line: use it exactly "
	"as old_string and the new item as new_string. Otherwise, to add an item, "
	"call edit_file with "
	"old_string='## User Preferences' and new_string='## User Preferences\\n- <new item>'. "
	"To remove an item, use its exact line as old_string. If edit_file fails, call read_file on the memory file, then "
	"retry with a string copied exactly from it; never give up after one failure. "
	"Always follow what your memory says; if two items conflict, the one "
	"higher in the list is newer and wins. "
	"Answer general-knowledge questions directly from what you know; only use "
	"a skill when the request is about that skill's topic."
)


@tool
def fetch_url(url: str) -> str:
	"""Fetch a web page and return its text (first 10,000 characters)."""
	response = requests.get(url, timeout=20)
	response.raise_for_status()
	return response.text[:10_000]


def store_location(path):
	"""Virtual path -> (namespace, key) in the store, the way CompositeBackend routes it."""
	for prefix, namespace in ROUTES.items():
		if path.startswith(prefix):
			return namespace, "/" + path[len(prefix):]
	raise ValueError(f"{path} is not under a store route")


def seed_store(store):
	for path, content in SEED_FILES.items():
		namespace, key = store_location(path)
		if store.get(namespace, key) is None:
			store.put(namespace, key, create_file_data(content))


async def aseed_store(store):
	for path, content in SEED_FILES.items():
		namespace, key = store_location(path)
		if await store.aget(namespace, key) is None:
			await store.aput(namespace, key, create_file_data(content))


class SeedMemoryMiddleware(AgentMiddleware):
	"""Seeds the store on the first run.

	`langgraph dev` / Studio supplies its own store, so it can't be seeded at
	import time like the InMemoryStore in main(). User middleware runs before
	MemoryMiddleware, so the seed is in place before AGENTS.md is loaded.
	"""

	def before_agent(self, state, runtime):
		if runtime.store is not None:
			seed_store(runtime.store)

	async def abefore_agent(self, state, runtime):
		# The dev server runs graphs async and rejects sync store calls.
		if runtime.store is not None:
			await aseed_store(runtime.store)


def build_backend():
	# Memory and skills are shared by every thread for this agent; the rest of
	# the filesystem is scratch space that lives only in the thread's state.
	return CompositeBackend(
		default=StateBackend(),
		routes={
			prefix: StoreBackend(namespace=lambda rt, ns=namespace: ns)
			for prefix, namespace in ROUTES.items()
		},
	)


def build_agent(store=None):
	"""Pass a store for local runs; leave it None under `langgraph dev`, which provides one."""
	return create_deep_agent(
		model=build_model(),
		system_prompt=SYSTEM_PROMPT,
		tools=[fetch_url],
		memory=[MEMORY_PATH],
		skills=["/skills/"],
		backend=build_backend(),
		store=store,
		middleware=[SeedMemoryMiddleware(), FlattenSystemPromptMiddleware()],
	)


def read_memory(store):
	item = store.get(*store_location(MEMORY_PATH))
	if item is None:
		return "(empty)"
	content = item.value["content"]
	return content if isinstance(content, str) else "\n".join(content)


def new_config():
	return {"configurable": {"thread_id": str(uuid7())}}


def ask(agent, prompt, config):
	result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
	return result["messages"][-1].text


# Module-level graph for `langgraph dev` / Studio.
agent = build_agent()


def demo(local_agent, store):
	turns = [
		"I prefer detailed explanations. Remember that.",
		"Explain how transformers work.",
	]
	print(f"Memory before:\n{read_memory(store)}")
	for i, prompt in enumerate(turns, 1):
		print(f"\n{'=' * 70}\nTHREAD {i} | USER: {prompt}\n{'-' * 70}")
		print(ask(local_agent, prompt, new_config()))
		print(f"{'-' * 70}\nMemory after thread {i}:\n{read_memory(store)}")


def chat(local_agent, store):
	print("Commands: /new = new thread (memory carries over), /memory = show AGENTS.md, /quit")
	config = new_config()
	while True:
		try:
			prompt = input("\nyou> ").strip()
		except (EOFError, KeyboardInterrupt):
			break
		if prompt in ("/quit", "/exit"):
			break
		if prompt == "/new":
			config = new_config()
			print("Started a new thread.")
		elif prompt == "/memory":
			print(read_memory(store))
		elif prompt:
			print(f"\nagent> {ask(local_agent, prompt, config)}")


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--chat", action="store_true", help="interactive chat instead of the demo")
	args = parser.parse_args()

	store = InMemoryStore()  # use the platform store when deploying to LangSmith
	seed_store(store)
	local_agent = build_agent(store)
	(chat if args.chat else demo)(local_agent, store)


if __name__ == "__main__":
	main()
