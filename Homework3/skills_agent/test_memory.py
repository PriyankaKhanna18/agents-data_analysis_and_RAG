"""Offline checks for memory_agent's store layout and seeding.

These don't call the model. Run with:  python -m pytest -q
For a live run against Cohere, use:  python memory_agent.py
"""

from langgraph.store.memory import InMemoryStore

import memory_agent
from memory_agent import MEMORY_PATH, read_memory, seed_store, store_location


def test_store_location_strips_route_prefix():
	assert store_location(MEMORY_PATH) == (("memory-agent", "memories"), "/AGENTS.md")
	assert store_location("/skills/langgraph-docs/SKILL.md") == (("memory-agent", "skills"), "/langgraph-docs/SKILL.md")


def test_seed_store_does_not_overwrite_learned_memory():
	store = InMemoryStore()
	seed_store(store)
	assert "Keep responses concise" in read_memory(store)
	namespace, key = store_location(MEMORY_PATH)
	store.put(namespace, key, memory_agent.create_file_data("- Prefers detailed explanations\n"))
	seed_store(store)
	assert read_memory(store) == "- Prefers detailed explanations\n"


def test_backend_reads_seeded_memory_through_route():
	store = InMemoryStore()
	seed_store(store)
	backend = memory_agent.build_backend()
	backend.routes["/memories/"]._store = store  # outside a graph run there's no store context
	(response,) = backend.download_files([MEMORY_PATH])
	assert response.error is None
	assert b"Keep responses concise" in response.content
