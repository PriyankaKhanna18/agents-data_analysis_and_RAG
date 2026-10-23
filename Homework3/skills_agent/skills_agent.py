"""Deep agent that loads skills, following
https://docs.langchain.com/oss/python/deepagents/skills

Skills live in ./skills/<name>/SKILL.md. The agent only sees each skill's name
and description at startup; it reads the full SKILL.md (and any supporting
files) with read_file when a request matches (progressive disclosure).

Two ways to load the same skills are shown:
- "filesystem": FilesystemBackend reads ./skills/ straight from disk.
- "state": StateBackend keeps files in agent state, so the skill files are
  passed in with each invoke() via create_file_data().

Usage:
    python skills_agent.py                       # run the sample prompts
    python skills_agent.py --backend state       # same, skills loaded via state
    python skills_agent.py --prompt "Write a haiku about rain"
"""

import argparse
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.utils import create_file_data
from deepagents.middleware.skills import _list_skills
from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

HERE = Path(__file__).resolve().parent
SKILLS_DIR = HERE / "skills"
load_dotenv(HERE.parents[1] / ".env")  # COHERE_API_KEY lives in the repo root .env

# Skill sources are directories that contain skill folders, not single skills.
SKILL_SOURCES = ["/skills/"]

SYSTEM_PROMPT = (
	"You are a helpful assistant with a library of skills. When a request "
	"matches a skill, read its SKILL.md before answering and follow it exactly. "
	"Do the work yourself with your tools."
)

SAMPLE_PROMPTS = {
	"haiku-poet": "Write a haiku about autumn leaves in Seattle.",
	"release-notes": (
		"Write release notes for v2.3.0 from these changes: added dark mode; "
		"CSV export is 3x faster; fixed crash when uploading empty files; "
		"bumped pytest to 9.1; refactored the auth module."
	),
	"sql-review": (
		"Review this query: \"SELECT * FROM orders WHERE customer = '\" + name + "
		"\"' AND YEAR(created_at) = 2024 AND shipped_at = NULL\""
	),
}


def build_model():
	# Cohere trial keys allow 20 chat requests/minute; stay under it.
	rate_limiter = InMemoryRateLimiter(requests_per_second=15 / 60, check_every_n_seconds=1)
	# disable_streaming: langchain-cohere's streaming path crashes when Cohere
	# omits billed token counts (same workaround as deep_agent_from_scratch.py).
	return ChatCohere(model="command-a-03-2025", rate_limiter=rate_limiter, disable_streaming=True)


class FlattenSystemPromptMiddleware(AgentMiddleware):
	"""Sends the system prompt to the model as one plain string.

	Deep Agents builds the system prompt as a list of text blocks (base prompt,
	filesystem tools, skills list, ...). langchain-cohere passes that list
	through and Cohere silently ignores it, so without this the model never
	sees the skills list. User middleware runs innermost, after every built-in
	middleware has added its section.
	"""

	def _flatten(self, request):
		if request.system_message is None or isinstance(request.system_message.content, str):
			return request
		return request.override(system_message=SystemMessage(request.system_message.text))

	def wrap_model_call(self, request, handler):
		return handler(self._flatten(request))

	async def awrap_model_call(self, request, handler):
		return await handler(self._flatten(request))


def build_agent(backend_kind="filesystem"):
	if backend_kind == "filesystem":
		# virtual_mode maps "/" to HERE, so "/skills/" is ./skills on disk and
		# the agent can't reach files outside this folder.
		backend = FilesystemBackend(root_dir=HERE, virtual_mode=True)
	elif backend_kind == "state":
		backend = StateBackend()
	else:
		raise ValueError(f"unknown backend: {backend_kind}")
	return create_deep_agent(
		model=build_model(),
		system_prompt=SYSTEM_PROMPT,
		backend=backend,
		skills=SKILL_SOURCES,
		middleware=[FlattenSystemPromptMiddleware()],
	)


def skill_files():
	"""Every file under ./skills as {"/skills/...": FileData}, for StateBackend."""
	return {
		f"/skills/{path.relative_to(SKILLS_DIR).as_posix()}": create_file_data(path.read_text())
		for path in sorted(SKILLS_DIR.rglob("*"))
		if path.is_file()
	}


def files_read(messages):
	"""Paths the agent opened with read_file, in order."""
	paths = []
	for message in messages:
		for call in getattr(message, "tool_calls", None) or []:
			if call["name"] == "read_file":
				args = call["args"]
				paths.append(args.get("file_path") or args.get("path"))
	return paths


def run(agent, prompt, backend_kind="filesystem", thread_id="1"):
	state = {"messages": [{"role": "user", "content": prompt}]}
	if backend_kind == "state":
		state["files"] = skill_files()
	return agent.invoke(state, config={"configurable": {"thread_id": thread_id}})


# Module-level graph for `langgraph dev` / Studio.
agent = build_agent()


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--backend", choices=["filesystem", "state"], default="filesystem")
	parser.add_argument("--prompt", help="run one prompt instead of the samples")
	args = parser.parse_args()

	runner = agent if args.backend == "filesystem" else build_agent(args.backend)
	prompts = {"custom": args.prompt} if args.prompt else SAMPLE_PROMPTS
	# skills_metadata is OmitFromOutput, so invoke() results never include it;
	# list the skills straight from disk instead.
	discovered = _list_skills(FilesystemBackend(root_dir=HERE, virtual_mode=True), SKILL_SOURCES[0])
	print(f"Skills discovered: {[s['name'] for s in discovered]}")
	for i, (expected, prompt) in enumerate(prompts.items()):
		print(f"\n{'=' * 70}\nUSER: {prompt}\n{'-' * 70}")
		result = run(runner, prompt, args.backend, thread_id=str(i))
		print(result["messages"][-1].text)
		print(f"{'-' * 70}\nFiles read: {files_read(result['messages'])}")
		if expected != "custom":
			used = any(f"/skills/{expected}/SKILL.md" in (p or "") for p in files_read(result["messages"]))
			print(f"Expected skill '{expected}' loaded: {'YES' if used else 'NO'}")


if __name__ == "__main__":
	main()
