"""Offline checks that the sample skills are valid and discoverable.

These don't call the model. Run with:  python -m pytest -q
For a live end-to-end run against Cohere, use:  python skills_agent.py
"""

import re

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware, _list_skills
from langchain_core.messages import AIMessage

import skills_agent
from skills_agent import HERE, SKILLS_DIR, files_read, skill_files

EXPECTED = {"haiku-poet", "release-notes", "sql-review"}


def frontmatter(path):
	match = re.match(r"^---\n(.*?)\n---\n", path.read_text(), re.S)
	assert match, f"{path} has no YAML frontmatter"
	return dict(line.split(":", 1) for line in match.group(1).splitlines() if re.match(r"^\w[\w-]*:", line))


@pytest.mark.parametrize("skill_dir", sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir()), ids=lambda p: p.name)
def test_skill_follows_spec(skill_dir):
	meta = frontmatter(skill_dir / "SKILL.md")
	name, description = meta["name"].strip(), meta["description"].strip()
	assert name == skill_dir.name  # name must match its folder
	assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and len(name) <= 64
	assert 0 < len(description) <= 1024
	assert len((skill_dir / "SKILL.md").read_text().splitlines()) < 500


def test_supporting_files_referenced_by_skills_exist():
	for skill_md in SKILLS_DIR.glob("*/SKILL.md"):
		for ref in re.findall(r"`(/skills/[^`]+)`", skill_md.read_text()):
			assert (HERE / ref.lstrip("/")).is_file(), f"{skill_md} points to missing {ref}"


def test_filesystem_backend_discovers_all_skills():
	backend = FilesystemBackend(root_dir=HERE, virtual_mode=True)
	skills = {s["name"]: s for s in _list_skills(backend, "/skills/")}
	assert set(skills) == EXPECTED
	assert skills["release-notes"]["metadata"] == {"author": "homework3", "version": "1.0"}


def test_state_backend_files_cover_every_skill_file():
	files = skill_files()
	assert "/skills/release-notes/assets/template.md" in files
	assert "/skills/sql-review/references/checklist.md" in files
	assert {p.split("/")[2] for p in files} == EXPECTED


def test_skills_listed_in_system_prompt():
	backend = FilesystemBackend(root_dir=HERE, virtual_mode=True)
	middleware = SkillsMiddleware(backend=backend, sources=skills_agent.SKILL_SOURCES)
	prompt = middleware._format_skills_list(_list_skills(backend, "/skills/"))
	for name in EXPECTED:
		assert name in prompt and f"/skills/{name}/SKILL.md" in prompt


def test_files_read_extracts_read_file_paths():
	messages = [
		AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": "/skills/haiku-poet/SKILL.md"}, "id": "1"}]),
		AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/x.md", "content": ""}, "id": "2"}]),
	]
	assert files_read(messages) == ["/skills/haiku-poet/SKILL.md"]


def test_system_prompt_reaches_model_as_plain_string_with_skills():
	"""Cohere ignores block-list system prompts, so the skills list must arrive as a string."""
	from deepagents import create_deep_agent
	from langchain.agents.middleware import AgentMiddleware
	from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

	class FakeModel(GenericFakeChatModel):
		def bind_tools(self, *args, **kwargs):
			return self

	seen = []

	class Spy(AgentMiddleware):
		def wrap_model_call(self, request, handler):
			seen.append(request.system_message.content)
			return handler(request)

	agent = create_deep_agent(
		model=FakeModel(messages=iter([AIMessage("ok")])),
		backend=FilesystemBackend(root_dir=HERE, virtual_mode=True),
		skills=skills_agent.SKILL_SOURCES,
		middleware=[skills_agent.FlattenSystemPromptMiddleware(), Spy()],
	)
	agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
	assert isinstance(seen[0], str)
	assert all(f"/skills/{name}/SKILL.md" in seen[0] for name in EXPECTED)
