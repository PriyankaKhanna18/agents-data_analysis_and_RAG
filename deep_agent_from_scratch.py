"""Data analysis agent built step by step from Deep Agents middleware.

Follows https://docs.langchain.com/oss/python/langchain/deep-agent-from-scratch
with these changes:
- Model: Cohere instead of Gemini.
- Charts: ChartDisplayMiddleware shows PNGs the agent saves in the chat.
- Backend: a local workspace folder instead of a LangSmith sandbox (sandboxes
  aren't enabled for this LangSmith org). There is no isolation: the agent's
  shell commands run directly on this machine.
"""

import asyncio
import csv
import io
import os
import re
import time
from pathlib import Path
from typing import NotRequired
from urllib.parse import quote

from deepagents import SubAgent
from deepagents.backends import LocalShellBackend
from deepagents.middleware import (
	FilesystemMiddleware,
	SkillsMiddleware,
	SubAgentMiddleware,
	SummarizationMiddleware,
)
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, TodoListMiddleware
from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()  # reads COHERE_API_KEY from .env

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE / "analysis_workspace"
SKILLS_DIR = HERE / "skills"
# Where `langgraph dev` serves chart_server.py's /charts/ route.
CHART_BASE_URL = os.environ.get("CHART_BASE_URL", "http://127.0.0.1:2024")

# File tools see the workspace as "/", while the shell runs inside it, so the
# model needs to be told how the two path styles relate.
PATH_RULES = (
	"Your workspace is a single folder. File tools see it as '/', e.g. "
	"'/sales.csv'. Shell commands already run inside that folder, so use "
	"relative paths there (e.g. 'python3 analyze.py', 'ls'). Never use "
	"absolute paths in shell commands. Do the work yourself with your tools; "
	"never just describe code for the user to run. Only report numbers, names "
	"and IDs that appear in tool output; if a script doesn't print a value you "
	"need, make it print it and run it again. Never guess. "
	"You CAN run code: the execute tool runs shell commands. To run Python, "
	"first save the script with write_file (e.g. '/top_stores.py'), then "
	"execute 'python3 top_stores.py'. If execute reports 'No such file', the "
	"script was never written: write it, don't retry other paths."
)

# Rules for any chart, since either the main agent or the visualizer may draw it.
CHART_RULES = (
	" Chart rules: treat ID columns such as Store as categories, not numbers "
	"(e.g. df['Store'].astype(str)), so a bar chart has one evenly spaced, "
	"labelled bar per item instead of bars scattered along a number line. "
	"Sort bars from largest to smallest. Show money in readable units (e.g. "
	"divide by 1e6 and label the axis 'Sales ($ millions)'). Make the script "
	"print the plotted values, and include them as a small table in your reply."
)

# A Markdown image: ![alt](target), where target may be quoted or <wrapped>.
MARKDOWN_IMAGE = re.compile(r'!\[([^\]]*)\]\(\s*<?["\']?([^)"\'<>\s]+)["\']?>?\s*\)')


class ChartState(AgentState):
	chart_turn_started: NotRequired[float]


class ChartDisplayMiddleware(AgentMiddleware):
	"""Show charts saved during a turn as images in the chat (e.g. Studio).

	Charts are PNG files on disk, which a chat UI can't load, so after each turn
	this adds a Markdown message linking each new PNG via the dev server's
	/charts/ route (chart_server.py). These messages are for the person reading
	the chat, so they're stripped from every model request.
	"""

	state_schema = ChartState

	def __init__(self, workspace: Path):
		super().__init__()
		self.workspace = workspace

	def before_agent(self, state, runtime):
		return {"chart_turn_started": time.time()}

	def _chart_url(self, chart: Path) -> str:
		name = chart.relative_to(self.workspace).as_posix()
		# ?v= busts the browser cache when a chart is regenerated under the same name.
		return f"{CHART_BASE_URL}/charts/{quote(name)}?v={int(chart.stat().st_mtime)}"

	def _fix_image_links(self, text: str, shown: set) -> str:
		"""Point the model's own Markdown images at the chart route.

		The model often embeds a chart by its file path, which a browser can't
		load. Existing PNGs get a working URL; any other image is reduced to its
		alt text so no broken-image icon is left behind.
		"""

		def fix(match):
			alt, target = match.group(1), match.group(2)
			if target.startswith(("http://", "https://")):
				return match.group(0)
			chart = (self.workspace / target.lstrip("./")).resolve()
			if chart.suffix.lower() == ".png" and chart.is_relative_to(self.workspace.resolve()) and chart.is_file():
				shown.add(chart)
				return f"![{alt}]({self._chart_url(chart)})"
			return alt

		return MARKDOWN_IMAGE.sub(fix, text)

	def after_agent(self, state, runtime):
		started = state.get("chart_turn_started")
		if started is None:
			return None
		updates, shown = [], set()
		# Messages since the user's last message belong to this turn.
		turn = []
		for message in reversed(state["messages"]):
			if isinstance(message, HumanMessage):
				break
			turn.append(message)
		for message in reversed(turn):
			if isinstance(message, AIMessage) and isinstance(message.content, str) and "![" in message.content:
				fixed = self._fix_image_links(message.content, shown)
				if fixed != message.content:
					# Same id, so this replaces the message instead of adding one.
					updates.append(message.model_copy(update={"content": fixed}))
		charts = sorted(
			(
				p for p in self.workspace.rglob("*.png")
				if p.stat().st_mtime >= started and p.resolve() not in shown
			),
			key=lambda p: p.stat().st_mtime,
		)
		if charts:
			lines = [
				f"**{c.relative_to(self.workspace).as_posix()}**\n\n"
				f"![{c.relative_to(self.workspace).as_posix()}]({self._chart_url(c)})"
				for c in charts
			]
			updates.append(AIMessage(content="\n\n".join(lines), additional_kwargs={"chart_display": True}))
		return {"messages": updates} if updates else None

	async def aafter_agent(self, state, runtime):
		# Servers like `langgraph dev` forbid blocking file I/O on the event loop.
		return await asyncio.to_thread(self.after_agent, state, runtime)

	@staticmethod
	def _without_charts(request):
		messages = [m for m in request.messages if not m.additional_kwargs.get("chart_display")]
		if len(messages) == len(request.messages):
			return request
		return request.override(messages=messages)

	def wrap_model_call(self, request, handler):
		return handler(self._without_charts(request))

	async def awrap_model_call(self, request, handler):
		return await handler(self._without_charts(request))


def build_backend():
	"""Step 2: an execution environment with file access."""
	WORKSPACE.mkdir(exist_ok=True)
	# Only pass PATH to the agent's shell so it can't read the API keys.
	return LocalShellBackend(root_dir=WORKSPACE, env={"PATH": os.environ["PATH"]})


def upload_sample_data(backend):
	rows = [
		["Date", "Product", "Units", "Revenue"],
		["2025-08-01", "Widget A", 10, 250],
		["2025-08-02", "Widget B", 5, 125],
		["2025-08-03", "Widget A", 7, 175],
		["2025-08-04", "Widget C", 3, 90],
	]
	buf = io.StringIO()
	csv.writer(buf).writerows(rows)
	backend.upload_files([("/sales.csv", buf.getvalue().encode())])


def upload_skills(backend):
	"""Step 4: copy local skill files into the workspace under /skills/."""
	skill_files = []
	for path in sorted(SKILLS_DIR.rglob("*")):
		if not path.is_file():
			continue
		rel = path.resolve().relative_to(SKILLS_DIR.resolve())
		skill_files.append((f"/skills/{rel.as_posix()}", path.read_bytes()))
	backend.upload_files(skill_files)


def build_agent(backend):
	# Step 1: the model. Cohere trial keys allow 20 chat requests/minute, and the
	# main agent and subagent share this limiter, so stay under it together.
	rate_limiter = InMemoryRateLimiter(requests_per_second=15 / 60, check_every_n_seconds=1)
	# disable_streaming: langchain-cohere 0.6.0's streaming path crashes when
	# Cohere omits billed token counts ("'NoneType' ... 'input_tokens'"), and
	# Studio streams by default. The non-streaming path handles it safely.
	model = ChatCohere(
		model="command-a-03-2025",
		rate_limiter=rate_limiter,
		disable_streaming=True,
	)

	# Step 5: a visualization subagent. Raw subagent specs get no tools of their
	# own, so give it filesystem middleware to write and run its scripts.
	visualizer: SubAgent = {
		"name": "visualizer",
		"description": "Generates charts and visualizations from data files in the workspace.",
		"system_prompt": (
			"You are a data visualization specialist. Write Python scripts using "
			"matplotlib and seaborn, run them, and save all figures as PNG files. "
			"Use plt.savefig(..., bbox_inches='tight') so labels aren't cut off. "
			"Make each script print the values it plots, and list those printed "
			"values in your final reply, since nobody reading the conversation "
			"later can look inside the image. "
			+ PATH_RULES
			+ CHART_RULES
		),
		"tools": [],
		"model": model,
		"middleware": [FilesystemMiddleware(backend=backend)],
	}

	return create_agent(
		model=model,
		tools=[],
		system_prompt="You are a data analysis assistant. " + PATH_RULES + CHART_RULES,
		middleware=[
			# First, so chart images are stripped before any other middleware
			# (e.g. summarization) sends history to the model.
			ChartDisplayMiddleware(WORKSPACE),
			FilesystemMiddleware(backend=backend),  # Step 2
			SummarizationMiddleware(model=model, backend=backend),  # Step 3
			SkillsMiddleware(backend=backend, sources=["/skills/"]),  # Step 4
			TodoListMiddleware(),  # Step 5
			SubAgentMiddleware(backend=backend, subagents=[visualizer]),  # Step 5
		],
	)


def make_agent():
	backend = build_backend()
	upload_sample_data(backend)
	upload_skills(backend)
	return build_agent(backend)


# Module-level graph that LangGraph Studio loads (see langgraph.json).
agent = make_agent()


def main():
	result = agent.invoke(
		{
			"messages": [
				{
					"role": "user",
					"content": (
						"Analyze /sales.csv: compute total units and revenue per "
						"product, have the visualizer make a bar chart of revenue by "
						"product, and write a summary to report.md."
					),
				}
			]
		}
	)
	print(result["messages"][-1].text)


if __name__ == "__main__":
	main()
