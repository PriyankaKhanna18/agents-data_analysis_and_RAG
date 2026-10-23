"""Deep agent graded against a rubric, following
https://docs.langchain.com/oss/python/deepagents/rubric

RubricMiddleware adds an LLM-as-a-judge loop: when the agent finishes, a
grader sub-agent reads the transcript and checks it against the `rubric`
passed in the input. If any criterion fails, the grader's feedback is added
to the conversation and the agent revises. This repeats until the grader is
satisfied or `max_iterations` grading passes have run.

The grader has the doc's `run_test_suite` tool, so for the find_duplicates
task it can run real tests instead of guessing whether the code works.

With no `rubric` in the input the middleware does nothing and the agent
behaves like a plain deep agent.

Usage:
    python grading_rubric.py                     # run the doc's examples
    python grading_rubric.py --prompt "Write a limerick about cats" \
        --rubric "- Five lines\n- AABBA rhyme scheme"

For a UI, run `langgraph dev` from the repo root, open `grading_rubric` in
LangGraph Studio, and fill in both `messages` and `rubric` in the input. In
Studio's chat view you can instead end the message with "RUBRIC:" followed by
the criteria; InlineRubricMiddleware moves them into `rubric`.
"""

import argparse
import re
import sys
from pathlib import Path

from deepagents import RubricMiddleware, create_deep_agent
from deepagents.middleware.rubric import RubricEvaluation, RubricState
from langchain.agents.middleware import AgentMiddleware
from cohere.errors import UnprocessableEntityError
from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

# `langgraph dev` loads this file by path, so sibling folders aren't on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent / "skills_agent"))
from skills_agent import FlattenSystemPromptMiddleware, build_model  # noqa: E402


@tool
def run_test_suite(code: str) -> dict:
	"""Run the find_duplicates test suite against Python source code."""
	namespace: dict = {"__builtins__": __builtins__}
	try:
		exec(code, namespace)
	except Exception as exc:
		return {"ok": False, "failures": [f"Failed to execute code: {exc}"]}

	find_duplicates = namespace.get("find_duplicates")
	if find_duplicates is None:
		return {"ok": False, "failures": ["Function find_duplicates is not defined"]}

	tests = [
		("test_basic", [1, 2, 2, 3, 1], [2, 1]),
		("test_empty", [], []),
		("test_no_duplicates", [1, 2, 3], []),
		("test_unhashable", [[1], [1], 2], [[1]]),
	]
	failures: list[str] = []
	for name, args, expected in tests:
		try:
			actual = find_duplicates(args)
			if actual != expected:
				failures.append(f"{name}: expected {expected}, got {actual}")
		except Exception as exc:
			failures.append(f"{name}: {exc}")

	return {"ok": not failures, "failures": failures}


def log_evaluation(ev: RubricEvaluation) -> None:
	print(f"  [grader] iteration {ev['iteration']}: {ev['result']} — {ev['explanation']}")


# "RUBRIC" (any case) then ":", "-" or a line break, then the criteria.
_INLINE_RUBRIC = re.compile(r"\bRUBRIC\b[ \t]*(?:[:\-]|\n)\s*(.+)\Z", re.IGNORECASE | re.DOTALL)


def _message_text(content) -> str:
	if isinstance(content, str):
		return content
	return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def extract_inline_rubric(text: str) -> str | None:
	"""Pull "RUBRIC: - a - b" out of a chat message as one criterion per line."""
	match = _INLINE_RUBRIC.search(text)
	if not match:
		return None
	body = re.sub(r"<br\s*/?>", "\n", match.group(1))
	# Criteria typed on one line ("- a - b") become one per line.
	body = re.sub(r"\s+-\s+", "\n- ", " " + body)
	criteria = [c.strip(" -\t") for c in body.splitlines()]
	criteria = [c for c in criteria if c]
	return "\n".join(f"- {c}" for c in criteria) or None


class InlineRubricMiddleware(AgentMiddleware):
	"""Sets `rubric` from a "RUBRIC:" section in the latest user message.

	Studio's chat view only sends `messages`, so without this a rubric typed
	into the chat is never graded. Must run before RubricMiddleware.
	"""

	state_schema = RubricState

	def before_agent(self, state, runtime):
		messages = state.get("messages") or []
		if not messages or messages[-1].type != "human":
			return None
		rubric = extract_inline_rubric(_message_text(messages[-1].content))
		if rubric is None or rubric == state.get("rubric"):
			return None
		return {"rubric": rubric}


GRADE_REPORT = "grade_report"


def format_grade_report(evaluations: list[RubricEvaluation]) -> str:
	lines = ["**Grader report**"]
	for ev in evaluations:
		lines.append(f"\nPass {ev['iteration'] + 1}: **{ev['result']}** — {ev['explanation']}")
		for c in ev["criteria"]:
			mark = "✅" if c["passed"] else f"❌ {c['gap']}"
			lines.append(f"- {c['name']}: {mark}")
	return "\n".join(lines)


class GradeReportMiddleware(AgentMiddleware):
	"""Appends the grader's verdicts to the chat once grading finishes.

	RubricMiddleware keeps its verdicts in private state, so in Studio a run
	that passes on the first try looks the same as an ungraded one. Listed
	before RubricMiddleware so its after_agent runs after the grader's (after
	hooks run in reverse order); while the grader asks for a revision it jumps
	back to the model and this hook doesn't run.
	"""

	state_schema = RubricState

	def after_agent(self, state, runtime):
		run_id = state.get("_current_grading_run_id")
		if not state.get("rubric") or not run_id:
			return None
		last = state["messages"][-1]
		if last.name == GRADE_REPORT and last.additional_kwargs.get("grading_run_id") == run_id:
			return None
		evaluations = [e for e in state.get("_rubric_evaluations", []) if e["grading_run_id"] == run_id]
		if not evaluations:
			return None
		report = AIMessage(
			content=format_grade_report(evaluations),
			name=GRADE_REPORT,
			additional_kwargs={"grading_run_id": run_id},
		)
		return {"messages": [report]}


class GraderChatCohere(ChatCohere):
	"""ChatCohere that accepts LangChain's tool_choice="any".

	The grader returns its verdict through a forced tool call (ToolStrategy),
	which LangChain requests as tool_choice="any". Cohere only understands
	"REQUIRED" and rejects "any" with HTTP 422, so every grading pass would end
	in grader_error.
	"""

	def bind_tools(self, tools, **kwargs):
		if kwargs.get("tool_choice") == "any":
			kwargs["tool_choice"] = "REQUIRED"
		return super().bind_tools(tools, **kwargs)

	# Cohere sometimes emits a malformed tool call (often when passing code to
	# run_test_suite) and fails with 422 INVALID_TOOL_GENERATION. It's random,
	# so retrying usually works; without this the pass ends in grader_error.
	_RETRIES = 3

	@staticmethod
	def _is_bad_tool_generation(exc: Exception) -> bool:
		return isinstance(exc, UnprocessableEntityError) and "INVALID_TOOL_GENERATION" in str(exc)

	def _generate(self, *args, **kwargs):
		for attempt in range(self._RETRIES):
			try:
				return super()._generate(*args, **kwargs)
			except UnprocessableEntityError as exc:
				if not self._is_bad_tool_generation(exc) or attempt == self._RETRIES - 1:
					raise

	async def _agenerate(self, *args, **kwargs):
		for attempt in range(self._RETRIES):
			try:
				return await super()._agenerate(*args, **kwargs)
			except UnprocessableEntityError as exc:
				if not self._is_bad_tool_generation(exc) or attempt == self._RETRIES - 1:
					raise


model = build_model()
# Same settings and rate limiter as the agent's model, so together they stay
# under the Cohere trial limit.
grader_model = GraderChatCohere(
	model=model.model, rate_limiter=model.rate_limiter, disable_streaming=model.disable_streaming
)

agent = create_deep_agent(
	model=model,
	system_prompt=(
		"You are a careful assistant and Python engineer. Write correct, readable "
		"code and follow the user's instructions exactly. Put your final answer "
		"(including any code) in your last message."
	),
	middleware=[
		InlineRubricMiddleware(),
		GradeReportMiddleware(),
		RubricMiddleware(
			model=grader_model,
			system_prompt=(
				"You are a strict reviewer grading the assistant's final answer against "
				"a rubric. Check every criterion separately. When a criterion mentions "
				"run_test_suite, call it with the assistant's latest code and base the "
				"verdict on its result. Pass only the function definition, with "
				"docstrings and comments removed: Cohere can't generate a tool call "
				"whose code argument contains them. For syllable counts, count each word's syllables and "
				"write the per-line totals before deciding. For every failing criterion, say exactly what is wrong "
				"so the assistant can fix it: for failed tests, quote each failure "
				"message (test name, input, expected and actual output)."
			),
			tools=[run_test_suite],
			# Cohere ignores list-style system prompts; the grader needs this too.
			grader_middleware=[FlattenSystemPromptMiddleware()],
			max_iterations=3,
			on_evaluation=log_evaluation,
		),
		FlattenSystemPromptMiddleware(),
	],
)

EXAMPLES = [
	(
		"Write a haiku about spring.",
		"- The poem has three lines\n- Lines follow a 5-7-5 syllable pattern\n- The theme is spring",
	),
	(
		"Write a one-sentence summary of photosynthesis.",
		"- The answer is one sentence\n- The answer mentions light and chlorophyll",
	),
	(
		"Write a Python function `find_duplicates(lst)` that returns a list of all elements "
		"that appear more than once in the input list, in the order they first appear.",
		"- All tests pass in run_test_suite\n"
		"- The function is named `find_duplicates` and accepts a single list argument",
	),
]


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--prompt", help="run one prompt instead of the examples")
	parser.add_argument("--rubric", default="", help="rubric for --prompt (use \\n between criteria)")
	args = parser.parse_args()

	examples = [(args.prompt, args.rubric.replace("\\n", "\n"))] if args.prompt else EXAMPLES
	for i, (prompt, rubric) in enumerate(examples):
		print(f"\n{'=' * 70}\nUSER: {prompt}\nRUBRIC:\n{rubric}\n{'-' * 70}")
		result = agent.invoke(
			{"messages": [{"role": "user", "content": prompt}], "rubric": rubric},
			config={"configurable": {"thread_id": str(i)}},
		)
		answer = next(m for m in reversed(result["messages"]) if m.name != GRADE_REPORT)
		print(f"{'-' * 70}\n{answer.text}")


if __name__ == "__main__":
	main()
