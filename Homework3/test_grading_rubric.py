"""Offline checks for grading_rubric.py (no Cohere calls)."""

from langchain_core.messages import AIMessage, HumanMessage

from grading_rubric import (
	GRADE_REPORT,
	GradeReportMiddleware,
	InlineRubricMiddleware,
	extract_inline_rubric,
)


def test_inline_rubric_from_studio_chat():
	text = 'Write a haiku. RUBRIC - - Three lines<br>- 5-7-5 syllables<br>- The word "salt" appears'
	assert extract_inline_rubric(text) == '- Three lines\n- 5-7-5 syllables\n- The word "salt" appears'


def test_inline_rubric_multiline_keeps_hyphenated_words():
	assert extract_inline_rubric("Write x.\nRubric:\n- a\n- b-c is fine") == "- a\n- b-c is fine"


def test_word_rubric_alone_is_not_a_rubric():
	assert extract_inline_rubric("explain what a rubric is") is None


def test_inline_middleware_sets_rubric_from_last_user_message():
	state = {"messages": [HumanMessage("Do it.\nRUBRIC:\n- one")]}
	assert InlineRubricMiddleware().before_agent(state, None) == {"rubric": "- one"}
	assert InlineRubricMiddleware().before_agent({**state, "rubric": "- one"}, None) is None


def _evaluation(run_id, iteration, result, passed):
	criterion = {"name": "Tests pass", "passed": True} if passed else {"name": "Tests pass", "passed": False, "gap": "test_basic fails"}
	return {"grading_run_id": run_id, "iteration": iteration, "result": result, "explanation": "x", "criteria": [criterion], "unverified": False}


def test_report_covers_current_run_only_and_is_added_once():
	state = {
		"rubric": "- Tests pass",
		"_current_grading_run_id": "run2",
		"messages": [AIMessage("answer")],
		"_rubric_evaluations": [
			_evaluation("run1", 0, "satisfied", True),
			_evaluation("run2", 0, "needs_revision", False),
			_evaluation("run2", 1, "satisfied", True),
		],
	}
	report = GradeReportMiddleware().after_agent(state, None)["messages"][0]
	assert report.name == GRADE_REPORT
	assert "Pass 1: **needs_revision**" in report.content and "❌ test_basic fails" in report.content
	assert "Pass 2: **satisfied**" in report.content and "Pass 3" not in report.content
	assert GradeReportMiddleware().after_agent({**state, "messages": [report]}, None) is None
