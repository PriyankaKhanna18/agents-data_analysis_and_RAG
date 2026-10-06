import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()  # reads COHERE_API_KEY from .env


def main():
	# No isolation: the agent runs shell commands directly on this machine,
	# confined (for file tools) to this throwaway workspace folder.
	workspace = Path(__file__).parent / "agent_workspace"
	workspace.mkdir(exist_ok=True)
	# Only pass PATH to the agent's shell so it can't read the API keys.
	backend = LocalShellBackend(root_dir=workspace, env={"PATH": os.environ["PATH"]})

	# Cohere trial keys allow 20 chat requests/minute; stay under it.
	rate_limiter = InMemoryRateLimiter(requests_per_second=15 / 60, check_every_n_seconds=1)

	agent = create_deep_agent(
		model=ChatCohere(model="command-a-03-2025", rate_limiter=rate_limiter),
		system_prompt=(
			"You are a Python coding assistant with shell access.\n"
			"Your workspace is a single folder. File tools see it as '/', so write "
			"files to paths like '/mypkg/__init__.py'. Shell commands already run "
			"inside that folder, so use relative paths there (e.g. 'pytest -q', "
			"'ls mypkg'). Never use absolute paths in shell commands.\n"
			"If pytest reports failures, fix the code yourself with your file tools "
			"and rerun pytest. Repeat until all tests pass; never just describe a "
			"fix for the user to apply. Finish with a short summary."
		),
		backend=backend,
	)

	result = agent.invoke(
		{
			"messages": [
				{
					"role": "user",
					"content": "Create a small Python package and run pytest",
				}
			]
		}
	)
	print(result["messages"][-1].text)


if __name__ == "__main__":
	main()
