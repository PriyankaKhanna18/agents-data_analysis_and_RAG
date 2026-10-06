"""Extra route for `langgraph dev`: serves the agent's PNG charts over HTTP.

Studio's chat renders Markdown images but can't read files from disk, so the
agent's chart messages link to /charts/<name>.png on the local dev server.
"""

from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, Response
from starlette.routing import Route

WORKSPACE = (Path(__file__).resolve().parent / "analysis_workspace").resolve()


async def chart(request):
	path = (WORKSPACE / request.path_params["name"]).resolve()
	# Only PNGs inside the workspace; blocks "../" tricks reaching other files.
	if path.suffix.lower() != ".png" or not path.is_relative_to(WORKSPACE) or not path.is_file():
		return Response(status_code=404)
	return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


app = Starlette(routes=[Route("/charts/{name:path}", chart)])
