"""Main FastMCP server — mounts all sub-servers."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from .client import GeminiClient
from . import context_cache, tracing
from .config import get_config
from .weaviate_client import WeaviateClient
from .tools.video import video_server
from .academic_client import SemanticScholarClient
from .tools.research import research_server, _ensure_document_tool, _ensure_web_tools, _ensure_academic_tools
from .tools.content import content_server, _ensure_batch_tool
from .tools.search import search_server
from .tools.infra import infra_server
from .tools.youtube import youtube_server
from .tools.knowledge import knowledge_server

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(server: FastMCP):
    """Startup/shutdown hook — sets up tracing, tears down shared clients."""
    tracing.setup()
    yield {}
    tracing.shutdown()
    if get_config().clear_cache_on_shutdown:
        await context_cache.clear()
    await SemanticScholarClient.close()
    await WeaviateClient.aclose()
    closed = await GeminiClient.close_all()
    logger.info("Lifespan shutdown: closed %d client(s)", closed)


app = FastMCP(
    "video-research",
    instructions=(
        "Unified Gemini research partner — video analysis, deep research, "
        "content extraction. Uses the configured Gemini models with thinking support."
    ),
    lifespan=_lifespan,
)

app.mount(video_server)
_ensure_document_tool()  # register research_document on research_server
_ensure_web_tools()  # register Deep Research tools on research_server
_ensure_academic_tools()  # register Semantic Scholar tools on research_server
app.mount(research_server)
_ensure_batch_tool()  # register content_batch_analyze on content_server
app.mount(content_server)
app.mount(search_server)
app.mount(infra_server)
app.mount(youtube_server)
app.mount(knowledge_server)


def _warn_if_no_local_access_roots() -> None:
    """Emit a single startup warning when no local-file allow-list is configured.

    Tools that read local files accept LLM-supplied paths. When the LLM is
    influenced by untrusted content (web pages, transcripts), an unrestricted
    server is a credential-exfiltration risk. The deny-list still blocks the
    obvious credential paths, but a positive allow-list is the recommended
    posture.
    """
    cfg = get_config()
    if cfg.local_file_access_roots:
        return
    logger.warning(
        "LOCAL_FILE_ACCESS_ROOTS is not set, so local file tools will accept "
        "any path the OS user can read (the sensitive-path deny-list still "
        "applies). Set LOCAL_FILE_ACCESS_ROOTS=<dir>[%s<dir>...] to restrict "
        "the server to specific directories. See README 'Local file access "
        "boundaries' for details.",
        os.pathsep,
    )


def main() -> None:
    """Entry-point for ``video-research-mcp`` console script."""
    _warn_if_no_local_access_roots()
    app.run()


if __name__ == "__main__":
    main()
