"""Agentic video processing through the Gemini Interactions API.

With ``processing="agentic"`` on a video input, the model navigates the
timeline and loads only the transcript, frames or audio it needs, instead of
sampling the whole video up front (Google's video-understanding docs,
"agentic" section). Google documents it for Gemini 3.8, 3.7 and 3.6 Flash
and 3.5 Flash Lite, through the Interactions API only.

Doc facts this module relies on:

- ``media_resolution`` and ``processing`` are independent; the Interactions
  video block carries resolution as ``resolution``.
- Clipping and fps are "only supported when processing the video in
  'static' mode"; in the SDK they live on the static processing object.
- Structured output uses ``response_format={"type": "text", "mime_type":
  "application/json", "schema": ...}`` (structured-output docs).
- Interactions are stored server side by default (55 days on the paid
  tier); ``store=False`` opts out. Nothing here needs a stored interaction.
"""

from __future__ import annotations

import copy
import logging
import re

from ..client import GeminiClient, _resolve_thinking_level
from ..config import get_config, validate_model_thinking

logger = logging.getLogger(__name__)

_AGENTIC_MODEL_RE = re.compile(
    r"^gemini-3\.(?:[678]-flash|5-flash-lite)(?:-preview(?:-[\w.-]+)?)?$"
)
AGENTIC_MODELS_LABEL = "gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash or gemini-3.5-flash-lite"

_USAGE_FIELDS = (
    "total_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "total_thought_tokens",
    "total_tool_use_tokens",
    "total_cached_tokens",
)


def supports_agentic(model: str) -> bool:
    """True when Google documents agentic video processing for ``model``."""
    return bool(_AGENTIC_MODEL_RE.match(model.removeprefix("models/")))


def check_agentic_model(model: str) -> None:
    """Raise ValueError when ``model`` is not documented for agentic video processing."""
    if not supports_agentic(model):
        raise ValueError(
            f"processing='agentic' needs {AGENTIC_MODELS_LABEL} (the models Google "
            f"documents for agentic video understanding); the configured model is {model!r}. "
            "Set GEMINI_MODEL or use processing='static'."
        )


_SCHEMA_KEYWORDS = ("items", "additionalProperties", "not")
_SCHEMA_LIST_KEYWORDS = ("prefixItems", "anyOf", "oneOf", "allOf")
_SCHEMA_MAP_KEYWORDS = ("$defs", "definitions")


def _is_object_schema(node: dict) -> bool:
    kind = node.get("type")
    if isinstance(kind, list):
        return "object" in kind
    return kind in (None, "object") and isinstance(node.get("properties"), dict)


def require_all_properties(schema: dict) -> dict:
    """Return a copy of ``schema`` where every object lists all its properties as required.

    Google's structured-output docs define ``required`` as "An array of
    strings, listing which properties are mandatory", so any property left
    out of it is optional and the model may omit it. Through the Interactions
    API it does: the default VideoResult schema declares no ``required``, and
    agentic runs came back with only some fields. Marking every property
    required asks for all of them; values can still be empty strings or lists.

    Only nodes in schema position are touched (the root, property values,
    ``items``, ``anyOf`` members, ``$defs`` entries and similar), so a
    property that is itself named ``properties`` or ``required`` is left as is.
    Applied to the server's default schema only; a caller's ``output_schema``
    is sent unchanged.
    """
    out = copy.deepcopy(schema)

    def walk(node) -> None:
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict) and _is_object_schema(node):
            node["required"] = list(properties)
            for child in properties.values():
                walk(child)
        for key in _SCHEMA_KEYWORDS:
            walk(node.get(key))
        for key in _SCHEMA_LIST_KEYWORDS:
            for child in node.get(key) or []:
                walk(child)
        for key in _SCHEMA_MAP_KEYWORDS:
            for child in (node.get(key) or {}).values():
                walk(child)

    walk(out)
    return out


def agentic_video_block(uri: str, *, mime_type: str = "", resolution: str | None = None) -> dict:
    """Build an Interactions video content block with agentic processing.

    Args:
        uri: File API URI or YouTube URL.
        mime_type: Video MIME type; omitted for YouTube URLs.
        resolution: Interactions media resolution; only 'high' is sent, as in
            the static path, so default requests leave the field off.
    """
    block: dict = {"type": "video", "uri": uri, "processing": "agentic"}
    if mime_type:
        block["mime_type"] = mime_type
    if resolution == "high":
        block["resolution"] = "high"
    return block


def _usage_totals(interaction) -> dict:
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return {}
    return {
        name: value
        for name in _USAGE_FIELDS
        if (value := getattr(usage, name, None)) is not None
    }


async def generate_agentic(
    video: dict,
    text: str,
    *,
    schema: dict,
    thinking_level: str | None = None,
) -> tuple[str, dict]:
    """Run one agentic video interaction with JSON structured output.

    The text goes after the video, per Google's guidance for a single video
    plus a prompt. The call is synchronous (no ``background``, which needs
    ``store=True``).

    Not wrapped in ``with_retry``: google-genai sends Interactions calls
    through its generated NextGen client, which retries 408/409/429/5XX with
    backoff (3 retries, 4 attempts in all) when ``http_options`` has no
    ``retry_options`` (``google/genai/_gaos/google_genai.py``,
    ``_translate_retry_config``; pinned by ``TestSdkRetries``). That differs
    from ``models.generate_content``, where ``retry_args(None)`` means a
    single attempt and ``client.py`` adds ``with_retry``. Stacking both here
    would multiply attempts on a paid agentic run.

    Returns:
        (output_text, usage_totals)

    Raises:
        ValueError: On an unsupported model or an empty response.
        RuntimeError: When the interaction ends in a status other than completed.
    """
    cfg = get_config()
    model = cfg.default_model
    check_agentic_model(model)
    level = _resolve_thinking_level(thinking_level or cfg.default_thinking_level)
    validate_model_thinking(model, level)

    client = GeminiClient.get()
    interaction = await client.aio.interactions.create(
        model=model,
        input=[video, {"type": "text", "text": text}],
        store=False,
        generation_config={"thinking_level": level},
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        },
    )
    status = getattr(interaction, "status", None)
    if status != "completed":
        errors = getattr(interaction, "errors", None) or []
        detail = "; ".join(str(getattr(e, "message", e)) for e in errors)
        raise RuntimeError(
            f"Agentic video interaction ended with status {status!r}"
            + (f": {detail}" if detail else "")
        )
    raw = (getattr(interaction, "output_text", None) or "").strip()
    if not raw:
        raise ValueError("Gemini returned no output text for the agentic video interaction")
    return raw, _usage_totals(interaction)
