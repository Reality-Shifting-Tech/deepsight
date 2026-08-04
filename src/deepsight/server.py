"""FastAPI application: OpenAI-compatible surface for deepsight.

Exposes ``POST /v1/chat/completions`` and ``GET /v1/models`` so any
OpenAI-compatible client (curl, openai SDK, OpenWebUI, LibreChat) can
talk to DeepSight without changes. The orchestrator does the vision
work; the server just maps the OpenAI message shape to the session.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .backends import ReasoningBackend, build_vision_backend
from .cache import PerceptionCache
from .config import Settings, get_settings
from .orchestrator import Orchestrator, _usage_sum

# ---------------------------------------------------------------------------
# Request/response models (OpenAI-compatible subset)
# ---------------------------------------------------------------------------


class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart] | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "deepsight"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the deepsight application with its long-lived components."""
    settings = settings or get_settings()
    reasoning = ReasoningBackend(
        base_url=settings.reasoning_base_url,
        api_key=settings.reasoning_key,
        model=settings.reasoning_model,
        temperature=settings.reasoning_temperature,
        max_tokens=settings.reasoning_max_tokens,
    )
    vision = build_vision_backend(settings)
    cache = PerceptionCache(ttl_seconds=settings.cache_ttl_seconds)
    orchestrator = Orchestrator(
        reasoning=reasoning,
        vision=vision,
        cache=cache if settings.cache_enabled else None,
        max_look_rounds=settings.max_look_rounds,
        sketch_enabled=settings.sketch_enabled,
        tool_round_max_tokens=settings.reasoning_tool_round_max_tokens,
        final_max_tokens=settings.reasoning_max_tokens,
        vision_tool_max_tokens=settings.vision_tool_max_tokens,
    )

    app = FastAPI(
        title="deepsight",
        description=(
            "OpenAI-compatible vision proxy that gives text-only LLMs "
            "interactive vision via a sketch + targeted tool-call loop."
        ),
        version=__version__,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.orchestrator = orchestrator
    app.state.settings = settings

    @app.get("/v1/models")
    def list_models() -> ModelList:
        now = int(time.time())
        return ModelList(
            data=[
                ModelInfo(id=settings.reasoning_model, created=now),
                ModelInfo(id=f"{settings.reasoning_model}+vision", created=now),
            ]
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest) -> Any:
        image_url, user_text = _extract_content(req)
        if not image_url:
            raise HTTPException(status_code=400, detail="no image_url found in messages")

        if req.stream:
            return StreamingResponse(
                _stream_live(
                    orchestrator,
                    req.model or settings.reasoning_model,
                    image_url,
                    user_text,
                ),
                media_type="text/event-stream",
            )
        session = await _run_session(orchestrator, image_url, user_text)
        return _openai_response(req.model or settings.reasoning_model, session)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_content(req: ChatRequest) -> tuple[str | None, str]:
    """Pull the last image URL and the accumulated user text from messages."""
    image_url: str | None = None
    texts: list[str] = []
    for msg in req.messages:
        if isinstance(msg.content, str):
            if msg.role == "user":
                texts.append(msg.content)
            continue
        for part in msg.content or []:
            if part.type == "image_url" and part.image_url:
                image_url = part.image_url.url
            elif part.type == "text" and part.text:
                texts.append(part.text)
    return image_url, "\n".join(t for t in texts if t.strip())


async def _run_session(
    orchestrator: Orchestrator,
    image_url: str,
    user_text: str,
    on_event: Callable[[str], None] | None = None,
) -> Any:
    """Run the vision session off the event loop (blocking backend calls)."""
    import asyncio

    return await asyncio.to_thread(
        orchestrator.run,
        image_url,
        user_text or "What do you see in this image?",
        on_event,
    )


def _openai_response(model: str, session: Any) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": session.content},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_sum(
            session.prompt_tokens,
            session.completion_tokens,
            session.cache_hit_tokens,
            session.cache_miss_tokens,
        ),
    }


async def _stream_live(
    orchestrator: Orchestrator,
    model: str,
    image_url: str,
    user_text: str,
) -> AsyncIterator[str]:
    """Stream an OpenAI SSE session with live progress status chunks.

    The vision session runs as a background task; each milestone event
    from the orchestrator (e.g. "👁️ viewing image...") is emitted
    immediately as a ``delta.status`` chunk, and the final answer arrives
    as a normal ``delta.content`` chunk when the session completes.
    """
    import asyncio
    import queue as q

    events: q.Queue[str] = q.Queue()
    task = asyncio.create_task(
        _run_session(orchestrator, image_url, user_text, on_event=events.put)
    )

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    yield sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )

    while True:
        try:
            status = events.get_nowait()
        except q.Empty:
            if task.done():
                break
            await asyncio.sleep(0.05)
            continue
        yield sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "status": status},
                        "finish_reason": None,
                    }
                ],
            }
        )

    session = task.result()
    usage = _usage_sum(
        session.prompt_tokens,
        session.completion_tokens,
        session.cache_hit_tokens,
        session.cache_miss_tokens,
    )

    yield sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": session.content}, "finish_reason": None}
            ],
        }
    )
    yield sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage,
        }
    )
    yield "data: [DONE]\n\n"


app = create_app()
