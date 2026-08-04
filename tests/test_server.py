"""Tests for the FastAPI OpenAI-compatible surface."""

import json

import pytest
from fastapi.testclient import TestClient

from deepsight.orchestrator import SessionResult
from deepsight.server import create_app


def make_session(content: str = "42", prompt: int = 30, completion: int = 12) -> SessionResult:
    return SessionResult(
        content=content,
        prompt_tokens=prompt,
        completion_tokens=completion,
        rounds=2,
        tool_calls=1,
        cache_hits=0,
    )


@pytest.fixture
def client(monkeypatch, png_data_url):
    app = create_app()

    async def fake_run(orchestrator, image_url, user_text, on_event=None):
        if on_event:
            on_event("👁️ viewing image...")
            on_event("✏️ sketching scene...")
        return make_session()

    monkeypatch.setattr("deepsight.server._run_session", fake_run)
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2
    assert any(m["id"].endswith("+vision") for m in body["data"])


def test_chat_completion_ok(client, png_data_url):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepsight",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": png_data_url}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "42"
    assert body["usage"]["total_tokens"] == 42


def test_chat_completion_missing_image_400(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "no image here"}]},
    )
    assert resp.status_code == 400
    assert "image_url" in resp.json()["detail"]


def test_chat_completion_stream(client, png_data_url):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepsight",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": png_data_url}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    chunks = [line for line in resp.text.splitlines() if line.startswith("data:")]
    assert chunks[-1].strip() == "data: [DONE]"
    payloads = [json.loads(c[6:]) for c in chunks[:-1]]
    assert any("42" in json.dumps(p) for p in payloads)
    assert any(p.get("usage") for p in payloads)


def test_chat_completion_stream_emits_status_chunks(client, png_data_url):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepsight",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": png_data_url}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    chunks = [line for line in resp.text.splitlines() if line.startswith("data:")]
    payloads = [json.loads(c[6:]) for c in chunks if c.strip() != "data: [DONE]"]
    statuses = [
        p["choices"][0]["delta"].get("status")
        for p in payloads
        if p.get("choices") and p["choices"][0].get("delta")
    ]
    assert "👁️ viewing image..." in statuses
    assert "✏️ sketching scene..." in statuses
    assert statuses.index("👁️ viewing image...") < statuses.index("✏️ sketching scene...")


def test_text_only_message_with_image_url(client, png_data_url):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "plain text"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": png_data_url}},
                    ],
                },
            ]
        },
    )
    assert resp.status_code == 200
