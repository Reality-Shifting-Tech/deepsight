"""Tests for the vision-session orchestrator: tool loop, markers, caps."""

from conftest import simple_result

from deepsight.backends import ToolCall
from deepsight.cache import PerceptionCache
from deepsight.orchestrator import Orchestrator, SessionResult


def make_orchestrator(reasoning, vision, **kwargs) -> Orchestrator:
    return Orchestrator(
        reasoning=reasoning,
        vision=vision,
        cache=kwargs.pop("cache", None),
        max_look_rounds=kwargs.pop("max_look_rounds", 5),
        sketch_enabled=kwargs.pop("sketch_enabled", True),
    )


def test_plain_answer_no_tools(fake_reasoning, fake_vision, png_data_url):
    reasoning = fake_reasoning([simple_result("14")])
    vision = fake_vision(text='{"objects":[]}', prompt=3, completion=1)
    orch = make_orchestrator(reasoning, vision)

    result = orch.run(png_data_url, "How many items?")

    assert isinstance(result, SessionResult)
    assert result.content == "14"
    assert result.rounds == 1
    assert result.tool_calls == 0
    # sketch vision call + one reasoning call
    assert vision.calls == 1
    assert reasoning.calls == 1
    assert result.prompt_tokens >= 3 + 10


def test_tool_call_path(fake_reasoning, fake_vision, png_data_url):
    look = ToolCall(id="c1", name="look", arguments={"x": 0, "y": 0, "w": 50, "h": 50})
    reasoning = fake_reasoning([simple_result("", tool_calls=[look]), simple_result("42")])
    vision = fake_vision(text="a bar", prompt=4, completion=2)
    orch = make_orchestrator(reasoning, vision)

    result = orch.run(png_data_url, "What value?")

    assert result.content == "42"
    assert result.rounds == 2
    assert result.tool_calls == 1
    # sketch + the look
    assert vision.calls == 2
    # tool result fed back before the final turn
    assert any(m.get("role") == "tool" and m.get("content") for m in reasoning.last_messages)


def test_text_marker_fallback(fake_reasoning, fake_vision, png_data_url):
    reasoning = fake_reasoning([simple_result("[LOOK 0,0,50,50]"), simple_result("7")])
    vision = fake_vision(text="region content", prompt=4, completion=2)
    orch = make_orchestrator(reasoning, vision)

    result = orch.run(png_data_url, "What's there?")

    assert result.content == "7"
    assert result.rounds == 2
    assert result.tool_calls == 1
    assert vision.calls == 2


def test_max_rounds_cap(fake_reasoning, fake_vision, png_data_url):
    look = ToolCall(id="c1", name="look", arguments={"x": 0, "y": 0, "w": 50, "h": 50})
    reasoning = fake_reasoning([simple_result("", tool_calls=[look])] * 5)
    vision = fake_vision(text="x", prompt=4, completion=2)
    orch = make_orchestrator(reasoning, vision, max_look_rounds=5)

    result = orch.run(png_data_url, "keep looking")

    assert result.content.startswith("[deepsight] reached max look rounds")
    assert result.rounds == 5
    assert result.tool_calls == 5


def test_sketch_disabled(fake_reasoning, fake_vision, png_data_url):
    reasoning = fake_reasoning([simple_result("ok")])
    vision = fake_vision(text="sketch text", prompt=3, completion=1)
    orch = make_orchestrator(reasoning, vision, sketch_enabled=False)

    result = orch.run(png_data_url, "hi")

    assert result.content == "ok"
    assert vision.calls == 0
    assert result.prompt_tokens == 10


def test_cache_hits_reduce_vision_calls(fake_reasoning, fake_vision, png_data_url):
    look = ToolCall(id="c1", name="look", arguments={"x": 0, "y": 0, "w": 50, "h": 50})
    reasoning = fake_reasoning([simple_result("", tool_calls=[look])] * 4 + [simple_result("done")])
    vision = fake_vision(text="cached region", prompt=4, completion=2)
    orch = make_orchestrator(reasoning, vision, cache=PerceptionCache(ttl_seconds=3600))

    result = orch.run(png_data_url, "loop")

    assert result.content == "done"
    # sketch once + look once (subsequent identical looks hit the cache)
    assert vision.calls == 2
    assert result.cache_hits >= 1
