from __future__ import annotations

import json

from generatetd.env import env_default, load_env_file
from generatetd.providers import MiniMaxClient, OpenRouterClient, ProviderError, ProviderParseError, parse_json_content


def test_parse_json_content_accepts_fenced_json():
    payload = {
        "disposition": "training_row",
        "label": "valid",
        "confidence": 0.9,
        "rationale": "ok",
        "corrected_candidate_call": None,
        "corrected_final_response": None,
        "required_facts": [],
        "metadata": {},
        "privacy_warnings": [],
    }
    assert parse_json_content("```json\n" + json.dumps(payload) + "\n```") == payload


def test_parse_json_content_extracts_json_after_reasoning_text():
    payload = review_payload()
    content = "<think>I considered the trace.</think>\nFinal answer:\n" + json.dumps(payload)
    assert parse_json_content(content) == payload


def test_parse_json_content_ignores_echoed_schema_object():
    payload = review_payload()
    echoed_schema = {"type": "object", "required": ["disposition"], "properties": {}}
    content = "I should follow this schema:\n" + json.dumps(echoed_schema) + "\nDecision:\n" + json.dumps(payload)
    assert parse_json_content(content) == payload


def test_parse_json_content_error_includes_preview():
    try:
        parse_json_content("not json from provider")
    except ProviderParseError as exc:
        assert "content_preview='not json from provider'" in str(exc)
    else:
        raise AssertionError("expected ProviderParseError")


class FakeResponse:
    def __init__(self, content, status_code=200, text=""):
        self.content = content
        self.status_code = status_code
        self.text = text

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self.content)}}]}


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses) or [FakeResponse(review_payload())]
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def review_payload():
    return {
        "disposition": "training_row",
        "label": "valid",
        "confidence": 0.8,
        "rationale": "sanitized success",
        "corrected_candidate_call": None,
        "corrected_final_response": None,
        "required_facts": [],
        "metadata": {},
        "privacy_warnings": [],
    }


def verification_payload():
    return {
        "approve_training_row": True,
        "confidence": 0.91,
        "rationale": "supported by visible transcript",
        "corrected_disposition": None,
        "corrected_label": None,
        "privacy_warnings": [],
    }


def test_minimax_client_uses_bearer_auth_without_live_call():
    session = FakeSession(FakeResponse(review_payload()))
    client = MiniMaxClient("key", "MiniMax-M2.7", session=session)
    decision = client.review_tool_call({"input": {}})
    assert decision.label == "valid"
    _, kwargs = session.calls[0]
    assert kwargs["headers"]["Authorization"] == "Bearer key"
    assert kwargs["json"]["model"] == "MiniMax-M2.7"
    assert "Output fields, all required" in kwargs["json"]["messages"][1]["content"]


def test_client_verifies_review_without_live_call():
    session = FakeSession(FakeResponse(verification_payload()))
    client = OpenRouterClient("key", "openrouter/owl-alpha", session=session)
    verification = client.verify_review({"input": {}}, review_payload())
    assert verification.approve_training_row is True
    assert verification.confidence == 0.91
    _, kwargs = session.calls[0]
    assert "Verify a proposed Forge verifier training row" in kwargs["json"]["messages"][1]["content"]


def test_openrouter_client_uses_strict_json_schema_without_live_call():
    session = FakeSession(FakeResponse(review_payload()))
    client = OpenRouterClient("key", "openai/gpt-4o-mini", session=session)
    decision = client.review_tool_call({"input": {}})
    assert decision.confidence == 0.8
    _, kwargs = session.calls[0]
    assert kwargs["headers"]["Authorization"] == "Bearer key"
    assert kwargs["json"]["response_format"]["type"] == "json_schema"
    assert kwargs["json"]["response_format"]["json_schema"]["strict"] is True


def test_openrouter_falls_back_when_strict_schema_unavailable():
    messages = []
    session = FakeSession(
        FakeResponse(
            {},
            status_code=404,
            text='{"error":{"message":"No endpoints found that can handle the requested parameters."}}',
        ),
        FakeResponse(review_payload()),
    )
    client = OpenRouterClient(
        "key",
        "deepseek/deepseek-v4-flash:free",
        session=session,
        max_attempts=1,
        on_retry=messages.append,
        sleep=lambda _: None,
    )
    decision = client.review_tool_call({"input": {}})
    assert decision.label == "valid"
    assert len(session.calls) == 2
    first_body = session.calls[0][1]["json"]
    second_body = session.calls[1][1]["json"]
    assert first_body["response_format"]["type"] == "json_schema"
    assert first_body["provider"]["require_parameters"] is True
    assert "response_format" not in second_body
    assert "provider" not in second_body
    assert client.use_strict_schema is False
    assert messages == [
        "api fallback api=openrouter model=deepseek/deepseek-v4-flash:free "
        "reason=strict_json_schema_unavailable status=404"
    ]


def test_client_normalizes_label_like_disposition():
    payload = review_payload()
    payload["disposition"] = "valid"
    session = FakeSession(FakeResponse(payload))
    client = OpenRouterClient("key", "openrouter/owl-alpha", session=session)
    decision = client.review_tool_call({"input": {}})
    assert decision.disposition == "training_row"
    assert decision.label == "valid"


def test_client_retries_transient_status_with_backoff_message():
    retry_messages = []
    sleeps = []
    session = FakeSession(
        FakeResponse({}, status_code=429, text="rate limited"),
        FakeResponse(review_payload()),
    )
    client = OpenRouterClient(
        "key",
        "deepseek/deepseek-v4-flash:free",
        session=session,
        max_attempts=2,
        backoff_seconds=0.5,
        on_retry=retry_messages.append,
        sleep=sleeps.append,
    )
    decision = client.review_tool_call({"input": {}})
    assert decision.label == "valid"
    assert len(session.calls) == 2
    assert sleeps == [0.5]
    assert retry_messages == [
        "api retry api=openrouter model=deepseek/deepseek-v4-flash:free attempt=2/2 wait=0.5s status=429"
    ]


def test_client_does_not_retry_non_transient_status():
    session = FakeSession(FakeResponse({}, status_code=400, text="bad request"))
    client = MiniMaxClient("key", "MiniMax-M2.7", session=session, max_attempts=3, sleep=lambda _: None)
    try:
        client.review_tool_call({"input": {}})
    except ProviderError as exc:
        assert "400 bad request" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_env_loader_sets_defaults_without_overriding_shell_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GENERATETD_OPENROUTER_MODEL", raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "GENERATETD_OPENROUTER_MODEL=deepseek/deepseek-v4-flash:free\n"
        "OPENROUTER_API_KEY=from-file\n"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    loaded = load_env_file(path)
    assert loaded["GENERATETD_OPENROUTER_MODEL"] == "deepseek/deepseek-v4-flash:free"
    assert env_default("GENERATETD_OPENROUTER_MODEL", "fallback") == "deepseek/deepseek-v4-flash:free"
    assert env_default("OPENROUTER_API_KEY", "fallback") == "from-shell"
