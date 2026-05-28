from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import ValidationError, validate

from .models import ReviewDecision, ReviewVerification
from .prompts import (
    REVIEW_SCHEMA,
    REVIEW_VERIFICATION_SCHEMA,
    SYSTEM_PROMPT,
    final_response_review_prompt,
    review_verification_prompt,
    tool_call_review_prompt,
)

REVIEW_REQUIRED_KEYS = frozenset(str(key) for key in REVIEW_SCHEMA["required"])
VERIFICATION_REQUIRED_KEYS = frozenset(str(key) for key in REVIEW_VERIFICATION_SCHEMA["required"])


class ProviderError(RuntimeError):
    pass


class ProviderRequestError(ProviderError):
    def __init__(self, provider: str, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"{provider} request failed: {status_code} {body}")


class ProviderParseError(ProviderError):
    def __init__(self, message: str, content: str) -> None:
        self.content_preview = preview_content(content)
        super().__init__(f"{message}; content_preview={self.content_preview!r}")


@dataclass
class ProviderConfig:
    provider: str
    minimax_model: str = "MiniMax-M2.7"
    openrouter_model: str = "deepseek/deepseek-v4-flash:free"
    max_attempts: int = 4
    backoff_seconds: float = 1.0
    on_retry: Callable[[str], None] | None = None


class ReviewClient:
    provider_name: str = "unknown"
    endpoint: str = ""
    model: str = ""

    def describe(self) -> str:
        return f"api={self.provider_name} endpoint={self.endpoint} model={self.model}"

    def review_tool_call(self, payload: dict[str, Any]) -> ReviewDecision:
        raise NotImplementedError

    def review_final_response(self, payload: dict[str, Any]) -> ReviewDecision:
        raise NotImplementedError

    def verify_review(self, payload: dict[str, Any], decision: dict[str, Any]) -> ReviewVerification:
        raise NotImplementedError


class HttpReviewClient(ReviewClient):
    endpoint: str
    provider_name: str = "http"
    retry_statuses = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        model: str,
        session: Any | None = None,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        on_retry: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_attempts = max(1, max_attempts)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.on_retry = on_retry
        self.sleep = sleep
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def review_tool_call(self, payload: dict[str, Any]) -> ReviewDecision:
        content = self._complete_content(tool_call_review_prompt(payload), response_schema=REVIEW_SCHEMA)
        return self._decision_from_content(content)

    def review_final_response(self, payload: dict[str, Any]) -> ReviewDecision:
        content = self._complete_content(final_response_review_prompt(payload), response_schema=REVIEW_SCHEMA)
        return self._decision_from_content(content)

    def verify_review(self, payload: dict[str, Any], decision: dict[str, Any]) -> ReviewVerification:
        content = self._complete_content(
            review_verification_prompt(payload, decision),
            response_schema=REVIEW_VERIFICATION_SCHEMA,
        )
        return self._verification_from_content(content)

    def _complete_content(self, prompt: str, response_schema: dict[str, Any]) -> str:
        raise NotImplementedError

    def _post_with_retries(self, **kwargs: Any) -> Any:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.post(self.endpoint, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    raise ProviderError(f"{self.provider_name} request failed after {attempt} attempts: {exc}") from exc
                self._sleep_before_retry(attempt, f"exception={type(exc).__name__}: {exc}")
                continue

            if response.status_code < 400:
                return response

            body = str(getattr(response, "text", ""))[:500]
            if response.status_code not in self.retry_statuses or attempt >= self.max_attempts:
                raise ProviderRequestError(self.provider_name, response.status_code, body)
            self._sleep_before_retry(attempt, f"status={response.status_code}")
        raise ProviderError(f"{self.provider_name} request failed: {last_error}")

    def _report(self, message: str) -> None:
        if self.on_retry is not None:
            self.on_retry(message)

    def _sleep_before_retry(self, attempt: int, reason: str) -> None:
        delay = self.backoff_seconds * (2 ** (attempt - 1))
        if self.on_retry is not None:
            self.on_retry(
                f"api retry api={self.provider_name} model={self.model} "
                f"attempt={attempt + 1}/{self.max_attempts} wait={delay:.1f}s {reason}"
            )
        if delay > 0:
            self.sleep(delay)

    def _decision_from_content(self, content: str) -> ReviewDecision:
        data = normalize_review_data(parse_json_content(content, REVIEW_REQUIRED_KEYS))
        try:
            validate(instance=data, schema=REVIEW_SCHEMA)
        except ValidationError as exc:
            raise ProviderParseError(f"review response failed schema validation: {exc.message}", content) from exc
        return ReviewDecision(
            disposition=str(data["disposition"]),
            label=str(data["label"]),
            confidence=float(data["confidence"]),
            rationale=str(data["rationale"]),
            corrected_candidate_call=data.get("corrected_candidate_call"),
            corrected_final_response=data.get("corrected_final_response"),
            required_facts=[str(item) for item in data.get("required_facts", [])],
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
            privacy_warnings=[str(item) for item in data.get("privacy_warnings", [])],
        )

    def _verification_from_content(self, content: str) -> ReviewVerification:
        data = parse_json_content(content, VERIFICATION_REQUIRED_KEYS)
        try:
            validate(instance=data, schema=REVIEW_VERIFICATION_SCHEMA)
        except ValidationError as exc:
            raise ProviderParseError(f"review verification failed schema validation: {exc.message}", content) from exc
        return ReviewVerification(
            approve_training_row=bool(data["approve_training_row"]),
            confidence=float(data["confidence"]),
            rationale=str(data["rationale"]),
            corrected_disposition=data.get("corrected_disposition"),
            corrected_label=data.get("corrected_label"),
            privacy_warnings=[str(item) for item in data.get("privacy_warnings", [])],
        )

    def _content_from_response(self, response: Any) -> str:
        try:
            data = response.json()
        except Exception as exc:
            body = str(getattr(response, "text", ""))
            raise ProviderParseError(f"{self.provider_name} response was not JSON: {exc}", body) from exc
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderParseError(f"{self.provider_name} response missing choices[0].message", json.dumps(data)) from exc
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            content = "\n".join(parts)
        if content is None or str(content).strip() == "":
            raise ProviderParseError(f"{self.provider_name} response message.content was empty", json.dumps(data))
        return str(content)


class MiniMaxClient(HttpReviewClient):
    provider_name = "minimax"
    endpoint = "https://api.minimax.io/v1/chat/completions"

    def _complete_content(self, prompt: str, response_schema: dict[str, Any]) -> str:
        response = self._post_with_retries(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_completion_tokens": 1200,
            },
            timeout=90,
        )
        return self._content_from_response(response)


class OpenRouterClient(HttpReviewClient):
    provider_name = "openrouter"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_strict_schema = True

    def _complete_content(self, prompt: str, response_schema: dict[str, Any]) -> str:
        try:
            response = self._post_with_retries(**self._request_kwargs(prompt, response_schema, strict=self.use_strict_schema))
        except ProviderRequestError as exc:
            if self.use_strict_schema and self._strict_schema_unavailable(exc):
                self.use_strict_schema = False
                self._report(
                    "api fallback api=openrouter "
                    f"model={self.model} reason=strict_json_schema_unavailable status={exc.status_code}"
                )
                response = self._post_with_retries(**self._request_kwargs(prompt, response_schema, strict=False))
            else:
                raise
        return self._content_from_response(response)

    def _request_kwargs(self, prompt: str, response_schema: dict[str, Any], strict: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_completion_tokens": 1200,
        }
        if strict:
            body.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "forge_training_review",
                        "strict": True,
                        "schema": response_schema,
                    },
                },
                "provider": {"require_parameters": True},
            })
        return {
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "forge-rs training data generator",
            },
            "json": body,
            "timeout": 90,
        }

    def _strict_schema_unavailable(self, exc: ProviderRequestError) -> bool:
        return exc.status_code == 404 and "No endpoints found that can handle the requested parameters" in exc.body


def parse_json_content(content: str, required_keys: frozenset[str] = REVIEW_REQUIRED_KEYS) -> dict[str, Any]:
    text = content.strip()
    if not text:
        raise ProviderParseError("review response content was empty", content)
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        fallback_objects: list[dict[str, Any]] = []
        for extracted in extract_json_objects(text):
            try:
                candidate = json.loads(extracted)
            except json.JSONDecodeError as nested_exc:
                raise ProviderParseError(
                    (
                        "review response JSON object was malformed "
                        f"at line {nested_exc.lineno} column {nested_exc.colno}: {nested_exc.msg}"
                    ),
                    text,
                ) from nested_exc
            if not isinstance(candidate, dict):
                continue
            if required_keys.issubset(candidate):
                return candidate
            fallback_objects.append(candidate)
        if fallback_objects:
            raise ProviderParseError("review response did not contain a decision JSON object", text) from exc
        if not fallback_objects:
            raise ProviderParseError(
                f"review response was not valid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
                text,
            ) from exc
    if not isinstance(data, dict):
        raise ProviderParseError("review response was not a JSON object", text)
    return data


def normalize_review_data(data: dict[str, Any]) -> dict[str, Any]:
    disposition = data.get("disposition")
    label = data.get("label")
    if disposition == label and isinstance(label, str) and label != "needs_human_review":
        normalized = dict(data)
        normalized["disposition"] = "training_row"
        return normalized
    if disposition == label == "needs_human_review":
        normalized = dict(data)
        normalized["disposition"] = "quarantine"
        return normalized
    return data


def extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start:index + 1])
                    start = text.find("{", index + 1)
                    break
        else:
            start = text.find("{", start + 1)
            continue
        continue
    return objects


def extract_first_json_object(text: str) -> str | None:
    objects = extract_json_objects(text)
    if objects:
        return objects[0]
    return None


def preview_content(content: str, limit: int = 500) -> str:
    compact = re.sub(r"\s+", " ", str(content)).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "...<truncated>"


def build_review_client(config: ProviderConfig) -> ReviewClient | None:
    provider = config.provider
    if provider == "none":
        return None
    if provider == "auto":
        if os.getenv("MINIMAX_API_KEY"):
            provider = "minimax"
        elif os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        else:
            return None
    if provider == "minimax":
        key = os.getenv("MINIMAX_API_KEY")
        if not key:
            raise ProviderError("MINIMAX_API_KEY is required for --provider minimax")
        return MiniMaxClient(
            key,
            config.minimax_model,
            max_attempts=config.max_attempts,
            backoff_seconds=config.backoff_seconds,
            on_retry=config.on_retry,
        )
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ProviderError("OPENROUTER_API_KEY is required for --provider openrouter")
        return OpenRouterClient(
            key,
            config.openrouter_model,
            max_attempts=config.max_attempts,
            backoff_seconds=config.backoff_seconds,
            on_retry=config.on_retry,
        )
    raise ProviderError(f"unknown provider: {config.provider}")
