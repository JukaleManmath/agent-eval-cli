from __future__ import annotations

import json
import os
import re

import httpx
import jmespath

from agenteval.schema.test_case import AgentConfig


class AgentClient:
    def __init__(self, agent_config: AgentConfig) -> None:
        self.agent_config = agent_config

    async def send(self, session_id: str, user_message: str) -> str:
        request_body = self._build_request_body(session_id, user_message)
        headers = {
            key: self._substitute(value, session_id, user_message)
            for key, value in self.agent_config.headers.items()
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.agent_config.timeout_seconds
            ) as client:
                response = await client.post(
                    self.agent_config.endpoint,
                    json=request_body,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise RuntimeError(f"Agent request failed: {error}") from error

        try:
            json_body = response.json()
        except ValueError as error:
            raise RuntimeError("Agent returned non-JSON response") from error

        reply = jmespath.search(self.agent_config.response_path, json_body)
        if reply is None:
            raise RuntimeError(
                f"response_path '{self.agent_config.response_path}' "
                "not found in agent response"
            )

        return str(reply)

    def _build_request_body(self, session_id: str, user_message: str) -> object:
        request_template = self._substitute(
            self.agent_config.request_template,
            session_id,
            user_message,
        )
        return json.loads(request_template)

    def _substitute(self, value: str, session_id: str, user_message: str) -> str:
        value = value.replace("${SESSION_ID}", session_id)
        value = value.replace("${USER_MESSAGE}", json.dumps(user_message)[1:-1])
        return re.sub(r"\$\{([^}]+)\}", self._env_replacement, value)

    def _env_replacement(self, match: re.Match[str]) -> str:
        return os.getenv(match.group(1), "")
