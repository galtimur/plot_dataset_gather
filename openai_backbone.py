import os
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

import backoff
import openai

class ChatMessage(TypedDict):
    role: str
    content: str


class OpenAIBackbone:
    name: str = "openai"

    def __init__(
        self,
        model_name: str,
        parameters: Dict[str, Any] = {},
        api_key: Optional[str] = None,
    ):
        self._client = openai.OpenAI(api_key=api_key if api_key else os.environ.get("OPENAI_API_KEY"))
        self._model_name = model_name
        self._parameters = parameters

    @backoff.on_exception(backoff.expo, openai.APIError)
    def _get_chat_completion(self, messages: List[ChatMessage]) -> openai.types.chat.ChatCompletion:
        return self._client.chat.completions.create(messages=messages, model=self._model_name, **self._parameters)

    def generate_msg(self, message: List[ChatMessage]) -> Dict[str, Optional[str]]:
        response = self._get_chat_completion(messages=message)
        assert response.choices[0].message.content, "Empty content in OpenAI API response."
        return {
            "prediction": response.choices[0].message.content,
            "created": str(response.created),
            "model": response.model,
            "system_fingerprint": response.system_fingerprint,
        }