import logging
from types import SimpleNamespace
from typing import Any, TypeAlias, TypedDict, cast

import requests

logger = logging.getLogger(__name__)


class RetrieverDocument(TypedDict):
    id: str
    contents: str


BatchSearchDocs: TypeAlias = list[list[RetrieverDocument]]
OpenAIChatMessage: TypeAlias = dict[str, Any]


class RetrieverClient:
    def __init__(self, base_url: str, top_k: int = 5, timeout: int | None = None):
        self.batch_search_endpoint = f"{base_url}/batch_search"
        self.top_k = top_k
        self.timeout = timeout
        self.session = requests.Session()

    def batch_search(self, queries: list[str]) -> BatchSearchDocs:
        """
        Returns: BatchSearchDocs 每个 query 对应的 doc 列表
        """
        if not queries:
            return []
        payload = {"query": queries, "top_n": self.top_k, "return_score": False}
        response = self.session.post(
            self.batch_search_endpoint, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        batch_results = cast(BatchSearchDocs, response.json())
        return batch_results

    def batch_search_formatted(self, queries: list[str]) -> list[str]:
        """
        Returns: List[str] 格式化后的 XML 字符串列表
        """
        batch_results = self.batch_search(queries)
        formatted_outputs: list[str] = []
        for search_results in batch_results:
            contents = [doc["contents"] for doc in search_results]
            formatted_outputs.append(
                "\n<information>\n" + ";\n".join(contents) + "\n</information>\n"
            )
        return formatted_outputs


class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float | None = None,
        endpoint: str | None = None,
        extra_headers: dict[str, str] | None = None,
        use_chat_template: bool = False,
    ):
        default_endpoint = "/v1/completions"
        chosen_endpoint = endpoint if endpoint is not None else default_endpoint
        self.completions_endpoint = f"{base_url.rstrip('/')}{chosen_endpoint}"
        self.model = model
        self.timeout = timeout
        self.use_chat_template = use_chat_template
        self.session = requests.Session()
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self.headers = headers

    def _validate_openai_chat_messages(self, messages: Any) -> list[OpenAIChatMessage]:
        if not isinstance(messages, list):
            raise ValueError(
                "When use_chat_template=True, prompts must be list[openai_chat_message]"
            )
        if not messages:
            raise ValueError("Chat messages cannot be empty")

        valid_roles = {"system", "user", "assistant", "tool"}
        validated: list[OpenAIChatMessage] = []
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Chat message at index {i} must be a dict")

            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or role not in valid_roles:
                raise ValueError(
                    f"Chat message at index {i} has invalid role: {role!r}"
                )
            if not isinstance(content, str):
                raise ValueError(f"Chat message at index {i} must have string content")

            validated.append(message)
        return validated

    def _looks_like_chat_message_list(self, value: Any) -> bool:
        if not isinstance(value, list) or not value:
            return False
        return all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in value
        )

    def _prepare_prompts(
        self, prompts: list[Any], tokenizer: Any = None, enable_thinking: bool = False
    ) -> list[str]:
        if not self.use_chat_template:
            if all(isinstance(prompt, str) for prompt in prompts):
                return prompts
            raise ValueError("Prompt must be list[str] for /v1/completions")

        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "tokenizer with apply_chat_template is required when use_chat_template=True"
            )

        conversations: list[list[OpenAIChatMessage]] = []
        if self._looks_like_chat_message_list(prompts):
            conversations = [self._validate_openai_chat_messages(prompts)]
        else:
            for i, prompt in enumerate(prompts):
                try:
                    conversations.append(self._validate_openai_chat_messages(prompt))
                except ValueError as e:
                    raise ValueError(
                        f"Invalid chat conversation at index {i}: {e}"
                    ) from e

        rendered_prompts: list[str] = []
        for messages in conversations:
            rendered_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=enable_thinking,    add_generation_prompt=True,
            )
            if not isinstance(rendered_prompt, str):
                raise ValueError("tokenizer.apply_chat_template must return str")
            rendered_prompts.append(rendered_prompt)
        return rendered_prompts

    def _build_payload(
        self, prompts: Any, config, stop_tokens: list[str]
    ) -> dict[str, Any]:

        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        max_tokens = getattr(config, "max_completion_length", 1024)
        temperature = getattr(config, "temperature", 1.0)
        top_p = getattr(config, "top_p", 0.9)
        n = getattr(config, "vllm_n", 1)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
        }
        repetition_penalty = getattr(config, "repetition_penalty", None)
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty

        prompt_payload: list[str]
        if isinstance(prompts, str):
            prompt_payload = [prompts]
        elif isinstance(prompts, list) and all(
            isinstance(item, str) for item in prompts
        ):
            prompt_payload = prompts
        else:
            raise ValueError("Prompt must be str or list[str] for /v1/completions")
        payload["prompt"] = prompt_payload

        if stop_tokens:
            payload["stop"] = stop_tokens
            payload["include_stop_str_in_output"] = True
        return payload

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            self.completions_endpoint,
            json=payload,
            timeout=self.timeout,
            headers=self.headers if self.headers else None,
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def _parse_choice(self, choice: dict[str, Any]) -> dict[str, Any]:
        text = choice.get("text")
        return {
            "text": text or "",
            "finish_reason": choice.get("finish_reason"),
        }

    def generate(
        self,
        prompts: list[Any],
        config,
        stop_tokens: list[str],
        tokenizer=None,
        enable_thinking: bool = False,
    ) -> list[dict[str, Any]]:
        if not prompts:
            return []

        prepared_prompts = self._prepare_prompts(prompts, tokenizer, enable_thinking)
        payload = self._build_payload(prepared_prompts, config, stop_tokens)
        try:
            data = self._post_json(payload)
        except Exception as e:
            logger.error(f"OpenAI-compatible call failed: {e}")
            # Return one empty choice per prompt instead of [] so downstream
            # code that aligns outputs with prompts by index never crashes:
            # a single bad request degrades to empty generations for this
            # call instead of aborting the whole run.
            return [
                {"text": "", "finish_reason": "error"}
                for _ in prepared_prompts
            ]

        prompt_count = len(prepared_prompts)
        grouped_choices: list[list[dict[str, Any]]] = [[] for _ in range(prompt_count)]
        for choice in data.get("choices", []):
            idx = choice.get("index", 0)
            record = self._parse_choice(choice)
            if 0 <= idx < prompt_count:
                grouped_choices[idx].append(record)
            else:
                logger.warning("Received choice index %s outside prompt range", idx)

        ordered_choices: list[dict[str, Any]] = []
        for idx in range(prompt_count):
            if grouped_choices[idx]:
                ordered_choices.append(grouped_choices[idx][0])
            else:
                ordered_choices.append(
                    {
                        "text": "",
                        "finish_reason": None,
                    }
                )
        return ordered_choices

    def generate_text(
        self,
        prompts: list[Any],
        config,
        stop_tokens: list[str],
        tokenizer=None,
        enable_thinking: bool = False,
    ) -> list[str]:
        return [
            choice["text"]
            for choice in self.generate(
                prompts,
                config,
                stop_tokens,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
            )
        ]
