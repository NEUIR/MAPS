import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .clients import OpenAIClient, RetrieverClient

THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
ANSWER_TAG_PATTERN = re.compile(
    r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE
)
BOXED_TAG_PATTERN = re.compile(r"<boxed>(.*?)</boxed>", flags=re.DOTALL | re.IGNORECASE)


class PromptedGenerationBase:
    def __init__(
        self,
        llm_client: OpenAIClient,
        generation_max_tokens: int = 1024,
        generation_temperature: float = 0.7,
        generation_top_p: float = 0.9,
        stop_tokens: list[str] | None = None,
        use_chat_template: bool = False,
        tokenizer: Any = None,
        enable_thinking: bool = False,
    ):
        self.llm_client = llm_client
        self.generation_max_tokens = generation_max_tokens
        self.generation_temperature = generation_temperature
        self.generation_top_p = generation_top_p
        self.stop_tokens = stop_tokens or []
        self.use_chat_template = use_chat_template
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

        self.llm_client.use_chat_template = use_chat_template
        if self.use_chat_template and self.tokenizer is None:
            raise ValueError("tokenizer is required when use_chat_template=True")

    def _make_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            max_completion_length=self.generation_max_tokens,
            temperature=self.generation_temperature,
            top_p=self.generation_top_p,
            vllm_n=1,
        )

    def _extract_last_tag(self, pattern: re.Pattern[str], text: str) -> str:
        matches = pattern.findall(text)
        if not matches:
            return ""
        return matches[-1].strip()

    def _extract_answer(self, text: str) -> str:
        answer = self._extract_last_tag(ANSWER_TAG_PATTERN, text)
        return answer if answer else text.strip()

    def _extract_boxed(self, text: str) -> str:
        boxed = self._extract_last_tag(BOXED_TAG_PATTERN, text)
        if boxed:
            return boxed
        answer = self._extract_last_tag(ANSWER_TAG_PATTERN, text)
        if answer:
            boxed_in_answer = self._extract_last_tag(BOXED_TAG_PATTERN, answer)
            if boxed_in_answer:
                return boxed_in_answer
        return ""

    def _extract_think(self, text: str) -> str:
        think = self._extract_last_tag(THINK_TAG_PATTERN, text)
        return think if think else ""

    def format_prompt(self, system_content: str, user_content: str) -> Any:
        if not self.use_chat_template:
            return f"{system_content}\n\n{user_content}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _prompt_to_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            chunks: list[str] = []
            for message in prompt:
                if isinstance(message, dict):
                    role = str(message.get("role", ""))
                    content = str(message.get("content", ""))
                    chunks.append(f"[{role}] {content}")
            return "\n".join(chunks)
        return str(prompt)

    def _generate_text_batch(self, prompts: list[Any]) -> list[str]:
        if not prompts:
            return []
        config = self._make_config()
        if not self.use_chat_template:
            prompt_texts = [self._prompt_to_text(prompt) for prompt in prompts]
            return self.llm_client.generate_text(prompt_texts, config, self.stop_tokens)
        return self.llm_client.generate_text(
            prompts,
            config,
            self.stop_tokens,
            tokenizer=self.tokenizer,
            enable_thinking=self.enable_thinking,
        )

    def _format_external_context(self, label: str, content: str) -> str:
        cleaned = (content or "").strip()
        if not cleaned:
            cleaned = "None"
        if "\n" in cleaned:
            return f"{label}:\n```\n{cleaned}\n```"
        return f"{label}: `{cleaned}`"


def resolve_chat_template_components(args) -> tuple[bool, Any]:
    use_chat_template = getattr(args, "use_chat_template", False)
    if isinstance(use_chat_template, str):
        use_chat_template = use_chat_template.lower() in {"1", "true", "yes", "y", "on"}
    else:
        use_chat_template = bool(use_chat_template)

    tokenizer = getattr(args, "tokenizer", None)
    model_path = getattr(args, "model_path", None)
    if use_chat_template and tokenizer is None and model_path:
        from transformers import AutoTokenizer
        model_path_expanded = str(Path(model_path).expanduser())
        tokenizer = AutoTokenizer.from_pretrained(model_path_expanded,
                                                   trust_remote_code=True)
    return use_chat_template, tokenizer


def build_openai_client_from_args(
    args, use_chat_template: bool = False
) -> OpenAIClient:
    openai_base_url = getattr(args, "openai_base_url", "http://127.0.0.1:8001")
    openai_api_key = getattr(args, "openai_api_key", "TEST")
    model = getattr(args, "model", None) or getattr(args, "llm_model", "Qwen3-14B")
    llm_timeout = getattr(args, "llm_timeout", None)
    return OpenAIClient(
        base_url=openai_base_url,
        model=model,
        api_key=openai_api_key,
        timeout=llm_timeout,
        use_chat_template=use_chat_template,
    )


def build_retriever_client_from_args(args) -> RetrieverClient:
    retriever_base_url = getattr(args, "retriever_base_url", "http://127.0.0.1:9100")
    retriever_top_k = int(getattr(args, "retriever_top_k", 5))
    retriever_timeout = getattr(args, "retriever_timeout", None)
    return RetrieverClient(
        base_url=retriever_base_url, top_k=retriever_top_k, timeout=retriever_timeout
    )


def parse_stop_tokens(args) -> list[str]:
    stop_tokens = getattr(args, "stop_tokens", None)
    if stop_tokens is None:
        return []
    if isinstance(stop_tokens, str):
        return [token for token in stop_tokens.split(",") if token]
    return list(stop_tokens)
