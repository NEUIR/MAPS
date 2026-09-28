"""MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in
Retrieval-Augmented Generation.

MAPS maintains an accumulating *retrieval experience store*. In each iteration
the Refine Agent receives the current experience store plus the latest search
results and produces a COMPLETE UPDATED experience set: it keeps valid
experiences, refines them with new evidence, adds new lessons, and drops
unreliable ones. The accumulating store is fed back to the Intent Agent (better
search directions) and the Query Agents (better search queries), forming a
self-improving search loop.
"""

import hashlib
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeAlias, TypedDict, cast

logger = logging.getLogger(__name__)

from .clients import (
    BatchSearchDocs,
    OpenAIClient,
    RetrieverClient,
    RetrieverDocument,
)
from .base import (
    PromptedGenerationBase,
    build_retriever_client_from_args,
    parse_stop_tokens,
)

Prompt: TypeAlias = str | list[dict[str, str]]

# ---------------------------------------------------------------------------
# XML tag patterns
# ---------------------------------------------------------------------------
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
SEARCH_TAG_PATTERN = re.compile(
    r"<search>(.*?)</search>", flags=re.DOTALL | re.IGNORECASE
)
ANSWER_TAG_PATTERN = re.compile(
    r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE
)
SEARCH_DIRECTIONS_PATTERN = re.compile(
    r"<search_directions>(.*?)</search_directions>", flags=re.DOTALL | re.IGNORECASE
)
DIRECTION_TAG_PATTERN = re.compile(
    r"<direction(?:\s+id=[\"']?(.*?)[\"']?)?>(.*?)</direction>",
    flags=re.DOTALL | re.IGNORECASE,
)
EXPERIENCE_TAG_PATTERN = re.compile(
    r"<experience>(.*?)</experience>", flags=re.DOTALL | re.IGNORECASE
)
EXPERIENCES_BLOCK_PATTERN = re.compile(
    r"<experiences>(.*?)</experiences>", flags=re.DOTALL | re.IGNORECASE
)
REFINED_INFORMATION_PATTERN = re.compile(
    r"<refined_information>(.*?)</refined_information>",
    flags=re.DOTALL | re.IGNORECASE,
)
LEGACY_INFORMATION_PATTERN = re.compile(
    r"Helpful\s+Information\s*:\s*(.*?)(?=<experiences?\b|$)",
    flags=re.DOTALL | re.IGNORECASE,
)
# Matches placeholder-only values such as "...", "…", or whitespace — these
# come from models echoing the prompt's format examples (e.g. the literal
# "<answer>...</answer>" instruction) and must never be treated as real values.
PLACEHOLDER_VALUE_PATTERN = re.compile(r"[\s.…]*")
# Stage 3 query-mining patterns: models that deliberate on query wording
# inside their think monologue often state the committed query explicitly
# before hitting the token limit (e.g. `Query: "Ulli Ganga" film release date.`).
QUERY_DECLARATION_PATTERN = re.compile(
    r"(?:final\s+)?query\s*:\s*(.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE
)
QUERY_COMMITMENT_PATTERN = re.compile(
    r"\bi(?:'ll| will)\s+(?:use|output|go with)\s*:?\s*(.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
# Structural tags of this pipeline. Experiences containing them are prompt
# echoes / meta monologue, not retrieval lessons.
STRUCTURAL_TAG_PATTERN = re.compile(
    r"</?(?:think|search_directions|direction|search|answer|experiences?|refined_information)\b[^>]*>",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        str(Path(model_path).expanduser()), trust_remote_code=True
    )


def _format_external_context(title: str, content: str) -> str:
    """Helper to format a titled context block."""
    return f"<{title}>\n{content}\n</{title}>"


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class SearchDirection(TypedDict):
    direction_id: str
    direction: str


class AgentAnswerRecord(TypedDict):
    prompt: str
    output: str
    answer: str


class QueryAgentCoreRecord(TypedDict):
    query_id: int
    direction_id: str
    direction: str
    search_query: str


class QueryAgentIterationRecord(QueryAgentCoreRecord):
    prompt: str
    intent_agent_think: str
    query_agent_think: str
    query_agent_output: str


class RefinedPathResult(QueryAgentCoreRecord):
    think: str
    refined_information: str
    selected_doc_ids: list[str]


class PooledDocument(TypedDict):
    doc_id: str
    contents: str
    sources: list[str]
    source_directions: list[str]
    source_queries: list[str]


class IntentIterationRecord(AgentAnswerRecord):
    think: str
    search_directions: list[SearchDirection]


class RefineIterationRecord(TypedDict):
    prompt: str
    refine_agent_output: str
    refined_information: str
    retrieval_experiences: list[str]
    pooled_docs: list[PooledDocument]
    refined_paths: list[RefinedPathResult]


class ReasonRecord(AgentAnswerRecord):
    pass


class IterationRecord(TypedDict, total=False):
    iteration: int
    intent: IntentIterationRecord
    query_agents: list[QueryAgentIterationRecord]
    refine: RefineIterationRecord


class ParallelO1Result(TypedDict):
    query: str
    max_iterations: int
    executed_iterations: int
    iterations: list[IterationRecord]
    final_reason: ReasonRecord
    final_answer: str
    refined_information: str
    retrieval_experiences: list[str]


# ===================================================================
# Main class
# ===================================================================

class MAPS(PromptedGenerationBase):
    """
    MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in
    Retrieval-Augmented Generation.

    The refine agent does triple duty each iteration:
      1. Extracts helpful information for answering the question.
      2. Reviews the current experience store and decides what to keep,
         refine, modify, or delete based on the latest search results.
      3. Produces a COMPLETE UPDATED experience store that replaces the old one.

    The accumulating store is passed to the Intent Agent (better directions)
    and the Query Agents (better queries) in subsequent rounds.
    """

    def __init__(
        self,
        retriever: RetrieverClient,
        intent_agent_llm_client: OpenAIClient,
        refine_agent_llm_client: OpenAIClient,
        query_agent_llm_client: OpenAIClient,
        *,
        # --- Intent agent ---
        intent_agent_max_tokens: int = 512,
        intent_agent_temperature: float = 0.6,
        intent_agent_top_p: float = 0.9,
        # --- Query agents ---
        query_agent_max_tokens: int = 512,
        query_agent_temperature: float = 0.8,
        query_agent_top_p: float = 0.95,
        # --- Refine agent ---
        refine_agent_max_tokens: int = 512,
        refine_agent_temperature: float = 0.2,
        refine_agent_top_p: float = 0.9,
        # --- Synthesis ---
        reason_max_tokens: int = 512,
        reason_temperature: float = 0.3,
        reason_top_p: float = 0.9,
        # --- Sampling ---
        repetition_penalty: float | None = None,
        # --- Stop tokens ---
        stop_tokens: list[str] | None = None,
        # --- Chat template & thinking flags ---
        intent_agent_use_chat_template: bool = False,
        intent_agent_tokenizer: Any = None,
        intent_agent_enable_thinking: bool = False,
        refine_agent_use_chat_template: bool = False,
        refine_agent_tokenizer: Any = None,
        refine_agent_enable_thinking: bool = False,
        query_agent_use_chat_template: bool = False,
        query_agent_tokenizer: Any = None,
        query_agent_enable_thinking: bool = False,
    ):
        super().__init__(
            llm_client=intent_agent_llm_client,
            generation_max_tokens=reason_max_tokens,
            generation_temperature=reason_temperature,
            generation_top_p=reason_top_p,
            stop_tokens=stop_tokens,
            use_chat_template=intent_agent_use_chat_template,
            tokenizer=intent_agent_tokenizer,
        )
        self.retriever = retriever
        self.intent_agent_llm_client = intent_agent_llm_client
        self.refine_agent_llm_client = refine_agent_llm_client
        self.query_agent_llm_client = query_agent_llm_client

        self.intent_agent_max_tokens = intent_agent_max_tokens
        self.intent_agent_temperature = intent_agent_temperature
        self.intent_agent_top_p = intent_agent_top_p

        self.query_agent_max_tokens = query_agent_max_tokens
        self.query_agent_temperature = query_agent_temperature
        self.query_agent_top_p = query_agent_top_p

        self.refine_agent_max_tokens = refine_agent_max_tokens
        self.refine_agent_temperature = refine_agent_temperature
        self.refine_agent_top_p = refine_agent_top_p

        self.reason_max_tokens = reason_max_tokens
        self.reason_temperature = reason_temperature
        self.reason_top_p = reason_top_p

        self.repetition_penalty = repetition_penalty

        self.intent_agent_use_chat_template = intent_agent_use_chat_template
        self.intent_agent_tokenizer = intent_agent_tokenizer
        self.intent_agent_enable_thinking = intent_agent_enable_thinking

        self.refine_agent_use_chat_template = refine_agent_use_chat_template
        self.refine_agent_tokenizer = refine_agent_tokenizer
        self.refine_agent_enable_thinking = refine_agent_enable_thinking

        self.query_agent_use_chat_template = query_agent_use_chat_template
        self.query_agent_tokenizer = query_agent_tokenizer
        self.query_agent_enable_thinking = query_agent_enable_thinking

        if self.refine_agent_use_chat_template and self.refine_agent_tokenizer is None:
            raise ValueError(
                "refine_agent_tokenizer is required when refine_agent_use_chat_template=True"
            )
        if self.query_agent_use_chat_template and self.query_agent_tokenizer is None:
            raise ValueError(
                "query_agent_tokenizer is required when query_agent_use_chat_template=True"
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> "MAPS":
        intent_agent_use_chat_template = _coerce_bool(
            _first_not_none(
                getattr(args, "intent_agent_use_chat_template", None),
                getattr(args, "shared_use_chat_template", None),
                getattr(args, "use_chat_template", None),
            ),
            default=False,
        )
        intent_agent_enable_thinking = _coerce_bool(
            _first_not_none(
                getattr(args, "intent_agent_enable_thinking", None),
                getattr(args, "shared_enable_thinking", None),
                getattr(args, "enable_thinking", None),
            ),
            default=False,
        )
        refine_agent_use_chat_template = _coerce_bool(
            getattr(args, "refine_agent_use_chat_template", None),
            default=intent_agent_use_chat_template,
        )
        refine_agent_enable_thinking = _coerce_bool(
            getattr(args, "refine_agent_enable_thinking", None),
            default=intent_agent_enable_thinking,
        )
        query_agent_use_chat_template = _coerce_bool(
            getattr(args, "query_agent_use_chat_template", None),
            default=intent_agent_use_chat_template,
        )
        query_agent_enable_thinking = _coerce_bool(
            getattr(args, "query_agent_enable_thinking", None),
            default=intent_agent_enable_thinking,
        )

        intent_agent_model = _first_not_none(
            getattr(args, "intent_agent_model", None),
            getattr(args, "shared_model", None),
            getattr(args, "model", None),
            getattr(args, "llm_model", None),
            "Qwen3-14B",
        )
        refine_agent_model = _first_not_none(
            getattr(args, "refine_agent_model", None),
            getattr(args, "shared_model", None),
            getattr(args, "model", None),
            getattr(args, "llm_model", None),
            intent_agent_model,
        )
        query_agent_model = (
            getattr(args, "query_agent_model", None) or intent_agent_model
        )

        intent_agent_model_path = _first_not_none(
            getattr(args, "intent_agent_model_path", None),
            getattr(args, "shared_model_path", None),
            getattr(args, "model_path", None),
            intent_agent_model,
        )
        refine_agent_model_path = _first_not_none(
            getattr(args, "refine_agent_model_path", None),
            getattr(args, "shared_model_path", None),
            getattr(args, "model_path", None),
            refine_agent_model,
        )
        query_agent_model_path = (
            getattr(args, "query_agent_model_path", None) or query_agent_model
        )

        intent_agent_tokenizer = _first_not_none(
            getattr(args, "intent_agent_tokenizer", None),
            getattr(args, "shared_tokenizer", None),
        )
        if intent_agent_use_chat_template and intent_agent_tokenizer is None:
            intent_agent_tokenizer = _load_tokenizer(intent_agent_model_path)

        refine_agent_tokenizer = _first_not_none(
            getattr(args, "refine_agent_tokenizer", None),
            getattr(args, "shared_tokenizer", None),
        )
        if refine_agent_use_chat_template and refine_agent_tokenizer is None:
            refine_agent_tokenizer = _load_tokenizer(refine_agent_model_path)

        query_agent_tokenizer = getattr(args, "query_agent_tokenizer", None)
        if query_agent_use_chat_template and query_agent_tokenizer is None:
            query_agent_tokenizer = _load_tokenizer(query_agent_model_path)

        retriever_client = build_retriever_client_from_args(args)

        intent_agent_llm_client = OpenAIClient(
            base_url=_first_not_none(
                getattr(args, "intent_agent_openai_base_url", None),
                getattr(args, "shared_openai_base_url", None),
                getattr(args, "openai_base_url", None),
                "http://127.0.0.1:8001",
            ),
            model=intent_agent_model,
            api_key=_first_not_none(
                getattr(args, "intent_agent_openai_api_key", None),
                getattr(args, "shared_openai_api_key", None),
                getattr(args, "openai_api_key", None),
                "TEST",
            ),
            timeout=_first_not_none(
                getattr(args, "intent_agent_llm_timeout", None),
                getattr(args, "shared_llm_timeout", None),
                getattr(args, "llm_timeout", None),
            ),
            use_chat_template=intent_agent_use_chat_template,
        )
        refine_agent_llm_client = OpenAIClient(
            base_url=_first_not_none(
                getattr(args, "refine_agent_openai_base_url", None),
                getattr(args, "shared_openai_base_url", None),
                getattr(args, "openai_base_url", None),
                "http://127.0.0.1:8001",
            ),
            model=refine_agent_model,
            api_key=_first_not_none(
                getattr(args, "refine_agent_openai_api_key", None),
                getattr(args, "shared_openai_api_key", None),
                getattr(args, "openai_api_key", None),
                "TEST",
            ),
            timeout=_first_not_none(
                getattr(args, "refine_agent_llm_timeout", None),
                getattr(args, "shared_llm_timeout", None),
                getattr(args, "llm_timeout", None),
            ),
            use_chat_template=refine_agent_use_chat_template,
        )
        query_agent_llm_client = OpenAIClient(
            base_url=_first_not_none(
                getattr(args, "query_agent_openai_base_url", None),
                getattr(args, "intent_agent_openai_base_url", None),
                getattr(args, "shared_openai_base_url", None),
                getattr(args, "openai_base_url", None),
                "http://127.0.0.1:8001",
            ),
            model=query_agent_model,
            api_key=_first_not_none(
                getattr(args, "query_agent_openai_api_key", None),
                getattr(args, "intent_agent_openai_api_key", None),
                getattr(args, "shared_openai_api_key", None),
                getattr(args, "openai_api_key", None),
                "TEST",
            ),
            timeout=_first_not_none(
                getattr(args, "query_agent_llm_timeout", None),
                getattr(args, "intent_agent_llm_timeout", None),
                getattr(args, "shared_llm_timeout", None),
                getattr(args, "llm_timeout", None),
            ),
            use_chat_template=query_agent_use_chat_template,
        )

        intent_agent_max_tokens = int(getattr(args, "intent_agent_max_tokens", 256))
        intent_agent_temperature = float(getattr(args, "intent_agent_temperature", 0.6))
        intent_agent_top_p = float(getattr(args, "intent_agent_top_p", 0.9))

        query_agent_max_tokens = int(getattr(args, "query_agent_max_tokens", 384))
        query_agent_temperature = float(getattr(args, "query_agent_temperature", 0.8))
        query_agent_top_p = float(getattr(args, "query_agent_top_p", 0.95))

        refine_agent_max_tokens = int(
            getattr(args, "refine_agent_max_tokens",
                    getattr(args, "refine_max_tokens", 384))
        )
        refine_agent_temperature = float(
            getattr(args, "refine_agent_temperature",
                    getattr(args, "refine_temperature", 0.2))
        )
        refine_agent_top_p = float(
            getattr(args, "refine_agent_top_p",
                    getattr(args, "refine_top_p", 0.9))
        )

        reason_max_tokens = int(getattr(args, "reason_max_tokens", 768))
        reason_temperature = float(getattr(args, "reason_temperature", 0.3))
        reason_top_p = float(getattr(args, "reason_top_p", 0.9))

        repetition_penalty_arg = getattr(args, "repetition_penalty", None)
        repetition_penalty = (
            float(repetition_penalty_arg) if repetition_penalty_arg is not None else None
        )

        return cls(
            retriever=retriever_client,
            intent_agent_llm_client=intent_agent_llm_client,
            refine_agent_llm_client=refine_agent_llm_client,
            query_agent_llm_client=query_agent_llm_client,
            intent_agent_max_tokens=intent_agent_max_tokens,
            intent_agent_temperature=intent_agent_temperature,
            intent_agent_top_p=intent_agent_top_p,
            query_agent_max_tokens=query_agent_max_tokens,
            query_agent_temperature=query_agent_temperature,
            query_agent_top_p=query_agent_top_p,
            refine_agent_max_tokens=refine_agent_max_tokens,
            refine_agent_temperature=refine_agent_temperature,
            refine_agent_top_p=refine_agent_top_p,
            reason_max_tokens=reason_max_tokens,
            reason_temperature=reason_temperature,
            reason_top_p=reason_top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=parse_stop_tokens(args),
            intent_agent_use_chat_template=intent_agent_use_chat_template,
            intent_agent_tokenizer=intent_agent_tokenizer,
            intent_agent_enable_thinking=intent_agent_enable_thinking,
            refine_agent_use_chat_template=refine_agent_use_chat_template,
            refine_agent_tokenizer=refine_agent_tokenizer,
            refine_agent_enable_thinking=refine_agent_enable_thinking,
            query_agent_use_chat_template=query_agent_use_chat_template,
            query_agent_tokenizer=query_agent_tokenizer,
            query_agent_enable_thinking=query_agent_enable_thinking,
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _make_config(
        self, max_tokens: int, temperature: float, top_p: float
    ) -> SimpleNamespace:
        config = SimpleNamespace(
            max_completion_length=max_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_n=1,
        )
        if getattr(self, "repetition_penalty", None) is not None:
            config.repetition_penalty = self.repetition_penalty
        return config

    def _format_prompt_by_template(
        self, system_prompt: str, user_prompt: str, use_chat_template: bool
    ) -> Any:
        if not use_chat_template:
            return f"{system_prompt}\n\n{user_prompt}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_searches(self, text: str) -> list[str]:
        # Strip the think region so prompt echoes inside it (e.g. the literal
        # "<search>...</search>" instruction) are not mistaken for queries.
        stripped = self._strip_think_region(text)
        searches: list[str] = []
        for matched in SEARCH_TAG_PATTERN.findall(stripped):
            q = matched.strip()
            if not q or self._is_placeholder_value(q):
                continue
            if q not in searches:
                searches.append(q)
        if searches:
            return searches
        recovered = self._recover_malformed_search(stripped)
        if recovered:
            return recovered
        # Stage 3: no usable <search> markup at all (model deliberated on
        # wording until truncation) — mine the committed query from the
        # original text, think region included, where the candidates live.
        return self._mine_query_from_text(text)

    def _mine_query_from_text(self, text: str) -> list[str]:
        """Stage 3 fallback: recover the query a model committed to in its
        think monologue but never wrapped in <search> (token-limit truncation).

        Tries, from strongest to weakest signal:
          1. explicit declarations: ``Query: ...`` / ``Final Query: ...``
          2. commitments: ``I'll use ...`` / ``I will output ...``

        The LAST match wins — models refine wording iteratively, so the final
        mention is the committed choice. Returns at most one query.
        """
        for pattern in (QUERY_DECLARATION_PATTERN, QUERY_COMMITMENT_PATTERN):
            matches = pattern.findall(text)
            for candidate in reversed(matches):
                q = candidate.strip().rstrip(".")
                # Unwrap quotes only when they enclose the whole candidate.
                if len(q) >= 2 and q[0] in "\"'`" and q[-1] == q[0]:
                    q = q[1:-1].strip()
                if self._is_plausible_mined_query(q):
                    return [q]
        return []

    def _is_plausible_mined_query(self, query: str) -> bool:
        """Sanity checks for Stage 3 mined queries (think monologue is noisy)."""
        if not query or self._is_placeholder_value(query):
            return False
        if len(query) > 200 or " " not in query:
            return False
        if query.lower() in {"this", "that", "it"}:
            return False
        if STRUCTURAL_TAG_PATTERN.search(query):
            return False
        if re.search(r"the prompt|experience", query, flags=re.IGNORECASE):
            return False
        return True

    def _recover_malformed_search(self, text: str) -> list[str]:
        """Fallback for outputs with a closing ``</search>`` but a malformed
        opening tag, e.g. ``<Gigi Causey> wedding</search>`` (model used the
        entity name as the tag).

        Takes the last line before the last ``</search>`` and unfolds
        pseudo-tags (``<Gigi Causey> wedding`` -> ``Gigi Causey wedding``) —
        the pseudo-tag content is part of the intended query and must be
        kept. Returns at most one recovered query; empty when the fragment
        fails sanity checks.
        """
        closing_matches = list(re.finditer(r"</search>", text, flags=re.IGNORECASE))
        if not closing_matches:
            return []
        fragment = text[: closing_matches[-1].start()].split("\n")[-1].strip()
        if not fragment or STRUCTURAL_TAG_PATTERN.search(fragment):
            return []
        fragment = re.sub(r"<([^<>\s][^<>]*)>", r"\1", fragment).strip()
        if (
            not fragment
            or self._is_placeholder_value(fragment)
            or len(fragment) > 200
        ):
            return []
        return [fragment]

    def _extract_think(self, text: str) -> str:
        """Extract think content, tolerating missing/malformed think tags.

        Strategy (some models emit only the closing ``</think>`` tag):
          1. If a ``</think>`` closing tag exists, everything before it is
             the think content.
          2. Otherwise, if ``<search_directions>`` or ``<search>`` exists,
             everything before the earliest such tag is the think content.
          3. Otherwise, fall back to the whole output.

        Any residual ``<think>``/``</think>`` tokens are stripped so they
        never leak into downstream agents (e.g. the Refine Agent).
        """
        closing_match = re.search(r"</think>", text, flags=re.IGNORECASE)
        if closing_match:
            think = text[: closing_match.start()]
        else:
            # Intent Agent outputs are followed by <search_directions>, query
            # agent outputs by <search>; cut at the earliest such tag.
            boundary_match = re.search(
                r"<search_directions>|<search>", text, flags=re.IGNORECASE
            )
            think = text[: boundary_match.start()] if boundary_match else text
        # Strip think special tokens so they never reach downstream agents.
        think = re.sub(r"</?think>", "", think, flags=re.IGNORECASE)
        return think.strip()

    def _strip_think_region(self, text: str) -> str:
        """Remove the think region before structural parsing.

        Models frequently echo the prompt's format instructions inside their
        think monologue (e.g. the literal "<answer>...</answer>" or
        "<search_directions>...</search_directions>"). First-match parsing
        would latch onto those echoes; removing the think region first
        guarantees only real structural tags are seen.

        Mirrors the boundary logic of ``_extract_think``:
          1. If ``</think>`` exists, drop everything up to and including the
             first closing tag.
          2. If think was opened but never closed, keep only content from the
             first structural boundary tag onwards (empty when truncated).
          3. Otherwise return the text unchanged.
        """
        closing_match = re.search(r"</think>", text, flags=re.IGNORECASE)
        if closing_match:
            return text[closing_match.end():]
        opening_match = re.search(r"<think>", text, flags=re.IGNORECASE)
        if opening_match:
            boundary_match = re.search(
                r"<search_directions>|<search>|<answer>",
                text[opening_match.end():],
                flags=re.IGNORECASE,
            )
            if boundary_match:
                return text[opening_match.end() + boundary_match.start():]
            return ""  # unclosed think with no structured content (truncated)
        return text

    @staticmethod
    def _is_placeholder_value(value: str) -> bool:
        """True for echo-only values like "...", "…", or empty strings."""
        return PLACEHOLDER_VALUE_PATTERN.fullmatch(value) is not None

    def _parse_answer(self, text: str) -> str:
        """Return the last non-placeholder <answer> outside the think region."""
        text = self._strip_think_region(text)
        answer = ""
        for match in ANSWER_TAG_PATTERN.finditer(text):
            candidate = match.group(1).strip()
            if not candidate or self._is_placeholder_value(candidate):
                continue
            answer = candidate  # keep last valid, matching _extract_last_tag
        return answer

    def _parse_search_directions(self, text: str) -> list[SearchDirection]:
        # Prompt echoes can produce empty/ellipsized blocks before the real
        # one; scan every block and keep the first that yields directions.
        for candidate in (self._strip_think_region(text), text):
            directions = self._parse_directions_from_text(candidate)
            if directions:
                return directions
        return []

    def _parse_directions_from_text(self, text: str) -> list[SearchDirection]:
        for block_match in SEARCH_DIRECTIONS_PATTERN.finditer(text):
            block = block_match.group(1)
            directions: list[SearchDirection] = []
            for idx, match in enumerate(DIRECTION_TAG_PATTERN.finditer(block), start=1):
                direction_id = (match.group(1) or str(idx)).strip() or str(idx)
                direction_text = re.sub(r"\s+", " ", match.group(2)).strip()
                if not direction_text or self._is_placeholder_value(direction_text):
                    continue
                directions.append({
                    "direction_id": direction_id,
                    "direction": direction_text,
                })
            if directions:
                return directions
        return []

    @staticmethod
    def _is_valid_experience(experience: str) -> bool:
        """Filter out junk experiences produced by prompt echoing.

        Drops:
          - placeholder-only values ("...", empty, punctuation-only)
          - bare connective words ("and", "or") from echoed format examples
          - very short fragments that cannot carry a retrieval lesson
          - entries containing structural tags (think/refined_information/
            experience/answer/...), which are meta monologue, not lessons
        """
        if len(experience) < 10:
            return False
        if PLACEHOLDER_VALUE_PATTERN.fullmatch(experience):
            return False
        if re.fullmatch(r"(?i)(and|or|the|a|an)\.?", experience):
            return False
        if STRUCTURAL_TAG_PATTERN.search(experience):
            return False
        return True

    def _parse_retrieval_experiences(self, text: str) -> list[str]:
        """Parse <experience>...</experience> tags from refine agent output."""
        experiences: list[str] = []
        for matched in EXPERIENCE_TAG_PATTERN.findall(text):
            exp = matched.strip()
            if exp and self._is_valid_experience(exp):
                # Deduplicate
                if exp not in experiences:
                    experiences.append(exp)
        # Also try to extract from <experiences> block
        block_match = EXPERIENCES_BLOCK_PATTERN.search(text)
        if block_match:
            for matched in EXPERIENCE_TAG_PATTERN.findall(block_match.group(1)):
                exp = matched.strip()
                if exp and self._is_valid_experience(exp) and exp not in experiences:
                    experiences.append(exp)
        return experiences

    def _parse_refined_information(self, refine_agent_output: str) -> str:
        tagged_match = REFINED_INFORMATION_PATTERN.search(refine_agent_output)
        if tagged_match and tagged_match.group(1).strip():
            return tagged_match.group(1).strip()
        legacy_match = LEGACY_INFORMATION_PATTERN.search(refine_agent_output)
        if legacy_match and legacy_match.group(1).strip():
            return legacy_match.group(1).strip()
        return refine_agent_output.strip()

    def _parse_experience_snapshot(
        self, refine_agent_output: str
    ) -> list[str] | None:
        has_container = EXPERIENCES_BLOCK_PATTERN.search(refine_agent_output) is not None
        matches = EXPERIENCE_TAG_PATTERN.findall(refine_agent_output)
        if not has_container and not matches:
            return None
        experiences: list[str] = []
        for matched in matches:
            experience = re.sub(r"\s+", " ", matched).strip()
            if (
                experience
                and self._is_valid_experience(experience)
                and experience not in experiences
            ):
                experiences.append(experience)
        return experiences

    def _merge_all_experiences(
        self, current_experiences: list[str], refine_agent_output: str
    ) -> list[str]:
        parsed_experiences = self._parse_experience_snapshot(refine_agent_output)
        if parsed_experiences is None:
            return list(current_experiences)
        merged: list[str] = []
        for experience in [*current_experiences, *parsed_experiences]:
            normalized = re.sub(r"\s+", " ", experience).strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged

    def _format_experience_block(self, experiences: list[str]) -> str:
        """Format the accumulated retrieval experience store for inclusion in prompts."""
        if not experiences:
            return "No prior retrieval experiences yet."
        blocks = [
            f"  {i + 1}. {exp}"
            for i, exp in enumerate(experiences)
        ]
        return "\n".join(blocks)

    def _generate_text_batch(
        self,
        prompts: list[Any],
        config: SimpleNamespace,
        llm_client: OpenAIClient,
        tokenizer: Any = None,
        use_chat_template: bool | None = None,
        enable_thinking: bool = False,
    ) -> list[str]:
        if not prompts:
            return []
        active_use_chat_template = (
            self.use_chat_template
            if use_chat_template is None
            else bool(use_chat_template)
        )
        if not active_use_chat_template:
            prompt_texts = [self._prompt_to_text(p) for p in prompts]
            return llm_client.generate_text(prompt_texts, config, self.stop_tokens)
        return llm_client.generate_text(
            prompts, config, self.stop_tokens,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_intent_agent_prompt(
        self,
        question: str,
        historical_refinements: list[str],
        retrieval_experiences: list[str],
        use_chat_template: bool,
    ) -> Prompt:
        history_block = (
            "\n\n".join(
                f"R_{idx + 1}:\n{report}"
                for idx, report in enumerate(historical_refinements)
            )
            if historical_refinements
            else "None"
        )
        experience_block = self._format_experience_block(retrieval_experiences)

        system_prompt = (
            "You are a navigator agent in a multi-stage retrieval reasoning system to answer the user's question. "
            "You receive the original question, the historical global refinement reports, "
            "and accumulated retrieval experiences refined over previous iterations.\n\n"
            "Your task:\n"
            "1. Think globally inside <think>...</think>.\n"
            "2. Carefully assess whether the information already gathered is sufficient for a confident answer. "
            "In the first round, when little or no information has been gathered, consider retrieving useful evidence before answering.\n"
            "3. If the available information is sufficient, use it to output the final concise short answer inside <answer>...</answer>.\n"
            "4. If important information is still missing, output a set of retrieval directions inside "
            "<search_directions>...</search_directions>. "
            'Each direction must be wrapped in <direction id="k">...</direction>.\n\n'
            "IMPORTANT — Learning from accumulated retrieval experiences:\n"
            "Review the accumulated retrieval experiences below. They have been continuously refined across "
            "iterations — capturing what search strategies and query patterns work, what doesn't, "
            "and how to improve. Use these lessons to produce better-targeted directions that "
            "avoid past mistakes and replicate past successes.\n\n"
            "Also use the Historical Refined Information as the evidence already gathered: identify what it "
            "already resolves, what remains uncertain, and whether another search would add useful information.\n\n"
            "CRITICAL: Every direction must be immediately searchable. Use only known entity names — "
            'never use placeholders, variable references, or unresolved references like "[Name]", "[Entity]". '
            "If a direction depends on information from another direction, merge them or omit the dependent direction.\n\n"
            "COMMIT AND PRODUCE OUTPUT: After your <think> reasoning, immediately emit either "
            "<answer>...</answer> or <search_directions>...</search_directions>. Do not re-deliberate, "
            "do not restate these instructions, do not iterate on wording — commit to your first "
            "reasonable plan. Your output MUST end with a complete <answer>...</answer> or "
            "<search_directions>...</search_directions> block."
        )
        user_prompt = "\n\n".join([
            _format_external_context("Original Question", question),
            _format_external_context("Historical Refined Information", history_block),
            _format_external_context(
                "Accumulated Retrieval Experiences", experience_block
            ),
            "Decide whether to answer now or propose the next retrieval directions.",
        ])
        return self._format_prompt_by_template(
            system_prompt, user_prompt, use_chat_template
        )

    def _build_query_agent_prompt(
        self,
        original_question: str,
        intent_agent_think: str,
        direction: SearchDirection,
        retrieval_experiences: list[str],
        use_chat_template: bool,
    ) -> Any:
        experience_block = self._format_experience_block(retrieval_experiences)

        system_prompt = (
            "You are a helpful assistant that helps the Navigator Agent answer the user's question. "
            "You receive the original question, one assigned retrieval direction, "
            "and accumulated retrieval experiences refined over previous iterations.\n\n"
            "Your task is to convert the retrieval direction into one concrete search-engine-friendly query.\n\n"
            "IMPORTANT — Learning from accumulated retrieval experiences:\n"
            "Review the accumulated retrieval experiences below. They have been continuously refined across "
            "iterations — capturing query patterns that yielded useful or non-useful results. "
            "Use these lessons to write a better query for the current direction.\n\n"
            "Rules:\n"
            "- Use only entity names explicitly mentioned in the direction or in the Navigator's thoughts.\n"
            '- Never use placeholders like "[Name]", "[Entity]", or unresolved references.\n'
            "- If the direction depends on an unknown entity, skip and output <search></search> instead.\n"
            "- Reason locally inside <think>...</think>, then output exactly one query inside <search>...</search>.\n"
            "- Do not iterate on query wording or second-guess yourself — commit to your first "
            "reasonable query and output it. Your output MUST end with a complete "
            "<search>...</search> block."
        )
        user_prompt = "\n\n".join([
            _format_external_context("Original Question", original_question),
            _format_external_context(
                f"Assigned Direction {direction['direction_id']}", direction["direction"]
            ),
            _format_external_context("Accumulated Retrieval Experiences", experience_block),
            "Now think and put your search query in <search>...</search>.",
        ])
        return self._format_prompt_by_template(system_prompt, user_prompt, use_chat_template)

    def _build_refine_agent_prompt(
        self,
        question: str,
        intent_agent_think: str,
        directions: list[SearchDirection],
        query_plans: list[QueryAgentIterationRecord],
        pooled_docs: list[PooledDocument],
        current_experiences: list[str],
        use_chat_template: bool,
    ) -> Prompt:
        direction_block = (
            "\n".join(
                f"- Direction {direction['direction_id']}: {direction['direction']}"
                for direction in directions
            )
            if directions
            else "None"
        )
        query_block = (
            "\n".join(
                f"- Search Direction: {plan['direction_id']} | Search Query: {plan['search_query']}"
                for plan in query_plans
            )
            if query_plans
            else "None"
        )
        docs_block = (
            "\n\n".join(
                "\n".join([
                    f"DocID: {doc['doc_id']}",
                    f"Sources: {', '.join(doc['sources'])}",
                    f"Source Directions: {' | '.join(doc['source_directions'])}",
                    f"Source Queries: {' | '.join(doc['source_queries'])}",
                    f"Content: {doc['contents']}",
                ])
                for doc in pooled_docs
            )
            if pooled_docs
            else "No documents in the pooled knowledge base."
        )
        current_experience_block = (
            "\n".join(
                f"{index + 1}. {experience}"
                for index, experience in enumerate(current_experiences)
            )
            if current_experiences
            else "No prior experiences yet."
        )

        system_prompt = (
            "You are a helpful assistant tasked with analyzing the current iteration's search results "
            "and producing BOTH helpful information AND an updated retrieval experience store.\n\n"
            "Your objectives:\n\n"
            "1. EXTRACT HELPFUL INFORMATION FOR THE ANSWER:\n"
            "   Read the searched documents and extract information relevant to the concrete search "
            "queries and the original question. Integrate this into the Navigator Agent's reasoning.\n"
            "   When presenting a fact, prefer attributing it to the concrete search query that retrieved "
            "the supporting document rather than to its DocID, so the retrieval context remains clear.\n"
            "   When documents mention different entities with the same or similar name, preserve their "
            "distinguishing context rather than merging their facts.\n"
            "   Output the complete refined information inside "
            "<refined_information> and </refined_information>.\n"
            "   If no helpful information is found, summarize what was searched and retrieved.\n\n"
            "2. UPDATE THE RETRIEVAL EXPERIENCE STORE (CRITICAL):\n"
            "   Below is the CURRENT experience store accumulated from previous iterations. "
            "Based on THIS iteration's search results, produce a COMPLETE UPDATED experience store.\n\n"
            "   For each existing experience, decide whether to:\n"
            "   - KEEP it as-is (still valid)\n"
            "   - REFINE/MODIFY it based on new evidence from this round\n"
            "   - DELETE it (proved unreliable or no longer relevant)\n\n"
            "   Additionally, ADD new experiences based on what you learned in THIS round:\n"
            "   What query patterns or search strategies worked well? Which ones did not?\n"
            "   What could be done differently next time?\n\n"
            "   Output the COMPLETE UPDATED set of experiences, with each lesson inside "
            "<experience> and </experience>. Include ALL experiences that should be kept "
            "(refined or as-is), not just the new ones. For example:\n"
            '   <experience>Queries with specific entity names (e.g. "Alice Chen") yield more relevant '
            "documents than queries with generic roles (e.g. 'the CEO').</experience>\n"
        )
        user_prompt = "\n\n".join([
            _format_external_context("Original Question", question),
            _format_external_context(
                "Previous Navigator Agent Reasoning", intent_agent_think
            ),
            _format_external_context("Abstract Search Directions", direction_block),
            _format_external_context("Concrete Search Queries", query_block),
            _format_external_context("Global Searched Document Pool", docs_block),
            _format_external_context(
                "Current Retrieval Experience Store (to review and update)",
                current_experience_block,
            ),
        ])
        return self._format_prompt_by_template(
            system_prompt, user_prompt, use_chat_template
        )

    def _build_refinement_appendix(self, report: str) -> str:
        return f"\n<information>\n{report}\n</information>\n"

    # ---- 5. Final answer prompt (unchanged) ----

    def _build_reason_prompt(
        self,
        question: str,
        historical_refinements: list[str],
        retrieval_experiences: list[str],
        use_chat_template: bool,
    ) -> Any:
        history_block = (
            "\n\n".join(
                f"R_{idx + 1}:\n{report}"
                for idx, report in enumerate(historical_refinements)
            )
            if historical_refinements
            else "None"
        )
        experience_block = self._format_experience_block(retrieval_experiences)

        system_prompt = (
            "You are the final answer synthesis agent. "
            "All evidence has already been gathered in the historical global refinement "
            "reports and accumulated retrieval experiences below.\n\n"
            "CRITICAL OUTPUT RULE: The FIRST thing you output must be the answer block "
            "<answer>...</answer>. Do not think out loud, do not restate the question or "
            "these instructions, do not re-analyze the reports — just answer.\n"
            "The answer must be concise: an entity name, date, place, or yes/no — "
            "no explanation, no markdown formatting, no bullet points."
        )
        user_prompt = "\n\n".join([
            _format_external_context("Original Question", question),
            _format_external_context("Historical Global Refinements", history_block),
            _format_external_context("Accumulated Retrieval Experiences", experience_block),
        ])
        return self._format_prompt_by_template(system_prompt, user_prompt, use_chat_template)

    # ------------------------------------------------------------------
    # Document pooling (unchanged from adaptive_parallel_o1)
    # ------------------------------------------------------------------

    def _make_doc_key(self, doc: RetrieverDocument) -> str:
        doc_id = str(doc.get("id", "")).strip()
        if doc_id:
            return doc_id
        contents = str(doc.get("contents", ""))
        return hashlib.sha1(contents.encode("utf-8")).hexdigest()

    def _pool_documents(
        self,
        query_plans: list[QueryAgentIterationRecord],
        docs_map: dict[tuple[int, int], list[RetrieverDocument]],
        sample_i: int,
    ) -> list[PooledDocument]:
        pooled_by_key: dict[str, PooledDocument] = {}
        for plan in query_plans:
            docs = docs_map.get((sample_i, plan["query_id"]), [])
            for doc in docs:
                key = self._make_doc_key(doc)
                contents = str(doc.get("contents", "")).strip()
                if key not in pooled_by_key:
                    pooled_by_key[key] = {
                        "doc_id": str(doc.get("id", "")).strip() or key,
                        "contents": contents,
                        "sources": [plan["direction_id"]],
                        "source_directions": [plan["direction"]],
                        "source_queries": [plan["search_query"]],
                    }
                    continue
                pooled_doc = pooled_by_key[key]
                if plan["direction_id"] not in pooled_doc["sources"]:
                    pooled_doc["sources"].append(plan["direction_id"])
                if plan["direction"] not in pooled_doc["source_directions"]:
                    pooled_doc["source_directions"].append(plan["direction"])
                if plan["search_query"] not in pooled_doc["source_queries"]:
                    pooled_doc["source_queries"].append(plan["search_query"])
        return list(pooled_by_key.values())

    # ------------------------------------------------------------------
    # Runtime initialisation
    # ------------------------------------------------------------------

    def _init_batch_runtime(
        self, questions: list[str], max_iterations: int
    ) -> tuple[list[bool], list[ParallelO1Result], list[list[str]]]:
        completed = [False] * len(questions)
        results: list[ParallelO1Result] = [
            {
                "query": q,
                "max_iterations": max_iterations,
                "executed_iterations": 0,
                "iterations": [],
                "final_reason": {"prompt": "", "output": "", "answer": ""},
                "final_answer": "",
                "refined_information": "",
                "retrieval_experiences": [],
            }
            for q in questions
        ]
        refinement_histories: list[list[str]] = [[] for _ in questions]
        return completed, results, refinement_histories

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, question: str, max_iterations: int = 4) -> ParallelO1Result:
        return self.run_batch([question], max_iterations=max_iterations)[0]

    def run_batch(
        self, questions: list[str], max_iterations: int = 4
    ) -> list[ParallelO1Result]:
        # -------------------------------------------------------------
        # Phase 0: Runtime initialisation
        # -------------------------------------------------------------
        if not questions:
            return []

        logger.info(
            "[maps] run_batch start: n_questions=%d max_iterations=%d "
            "intent_max_tokens=%d query_max_tokens=%d refine_max_tokens=%d rp=%s",
            len(questions), max_iterations,
            self.intent_agent_max_tokens, self.query_agent_max_tokens,
            self.refine_agent_max_tokens, self.repetition_penalty,
        )

        completed, results, refinement_histories = self._init_batch_runtime(
            questions=questions, max_iterations=max_iterations
        )
        latest_intent_thinks = ["" for _ in questions]

        retrieval_experiences_by_sample: list[list[str]] = [[] for _ in questions]

        # Static generation configs
        intent_agent_config = self._make_config(
            max_tokens=self.intent_agent_max_tokens,
            temperature=self.intent_agent_temperature,
            top_p=self.intent_agent_top_p,
        )
        query_config = self._make_config(
            max_tokens=self.query_agent_max_tokens,
            temperature=self.query_agent_temperature,
            top_p=self.query_agent_top_p,
        )
        refine_agent_config = self._make_config(
            max_tokens=self.refine_agent_max_tokens,
            temperature=self.refine_agent_temperature,
            top_p=self.refine_agent_top_p,
        )

        for iteration_idx in range(max_iterations):
            # ---------------------------------------------------------
            # Phase 1: Intent agent planning (with accumulated experiences)
            # ---------------------------------------------------------
            active_indices = [i for i, c in enumerate(completed) if not c]
            if not active_indices:
                break

            logger.info(
                "[maps] iter %d/%d start: active_samples=%d/%d",
                iteration_idx + 1, max_iterations,
                len(active_indices), len(questions),
            )

            active_prompts = [
                self._build_intent_agent_prompt(
                    question=questions[i],
                    historical_refinements=refinement_histories[i],
                    retrieval_experiences=retrieval_experiences_by_sample[i],
                    use_chat_template=self.intent_agent_use_chat_template,
                )
                for i in active_indices
            ]
            intent_agent_outputs = self._generate_text_batch(
                active_prompts,
                intent_agent_config,
                llm_client=self.intent_agent_llm_client,
                tokenizer=self.intent_agent_tokenizer,
                use_chat_template=self.intent_agent_use_chat_template,
                enable_thinking=self.intent_agent_enable_thinking,
            )

            logger.info(
                "[maps] iter %d phase1 intent: %d prompts",
                iteration_idx + 1, len(active_prompts),
            )

            # Collect path-dispatch inputs from Intent Agent outputs
            query_agent_generation_prompts: list[Any] = []
            query_meta: list[tuple[int, int, str, str, str, str]] = []
            query_plans_by_sample: list[list[QueryAgentIterationRecord]] = [
                [] for _ in range(len(questions))
            ]
            query_records_by_sample: list[list[QueryAgentIterationRecord]] = [
                [] for _ in range(len(questions))
            ]
            directions_by_sample: list[list[SearchDirection]] = [
                [] for _ in range(len(questions))
            ]
            iteration_records_by_sample: list[IterationRecord | None] = [
                None for _ in range(len(questions))
            ]

            for local_i, sample_i in enumerate(active_indices):
                intent_agent_output = intent_agent_outputs[local_i]
                intent_prompt_text = self._prompt_to_text(active_prompts[local_i])
                results[sample_i]["executed_iterations"] += 1
                intent_agent_think = self._extract_think(intent_agent_output)
                latest_intent_thinks[sample_i] = intent_agent_think

                answer = self._parse_answer(intent_agent_output)
                directions = self._parse_search_directions(intent_agent_output)
                iteration_record: IterationRecord = {
                    "iteration": iteration_idx + 1,
                    "intent": {
                        "prompt": intent_prompt_text,
                        "output": intent_agent_output,
                        "think": intent_agent_think,
                        "answer": answer,
                        "search_directions": directions,
                    },
                    "query_agents": [],
                }
                results[sample_i]["iterations"].append(iteration_record)
                iteration_records_by_sample[sample_i] = iteration_record

                if answer:
                    results[sample_i]["final_answer"] = answer
                    completed[sample_i] = True
                    continue

                directions_by_sample[sample_i] = directions
                if not directions:
                    continue

                for query_agent_id, direction in enumerate(directions, start=1):
                    query_agent_prompt = self._build_query_agent_prompt(
                        original_question=questions[sample_i],
                        intent_agent_think=intent_agent_think,
                        direction=direction,
                        retrieval_experiences=retrieval_experiences_by_sample[sample_i],
                        use_chat_template=self.query_agent_use_chat_template,
                    )
                    query_agent_generation_prompts.append(query_agent_prompt)
                    query_meta.append((
                        sample_i,
                        query_agent_id,
                        direction["direction_id"],
                        direction["direction"],
                        intent_agent_think,
                        self._prompt_to_text(query_agent_prompt),
                    ))

            n_answered = sum(1 for i in active_indices if completed[i])
            logger.info(
                "[maps] iter %d intent parse: answered=%d directions=%d query_prompts=%d",
                iteration_idx + 1, n_answered,
                sum(len(d) for d in directions_by_sample),
                len(query_agent_generation_prompts),
            )

            if not query_agent_generation_prompts:
                continue

            # ---------------------------------------------------------
            # Phase 2: Path-agent concretisation
            # ---------------------------------------------------------
            query_agent_outputs = self._generate_text_batch(
                query_agent_generation_prompts,
                query_config,
                llm_client=self.query_agent_llm_client,
                tokenizer=self.query_agent_tokenizer,
                use_chat_template=self.query_agent_use_chat_template,
                enable_thinking=self.query_agent_enable_thinking,
            )

            logger.info(
                "[maps] iter %d phase2 query agents: %d prompts",
                iteration_idx + 1, len(query_agent_generation_prompts),
            )

            flat_queries: list[str] = []
            query_meta: list[tuple[int, int]] = []

            for idx, query_agent_output in enumerate(query_agent_outputs):
                (
                    sample_i,
                    query_id,
                    direction_id,
                    direction_text,
                    intent_agent_think,
                    query_prompt_text,
                ) = query_meta[idx]
                query_agent_think = self._extract_think(query_agent_output)
                query_agent_queries = self._extract_searches(query_agent_output)
                query_agent_query = query_agent_queries[0] if query_agent_queries else ""

                query_record: QueryAgentIterationRecord = {
                    "prompt": query_prompt_text,
                    "query_agent_output": query_agent_output,
                    "query_id": query_id,
                    "direction_id": direction_id,
                    "direction": direction_text,
                    "intent_agent_think": intent_agent_think,
                    "query_agent_think": query_agent_think,
                    "search_query": query_agent_query,
                }
                query_records_by_sample[sample_i].append(query_record)

                query_plan: QueryAgentIterationRecord = {
                    "prompt": query_prompt_text,
                    "query_agent_output": query_agent_output,
                    "query_id": query_id,
                    "direction_id": direction_id,
                    "direction": direction_text,
                    "intent_agent_think": intent_agent_think,
                    "query_agent_think": query_agent_think,
                    "search_query": query_agent_query,
                }
                query_plans_by_sample[sample_i].append(query_plan)

                if query_agent_query:
                    flat_queries.append(query_agent_query)
                    query_meta.append((sample_i, query_id))

            for sample_i in range(len(questions)):
                if (
                    iteration_records_by_sample[sample_i] is not None
                    and query_records_by_sample[sample_i]
                ):
                    iteration_record = cast(IterationRecord, iteration_records_by_sample[sample_i])
                    iteration_record["query_agents"] = query_records_by_sample[sample_i]

            # ---------------------------------------------------------
            # Phase 3: Retrieval + global document pooling
            # ---------------------------------------------------------
            flat_docs = self.retriever.batch_search(flat_queries)

            logger.info(
                "[maps] iter %d phase3 retrieval: %d queries",
                iteration_idx + 1, len(flat_queries),
            )

            docs_map: dict[tuple[int, int], list[RetrieverDocument]] = {}
            for idx, docs in enumerate(flat_docs):
                meta = query_meta[idx]
                docs_map[meta] = docs

            refine_prompts: list[Any] = []
            refine_meta: list[tuple[int, str, list[str], list[str]]] = []
            refined_paths_by_sample: list[list[RefinedPathResult]] = [
                [] for _ in range(len(questions))
            ]
            pooled_docs_by_sample: list[list[PooledDocument]] = [
                [] for _ in range(len(questions))
            ]
            retrieved_by_sample: list[BatchSearchDocs] = [
                [] for _ in range(len(questions))
            ]

            # Build one global-refine prompt per active sample
            for sample_i in active_indices:
                sample_query_plans = query_plans_by_sample[sample_i]
                if not sample_query_plans:
                    continue
                sample_filtered_query_plans = [
                    plan for plan in sample_query_plans if plan["search_query"]
                ]
                if not sample_filtered_query_plans:
                    continue

                retrieved_by_sample[sample_i] = [
                    docs_map.get((sample_i, plan["query_id"]), [])
                    for plan in sample_filtered_query_plans
                ]
                pooled_docs = self._pool_documents(
                    sample_filtered_query_plans, docs_map, sample_i
                )
                pooled_docs_by_sample[sample_i] = pooled_docs

                refined_paths_by_sample[sample_i] = [
                    {
                        "query_id": plan["query_id"],
                        "direction_id": plan["direction_id"],
                        "direction": plan["direction"],
                        "think": plan["query_agent_think"],
                        "search_query": plan["search_query"],
                        "refined_information": "",
                        "selected_doc_ids": [
                            doc["doc_id"]
                            for doc in pooled_docs
                            if plan["direction_id"] in doc["sources"]
                        ],
                    }
                    for plan in sample_filtered_query_plans
                ]

                refine_prompt = self._build_refine_agent_prompt(
                    question=questions[sample_i],
                    intent_agent_think=latest_intent_thinks[sample_i],
                    directions=directions_by_sample[sample_i],
                    query_plans=sample_filtered_query_plans,
                    pooled_docs=pooled_docs,
                    current_experiences=retrieval_experiences_by_sample[sample_i],
                    use_chat_template=self.refine_agent_use_chat_template,
                )
                refine_prompts.append(refine_prompt)
                refine_meta.append((
                    sample_i,
                    self._prompt_to_text(refine_prompt),
                    [doc["doc_id"] for doc in pooled_docs],
                    [plan["direction_id"] for plan in sample_filtered_query_plans],
                ))

            # ---------------------------------------------------------
            # Phase 4: Global refine + context update + EXPERIENCE DISTILLATION
            # ---------------------------------------------------------
            refine_agent_outputs = self._generate_text_batch(
                refine_prompts,
                refine_agent_config,
                llm_client=self.refine_agent_llm_client,
                tokenizer=self.refine_agent_tokenizer,
                use_chat_template=self.refine_agent_use_chat_template,
                enable_thinking=self.refine_agent_enable_thinking,
            )
            logger.info(
                "[maps] iter %d phase4 refine: %d prompts",
                iteration_idx + 1, len(refine_prompts),
            )

            for idx, refine_agent_output in enumerate(refine_agent_outputs):
                sample_i, refine_prompt_text, _, _ = refine_meta[idx]

                refined_information = self._parse_refined_information(refine_agent_output)
                updated_experiences = self._merge_all_experiences(
                    retrieval_experiences_by_sample[sample_i], refine_agent_output
                )
                retrieval_experiences_by_sample[sample_i] = updated_experiences
                results[sample_i]["refined_information"] = refined_information
                results[sample_i]["retrieval_experiences"] = updated_experiences

                # Store refine record in iteration
                if iteration_records_by_sample[sample_i] is not None:
                    iteration_record = cast(IterationRecord, iteration_records_by_sample[sample_i])
                    refine_record: RefineIterationRecord = {
                        "prompt": refine_prompt_text,
                        "refine_agent_output": refine_agent_output,
                        "refined_information": refined_information,
                        "retrieval_experiences": updated_experiences,
                        "pooled_docs": pooled_docs_by_sample[sample_i],
                        "refined_paths": refined_paths_by_sample[sample_i],
                    }
                    iteration_record["refine"] = refine_record
                refinement_histories[sample_i].append(refined_information)

            logger.info(
                "[maps] iter %d done: completed=%d/%d",
                iteration_idx + 1, sum(completed), len(questions),
            )

        # -------------------------------------------------------------
        # Phase 5: Final answer synthesis
        # -------------------------------------------------------------
        n_to_reason = sum(1 for r in results if not r["final_answer"])
        logger.info(
            "[maps] phase5 reason: %d samples need final answer",
            n_to_reason,
        )
        for idx, result in enumerate(results):
            if result["final_answer"]:
                continue

            final_prompt = self._build_reason_prompt(
                question=questions[idx],
                historical_refinements=refinement_histories[idx],
                retrieval_experiences=retrieval_experiences_by_sample[idx],
                use_chat_template=self.intent_agent_use_chat_template,
            )
            final_prompt_text = self._prompt_to_text(final_prompt)
            final_outputs = self._generate_text_batch(
                [final_prompt],
                self._make_config(
                    max_tokens=self.reason_max_tokens,
                    temperature=self.reason_temperature,
                    top_p=self.reason_top_p,
                ),
                llm_client=self.intent_agent_llm_client,
                tokenizer=self.intent_agent_tokenizer,
                use_chat_template=self.intent_agent_use_chat_template,
                enable_thinking=self.intent_agent_enable_thinking,
            )
            final_output = final_outputs[0] if final_outputs else ""
            parsed = self._parse_answer(final_output)
            final_answer = parsed if parsed else (final_output.strip() if final_output else "")
            result["final_reason"] = {
                "prompt": final_prompt_text,
                "output": final_output,
                "answer": final_answer,
            }
            result["final_answer"] = final_answer

        n_completed = sum(1 for r in results if r["final_answer"])
        logger.info(
            "[maps] run_batch done: %d/%d completed",
            n_completed, len(questions),
        )

        return results
