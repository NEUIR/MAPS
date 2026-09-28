import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from maps import MAPS
from maps.utils import AsyncJsonlWriter, build_output_dir, read_jsonlines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in Retrieval-Augmented Generation — runner"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input jsonlines test file",
    )
    parser.add_argument(
        "--result_file", type=str, default=None, help="Path to results jsonlines file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for batched testing"
    )

    parser.add_argument(
        "--retriever_base_url", type=str, default="http://127.0.0.1:9100"
    )
    parser.add_argument("--retriever_top_k", type=int, default=5)
    parser.add_argument("--retriever_timeout", type=float, default=None)

    parser.add_argument(
        "--intent_agent_model",
        type=str,
        default=None,
        help="Model used by intent agent",
    )
    parser.add_argument(
        "--intent_agent_openai_base_url",
        type=str,
        default=None,
        help="Optional dedicated OpenAI base URL for intent agent client",
    )
    parser.add_argument(
        "--intent_agent_openai_api_key",
        type=str,
        default=None,
        help="Optional dedicated OpenAI API key for intent agent client",
    )
    parser.add_argument(
        "--intent_agent_llm_timeout",
        type=float,
        default=None,
        help="Optional dedicated timeout for intent agent client",
    )
    parser.add_argument(
        "--intent_agent_model_path",
        type=str,
        default=None,
        help="Tokenizer source for intent model when using chat template",
    )
    parser.add_argument(
        "--intent_agent_use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether intent agent should use chat template; default inherits use_chat_template",
    )
    parser.add_argument(
        "--intent_agent_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether intent agent enables thinking mode; default inherits shared/legacy flag",
    )

    parser.add_argument(
        "--refine_agent_model",
        type=str,
        default=None,
        help="Model used by refine agent",
    )
    parser.add_argument(
        "--refine_agent_openai_base_url",
        type=str,
        default=None,
        help="Optional dedicated OpenAI base URL for refine agent client",
    )
    parser.add_argument(
        "--refine_agent_openai_api_key",
        type=str,
        default=None,
        help="Optional dedicated OpenAI API key for refine agent client",
    )
    parser.add_argument(
        "--refine_agent_llm_timeout",
        type=float,
        default=None,
        help="Optional dedicated timeout for refine agent client",
    )
    parser.add_argument(
        "--refine_agent_model_path",
        type=str,
        default=None,
        help="Tokenizer source for refine model when using chat template",
    )
    parser.add_argument(
        "--refine_agent_use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether refine agent should use chat template; default inherits intent_agent_use_chat_template",
    )
    parser.add_argument(
        "--refine_agent_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether refine agent enables thinking mode; default inherits intent_agent_enable_thinking",
    )

    parser.add_argument(
        "--query_agent_model",
        type=str,
        default=None,
        help="Model used by query agent",
    )
    parser.add_argument(
        "--query_agent_openai_base_url",
        type=str,
        default=None,
        help="Optional dedicated OpenAI base URL for query agent client",
    )
    parser.add_argument(
        "--query_agent_openai_api_key",
        type=str,
        default=None,
        help="Optional dedicated OpenAI API key for query agent client",
    )
    parser.add_argument(
        "--query_agent_llm_timeout",
        type=float,
        default=None,
        help="Optional dedicated timeout for query agent client",
    )
    parser.add_argument(
        "--query_agent_model_path",
        type=str,
        default=None,
        help="Tokenizer source for query model when using chat template",
    )
    parser.add_argument(
        "--query_agent_use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether query agent should use chat template; default inherits intent_agent_use_chat_template",
    )
    parser.add_argument(
        "--query_agent_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether query agent enables thinking mode; default inherits intent_agent_enable_thinking",
    )

    parser.add_argument("--intent_agent_max_tokens", type=int, default=256)
    parser.add_argument("--intent_agent_temperature", type=float, default=0.6)
    parser.add_argument("--intent_agent_top_p", type=float, default=0.9)

    parser.add_argument("--query_agent_max_tokens", type=int, default=512)
    parser.add_argument("--query_agent_temperature", type=float, default=0.8)
    parser.add_argument("--query_agent_top_p", type=float, default=0.9)

    parser.add_argument("--refine_agent_max_tokens", type=int, default=1024)
    parser.add_argument("--refine_agent_temperature", type=float, default=0.8)
    parser.add_argument("--refine_agent_top_p", type=float, default=0.9)

    # Backward-compatible aliases.
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--llm_timeout", type=float, default=None)
    parser.add_argument("--use_chat_template", action="store_true", default=None)
    parser.add_argument(
        "--enable_thinking", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--shared_openai_base_url", type=str, default=None)
    parser.add_argument("--shared_openai_api_key", type=str, default=None)
    parser.add_argument("--shared_llm_timeout", type=float, default=None)
    parser.add_argument("--shared_model", type=str, default=None)
    parser.add_argument("--shared_model_path", type=str, default=None)
    parser.add_argument(
        "--shared_use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--shared_enable_thinking", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--refine_max_tokens", type=int, default=None)
    parser.add_argument("--refine_temperature", type=float, default=None)
    parser.add_argument("--refine_top_p", type=float, default=None)
    parser.add_argument(
        "--repetition_penalty", type=float, default=None,
        help="Repetition penalty for all generation calls (vLLM sampling param)",
    )

    parser.add_argument(
        "--stop_tokens", type=str, default=None
    )
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=None)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_top_dir", default="outputs", type=str)
    return parser


def main() -> None:
    parser = build_parser()
    parser.description = "MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in Retrieval-Augmented Generation — runner"
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-based index of the first sample to use; takes num_samples samples starting from here",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file (default: <output_dir>/run.log)",
    )
    args = parser.parse_args()
    input_file_path = Path(args.input_file).expanduser()

    output_dir = build_output_dir(
        input_file_path,
        method_name="maps",
        model_name=args.intent_agent_model,
        top_dir=args.output_top_dir,
    )
    dataset_name = input_file_path.parent.name

    if args.log_file:
        log_path = args.log_file
    elif args.result_file:
        log_path = str(Path(args.result_file).expanduser().parent / "run.log")
    else:
        log_path = str(output_dir / "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=log_path,
        filemode="a",
    )

    samples = read_jsonlines(input_file_path)
    start = max(0, args.start)
    if args.num_samples is not None:
        samples = samples[start : start + max(0, args.num_samples)]
    else:
        samples = samples[start:]

    pipeline = MAPS.from_args(args)

    if args.result_file:
        output_file_path = Path(args.result_file).expanduser()
    else:
        output_file_path = output_dir / f"{dataset_name}.jsonl"
        config_file_path = output_dir / "config.json"
        with config_file_path.open("w", encoding="utf-8") as config_file:
            json.dump(vars(args), config_file, ensure_ascii=False, indent=0)
    writer = AsyncJsonlWriter(output_file_path)
    sample_pbar = tqdm(total=len(samples), desc="Samples", unit="sample")

    batch_size = max(1, int(args.batch_size))
    batch_starts = range(0, len(samples), batch_size)

    if args.debug:
        import debugpy

        debugpy.listen(9999)
        print("waiting for debugger attaching")
        debugpy.wait_for_client()
        print("debugger attached")
        print("break point")
        debugpy.breakpoint()

    for batch_start in tqdm(batch_starts, desc="Processing batches", unit="batch"):
        batch_samples = samples[batch_start : batch_start + batch_size]
        questions: list[str] = []
        validated_items: list[tuple[int, str, list[str]]] = []
        for offset, sample in enumerate(batch_samples):
            idx = start + batch_start + offset
            question = sample.get("question", "")
            golden_answers = sample.get("golden_answers", [])
            validated_items.append((idx, question, golden_answers))
            questions.append(question)

        batch_results = [
            dict(item) for item in pipeline.run_batch(questions, args.max_iterations)
        ]

        for (idx, question, golden_answers), result in zip(
            validated_items, batch_results, strict=False
        ):
            final_answer = result.get("final_answer", "")
            output_record = {
                    "index": idx,
                    "question": question,
                    "golden_answers": golden_answers,
                    "predicted_answer_in_tag": final_answer,
                    "pipeline_result": result,
                }
            writer.submit(output_record)
            sample_pbar.update(1)

    sample_pbar.close()
    writer.close()


if __name__ == "__main__":
    main()