# MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in Retrieval-Augmented Generation

Official implementation of **MAPS**, a multi-agent, multi-round retrieval-augmented
reasoning framework that regulates *search momentum* by maintaining an explicit,
continuously refined.

Each round, MAPS decomposes the search problem into a set of abstract
**search directions**, concretizes them into parallel **search queries**, pools and
de-duplicates the retrieved evidence, and then jointly produces (i) refined
information.
trajectory self-corrects over rounds instead of drifting.


---

## 🎯 Method Overview

Let $q_0$ be the original query, $t = 1,\dots,T$ the iteration index, and $N$ the number
of parallel retrieval directions. MAPS carries two pieces of state across iterations:
the **knowledge observations** $O_{t-1} = (o_1,\dots,o_{t-1})$ and the **refined
reasoning state** $\tilde{r}_{t-1}$.

### Parallel reasoning for search direction calibration

Each iteration proposes $N$ complementary directions, realizes each as exactly one
sub-query, and retrieves them all in a single batch:

$$
\{s^1_t, \dots, s^N_t\} = M_{\mathrm{Intent}}\left(q_0, O_{t-1}, \tilde{r}_{t-1}\right)
$$

$$
q^i_t = M_{\mathrm{Query}}\left(q_0, s^i_t, \tilde{r}_{t-1}\right), \qquad Q_t = \{q^1_t, \dots, q^N_t\}
$$

$$
D_t = \bigcup_{q^i_t \in Q_t} \mathrm{TopK}\left(q^i_t, \mathcal{D}\right)
$$

### Refinement distillation

$$
o_t,\; \tilde{r}_t = M_{\mathrm{Refine}}\left(q_0, \{s^1_t, \dots, s^N_t\}, Q_t, D_t, O_{t-1}, \tilde{r}_{t-1}\right)
$$

The refine agent returns the knowledge observation $o_t$

### Answering

The intent agent may terminate early; otherwise, after $T$ iterations the reasoning
model produces the final answer

$$
y = M_{\mathrm{Reason}}\left(q_0, O_T, \tilde{r}_T\right).
$$

---

## ⚙️ Installation

MAPS talks to two HTTP services (an OpenAI-compatible LLM server and the retrieval
server), so the client-side install is lightweight:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the full research environment (vLLM, FlashRAG, PyTorch,
FAISS-GPU, etc.) used to reproduce the paper. It targets **Linux + NVIDIA GPUs +
CUDA 12.x**. If you only need to run inference against already-running services, the
minimal client dependencies are `requests`, `tqdm`, `transformers`, and `openai`.

---

## 🚀 Quick Start

### 1️⃣ Start the retrieval service

```bash
# Edit retrieval_server/retriever_config.yaml first: set your embedding model,
# FAISS index and corpus paths.
bash scripts/retrieval_server.sh
# -> GET /health, POST /search, POST /batch_search on port 9100
```

### 2️⃣ Start an LLM server

Serve a model with vLLM (see `scripts/vllm_server.sh` for example commands):

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve <MODEL> --port 9101 -tp 4 -dp 1 \
  --gpu-memory-utilization 0.7 --api-key TEST
```

MAPS supports per-agent model/endpoint overrides, so the Intent, Query, Refine
and Reason roles can each point at a different server. By default
they all share the endpoint given by `--openai_base_url`.

### 3️⃣ Run MAPS

```bash
python scripts/run_maps.py \
  --input_file /path/to/<dataset>/test.jsonl \
  --openai_base_url http://127.0.0.1:9101/ \
  --openai_api_key TEST \
  --model <MODEL> \
  --retriever_base_url http://127.0.0.1:9100 \
  --max_iterations 5 \
  --batch_size 512 \
  --num_samples 1024
```

Or run the whole suite:

```bash
bash scripts/run_maps_all.sh
```

Or boot the vLLM + retrieval servers and run an experiment in one shot. Every
setting can be given as an environment variable (default) or as a CLI flag
(which overrides it):

```bash
# run with the defaults
bash scripts/pipeline.sh

# override model / port / GPUs
bash scripts/pipeline.sh --model Qwen3.6-35B-A3B --port 9101 --gpu-ids "4,5,6,7"

# skip a server that is already running elsewhere
bash scripts/pipeline.sh --no-vllm

bash scripts/pipeline.sh --help
```

`pipeline.sh` reuses `scripts/retrieval_server.sh` for the retrieval service and
starts vLLM following the commands in `scripts/vllm_server.sh`. It waits until
both ports are open, exports `OPENAI_BASE_URL`, `OPENAI_API_KEY` and
`RETRIEVER_BASE_URL` to `RUN_SCRIPT`, and shuts the servers down on exit.

---

## 📄 Input Data Format

`--input_file` is a JSON Lines file with one object per line:

```json
{"question": "...", "golden_answers": ["..."]}
```

The output directory is derived from the input path as
`outputs/maps/<model>/<dataset>/<n>/`, where `<n>` auto-increments so repeated runs
never overwrite each other. Unless `--result_file` is given, the runner writes:

| File | Contents |
|---|---|
| `<dataset>.jsonl` | One record per sample: `index`, `question`, `golden_answers`, `predicted_answer_in_tag`, `pipeline_result` (full per-iteration trace) |
| `config.json` | Snapshot of the CLI arguments |
| `run.log` | Run log |

---

## 🛠️ Key CLI Options

| Option | Default | Description |
|---|---|---|
| `--input_file` | required | Input JSONL test file |
| `--openai_base_url` | `http://127.0.0.1:8001` | Shared OpenAI-compatible endpoint |
| `--model` | — | Shared model name |
| `--retriever_base_url` | `http://127.0.0.1:9100` | Retrieval service endpoint |
| `--retriever_top_k` | `5` | Documents retrieved per query |
| `--max_iterations` | `5` | Maximum intent/refine rounds |
| `--batch_size` | `512` | Samples processed per batch |
| `--num_samples` | all | Truncate the input to the first N samples |
| `--use_chat_template` | off | Render prompts with the tokenizer chat template |
| `--enable_thinking` | off | Enable thinking mode (e.g. Qwen3) |

Per-agent overrides follow the pattern
`--{intent_agent,query_agent,refine_agent}_{model,openai_base_url,openai_api_key,model_path,max_tokens,temperature,top_p,llm_timeout,use_chat_template,enable_thinking}`,
falling back to the shared values.

Run `python scripts/run_maps.py --help` for the complete list.

---

## 🔍 Retrieval Service

`retrieval_server/` is a small FastAPI wrapper around a FlashRAG dense retriever:

- `GET /health` → `{"status", "retrievers": {...}}`
- `POST /search` → single query, top-N documents (`{"id", "contents"}`)
- `POST /batch_search` → list of queries → list of document lists

See [`retrieval_server/README.md`](retrieval_server/README.md) for details. Any
retriever exposing the same `/batch_search` contract can be used as a drop-in
replacement.

---

## 📚 Data and Models

### Downloading the assets

`scripts/download_assets.sh` downloads the datasets and models below. It probes
`huggingface.co` first and automatically falls back to **ModelScope** when
Hugging Face is unreachable, and it installs the `huggingface_hub` /
`modelscope` libraries if they are missing:

```bash
bash scripts/download_assets.sh                                    # auto backend
bash scripts/download_assets.sh --backend modelscope               # force ModelScope
bash scripts/download_assets.sh --datasets "nq hotpotqa" --models "Qwen3-32B"
bash scripts/download_assets.sh --include "nq/*" --revision master
bash scripts/download_assets.sh --help
```

Datasets land in `${DATA_ROOT}/<dataset>/test.jsonl` and models in
`${MODEL_ROOT}/<ModelName>`, where `DATA_ROOT` defaults to `$HOME/llm/datasets`
and `MODEL_ROOT` to `$HOME/llm/models` — the same paths that
`scripts/run_maps_all.sh` reads.

ModelScope mirrors: [FlashRAG datasets](https://www.modelscope.cn/datasets/RUC-NLPIR/FlashRAG_datasets) ·
[Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B) ·
[Qwen3.6-35B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B) ·
[e5-base-v2](https://www.modelscope.cn/models/intfloat/e5-base-v2).

### Retrieval corpus, index and datasets

The benchmark datasets, the Wikipedia retrieval corpus and a **prebuilt FAISS
index** all ship in the same [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
dataset repository, under `retrieval-corpus/`:

| File | Contents |
|---|---|
| `retrieval-corpus/e5_flat_inner.index` | FAISS inner-product index over the corpus (built with e5-base-v2) |
| `retrieval-corpus/wiki18_100w.zip` | zipped `wiki18_100w.jsonl`, the ~18M-passage Wikipedia corpus |

- Hugging Face — <https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus>
- ModelScope — <https://www.modelscope.cn/datasets/RUC-NLPIR/FlashRAG_datasets/tree/master/retrieval-corpus>

`scripts/download_assets.sh` downloads the repository into `${DATA_ROOT}`, so
these files land in `${DATA_ROOT}/retrieval-corpus/` (when you narrow the download
with `--include`, add `--with-retrieval-corpus` to keep them). Unzip the corpus:

```bash
DATA_ROOT="${DATA_ROOT:-$HOME/llm/datasets}"
unzip -o "${DATA_ROOT}/retrieval-corpus/wiki18_100w.zip" -d "${DATA_ROOT}/retrieval-corpus"
```

Then point `retrieval_server/retriever_config.yaml` at the local files — with the
default `DATA_ROOT=$HOME/llm/datasets` (expand `$HOME` to an absolute path, the
config does not expand environment variables):

```yaml
retrieval_method: dense
retrieval_model_path: /path/to/e5-base-v2                              # intfloat/e5-base-v2
index_path: $HOME/llm/datasets/retrieval-corpus/e5_flat_inner.index
corpus_path: $HOME/llm/datasets/retrieval-corpus/wiki18_100w.jsonl
```

The benchmark `test.jsonl` files consumed by `scripts/run_maps_all.sh` live
alongside it as `${DATA_ROOT}/<dataset>/test.jsonl`.

### Embedding model

Dense retrieval is performed with **e5-base-v2**:

- Hugging Face — <https://huggingface.co/intfloat/e5-base-v2>
- ModelScope — <https://www.modelscope.cn/models/intfloat/e5-base-v2>

### Backbone models

MAPS is evaluated with two Qwen3 backbones:

- **Qwen/Qwen3-32B** — <https://huggingface.co/Qwen/Qwen3-32B>
- **Qwen/Qwen3.6-35B-A3B** — <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>

`scripts/vllm_server.sh` contains ready-to-edit `vllm serve` commands for both.

---


## ⚖️ License

Released under the [MIT License](LICENSE).

---

## 🔗 Relevant Resources

| Resource | Hugging Face | ModelScope |
|---|---|---|
| FlashRAG datasets (benchmark data + Wikipedia retrieval corpus) | <https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets> | <https://www.modelscope.cn/datasets/RUC-NLPIR/FlashRAG_datasets> |
| Embedding model — e5-base-v2 | <https://huggingface.co/intfloat/e5-base-v2> | <https://www.modelscope.cn/models/intfloat/e5-base-v2> |
| Backbone model — Qwen3-32B | <https://huggingface.co/Qwen/Qwen3-32B> | <https://www.modelscope.cn/models/Qwen/Qwen3-32B> |
| Backbone model — Qwen3.6-35B-A3B | <https://huggingface.co/Qwen/Qwen3.6-35B-A3B> | <https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B> |

