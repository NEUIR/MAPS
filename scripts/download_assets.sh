#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Download the datasets and models required by MAPS.
#
# Backend selection:
#   - If huggingface.co is reachable  -> Hugging Face (huggingface_hub)
#   - Otherwise                       -> ModelScope (modelscope)
#   (force with --backend hf | modelscope)
#
# Required python packages are detected and installed on demand:
#   huggingface_hub, modelscope
#
# Layout (identical to scripts/run_maps_all.sh, so the runner works as-is):
#   datasets -> ${DATA_ROOT}/<dataset>/test.jsonl
#   models   -> ${MODEL_ROOT}/<ModelName>
#
# Usage:
#   bash scripts/download_assets.sh
#   bash scripts/download_assets.sh --backend modelscope
#   bash scripts/download_assets.sh --datasets "nq hotpotqa" --models "Qwen3-32B"
#   bash scripts/download_assets.sh --include "nq/*" "hotpotqa/*"
#   bash scripts/download_assets.sh --skip-datasets
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- paths (must match scripts/run_maps_all.sh) ---
DATA_ROOT="${DATA_ROOT:-$HOME/llm/datasets}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/llm/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- assets ---
DATASETS="${DATASETS:-2wikimultihopqa hotpotqa musique gpqa popqa nq}"
MODELS="${MODELS:-Qwen3-32B Qwen3.6-35B-A3B e5-base-v2}"

# --- behaviour ---
BACKEND="${BACKEND:-auto}"          # auto | hf | modelscope
HF_TIMEOUT="${HF_TIMEOUT:-5}"       # seconds, connectivity probe
DO_INSTALL="${DO_INSTALL:-1}"       # 1 = pip install missing libs
SKIP_DATASETS="${SKIP_DATASETS:-0}"
SKIP_MODELS="${SKIP_MODELS:-0}"
REVISION="${REVISION:-}"
INCLUDE="${INCLUDE:-}"
EXCLUDE="${EXCLUDE:-}"
WITH_CORPUS="${WITH_CORPUS:-0}"

usage() {
  cat <<EOF
Usage: bash scripts/download_assets.sh [OPTIONS]

Downloads the MAPS datasets and models, preferring Hugging Face and falling
back to ModelScope when huggingface.co is unreachable.

Options:
  --backend auto|hf|modelscope   Download backend (env: BACKEND, default: auto)
  --data-root PATH               Dataset root (env: DATA_ROOT, default: \$HOME/llm/datasets)
  --model-root PATH              Model root   (env: MODEL_ROOT, default: \$HOME/llm/models)
  --datasets "a b c"             Dataset names to verify afterwards (env: DATASETS)
  --models "a b c"               Models to download (env: MODELS)
  --include "glob ..."           Dataset files to download, glob patterns
                                 (env: INCLUDE; HF allow_patterns / ModelScope --include)
  --exclude "glob ..."           Dataset files to skip (env: EXCLUDE)
                                 --include/--exclude apply to the dataset download only
  --with-retrieval-corpus        When --include narrows the download, also fetch
                                 retrieval-corpus/ (prebuilt FAISS index + Wikipedia corpus)
  --revision REV                 Branch / tag / commit (env: REVISION)
  --hf-timeout SEC               Connectivity probe timeout (env: HF_TIMEOUT, default: ${HF_TIMEOUT})
  --no-install                   Do not pip-install missing libraries
  --skip-datasets                Do not download datasets
  --skip-models                  Do not download models
  -h, --help                     Show this help

Repositories:
  Hugging Face : RUC-NLPIR/FlashRAG_datasets, Qwen/Qwen3-32B,
                 Qwen/Qwen3.6-35B-A3B, intfloat/e5-base-v2
  ModelScope   : RUC-NLPIR/FlashRAG_datasets, Qwen/Qwen3-32B,
                 Qwen/Qwen3.6-35B-A3B, intfloat/e5-base-v2

ModelScope downloads use its CLI:
  modelscope download --dataset RUC-NLPIR/FlashRAG_datasets --local_dir <dir>
  modelscope download --model   Qwen/Qwen3-32B               --local_dir <dir>

After this script finishes you still need the retrieval index: build the FAISS
index over the Wikipedia corpus with FlashRAG and point
retrieval_server/retriever_config.yaml at it.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) shift; BACKEND="${1:?--backend requires a value}"; shift ;;
    --data-root) shift; DATA_ROOT="${1:?--data-root requires a value}"; shift ;;
    --model-root) shift; MODEL_ROOT="${1:?--model-root requires a value}"; shift ;;
    --datasets) shift; DATASETS="${1:?--datasets requires a value}"; shift ;;
    --models) shift; MODELS="${1:?--models requires a value}"; shift ;;
    --include) shift; INCLUDE="${1:?--include requires a value}"; shift ;;
    --exclude) shift; EXCLUDE="${1:?--exclude requires a value}"; shift ;;
    --with-retrieval-corpus) WITH_CORPUS=1; shift ;;
    --revision) shift; REVISION="${1:?--revision requires a value}"; shift ;;
    --hf-timeout) shift; HF_TIMEOUT="${1:?--hf-timeout requires a value}"; shift ;;
    --no-install) DO_INSTALL=0; shift ;;
    --skip-datasets) SKIP_DATASETS=1; shift ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[download] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[download] ERROR: python interpreter '${PYTHON_BIN}' not found." >&2
  echo "[download]        set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

export DATA_ROOT MODEL_ROOT BACKEND HF_TIMEOUT DO_INSTALL
export ASSET_DATASETS="${DATASETS}"
export ASSET_MODELS="${MODELS}"
export SKIP_DATASETS SKIP_MODELS REVISION INCLUDE EXCLUDE WITH_CORPUS

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ["DATA_ROOT"]).expanduser()
MODEL_ROOT = Path(os.environ["MODEL_ROOT"]).expanduser()
BACKEND = os.environ.get("BACKEND", "auto")
HF_TIMEOUT = float(os.environ.get("HF_TIMEOUT", "5"))
DO_INSTALL = os.environ.get("DO_INSTALL", "1") == "1"
DATASETS = os.environ["ASSET_DATASETS"].split()
MODELS = os.environ["ASSET_MODELS"].split()
SKIP_DATASETS = os.environ.get("SKIP_DATASETS", "0") == "1"
SKIP_MODELS = os.environ.get("SKIP_MODELS", "0") == "1"
REVISION = os.environ.get("REVISION", "").strip() or None
INCLUDE = os.environ.get("INCLUDE", "").split()
EXCLUDE = os.environ.get("EXCLUDE", "").split()
WITH_CORPUS = os.environ.get("WITH_CORPUS", "0") == "1"

# retrieval-corpus/ ships inside the dataset repository and holds the prebuilt
# FAISS index plus the Wikipedia corpus. An unfiltered download already
# contains it; when --include narrows the download, add it explicitly.
CORPUS_PATTERN = "retrieval-corpus/*"
if WITH_CORPUS and INCLUDE and CORPUS_PATTERN not in INCLUDE:
    INCLUDE.append(CORPUS_PATTERN)

HF_DATASET_ID = "RUC-NLPIR/FlashRAG_datasets"
MS_DATASET_ID = "RUC-NLPIR/FlashRAG_datasets"
HF_MODEL_IDS = {
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
    "e5-base-v2": "intfloat/e5-base-v2",
}
MS_MODEL_IDS = {
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
    "e5-base-v2": "intfloat/e5-base-v2",
}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Library detection / installation
# ---------------------------------------------------------------------------
def ensure_pkg(module: str, pip_name: str) -> bool:
    if importlib.util.find_spec(module) is not None:
        log(f"[lib] {pip_name}: already installed")
        return True
    if not DO_INSTALL:
        log(f"[lib] {pip_name}: MISSING (--no-install given, not installing)")
        return False
    log(f"[lib] {pip_name}: not found -> pip install -U {pip_name}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pip_name])
    importlib.invalidate_caches()
    ok = importlib.util.find_spec(module) is not None
    log(f"[lib] {pip_name}: {'installed' if ok else 'STILL MISSING'}")
    return ok


# ---------------------------------------------------------------------------
# Connectivity probe
# ---------------------------------------------------------------------------
def hf_reachable() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=HF_TIMEOUT).close()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# ModelScope CLI resolution
# ---------------------------------------------------------------------------
def ms_cli() -> str:
    """Locate the `modelscope` console script (prefer the one beside sys.executable)."""
    sibling = Path(sys.executable).resolve().parent / "modelscope"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("modelscope")
    if found:
        return found
    raise RuntimeError(
        "the 'modelscope' CLI was not found. Install it with:\n"
        f"  {sys.executable} -m pip install -U modelscope"
    )


def ms_download(repo_id: str, target: Path, *, dataset: bool) -> None:
    cmd = [ms_cli(), "download", "--dataset" if dataset else "--model", repo_id]
    cmd += ["--local_dir", str(target)]
    if REVISION:
        cmd += ["--revision", REVISION]
    if dataset:                       # file filters only make sense for datasets
        for pattern in INCLUDE:
            cmd += ["--include", pattern]
        for pattern in EXCLUDE:
            cmd += ["--exclude", pattern]
    target.mkdir(parents=True, exist_ok=True)
    log(f"[ms] {' '.join(cmd)}")
    subprocess.check_call(cmd)


# ---------------------------------------------------------------------------
# Hugging Face download
# ---------------------------------------------------------------------------
def hf_download(repo_id: str, target: Path, *, dataset: bool = False) -> None:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    kwargs = {"repo_id": repo_id, "local_dir": str(target)}
    if dataset:
        kwargs["repo_type"] = "dataset"
    if REVISION:
        kwargs["revision"] = REVISION
    if dataset:                       # file filters only make sense for datasets
        if INCLUDE:
            kwargs["allow_patterns"] = INCLUDE
        if EXCLUDE:
            kwargs["ignore_patterns"] = EXCLUDE
    log(f"[hf] snapshot_download({repo_id!r}, dataset={dataset}) -> {target}")
    snapshot_download(**kwargs)


# ---------------------------------------------------------------------------
# Post-download verification (must match scripts/run_maps_all.sh)
# ---------------------------------------------------------------------------
def verify_dataset(name: str) -> None:
    expected = DATA_ROOT / name / "test.jsonl"
    if expected.is_file():
        log(f"[ok]   dataset {name}: {expected}")
        return
    nested = sorted(DATA_ROOT.glob(f"*/{name}/test.jsonl"))
    if nested:
        log(f"[warn] dataset {name}: expected {expected}, found {nested[0]}")
        log(f"[warn]        move it to {DATA_ROOT / name} or point DATA_ROOT at the parent")
    elif (DATA_ROOT / name).is_dir():
        found = [p.name for p in sorted((DATA_ROOT / name).iterdir())[:10]]
        log(f"[warn] dataset {name}: no test.jsonl in {DATA_ROOT / name} (found: {found})")
    else:
        log(f"[warn] dataset {name}: not found under {DATA_ROOT}")


def verify_model(name: str) -> None:
    target = MODEL_ROOT / name
    if target.is_dir() and any(target.iterdir()):
        n = sum(1 for _ in target.rglob("*"))
        log(f"[ok]   model {name}: {target} ({n} entries)")
    else:
        log(f"[warn] model {name}: {target} is missing or empty")


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def verify_retrieval_corpus() -> None:
    """The prebuilt FAISS index and Wikipedia corpus live in retrieval-corpus/."""
    base = DATA_ROOT / "retrieval-corpus"
    if not base.is_dir():
        log(f"[warn] retrieval-corpus/: not found under {DATA_ROOT}")
        log("[warn]        fetch it with --include \"<dataset>/*\" --with-retrieval-corpus,")
        log("[warn]        or download the dataset repository without --include")
        return
    index = base / "e5_flat_inner.index"
    zipped = base / "wiki18_100w.zip"
    corpus = base / "wiki18_100w.jsonl"
    if index.is_file():
        log(f"[ok]   {index.name}: {index} ({_mb(index)})")
    else:
        log(f"[warn] {index.name}: missing in {base}")
    if corpus.is_file():
        log(f"[ok]   {corpus.name}: {corpus} ({_mb(corpus)})")
    elif zipped.is_file():
        log(f"[ok]   {zipped.name}: {zipped} ({_mb(zipped)})")
        log(f"[info] unzip it:  unzip -o {zipped} -d {base}")
    else:
        log(f"[warn] neither {zipped.name} nor {corpus.name} in {base}")
    log("[info] retrieval_server/retriever_config.yaml:")
    log(f"[info]   index_path:  {index}")
    log(f"[info]   corpus_path: {corpus}")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
log(f"[paths] DATA_ROOT  = {DATA_ROOT}")
log(f"[paths] MODEL_ROOT = {MODEL_ROOT}")

if BACKEND == "auto":
    reachable = hf_reachable()
    log(f"[probe] huggingface.co reachable within {HF_TIMEOUT}s: {reachable}")
    backend = "hf" if reachable else "modelscope"
    if not reachable:
        log("[probe] huggingface.co unreachable -> falling back to ModelScope")
else:
    backend = BACKEND
    log(f"[probe] backend forced to {backend}")
    if backend == "hf" and not hf_reachable():
        log("[probe] WARNING: --backend hf forced but huggingface.co looks unreachable;")
        log("[probe]          the download will likely fail.")

# Both libraries are checked/installed as requested.
have_hf = ensure_pkg("huggingface_hub", "huggingface_hub")
have_ms = ensure_pkg("modelscope", "modelscope")
if backend == "hf" and not have_hf:
    log("[error] huggingface_hub is required for --backend hf")
    raise SystemExit(1)
if backend == "modelscope" and not have_ms:
    log("[error] modelscope is required for --backend modelscope")
    raise SystemExit(1)

DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
if SKIP_DATASETS:
    log("[skip] datasets")
elif backend == "hf":
    hf_download(HF_DATASET_ID, DATA_ROOT, dataset=True)
else:
    try:
        ms_download(MS_DATASET_ID, DATA_ROOT, dataset=True)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        log(f"[warn] ModelScope dataset download failed: {exc}")
        log("[warn] fetch the FlashRAG datasets manually so that")
        log(f"[warn]   {DATA_ROOT}/<dataset>/test.jsonl")
        log("[warn] exists for each dataset you want to run.")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
if SKIP_MODELS:
    log("[skip] models")
else:
    for name in MODELS:
        target = MODEL_ROOT / name
        if backend == "hf":
            repo_id = HF_MODEL_IDS.get(name)
            if repo_id is None:
                log(f"[warn] no Hugging Face repo id configured for {name}; skipping")
                continue
            hf_download(repo_id, target)
        else:
            repo_id = MS_MODEL_IDS.get(name)
            if repo_id is None:
                log(f"[warn] {name}: no ModelScope mirror configured "
                    f"(Hugging Face id: {HF_MODEL_IDS.get(name)!r}); skipping")
                continue
            ms_download(repo_id, target, dataset=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log("")
log("=== verification ===")
if not SKIP_DATASETS:
    for name in DATASETS:
        verify_dataset(name)
    verify_retrieval_corpus()
if not SKIP_MODELS:
    for name in MODELS:
        verify_model(name)

log("")
log("=== next steps ===")
log(f"1. Models are under {MODEL_ROOT}; use them for --model_path / `vllm serve`.")
log(f"2. Datasets are under {DATA_ROOT}/<dataset>/test.jsonl, which is what")
log("   scripts/run_maps_all.sh reads (via its DATA_ROOT).")
log("3. Set retrieval_server/retriever_config.yaml to your e5-base-v2 weights,")
log("   Wikipedia corpus and FAISS index (the index is built with FlashRAG).")
PY
