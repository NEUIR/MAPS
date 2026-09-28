# retrieval_server

A minimal FastAPI wrapper around a [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
dense retriever. The retriever is `e5-base-v2` (dense embeddings) with a FAISS
inner-product (flat) index built over the `wiki18_100w` corpus (the first 1,000,000
passages of the Wikipedia 2018 corpus).

## Endpoints

| Method | Path           | Description                                  |
|--------|----------------|----------------------------------------------|
| GET    | `/health`      | Health check; reports retriever availability |
| POST   | `/search`      | Single-query retrieval (body: `query`, `top_n`, `return_score`) |
| POST   | `/batch_search`| Batch retrieval (body: `query` as a list of strings, `top_n`, `return_score`) |

## Start

From the repository root:

```bash
python retrieval_server/app.py \
  --config_file_path retrieval_server/retriever_config.yaml \
  --num_retriever 1 \
  --port 9100 \
  --gpu_id "0,1"
```

Arguments:

- `--config_file_path`: path to the FlashRAG retriever config YAML.
- `--num_retriever`: number of retriever instances to load (used as a concurrency
  semaphore for parallel requests).
- `--port`: HTTP port to listen on (default `9100`).
- `--gpu_id`: overrides the `gpu_id` in the config file, e.g. `"0"` or `"0,1"`.

## Configuration

**Important:** the `retrieval_model_path`, `index_path` and `corpus_path` entries in
`retriever_config.yaml` point to machine-specific locations and **must be edited by
the user** to match their local model and index files before starting the server.