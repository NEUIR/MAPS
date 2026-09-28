# app.py
import argparse
import asyncio
from collections import deque

from fastapi import FastAPI, HTTPException
from flashrag.config import Config
from flashrag.utils import get_retriever
from pydantic import BaseModel

app = FastAPI()

retriever_list = []
available_retrievers = deque()
retriever_semaphore: asyncio.Semaphore


def init_retriever(args):
    global retriever_semaphore
    config_dict = {}
    if args.gpu_id is not None:
        config_dict["gpu_id"] = args.gpu_id
    config = Config(args.config_file_path, config_dict=config_dict)
    # 根据 num_retriever 参数初始化指定数量的检索器
    for i in range(args.num_retriever):
        print(f"Initializing retriever {i + 1}/{args.num_retriever}...")
        retriever = get_retriever(config)
        retriever_list.append(retriever)
        available_retrievers.append(i)

    print(f"Successfully initialized {len(retriever_list)} retrievers.")
    # 创建信号量，限制并发访问
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers),
        },
    }


class QueryRequest(BaseModel):
    query: str
    top_n: int = 10
    return_score: bool = False


class BatchQueryRequest(BaseModel):
    query: list[str]
    top_n: int = 10
    return_score: bool = False


class Document(BaseModel):
    id: str
    contents: str


@app.post("/search")
async def search(request: QueryRequest):
    """处理单个查询的检索"""
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    if not query or not query.strip():
        print(f"Query content cannot be empty: {query}")
        raise HTTPException(status_code=400, detail="Query content cannot be empty")

    # 异步等待一个可用的检索器
    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        # print(f"Using retriever {retriever_idx} for query: {query[:20]}...")
        try:
            if return_score:
                results, scores = retriever_list[retriever_idx].search(
                    query, top_n, return_score
                )
                return [
                    Document(id=result["id"], contents=result["contents"])
                    for result in results
                ], scores
            results = retriever_list[retriever_idx].search(query, top_n, return_score)
            return [
                Document(id=result["id"], contents=result["contents"])
                for result in results
            ]
        except Exception as e:
            print(f"Error during search: {e}")
            raise HTTPException(status_code=500, detail="Search failed")
        finally:
            # 释放检索器，使其可用于其他请求
            available_retrievers.append(retriever_idx)


@app.post("/batch_search")
async def batch_search(request: BatchQueryRequest):
    """处理批量查询的检索"""
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    if not query or len(query) == 0:
        raise HTTPException(status_code=400, detail="Query list cannot be empty")

    # 异步等待一个可用的检索器
    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        print(
            f"Using retriever {retriever_idx} for batch search of {len(query)} queries..."
        )
        try:
            if return_score:
                results, scores = retriever_list[retriever_idx].batch_search(
                    query, top_n, return_score
                )
                return [
                    [
                        Document(id=result["id"], contents=result["contents"])
                        for result in res_list
                    ]
                    for res_list in results
                ], scores
            results = retriever_list[retriever_idx].batch_search(
                query, top_n, return_score
            )
            return [
                [
                    Document(id=result["id"], contents=result["contents"])
                    for result in res_list
                ]
                for res_list in results
            ]
        except Exception as e:
            print(f"Error during batch search: {e}")
            raise HTTPException(status_code=500, detail="Batch search failed")
        finally:
            # 释放检索器
            available_retrievers.append(retriever_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        type=str,
        default="./retriever_config.yaml",
        help="Path to the retriever config file",
    )
    parser.add_argument(
        "--num_retriever",
        type=int,
        default=1,
        help="Number of retriever instances to load",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help='Override gpu_id in config file, e.g. "0" or "0,1"',
    )
    parser.add_argument(
        "--port", type=int, default=9100, help="Port to run the server on"
    )
    args = parser.parse_args()

    # 初始化检索器
    init_retriever(args)

    # 启动 FastAPI/Uvicorn 服务
    import uvicorn

    print(f"Starting server on http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
