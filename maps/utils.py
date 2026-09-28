import json
import queue
import threading
from pathlib import Path
from typing import Any


def read_jsonlines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            records.append(obj)
    return records


def write_jsonlines(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class AsyncJsonlWriter:
    """Background-thread JSONL appender: submit(record) is non-blocking.

    Records are written asynchronously by a daemon thread, one JSON object per
    line, flushed after every write so partial results survive a crash.
    close() drains the queue and joins the thread.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            while True:
                record = self._queue.get()
                if record is None:
                    f.flush()
                    break
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                self._queue.task_done()

    def submit(self, record: dict[str, Any]) -> None:
        if self._closed:
            return
        self._queue.put(record)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=60)


def build_output_dir(
    input_path: Path, method_name: str, model_name: str, top_dir: str = "outputs"
) -> Path:
    dataset_name = input_path.parent.name
    dataset_root = Path(top_dir) / method_name / model_name / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    existing_indices: list[int] = []
    for child in dataset_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_indices.append(int(child.name))

    next_index = 0 if not existing_indices else max(existing_indices) + 1
    output_dir = dataset_root / str(next_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir