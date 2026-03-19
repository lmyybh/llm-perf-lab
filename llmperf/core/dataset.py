import typer
import json
from pathlib import Path

from llmperf.core.data import OpenAIRequestInput


def check_file(file: Path, suffixs=None):
    if not file.is_file():
        raise typer.BadParameter(f"file not found: {file}")

    if suffixs is not None and file.suffix not in suffixs:
        raise typer.BadParameter(f"unsupported file format: {file.suffix}")


def read_zss_file(file: Path, num_requests: int):
    check_file(file)

    def parse_line(line: str):
        data = json.loads(line)

        return OpenAIRequestInput(
            messages=data["messages"],
            model=data["model"],
            stream=data["stream"],
            tools=data.get("tools"),
            rid=data["rid"],
            temperature=data["temperature"],
            max_tokens=data["max_tokens"],
            seed=data["seed"],
            frequency_penalty=data["frequency_penalty"],
            repetition_penalty=data["repetition_penalty"],
            presence_penalty=data["presence_penalty"],
            chat_template_kwargs=data["chat_template_kwargs"],
        )

    requests = []
    with file.open() as f:
        for i, line in enumerate(f):
            if i >= num_requests:
                break
            requests.append(parse_line(line))

    return requests


if __name__ == "__main__":
    file = Path(
        "/data/cgl/download/datasets/boss/zhishanshan/qwen235-2507-fp8/raw_request/request_235b_20min.jsonl"
    )

    data = read_zss_file(file, 10)

    print(data[0])
