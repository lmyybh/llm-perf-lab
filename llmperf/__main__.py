"""允许通过 `python -m llmperf` 启动 CLI。"""

from llmperf.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
