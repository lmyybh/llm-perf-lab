import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from llmperf.cli import app
from llmperf.commands.bench.args import (
    BenchCommandArgs,
    BenchCommonArgs,
    FileDatasetArgs,
)
from llmperf.commands.bench.service import build_dataset
from llmperf.common.tokenization import load_tokenizer
from llmperf.datasets.file import select_jsonl_parser
from llmperf.errors import ConfigError, DatasetFormatError


class ErrorBoundaryTest(unittest.TestCase):
    def test_load_tokenizer_raises_project_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            load_tokenizer(
                tokenizer_path=None,
                model_name=None,
                purpose="request",
                required=True,
            )

    def test_select_jsonl_parser_raises_project_dataset_error(self) -> None:
        with self.assertRaises(DatasetFormatError):
            select_jsonl_parser("unsupported")

    def test_build_dataset_requires_num_requests_for_random_mode(self) -> None:
        args = BenchCommandArgs(
            common=BenchCommonArgs(url="http://localhost", num_requests=None),
            dataset={
                "mode": "random",
                "seed": 0,
                "min_input_tokens": 1,
                "max_input_tokens": 2,
                "min_output_tokens": 1,
                "max_output_tokens": 2,
            },
        )

        with self.assertRaises(ConfigError):
            build_dataset(args)

    def test_cli_translates_project_errors_for_request_command(self) -> None:
        runner = CliRunner()
        with patch("llmperf.cli.run_request_command", side_effect=ConfigError("boom")):
            result = runner.invoke(app, ["request", "--user", "hello"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("boom", result.output)

    def test_cli_translates_project_errors_for_bench_command(self) -> None:
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            dataset_path = Path(tmp.name)
            with patch(
                "llmperf.cli.run_bench_command", side_effect=ConfigError("bench boom")
            ):
                result = runner.invoke(
                    app,
                    [
                        "bench",
                        "--file",
                        str(dataset_path),
                        "--mode",
                        "openai-jsonl",
                    ],
                )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("bench boom", result.output)


if __name__ == "__main__":
    unittest.main()
