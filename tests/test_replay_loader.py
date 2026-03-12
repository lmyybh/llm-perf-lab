"""replay dump 加载测试。"""

import pickle
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from llmperf.core.errors import ConfigError, InputError
from llmperf.core.replay_loader import load_replay_requests, validate_replay_endpoint


@dataclass
class FakeGenerateReqInput:
    """用于测试的最小化假请求结构。"""

    input_ids: list[int] | None = None
    text: str | None = None
    sampling_params: dict | None = None
    stream: bool = False
    rid: str | None = None
    received_time: float | None = None


class ReplayLoaderTest(unittest.TestCase):
    def test_load_requests_from_directory_keeps_payload_fields(self):
        """目录模式下应按文件名排序读取，并保留有效 payload 字段。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            payload = {
                "requests": [
                    (
                        FakeGenerateReqInput(
                            input_ids=[1, 2, 3],
                            sampling_params={"temperature": 0.5},
                            stream=True,
                            rid="old-rid",
                            received_time=1.0,
                        ),
                        {"ignored": True},
                        0.0,
                        1.0,
                    )
                ]
            }
            (base / "b.pkl").write_bytes(pickle.dumps(payload))
            (base / "a.pkl").write_bytes(pickle.dumps(payload))

            requests = load_replay_requests(str(base), limit=1)

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].source_file, "a.pkl")
        self.assertEqual(requests[0].payload["input_ids"], [1, 2, 3])
        self.assertEqual(requests[0].payload["sampling_params"], {"temperature": 0.5})
        self.assertTrue(requests[0].stream)
        self.assertNotIn("rid", requests[0].payload)
        self.assertNotIn("received_time", requests[0].payload)

    def test_validate_replay_endpoint_requires_generate(self):
        """replay 只能指向 `/generate`。"""
        with self.assertRaises(ConfigError):
            validate_replay_endpoint("http://127.0.0.1:8000/v1/chat/completions")

    def test_load_requests_rejects_missing_prompt_data(self):
        """既没有 text 也没有 input_ids 的请求不可回放。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            payload = {"requests": [(FakeGenerateReqInput(),)]}
            dump_path = base / "bad.pkl"
            dump_path.write_bytes(pickle.dumps(payload))

            with self.assertRaises(InputError):
                load_replay_requests(str(dump_path))


if __name__ == "__main__":
    unittest.main()
