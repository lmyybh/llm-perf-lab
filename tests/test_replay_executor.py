"""replay 执行器测试。

这里重点覆盖异步调度、QPS 节拍以及聚合统计结果。
"""

import asyncio
import itertools
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from llmperf.core.models import ReplayItemResult, ReplayRequest
from llmperf.core.replay_executor import execute_replay, summarize_replay


class ReplayExecutorTest(unittest.TestCase):
    def test_execute_replay_updates_tqdm_progress(self):
        """异步执行完成后应正确刷新 tqdm 进度。"""
        requests = [
            ReplayRequest(
                source_file="a.pkl",
                source_index=0,
                endpoint_type="generate",
                payload={"input_ids": [1]},
                stream=False,
            ),
            ReplayRequest(
                source_file="a.pkl",
                source_index=1,
                endpoint_type="generate",
                payload={"input_ids": [2]},
                stream=False,
            ),
        ]

        async def fake_execute(*args, **kwargs):
            return ReplayItemResult(
                source_file="a.pkl",
                source_index=0,
                status="ok",
                latency_ms=10,
                request_start_time=0.0,
                output_text="ok",
            )

        progress = MagicMock()
        bar_ctx = MagicMock()
        bar_ctx.__aenter__ = AsyncMock(return_value=progress)
        bar_ctx.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "llmperf.core.replay_executor.tqdm_async", return_value=bar_ctx
        ), patch(
            "llmperf.core.replay_executor.execute_replay_request_async",
            side_effect=fake_execute,
        ):
            results, summary = execute_replay(
                requests=requests,
                endpoint="http://127.0.0.1:8000/generate",
                timeout_ms=1000,
                api_key=None,
                qps=None,
                max_concurrency=2,
            )
        self.assertEqual(len(results), 2)
        self.assertEqual(summary.requests_succeeded, 2)
        self.assertEqual(progress.update.call_count, 2)

    def test_summarize_replay_includes_accept_metrics(self):
        """summary 应包含 accept 指标和基于发送时间计算的 actual_qps。"""
        results = [
            ReplayItemResult(
                source_file="a.pkl",
                source_index=0,
                status="ok",
                latency_ms=100,
                request_start_time=0.0,
                ttft_ms=20,
                output_text="ok",
                output_tokens=5,
                tpot_ms=20.0,
                accept_len=2.0,
                accept_rate=0.25,
                status_code=200,
            ),
            ReplayItemResult(
                source_file="a.pkl",
                source_index=1,
                status="ok",
                latency_ms=200,
                request_start_time=0.5,
                ttft_ms=30,
                output_text="ok",
                output_tokens=10,
                tpot_ms=18.0,
                accept_len=4.0,
                accept_rate=0.5,
                status_code=200,
            ),
        ]
        summary = summarize_replay(results, elapsed_ms=500)
        self.assertEqual(summary.actual_qps, 2.0)
        self.assertEqual(summary.accept_len, 3.0)
        self.assertEqual(summary.accept_rate, 0.375)

    def test_qps_mode_uses_async_sleep_for_request_rate(self):
        """QPS 模式应通过 async sleep 控制发起节拍。"""
        requests = [
            ReplayRequest(
                source_file="a.pkl",
                source_index=i,
                endpoint_type="generate",
                payload={"input_ids": [i]},
                stream=False,
            )
            for i in range(3)
        ]

        async def fake_execute(*args, **kwargs):
            return ReplayItemResult(
                source_file="a.pkl",
                source_index=0,
                status="ok",
                latency_ms=10,
                request_start_time=0.0,
                output_text="ok",
            )

        progress = MagicMock()
        bar_ctx = MagicMock()
        bar_ctx.__aenter__ = AsyncMock(return_value=progress)
        bar_ctx.__aexit__ = AsyncMock(return_value=None)

        sleep_calls: list[float] = []

        async def fake_sleep(delay: float):
            sleep_calls.append(delay)

        perf_values = itertools.chain(
            [0.0, 0.0, 0.3, 0.6, 0.9, 1.2], itertools.repeat(1.2)
        )

        def fake_perf_counter():
            return next(perf_values)

        with patch(
            "llmperf.core.replay_executor.tqdm_async", return_value=bar_ctx
        ), patch(
            "llmperf.core.replay_executor.execute_replay_request_async",
            side_effect=fake_execute,
        ), patch(
            "llmperf.core.replay_executor.asyncio.sleep", side_effect=fake_sleep
        ), patch(
            "llmperf.core.replay_executor.time.perf_counter",
            side_effect=fake_perf_counter,
        ):
            execute_replay(
                requests=requests,
                endpoint="http://127.0.0.1:8000/generate",
                timeout_ms=1000,
                api_key=None,
                qps=2.0,
                max_concurrency=None,
            )

        self.assertGreaterEqual(len(sleep_calls), 1)
        self.assertAlmostEqual(sleep_calls[0], 0.1, places=3)


if __name__ == "__main__":
    unittest.main()
