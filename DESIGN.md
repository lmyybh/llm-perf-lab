# LLM 推理服务 CLI 设计（Request + Replay）

## 1. 目标

实现两个命令：

1. `llmperf request`
2. `llmperf replay`

支持两种接口，并且输入参数严格分离：

1. `/v1/chat/completions`：只允许 `--messages`（JSON 格式对话）
2. `/generate`：只允许 `--prompt`（字符串）

`request` 不支持文件输入；`replay` 用于回放 SGLang dump 的原始推理请求。

## 2. 命令形态

统一命令：`llmperf request`

chat 接口示例：

```bash
llmperf request \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --messages '[{"role":"system","content":"你是代码助手"},{"role":"user","content":"给我一个快排思路"},{"role":"assistant","content":"可以用分治..."},{"role":"user","content":"给出 Go 代码"}]' \
  --max-new-tokens 128 \
  --stream
```

generate 接口示例：

```bash
llmperf request \
  --endpoint http://127.0.0.1:8000/generate \
  --prompt "写一段 Python 快排" \
  --max-new-tokens 256
```

## 3. 参数设计

### 3.1 必填参数

- `--endpoint`：完整 URL，必须以 `/generate` 或 `/v1/chat/completions` 结尾。

### 3.2 接口专用输入参数

- chat 专用：`--messages`（JSON 数组字符串）
- generate 专用：`--prompt`（字符串）

校验规则：

- endpoint 为 `/v1/chat/completions` 时：
  - 必须提供 `--messages`
  - 不允许提供 `--prompt`
- endpoint 为 `/generate` 时：
  - 必须提供 `--prompt`
  - 不允许提供 `--messages`

### 3.3 通用参数

- `--api-key` / `--api-key-env`
- `--timeout-ms`
- `--format table|json`
- `--save-output <path>`

### 3.4 推理参数

- `--model`（chat 接口通常必填）
- `--max-new-tokens`（chat/generate 通用）
- `--temperature`
- `--top-p`
- `--stream/--no-stream`

## 4. 接口适配规则

通过 endpoint 后缀自动选择 adapter。

### 4.1 `/v1/chat/completions`

请求映射：

- `--messages` 直接映射到 `messages`
- 参数映射：`model/max_new_tokens/temperature/top_p/stream`

`--messages` 格式要求：

- 必须是 JSON 数组
- 每项结构：`{"role":"system|user|assistant|tool","content":"..."}`
- 至少包含一条 `user` 消息

输出归一化字段：

- `status`
- `latency_ms`
- `ttft_ms`（流式）
- `output_text`
- `usage`（可用时）

### 4.2 `/generate`

请求映射：

- `--prompt` 映射为 `prompt`（或后端所需字段，由适配层处理）
- 参数映射：`max_new_tokens/temperature/top_p/stream`

输出归一化字段：

- `status`
- `latency_ms`
- `ttft_ms`（流式）
- `output_text`
- `token_stats`（可用时）

## 5. 输出设计

终端输出：

- `status`
- `latency_ms`
- `ttft_ms`（可用时）
- `output_text`（可截断预览）

文件输出（`--save-output`）：

- 单个 JSON 结果对象。

## 6. 错误与退出码

错误分类：

1. `config_error`
2. `input_error`
3. `network_error`
4. `http_error`
5. `protocol_error`
6. `timeout_error`

退出码：

- `0`：成功。
- `2`：参数错误。
- `3`：请求执行失败。

## 7. MVP 验收标准

1. 单一 `llmperf request` 命令可稳定发送请求。
2. `/v1/chat/completions` 严格使用 `--messages`。
3. `/generate` 严格使用 `--prompt`。
4. 响应可统一输出 `status/latency/output_text`。
5. 错误分类与退出码行为一致。

## 8. Replay 命令

### 8.1 目标

从 SGLang dump 的 `.pkl` 文件中提取 `GenerateReqInput`，按原始推理输入回放到 `/generate`。

由于 dump 中通常只保留 `input_ids` 与采样参数，不能可靠恢复原始 `/v1/chat/completions` 的 `messages`，因此 replay 的一致性定义为：

- 与 SGLang chat 接口内部转换后的 `GenerateReqInput` 一致
- 不要求恢复 OpenAI 风格 chat body

### 8.2 命令形态

```bash
llmperf replay \
  --endpoint http://127.0.0.1:8000/generate \
  --dump-path /path/to/raw_log_data \
  --qps 8
```

### 8.3 参数设计

- `--endpoint`：完整 URL，必须以 `/generate` 结尾
- `--dump-path`：单个 `.pkl` 文件或包含多个 `.pkl` 的目录
- `--qps`
- `--max-concurrency`
- `--timeout-ms`
- `--api-key`
- `--save-output`
- `--num-requests`

约束：

- `--qps` 与 `--max-concurrency` 可同时提供
- 若同时提供，仅 `--qps` 生效，并输出提示

### 8.4 回放字段

默认保留以下请求语义字段：

- `input_ids/text`
- `sampling_params`
- `stream`
- `return_logprob`
- `logprob_start_len`
- `top_logprobs_num`
- 多模态输入字段
- 其他非空的生成相关字段

默认不回放以下运行时字段：

- `rid`
- `validation_time`
- `received_time`
- `received_time_perf`
- `http_worker_ipc`
- `log_metrics`

### 8.5 输出设计

终端输出聚合统计：

- `total/succeeded/failed`
- `success_rate`
- `elapsed_ms`
- `throughput_rps`
- `latency p50/p95/p99`
- `ttft p50/p95/p99`

文件输出（`--save-output`）：

- JSONL 格式逐请求结果
