# llm-perf-lab

当前包含“发送请求”和“回放 dump 请求”两部分能力。

## 文档入口

1. [DESIGN.md](/data/cgl/projects/llm-perf-lab/DESIGN.md): `llmperf request` 与 `llmperf replay` 的命令设计、配置设计、模块边界与错误处理。

## 运行

```bash
python3 -m llmperf request --help
./bin/llmperf request --help
python3 -m llmperf replay --help
./bin/llmperf replay --help
```

## 测试

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
