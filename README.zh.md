# branchmem（中文说明）

对话和 Agent 的历史是一棵树：重新生成回复、编辑历史消息、从检查点分叉，都会产生互为兄弟的分支。原始消息按当前分支过滤很容易，难的是长期记忆：摘要、事实、画像、关系图的边，都是在旧分支还是"当前分支"时就生成好的，分支被放弃以后它们还留在库里。

branchmem 的规则只有一条：**派生记忆只有当它的全部来源事件都在当前锚点的祖先路径上时才可见。**

由此得到的几条不变量：

- **兄弟不互见**：一条分支只看到自己的锚点和锚点的祖先。共享前缀对所有分支可见，兄弟分支的内容和由它派生的记忆都不可见。
- **按来源判断，而不是打标签**：可见性不由 `lineage_id`、`branch_id` 或 `inactive` 标记决定。标签会过时，来源事件 ID 不会。只要有一条来源在兄弟分支上，整条笔记就隐藏。
- **切回旧分支只改锚点**：切线时不标记、不删除、不改写任何东西。切回旧分支，旧分支的记忆自然恢复。
- **缺锚点即关闭**：严格模式下，没有锚点的树什么都不返回，绝不默认放行整棵树。
- **拓扑确定**：同一组事件不论到达顺序（子节点先到、重复投递、迟到事件），建出同一棵树、同一组分段和 ID。
- **内容门控不剪断树**：图片占位、思考记录、系统事件可以不进提取，但仍留在树里，父链不断。
- **共享需显式**：`namespace_shared` 只豁免笔记本身，展开来源时仍按原文路径复核。
- **不静默丢数据**：每个提取作业最终都是完成、跳过（带原因）、失败（可重试）或待处理之一。

精确的谓词、数据契约和原因码见 [docs/invariants.md](docs/invariants.md)。

## 使用

Python 3.10+，核心库零依赖。

```bash
pip install -e .            # 核心库 + 基准
pip install -e ".[notes]"   # 向量检索需要 numpy
pip install -e ".[zh]"      # 中文 BM25 分词需要 jieba
pip install -e ".[test]"    # pytest
```

用法示例见 [README.md](README.md) 的 Quick start 和 [examples/](examples/)。

## 跑测试和基准

```bash
python -m pytest -q
python -m branchbench validate                 # 全部 30 个场景；--split dev|test 只看一个集合
python -m branchbench context-report --split test
python -m branchbench estimate --split test --runs 3   # 估算调用次数和 token，不调用 API
python -m branchbench answer-run --split test --runs 3 --label test-v1
python -m branchbench answer-report results/test-v1 --markdown docs/results_test.md
```

场景集：30 个中文合成对话、98 道题，五种分叉方式各 6 个。开发集 `bench/scenarios/dev/`（10 个场景、29 道题，可用来调提示词和流程）；测试集 `bench/scenarios/test/`（20 个场景、69 道题，冻结后不再修改，正式结果只报测试集，`FROZEN.json` 记录每个文件的 SHA-256）。回答级运行的模型配置只从环境变量读取（`BB_LLM_*`、`BB_EMBED_*`），原始产物写进不入库的 `results/`，汇总表入库到 `docs/`。

### 结果

回答级结果待测试集正式运行后补上。

基准场景是合成的分支对话（编辑用户消息、重新生成、从检查点分叉、切回旧分支、分支内改口）。校验器保证每道题真的在测它声称要测的东西；评分只看关键词，不用 LLM 判官，结果可复现。上下文级结果：只过滤原文、保留派生记忆的读法，泄漏题全部泄漏；按来源判断的读法泄漏为 0，命中率 100%。

## 来源

代码整理自一个私有应用的记忆系统。与原实现做了差分对照：在 400 棵随机打乱到达顺序的树上，建树、约 10.7 万次可见性判定、约 4700 次读闸门调用、约 2.4 万次笔记判定、两版分段器和重新生成推断，结果完全一致。原部署中的聚合结果（不公开任何数据）：5 个会话约 2.3 万条事件配回消息树，模拟请求下约 1 万条被丢弃分支的内容被挡下，兄弟线泄漏 0，主线误隐藏 0。

## 许可证

尚未确定。
