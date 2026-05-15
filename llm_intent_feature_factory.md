# LLM 离线意图特征工厂（v7）

目标：在现有统计特征（CTR、曝光、ID Embedding）之外，新增用户-物品对的语义意图特征，提升转化预估。

## 一、总体设计

离线流水线分 5 个阶段：

1. **样本构建**：从训练集抽取 `(user_id, item_id, context)`。
2. **提示词组装**：将用户行为摘要、物品属性、上下文拼接为 LLM Prompt。
3. **LLM 推理**：批量调用大模型，输出结构化 JSON 意图标签。
4. **特征编码**：将 JSON 转为模型可消费的稠密/离散特征。
5. **Join 回流**：按 `(user_id, item_id)` 回填到训练 Parquet。

## 二、建议输出 Schema（结构化）

LLM 输出 JSON（示例）：

```json
{
  "intent_primary": "price_sensitive",
  "intent_secondary": ["fast_shipping", "trusted_brand"],
  "purchase_stage": "comparison",
  "confidence": 0.84,
  "reason": "用户近期多次比较同品类并关注促销"
}
```

入模特征建议：

- `llm_intent_primary_id`（离散）
- `llm_intent_secondary_multi_hot`（多值离散）
- `llm_purchase_stage_id`（离散）
- `llm_intent_confidence`（连续）
- `llm_reason_emb[dim]`（文本向量，可选）

## 三、离线实现要点

- **稳定性**：固定模型版本与 prompt 模板版本（`prompt_ver`）。
- **一致性**：训练/验证/推理使用同一套 label->id 字典。
- **可追踪**：落地 `request_id`, `model_name`, `latency_ms`, `err_code`。
- **成本控制**：
  - 对 `(user_id,item_id)` 做去重缓存。
  - 对长文本做摘要压缩。
  - 对低价值样本（如极低曝光）降采样。

## 四、质量监控

- 覆盖率：`intent_non_null_rate`
- 稳定性：同样本多次推理一致率
- 业务增益：AUC / LogLoss / CVR uplift
- 风险监控：异常标签占比、解析失败率、超时率

## 五、与现有训练代码衔接

当前仓库的训练入口在 `train.py`，数据读取在 `dataset.py`。建议先采用“旁路特征”方式：

1. 在产出 Parquet 中新增 `user_dense_feats_xxx` 或 `item_dense_feats_xxx` 列；
2. 同步更新 `schema.json`，让 `PCVRParquetDataset` 自动装载；
3. 通过 `train.py` 原有流程直接训练，无需改动主干模型结构。

这样改动最小，便于快速验证 +10‰～25‰ 的离线收益区间。
