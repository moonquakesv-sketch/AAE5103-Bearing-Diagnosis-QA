# E01-S02 四情景评价准备 v2

版本 `e01.four_scenario.v2_rev027`；只准备材料，正式放行false。业务请求14题，评分66点；debug12、运行保留2。四类各覆盖explanation/inspection/preparation，外圈另有三任务组合。10个Timken正向点评价来源真值挂起。旧v1的15题/68点和rev015原字节保留。

先读 `docs/E01_S02_运行与评分协议.md`、coverage.csv、exposure_registry.json、source_review.json和release_checklist.json。类别/任务均为作者指定模块输入，原因独立假定；不得借I01真实来源文案冒充实际推理。normal不证明无损伤、健康或放行。

业务侧只有两份requests.jsonl的既有五字段。question_bindings.json保存评价侧ID、family、split、行号与规范请求哈希；rubrics.json逐点评分和完整证据不进入业务输入。来源状态、来源版本登记、盲码与评分始终留在评价侧。字段隔离不能识别自由文本中的所有答案泄漏。

题集由先行design_contract.json确定。12个公开/近义目标放debug；sample_provenance与source_verifiability两个窄目标留作作者可见的运行保留集。保留题只有rolling_element、outer_race各一题，inner_race和normal无保留题。不能据此宣称四类均衡盲测或四类问答收益；若后续要求此类结论，须先另行设计并审核未暴露的保留目标，或批准更窄的描述性研究范围。知识已开发可见；多题和重复不构成多个独立设备。

固定共同知识为knowledge/i01/v1（7来源/4情景/16卡），corpus_binding.json逐项哈希，重新核对旧外圈材料与共同语料；reuse_mapping.json逐题映射旧目标并保留三项rev015修复。AND/OR支持组合仅静态枚举，不代表实际检索卡。coverage中最少卡数未查看排名；四组全局top_k=3完整卡，不补卡、不裁条件，等卡不等长。

source_review.json按全部66点登记，10点needs_source_review/suspended。未获同版Timken本地PDF/哈希/视觉审核；不得给这些点确定性来源真值分、静默删点或缩分母。NSK本地原文已静态对照，仍不等于人类/领域专家签核。角接触球轴承、调心滚子案例与CWRU深沟球人工缺陷之间的限制保留。

validation代码是新run内对既有校验器的必要适配，原脚本和旧行为保持。只可在新的不存在输出路径作静态校验；合成测试不是真题系统运行。不要运行本run的authoring/封存脚本重建历史产物，不执行旧finalize/freeze/acceptance/launch，不重跑已消费D04/I01/Q01，不复位预算。

本轮证据位于runs/e01/20261026_s02_695c0994/；实际验证与独立复核以该目录验收/最终读回为准。release_checklist.json中的服务、模板、计数器和价格仅候选/历史记录；批准配置、现价、新硬预算和人类人员均未确定。完成准备封存后停止，E02另行明确授权。
