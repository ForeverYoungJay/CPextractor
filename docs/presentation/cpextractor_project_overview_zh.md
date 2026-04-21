# CPextractor 项目介绍

## 1. 标题
- CPextractor
- 从晶体塑性论文到可追溯、可检索、可复核的参数数据库
- 关键词：Extraction, Grounding, Committee, Confidence, Database

## 2. 项目要解决的问题
- 晶体塑性参数散落在正文、表格、补充材料和引用文献里
- 仅抽取数值不够，科研复用还需要单位归一、作用范围、来源与证据
- 手工整理成本高，难以支撑规模化数据库建设和 RAG 问答

## 3. CPextractor 总览
- 输入：Scopus 检索、DOI 列表、本地全文目录
- 解析：Elsevier XML 转为 sections、tables、references
- 抽取：两阶段 LLM，先选文件，再做 schema-constrained extraction
- 后处理：参数归一化、材料/相/scope/binding 解析、evidence grounding
- 输出：parameter_claims、审查材料、PostgreSQL、向量库、聊天检索

## 4. 端到端流水线
- 获取文献并建立本地全文资产
- 解析正文、表格、引用关系
- 抽取候选参数记录到 `parameters.registry`
- 确定性后处理和证据回链
- 运行 LLM committee 与 meta judge
- 融合规则分和 judge 分数，决定置信度、质量层级和是否入库

## 5. 质量控制栈
- Layer 1: Rule validation
- Layer 2: Evidence grounding
- Layer 3: Multi-agent LLM evaluation
- Layer 4: Confidence fusion and quality tiering
- 设计重点：低质量论文可以被 gate 掉，但完整审计痕迹仍然保留

## 6. 项目里有没有 rule-based judge
- 严格说，没有一个单独命名为 “rule-based judge” 的 agent
- 但有一层确定性规则校验，功能上相当于 rule-based QA / validator
- 典型检查：evidence 缺失、关键参数缺值、单位缺失、明显负值、scope 绑定不一致
- 规则结果会产出 `quality_checks` 报告和 `rule_score`
- 后续会与 LLM committee 输出一起参与最终 confidence fusion

## 7. LLM committee 怎么工作
- Evidence judge：只判断参数是否被当前证据支持
- Normalization judge：检查 canonical mapping、unit、SI conversion
- Consistency judge：检查跨参数、跨材料、binding、model scope 的一致性
- Meta judge：综合 committee 输出、rule report、evidence report，给出文档级 verdict
- 合并策略：参数级三票先做 consensus，再做 policy adjustment，最后形成 review_required 与 final audit

## 8. Rule + Committee 融合逻辑
- 参数级分数同时考虑 rule penalty、LLM audit score、基础证据质量
- 文档级分数同时考虑 `rule_score` 与 `overall_score`
- 典型降级策略：
- 对低风险 table grounding 分歧放宽处罚
- 对实际上 SI 换算正确的误报做 suppress
- 对仅 normalization 的低风险分歧降级，不强制人工复核
- 输出：document confidence、parameter confidence、quality tier、review escalation

## 9. 数据模型与数据库
- 中间结构：`parameters.registry`
- 最小可信单元：`parameter_claims`
- 证据对象：`evidence_objects`
- 最终层级：materials -> phases -> conditions -> models -> claims
- 数据库存储：papers、extractions、chunks、parameter_vectors、references、evaluation tables

## 10. 评估与应用场景
- Extraction correctness：字段、数值、单位、引用信息
- Judge correctness：与人工 reviewed queue 对比，评估 judge 准确率与 calibration
- Database utility：检索命中率、问答 grounding、分析任务成功率
- 应用方向：参数数据库、证据约束 RAG、材料/相/机制/工况对比分析

## 11. 推荐演示路线
- 选一篇 DOI
- 展示 parse 后的 sections / tables / references
- 展示 extractor 输出与 postprocess 结果
- 展示 evidence grounding、committee audit、confidence fusion
- 展示最终入库、检索与聊天问答

## 12. 收尾
- CPextractor 不是单一抽取器，而是一条 evidence-grounded curation pipeline
- 关键价值在于：抽取、规范化、审计、置信度建模、数据库落地被放进同一系统
- 适合两个论文方向：scientific IE + evidence-grounded database system
