# 正式实验完整性与质量审计

- 实验：`governance-formal-v2-calibration`
- 总体结论：**通过**
- 分片：110 / 110
- 治理证据：3,861,000 行；估值记录：520,000 行
- 误差分解最大绝对残差：`2.220e-16`

## 检查项

| 检查 | 结果 |
|---|---|
| `shard_plan_count_is_110` | 通过 |
| `completion_marker_count_matches_plan` | 通过 |
| `no_root_failed_markers` | 通过 |
| `no_attempt_failures_in_selected_results` | 通过 |
| `all_shards_have_expected_evidence_rows` | 通过 |
| `evidence_primary_key_unique` | 通过 |
| `valuation_primary_key_unique` | 通过 |
| `ok_quantifier_rows_have_finite_estimates` | 通过 |
| `failed_quantifier_rows_do_not_carry_estimates` | 通过 |
| `estimates_are_in_unit_interval` | 通过 |
| `error_decomposition_residual_within_1e_10` | 通过 |
| `all_expected_seeds_present` | 通过 |
| `all_expected_protocols_present` | 通过 |
| `all_expected_policies_present` | 通过 |
| `all_expected_quantifiers_present` | 通过 |
| `artifact_rows_are_seed_by_protocol` | 通过 |
| `artifact_gate_has_no_failures` | 通过 |

## 解释边界

完整性审计通过只表示冻结分片、主键、数值和制品一致。量化器在预注册条件下不可定义时仍会保留失败状态，是否进入某张统计表由逐组合纳入规则决定，不能用插值或回退值替代。
