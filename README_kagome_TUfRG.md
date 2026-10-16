# Kagome spinful TUfRG — Chatbox 接手指南（Stage 3 v1.0）

**当前状态：Stage 1（理论）、Stage 2（divERGe 源码和 kernel 审计）、Stage 3（正式代码）均完成；Stage 4（重新验证 PRR 和 SOC 相图）尚未开始。** Stage 3 的最终 PACE 验收作业 `14089745` 通过。此包供后续 Chatbox 在 PACE Phoenix 上以“**给模型参数 → 跑 FRG → 做 diagnosis → 收集结果**”的流程继续研究。

## 1. 文件职责（真正的工作流程只有两步）

| 文件 | 用途 |
|---|---|
| `TUfRG_model.py` | Kagome 模型、hopping、NN/NNN KM SOC、`U,V1,V2,V3`、几何与填充率 |
| `TUfRG_runner.py` | 用 divERGe 运行 flow；输出 `T.dvg`、`O.dvg`、`F.dat`、`metadata.json` |
| `TUfRG_diagnose.py` | **Diagnosis 唯一入口**：验证数据、恢复 full-projected vertex、调用物理 kernel、保存本征模/报告 |
| `TUfRG_physics.py` | PH longitudinal/transverse、Cooper source bases、signed kernel、本征空间及权重 |
| `TUfRG_archive.py` | Diagnosis 内部的只读 `T/O`、网格、FF、二进制版本一致性检查，**无需单独调用** |
| `TUfRG_restore.c` | 调用 divERGe 官方 C projector，从原始 P/C/D 恢复 full-projected `p,d,m`；不是另一套 FRG |
| `stage3_final_gate.py` | Diagnosis 后独立检查 JSON/NPZ、子空间正交性及权重 |
| `run_stage3_diagnose_check.sbatch` | 已验证的 Slurm diagnosis wrapper |
| `run_tufrg_flow.sbatch` | 此交接包新增的 **flow 提交模板**；要求显式提供 flow 参数 |
| `collect_tufrg_results.sh` | 把小型结果打包；不包含十几 GB 的 `O.dvg` |

物理研究流程：

```
[U,V1,V2,V3, lambda_nn,lambda_nnn, filling, mesh, flow controls]
  TUfRG_model.py + TUfRG_runner.py
       -> <FLOW_OUT>/<TAG>/{T.dvg,O.dvg,F.dat,metadata.json}
  TUfRG_diagnose.py  (内部自动使用 archive/physics/C projector)
       -> stage3_diagnosis_<JOBID>/{diagnosis.json,leading_modes.npz,stage3_final_gate.json}
```

## 2. 物理适用范围与重要限制

- 目前 Diagnosis **只支持**三子格、`n_spin=2`、`SU2=0`、spin-conserving intrinsic Kane–Mele NN/NNN SOC（也包括无 SOC），**不支持 Rashba**。Model 中可以生成 Rashba，但不要把其结果交给当前 diagnosis。
- PH 只允许 **self-inverse momenta**：`Gamma, M_a, M_b, M_c`（`--ph-q trim`，默认）；Cooper 为 **零总动量 Γ**。`trim` 不等于全 BZ！任意一般 `q` 需要重新推导并实现 `q,-q` 成对复 source。不可使用 `--ph-q all` 冒充完整相图。
- 使用 Stage 1/2 审定的 spin-resolved physical kernels，不可替换为旧的 `2D-C` spinful shortcut、裸 `D-C`、SVD 直接比较等。`K_PH=T†dT`，`H_PH=K_PH L_PH`；`K_PP=½ T†pT`，`H_PP=-K_PP L_PP`。积分 loop **有符号**，不得裁剪成正定矩阵。次级复特征值不得默默删除；只有经过检查的 leading-real branch 作为指标。
- 输出保存完整 leading eigenspace；简并时 `gap_coefficients` 和 `induced_response_coefficients` 是**投影子空间不变量**，不能只看单条 arbitrary eigenvector。
- 此诊断得到的是 **signed integrated-kernel instability indicators**，**不是**自动 FM/CBO/LCO/SC 相标签或严格热力学易感性证明。判断相须结合 momentum、spin、TR parity、on-site/bond form factor、对称性和 cutoff。SOC 下 physical bond current 不能只用 `Im charge bond` 代替。
- `U,V1` / intrinsic NNN-SOC 的 141-FF 生产 case 已验收；`V2,V3` 虽在 Model 中实现，正式相图仍需各自 benchmark 验证。
- Stage 3 的验收基于 divERGe v0.10 原生库 SHA256：`cd300dd6ccc9b1f972ff8d93a3478db0695ba1465e3423951bdb9674d3e4e69b`。换库/ABI 时不要强行绕过检查。

## 3. PACE 环境（已经使用过的路径）

```bash
RUNS=/storage/home/hcoda1/4/mli826/scratch/divERGe_test_run/runs
PY=/storage/scratch1/4/mli826/conda_envs/diverge-test/bin/python
SRC=/storage/scratch1/4/mli826/divERGe_test_run/official/diverge/src
LIB="$HOME/.local/lib/python3.10/site-packages/diverge/libdivERGe.so"
cd "$RUNS"
```

上述 `RUNS` 常解析到 `/storage/scratch1/4/mli826/divERGe_test_run/runs`（scratch symlink）。账户 `gts-ikimchi3`，分区 `cpu-medium`；示例诊断脚本申请 8 CPU、96 GB、2 小时。**修改前先核对当前 PACE 账户/配额**。不要删除历史 `O.dvg`、`T.dvg`、`metadata.json`、`F.dat`、任何既有 `runs` 子目录或 `archived`。

**交接包的 Model/Runner 是先前提供的原始快照**；PACE 当前这两个文件是实际运行权威版本。如哈希不同，不要把此包自动覆盖 PACE 原文件，应先 diff 并核实来源。

## 4. Step 1 — 在 Slurm 上运行新的 FRG flow

Runner 的**默认 `nk=4,nkf=1,niter=3,filling_mode=fixed_mu` 仅用于小型调试，绝不等于历史生产参数**。每次运行必须记录并明确选择 `nk,nkf,tu_distance,niter,maxvert,Lambda/dLambda`（或明示使用 divERGe Euler 默认）、filling protocol、SOC 和相互作用。复现历史案例时先看该案例原始 `metadata.json`、`F.dat` 和当时的 sbatch，而不是随意从本例推断。

此包附带 `run_tufrg_flow.sbatch`。示例调用（**自行根据目标计算指定 `NITER`；不能把这一示例当作历史重现参数**）：

```bash
cd "$RUNS"
mkdir -p logs
export TAG="my_unique_case_001"  # 必须是尚不存在的新名字，禁止覆盖旧结果
export OUTROOT="$RUNS/my_new_flows"
export U=1 V1=1.5 V2=0 V3=0
export LAMBDA_NN=0 LAMBDA_NNN=0.1
export FILLING_MODE=fixed_filling TARGET_FILLING=0.4166666666666667
export NK=24 NKF=12 TU_DISTANCE=2.01
export NITER="<chosen_integer>"          # 必须替换成实际正整数
export MAXVERT="<chosen_threshold>"      # 必须替换成实际阈值
# 按需指定 LAMBDA0、DLAMBDA0；若未指定就明确使用 divERGe Euler 默认，不能当作历史复现
sbatch --export=ALL run_tufrg_flow.sbatch
```

作业输出在 `$OUTROOT/$TAG/{T.dvg,O.dvg,F.dat,metadata.json}`。它是可直接传给 diagnosis 的 `--case` 目录，不要求叫 `snapshots/final`。请确保 `O.dvg` 已生成（**不能用 `--no-postprocess`**），且 `--tu-channel-calc-project 0`。可以将 `RUNS/my_new_flows/...` 纳入最终阶段的长期 archive，但不要让临时清理脚本扫描这些数据目录。

## 5. Step 2 — 在 Slurm 上运行 Diagnosis

诊断**不重新运行 FRG**。对已存在的旧 case，可跳过 Step 1。使用 Stage 3 已验收的 wrapper：

```bash
cd "$RUNS"
CASE="$RUNS/my_new_flows/my_unique_case_001"  # 或某旧 case 的 .../snapshots/final
[ -f "$CASE/T.dvg" ] && [ -f "$CASE/O.dvg" ] && [ -f "$CASE/metadata.json" ] || exit 1
mkdir -p logs
export PH_Q=trim                # Γ + 3M；可用 Gamma 或 Gamma,M_a 等子集
export STAGE3_REFERENCE=0       # 仅验收过的指定 NNN-SOC 141FF case 才设为 1
unset PROJ                     # 默认从原始 T/O 自动恢复；不依赖旧缓存目录
sbatch --export=ALL,CASE="$CASE" run_stage3_diagnose_check.sbatch
```

注：旧 wrapper 会将结果存入 `$RUNS/stage3_diagnosis_<JOBID>/`，其中包含 `diagnosis.json`（谱和 source 分解）、`leading_modes.npz`（完整本征子空间、原始向量）、`stage3_final_gate.json`（独立验收）。原始 `T/O` **只读不改**。冷启动需要本目录的 `TUfRG_restore.c`、`SRC`、已审计的 `LIB`、GCC/OpenMP。可设置经 provenance 验收的 `PROJ` 缓存以加速，但正常工作不要求缓存。

监控：

```bash
squeue -u "$USER"
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
cat "logs/stage3_diagnose_<JOBID>.out"
cat "logs/stage3_diagnose_<JOBID>.err"
```

两个成功标志：`STAGE3_DIAGNOSIS_PASSED` 和 `STAGE3_FINAL_GATE_PASSED`。若出现 FAILED、`blockers` 非空、原生库 hash 不匹配、复 branch 异常，**不生成相标签，先诊断原因**。

## 6. 输出怎么看、怎么打包给下一个 Chatbox

1. **`diagnosis.json`**：先读 `status, blockers, warnings, parameters, resolved_parameters, q_coverage, sector_candidates`；对每个 `q` 再看 longitudinal/transverse/Cooper `leading_real, leading_multiplicity, gap_coefficients, induced_response_coefficients, nonreal_count, sanity`。`sector_candidates` 只是排序表，不等于 phase diagram。
2. **`leading_modes.npz`**：真正的 mode/subspace 数组；简并时重点看 `*_gap_basis` 和 `*_response_basis`，配合 JSON 的 `source_labels` 和几何作 symmetry/order 分析。
3. **`stage3_final_gate.json` + Slurm `.out/.err`**：确认物理源归一化、子空间正交性、代码真正跑完。对历史 flow 另读 `metadata.json, F.dat` 和临界 cutoff；**本代码默认只看最终 snapshot，不自动给完整 cutoff 随 Λ 的演化图**。

在 `$RUNS` 执行（`JOBID` 替换为本次 diagnosis 作业号）：

```bash
cd "$RUNS"
JOBID=<JOBID>
CASE="$RUNS/my_new_flows/my_unique_case_001"
bash collect_tufrg_results.sh "$JOBID" "$CASE"
```

会生成 `tufrg_diagnosis_<JOBID>_collect.tar.gz`，内容包括小型诊断结果、该 case 的 metadata/可用 F.dat 和对应日志；**不包含 `O.dvg/T.dvg`**。把包交给 Chatbox，之后它应先核查 `status` 与 coverage，再谈物理。原始大文件仍保存在 PACE。

## 7. 长期交接与注意事项

- 一次性交接：此 package + `PRR_to_Kagome_Kane_Mele_SOC_Theory_Note.pdf` + `divERGe_Kagome_KM_Stage2_Verification.pdf` + `kagome_TUfRG_Stage3_signoff.md`；新 Chatbox 只需询问物理参数、预期计算区域、RG 参数/已有 metadata，然后生成 sbatch/分析结果。
- `kagome_TUfRG_PACE_handoff.tar.gz` 可由附带的 `make_pace_handoff.sh` 从 **PACE 实际在用的文件** 重新封装；不修改原文件。
- 禁止用此工具直接宣称全 BZ 主导相或 LCO。Stage 4 必须先在四个 no-SOC PRR benchmark 上重新验证 CBO/FM/A1/E2，再做 SOC scan 与对称性判定。
- 有问题先对照 Stage 1、Stage 2 技术文档和 Stage 3 final gate；不要回退使用旧 `TUfRG_instability.py` 进行正式诊断。
