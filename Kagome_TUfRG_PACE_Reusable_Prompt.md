# Kagome TUfRG / divERGe：PACE 通用提交与诊断指令模板

> **用途**：把下面的「可复制主提示词」完整交给一个新的 Chatbox，再填写最下方的参数表。它应该在**不重新发明物理模型或 diagnosis 方法**的前提下，生成在 Georgia Tech PACE 上可执行的 Slurm flow 和 diagnosis 脚本。
>
 **来源与核对**：已逐项核对现存的 `TUfRG_runner.py`、`TUfRG_model.py`、`TUfRG_diagnose.py`、`TUfRG_archive.py`、`stage3_final_gate.py`，以及原始 `tufrg_nk32_nkf16_5point.sbatch` 和 `tufrg_stage3_nk32_nkf16_5point.sbatch`。此文件是工作流程模板，不代表已经提交新任务。

---

## 一、可复制主提示词（直接转发给新 Chatbox）

```text
你将协助我在 Georgia Tech PACE 集群上使用既有 Kagome TUfRG / divERGe
Python pipeline 完成新的参数扫描、RG flow 以及随后的物理 diagnosis。

重要：我不是请你从零写 fRG 或者重建一套诊断框架。请严格复用已经
验证的源代码和 API，生成我可以粘贴到 PACE 终端执行的 Slurm 脚本、
预检查命令、提交命令及收集结果命令。不要在当前 Chatbox 里实际启动计算。

【A. 项目目录（先检查，禁止猜测）】
PACE 工作目录：
  /storage/home/hcoda1/4/mli826/scratch/divERGe_test_run/runs
Python 环境：
  /storage/scratch1/4/mli826/conda_envs/diverge-test/bin/python
divERGe C 源头文件目录（用于 diagnosis 的 C bridge）：
  /storage/scratch1/4/mli826/divERGe_test_run/official/diverge/src
生产环境共享库（需通过审计哈希验证）：
  /storage/home/hcoda1/4/mli826/.local/lib/python3.10/site-packages/diverge/libdivERGe.so

PACE 已有且必须复用的文件（通常在 runs/ 根目录）：
  TUfRG_model.py          — Kagome spinful KM + U/V1/V2/V3 模型、填充数及网格
  TUfRG_runner.py         — 主 flow CLI，写 T.dvg、F.dat、O.dvg、metadata.json
  TUfRG_archive.py        — Stage-3 输入兼容性/哈希预检查
  TUfRG_physics.py        — Stage-3 物理核和模式分析
  TUfRG_diagnose.py       — Stage-3 diagnosis 主 CLI
  TUfRG_restore.c         — 官方 divERGe 原生数据投影桥
  stage3_final_gate.py    — 独立的 Stage-3 结果一致性检查

如果我另外提供 publication archive，则这七个源文件在 archive 的 code/
下，而不是自动出现在 PACE 的 runs/ 中。必须检查实际部署位置，不能仅凭
archive 的存在就认为 PACE 可以直接运行。

【B. 已验证的计算链】
第一步：TUfRG_runner.py 进行 divERGe TU flow (P/C/D)，
        输出每个参数点的 T.dvg、F.dat、metadata.json、最终 O.dvg。
第二步：TUfRG_diagnose.py 对同一参数点最终 O.dvg 作只读诊断，
        使用 TUfRG_restore.c + 官方 divERGe 头文件 + 审计过的 .so。
第三步：stage3_final_gate.py --out-dir <diagnosis_dir>，
        对 diagnosis.json 和 leading_modes.npz 作独立一致性检查。
第四步：仅收集小型数值输出及文本报告用于后续检查；不要压缩或复制
        数十 GB 的 O.dvg，也不要自动删除原始文件。

【C. 物理和 API 约定（不得擅自改）】
1. 模型：三子晶格 Kagome、显式自旋 n_spin=2、SU2=False；t=1
   时实 NN hopping = -t。默认 vertex convention = official_s1minus1。
2. 交互：--U、--V1、--V2、--V3；旧参数 --V 是 --V1 的别名。
3. SOC：--lambda-nn、--lambda-nnn、--lambda-rashba。
   若一次 convergence scan 中每个 task 的 SOC、Nk/Nkf、TU cutoff
   不同，必须为每个 task 单独传参；不能用 runner 的 `--case`
   选项假装它支持逐 case 改变全局 mesh/SOC。
4. 填充：--filling-mode fixed_filling + --target-filling 0.4166666666666667
   表示 spinful 全六带希尔伯特空间中的 5/12；不同于 fixed_mu 模式。
   若用户选择 fixed_mu，则须改用 --filling-mode fixed_mu 和 --mu，
   不能同时声称固定填充。
5. 网格：--nk 是粗 q 网格每方向点数，--nkf 是精细 loop 倍数，
   --tu-distance 是 TU real-space form-factor cutoff；不能混淆 nk/nkf。
6. Flow：默认 --flow-mode tu、--channels PCD，Euler 参数
   --niter、--maxvert、--dLambda-fac 等从我的参数表读取。
7. 为现有 Stage-3 raw-channel diagnosis 保持
   --tu-channel-calc-project 0 和 --vertex-convention official_s1minus1。
   必须保留最终 O.dvg；使用 --no-postprocess 会令这一套 diagnosis 无法运行。
8. 现有 TUfRG_diagnose.py 只支持 intrinsic NN/NNN Kane-Mele（包括
   两者为零）、无 Rashba、自旋守恒、2D、偶数 nk、方形未压缩的 q 网格。
   只检查 Γ 和 M 的 TRIM particle-hole 以及 Γ Cooper。
   若 lambda_rashba != 0、nk 为奇数、需要一般非 TRIM Q 或需要
   不同诊断物理，必须明确报告“不在当前 Stage-3 适用范围”，
   不能直接使用这套 diagnosis 得出结论；应先征求我是否开发扩展。
9. 现有 diagnosis 是 signed flow-integrated loops 的 sector/mode
   比较，不是严格的热力学相变证明，也不能仅凭一个虚键权重就宣称
   已验证物理电流。不可根据预计相标签强行指定 leading order。
10. 可使用中性的 case tag（如 U3p0_V1p5），不要在输出路径里
    暗示未经数据验证的 LCO / SC / CBO 等结论。

【D. PACE 的历史可信基准（可以作为默认，但不应硬编码所有以后任务）】
上一次 Nk=32、Nkf=16 的 5-point convergence 测试为：
  #SBATCH --account=gts-ikimchi3
  #SBATCH --partition=cpu-medium
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=160G
  #SBATCH --time=72:00:00
  #SBATCH --array=0-4
  无额外 --qos；请不要猜测 QoS。
  通用环境变量：
    OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    DIVERGE_OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    DIVERGE_FFTW_NUM_THREADS=$SLURM_CPUS_PER_TASK
    DIVERGE_OMP_IMPLICIT_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
    PYTHONUNBUFFERED=1
  精确的历史 flow 参数：
    --nk 32 --nkf 16 --tu-distance 2.01
    --lambda-nn 0.5 --lambda-nnn 0.0
    --filling-mode fixed_filling --target-filling 0.4166666666666667
    --niter 120 --maxvert 100 --dLambda-fac 0.1
    --tu-channel-calc-project 0 --vertex-convention official_s1minus1
  历史 Stage-3 diagnosis 使用同样的 8 cores、160G、72h，
  并用 `--ph-q trim`；对于 SOC 非零的数据使用 `--require-soc`。
  以上是已成功运行的一个规模，不代表任何尺寸或参数都固定使用该资源。
  Nk32/Nkf16 时一个 O.dvg 约 25GB；扫描时要检查容量。

【E. 你必须生成的交付物】

首先：若缺少对结果有实质影响的参数（参数点列表、SOC、filling
模式、Nk/Nkf、资源预算等），请只问清楚缺失部分，不要私自补齐。

然后按顺序交付：

1. 一个只读 PACE preflight 命令块：
   - 检查 runs/ 中七个文件、Python 路径、divERGe import 和 --help；
   - 检查 diverge_model.h、libdivERGe.so；
   - 检查 SHA256 与 TUfRG_archive.py 中审计值一致；
   - 检查拟使用的输出文件夹不存在，磁盘容量足够；
   - 检查目录是否有旧任务，不能覆盖。
   如果用户还没给 PACE 文件树，让我运行 `ls`/`find` 等命令
   并贴回输出，不能假装已验证。

2. `slurm/<unique_name>_flow.sbatch` 的**完整内容**：
   - 以 Slurm array 一点一任务；array 范围按实际点数自动适配；
   - #SBATCH 路径及 CPU/memory/time 与所选资源匹配；
   - 执行真实已有的 TUfRG_runner.py，不创建另一套求解器；
   - 清楚传递每个 U、V1、V2、V3 及模型/SOC/mesh 参数；
     支持不同 task 具有不同 Nk、Nkf 或 SOC 的 convergence scan；
   - 用唯一 tag 和独立 OUTROOT；拒绝覆盖旧目录；
   - 显式传递 --lambda-rashba 0（若适用）、--tu-channel-calc-project 0；
   - 未指定的非默认物理参数不能被悄悄省略；
   - 用 `set -eo pipefail` 或经本地环境验证的安全 shell 设置，
     避免登录环境因为 `set -u` 和 /etc/bashrc 冲突；
   - 限制 OpenBLAS/MKL 线程；不要在一个 array task 内把整组点串行计算。

3. `slurm/<unique_name>_diagnose.sbatch` 的**完整内容**：
   - 只在 flow 成功、每个 case 的 T.dvg / O.dvg / metadata.json
     均存在且非空时执行；与 flow case list 一一对应；
   - diagnosis 输出在独立目录，不准写回原 case；
   - 调用（参数名称必须准确）：
       python -u TUfRG_diagnose.py \
         --case "$CASE" --out-dir "$DIAGOUT" \
         --diverge-src "$DIVERGE_SRC" --binary "$DIVERGE_BINARY" \
         --ph-q trim
     在 lambda_nn 或 lambda_nnn 非零时，加 `--require-soc`；
     在两者都为零时，不得加这个 flag；
   - 随后执行 `python -u stage3_final_gate.py --out-dir "$DIAGOUT"`；
   - 编译桥文件必须匹配原始 .so 与 header；不要静默使用其他版本；
   - 不用仅含 F.dat、T.dvg、leading_modes.npz 的 publication 小归档
     冒充可直接从头运行 Stage-3 diagnosis 的完整输入。

4. PACE 终端执行的逐步命令：
   - 从 runs/ 目录预建 logs/、diagnosis/logs/；
   - 用 `sbatch` 提交 flow，显示 `squeue`、`sacct` 和日志检查命令；
   - 等 flow 完成并验证每个任务成功后，单独提交 diagnosis，
     或提供带 `afterok` 的 dependency（但说明整组 array 的依赖语义）；
   - 验证报告和 leading_modes.npz 存在，收集小型结果包的命令。

5. 简短解释：
   - 输出文件结构、各文件的含义；
   - 此次相对历史基准改动的参数；
   - 资源/存储与物理诊断适用性风险；
   - 哪些事情是“已在脚本中保证”，哪些必须上 PACE 才能验证。

【F. 禁止事项】
- 不要在未授权情况下实际 `sbatch`、运行 FRG 或进行新数值运算。
- 不要重写 `TUfRG_model.py`、`TUfRG_runner.py` 或 diagnosis 核心代码，
  除非我明确要求源代码改动且先做兼容性审计。
- 不要将 `O.dvg` 复制到每个 diagnosis 输出目录（直接原地读取）。
- 不要自动删除 `O.dvg`、snapshot、运行输出或其他研究数据。
- 不要因为文件名里出现“LCO/SC”等字样就把它写成已确认相。
- 不要复用已经存在的 OUTROOT/DIAGROOT 并覆盖旧结果。
- 不要把另一种 N-patch FRG、VMC 或 mean-field pipeline 混入此任务。

【G. 本次任务的参数】
请读取本文下方的“二、任务参数单”，据此生成代码；
如果我在对话中给出的具体参数与历史基准不同，以我本次参数为准。
```

---

## 二、任务参数单（每次新实验只修改这一块）

复制以下字段并填写；可以一次给多个 `U,V1,V2,V3` 点。

```yaml
project: kagome_TUfRG_divERGe
job_label: "<new_unique_scan_name>"

physics:
  t: 1.0
  filling_mode: fixed_filling       # fixed_filling | fixed_mu
  target_filling: 0.4166666666666667 # 仅 fixed_filling 使用
  mu: null                         # fixed_mu 时填写；fixed_filling 时不用
  filling_ngrid: 120
  lambda_nn: 0.5
  lambda_nnn: 0.0
  lambda_rashba: 0.0                # !=0：现有 Stage-3 诊断不适用

mesh:
  nk: 32                           # Stage-3 需要偶数
  nkf: 16
  tu_distance: 2.01

flow:
  niter: 120
  maxvert: 100.0
  dLambda_fac: 0.1
  dLambda_fac_scale: 1.0
  flow_mode: tu
  channels: PCD
  tu_channel_calc_project: 0
  vertex_convention: official_s1minus1
  keep_final_O_dvg: true

cases:                              # 仅为演示：实际运行时替换为本次的点
  - {name: U3p0_V1p5, U: 3.0, V1: 1.5, V2: 0.0, V3: 0.0}
  - {name: U3p0_V0p0, U: 3.0, V1: 0.0, V2: 0.0, V3: 0.0}
  - {name: U0p0_V2p0, U: 0.0, V1: 2.0, V2: 0.0, V3: 0.0}
  # 若做 mesh 收敛检验，可为某个 case 额外设置 nk/nkf/tu_distance，
  # 或提供 (U,V1,...,nk,nkf,lambda_nn,lambda_nnn) 的逐 task 表。
  # 此时 Chatbox 必须生成逐 task 的完整调用及唯一 tag。

pace:
  account: gts-ikimchi3
  partition: cpu-medium
  qos: null                       # 过去 32x16 的脚本没有显式 QoS
  cpus_per_task: 8
  mem_per_task: 160G
  time_per_task: "72:00:00"
  array_max_concurrent: null       # 若需限制同时运行数量，在提交前明确设置

diagnosis:
  enabled: true
  ph_q: trim                       # Gamma + 3M；不是全 BZ 扫描
  stage3_gate: true
  collect: ["F.dat", "T.dvg", "metadata.json", "diagnosis.json", "leading_modes.npz", "stage3_final_gate.json"]
  include_large_O_in_upload: false

safety:
  never_overwrite: true
  never_delete_original_data: true
  submit_jobs_only_after_my_confirmation: true
  ask_if_unsupported_diagnosis: true
```

---

## 三、已核实的命令接口（给 Chatbox 交叉检查）

以下调用形态来自历史已经使用的源文件。`<...>` 为未来脚本生成时必须替换的值，不是可以照抄运行的 shell。

**Flow：每个数组任务运行一个参数点。**

```bash
"$PY" -u TUfRG_runner.py \
  --U "$U" --V1 "$V1" --V2 "$V2" --V3 "$V3" \
  --t "$T" \
  --lambda-nn "$LAMBDA_NN" --lambda-nnn "$LAMBDA_NNN" \
  --lambda-rashba "$LAMBDA_RASHBA" \
  --filling-mode fixed_filling --target-filling "$TARGET_FILLING" \
  --filling-ngrid "$FILLING_NGRID" \
  --nk "$NK" --nkf "$NKF" --tu-distance "$TU_DISTANCE" \
  --niter "$NITER" --maxvert "$MAXVERT" \
  --dLambda-fac "$DLAMBDA_FAC" --dLambda-fac-scale "$DLAMBDA_FAC_SCALE" \
  --flow-mode tu --channels PCD \
  --tu-channel-calc-project 0 \
  --vertex-convention official_s1minus1 \
  --tag "$TAG" --outdir "$OUTROOT"
```

`fixed_mu` 模式必须由新 Chatbox 明确替换上述 filling 相关 flags。

**Diagnosis：独立脚本，使用 flow 生成的完整原始 case 目录。**

```bash
"$PY" -u TUfRG_diagnose.py \
  --case "$CASE" \
  --out-dir "$DIAGOUT" \
  --diverge-src "$DIVERGE_SRC" \
  --binary "$DIVERGE_BINARY" \
  --ph-q trim \
  --require-soc  # 仅当 NN/NNN intrinsic SOC 至少一个非零；否则删除本行并去掉上一行反斜杠

"$PY" -u stage3_final_gate.py --out-dir "$DIAGOUT"
```

**注意：`TUfRG_archive.py` 会核对生产库 SHA256。当前归档源代码要求：**

```text
cd300dd6ccc9b1f972ff8d93a3478db0695ba1465e3423951bdb9674d3e4e69b
```

若 PACE 上的 `libdivERGe.so` 哈希不同，停止 Stage-3 diagnosis 并核对所用版本；不要为了绕过检查擅自修改验证常量。

## 四、输出与解释的边界

| 文件 | 产生阶段 | 用途 | 是否适合上传小归档 |
| --- | --- | --- | --- |
| `T.dvg` | Flow | 模型与网格二进制快照 | 是 |
| `F.dat` | Flow | 每一步的 cutoff、vertex/channel 最大值 | 是 |
| `metadata.json` | Flow | 参数、解析出的化学势、运行状态、路径 | 是 |
| `O.dvg` | Flow postprocess | 完整 P/C/D 数据；Stage-3 的输入 | **否，通常极大** |
| `diagnosis.json` | Stage-3 | 诊断数值和报告 | 按需要；不是原始数据 |
| `leading_modes.npz` | Stage-3 | 模式数值数组、分量标签 | 是 |
| `stage3_final_gate.json` | Stage-3 gate | 一致性检查结果 | 按需要 |

原来准备 publication repo 的 compact 三点归档只有 `T.dvg/F.dat/metadata.json/leading_modes.npz`，没有 `O.dvg`：它可用于查看已保留的模式数组，**不能重新计算完整 Stage-3 eigenproblem**。若任务是生成新的完整 diagnosis，必须找到 PACE 上对应的原始 `O.dvg`。

模板中的三个参数点只是填表示例；不代表在任意新的 SOC 或网格下已验证了相标签。尤其是 `(U,V1)=(0,2)`，不得直接把 archive 文件名或历史猜测当作“SC leading”的证明。
