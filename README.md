# DiffCSP Simple

这是一个独立、可读优先的 DiffCSP 重实现。它不导入原 `diffcsp` 包，不使用
Hydra、PyTorch Lightning、W&B、E3NN 或 PyG。

## 目录

```text
diffcsp_simple/
├─ data.py          # CSV/CIF 读取与缓存
├─ model1.py        # Crysformer、配置类与 CrysformerDenoiser 适配器
├─ model2.py        # 双流 Transformer、配置类与 CrystalDualStreamDenoiser 适配器
├─ model.py         # 原有的 CSPNet（--model cspnet 时使用）
├─ diffusion.py     # 联合晶格/坐标扩散训练与采样
├─ train.py         # 普通 PyTorch 训练循环
├─ evaluate.py      # 测试集重构与结构匹配
├─ sample.py        # 按化学式生成 CIF
├─ utils.py         # checkpoint、CIF 和小工具
└─ requirements.txt
```

## 算法对应关系

- 去噪网络通过 `--model` 选择：
  - `crysformer`（默认）：`model1.py` 中的 Crysformer，全对密集注意力、晶格
    Gram 矩阵 + 傅里叶位移边特征、门控残差；控制参数集中在
    `model1.py` 的 `CrysformerConfig`，训练时按该配置创建模型。
  - `cspnet`：`model.py` 中原来的全连接消息传递网络。
  - `dualstream`：`model2.py` 中的 `CrystalDualStreamTransformer`，一条原子
    不变流（图注意力 + 晶格→原子交叉注意力）和一条晶格等变向量流
    （`[B, 3, 3, D]`，由晶格矩阵列向量初始化）相互更新；控制参数集中在
    `model2.py` 的 `CrystalDualStreamConfig`。`CrystalDualStreamDenoiser`
    负责原子类型嵌入、时间嵌入、padded `[B, N]` 张量、padding mask，以及把
    向量流投影为晶格噪声、把原子流投影为分数坐标 score。
- 时间步编码沿用 `diffusion.py` 的 `SinusoidalTimeEmbedding`：`DiffCSP`
  先算出 `[B, time_dim]` 的时间特征，再传给去噪网络，因此扩散文件保持
  原有接口不变。
- `model1.py` 中的 `CrysformerDenoiser` 只做形状适配：把 DiffCSP 的
  “拼接原子 + 图边界”批量转换为 Crysformer 需要的 padded `[B, N]` 张量与
  node mask，再把每个原子的分数坐标 score 按原顺序摊平。
- `model2.py` 中的 `CrystalDualStreamDenoiser` 采用同样的适配方式。由于
  模型主体只接收不变量原子特征，适配层额外注入时间嵌入和分数坐标傅里叶
  嵌入；晶格噪声来自等变向量流的线性投影，分数坐标 score 来自原子流。
- 训练时把通用超参数（`hidden_dim`、`time_dim`、`num_layers`、`max_atoms`；
  Crysformer 与 cspnet 另用 `num_frequencies`）填入对应模型的配置；
  checkpoint 中保存模型名与配置，评估/采样据此自动重建。
- 与原 DiffCSP 一致：晶格 DDPM、分数坐标 wrapped-normal score matching、
  时间步条件网络、联合晶格与坐标生成。
- 训练使用普通 PyTorch 循环，不再依赖 Lightning 的回调和 Hydra 配置系统。
- 采样保留原算法的 PC corrector + predictor 形式。
- 坐标 score 方差使用固定 Monte Carlo 估计，保留原始归一化思路，但样本数较少。

## 安装

新目录只需要一套最小依赖：

```bash
pip install -r diffcsp_simple/requirements.txt
```

如果需要 GPU 版 PyTorch，请先按照本机 CUDA 版本安装 PyTorch，再安装其余依赖。

## 训练

```bash
python -m diffcsp_simple.train \
  --data-dir data/carbon_24 \
  --output-dir outputs/diffcsp_simple \
  --epochs 1000 \
  --batch-size 128 \
  --device cuda
```

默认使用 Crysformer。对比原 CSPNet 时只需切换模型：

```bash
python -m diffcsp_simple.train \
  --data-dir data/carbon_24 \
  --output-dir outputs/diffcsp_simple \
  --model cspnet \
  --epochs 1000 \
  --batch-size 256
```

使用 `model2.py` 的双流 Transformer：

```bash
python -m diffcsp_simple.train \
  --data-dir data/carbon_24 \
  --output-dir outputs/diffcsp_simple \
  --model dualstream \
  --epochs 1000 \
  --batch-size 256
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--data-dir` | 包含 `train.csv`、`val.csv`、`test.csv` 的目录 |
| `--model` | 去噪模型：`crysformer`（默认）、`dualstream` 或 `cspnet` |
| `--output-dir` | checkpoint 和日志输出目录 |
| `--epochs` | 最大训练轮数 |
| `--batch-size` | 每批晶体数 |
| `--hidden-dim` | 节点与边特征维度（Crysformer/dualstream 需被默认 8 头整除） |
| `--num-layers` | 网络层数 |
| `--num-frequencies` | 边特征傅里叶频率数（Crysformer 与 cspnet 使用） |
| `--max-atoms` | 原子序数词表上限（Crysformer 映射到 `max_atomic_number`） |
| `--timesteps` | 扩散步数 |
| `--patience` | 连续多少轮验证无改善后早停 |
| `--resume` | 从 checkpoint 继续训练 |
| `--limit-train` | 调试时只读取前 N 个训练样本 |
| `--device` | `auto`、`cpu` 或 `cuda` |

`dualstream` 额外使用 `CrystalDualStreamConfig` 中的 `heads=8`、`expansion=4`
和 `dropout=0.0` 默认值，这些暂未暴露为命令行参数。

首次读取 CSV 后会缓存为：

```text
data/<dataset>/.simple_train.pt
data/<dataset>/.simple_val.pt
data/<dataset>/.simple_test.pt
```

训练输出：

```text
outputs/diffcsp_simple/<dataset>/
├─ config.json
├─ metrics.csv
├─ best.pt
└─ last.pt
```

`metrics.csv` 每个 epoch 只有一行，控制台也只打印一行汇总，不产生 Hydra 的
`version_0` 目录或 Lightning 的稀疏指标行。

## 测试集评估

```bash
python -m diffcsp_simple.evaluate \
  --checkpoint outputs/diffcsp_simple/carbon_24/best.pt \
  --data-dir data/carbon_24 \
  --split test \
  --batch-size 16 \
  --max-samples 0 \
  --num-samples 1
```

脚本对测试集做重构采样，并使用 pymatgen `StructureMatcher` 报告匹配率和平均
RMSD；同时按原 CSP 评估流程过滤无效结构和组成，结果写入 checkpoint 同目录的
`eval_metrics.json`。`--max-samples 0` 表示完整测试集，调试时可以改成 `100`。

多次采样时使用：

```bash
--num-samples 20
```

如果不传 `--step-lr`，脚本会按数据集使用原实现的推荐采样步长。

## 按化学式生成

```bash
python -m diffcsp_simple.sample \
  --checkpoint outputs/diffcsp_simple/carbon_24/best.pt \
  --formula C6 \
  --num-samples 10 \
  --output-dir outputs/diffcsp_simple/samples \
  --step-lr 1e-5
```

生成结果保存在：

```text
outputs/diffcsp_simple/samples/C6/C6_0001.cif
```

## 快速调试

只解析少量数据并跑一个 epoch：

```bash
python -m diffcsp_simple.train \
  --data-dir data/carbon_24 \
  --output-dir outputs/diffcsp_simple_smoke \
  --epochs 1 \
  --batch-size 2 \
  --hidden-dim 32 \
  --time-dim 32 \
  --num-layers 1 \
  --limit-train 4 \
  --limit-val 4 \
  --device cpu
```

该版本的实现目标是清晰、可审查、可做 CSP 实验，不追求复刻原仓库的所有可选模块。
