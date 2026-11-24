# BTNHG

**Bitcoin Transaction Network Heterogeneous Graph** — reference code for the paper
*Task-Matched Simplicity: A Lightweight Temporal-Feature Network for Bitcoin Address Classification*.

The repository contains two parallel views of the same 15,218 labeled Bitcoin
addresses over 14 address-cluster classes, and two parallel training frameworks
that share one loss, one evaluation procedure, and one reporting format:

| view | representation | models |
|---|---|---|
| **graph** | heterogeneous graph of `address` / `coin` / `tx` nodes (546,649 / 2,266,449 / 1,000,468 nodes, 5,971,294 raw edges, 11,942,588 after adding reverse edges) | GCN, GAT, SAGE, HAN, HGT, RGCN, SAGETF, GINETF |
| **temporal** | address-temporal feature dataset (ATFD): each address is a block-ordered tensor of 13 per-block features plus 1 interval channel, sparse-compressed along time | CNN2D, Transformer-only, depthwise CNN variants, **CNN1D_DW_TF** (proposed) |

Dataset: <https://www.kaggle.com/datasets/wy000000/btnhgv2>

---

## 1. News / version history

* `2026.11.24.1` — repository completed for the paper release: added the raw-sequence
  and no-interval ablations (`NEW1_*`, `NEW2_*`), the two strongest graph baselines
  (`SAGE_TF.py`, `GINE_TF.py`), a ten-round driver loop for both frameworks, a
  working default model in `addressTF.py`, and a portable `dataPath`.
* `2026.1.26.1` — added learning-rate scheduler.
* `2026.1.20.1` — training accuracy is now recorded (overfitting monitoring).
* `2026.1.14.1` — fixed the bug where cross-validation did not re-initialize the model.
* `2025.12.18.1` — added k-fold cross-validation.
* `2025.12.6.1` — initial release on the BTNHG2013v2 dataset (HAN, HGT, RGCN, SAGE, GAT, GraphConv).

---

## 2. Dataset

Download the dataset from Kaggle and put the CSV files in `data/` (the default
`BTNHGV2ParameterClass.dataPath`), or point `BTNHG_DATA_PATH` at an existing
directory:

```bash
# Windows (PowerShell)
$env:BTNHG_DATA_PATH = "D:\BTNHG\BTNHGV2"

# Linux / macOS
export BTNHG_DATA_PATH=/data/BTNHGV2
```

Files expected in the data directory:

| file | used by | content |
|---|---|---|
| `addressFeature.csv` | graph | address node features + `addressID` |
| `coinFeature.csv` | graph (and GINETF edge features) | coin node features + `coinID` |
| `TxFeature.csv` | graph (and GINETF edge features) | transaction node features + `txID` |
| `hgEdgeV2.csv` | graph | heterogeneous edges + `clusterID` labels |
| `addressTimeData.csv` | temporal | per-address, per-block activity records |

Observation window: block heights 272,375–277,995 (5,621 consecutive blocks, 5,504
of which contain transactions). 15,218 of 546,649 parsed addresses carry a label.
The temporal framework caches its built tensor as `addressTimeFeature_dataSet.pt`
(next to the CSVs); delete it to force a rebuild.

---

## 3. Requirements

Python 3.10+ and:

```bash
pip install torch torch-geometric transformers pandas numpy scikit-learn \
            matplotlib xlsxwriter openpyxl psutil py-cpuinfo
```

A CUDA GPU is required for the temporal models (the uncompressed sequence does not
fit in GPU memory — that is itself a reported result).

---

## 4. Repository layout

```
BTNHGV2.py                      graph driver: eight baselines x runs x 5-fold
BTNHGV2ParameterClass.py        all hyperparameters, paths, seeds
BTNHGV2HeteroDataClass.py       loads the CSVs into a PyG HeteroData
HeteroModelTrainerTesterClass.py  graph training/testing/k-fold loop
ExtendedNNModule.py             nn.Module subclass holding batch size/seeds
EarlyStoppingClass.py           early stopping (patience / min_delta)
resultAnalysisClass.py          metrics, confusion data, timestamped result folders

Graph baselines (paper Table 3)
  GraphConvClass.py             GCN   (PyG GraphConv: W1 x_i + W2 sum_j x_j, unnormalised sum)
  GATClass.py                   GAT   (masked self-attention over neighbourhoods)
  SAGEClass.py                  SAGE  (inductive mean aggregation)
  HANClass.py                   HAN   (node-level + meta-path-level attention)
  HGTClass.py                   HGT   (heterogeneous node/edge-type attention)
  RGCNClass.py                  RGCN  (per-relation weight matrices)
  SAGE_TF.py                    SAGETF (SAGE + TransformerConv fusion)
  GINE_TF.py                    GINETF (GINEConv with edge features + TransformerConv)

Additional / experimental graph models (not reported in the paper)
  HeteroGCNClass.py, GraphConvClass-based variants, GPSClass.py, GraphGPSClass.py

addressTimeFeature/             temporal framework (ATFD + CNN1D_DW_TF family)
  addressTF.py                  main driver (default: CNN1D_DW_TF)
  addressTimeDataClass.py       builds the ATFD tensor + time compression
  addressTimeFeatureClass.py    13 per-block features per address
  DataSetModelTrainerTesterClass.py  temporal training/testing/k-fold loop
  nnModule.py                   SEBlock, ConvPE, MambaBlock
  CNN2D.py                      CNN2D baseline (2-D conv over time x feature)
  CNN1D_TF_class.py             Transformer encoder only
  CNN1D_DW_class.py             depthwise 1-D conv + flatten + classifier
  CNN1D_DW_PW_class.py          depthwise + pointwise
  CNN1D_DW_SE_class.py          depthwise + squeeze-and-excitation
  CNN1D_DW_TF_class.py          **CNN1D_DW_TF (proposed)**
  CNN1D_DW_SE_TF_class.py       depthwise + SE + Transformer
  CNN1D_DW_SE_PE_TF_class.py    + residual depthwise conv ("PE")
  CNN1D_DW_SE_PE_TF_CLS_class.py  + CLS-token readout
  CNN1D_DW_SE_PE_MB_class.py    + Mamba-style gated block
  NEW1_addressTimeDataRawClass.py            raw sequence (no compression)
  NEW1_addressTF_raw_sequence.py             driver for the OOM row of Table 1
  NEW2_addressTimeDataCompressNoIntervalClass.py  compression without interval channel
  NEW2_addressTF_compress_no_interval.py     driver for the no-interval row
  simple2DCNNClass.py           backwards-compatible alias of CNN2D.py
```

Standalone demos (not part of the paper): `example.py` (toy GCN on synthetic graphs),
`test.py`, `testFunction.py`, `test.ipynb`.

Every model class name in the paper's tables is the class name in this repository
(`CNN1D_DW_TF_class` → `CNN1D_DW_TF`, `SAGE_TF` → `SAGETF`, `GINE_TF` → `GINETF`,
`GraphConvClass` → `GCN`, `simple2DCNNClass` in `CNN2D.py` → `CNN2D`).

---

## 5. Quick start

### 5.1 Graph baselines

```bash
python BTNHGV2.py
```

Select the model with `build_model()` in `BTNHGV2.py` (one uncommented line).
`BTNHGV2ParameterClass.graph_run_times` (default 10) controls how many
five-fold rounds are run; each round re-draws the fold partition and re-initializes
the parameters.

### 5.2 Proposed temporal network and its ablations

```bash
python addressTimeFeature/addressTF.py
```

Default model: `CNN1D_DW_TF_class`. Uncomment another line in `build_model()` for the
structure ablation of Table 2. `BTNHGV2ParameterClass.addressTF_run_times`
(default 10) controls the number of rounds.

### 5.3 Representation ablation (Table 1)

```bash
# row 1: full 5,621-step sequence (runs out of GPU memory, as reported)
python addressTimeFeature/NEW1_addressTF_raw_sequence.py

# row 2: compressed, no interval channel (13 features)
python addressTimeFeature/NEW2_addressTF_compress_no_interval.py
```

### 5.4 Hyperparameter sweeps (Table 4 / Table 5)

Edit `BTNHGV2ParameterClass` and re-run `addressTF.py`:

* learning rate — `lr` ∈ {0.005, 0.010, 0.020, 0.035, 0.050, 0.060};
* encoder depth — `tf_num_layers` ∈ {1, 2, 3, 4}.

Because the model constructors bind these values as default arguments at import
time, change them in the parameter file (not at runtime) and start a fresh process.

### 5.5 Interactive notebooks

* `BTNHGV2For_a_nn.ipynb` — graph view.
* `addressTimeFeature/addressTimeDataFor_a_nn.ipynb` — temporal view.

---

## 6. Training protocol

Identical for both frameworks:

* loss: cross-entropy; stratified 5-fold cross-validation, re-initialized per fold;
* optimizer: AdamW, `weight_decay=1e-4`, `lr=0.05` (selected);
* schedule: cosine decay with a 10% linear warm-up over the epoch budget;
* batch size 4096, at most 512 epochs;
* early stopping: patience 128 epochs, `min_delta=0.01`, monitored on the
  **training** loss, best weights restored before testing;
* one "round" = one complete 5-fold partition; the paper reports 10 rounds
  (50 training runs per configuration).

Each round writes a timestamped folder under `<dataPath>/result/`:

```
<ModelClass>-kFold-<YYYY.MM.DD HH.MM.SS>-acc <mean>/
    extendedAttributes.txt        environment + configuration + metrics
    kFold_evaluationMetrics.xlsx  per-fold metrics and the 5-fold average
    model.state_dict.pt           best state dict of the round
    <ModelClass>.py, BTNHGV2ParameterClass.py   snapshot of the code used
```

**Peak Acc** = highest accuracy among the 5 folds of a round. **Std** = standard
deviation of the 10 round-level mean accuracies (compute from the ten folders).

---

## 7. Implementation notes (differences that matter)

* **Interval feature.** After compression, column 0 stores the gap *to the previous*
  active block, `delta_t = t_t - t_{t-1}`, with `t_0 = 272375` (window start), so
  `delta_t = 1` exactly when two retained blocks are consecutive. It is prepended to
  the 13 block features, giving 14 channels.
* **13 block features.** `blockID`, coin counts/amounts (total, received, spent,
  difference), transaction counts (total, received, spent, difference). The three
  amount columns are stored as `log(x + 1e-10)`.
* **No positional encoding** is used in `CNN1D_DW_TF`: temporal order enters through
  the depthwise convolution and through the position-specific linear readout that
  flattens the encoder output.
* **Padded rows are not masked.** Compression pads each address to
  `L = max_i L_i` rows with zeros; those rows take part in the convolution,
  self-attention, and flattens. They are all-zero and their interval entry is 0,
  while real rows carry `delta >= 1` (except a first activity in block 272375).
* **Channel count.** `C = 14` doubles as the Transformer `d_model` (single head,
  `dim_feedforward=32`), so the classifier input is `14 * L`.

---

## 8. Citation

```bibtex
@article{wu2026taskmatched,
  title  = {Task-Matched Simplicity: A Lightweight Temporal-Feature Network
            for Bitcoin Address Classification},
  author = {Wu, Yan and Ge, Chen and Yin, Shuo and Xiao, Saisai and Zhou, Lei
            and Liu, Lu and Pan, Senshan},
  year   = {2026},
  note   = {Manuscript}
}
```

---

## 9. 中文速览

* **数据**：从 Kaggle 下载 `btnhgv2`，把 5 个 CSV 放到 `data/`（或设置环境变量
  `BTNHG_DATA_PATH` 指向已有目录）。
* **图基线**：`python BTNHGV2.py`，在 `build_model()` 里取消注释选择 GCN/GAT/SAGE/
  HAN/HGT/RGCN/SAGETF/GINETF。
* **主模型与结构消融**：`python addressTimeFeature/addressTF.py`，默认即
  `CNN1D_DW_TF`；换模型同样改 `build_model()`。
* **表示消融**：`NEW1_addressTF_raw_sequence.py`（原始 5,621 步，会 OOM）、
  `NEW2_addressTF_compress_no_interval.py`（压缩但不加区间通道）。
* **十轮协议**：`BTNHGV2ParameterClass.addressTF_run_times` / `graph_run_times`
  默认 10，快速冒烟测试改成 1。
* **结果**：每轮写入 `data/result/<模型>-kFold-<时间>-acc <均值>/`。
