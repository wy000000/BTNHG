# Data directory

Place the BTNHG dataset files here, or point the environment variable
`BTNHG_DATA_PATH` at an existing data directory.

Download: <https://www.kaggle.com/datasets/wy000000/btnhgv2>

Expected files:

| file | content |
|---|---|
| `addressFeature.csv` | address node features (`addressID` + features) |
| `coinFeature.csv` | coin/UTXO node features (`coinID` + features) |
| `TxFeature.csv` | transaction node features (`txID` + features) |
| `hgEdgeV2.csv` | heterogeneous edges + `clusterID` labels |
| `addressTimeData.csv` | per-address per-block activity records (temporal view) |

Generated artefacts (safe to delete):

* `addressTimeFeature_dataSet.pt` — cached compressed ATFD tensor (14 features);
* `NEW2_addressTimeFeature_compress_no_interval_dataSet.pt` — cached 13-feature tensor;
* `NEW1_addressTimeFeature_raw_dataSet.pt` — cached uncompressed tensor (~4.5 GB);
* `result/` — one timestamped folder per five-fold round.
