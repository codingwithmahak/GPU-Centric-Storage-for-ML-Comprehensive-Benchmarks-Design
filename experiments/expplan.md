# Experiment Plan & Matrix

## Factors
- Storage: local NVMe, NAS (NFS), object store (S3-compatible).
- Formats: Parquet, ORC, TFRecord, WebDataset (tar shards).
- Pipelines: PyTorch DataLoader, DALI, FFCV; Spark CPU vs RAPIDS+UCX.
- Caching: none, OS page cache, node-local NVMe cache (copy-on-read), RAPIDS FileCache.
- Concurrency: dataloader workers (w), prefetch depth (p), batch size (b).

## Matrix (example)
| Storage | Format     | Pipeline     | Cache | w | p | b |
|---------|------------|--------------|-------|---|---|---|
| NVMe    | WebDataset | DALI         | yes   | 8 | 4 | 256|
| S3      | Parquet    | PyTorch base | none  | 8 | 2 | 128|
| NFS     | TFRecord   | FFCV         | yes   | 16| 8 | 256|

## Method
- 5 runs per cell, discard 1 warm-up.
- Record: throughput, GPU util (nvidia-smi), input stall %, epoch time, CPU util.
- Save CSVs to `results/` and push to GitHub.
