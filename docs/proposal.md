# Proposal: GPU-Centric Storage Architectures for Scalable AI/ML Training and Inference

## Motivation
Modern GPU clusters can process **terabytes per hour** of training data, yet **I/O stalls** frequently leave GPUs underutilized. Traditional storage (NFS/NAS, HDFS, basic S3 clients) and default data loaders are optimized for throughput-agnostic analytics, not for **low-latency, high-parallelism** model training. This project studies how storage and ingest layers bottleneck ML and proposes a **GPU-centric, Spark-integrated** design.

## Research Questions
1. What are the dominant I/O bottlenecks for training/inference on GPU clusters across **local NVMe, NAS, and object storage**?
2. How do **file formats** (Parquet/ORC vs. TFRecord/WebDataset/Arrow) and **pipelines** (PyTorch DataLoader vs DALI/FFCV) affect end-to-end time-to-train and GPU utilization?
3. Can **Spark + RAPIDS (UCX shuffle)** close the ETL-to-training gap when paired with **Arrow Flight** or **GPUDirect Storage (GDS)**?
4. What **caching/sharding** strategies (node-local NVMe, SPDK user-space I/O, Arrow Flight servers) maximize effective throughput under cost constraints?
5. What **architecture** emerges that others can reproduce with open tooling on AWS/GCP/on‑prem?

## Methods
- **Literature synthesis** on lakehouse formats and GPU I/O (Delta/Iceberg/Arrow, UCX, NVMe‑oF, GDS, DALI/FFCV).
- **Microbenchmarks** (`src/bench/io_bench.py` + `notebooks/01_io_microbench.ipynb`): sequential vs random reads; small-files vs sharded-tar; prefetch depth.
- **ETL pipelines**: Spark CPU vs Spark+RAPIDS with UCX shuffle; measure scan/filter/join throughput and spill.
- **Training**: PyTorch baseline vs DALI vs FFCV loaders on image/tabular datasets; measure **images/sec**, **GPU util**, **input pipeline wait**.
- **Inference**: Batch vs real-time (TorchServe/ONNXRuntime); measure p50/p95 latency vs RPS under different storage/caches.
- **Proposed design**: GPU-aware prefetch + Arrow Flight + UCX + node-local NVMe cache (optional SPDK) + object store backing; integrate with Delta/Iceberg metadata for governance.
- **Statistical rigor**: 5 independent runs; 95% CI via bootstrap; report medians and p95.

## Datasets
- Image: CIFAR‑10 / ImageNet‑subset (public mirrors).
- Tabular: NYC taxi, Criteo 1TB (subset), synthetic skewed datasets for joins.

## Metrics
- **Throughput** (MB/s, samples/s), **GPU utilization**, **stall time**, **epoch time**.
- **Latency** (p50/p95/p99) for inference.
- **Cost-efficiency**: $/epoch and $/10k inferences (cloud list prices).

## Expected Contributions
- A **reproducible benchmark suite** for storage-driven ML performance.
- A **reference architecture** unifying Spark ETL (RAPIDS+UCX) with **high-throughput training ingest** (Arrow Flight / GDS / DALI/FFCV) and **NVMe caching**.
- Guidance for choosing **formats**, **sharding**, and **caches** to avoid I/O stalls.

## Risks & Mitigations
- Hardware variance ⇒ normalize by device specs and publish config.
- Cloud variance ⇒ pin instance types; repeat across times of day.
- Library setup pains ⇒ provide `environment.yml`, configs, and runbooks.

## Deliverables
- 10–15 page paper (3,000–5,000 words), APA style, ≥30 citations.
- Notebooks, CSVs, and plots committed to a **private GitHub repo**.
- Architecture diagrams and a decision checklist for practitioners.
