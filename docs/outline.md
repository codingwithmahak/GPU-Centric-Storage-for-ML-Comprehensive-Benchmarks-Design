# Final Paper Outline (Draft)

1. **Introduction**
   - Problem statement; why GPUs idle; motivation and scope.
2. **Background**
   - Storage stacks (POSIX vs SPDK/io_uring; NVMe-oF; object stores).
   - Data formats & lakehouse layers (Parquet/ORC, Delta/Iceberg, Arrow/Flight).
   - GPU ingest (PyTorch DataLoader, DALI, FFCV) and Spark+RAPIDS.
3. **Related Work**
   - Prior systems (Ceph, DAOS, Alluxio, AIStore, WebDataset, Arrow Flight).
4. **Methodology**
   - Environments (local NVMe, NAS, S3/GCS), datasets, instrumentation.
   - Experiment matrix and hypotheses; statistics (CI/bootstrap).
5. **Microbenchmarks**
   - File size, sharding, prefetch, concurrency; sequential vs random I/O.
6. **ETL Benchmarks (Spark)**
   - CPU vs GPU; UCX shuffle; cache effects; format scans; join skew.
7. **Training Benchmarks**
   - Image & tabular pipelines; DataLoader vs DALI vs FFCV; GDS if available.
8. **Inference Benchmarks**
   - Latency under load; caching tiers; warm vs cold start.
9. **Proposed Architecture**
   - GPU‑aware prefetch + Arrow Flight server + UCX + NVMe cache + object store.
   - Control/data paths; failure modes; ops considerations.
10. **Results**
    - Plots and tables; effect sizes; cost/perf analysis.
11. **Discussion**
    - When each approach wins; portability; limitations.
12. **Conclusion & Future Work**
    - Open questions; next steps (multi-tenant isolation, multi-node GDS).
13. **References (≥30)**
