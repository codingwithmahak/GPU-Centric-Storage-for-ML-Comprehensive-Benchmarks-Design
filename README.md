# 🚀 GPU-Centric Storage for Machine Learning  
## Comprehensive Benchmarks & Architecture Exploration

A practical benchmarking and experimentation framework for analyzing storage bottlenecks in GPU-accelerated machine learning workflows.

This project provides reproducible experiments, performance comparisons, and optimization strategies to improve end-to-end ML pipeline efficiency.

---

## 🔬 What This Repository Includes

### 📊 Benchmarking Modules

- I/O microbenchmarks (sequential vs random access)
- CPU vs GPU Spark ETL comparisons
- Training data loader performance analysis
- Inference latency evaluation
- Caching and storage tier experiments
- Arrow Flight performance testing

---

### 🧪 Reproducible Experiments

- Structured notebook workflow  
- Configurable Spark environments (CPU & GPU)  
- Synthetic dataset generation tools  
- Automated benchmarking scripts  

---

### 📈 Performance Analysis

- Throughput comparison plots  
- GPU idle time analysis  
- Memory usage tracking  
- Cost-performance tradeoff insights  

---

## 📚 Notebook Workflow

Run notebooks in order for a complete analysis:

| Order | Notebook | Description |
|-------|----------|-------------|
| 1 | `00_environment_check` | System validation |
| 2 | `01_io_microbenchmarks` | Storage performance testing |
| 3 | `02_spark_etl_cpu_vs_gpu` | ETL comparison |
| 4 | `03_training_throughput` | Data loading optimization |
| 5 | `04_inference_latency` | Serving performance |
| 6 | `05_caching_strategies` | Storage tier evaluation |
| 7 | `06_arrow_flight` | High-throughput data serving |

---

## 🌐 Supported Platforms

- Local development environments  
- Google Colab  
- AWS SageMaker  
- EMR Spark clusters  

---

## 🛠️ Technology Stack

### ML & Processing Frameworks

- PyTorch  
- Apache Spark  
- RAPIDS cuDF  
- NVIDIA DALI  
- FFCV  

### Storage & Data Formats

- Parquet  
- ORC  
- Apache Iceberg  
- Delta Lake  
- Apache Arrow  
- Arrow Flight  

---

## 📁 Repository Structure

gpu_storage_ml_project/
├── notebooks/ # Interactive benchmarking notebooks
├── src/bench/ # Benchmark utilities
├── configs/ # Spark & storage configurations
├── scripts/ # Automation scripts
├── results/ # Benchmark outputs
├── experiments/ # Experiment logs
└── data/ # Generated datasets


---

## 🎯 Key Research Themes

- Quantifying I/O bottlenecks in ML pipelines  
- Evaluating GPU acceleration impact on ETL  
- Comparing storage formats for training workloads  
- Designing GPU-centric storage architectures  
- Improving GPU utilization through data optimization  

---

## 🚀 Getting Started

1. Clone the repository  
2. Set up the environment  
3. Run `00_environment_check`  
4. Execute notebooks sequentially  
5. Analyze benchmark results

