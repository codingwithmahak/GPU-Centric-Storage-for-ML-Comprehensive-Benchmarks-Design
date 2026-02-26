🚀 GPU-Centric Storage for Machine Learning
Comprehensive Benchmarks & Architecture Exploration

A practical benchmarking and experimentation framework designed to analyze storage bottlenecks in GPU-accelerated machine learning workflows.

This repository provides reproducible experiments, performance comparisons, and optimization strategies for improving end-to-end ML pipeline efficiency.

🎯 Project Objective

Modern ML workloads are often GPU-bound in theory but I/O-bound in practice.
This project explores how storage systems impact:

Training throughput

ETL performance

Inference latency

GPU utilization efficiency

End-to-end pipeline performance

The goal is to identify bottlenecks and propose GPU-centric storage optimization strategies.

🔬 What This Repository Includes
📊 Benchmarking Modules

I/O microbenchmarks (sequential vs random access)

CPU vs GPU Spark ETL comparisons

Training data loader performance analysis

Inference latency evaluation

Caching and storage tier experiments

Arrow Flight performance testing

🧪 Reproducible Experiments

Structured notebook workflow

Configurable Spark environments (CPU & GPU)

Synthetic dataset generation tools

Automated benchmarking scripts

📈 Performance Analysis

Throughput comparison plots

GPU idle time analysis

Memory usage tracking

Cost-performance tradeoff insights

📚 Notebook Workflow

Run notebooks in order for full analysis:

00_environment_check – System validation

01_io_microbenchmarks – Storage performance testing

02_spark_etl_cpu_vs_gpu – ETL comparison

03_training_throughput – Data loading optimization

04_inference_latency – Serving performance

05_caching_strategies – Storage tier evaluation

06_arrow_flight – High-throughput data serving

🌐 Supported Platforms

Local development environments

Google Colab

AWS SageMaker

EMR Spark clusters

🛠️ Technology Stack
ML & Processing Frameworks

PyTorch

Spark

RAPIDS cuDF

NVIDIA DALI

FFCV

Storage & Data Formats

Parquet

ORC

Apache Iceberg

Delta Lake

Apache Arrow

Arrow Flight

🏗️ Advanced Experiments

Data lakehouse performance analysis

Snapshot-based reproducibility workflows

GPU memory optimization experiments

Distributed data serving performance

📁 Repository Structure
gpu_storage_ml_project/
├── notebooks/        # Interactive benchmarking notebooks
├── src/bench/        # Benchmark utilities
├── configs/          # Spark & storage configurations
├── scripts/          # Automation scripts
├── results/          # Benchmark outputs
├── experiments/      # Experiment logs
└── data/             # Generated datasets
🎯 Key Research Themes

Quantifying I/O bottlenecks in ML pipelines

Evaluating GPU acceleration impact on ETL

Comparing storage formats for training workloads

Designing GPU-centric storage architectures

Improving GPU utilization through data optimization

🚀 Getting Started

Clone the repository

Set up the environment

Run 00_environment_check

Execute notebooks sequentially

Analyze benchmark results