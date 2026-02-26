# Runbook

1) Create env: `conda env create -f environment.yml`
2) Launch: `jupyter lab`
3) Edit `configs/spark-*.conf` for your cluster.
4) Run notebooks in order:
   - `00_environment_check.ipynb`
   - `01_io_microbench.ipynb`
   - `02_spark_etl_gpu_vs_cpu.ipynb`
   - `03_training_throughput_pytorch.ipynb`
   - `04_inference_latency_torchserve.ipynb`
   - `05_cache_and_formats_experiments.ipynb`
   - `06_arrow_flight_experiment.ipynb`
5) Commit CSVs/plots in `results/`.
