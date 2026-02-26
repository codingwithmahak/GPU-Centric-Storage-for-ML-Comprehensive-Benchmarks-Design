# GPU Storage ML Project: Deployment Guide

This guide provides step-by-step instructions for deploying and running the GPU Storage ML benchmarks across different platforms, from local development to cloud-scale deployments.

## 🎯 Platform-Specific Deployment

### 📱 Google Colab (Recommended for Beginners)

**Perfect for**: Learning, experimentation, free GPU access

#### Setup Steps:
1. **Open Colab and create new notebook**
   ```python
   # Clone the repository
   !git clone https://github.com/knkarthik01/gpu_storage_ml_project.git
   %cd gpu_storage_ml_project
   ```

2. **Install dependencies**
   ```python
   # Install core packages
   !pip install pandas numpy matplotlib pyspark pyarrow tqdm psutil scipy scikit-learn
   
   # Optional GPU packages (if GPU runtime selected)
   !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Mount Google Drive for data persistence**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Create symlink for persistent storage
   !ln -sf /content/drive/MyDrive/gpu_storage_results /content/gpu_storage_ml_project/results
   ```

4. **Run environment check**
   ```python
   # Execute the environment check notebook
   %run notebooks/00_environment_check.ipynb
   ```

#### Colab-Specific Tips:
- **Enable GPU Runtime**: Runtime → Change runtime type → GPU
- **Memory Management**: Use smaller dataset sizes to avoid memory limits
- **Session Persistence**: Save intermediate results to Drive
- **Batch Size Optimization**: Reduce batch sizes for memory constraints

---

### 🔬 AWS SageMaker Studio

**Perfect for**: Production ML workflows, enterprise environments

#### Setup Steps:
1. **Launch SageMaker Studio**
   ```bash
   # From AWS Console: SageMaker → Studio → Launch Studio
   # Or use AWS CLI:
   aws sagemaker create-domain --domain-name gpu-storage-ml --auth-mode IAM
   ```

2. **Clone repository in Studio**
   ```bash
   # In Studio terminal:
   git clone https://github.com/knkarthik01/gpu_storage_ml_project.git
   cd gpu_storage_ml_project
   ```

3. **Set up SageMaker environment**
   ```python
   # Install in notebook cell:
   !pip install pandas numpy matplotlib pyspark pyarrow tqdm psutil
   
   # For GPU instances:
   !pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
   ```

4. **Configure S3 integration**
   ```python
   import boto3
   import sagemaker
   
   session = sagemaker.Session()
   bucket = session.default_bucket()
   
   # Set S3 paths for large datasets
   s3_data_path = f's3://{bucket}/gpu-storage-ml-data'
   s3_results_path = f's3://{bucket}/gpu-storage-ml-results'
   ```

#### SageMaker-Specific Features:
- **Processing Jobs**: Scale experiments across multiple instances
- **S3 Integration**: Seamless large dataset handling
- **Distributed Training**: Multi-GPU experiments
- **Cost Optimization**: Spot instances for batch processing

#### Sample SageMaker Processing Job:
```python
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    command=['python3'],
    image_uri='683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.c5.2xlarge'
)

processor.run(
    code='notebooks/01_io_microbenchmarks.py',  # Convert notebook to script
    inputs=[ProcessingInput(source=s3_data_path, destination='/opt/ml/processing/input')],
    outputs=[ProcessingOutput(source='/opt/ml/processing/output', destination=s3_results_path)]
)
```

---

### 🔥 EMR Spark Clusters

**Perfect for**: Big data processing, distributed GPU workloads

#### Setup Steps:
1. **Create EMR cluster with GPU instances**
   ```bash
   aws emr create-cluster \
     --applications Name=Spark Name=Hadoop Name=Livy \
     --ec2-attributes KeyName=your-key-pair \
     --instance-type g4dn.xlarge \
     --instance-count 3 \
     --service-role EMR_DefaultRole \
     --ec2-attributes InstanceProfile=EMR_EC2_DefaultRole \
     --bootstrap-actions Path=s3://your-bucket/scripts/setup_emr.sh \
     --configurations file://configs/emr-config.json
   ```

2. **EMR Bootstrap Script** (`scripts/setup_emr.sh`):
   ```bash
   #!/bin/bash
   
   # Install Python packages
   sudo pip3 install pandas numpy matplotlib pyarrow tqdm psutil
   
   # Install RAPIDS (for GPU instances)
   if nvidia-smi > /dev/null 2>&1; then
       conda install -c rapidsai -c nvidia -c conda-forge cudf=23.10 python=3.9 cudatoolkit=11.8
   fi
   
   # Configure Spark for GPU
   sudo tee -a /etc/spark/conf/spark-defaults.conf << EOF
   spark.plugins com.nvidia.spark.SQLPlugin
   spark.rapids.sql.enabled true
   spark.executor.resource.gpu.amount 1
   spark.task.resource.gpu.amount 0.0625
   EOF
   ```

3. **Submit Spark jobs**
   ```bash
   # Submit ETL benchmark
   spark-submit \
     --properties-file configs/spark-gpu.conf \
     --py-files src/bench/spark_utils.py \
     scripts/run_spark_etl_benchmark.py
   ```

#### EMR Configuration (`configs/emr-config.json`):
```json
[
  {
    "Classification": "spark-defaults",
    "Properties": {
      "spark.sql.adaptive.enabled": "true",
      "spark.sql.adaptive.coalescePartitions.enabled": "true",
      "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
      "spark.dynamicAllocation.enabled": "true",
      "spark.dynamicAllocation.maxExecutors": "20"
    }
  }
]
```

---

### 🖥️ Local Development

**Perfect for**: Development, custom hardware testing, full control

#### Ubuntu/Linux Setup:
```bash
# 1. Clone repository
git clone https://github.com/knkarthik01/gpu_storage_ml_project.git
cd gpu_storage_ml_project

# 2. Install system dependencies
sudo apt update
sudo apt install python3-pip python3-venv build-essential

# 3. Create virtual environment
python3 -m venv gpu-storage-env
source gpu-storage-env/bin/activate

# 4. Install Python packages
pip install -r requirements.txt

# 5. Install PyTorch (CPU or GPU)
# For CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# For GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 6. Install Spark
wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xzf spark-3.5.0-bin-hadoop3.tgz
export SPARK_HOME=$PWD/spark-3.5.0-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH

# 7. Launch Jupyter
jupyter lab
```

#### macOS Setup:
```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python and dependencies
brew install python@3.11
brew install openjdk@11

# 3. Clone and setup environment
git clone https://github.com/knkarthik01/gpu_storage_ml_project.git
cd gpu_storage_ml_project
python3 -m venv gpu-storage-env
source gpu-storage-env/bin/activate
pip install -r requirements.txt

# 4. Install PyTorch (MPS for Apple Silicon)
pip install torch torchvision

# 5. Setup Spark
brew install apache-spark
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
```

#### Windows Setup:
```powershell
# 1. Install Python from python.org (3.9+ recommended)
# 2. Install Git for Windows

# 3. Clone repository
git clone https://github.com/knkarthik01/gpu_storage_ml_project.git
cd gpu_storage_ml_project

# 4. Create virtual environment
python -m venv gpu-storage-env
gpu-storage-env\Scripts\activate

# 5. Install packages
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 6. Install Java 11 and Spark
# Download from Apache Spark website
# Set JAVA_HOME and SPARK_HOME environment variables

# 7. Launch Jupyter
jupyter lab
```

---

## 🛠️ Advanced Configuration

### GPU Optimization
```python
# CUDA environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# PyTorch optimizations
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Memory optimization
torch.cuda.empty_cache()
```

### Storage Configuration
```bash
# NVMe optimization
echo mq-deadline | sudo tee /sys/block/nvme0n1/queue/scheduler

# Network storage testing
# NFS mount for network storage experiments
sudo mount -t nfs server:/path/to/data /mnt/nfs-data

# S3 configuration
aws configure set default.s3.max_concurrent_requests 20
aws configure set default.s3.max_bandwidth 100MB/s
```

### Spark Tuning
```python
# Spark configuration for large datasets
spark_config = {
    'spark.executor.memory': '8g',
    'spark.executor.cores': '4',
    'spark.sql.shuffle.partitions': '400',
    'spark.sql.adaptive.enabled': 'true',
    'spark.sql.adaptive.coalescePartitions.enabled': 'true',
    'spark.executor.memoryFraction': '0.8',
    'spark.executor.extraJavaOptions': '-XX:+UseG1GC'
}
```

---

## 🐳 Docker Deployment

### CPU-Only Container
```dockerfile
FROM python:3.11-slim

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

### GPU-Enabled Container
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

COPY . .
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

### Build and Run
```bash
# Build container
docker build -t gpu-storage-ml .

# Run with GPU support
docker run --gpus all -p 8888:8888 -v $(pwd)/results:/workspace/results gpu-storage-ml
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Verify PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"

# Fix CUDA version mismatch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Memory Issues
```python
# Reduce batch sizes in config
batch_size = 16  # instead of 64

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

#### 3. Spark Issues
```bash
# Increase driver memory
export SPARK_DRIVER_MEMORY=4g

# Fix Java heap space
export SPARK_DRIVER_OPTS="-Xmx4g"

# Check Spark configuration
spark-submit --help
```

#### 4. Storage Performance Issues
```bash
# Check disk I/O
iostat -x 1

# Monitor network storage
iftop

# Verify mount options
mount | grep data
```

---

## 📊 Performance Monitoring

### System Monitoring
```python
# Monitor GPU usage
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

# Monitor CPU and memory
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")
```

### Application Monitoring
```python
# PyTorch profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your training code here
    pass

prof.export_chrome_trace("trace.json")
```

---

## 🚀 Production Deployment

### CI/CD Pipeline
```yaml
# .github/workflows/benchmark.yml
name: GPU Storage Benchmarks

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run benchmarks
      run: python scripts/run_full_benchmark.py
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: results/
```

### Automated Reporting
```python
# scripts/generate_report.py
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def generate_daily_report():
    # Load benchmark results
    results = pd.read_csv('results/daily_benchmark.csv')
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Performance trends
    results.plot(x='date', y='throughput', ax=axes[0,0])
    axes[0,0].set_title('Throughput Trends')
    
    # Save report
    plt.savefig(f'reports/daily_report_{datetime.now().strftime("%Y%m%d")}.png')
    
    return 'reports/daily_report.html'
```

---

## 📞 Support

### Getting Help
- **Documentation**: Check the `docs/` directory for detailed guides
- **Issues**: Create GitHub issues for bugs or feature requests  
- **Discussions**: Use GitHub Discussions for questions and tips
- **Community**: Join our community for optimization insights

### Contributing
- Follow the contribution guidelines in `CONTRIBUTING.md`
- Run tests before submitting: `python -m pytest tests/`
- Include performance results with your contributions
- Update documentation for new features

---

**🎯 Ready to deploy? Choose your platform above and start optimizing your ML storage pipelines!**