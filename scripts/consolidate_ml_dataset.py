"""
Consolidate all benchmark results into a unified ML training dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict
from datetime import datetime

def load_benchmark_results(results_dir: str = "results") -> Dict[str, pd.DataFrame]:
    """Load all CSV benchmark results."""
    results_path = Path(results_dir)
    
    datasets = {}
    
    # Define expected result files
    result_files = {
        # I/O Benchmarks
        'io_sequential': 'io_sequential_detailed.csv',
        'io_random': 'io_random_detailed.csv',
        'io_mmap': 'io_mmap_detailed.csv',
        'io_concurrent': 'io_concurrent_detailed.csv',
        
        # ETL Benchmarks
        'etl_cpu': 'spark_etl_cpu_detailed.csv',
        'etl_gpu': 'spark_etl_gpu_detailed.csv',
        'etl_formats': 'format_comparison_detailed.csv',
        
        # Training Benchmarks
        'training': 'training_benchmarks_comprehensive.csv',
    }
    
    print("📂 Loading benchmark results...")
    
    for key, filename in result_files.items():
        filepath = results_path / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                datasets[key] = df
                print(f"   ✅ {key}: {len(df)} rows from {filename}")
            except Exception as e:
                print(f"   ⚠️  Failed to load {filename}: {e}")
        else:
            print(f"   ⚠️  Not found: {filename}")
    
    return datasets

def engineer_features(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Engineer features from raw benchmark data for ML model training.
    
    Creates a unified dataset with features for:
    - Regression: predicting throughput
    - Classification: recommending configurations
    """
    
    feature_dfs = []
    
    # Process I/O sequential benchmarks
    if 'io_sequential' in datasets:
        print("\n🔨 Processing sequential I/O benchmarks...")
        io_df = datasets['io_sequential'].copy()
        
        # Extract storage type from file path
        io_df['storage_type'] = io_df['file_path'].apply(lambda x: 
            'nvme' if 'nvme' in str(x).lower() else
            'nfs' if 'nfs' in str(x).lower() else
            's3' if 's3' in str(x).lower() else 
            'tmpfs' if 'tmp' in str(x).lower() else 'disk'
        )
        
        # Categorize file sizes
        io_df['file_size_category'] = pd.cut(
            io_df['file_size_mb'],
            bins=[0, 1, 10, 100, 1000, float('inf')],
            labels=['tiny', 'small', 'medium', 'large', 'xlarge']
        )
        
        # Select and rename columns
        io_features = io_df[[
            'block_kb', 'file_size_mb', 'throughput_mb_s', 
            'duration_seconds', 'storage_type', 'file_size_category'
        ]].copy()
        
        io_features['benchmark_type'] = 'io_sequential'
        io_features['access_pattern'] = 'sequential'
        io_features['target_throughput'] = io_features['throughput_mb_s']
        
        feature_dfs.append(io_features)
        print(f"   Added {len(io_features)} sequential I/O observations")
    
    # Process random I/O benchmarks
    if 'io_random' in datasets:
        print("\n🔨 Processing random I/O benchmarks...")
        rand_df = datasets['io_random'].copy()
        
        rand_df['storage_type'] = rand_df['file_path'].apply(lambda x: 
            'nvme' if 'nvme' in str(x).lower() else
            'nfs' if 'nfs' in str(x).lower() else
            's3' if 's3' in str(x).lower() else 
            'tmpfs' if 'tmp' in str(x).lower() else 'disk'
        )
        
        rand_features = rand_df[[
            'block_kb', 'file_size_mb', 'n_samples', 
            'throughput_mb_s', 'iops', 'storage_type'
        ]].copy()
        
        rand_features['benchmark_type'] = 'io_random'
        rand_features['access_pattern'] = 'random'
        rand_features['target_throughput'] = rand_features['throughput_mb_s']
        
        feature_dfs.append(rand_features)
        print(f"   Added {len(rand_features)} random I/O observations")
    
    # Process concurrent I/O benchmarks
    if 'io_concurrent' in datasets:
        print("\n🔨 Processing concurrent I/O benchmarks...")
        conc_df = datasets['io_concurrent'].copy()
        
        # Handle different column names
        if 'aggregate_throughput_mb_s' in conc_df.columns:
            throughput_col = 'aggregate_throughput_mb_s'
        elif 'total_throughput_mb_s' in conc_df.columns:
            throughput_col = 'total_throughput_mb_s'
        else:
            throughput_col = 'throughput_mb_s'
        
        conc_features = conc_df[['n_threads', throughput_col]].copy()
        conc_features['benchmark_type'] = 'io_concurrent'
        conc_features['access_pattern'] = 'concurrent'
        conc_features['target_throughput'] = conc_features[throughput_col]
        conc_features['storage_type'] = 'local'  # Default
        
        feature_dfs.append(conc_features)
        print(f"   Added {len(conc_features)} concurrent I/O observations")
    
    # Process CPU ETL benchmarks
    if 'etl_cpu' in datasets:
        print("\n🔨 Processing CPU ETL benchmarks...")
        etl_df = datasets['etl_cpu'].copy()
        
        etl_features = etl_df[[
            'workload', 'dataset_name', 'duration_seconds', 
            'throughput_mb_s', 'memory_peak_mb'
        ]].copy()
        
        etl_features['benchmark_type'] = 'etl_cpu'
        etl_features['compute_backend'] = 'cpu'
        etl_features['target_throughput'] = etl_features['throughput_mb_s']
        
        feature_dfs.append(etl_features)
        print(f"   Added {len(etl_features)} CPU ETL observations")
    
    # Process GPU ETL benchmarks
    if 'etl_gpu' in datasets:
        print("\n🔨 Processing GPU ETL benchmarks...")
        gpu_df = datasets['etl_gpu'].copy()
        
        gpu_features = gpu_df[[
            'workload', 'dataset_name', 'duration_seconds',
            'throughput_mb_s', 'memory_peak_mb'
        ]].copy()
        
        gpu_features['benchmark_type'] = 'etl_gpu'
        gpu_features['compute_backend'] = 'gpu'
        gpu_features['target_throughput'] = gpu_features['throughput_mb_s']
        
        feature_dfs.append(gpu_features)
        print(f"   Added {len(gpu_features)} GPU ETL observations")
    
    # Process training benchmarks
    if 'training' in datasets:
        print("\n🔨 Processing training benchmarks...")
        train_df = datasets['training'].copy()
        
        # Select available columns
        cols_to_use = []
        if 'framework' in train_df.columns:
            cols_to_use.append('framework')
        if 'dataset_type' in train_df.columns:
            cols_to_use.append('dataset_type')
        if 'batch_size' in train_df.columns:
            cols_to_use.append('batch_size')
        if 'samples_per_second' in train_df.columns:
            cols_to_use.append('samples_per_second')
        if 'data_loading_ratio' in train_df.columns:
            cols_to_use.append('data_loading_ratio')
        if 'num_workers' in train_df.columns:
            cols_to_use.append('num_workers')
        
        train_features = train_df[cols_to_use].copy()
        train_features['benchmark_type'] = 'training'
        train_features['target_throughput'] = train_features['samples_per_second']
        
        feature_dfs.append(train_features)
        print(f"   Added {len(train_features)} training observations")
    
    # Combine all features
    if feature_dfs:
        unified_df = pd.concat(feature_dfs, axis=0, ignore_index=True, sort=False)
        print(f"\n✅ Created unified dataset: {len(unified_df)} total observations")
        return unified_df
    else:
        print("\n⚠️  No data available for feature engineering")
        return pd.DataFrame()

def save_ml_dataset(df: pd.DataFrame, output_path: str = "results/ml_dataset/unified_ml_dataset.csv"):
    """Save the consolidated ML dataset with summary statistics."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    df.to_csv(output_file, index=False)
    
    print(f"\n📊 ML Dataset Statistics:")
    print(f"   Total observations: {len(df)}")
    print(f"   Features (columns): {len(df.columns)}")
    print(f"   Benchmark types: {df['benchmark_type'].nunique()}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Target variable: target_throughput")
    
    # Show value counts for categorical variables
    print(f"\n📈 Benchmark type distribution:")
    print(df['benchmark_type'].value_counts().to_string())
    
    if 'storage_type' in df.columns:
        print(f"\n💾 Storage type distribution:")
        print(df['storage_type'].value_counts().to_string())
    
    print(f"\n💾 Saved to: {output_file}")
    
    # Save summary statistics
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_observations': int(len(df)),
        'n_features': int(len(df.columns)),
        'columns': list(df.columns),
        'benchmark_types': df['benchmark_type'].value_counts().to_dict(),
        'column_dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df) > 0 else {}
    }
    
    summary_path = output_file.parent / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📄 Summary saved to: {summary_path}")

def main():
    """Main execution function."""
    print("="*60)
    print("🔧 CONSOLIDATING ML DATASET FROM BENCHMARKS")
    print("="*60)
    
    # Load all benchmark results
    datasets = load_benchmark_results("results")
    
    if not datasets:
        print("\n❌ No benchmark data found!")
        print("\nPlease run benchmarks first:")
        print("   python scripts/run_full_benchmark.py")
        print("\nOr run notebooks manually:")
        print("   jupyter lab notebooks/")
        return
    
    # Engineer features
    print("\n" + "="*60)
    ml_dataset = engineer_features(datasets)
    
    if ml_dataset.empty:
        print("\n❌ Failed to create ML dataset")
        return
    
    # Save consolidated dataset
    print("\n" + "="*60)
    save_ml_dataset(ml_dataset)
    
    print("\n" + "="*60)
    print("✅ ML DATASET PREPARATION COMPLETE!")
    print("="*60)
    print("\n📋 Next steps:")
    print("  1. Review the dataset: results/ml_dataset/unified_ml_dataset.csv")
    print("  2. Run exploratory data analysis:")
    print("     jupyter lab notebooks/phase2_eda_and_pca.ipynb")
    print("  3. Begin model training (Phase 3)")

if __name__ == "__main__":
    main()
