"""
Comprehensive benchmark execution script for ML model training data collection.
Runs all notebooks and collects results into a unified dataset.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys

def run_notebook(notebook_path: str, output_dir: str = "results/ml_dataset"):
    """Execute a Jupyter notebook and capture outputs."""
    print(f"\n{'='*60}")
    print(f"Running: {notebook_path}")
    print(f"{'='*60}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{Path(notebook_path).stem}_{timestamp}.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Execute notebook using nbconvert
        # Use absolute paths to avoid path confusion
        abs_output_path = Path(output_path).resolve()
        abs_notebook_path = Path(notebook_path).resolve()
        
        cmd = [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=3600",
            "--output", str(abs_output_path),
            str(abs_notebook_path)
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        
        print(f"✅ Completed in {duration:.1f}s")
        return {
            "notebook": notebook_path,
            "status": "success",
            "duration_seconds": duration,
            "timestamp": timestamp,
            "output_path": str(output_path)
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {
            "notebook": notebook_path,
            "status": "failed",
            "error": str(e),
            "timestamp": timestamp
        }

def main():
    """Run all benchmarking notebooks in sequence."""
    output_dir = Path("results/ml_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Notebook execution order
    notebooks = [
        "notebooks/00_placeholder.ipynb",  # Environment check
        "notebooks/01_placeholder.ipynb",  # I/O microbenchmarks
        "notebooks/02_placeholder.ipynb",  # Spark ETL benchmarks
        "notebooks/03_placeholder.ipynb",  # Training throughput
    ]
    
    results = []
    
    print("\n🚀 STARTING COMPREHENSIVE BENCHMARK SUITE")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for notebook in notebooks:
        if not Path(notebook).exists():
            print(f"⚠️  Skipping {notebook} (not found)")
            continue
            
        result = run_notebook(notebook, str(output_dir))
        results.append(result)
        
        # Brief pause between notebooks
        time.sleep(2)
    
    # Save execution summary
    summary_path = output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("📊 BENCHMARK EXECUTION SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_duration = sum(r.get('duration_seconds', 0) for r in results)
    
    print(f"Total notebooks: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"\n📄 Summary saved to: {summary_path}")
    
    # Show failed notebooks
    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        print("\n❌ Failed notebooks:")
        for r in failed:
            print(f"   - {r['notebook']}: {r.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    main()
