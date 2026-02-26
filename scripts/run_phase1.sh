#!/bin/bash
# Phase 1: Comprehensive data collection for ML model training

set -e  # Exit on error

echo "🚀 PHASE 1: DATA COLLECTION & BENCHMARKING"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d "notebooks" ]; then
    echo "❌ Error: notebooks directory not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Create results directory
mkdir -p results/ml_dataset

# Check for required tools
if ! command -v jupyter &> /dev/null; then
    echo "❌ Error: jupyter not found"
    echo "Installing jupyter..."
    pip install jupyter nbconvert
fi

# Step 1: Run all benchmarks
echo "📊 Step 1: Running comprehensive benchmarks..."
echo "This will take 30-60 minutes depending on your system"
echo ""
python scripts/run_full_benchmark.py

# Step 2: Consolidate results
echo ""
echo "🔨 Step 2: Consolidating benchmark data..."
python scripts/consolidate_ml_dataset.py

# Step 3: Quick validation
echo ""
echo "✅ Step 3: Validating dataset..."
python -c "
import pandas as pd
from pathlib import Path

csv_path = Path('results/ml_dataset/unified_ml_dataset.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f'✅ Dataset shape: {df.shape}')
    print(f'✅ Columns: {list(df.columns)[:5]}...')
    print(f'\nFirst 3 rows:')
    print(df.head(3).to_string())
else:
    print('⚠️  Dataset file not found')
"

echo ""
echo "=========================================="
echo "🎉 PHASE 1 COMPLETE!"
echo "=========================================="
echo ""
echo "📁 Dataset location: results/ml_dataset/unified_ml_dataset.csv"
echo ""
echo "📋 Next steps:"
echo "  1. Review results: cat results/ml_dataset/dataset_summary.json"
echo "  2. Start EDA: jupyter lab notebooks/phase2_eda_and_pca.ipynb"
echo ""
