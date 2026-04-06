#!/bin/bash
echo "=== Starting ML pipeline at $(date) ==="

# 1. Navigate to your project folder
cd /work/M6_Project_Anti_Food_Waste

# 2. Activate your virtual environment
source /work/.venv/bin/activate

# 3. Run the ML pipeline
python3 ml_pipeline/run_ml.py

echo "=== ML pipeline finished ==="