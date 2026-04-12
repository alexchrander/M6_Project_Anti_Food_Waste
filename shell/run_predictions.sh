#!/bin/bash
echo "=== Starting Predictions Pipeline at $(date) ==="

# 1. Navigate to your project folder
cd /work/M6_Project_Anti_Food_Waste

# 2. Activate your virtual environment
source /work/.venv/bin/activate

# 3. Run the Predictions Pipeline
python3 predict_pipeline/predict.py

echo "=== Predictions Pipeline finished ==="