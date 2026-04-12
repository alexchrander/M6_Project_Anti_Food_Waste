#!/bin/bash
echo "=== Starting Fetching & Predictions Pipeline at $(date) ==="

# 1. Navigate to your project folder
cd /work/M6_Project_Anti_Food_Waste

# 2. Activate your virtual environment
source /work/.venv/bin/activate

# 3. Run the Fetch Pipeline
echo "--- Fetch Pipeline starting at $(date) ---"
python3 fetch_pipeline/run_fetch.py || { echo "Fetch Pipeline failed — aborting"; exit 1; }
echo "--- Fetch Pipeline finished at $(date) ---"

# 4. Run the Predictions Pipeline
echo "--- Predictions Pipeline starting at $(date) ---"
python3 predict_pipeline/predict.py || { echo "Predictions Pipeline failed"; exit 1; }
echo "--- Predictions Pipeline finished at $(date) ---"

echo "=== Fetching & Predictions Pipeline finished at $(date) ==="