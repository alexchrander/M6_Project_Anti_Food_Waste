#!/bin/bash
echo "=== Starting fetch pipeline at $(date) ==="

# 1. Navigate to your project folder
cd /work/M6_Project_Anti_Food_Waste

# 2. Activate your virtual environment
source /work/.venv/bin/activate

# 3. Run the python script
python3 fetch_pipeline/run_fetch.py

echo "=== Fetch Pipeline finished ==="