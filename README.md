# Anti Food Waste — Aalborg
Predicts which clearance offers from Salling Group stores are likely to sell before expiry, using a live fetch-, prediction-, and ML pipeline served with a Streamlit dashboard.

**Live App:** [https://app-food-waste.cloud.sdu.dk/](https://app-food-waste.cloud.sdu.dk/)

## Project structure

```
# Folders
__pycache__/                 # Auto-generated Python cache files
app/                         # Script for Streamlit dashboard
data/                        # Raw-,feature-, and predictions data (gitignored)
fetch_prediction_pipeline/   # Fetches live clearance offers from the Salling Group API followed by prediction pipeline
ml_pipeline/                 # Evaluates and retrains new model (if triggered)
models/                      # Saved champion model artifacts
outputs/                     # Log outputs from the full pipeline
shell/                       # Shell scripts used by cron jobs

# Files
.dockerignore                # Files and folders excluded from the Docker build context
.env.example                 # Template for required environment variables
.gitignore                   # Files and folders excluded from Git tracking
Dockerfile                   # Instructions for building the Docker image
Pipeline_Diagram.png         # Visual overview of the full pipeline
README.md                    # This file
config.py                    # Shared configuration and constants across the full pipeline
crontab                      # Cron schedule for the fetch-, prediction-, and ML pipelines
docker-compose.yml           # Defines and orchestrates the db, app, and scheduler services
requirements.txt             # Python dependencies
```

## Requirements
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A Salling Group Food Waste API token — get one at [https://developer.sallinggroup.dev/](https://developer.sallinggroup.dev/catalog/8GAPSQHBBNZD6MEBFG3GGPHWRM)

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/alexchrander/M6_Project_Anti_Food_Waste.git
cd M6_Project_Anti_Food_Waste
```

**2. Create your `.env` file**
```bash
cp .env.example .env
```
Open `.env` and set your API token:
```
ANTI_FOOD_WASTE_API=your_token_here
```

**3. Start all services**
```bash
docker compose up --build
```

**4. Trigger the first run**

The scheduler only activates on its cron schedule, so the first fetch and prediction must be triggered manually.

In a new terminal:
```bash
docker compose run --rm scheduler python fetch_prediction_pipeline/run_fetch.py
docker compose run --rm scheduler python fetch_prediction_pipeline/predict.py
```

**5. Open the app**
Go to http://localhost:8501

The scheduler automatically runs the fetch- and prediction pipeline every 15 minutes from 06:00 to 00:00, and the ML pipeline every night at 02:00.

## Useful commands

| Command | Description |
|---|---|
| `docker compose up --build` | Start everything |
| `docker compose down` | Stop everything (keep data) |
| `docker compose down -v` | Stop everything and delete all data |
| `docker compose logs scheduler` | View pipeline logs |
| `docker compose run --rm scheduler python fetch_prediction_pipeline/run_fetch.py` | Run fetch manually |
| `docker compose run --rm scheduler python fetch_prediction_pipeline/predict.py` | Run predictions manually |
| `docker compose run --rm scheduler python ml_pipeline/run_ml.py` | Run ML pipeline manually |

## Pipeline Diagram

![Pipeline Diagram](Pipeline_Diagram.png)
