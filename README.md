# Anti Food Waste — Aalborg
Predicts which clearance offers from Salling Group stores are likely to sell before expiry, using a live ML pipeline and a Streamlit dashboard.

**Live app:** [https://app-food-waste.cloud.sdu.dk/](https://app-food-waste.cloud.sdu.dk/)

## Project structure

```
fetch_pipeline/   # Fetches live clearance offers from the Salling Group API and stores them
ml_pipeline/      # Builds dataset & features, trains and evaluates the model (runs nightly)
predict_pipeline/ # Loads the trained model and scores current offers
app/              # Streamlit dashboard
models/           # Saved model artifacts
data/             # Raw and feature data (gitignored)
outputs/          # Prediction outputs (gitignored)
```

## Requirements
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A Salling Group Food Waste API token — get one at [developer.sallinggroup.dev](https://developer.sallinggroup.dev/)

## Setup

**1. Clone the repository**
```bash
git clone <repo-url>
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
In a new terminal:
```bash
docker compose run --rm scheduler python fetch_pipeline/run_fetch.py
docker compose run --rm scheduler python predict_pipeline/predict.py
```

**5. Open the app**
Go to http://localhost:8501

The scheduler automatically runs fetch + predict every 15 minutes from 06:00 to 00:00, and the ML pipeline nightly at 02:00.

## Useful commands

| Command | Description |
|---|---|
| `docker compose up --build` | Start everything |
| `docker compose down` | Stop everything (keep data) |
| `docker compose down -v` | Stop everything and delete all data |
| `docker compose logs scheduler` | View pipeline logs |
| `docker compose run --rm scheduler python fetch_pipeline/run_fetch.py` | Run fetch manually |
| `docker compose run --rm scheduler python predict_pipeline/predict.py` | Run predictions manually |
| `docker compose run --rm scheduler python ml_pipeline/run_ml.py` | Retrain model manually |

## Reproducing the ML pipeline

To retrain the model from scratch after collecting enough data:

```bash
docker compose run --rm scheduler python ml_pipeline/run_ml.py
```

This runs `build_dataset.py` → `build_features.py` → `preprocessing.py` → `train.py` → `evaluate.py` in sequence and saves the new model to `models/`.

## Pipeline Diagram

![Pipeline Diagram](Pipeline_Diagram.png)
