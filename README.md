# Anti Food Waste — Aalborg
Predicts which clearance offers from Salling Group stores are likely to sell before expiry, using a live ML pipeline and a Streamlit dashboard.

## Requirements
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A Salling Group Food Waste API token

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

**4. Trigger the first fetch**
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
| `docker compose run --rm scheduler python ml_pipeline/run_ml.py` | Run ML pipeline manually |
