# Anti Food Waste — Aalborg
Predicts which clearance offers from Salling Group stores are likely to sell before expiry, and recommends recipes that use those products - served through a Streamlit dashboard with a live fetch-, prediction-, ML-, and RAG pipeline.

**Live App:** [https://app-food-waste.cloud.sdu.dk/](https://app-food-waste.cloud.sdu.dk/)

## Project structure

```
# Folders
analysis_&_monitoring/       # Notebooks for model development, evaluation, and connection tests
app/                         # Streamlit dashboard (Clearance Offers + Recipe Finder)
data/                        # Raw- and feature data
fetch_prediction_pipeline/   # Fetches live clearance offers from the Salling Group API followed by prediction pipeline
ml_pipeline/                 # Evaluates and retrains new model (if triggered)
models/                      # Saved champion model artifacts
outputs/                     # Log outputs from the full pipeline
rag_pipeline/                # Scrapes recipes, builds ChromaDB embeddings, and runs LLM queries
shell/                       # Shell scripts used by cron job

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

### .gitignore file
```
*.env                        # Required environment variables
*mlruns/                     # Saved model artifacts from each retraining
*data/predictions/           # Prediction data
```


## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A Salling Group Anti Food Waste API token — get one at [https://developer.sallinggroup.dev/](https://developer.sallinggroup.dev/catalog/8GAPSQHBBNZD6MEBFG3GGPHWRM)
- A MongoDB instance with recipes (used by the RAG pipeline)
- A Gemini API key (used by the RAG pipeline for LLM queries)

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
Open `.env` and fill in your credentials:
```
ANTI_FOOD_WASTE_API=your_token_here
MONGO_URI=your_mongodb_uri_here
GEMINI_API_KEY=your_gemini_key_here
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

**5. Build the RAG index (first time only)**
```bash
docker compose run --rm scheduler python rag_pipeline/build_index.py
```

**6. Open the app**
Go to http://localhost:8501

The scheduler automatically runs the fetch- and prediction pipeline every 15 minutes from 06:00 to 00:00, and the ML pipeline every night at 02:00. The RAG pipeline runs on demand via the Recipe Finder in the app.

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
| `docker compose run --rm scheduler python rag_pipeline/build_index.py` | Rebuild ChromaDB embeddings |

## Pipeline Diagram

![Pipeline Diagram](Pipeline_Diagram.png)
