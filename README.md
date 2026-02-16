# Used Car Price Prediction (UCPP) 🚗📈

UCPP is a **local-first**, Docker-friendly **Used Car Price Prediction** service for **two markets: India (IN) and United States (US)**.  
It ships with a clean preprocessing pipeline, persisted ML artifacts, and a **FastAPI** server exposing simple **single** and **batch** prediction endpoints.

---

## Problem Statement 🧩

Used car buyers, sellers, and dealerships still face a messy reality:

- Pricing varies wildly across **cities, trims, mileage, ownership history, fuel type, and condition**
- Online prices are often **anchored to aspirational listings**, not market clearing prices
- Data is fragmented and inconsistent (units, strings, missing fields, noisy values)
- Many tools are either **not reproducible**, **not deployable**, or depend on heavyweight infra just to run inference

**Net result:** price discovery is slow, uncertain, and trust is low.

---

## Solution ✅

UCPP provides a practical, production-shaped solution:

- **Two-market support**: `IN` and `US`
- **Local-first inference**: runs on your machine with installed dependencies
- **Dockerized serving**: reliable, reproducible execution across environments
- **Artifact-driven predictions**: models loaded from `./artifacts` (mounted read-only in Docker)
- **API-first interface**: `/v1/predict` and `/v1/batch`
- **Readiness & liveness probes**: `/ready` and `/health` for deployment sanity

---

## Why this can be a game-changer 🚀

If you can predict fair value quickly and consistently, you unlock **real leverage**:

- **Dealership ops**: faster pricing decisions, less negotiation drag
- **Consumer trust**: fewer “overpriced” traps and better decision confidence
- **Market efficiency**: tighter spreads between listed vs realized prices
- **Lead conversion**: better qualification and intent scoring for marketplaces
- **Financing**: improved underwriting and risk checks for auto loans

The “game changer” is not just ML — it’s the ability to **deploy** a predictable, testable, repeatable pricing engine with strong API contracts.

---

## Current Market Constraints 🧱

Real-world used car pricing systems typically struggle because:

1. **Unstructured inputs** (e.g., `"45,000 mi"`, `"1197 CC"`, `"10.5 Lakh"`)  
2. **High missingness** (buyers/sellers omit key fields)
3. **Non-standard schemas** across platforms and countries
4. **Model reproducibility issues** (works on one laptop, breaks elsewhere)
5. **Operational gaps** (no readiness checks, no batch interface, no CI gates)

UCPP is designed to be a deployable foundation that reduces these risks.

---

## How UCPP solves it 🛠️

- **Strong schema validation** with Pydantic  
- **Preprocessing built for messy real-world strings** (units, numeric extraction, normalization)
- **Stable artifacts** stored in `./artifacts/v1/<market>/...`
- **In-process model cache** for fast repeat inference
- **Batch mode** with `strict=true|false`
  - `strict=true`: fail fast on the first invalid row
  - `strict=false`: return `preds[]` + `errors[]` without crashing

---

## API Overview 🌐

### Endpoints
- `GET /health` → liveness
- `GET /ready` → readiness (checks artifacts exist + are loadable)
- `POST /v1/predict` → single prediction
- `POST /v1/batch` → batch predictions

### Response behavior (Batch)
- Always returns:
  - `preds`: successful predictions
  - `errors`: per-row failures (only populated when `strict=false`)

---

## Quickstart (Local) ⚡

### 1) Create & activate a virtual env
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install
```bash
pip install -e ".[dev]"
```

### 3) Run tests
```bash
make fmt
make lint
make test
```

### 4) Run the API
```bash
uvicorn ucpp.api.app:app --host 127.0.0.1 --port 8000 --reload
```

---

## Quickstart (Docker) 🐳

### 1) Build & run
```bash
docker compose up --build
```

### 2) Health & readiness checks
```bash
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8000/ready  | jq .
```

> ✅ `ready` verifies the artifacts exist and default models are loadable.

---

## Example Requests 📬

### IN: Single predict
```bash
curl -s -X POST 'http://localhost:8000/v1/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "market":"IN",
    "payload":{
      "Name":"Honda City",
      "Location":"Mumbai",
      "Year":2016,
      "Kilometers_Driven":42000,
      "Fuel_Type":"Petrol",
      "Transmission":"Manual",
      "Owner_Type":"First",
      "Colour":"White",
      "Seats":5,
      "No. of Doors":4,
      "Mileage":"18.0 kmpl",
      "Engine":"1197 CC",
      "Power":"82 bhp",
      "New_Price":"10.5 Lakh"
    }
  }' | jq .
```

### US: Single predict
```bash
curl -s -X POST 'http://localhost:8000/v1/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "market":"US",
    "payload":{
      "brand":"Toyota",
      "model":"Camry",
      "model_year":2018,
      "fuel_type":"Gasoline",
      "engine":"2.5L",
      "transmission":"Automatic",
      "ext_col":"White",
      "int_col":"Black",
      "accident":"None reported",
      "clean_title":"Yes",
      "milage":"45000 mi"
    }
  }' | jq .
```

### Batch (non-strict): returns `errors[]` for invalid rows
```bash
curl -s -X POST 'http://localhost:8000/v1/batch' \
  -H 'Content-Type: application/json' \
  -d '{
    "market":"US",
    "strict": false,
    "rows":[
      {"brand":"Toyota","model":"Camry","model_year":2018},
      {}
    ]
  }' | jq .
```

---

## Artifacts Layout 📦

Artifacts are mounted in Docker (read-only):

```
./artifacts/
  v1/
    in/
      lightgbm.joblib
      lightgbm_log.joblib
      catboost.joblib
      catboost_log.joblib
      metrics.json
    us/
      lightgbm.joblib
      lightgbm_log.joblib
      catboost.joblib
      catboost_log.joblib
      metrics.json
```

The API reads artifacts from `UCPP_ARTIFACTS_DIR`:

- Local default: `artifacts`
- Docker default: `/app/artifacts`

---

## Configuration ⚙️

Environment variables:

- `UCPP_ARTIFACTS_DIR` (default: `artifacts`)
- `UCPP_HOST` (default: `127.0.0.1` in local `__main__`)
- `UCPP_PORT` (default: `8000`)

---

## Notes & Known Limitations ⚠️

- Predictions depend heavily on input completeness. Sparse payloads may yield unrealistic outputs (this is expected).
- Current validation is designed to be permissive for real-world missingness, but rejects **completely empty payloads**.
- The service is designed as a foundation — tightening “minimum required fields” per market is a logical next step for V4.

---

## Roadmap 🔭

### ✅ V3 (Completed)
- Dockerized FastAPI serving
- `/health` and `/ready`
- Artifact-based model loading with caching
- Single and batch prediction endpoints
- Batch strict vs non-strict behavior
- Validation errors properly classified + JSON-serializable
- CI workflow + tests passing

### ➡️ V4 (Next)
- Input contract hardening (minimum viable fields per market)
- Better defaults + warnings for imputed/missing values
- API docs polish + examples + possibly versioned artifact routing

---

## License 📝

Private/experimental project. Add a license if you intend to open-source.
