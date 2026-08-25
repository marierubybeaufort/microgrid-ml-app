# Microgrid ML

**Machine learning for electricity-consumption forecasting and microgrid fault-risk detection.**

Microgrid ML is an end-to-end machine-learning system for analyzing community energy data. The 2026 rebuild introduces a reproducible backend built with **FastAPI, PyTorch, and scikit-learn**, with explicit preprocessing, chronological evaluation, saved model artifacts, production inference, and tested API endpoints.

The project currently supports two core ML tasks:

- **next-hour residential electricity-consumption forecasting**
- **temporal fault-risk detection for household generation data**

> The original Streamlit dashboard remains in the repository as a legacy interface while a new React + Vite frontend is developed.

---

## Results

### Electricity forecasting

A PyTorch LSTM is trained on hourly residential electricity-consumption data aggregated by Forward Sortation Area.

| Metric | Result |
|---|---:|
| Test normalized MSE | `0.008363` |
| Test MAE | `0.014958 kWh/premise` |
| Test RMSE | `0.024860 kWh/premise` |
| Persistence baseline MSE | `0.057425` |
| MSE reduction vs. persistence | **85.44%** |

The model is evaluated on a chronologically held-out test period and compared against a persistence baseline that predicts the next hour using the most recent observed value.

### Fault-risk detection

The fault detector uses a class-balanced Random Forest trained on household-relative temporal degradation features.

| Metric | Result |
|---|---:|
| Mean held-out-household PR-AUC | **`0.803`** |
| Alert threshold strategy | 99th percentile of OOB normal scores |

Fault evaluation is performed by holding out faulty households entirely rather than randomly mixing observations between train and test sets.

The detector is intentionally treated as a **conservative fault-risk model**, not as a generic high-recall fault classifier.

---

## Architecture

```text
                         ┌─────────────────────────┐
                         │      React + Vite       │
                         │      frontend WIP       │
                         └────────────┬────────────┘
                                      │
                                      │ HTTP / JSON
                                      ▼
                         ┌─────────────────────────┐
                         │        FastAPI          │
                         │                         │
                         │  /forecast   /fault     │
                         │  /health                │
                         └──────────┬──────┬───────┘
                                    │      │
                    ┌───────────────┘      └───────────────┐
                    ▼                                      ▼
        ┌────────────────────────┐             ┌────────────────────────┐
        │  Forecasting Service   │             │  Fault-Risk Service    │
        └────────────┬───────────┘             └────────────┬───────────┘
                     │                                      │
                     ▼                                      ▼
        ┌────────────────────────┐             ┌────────────────────────┐
        │    PyTorch LSTM        │             │    Random Forest       │
        │                        │             │                        │
        │  next-hour energy      │             │  temporal degradation │
        │  consumption forecast  │             │  risk score           │
        └────────────┬───────────┘             └────────────┬───────────┘
                     │                                      │
                     ▼                                      ▼
        ┌────────────────────────┐             ┌────────────────────────┐
        │ Saved model + scaler   │             │ Saved model + config   │
        │ config + metrics       │             │ threshold              │
        └────────────────────────┘             └────────────────────────┘
```

---

## Forecasting pipeline

The 2026 forecasting pipeline uses public hourly residential electricity-consumption data from the **Independent Electricity System Operator (IESO)**.

The raw monthly dataset contains more than **1.46 million records** across geographic regions, customer types, and pricing plans.

For forecasting, the pipeline:

1. filters to residential customers
2. aggregates price plans within each geographic region and hour
3. removes incomplete geographic time series
4. normalizes consumption by premise count
5. creates cyclical hour-of-day and day-of-week features
6. performs chronological train / validation / test splitting
7. fits normalization statistics using the training period only
8. generates 24-hour sliding input sequences
9. trains a global LSTM across all geographic series

### Training dataset

After preprocessing:

```text
534 complete geographic series
744 hourly timestamps per series
397,296 aggregated observations
384,480 forecasting windows
24-hour input sequence
5 input features
```

The target is:

```text
consumption_per_premise_kwh
```

Using consumption per premise prevents large geographic regions from dominating the model purely because they contain more customers.

### Input features

Each hourly LSTM input contains:

```text
consumption_per_premise_kwh
hour_sin
hour_cos
dow_sin
dow_cos
```

The model consumes the previous **24 hours** and predicts electricity consumption for the **next hour**.

---

## Forecasting model

The forecaster is implemented in PyTorch using an LSTM followed by a small feed-forward prediction head.

```text
24 × 5 input sequence
        │
        ▼
      LSTM
 hidden size 64
        │
        ▼
 Linear → ReLU → Linear
        │
        ▼
next-hour consumption
```

Training uses:

- Adam optimizer
- mean squared error loss
- chronological validation
- early-stopping support
- best-validation checkpoint restoration
- fixed random seed for reproducibility

The final model contains approximately **20K trainable parameters**.

---

## Forecasting evaluation

The forecasting model is evaluated against a **persistence baseline**:

```text
prediction(t + 1) = observation(t)
```

This is an important baseline for slowly varying energy time series because a model should demonstrate value beyond simply repeating the most recent observation.

Final held-out test performance:

```text
LSTM normalized MSE:        0.008363
Persistence normalized MSE: 0.057425

Reduction in normalized MSE: 85.44%
```

Real-unit errors:

```text
MAE:  0.014958 kWh/premise
RMSE: 0.024860 kWh/premise
```

The test set is not used for model selection.

---

## Fault-risk detection

The fault subsystem analyzes household generation time series and estimates whether the current observation resembles a temporal degradation event.

The original household dataset contains:

```text
50 households
9,650 observations
15-minute sampling
63 labeled fault observations
3 households containing faults
```

Because faults are highly imbalanced and concentrated in a small number of households, random row-level splitting would produce misleading evaluation results.

Instead, evaluation holds out each faulty household and tests whether the detector generalizes to an unseen household.

---

## Fault features

Rather than using household identity, the model focuses on changes relative to each household's recent behavior.

Features include:

```text
delta_1
drop_from_4
drop_from_12
relative_delta_1
relative_drop_4
relative_drop_12
z_drop_4
z_drop_12
```

These capture patterns such as:

```text
recent generation is healthy
            │
            ▼
current generation drops sharply
            │
            ▼
fault risk increases
```

Raw household IDs are deliberately excluded from the model.

---

## Fault model

Fault risk is estimated using a **Random Forest classifier** with:

```text
500 trees
balanced_subsample class weighting
minimum leaf size = 2
out-of-bag predictions enabled
```

Rather than relying on the default `0.50` classification threshold, the production threshold is derived from the distribution of normal out-of-bag scores.

```text
alert threshold
=
99th percentile of OOB normal-class probabilities
```

This produces a conservative alerting strategy designed to limit false alarms.

### Held-out evaluation

Each faulty household is evaluated without using its labels during training.

```text
House_01 PR-AUC: 0.619
House_07 PR-AUC: 0.991
House_13 PR-AUC: 0.799

Mean PR-AUC: 0.803
```

The detector performs best as a **fault-risk ranking system**. Recall at the conservative production threshold is intentionally limited, and the current dataset is too small to support claims of broad real-world fault-classification accuracy.

---

## API

The backend is implemented with FastAPI.

### Health

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

### Forecast electricity consumption

```http
POST /forecast
```

The endpoint requires exactly **24 hourly observations**.

Example request:

```json
{
  "observations": [
    {
      "timestamp": "2025-05-29T01:00:00",
      "consumption_per_premise_kwh": 0.82
    }
  ]
}
```

The complete request contains 24 chronologically ordered observations spaced exactly one hour apart.

Example response:

```json
{
  "prediction_kwh_per_premise": 0.7744
}
```

### Estimate fault risk

```http
POST /fault
```

The endpoint requires exactly **13 generation readings**:

```text
12 historical observations
+
1 current observation
```

Observations must be chronologically ordered and exactly **15 minutes apart**.

Example response:

```json
{
  "fault_risk": 0.6209,
  "threshold": 0.1658,
  "alert": true
}
```

---

## Repository structure

```text
microgrid-ml-cym2025/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── forecasting.py
│   │   │   │   └── fault_detection.py
│   │   │   ├── schemas/
│   │   │   │   ├── forecasting.py
│   │   │   │   └── fault_detection.py
│   │   │   └── router.py
│   │   │
│   │   ├── ml/
│   │   │   ├── forecasting/
│   │   │   │   ├── baseline.py
│   │   │   │   ├── dataset.py
│   │   │   │   ├── evaluation.py
│   │   │   │   ├── ieso.py
│   │   │   │   ├── inference.py
│   │   │   │   ├── loaders.py
│   │   │   │   ├── model.py
│   │   │   │   ├── preprocessing.py
│   │   │   │   ├── scaling.py
│   │   │   │   ├── split.py
│   │   │   │   ├── train.py
│   │   │   │   └── train_ieso.py
│   │   │   │
│   │   │   └── fault_detection/
│   │   │       ├── evaluate.py
│   │   │       ├── inference.py
│   │   │       ├── preprocessing.py
│   │   │       └── train.py
│   │   │
│   │   ├── services/
│   │   │   ├── forecasting.py
│   │   │   └── fault_detection.py
│   │   │
│   │   └── main.py
│   │
│   ├── artifacts/
│   │   ├── forecasting/
│   │   │   ├── model.pt
│   │   │   ├── config.json
│   │   │   ├── metrics.json
│   │   │   └── scaler.json
│   │   │
│   │   └── fault_detection/
│   │       ├── model.joblib
│   │       └── config.json
│   │
│   └── tests/
│       └── test_api.py
│
├── data/
│
├── app.py
├── requirements.txt
└── README.md
```

`app.py` contains the original Streamlit dashboard and is retained temporarily as a legacy interface.

---

## Local setup

### 1. Clone the repository

```bash
git clone <repository>
cd microgrid-ml-cym2025
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run the API

From the repository root:

```bash
python -m uvicorn backend.app.main:app --reload
```

The backend exposes:

```text
GET  /health
POST /forecast
POST /fault
```

FastAPI's interactive API documentation is available through the local `/docs` endpoint while the server is running.

---

## Train the forecasting model

```bash
python -m backend.app.ml.forecasting.train_ieso
```

Training produces:

```text
backend/artifacts/forecasting/
├── model.pt
├── scaler.json
├── config.json
└── metrics.json
```

The training script:

- preprocesses the IESO dataset
- builds chronological datasets
- fits the scaler using training data only
- trains the LSTM
- restores the best validation state
- evaluates against the test set
- compares against persistence
- saves model metadata and metrics

---

## Train the fault-risk model

```bash
python -m backend.app.ml.fault_detection.train
```

Artifacts are written to:

```text
backend/artifacts/fault_detection/
├── model.joblib
└── config.json
```

To reproduce held-out-household evaluation:

```bash
python -m backend.app.ml.fault_detection.evaluate
```

---

## Tests

Run the backend test suite with:

```bash
python -m pytest backend/tests -v
```

Current API coverage includes:

- health endpoint
- forecasting inference
- 24-observation forecast validation
- fault-risk inference
- negative generation rejection
- hourly forecast interval validation
- 15-minute fault interval validation

Current status:

```text
7 passed
```

---

## Design decisions

### Why chronological splitting?

Randomly splitting adjacent time-series windows can leak highly similar observations between training and test sets.

Forecasting data is therefore divided chronologically:

```text
past ──────────────────────────────► future

|       train       | validation | test |
```

### Why train-only scaling?

Normalization statistics are calculated using training-period observations only.

Validation and test values never influence the fitted scaler.

### Why consumption per premise?

Raw electricity consumption strongly depends on the number of customers within a geographic region.

Normalizing by premise count creates a more comparable forecasting target across regions.

### Why hold out entire households for fault evaluation?

Only three households in the current synthetic community dataset contain labeled faults.

A random observation-level split could allow nearly identical fault patterns from the same household to appear in both training and test data.

Holding out an entire faulty household produces a harder and more meaningful generalization test.

---

## Limitations

This project is an applied ML prototype and has several important limitations.

### Forecasting

The current IESO training data covers one month of hourly observations. Although the global model sees hundreds of geographic series, longer seasonal coverage would provide a much stronger evaluation of annual and weather-driven behavior.

The current model uses calendar and historical consumption features but does not yet incorporate:

- weather
- temperature
- holidays
- electricity prices
- longer seasonal context

### Fault detection

The available fault dataset is small and synthetic:

```text
63 labeled faults
3 faulty households
```

Fault labels are concentrated at synchronized timestamps, so the detector should not be interpreted as a validated real-world equipment-failure classifier.

The current model is best viewed as a demonstration of **temporal fault-risk scoring and anomaly detection methodology**.

---

## Roadmap

### Backend

- [x] FastAPI application
- [x] PyTorch forecasting model
- [x] chronological evaluation pipeline
- [x] persistence baseline
- [x] saved forecasting artifacts
- [x] forecasting inference endpoint
- [x] temporal fault-risk features
- [x] Random Forest risk model
- [x] saved fault-model artifacts
- [x] fault-risk inference endpoint
- [x] API validation
- [x] backend test coverage

### Frontend

- [ ] replace legacy Streamlit interface
- [ ] React + Vite dashboard
- [ ] forecasting visualization
- [ ] fault-risk timeline
- [ ] model-performance views
- [ ] responsive layout
- [ ] API integration

### Infrastructure

- [ ] Railway backend deployment
- [ ] production frontend deployment
- [ ] environment-based API configuration
- [ ] CI test workflow

### ML

- [ ] multi-month IESO training data
- [ ] weather features
- [ ] additional forecasting baselines
- [ ] per-region performance diagnostics
- [ ] larger fault dataset
- [ ] probability calibration
- [ ] anomaly-detection comparisons

---

## Legacy dashboard

The repository still includes the original Streamlit dashboard from the 2025 project.

It is retained for historical/reference purposes while the frontend is being rebuilt.

The 2026 backend should be treated as the current ML implementation.

---

## Project status

**Backend: operational**

```text
IESO data
   ↓
PyTorch LSTM
   ↓
saved model
   ↓
FastAPI /forecast
```

```text
household generation
   ↓
temporal degradation features
   ↓
Random Forest
   ↓
saved model
   ↓
FastAPI /fault
```

**Frontend: React/Vite rebuild in progress.**

---

## Tech stack

**Machine learning**

- PyTorch
- scikit-learn
- NumPy
- pandas

**Backend**

- FastAPI
- Pydantic
- Uvicorn

**Testing**

- pytest
- httpx

**Frontend**

- Streamlit — legacy
- React + Vite — planned replacement

---

## Background

Microgrid ML began as a CYM 2025 project exploring machine learning for community energy systems.

The 2026 rebuild focuses on making the ML workflow more rigorous and reproducible:

- explicit data provenance
- leakage-aware evaluation
- meaningful baselines
- saved model artifacts
- honest treatment of dataset limitations
- production-style model inference
- a clean API boundary between ML and the user interface