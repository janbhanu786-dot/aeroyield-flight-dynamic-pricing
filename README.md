# AeroYield — Flight Booking Simulator with Dynamic Pricing Engine

A full-stack airline revenue management, seat reservation, and dynamic pricing simulator powered by **FastAPI**, **SQLAlchemy**, **scikit-learn**, **SciPy**, and **Streamlit**.

---

## Project Walkthrough & Visual Previews

### Flight Booking & Reservation Engine

#### 1. Live Flight Schedule & Dynamic Fare Quotes
Search and filter across domestic and international routes (`DEL → BOM`, `HYD → BLR`, `BOM → DXB`, `BLR → SIN`, `DEL → LHR`, `MAA → DEL`) with real-time dynamic multipliers, cabin class switching (`Economy`, `Premium Economy`, `Business`), and EMSR-b active bucket indicators.

![Live Flight Schedule & Dynamic Fares](docs/screenshots/01_flight_routes_and_fares.jpg.png)

#### 2. Interactive 20×6 Cabin Seat Map & Checkout
Select available seats directly on the cabin layout with automatic ancillary surcharges for Exit Rows (+12%), Window seats (+6%), and Aisle seats (+2%), displaying a transparent fare breakdown prior to booking.

![Interactive Cabin Seat Map & Checkout](docs/screenshots/02_cabin_seat_map_checkout.jpg.png)

#### 3. PNR Confirmation, Cancellation & Market Simulator
Confirming a reservation issues a unique PNR and immediately reprices remaining seats as load factor increases. Cancelling a PNR releases seats back to inventory, while live sliders simulate Days to Departure, Booking Velocity, and Seasonality regimes.

![PNR Ledger & Market Simulator](docs/screenshots/03_pnr_ledger_market_sim.jpg.png)

---

### Dynamic Pricing & EMSR-b Optimization

#### 4. 90-Day Fare Trajectory & Pricing Multipliers
Tracks dynamic pricing curves across Low (25%), Current, and Surge (90%) cabin load regimes alongside real-time urgency, scarcity, velocity, and competitor index KPIs.

![90-Day Fare Trajectory & Pricing Multipliers](docs/screenshots/04_dynamic_pricing_trajectory.jpg.png)

#### 5. EMSR-b Nested Fare Buckets & Price Elasticity Curve
Calculates Belobaba's EMSR-b protection levels ($\theta_j$) and booking limits across nested buckets (`Y`, `B`, `M`, `Q`, `L`), paired with the iso-elastic revenue optimization curve $Q(P) = Q_0 \cdot (P/P_0)^{-\epsilon}$.

![EMSR-b Nested Fare Buckets & Elasticity Curve](docs/screenshots/05_emsr_b_bucket_allocation.jpg.png)

---

### ML Demand Forecasting (`scikit-learn`)

#### 6. Ensemble Model Accuracy & Demand Prediction
Benchmarks `RandomForestRegressor` and `GradientBoostingRegressor` models on synthetic flight booking telemetry using `R²`, `RMSE`, and `MAE`, with interactive seat-demand inference.

![ML Model Benchmark & Demand Prediction](docs/screenshots/06_ml_model_metrics_accuracy.jpg.png)

#### 7. Feature Importance & Willingness-to-Pay Drivers
Visualizes normalized feature importances across Days to Departure, Load Factor, Seasonality, and Historical Booking Velocity to explain fare predictions transparently.

![ML Feature Importance & Drivers](docs/screenshots/07_ml_feature_importance_curve.jpg.png)

---

### Non-Homogeneous Poisson Monte Carlo Simulator

#### 8. Static vs. Dynamic Cumulative Revenue Curve
Simulates stochastic passenger arrivals via a Non-Homogeneous Poisson Process (NHPP) and compares cumulative Static Fixed Pricing revenue against Dynamic EMSR-b Pricing to quantify net Yield Lift (%).

![Monte Carlo Cumulative Revenue Curve](docs/screenshots/08_monte_carlo_revenue_curve.jpg.png)

#### 9. Spoilage Reduction, Event Log & CSV Export
Displays final cabin utilization, rejected high-WTP vs. low-WTP arrival outcomes, tick-by-tick booking telemetry, and downloadable simulation CSV reports.

![Monte Carlo Event Log & Yield Metrics](docs/screenshots/09_monte_carlo_event_log.jpg.png)

---

## Key Capabilities

- **Multi-Factor Dynamic Pricing (`src/pricing_engine.py`)**: Exponential time-to-departure urgency, convex load-factor scarcity, booking velocity surge detection, competitor anchoring, and floor/ceiling guardrails.
- **EMSR-b Seat Inventory Nesting**: Normal inverse CDF (`scipy.stats.norm.ppf`) protection levels across 5 nested fare classes (`Y`, `B`, `M`, `Q`, `L`).
- **ML Demand Forecaster (`src/ml_demand_model.py`)**: `RandomForestRegressor` and `GradientBoostingRegressor` models evaluated with `RMSE`, `MAE`, `R²`, and feature importance ranking.
- **SQLite + SQLAlchemy Persistence (`src/database.py`, `src/models.py`)**: Atomic seat reservation, PNR generation, pricing history logs, and cancellation rollback.
- **Automated Test Suite (`tests/test_pricing.py`)**: `pytest` unit and FastAPI integration coverage.

---

## Architecture

```text
+-------------------------------------------------------------------+
|                  Streamlit Interactive Dashboard                   |
|   (Flight Booking | Seat Map | Pricing Lab | ML | Monte Carlo)    |
+---------------------------------+---------------------------------+
                                  | REST / JSON
                                  v
+-------------------------------------------------------------------+
|                      FastAPI Application (app.py)                  |
|  /flights   /quote   /bookings   /cancel   /ml/train   /simulate   |
+----------------+--------------------------------+-----------------+
                 |                                |
                 v                                v
+--------------------------------+  +-------------------------------+
|  Dynamic Pricing & EMSR-b Core |  |  ML Demand Model (sklearn)    |
|  - Time Decay & Load Factor    |  |  - RandomForestRegressor      |
|  - Littlewood / EMSR-b Buckets |  |  - GradientBoostingRegressor  |
|  - Elasticity & Guardrails     |  |  - Feature Importance & R2    |
+----------------+---------------+  +-------------------------------+
                 |
                 v
+-------------------------------------------------------------------+
|              SQLAlchemy ORM + SQLite (aeroyield.db)               |
|         Flights | Bookings | PricingLogs | SimulationRuns         |
+-------------------------------------------------------------------+
```

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/aeroyield-flight-dynamic-pricing.git
cd aeroyield-flight-dynamic-pricing

python -m venv venv
# Windows: venv\Scripts\activate
# macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start FastAPI Backend
```bash
uvicorn app:app --reload --port 8000
```
Open interactive Swagger API docs at `http://127.0.0.1:8000/docs`.

### 3. Start Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Open dashboard at `http://localhost:8501`.

### 4. Run Test Suite
```bash
pytest -v
```

---

## Repository Structure

```text
aeroyield-flight-dynamic-pricing/
├── app.py                              # FastAPI REST API server
├── streamlit_app.py                    # Interactive Streamlit UI dashboard
├── requirements.txt                    # Python dependencies
├── README.md                           # Project overview & visual walkthrough
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── models.py                       # SQLAlchemy ORM & Pydantic schemas
│   ├── database.py                     # SQLite engine & initial route seeder
│   ├── pricing_engine.py               # EMSR-b & multi-factor pricing core
│   ├── ml_demand_model.py              # scikit-learn demand & WTP forecaster
│   └── simulator.py                    # NHPP Monte Carlo simulation engine
├── tests/
│   └── test_pricing.py                 # Pytest unit & integration tests
└── docs/
    ├── ARCHITECTURE_AND_ALGORITHMS.md  # Mathematical specs & API reference
    └── screenshots/                    # UI explanation screenshots for GitHub
        ├── 01_flight_routes_and_fares.jpg.png
        ├── 02_cabin_seat_map_checkout.jpg.png
        ├── 03_pnr_ledger_market_sim.jpg.png
        ├── 04_dynamic_pricing_trajectory.jpg.png
        ├── 05_emsr_b_bucket_allocation.jpg.png
        ├── 06_ml_model_metrics_accuracy.jpg.png
        ├── 07_ml_feature_importance_curve.jpg.png
        ├── 08_monte_carlo_revenue_curve.jpg.png
        └── 09_monte_carlo_event_log.jpg.png
```

---

## License
MIT License
