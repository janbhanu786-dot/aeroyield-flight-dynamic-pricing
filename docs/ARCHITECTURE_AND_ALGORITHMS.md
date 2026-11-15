# Architecture & Algorithmic Design

## 1. System Overview
**AeroYield** is a full-stack airline revenue management and flight booking simulator built with Python (`FastAPI`, `SQLAlchemy`, `scikit-learn`, `SciPy`, `Streamlit`). It simulates how modern airline pricing engines adjust ticket fares in real time based on seat scarcity, booking velocity, time remaining until departure, competitor benchmarks, and passenger price elasticity.

---

## 2. Core Modules
1. **Seat Inventory & Reservation Engine (`src/database.py`, `src/models.py`)**
   - SQLAlchemy ORM backed by SQLite (`flights`, `bookings`, `pricing_logs`).
   - Manages 120-seat aircraft cabin layouts, deterministic PNR generation, seat position surcharges (Exit Rows 10–11 `+12%`, Window A/F `+6%`), and atomic booking/cancellation rollbacks.
2. **EMSR-b & Multi-Factor Dynamic Pricing Core (`src/pricing_engine.py`)**
   - Implements Belobaba's **Expected Marginal Seat Revenue - version b (EMSR-b)** across 5 nested fare buckets (`Y`, `B`, `M`, `Q`, `L`).
   - Computes protection levels $\theta_j$ and booking limits $BL_j$ using the inverse normal CDF (`scipy.stats.norm.ppf`).
   - Enforces floor ($0.72 \times P_{\text{base}}$) and ceiling ($3.25 \times P_{\text{base}}$) guardrails alongside elasticity dampening.
3. **Machine Learning Demand & Multiplier Predictor (`src/ml_demand_model.py`)**
   - Synthesizes realistic historical flight booking records and trains `RandomForestRegressor` and `GradientBoostingRegressor` models.
   - Evaluates models using `RMSE`, `MAE`, and `R²` and ranks feature importances (`days_to_departure`, `load_factor`, `booking_velocity`, `competitor_ratio`, `elasticity`).
4. **NHPP Monte Carlo Yield Simulator (`src/simulator.py`)**
   - Simulates passenger arrivals as a **Non-Homogeneous Poisson Process (NHPP)** whose arrival intensity $\lambda(t)$ increases as departure nears.
   - Draws customer willingness-to-pay from a log-normal distribution and compares **Static Fixed Pricing** against **Dynamic EMSR-b Pricing** across $N$ runs.

---

## 3. Mathematical Specifications

### Multi-Factor Dynamic Fare
$$P_{\text{dynamic}} = \text{clamp}\Big( P_{\text{base}} \cdot M_{\text{time}}(t) \cdot M_{\text{load}}(LF) \cdot M_{\text{velocity}}(v) \cdot M_{\text{season}} \cdot M_{\text{comp}}, \; P_{\min}, \; P_{\max} \Big)$$

- **Time Urgency**: $M_{\text{time}}(t) = 1 + 0.58 \cdot e^{-0.115 t}$
- **Load Scarcity**: $M_{\text{load}}(LF) = 1 + 0.72 \cdot LF^{2.35}$
- **Iso-Elastic Optimal Fare**: $P^* = MC \cdot \frac{\epsilon}{\epsilon - 1}$

---

## 4. API Endpoints (`app.py`)
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/flights` | List all flights with live dynamic fare quotes |
| `POST` | `/quote` | Compute real-time pricing breakdown & EMSR-b buckets |
| `POST` | `/bookings` | Reserve seats, issue PNR, and trigger inventory repricing |
| `POST` | `/bookings/{pnr}/cancel` | Cancel PNR, release seats, and recalculate fares |
| `POST` | `/ml/train` | Train Random Forest & Gradient Boosting demand models |
| `POST` | `/simulate/monte-carlo` | Run NHPP Static vs. Dynamic revenue benchmark |
