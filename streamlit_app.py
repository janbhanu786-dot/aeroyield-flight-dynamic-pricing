"""
AeroYield Interactive Streamlit Dashboard (Standalone Python UI).

Run locally:
    streamlit run streamlit_app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import SessionLocal, init_db
from src.ml_demand_model import DemandPricingMLModel
from src.models import FlightORM, MonteCarloRequest, PricingQuoteRequest
from src.pricing_engine import DynamicPricingEngine
from src.simulator import run_monte_carlo_comparison

st.set_page_config(page_title="AeroYield — Flight Dynamic Pricing Simulator", layout="wide")
init_db()

engine = DynamicPricingEngine()
if "ml_model" not in st.session_state:
    model = DemandPricingMLModel()
    model.train(2500)
    st.session_state["ml_model"] = model

st.title("AeroYield — Flight Booking & Dynamic Pricing Simulator")
st.caption("EMSR-b Revenue Management · scikit-learn Demand Forecasting · NHPP Monte Carlo")

tab_book, tab_lab, tab_ml, tab_mc = st.tabs(
    ["Flight Booking", "Dynamic Pricing Lab", "ML Demand Forecaster", "Monte Carlo Simulator"]
)

with tab_book:
    with SessionLocal() as db:
        flights = db.query(FlightORM).all()
        rows = []
        for f in flights:
            q = engine.quote(
                PricingQuoteRequest(
                    base_fare=f.base_fare,
                    days_to_departure=f.days_to_departure,
                    total_seats=f.total_seats,
                    booked_seats=f.booked_seats,
                    booking_velocity_per_hr=f.booking_velocity_per_hr,
                    elasticity=f.elasticity,
                    competitor_fare=f.competitor_fare,
                    season_mode=f.season_mode,
                )
            )
            rows.append(
                {
                    "Flight": f.flight_number,
                    "Route": f"{f.origin} -> {f.destination}",
                    "Aircraft": f.aircraft,
                    "Days Left": f.days_to_departure,
                    "Load Factor": f"{round(q.load_factor * 100, 1)}%",
                    "Bucket": q.active_bucket,
                    "Base Fare (INR)": f.base_fare,
                    "Dynamic Fare (INR)": q.dynamic_fare,
                    "Multiplier": f"{q.total_multiplier:.2f}x",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_lab:
    c1, c2 = st.columns([1, 2])
    with c1:
        base_fare = st.slider("Base Fare (INR)", 2500, 25000, 5400, step=100)
        days_left = st.slider("Days to Departure", 0, 90, 12)
        booked = st.slider("Booked Seats (of 120)", 0, 120, 74)
        velocity = st.slider("Booking Velocity (/hr)", 0.2, 10.0, 3.5, step=0.1)
        elasticity = st.slider("Price Elasticity", 0.5, 2.5, 1.25, step=0.05)
        season = st.selectbox("Seasonality", ["off_peak", "regular", "peak_business", "festival"])

    quote = engine.quote(
        PricingQuoteRequest(
            base_fare=base_fare,
            days_to_departure=days_left,
            total_seats=120,
            booked_seats=booked,
            booking_velocity_per_hr=velocity,
            elasticity=elasticity,
            competitor_fare=base_fare * 1.05,
            season_mode=season,
        )
    )
    with c2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dynamic Fare", f"INR {quote.dynamic_fare:,.0f}", f"{quote.total_multiplier:.2f}x")
        m2.metric("Active Bucket", quote.active_bucket)
        m3.metric("Load Factor", f"{quote.load_factor * 100:.1f}%")
        m4.metric("Optimal P*", f"INR {quote.revenue_optimal_price:,.0f}")

        curve_days = list(range(90, -1, -3))
        curve_fares = [
            engine.quote(
                PricingQuoteRequest(
                    base_fare=base_fare,
                    days_to_departure=d,
                    total_seats=120,
                    booked_seats=booked,
                    booking_velocity_per_hr=velocity,
                    elasticity=elasticity,
                    competitor_fare=base_fare * 1.05,
                    season_mode=season,
                )
            ).dynamic_fare
            for d in curve_days
        ]
        fig = px.line(x=curve_days, y=curve_fares, labels={"x": "Days to Departure", "y": "Dynamic Fare (INR)"})
        fig.update_xaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

with tab_ml:
    ml = st.session_state["ml_model"]
    st.subheader("scikit-learn Ensemble Evaluation (2,500 Historical Flights)")
    col_rf, col_gb = st.columns(2)
    col_rf.json({"RandomForestRegressor": ml.metrics["random_forest"]})
    col_gb.json({"GradientBoostingRegressor": ml.metrics["gradient_boosting"]})
    fi_df = pd.DataFrame(
        [{"Feature": k, "Importance": v} for k, v in ml.metrics["feature_importance"].items()]
    )
    st.bar_chart(fi_df.set_index("Feature"))

with tab_mc:
    sims = st.slider("Monte Carlo Horizon Simulations", 5, 100, 25)
    if st.button("Run NHPP Monte Carlo Benchmark"):
        res = run_monte_carlo_comparison(MonteCarloRequest(num_simulations=sims))
        st.success(f"Dynamic EMSR-b Yield Lift: +{res['yield_lift_pct']}% over Static Pricing")
        traj = pd.DataFrame(res["daily_trajectory"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=traj["day"], y=traj["static_revenue"], name="Static Pricing"))
        fig.add_trace(go.Scatter(x=traj["day"], y=traj["dynamic_revenue"], name="Dynamic EMSR-b Pricing"))
        st.plotly_chart(fig, use_container_width=True)
