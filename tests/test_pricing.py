"""Automated Pytest suite for Dynamic Pricing Engine and FastAPI endpoints."""

from fastapi.testclient import TestClient

from app import app
from src.models import MonteCarloRequest, PricingQuoteRequest
from src.pricing_engine import DynamicPricingEngine
from src.simulator import run_monte_carlo_comparison


engine = DynamicPricingEngine()


def test_time_urgency_increases_near_departure():
    far_quote = engine.quote(
        PricingQuoteRequest(
            base_fare=5000,
            days_to_departure=60,
            total_seats=120,
            booked_seats=40,
            competitor_fare=5000,
        )
    )
    near_quote = engine.quote(
        PricingQuoteRequest(
            base_fare=5000,
            days_to_departure=2,
            total_seats=120,
            booked_seats=40,
            competitor_fare=5000,
        )
    )
    assert near_quote.dynamic_fare > far_quote.dynamic_fare


def test_load_factor_scarcity_increases_fare():
    low_lf = engine.quote(
        PricingQuoteRequest(
            base_fare=5000,
            days_to_departure=15,
            total_seats=120,
            booked_seats=20,
            competitor_fare=5000,
        )
    )
    high_lf = engine.quote(
        PricingQuoteRequest(
            base_fare=5000,
            days_to_departure=15,
            total_seats=120,
            booked_seats=108,
            competitor_fare=5000,
        )
    )
    assert high_lf.dynamic_fare > low_lf.dynamic_fare


def test_price_respects_floor_and_ceiling_guardrails():
    quote = engine.quote(
        PricingQuoteRequest(
            base_fare=6000,
            days_to_departure=0,
            total_seats=120,
            booked_seats=120,
            booking_velocity_per_hr=15.0,
            competitor_fare=12000,
            season_mode="festival",
        )
    )
    assert quote.price_floor <= quote.dynamic_fare <= quote.price_ceiling


def test_emsrb_buckets_monotonically_nested():
    buckets = engine.compute_emsrb_buckets(base_fare=5000, total_seats=120, booked_seats=60, days_to_departure=10)
    assert len(buckets) == 5
    fares = [b.fare for b in buckets]
    assert fares == sorted(fares, reverse=True)


def test_monte_carlo_positive_output():
    result = run_monte_carlo_comparison(MonteCarloRequest(num_simulations=5, horizon_days=30, seed=7))
    assert result["avg_dynamic_revenue"] > 0
    assert len(result["daily_trajectory"]) == 30


def test_api_flights_and_booking_lifecycle():
    with TestClient(app) as client:
        flights_res = client.get("/flights")
        assert flights_res.status_code == 200
        flights = flights_res.json()
        assert len(flights) >= 1

        target = flights[0]
        occupied = set(target["occupied_seats"])
        candidate = next(f"{r}{c}" for r in range(1, 21) for c in "ABCDEF" if f"{r}{c}" not in occupied)

        book_res = client.post(
            "/bookings",
            json={
                "flight_id": target["id"],
                "passenger_name": "Aarav Sharma",
                "passenger_email": "aarav@example.com",
                "cabin_class": "Economy",
                "seats": [candidate],
            },
        )
        assert book_res.status_code == 200
        pnr = book_res.json()["pnr"]

        cancel_res = client.post(f"/bookings/{pnr}/cancel")
        assert cancel_res.status_code == 200
        assert cancel_res.json()["status"] == "CANCELLED"
