"""
AeroYield FastAPI Backend Server.

Run locally:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import secrets
import string
from contextlib import asynccontextmanager
from typing import Any, Dict, List
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.database import get_db, init_db
from src.ml_demand_model import DemandPricingMLModel
from src.models import (
    BookingORM,
    CreateBookingRequest,
    FlightORM,
    MonteCarloRequest,
    PricingLogORM,
    PricingQuoteRequest,
)
from src.pricing_engine import DynamicPricingEngine
from src.simulator import run_monte_carlo_comparison


pricing_engine = DynamicPricingEngine()
ml_model = DemandPricingMLModel()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    ml_model.train(n_samples=2500)
    yield


app = FastAPI(
    title="AeroYield Dynamic Pricing & Flight Booking Simulator API",
    description="Airline Revenue Management, EMSR-b Seat Optimization, ML Demand Forecasting & Monte Carlo Simulation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _seat_surcharge(seat_code: str, base_fare: float) -> float:
    """Window (A/F) and Exit Row (rows 10, 11) seat surcharges."""
    col = seat_code[-1].upper()
    row = int("".join(ch for ch in seat_code if ch.isdigit()) or "1")
    surcharge = 0.0
    if row in (10, 11):
        surcharge += round(base_fare * 0.12, 2)
    elif col in ("A", "F"):
        surcharge += round(base_fare * 0.06, 2)
    return surcharge


def _generate_pnr() -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "AY" + "".join(secrets.choice(alphabet) for _ in range(4))


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "AeroYield Dynamic Pricing API"}


@app.get("/flights")
def list_flights(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    flights = db.query(FlightORM).all()
    results = []
    for f in flights:
        quote = pricing_engine.quote(
            PricingQuoteRequest(
                base_fare=f.base_fare,
                days_to_departure=f.days_to_departure,
                total_seats=f.total_seats,
                booked_seats=f.booked_seats,
                booking_velocity_per_hr=f.booking_velocity_per_hr,
                elasticity=f.elasticity,
                competitor_fare=f.competitor_fare,
                season_mode=f.season_mode,
                cabin_class="Economy",
            )
        )
        occupied = [s for s in f.occupied_seat_codes.split(",") if s]
        results.append(
            {
                "id": f.id,
                "flight_number": f.flight_number,
                "airline": f.airline,
                "origin": f.origin,
                "origin_city": f.origin_city,
                "destination": f.destination,
                "destination_city": f.destination_city,
                "aircraft": f.aircraft,
                "departure_time": f.departure_time,
                "arrival_time": f.arrival_time,
                "duration_minutes": f.duration_minutes,
                "days_to_departure": f.days_to_departure,
                "total_seats": f.total_seats,
                "booked_seats": f.booked_seats,
                "occupied_seats": occupied,
                "base_fare": f.base_fare,
                "competitor_fare": f.competitor_fare,
                "elasticity": f.elasticity,
                "booking_velocity_per_hr": f.booking_velocity_per_hr,
                "season_mode": f.season_mode,
                "quote": quote.model_dump(),
            }
        )
    return results


@app.post("/quote")
def compute_quote(req: PricingQuoteRequest):
    return pricing_engine.quote(req)


@app.post("/bookings")
def create_booking(req: CreateBookingRequest, db: Session = Depends(get_db)):
    flight = db.query(FlightORM).filter(FlightORM.id == req.flight_id).first()
    if not flight:
        raise HTTPException(status_code=404, detail="Flight not found")

    occupied = set(s for s in flight.occupied_seat_codes.split(",") if s)
    requested = [s.strip().upper() for s in req.seats]

    conflict = [s for s in requested if s in occupied]
    if conflict:
        raise HTTPException(status_code=409, detail=f"Seats already occupied: {', '.join(conflict)}")

    if flight.booked_seats + len(requested) > flight.total_seats:
        raise HTTPException(status_code=400, detail="Insufficient seat capacity on flight")

    quote = pricing_engine.quote(
        PricingQuoteRequest(
            base_fare=flight.base_fare,
            days_to_departure=flight.days_to_departure,
            total_seats=flight.total_seats,
            booked_seats=flight.booked_seats,
            booking_velocity_per_hr=flight.booking_velocity_per_hr,
            elasticity=flight.elasticity,
            competitor_fare=flight.competitor_fare,
            season_mode=flight.season_mode,
            cabin_class=req.cabin_class,
        )
    )

    seat_surcharge = round(sum(_seat_surcharge(s, quote.dynamic_fare) for s in requested), 2)
    total_amount = round(quote.dynamic_fare * len(requested) + seat_surcharge, 2)

    pnr = _generate_pnr()
    booking = BookingORM(
        pnr=pnr,
        flight_id=flight.id,
        passenger_name=req.passenger_name,
        passenger_email=req.passenger_email,
        cabin_class=req.cabin_class,
        seats=",".join(requested),
        fare_bucket=quote.active_bucket,
        base_fare_snapshot=quote.base_fare,
        dynamic_multiplier=quote.total_multiplier,
        seat_surcharge=seat_surcharge,
        total_amount=total_amount,
        status="CONFIRMED",
    )

    occupied.update(requested)
    flight.occupied_seat_codes = ",".join(sorted(occupied))
    flight.booked_seats += len(requested)
    flight.booking_velocity_per_hr = round(flight.booking_velocity_per_hr + 0.4 * len(requested), 2)

    log = PricingLogORM(
        flight_id=flight.id,
        load_factor=round(flight.booked_seats / flight.total_seats, 4),
        days_to_departure=flight.days_to_departure,
        quoted_fare=quote.dynamic_fare,
        multiplier=quote.total_multiplier,
        active_bucket=quote.active_bucket,
    )

    db.add(booking)
    db.add(log)
    db.commit()
    db.refresh(booking)

    return {
        "pnr": booking.pnr,
        "flight_id": booking.flight_id,
        "passenger_name": booking.passenger_name,
        "seats": requested,
        "fare_bucket": booking.fare_bucket,
        "total_amount": booking.total_amount,
        "status": booking.status,
    }


@app.get("/bookings")
def list_bookings(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    rows = db.query(BookingORM).order_by(BookingORM.created_at.desc()).all()
    return [
        {
            "pnr": b.pnr,
            "flight_id": b.flight_id,
            "passenger_name": b.passenger_name,
            "passenger_email": b.passenger_email,
            "cabin_class": b.cabin_class,
            "seats": [s for s in b.seats.split(",") if s],
            "fare_bucket": b.fare_bucket,
            "dynamic_multiplier": b.dynamic_multiplier,
            "total_amount": b.total_amount,
            "status": b.status,
            "created_at": b.created_at.isoformat() if b.created_at else None,
        }
        for b in rows
    ]


@app.post("/bookings/{pnr}/cancel")
def cancel_booking(pnr: str, db: Session = Depends(get_db)):
    booking = db.query(BookingORM).filter(BookingORM.pnr == pnr).first()
    if not booking:
        raise HTTPException(status_code=404, detail="Booking PNR not found")
    if booking.status == "CANCELLED":
        raise HTTPException(status_code=400, detail="Booking already cancelled")

    flight = db.query(FlightORM).filter(FlightORM.id == booking.flight_id).first()
    if flight:
        seats_to_release = set(s for s in booking.seats.split(",") if s)
        occupied = set(s for s in flight.occupied_seat_codes.split(",") if s)
        occupied -= seats_to_release
        flight.occupied_seat_codes = ",".join(sorted(occupied))
        flight.booked_seats = max(0, flight.booked_seats - len(seats_to_release))

    booking.status = "CANCELLED"
    db.commit()
    return {"pnr": pnr, "status": "CANCELLED", "refund_amount": round(booking.total_amount * 0.85, 2)}


@app.post("/simulate/monte-carlo")
def simulate_monte_carlo(req: MonteCarloRequest):
    return run_monte_carlo_comparison(req)


@app.post("/ml/train")
def train_ml_demand_model(n_samples: int = 2500):
    return ml_model.train(n_samples=n_samples)
