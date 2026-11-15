"""SQLite database initialization and initial aviation route seeding."""

from __future__ import annotations

import os
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.models import Base, FlightORM

DATABASE_URL = os.getenv("AEROYIELD_DB_URL", "sqlite:///./aeroyield.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create database tables and seed default routes if empty."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if db.query(FlightORM).count() == 0:
            seed_flights(db)


def _generate_initial_occupied(total_seats: int, booked_count: int) -> str:
    """Deterministically pick realistic occupied seat codes (1A..20F)."""
    cols = ["A", "B", "C", "D", "E", "F"]
    rows = (total_seats + 5) // 6
    all_seats: List[str] = [f"{r}{c}" for r in range(1, rows + 1) for c in cols][:total_seats]
    target = min(booked_count, len(all_seats))
    occupied: List[str] = []
    used = set()
    idx = 0
    while len(occupied) < target:
        candidate = idx % len(all_seats)
        while candidate in used:
            candidate = (candidate + 1) % len(all_seats)
        used.add(candidate)
        occupied.append(all_seats[candidate])
        idx += 7
    return ",".join(occupied)


def seed_flights(db: Session) -> None:
    """Seed realistic domestic and international routes."""
    initial_routes = [
        {
            "id": "AY-204",
            "flight_number": "AY 204",
            "airline": "AeroYield Airways",
            "origin": "DEL",
            "origin_city": "New Delhi",
            "destination": "BOM",
            "destination_city": "Mumbai",
            "aircraft": "Airbus A320neo",
            "departure_time": "06:45",
            "arrival_time": "09:00",
            "duration_minutes": 135,
            "days_to_departure": 12,
            "total_seats": 120,
            "booked_seats": 74,
            "base_fare": 5200.0,
            "competitor_fare": 5650.0,
            "elasticity": 1.18,
            "booking_velocity_per_hr": 3.8,
            "season_mode": "peak_business",
        },
        {
            "id": "AY-418",
            "flight_number": "AY 418",
            "airline": "AeroYield Airways",
            "origin": "HYD",
            "origin_city": "Hyderabad",
            "destination": "BLR",
            "destination_city": "Bengaluru",
            "aircraft": "ATR 72-600 / A320",
            "departure_time": "09:30",
            "arrival_time": "10:45",
            "duration_minutes": 75,
            "days_to_departure": 24,
            "total_seats": 120,
            "booked_seats": 42,
            "base_fare": 3400.0,
            "competitor_fare": 3550.0,
            "elasticity": 1.42,
            "booking_velocity_per_hr": 1.9,
            "season_mode": "regular",
        },
        {
            "id": "AY-802",
            "flight_number": "AY 802",
            "airline": "AeroYield International",
            "origin": "BOM",
            "origin_city": "Mumbai",
            "destination": "DXB",
            "destination_city": "Dubai",
            "aircraft": "Boeing 787-8 Dreamliner",
            "departure_time": "14:15",
            "arrival_time": "16:10",
            "duration_minutes": 205,
            "days_to_departure": 6,
            "total_seats": 120,
            "booked_seats": 98,
            "base_fare": 14800.0,
            "competitor_fare": 16200.0,
            "elasticity": 0.92,
            "booking_velocity_per_hr": 5.4,
            "season_mode": "festival",
        },
        {
            "id": "AY-512",
            "flight_number": "AY 512",
            "airline": "AeroYield International",
            "origin": "BLR",
            "origin_city": "Bengaluru",
            "destination": "SIN",
            "destination_city": "Singapore",
            "aircraft": "Airbus A321neo",
            "departure_time": "22:50",
            "arrival_time": "05:55",
            "duration_minutes": 275,
            "days_to_departure": 18,
            "total_seats": 120,
            "booked_seats": 66,
            "base_fare": 16500.0,
            "competitor_fare": 15900.0,
            "elasticity": 1.15,
            "booking_velocity_per_hr": 2.7,
            "season_mode": "regular",
        },
        {
            "id": "AY-101",
            "flight_number": "AY 101",
            "airline": "AeroYield International",
            "origin": "DEL",
            "origin_city": "New Delhi",
            "destination": "LHR",
            "destination_city": "London Heathrow",
            "aircraft": "Boeing 777-300ER",
            "departure_time": "02:30",
            "arrival_time": "07:45",
            "duration_minutes": 585,
            "days_to_departure": 35,
            "total_seats": 120,
            "booked_seats": 54,
            "base_fare": 42000.0,
            "competitor_fare": 44500.0,
            "elasticity": 0.88,
            "booking_velocity_per_hr": 1.5,
            "season_mode": "regular",
        },
        {
            "id": "AY-309",
            "flight_number": "AY 309",
            "airline": "AeroYield Airways",
            "origin": "MAA",
            "origin_city": "Chennai",
            "destination": "DEL",
            "destination_city": "New Delhi",
            "aircraft": "Airbus A320neo",
            "departure_time": "17:20",
            "arrival_time": "20:10",
            "duration_minutes": 170,
            "days_to_departure": 4,
            "total_seats": 120,
            "booked_seats": 106,
            "base_fare": 6100.0,
            "competitor_fare": 6800.0,
            "elasticity": 0.95,
            "booking_velocity_per_hr": 6.1,
            "season_mode": "peak_business",
        },
    ]

    for route in initial_routes:
        occupied_codes = _generate_initial_occupied(route["total_seats"], route["booked_seats"])
        flight = FlightORM(**route, occupied_seat_codes=occupied_codes)
        db.add(flight)
    db.commit()


def get_db():
    """FastAPI dependency yielding a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
