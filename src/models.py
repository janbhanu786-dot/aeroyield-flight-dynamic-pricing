"""SQLAlchemy ORM models and Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class FlightORM(Base):
    __tablename__ = "flights"

    id = Column(String(32), primary_key=True, index=True)
    flight_number = Column(String(16), nullable=False)
    airline = Column(String(64), nullable=False)
    origin = Column(String(8), nullable=False)
    origin_city = Column(String(64), nullable=False)
    destination = Column(String(8), nullable=False)
    destination_city = Column(String(64), nullable=False)
    aircraft = Column(String(64), nullable=False)
    departure_time = Column(String(16), nullable=False)
    arrival_time = Column(String(16), nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    days_to_departure = Column(Integer, nullable=False, default=14)
    total_seats = Column(Integer, nullable=False, default=120)
    booked_seats = Column(Integer, nullable=False, default=42)
    occupied_seat_codes = Column(Text, nullable=False, default="")
    base_fare = Column(Float, nullable=False)
    competitor_fare = Column(Float, nullable=False)
    elasticity = Column(Float, nullable=False, default=1.25)
    booking_velocity_per_hr = Column(Float, nullable=False, default=2.4)
    season_mode = Column(String(32), nullable=False, default="regular")

    bookings = relationship("BookingORM", back_populates="flight", cascade="all, delete-orphan")


class BookingORM(Base):
    __tablename__ = "bookings"

    pnr = Column(String(16), primary_key=True, index=True)
    flight_id = Column(String(32), ForeignKey("flights.id"), nullable=False)
    passenger_name = Column(String(128), nullable=False)
    passenger_email = Column(String(128), nullable=False)
    cabin_class = Column(String(32), nullable=False, default="Economy")
    seats = Column(Text, nullable=False)  # Comma-separated seat codes
    fare_bucket = Column(String(8), nullable=False)
    base_fare_snapshot = Column(Float, nullable=False)
    dynamic_multiplier = Column(Float, nullable=False)
    seat_surcharge = Column(Float, nullable=False, default=0.0)
    total_amount = Column(Float, nullable=False)
    status = Column(String(24), nullable=False, default="CONFIRMED")
    created_at = Column(DateTime, default=datetime.utcnow)

    flight = relationship("FlightORM", back_populates="bookings")


class PricingLogORM(Base):
    __tablename__ = "pricing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    flight_id = Column(String(32), nullable=False, index=True)
    load_factor = Column(Float, nullable=False)
    days_to_departure = Column(Integer, nullable=False)
    quoted_fare = Column(Float, nullable=False)
    multiplier = Column(Float, nullable=False)
    active_bucket = Column(String(8), nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)


# --- Pydantic API Schemas ---

class PricingQuoteRequest(BaseModel):
    base_fare: float = Field(..., gt=0)
    days_to_departure: int = Field(..., ge=0, le=365)
    total_seats: int = Field(120, gt=0)
    booked_seats: int = Field(40, ge=0)
    booking_velocity_per_hr: float = Field(2.0, ge=0)
    elasticity: float = Field(1.25, ge=0.3, le=3.5)
    competitor_fare: float = Field(..., gt=0)
    season_mode: str = Field("regular")
    cabin_class: str = Field("Economy")


class FareBucketAllocation(BaseModel):
    code: str
    name: str
    fare: float
    mean_demand: float
    std_demand: float
    protection_level: int
    booking_limit: int
    is_active: bool


class PricingQuoteResponse(BaseModel):
    base_fare: float
    dynamic_fare: float
    total_multiplier: float
    load_factor: float
    active_bucket: str
    components: Dict[str, float]
    emsrb_buckets: List[FareBucketAllocation]
    price_floor: float
    price_ceiling: float
    revenue_optimal_price: float


class CreateBookingRequest(BaseModel):
    flight_id: str
    passenger_name: str = Field(..., min_length=2)
    passenger_email: str
    cabin_class: str = Field("Economy")
    seats: List[str] = Field(..., min_length=1)


class MonteCarloRequest(BaseModel):
    base_fare: float = Field(5400.0, gt=0)
    capacity: int = Field(120, ge=20, le=400)
    horizon_days: int = Field(60, ge=7, le=120)
    mean_daily_arrivals: float = Field(2.4, gt=0.1, le=20.0)
    elasticity: float = Field(1.25, ge=0.4, le=3.0)
    num_simulations: int = Field(25, ge=5, le=200)
    seed: Optional[int] = 42
