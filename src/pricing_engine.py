"""
Core Dynamic Pricing & EMSR-b Seat Inventory Optimization Engine.

Implements:
1. Belobaba's EMSR-b (Expected Marginal Seat Revenue - version b) nested fare buckets.
2. Exponential Time-to-Departure urgency curve.
3. Non-linear Load Factor scarcity multiplier.
4. Booking velocity surge detection.
5. Competitor price anchoring and revenue-optimal elasticity bounds.
"""

from __future__ import annotations

import math
from typing import Dict, List
from scipy.stats import norm

from src.models import FareBucketAllocation, PricingQuoteRequest, PricingQuoteResponse


SEASON_MULTIPLIERS: Dict[str, float] = {
    "off_peak": 0.88,
    "regular": 1.00,
    "peak_business": 1.14,
    "festival": 1.28,
}

CABIN_MULTIPLIERS: Dict[str, float] = {
    "Economy": 1.00,
    "Premium Economy": 1.48,
    "Business": 2.65,
}


class DynamicPricingEngine:
    """Algorithmic Airline Revenue Management & Dynamic Pricing Engine."""

    def __init__(self, floor_ratio: float = 0.72, ceiling_ratio: float = 3.25) -> None:
        self.floor_ratio = floor_ratio
        self.ceiling_ratio = ceiling_ratio

    @staticmethod
    def time_urgency_factor(days_to_departure: int) -> float:
        """
        Exponential urgency surge as departure approaches.
        At 60+ days out: ~1.00x. Within 7 days: rises sharply up to ~1.58x.
        """
        t = max(0, days_to_departure)
        return round(1.0 + 0.58 * math.exp(-0.115 * t), 4)

    @staticmethod
    def load_factor_scarcity(load_factor: float) -> float:
        """
        Smooth convex scarcity multiplier based on cabin occupancy (0..1).
        """
        lf = min(1.0, max(0.0, load_factor))
        return round(1.0 + 0.72 * (lf ** 2.35), 4)

    @staticmethod
    def velocity_surge_factor(velocity_per_hr: float) -> float:
        """
        Detects abnormal hourly booking acceleration relative to 2.0/hr baseline.
        """
        excess = max(0.0, velocity_per_hr - 2.0)
        return round(1.0 + min(0.35, 0.055 * excess), 4)

    @staticmethod
    def competitor_anchor_factor(base_fare: float, competitor_fare: float) -> float:
        """
        Dampened anchoring toward competitor market price (30% elasticity blend).
        """
        if base_fare <= 0:
            return 1.0
        ratio = competitor_fare / base_fare
        clamped_ratio = min(1.30, max(0.75, ratio))
        return round(1.0 + 0.30 * (clamped_ratio - 1.0), 4)

    @staticmethod
    def compute_emsrb_buckets(
        base_fare: float,
        total_seats: int,
        booked_seats: int,
        days_to_departure: int,
    ) -> List[FareBucketAllocation]:
        """
        Belobaba's EMSR-b algorithm across 5 nested fare buckets (Y > B > M > Q > L).
        Determines protection levels theta_j and booking limits BL_j.
        """
        bucket_specs = [
            ("Y", "Full Flex", 1.85, 0.14, 0.35),
            ("B", "Flex Plus", 1.48, 0.18, 0.30),
            ("M", "Standard", 1.18, 0.24, 0.28),
            ("Q", "Saver", 0.96, 0.24, 0.25),
            ("L", "Super Saver", 0.80, 0.20, 0.25),
        ]

        urgency_shift = max(0.5, min(1.6, 1.3 - days_to_departure / 60.0))
        allocations: List[FareBucketAllocation] = []

        fares = [round(base_fare * mult, 2) for _, _, mult, _, _ in bucket_specs]
        means = [total_seats * share * (urgency_shift if i < 2 else 1.0) for i, (_, _, _, share, _) in enumerate(bucket_specs)]
        stds = [max(1.5, m * cv) for m, (_, _, _, _, cv) in zip(means, bucket_specs)]

        protection_levels: List[int] = [total_seats]  # Highest class Y has access to all seats
        # Compute protection levels for nested classes j = 1..k-1
        for j in range(len(bucket_specs) - 1):
            agg_mean = sum(means[: j + 1])
            agg_var = sum(s ** 2 for s in stds[: j + 1])
            agg_std = math.sqrt(agg_var)
            weighted_fare = sum(fares[i] * means[i] for i in range(j + 1)) / max(1e-6, agg_mean)

            next_fare = fares[j + 1]
            prob_threshold = min(0.99, max(0.01, 1.0 - (next_fare / max(1e-6, weighted_fare))))
            theta = int(round(agg_mean + norm.ppf(prob_threshold) * agg_std))
            theta = min(total_seats, max(0, theta))
            protection_levels.append(theta)

        # Booking Limit for class j+1 = Capacity - Protection(j)
        booking_limits: List[int] = [total_seats]
        for j in range(1, len(bucket_specs)):
            bl = max(0, total_seats - protection_levels[j])
            booking_limits.append(bl)

        # Active bucket is the lowest-fare bucket whose booking limit exceeds booked_seats
        active_code = "Y"
        for idx in range(len(bucket_specs) - 1, -1, -1):
            if booked_seats < booking_limits[idx]:
                active_code = bucket_specs[idx][0]
                break

        for idx, (code, name, _, _, _) in enumerate(bucket_specs):
            allocations.append(
                FareBucketAllocation(
                    code=code,
                    name=name,
                    fare=fares[idx],
                    mean_demand=round(means[idx], 1),
                    std_demand=round(stds[idx], 1),
                    protection_level=protection_levels[idx] if idx < len(protection_levels) else 0,
                    booking_limit=booking_limits[idx],
                    is_active=(code == active_code),
                )
            )
        return allocations

    @staticmethod
    def revenue_optimal_price(base_fare: float, elasticity: float, load_factor: float) -> float:
        """
        Computes marginal-cost adjusted optimal fare under iso-elastic demand:
        P* = MC * epsilon / (epsilon - 1), guarded for inelastic regimes (epsilon <= 1.05).
        """
        marginal_cost = base_fare * (0.38 + 0.25 * load_factor)
        effective_eps = max(1.22, elasticity)
        optimal = marginal_cost * (effective_eps / (effective_eps - 1.0))
        return round(max(base_fare * 0.8, min(base_fare * 2.6, optimal)), 2)

    def quote(self, req: PricingQuoteRequest) -> PricingQuoteResponse:
        """Calculate complete dynamic pricing breakdown for a flight state."""
        load_factor = min(1.0, req.booked_seats / max(1, req.total_seats))

        m_time = self.time_urgency_factor(req.days_to_departure)
        m_load = self.load_factor_scarcity(load_factor)
        m_vel = self.velocity_surge_factor(req.booking_velocity_per_hr)
        m_season = SEASON_MULTIPLIERS.get(req.season_mode, 1.0)
        m_comp = self.competitor_anchor_factor(req.base_fare, req.competitor_fare)
        m_cabin = CABIN_MULTIPLIERS.get(req.cabin_class, 1.0)

        # Elasticity dampening: higher price sensitivity moderates aggressive surges
        raw_surge = m_time * m_load * m_vel * m_season * m_comp
        elasticity_dampener = 1.0 / (1.0 + max(0.0, req.elasticity - 1.15) * 0.18)
        adjusted_multiplier = 1.0 + (raw_surge - 1.0) * elasticity_dampener

        price_floor = round(req.base_fare * self.floor_ratio * m_cabin, 2)
        price_ceiling = round(req.base_fare * self.ceiling_ratio * m_cabin, 2)

        unclamped = req.base_fare * adjusted_multiplier * m_cabin
        dynamic_fare = round(min(price_ceiling, max(price_floor, unclamped)), 2)

        buckets = self.compute_emsrb_buckets(
            base_fare=req.base_fare * m_cabin,
            total_seats=req.total_seats,
            booked_seats=req.booked_seats,
            days_to_departure=req.days_to_departure,
        )
        active_bucket = next((b.code for b in buckets if b.is_active), "Y")
        optimal_p = self.revenue_optimal_price(req.base_fare * m_cabin, req.elasticity, load_factor)

        return PricingQuoteResponse(
            base_fare=round(req.base_fare * m_cabin, 2),
            dynamic_fare=dynamic_fare,
            total_multiplier=round(dynamic_fare / max(1e-6, req.base_fare * m_cabin), 4),
            load_factor=round(load_factor, 4),
            active_bucket=active_bucket,
            components={
                "time_urgency": m_time,
                "load_scarcity": m_load,
                "velocity_surge": m_vel,
                "seasonality": m_season,
                "competitor_anchor": m_comp,
                "cabin_multiplier": m_cabin,
                "elasticity_dampener": round(elasticity_dampener, 4),
            },
            emsrb_buckets=buckets,
            price_floor=price_floor,
            price_ceiling=price_ceiling,
            revenue_optimal_price=optimal_p,
        )
