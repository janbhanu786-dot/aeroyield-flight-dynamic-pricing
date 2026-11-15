"""
Non-Homogeneous Poisson Process (NHPP) Monte Carlo Booking Simulator.

Compares Static Fixed Pricing vs. Dynamic EMSR-b Pricing across N simulated
flight booking horizons to quantify Yield Lift %, Load Factor, and Spoilage.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np

from src.models import MonteCarloRequest, PricingQuoteRequest
from src.pricing_engine import DynamicPricingEngine


def run_monte_carlo_comparison(req: MonteCarloRequest) -> Dict[str, Any]:
    """Run N paired Monte Carlo booking horizons (Static vs. Dynamic)."""
    rng = np.random.default_rng(req.seed)
    engine = DynamicPricingEngine()

    static_revenues: List[float] = []
    dynamic_revenues: List[float] = []
    static_lfs: List[float] = []
    dynamic_lfs: List[float] = []

    sample_daily_curve: List[Dict[str, Any]] = []

    for sim_idx in range(req.num_simulations):
        static_booked = 0
        dynamic_booked = 0
        static_rev = 0.0
        dynamic_rev = 0.0

        for day_idx in range(req.horizon_days):
            days_left = req.horizon_days - 1 - day_idx
            # Arrival intensity increases as departure nears (NHPP)
            intensity = req.mean_daily_arrivals * (0.65 + 1.15 * math.exp(-0.07 * days_left))
            arrivals = int(rng.poisson(intensity))

            for _ in range(arrivals):
                # Draw customer willingness-to-pay (WTP) from log-normal distribution
                wtp_mult = float(rng.lognormal(mean=0.12 + 0.25 * math.exp(-0.09 * days_left), sigma=0.34))
                customer_wtp = req.base_fare * wtp_mult

                # 1. Static Pricing decision
                if static_booked < req.capacity and customer_wtp >= req.base_fare:
                    static_booked += 1
                    static_rev += req.base_fare

                # 2. Dynamic Pricing decision
                if dynamic_booked < req.capacity:
                    quote = engine.quote(
                        PricingQuoteRequest(
                            base_fare=req.base_fare,
                            days_to_departure=days_left,
                            total_seats=req.capacity,
                            booked_seats=dynamic_booked,
                            booking_velocity_per_hr=intensity / 8.0,
                            elasticity=req.elasticity,
                            competitor_fare=req.base_fare * 1.03,
                            season_mode="regular",
                            cabin_class="Economy",
                        )
                    )
                    # Purchase probability adjusted by price elasticity
                    rel_price = quote.dynamic_fare / max(1.0, req.base_fare)
                    accept_prob = min(1.0, (rel_price ** (-req.elasticity * 0.45)))
                    if customer_wtp >= quote.dynamic_fare and rng.random() <= accept_prob:
                        dynamic_booked += 1
                        dynamic_rev += quote.dynamic_fare

            if sim_idx == 0:
                sample_daily_curve.append(
                    {
                        "day": day_idx + 1,
                        "days_to_departure": days_left,
                        "static_revenue": round(static_rev, 2),
                        "dynamic_revenue": round(dynamic_rev, 2),
                        "static_load_factor": round(static_booked / req.capacity, 3),
                        "dynamic_load_factor": round(dynamic_booked / req.capacity, 3),
                    }
                )

        static_revenues.append(static_rev)
        dynamic_revenues.append(dynamic_rev)
        static_lfs.append(static_booked / req.capacity)
        dynamic_lfs.append(dynamic_booked / req.capacity)

    avg_static = float(np.mean(static_revenues))
    avg_dynamic = float(np.mean(dynamic_revenues))
    yield_lift_pct = ((avg_dynamic - avg_static) / max(1.0, avg_static)) * 100.0

    return {
        "simulations": req.num_simulations,
        "avg_static_revenue": round(avg_static, 2),
        "avg_dynamic_revenue": round(avg_dynamic, 2),
        "yield_lift_pct": round(yield_lift_pct, 2),
        "avg_static_load_factor": round(float(np.mean(static_lfs)) * 100.0, 2),
        "avg_dynamic_load_factor": round(float(np.mean(dynamic_lfs)) * 100.0, 2),
        "spoilage_static_seats": round((1.0 - float(np.mean(static_lfs))) * req.capacity, 1),
        "spoilage_dynamic_seats": round((1.0 - float(np.mean(dynamic_lfs))) * req.capacity, 1),
        "daily_trajectory": sample_daily_curve,
    }
