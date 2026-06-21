import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

CONSTITUENTS: Dict[str, float] = {
    "M2": 28.9841042,
    "S2": 30.0000000,
    "K1": 15.0410686,
    "O1": 13.9430356,
    "N2": 28.4397295,
    "P2": 29.9589333,
    "K2": 30.0821373,
    "Q1": 13.3986609,
    "M4": 57.9682084,
    "M6": 86.9523125,
    "S4": 60.0000000,
    "MK3": 44.0251728,
}


def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def normalize_phase(deg: float) -> float:
    while deg >= 360.0:
        deg -= 360.0
    while deg < 0.0:
        deg += 360.0
    return deg


def phase_diff(a_deg: float, b_deg: float) -> float:
    diff = normalize_phase(a_deg - b_deg)
    if diff > 180.0:
        diff -= 360.0
    return diff


@dataclass
class TideConstituent:
    name: str
    amplitude: float
    phase: float

    @property
    def omega_deg_per_hour(self) -> float:
        if self.name not in CONSTITUENTS:
            raise ValueError(f"Unknown constituent: {self.name}")
        return CONSTITUENTS[self.name]

    def value_at_hours(self, hours: float) -> float:
        omega = _deg_to_rad(self.omega_deg_per_hour)
        phase_rad = _deg_to_rad(self.phase)
        return self.amplitude * math.cos(omega * hours - phase_rad)


@dataclass
class TideModel:
    constituents: List[TideConstituent] = field(default_factory=list)
    z0: float = 0.0

    def height_at_hours(self, hours: float) -> float:
        h = self.z0
        for c in self.constituents:
            h += c.value_at_hours(hours)
        return h

    def heights_at_hours(self, hours_list: List[float]) -> List[float]:
        return [self.height_at_hours(t) for t in hours_list]

    def rate_at_hours(self, hours: float, dt: float = 0.01) -> float:
        h1 = self.height_at_hours(hours - dt / 2)
        h2 = self.height_at_hours(hours + dt / 2)
        return (h2 - h1) / dt

    def rates_at_hours(self, hours_list: List[float], dt: float = 0.01) -> List[float]:
        return [self.rate_at_hours(t, dt) for t in hours_list]

    def high_low_waters(self, start_hours: float, end_hours: float, step: float = 0.01) -> List[Tuple[float, float, str]]:
        results = []
        prev_rate = None
        t = start_hours
        while t <= end_hours:
            rate = self.rate_at_hours(t, step)
            if prev_rate is not None:
                if prev_rate > 0 and rate <= 0:
                    h = self.height_at_hours(t - step / 2)
                    results.append((t - step / 2, h, "high"))
                elif prev_rate < 0 and rate >= 0:
                    h = self.height_at_hours(t - step / 2)
                    results.append((t - step / 2, h, "low"))
            prev_rate = rate
            t += step
        return results


def hours_from_datetime(ref_dt: datetime, dt: datetime) -> float:
    delta = dt - ref_dt
    return delta.total_seconds() / 3600.0


def datetime_from_hours(ref_dt: datetime, hours: float) -> datetime:
    return ref_dt + timedelta(hours=hours)


def predict_tide(model: TideModel, times: List[datetime], ref_dt: Optional[datetime] = None) -> List[float]:
    if ref_dt is None:
        ref_dt = times[0]
    hours_list = [hours_from_datetime(ref_dt, t) for t in times]
    return model.heights_at_hours(hours_list)
