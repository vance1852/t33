import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from .core import (
    TideModel,
    hours_from_datetime,
    datetime_from_hours,
)


DANGER_REASONS = {
    "height_too_high": "潮高超过阈值",
    "flood_too_fast": "涨潮速度过快",
    "return_not_safe": "返回时潮高将超过阈值",
    "darkness": "天黑前无法返回",
    "near_danger": "接近危险边界，安全余量不足",
}


@dataclass
class WindowConfig:
    height_threshold: float = 1.5
    max_flood_rate: float = 0.15
    offshore_distance_km: float = 1.0
    walking_speed_kmh: float = 3.0
    safety_margin_height: float = 0.3
    safety_margin_time_minutes: float = 30.0
    sunset_dt: Optional[datetime] = None
    sunrise_dt: Optional[datetime] = None
    time_step_minutes: float = 10.0
    require_daylight: bool = True

    @property
    def return_time_hours(self) -> float:
        return self.offshore_distance_km / self.walking_speed_kmh

    @property
    def safe_height_with_margin(self) -> float:
        return self.height_threshold - self.safety_margin_height


@dataclass
class PointSafety:
    dt: datetime
    is_safe: bool
    height: float
    rate: float
    return_height: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class SafeWindow:
    start_dt: datetime
    end_dt: datetime
    min_height: float
    min_height_dt: datetime
    max_return_height: float
    duration_minutes: float
    reasons_near_start: List[str] = field(default_factory=list)
    reasons_near_end: List[str] = field(default_factory=list)


def _compute_return_height(
    model: TideModel,
    start_hours: float,
    return_time_hours: float,
    step_hours: float = 0.01,
) -> float:
    """
    计算从 start_hours 开始，返回时间内的最大潮高。
    """
    max_h = -float("inf")
    t = start_hours
    end_t = start_hours + return_time_hours
    while t <= end_t + 1e-9:
        h = model.height_at_hours(t)
        if h > max_h:
            max_h = h
        t += step_hours
    return max_h


def _classify_point(
    model: TideModel,
    t_hours: float,
    dt: datetime,
    config: WindowConfig,
) -> PointSafety:
    height = model.height_at_hours(t_hours)
    rate = model.rate_at_hours(t_hours)

    reasons = []
    is_safe = True

    if height > config.safe_height_with_margin:
        is_safe = False
        reasons.append("height_too_high")

    if rate > config.max_flood_rate:
        is_safe = False
        reasons.append("flood_too_fast")

    return_h = _compute_return_height(model, t_hours, config.return_time_hours)
    if return_h > config.safe_height_with_margin:
        is_safe = False
        reasons.append("return_not_safe")

    if config.require_daylight and config.sunset_dt is not None:
        return_end_dt = dt + timedelta(hours=config.return_time_hours) + timedelta(minutes=config.safety_margin_time_minutes)
        if return_end_dt > config.sunset_dt:
            is_safe = False
            reasons.append("darkness")

    if config.require_daylight and config.sunrise_dt is not None:
        if dt < config.sunrise_dt:
            is_safe = False
            reasons.append("darkness")

    return PointSafety(
        dt=dt,
        is_safe=is_safe,
        height=height,
        rate=rate,
        return_height=return_h,
        reasons=reasons,
    )


def find_safe_windows(
    model: TideModel,
    start_dt: datetime,
    end_dt: datetime,
    config: WindowConfig,
    ref_dt: Optional[datetime] = None,
) -> Tuple[List[SafeWindow], List[PointSafety]]:
    """
    查找给定时间范围内的安全赶海窗口。

    返回：(窗口列表, 逐点安全评估列表)
    """
    if ref_dt is None:
        ref_dt = start_dt

    points: List[PointSafety] = []
    step = timedelta(minutes=config.time_step_minutes)
    current_dt = start_dt

    while current_dt <= end_dt + timedelta(seconds=1):
        t_hours = hours_from_datetime(ref_dt, current_dt)
        point = _classify_point(model, t_hours, current_dt, config)
        points.append(point)
        current_dt += step

    windows: List[SafeWindow] = []
    i = 0
    n = len(points)

    while i < n:
        if points[i].is_safe:
            start_idx = i
            while i < n and points[i].is_safe:
                i += 1
            end_idx = i - 1

            win_points = points[start_idx : end_idx + 1]

            min_h = float("inf")
            min_h_dt = win_points[0].dt
            max_return_h = -float("inf")

            for p in win_points:
                if p.height < min_h:
                    min_h = p.height
                    min_h_dt = p.dt
                if p.return_height > max_return_h:
                    max_return_h = p.return_height

            start_reasons = []
            if start_idx > 0:
                start_reasons = points[start_idx - 1].reasons[:]
            end_reasons = []
            if end_idx < n - 1:
                end_reasons = points[end_idx + 1].reasons[:]

            duration = (win_points[-1].dt - win_points[0].dt).total_seconds() / 60.0

            windows.append(SafeWindow(
                start_dt=win_points[0].dt,
                end_dt=win_points[-1].dt,
                min_height=min_h,
                min_height_dt=min_h_dt,
                max_return_height=max_return_h,
                duration_minutes=duration,
                reasons_near_start=start_reasons,
                reasons_near_end=end_reasons,
            ))
        else:
            i += 1

    return windows, points


def describe_danger_reason(reason_key: str) -> str:
    return DANGER_REASONS.get(reason_key, reason_key)
