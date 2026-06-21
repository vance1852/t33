import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

from .core import (
    TideModel,
    TideConstituent,
    predict_tide,
    hours_from_datetime,
    datetime_from_hours,
)


@dataclass
class HarborInfo:
    name: str
    model: TideModel
    description: str = ""
    has_defect: bool = False
    defect_description: str = ""


def _build_harbor_model(amps: Dict[str, float], phases: Dict[str, float], z0: float = 0.0) -> TideModel:
    constituents = []
    for name, amp in amps.items():
        phase = phases.get(name, 0.0)
        constituents.append(TideConstituent(name=name, amplitude=amp, phase=phase))
    return TideModel(constituents=constituents, z0=z0)


def get_sample_harbors() -> Dict[str, HarborInfo]:
    harbors = {}

    harbors["qingdao"] = HarborInfo(
        name="青岛湾",
        model=_build_harbor_model(
            amps={"M2": 1.25, "S2": 0.42, "K1": 0.38, "O1": 0.32, "N2": 0.28},
            phases={"M2": 45.2, "S2": 120.5, "K1": 88.0, "O1": 65.3, "N2": 32.1},
            z0=2.1,
        ),
        description="正规半日潮港，潮差中等，适合赶海",
    )

    harbors["sanya"] = HarborInfo(
        name="三亚湾",
        model=_build_harbor_model(
            amps={"K1": 0.52, "O1": 0.48, "M2": 0.35, "S2": 0.18},
            phases={"K1": 210.0, "O1": 180.0, "M2": 95.0, "S2": 170.0},
            z0=1.5,
        ),
        description="不正规日潮港，日潮特征明显",
    )

    harbors["beihai"] = HarborInfo(
        name="北海银滩",
        model=_build_harbor_model(
            amps={"M2": 1.85, "S2": 0.62, "K1": 0.25, "O1": 0.22},
            phases={"M2": 350.0, "S2": 75.0, "K1": 45.0, "O1": 25.0},
            z0=2.8,
        ),
        description="正规半日潮港，潮差较大，相位接近0度",
        has_defect=True,
        defect_description="M2相位接近0度，拟合时需注意相位归一化",
    )

    harbors["xiangzhou"] = HarborInfo(
        name="香洲港（缺测版）",
        model=_build_harbor_model(
            amps={"M2": 0.95, "S2": 0.30, "K1": 0.28, "O1": 0.24},
            phases={"M2": 150.0, "S2": 220.0, "K1": 110.0, "O1": 90.0},
            z0=1.8,
        ),
        description="不正规半日潮港，数据缺测较多",
        has_defect=True,
        defect_description="实测数据缺测点多，时间不等间隔",
    )

    harbors["donghai"] = HarborInfo(
        name="东海岛（少分潮）",
        model=_build_harbor_model(
            amps={"M2": 1.50, "S2": 0.50, "K1": 0.30},
            phases={"M2": 60.0, "S2": 140.0, "K1": 70.0},
            z0=2.0,
        ),
        description="缺O1分潮的特殊港湾",
        has_defect=True,
        defect_description="缺少O1分潮，仅M2+S2+K1三个主要分潮",
    )

    return harbors


def generate_synthetic_observations(
    model: TideModel,
    start_dt: datetime,
    duration_hours: float,
    interval_minutes: float = 60.0,
    noise_std: float = 0.05,
    seed: Optional[int] = 42,
) -> Tuple[List[datetime], List[float]]:
    if seed is not None:
        random.seed(seed)

    times = []
    heights = []
    n_steps = int(duration_hours * 60.0 / interval_minutes)
    for i in range(n_steps + 1):
        dt = start_dt + timedelta(minutes=i * interval_minutes)
        h = model.height_at_hours(i * interval_minutes / 60.0)
        noise = random.gauss(0, noise_std)
        times.append(dt)
        heights.append(h + noise)
    return times, heights


def generate_irregular_observations(
    model: TideModel,
    start_dt: datetime,
    duration_hours: float,
    base_interval_minutes: float = 60.0,
    irregular_ratio: float = 0.3,
    missing_ratio: float = 0.15,
    noise_std: float = 0.08,
    seed: Optional[int] = 42,
) -> Tuple[List[datetime], List[float]]:
    if seed is not None:
        random.seed(seed)

    times = []
    heights = []
    t = 0.0
    while t <= duration_hours:
        dt = start_dt + timedelta(hours=t)
        if random.random() < missing_ratio:
            pass
        else:
            h = model.height_at_hours(t)
            noise = random.gauss(0, noise_std)
            times.append(dt)
            heights.append(h + noise)

        jitter = random.uniform(-base_interval_minutes * irregular_ratio / 60.0,
                                 base_interval_minutes * irregular_ratio / 60.0)
        t += base_interval_minutes / 60.0 + jitter

    return times, heights


def get_default_reference_time() -> datetime:
    return datetime(2026, 6, 21, 0, 0, 0)


def get_sample_observation_set(harbor_key: str) -> Tuple[List[datetime], List[float]]:
    harbors = get_sample_harbors()
    if harbor_key not in harbors:
        raise ValueError(f"Unknown harbor: {harbor_key}")

    ref_dt = get_default_reference_time()
    harbor = harbors[harbor_key]

    if harbor_key == "xiangzhou":
        return generate_irregular_observations(
            harbor.model,
            ref_dt,
            duration_hours=48.0,
            base_interval_minutes=60.0,
            irregular_ratio=0.4,
            missing_ratio=0.2,
            noise_std=0.06,
            seed=123,
        )
    else:
        return generate_synthetic_observations(
            harbor.model,
            ref_dt,
            duration_hours=36.0,
            interval_minutes=30.0,
            noise_std=0.05,
            seed=harbor_key.__hash__() % 10000,
        )


def get_tomorrow_sun_times(ref_dt: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    if ref_dt is None:
        ref_dt = get_default_reference_time()
    tomorrow = ref_dt + timedelta(days=1)
    sunrise = tomorrow.replace(hour=5, minute=30)
    sunset = tomorrow.replace(hour=19, minute=15)
    return sunrise, sunset
