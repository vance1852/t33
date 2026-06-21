import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime

from .core import (
    TideModel,
    TideConstituent,
    hours_from_datetime,
    normalize_phase,
    CONSTITUENTS,
)


@dataclass
class FitResult:
    model: TideModel
    residuals: List[float]
    rmse: float
    mae: float
    max_abs_residual: float
    n_points: int
    n_skipped: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def residual_stats(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "max_abs": self.max_abs_residual,
            "n_points": self.n_points,
            "n_skipped": self.n_skipped,
        }


def _is_valid_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return False
    return True


def _build_design_matrix(hours_list: List[float], constituent_names: List[str]) -> List[List[float]]:
    """
    构建设计矩阵 X。每行对应一个时刻，列结构：
    [1, cos(ω1*t), sin(ω1*t), cos(ω2*t), sin(ω2*t), ...]
    """
    n_rows = len(hours_list)
    n_cols = 1 + 2 * len(constituent_names)
    X = [[0.0] * n_cols for _ in range(n_rows)]

    for i, t in enumerate(hours_list):
        X[i][0] = 1.0
        for j, name in enumerate(constituent_names):
            omega = CONSTITUENTS[name] * math.pi / 180.0
            col_a = 1 + 2 * j
            col_b = 2 + 2 * j
            X[i][col_a] = math.cos(omega * t)
            X[i][col_b] = math.sin(omega * t)

    return X


def _lstsq(X: List[List[float]], y: List[float]) -> List[float]:
    """
    用 numpy 解最小二乘，如果没有 numpy 则回退到简化版。
    为了保证健壮性，我们用纯 Python 实现了高斯消元法。
    """
    try:
        import numpy as np

        X_np = np.array(X, dtype=float)
        y_np = np.array(y, dtype=float)
        coeffs, _, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
        return coeffs.tolist()
    except ImportError:
        return _lstsq_python(X, y)


def _lstsq_python(X: List[List[float]], y: List[float]) -> List[float]:
    """
    纯 Python 实现最小二乘（正规方程法 + 高斯消元）。
    解 X^T X x = X^T y
    """
    n_rows = len(X)
    n_cols = len(X[0])

    XtX = [[0.0] * n_cols for _ in range(n_cols)]
    Xty = [0.0] * n_cols

    for i in range(n_cols):
        for j in range(n_cols):
            s = 0.0
            for k in range(n_rows):
                s += X[k][i] * X[k][j]
            XtX[i][j] = s

    for i in range(n_cols):
        s = 0.0
        for k in range(n_rows):
            s += X[k][i] * y[k]
        Xty[i] = s

    aug = []
    for i in range(n_cols):
        row = XtX[i][:] + [Xty[i]]
        aug.append(row)

    for i in range(n_cols):
        max_row = i
        max_val = abs(aug[i][i])
        for k in range(i + 1, n_cols):
            if abs(aug[k][i]) > max_val:
                max_val = abs(aug[k][i])
                max_row = k
        if max_row != i:
            aug[i], aug[max_row] = aug[max_row], aug[i]

        pivot = aug[i][i]
        if abs(pivot) < 1e-10:
            raise ValueError("Singular matrix in least squares")

        for j in range(i, n_cols + 1):
            aug[i][j] /= pivot

        for k in range(n_cols):
            if k != i and abs(aug[k][i]) > 1e-15:
                factor = aug[k][i]
                for j in range(i, n_cols + 1):
                    aug[k][j] -= factor * aug[i][j]

    x = [row[n_cols] for row in aug]
    return x


def fit_tide(
    times: List[datetime],
    heights: List[float],
    constituent_names: Optional[List[str]] = None,
    ref_dt: Optional[datetime] = None,
) -> FitResult:
    """
    用最小二乘法从实测潮高数据中拟合分潮参数。

    支持：不等间隔时间序列、缺测数据（自动跳过 None/NaN/inf）、
          相位归一化到 [0, 360)
    """
    if len(times) != len(heights):
        raise ValueError("times and heights must have same length")

    if constituent_names is None:
        constituent_names = ["M2", "S2", "K1", "O1"]

    if ref_dt is None:
        ref_dt = times[0]

    warn_msgs = []

    valid_indices = []
    for i in range(len(times)):
        if _is_valid_value(heights[i]):
            valid_indices.append(i)
        else:
            if not warn_msgs:
                warn_msgs.append("观测数据包含缺测值，已自动跳过")

    n_skipped = len(times) - len(valid_indices)
    n_points = len(valid_indices)

    if n_points < 2 * len(constituent_names) + 1:
        warnings.warn(
            f"有效观测点({n_points})可能不足以拟合 {len(constituent_names)} 个分潮"
        )
        warn_msgs.append(f"有效观测点较少({n_points})，拟合结果可能不稳定")

    valid_times = [times[i] for i in valid_indices]
    valid_heights = [float(heights[i]) for i in valid_indices]

    hours_list = [hours_from_datetime(ref_dt, t) for t in valid_times]

    X = _build_design_matrix(hours_list, constituent_names)

    coeffs = _lstsq(X, valid_heights)

    z0 = coeffs[0]

    constituents = []
    for j, name in enumerate(constituent_names):
        a = coeffs[1 + 2 * j]
        b = coeffs[2 + 2 * j]
        amplitude = math.sqrt(a * a + b * b)
        phase = math.degrees(math.atan2(b, a))
        phase = normalize_phase(phase)

        if abs(phase - 0.0) < 5.0 or abs(phase - 360.0) < 5.0:
            warn_msgs.append(f"{name}分潮相位接近0度，已归一化")

        constituents.append(TideConstituent(
            name=name,
            amplitude=amplitude,
            phase=phase,
        ))

    model = TideModel(constituents=constituents, z0=z0)

    residuals = []
    for i in range(n_points):
        t_hours = hours_list[i]
        pred = model.height_at_hours(t_hours)
        residuals.append(valid_heights[i] - pred)

    if residuals:
        ss = sum(r * r for r in residuals)
        rmse = math.sqrt(ss / len(residuals))
        mae = sum(abs(r) for r in residuals) / len(residuals)
        max_abs = max(abs(r) for r in residuals)
    else:
        rmse = 0.0
        mae = 0.0
        max_abs = 0.0

    return FitResult(
        model=model,
        residuals=residuals,
        rmse=rmse,
        mae=mae,
        max_abs_residual=max_abs,
        n_points=n_points,
        n_skipped=n_skipped,
        warnings=warn_msgs,
    )
