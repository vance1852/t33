"""测试集合
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import math
from datetime import datetime, timedelta

from tide_tool.core import (
    TideModel,
    TideConstituent,
    normalize_phase,
    phase_diff,
    CONSTITUENTS,
    hours_from_datetime,
)
from tide_tool.fitting import fit_tide, FitResult
from tide_tool.window import find_safe_windows, WindowConfig, describe_danger_reason
from tide_tool.samples import (
    get_sample_harbors,
    get_sample_observation_set,
    generate_synthetic_observations,
    generate_irregular_observations,
    get_default_reference_time,
    get_tomorrow_sun_times,
)
from tide_tool.report import (
    export_tide_predictions_csv,
    export_windows_csv,
    export_points_csv,
    export_fit_result_csv,
    export_markdown_report,
)


class TestCore(unittest.TestCase):
    def test_normalize_phase(self):
        self.assertAlmostEqual(normalize_phase(0.0), 0.0)
        self.assertAlmostEqual(normalize_phase(360.0), 0.0)
        self.assertAlmostEqual(normalize_phase(720.0), 0.0)
        self.assertAlmostEqual(normalize_phase(-90.0), 270.0)
        self.assertAlmostEqual(normalize_phase(450.0), 90.0)
        self.assertAlmostEqual(normalize_phase(359.9), 359.9)

    def test_phase_diff(self):
        self.assertAlmostEqual(phase_diff(10.0, 0.0), 10.0)
        self.assertAlmostEqual(phase_diff(0.0, 10.0), -10.0)
        self.assertAlmostEqual(phase_diff(350.0, 10.0), -20.0, places=5)
        self.assertAlmostEqual(phase_diff(10.0, 350.0), 20.0, places=5)

    def test_tide_constituent(self):
        c = TideConstituent(name="M2", amplitude=1.0, phase=0.0)
        self.assertAlmostEqual(c.omega_deg_per_hour, 28.9841042)
        h = c.value_at_hours(0.0)
        self.assertAlmostEqual(h, 1.0)

    def test_tide_model(self):
        model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.0, phase=0.0),
            ],
            z0=2.0,
        )
        h0 = model.height_at_hours(0.0)
        self.assertAlmostEqual(h0, 3.0)

        period_hours = 360.0 / CONSTITUENTS["M2"]
        h_half = model.height_at_hours(period_hours / 2.0)
        self.assertAlmostEqual(h_half, 1.0)

    def test_high_low_waters(self):
        model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.0, phase=0.0),
            ],
            z0=2.0,
        )
        extremes = model.high_low_waters(0.0, 25.0, step=0.01)
        self.assertTrue(len(extremes) >= 2)
        highs = [e for e in extremes if e[2] == "high"]
        lows = [e for e in extremes if e[2] == "low"]
        self.assertTrue(len(highs) >= 1)
        self.assertTrue(len(lows) >= 1)


class TestFitting(unittest.TestCase):
    def test_synthetic_fit_perfect(self):
        """合成潮汐拟合回归测试：无噪声时应能几乎完美恢复参数"""
        true_model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.2, phase=45.0),
                TideConstituent(name="S2", amplitude=0.4, phase=120.0),
                TideConstituent(name="K1", amplitude=0.3, phase=80.0),
                TideConstituent(name="O1", amplitude=0.25, phase=60.0),
            ],
            z0=2.0,
        )

        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        duration_hours = 48.0
        interval_minutes = 30.0
        n = int(duration_hours * 60 / interval_minutes) + 1
        times = [ref_dt + timedelta(minutes=i * interval_minutes) for i in range(n)]
        heights = [true_model.height_at_hours(i * interval_minutes / 60.0) for i in range(n)]

        result = fit_tide(times, heights, constituent_names=["M2", "S2", "K1", "O1"], ref_dt=ref_dt)

        self.assertLess(result.rmse, 1e-6)
        self.assertLess(result.mae, 1e-6)

        fitted = {c.name: c for c in result.model.constituents}
        true_constits = {c.name: c for c in true_model.constituents}

        for name in ["M2", "S2", "K1", "O1"]:
            self.assertAlmostEqual(fitted[name].amplitude, true_constits[name].amplitude, places=4)
            pd = abs(phase_diff(fitted[name].phase, true_constits[name].phase))
            self.assertLess(pd, 0.01)

        self.assertAlmostEqual(result.model.z0, true_model.z0, places=4)

    def test_phase_near_zero(self):
        """相位跨0点测试：M2相位接近0度，拟合后应能正确归一化"""
        true_model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.5, phase=2.0),
                TideConstituent(name="S2", amplitude=0.5, phase=355.0),
            ],
            z0=2.5,
        )

        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        times = [ref_dt + timedelta(minutes=i * 20) for i in range(200)]
        heights = [true_model.height_at_hours(i * 20 / 60.0) for i in range(200)]

        result = fit_tide(times, heights, constituent_names=["M2", "S2"], ref_dt=ref_dt)

        fitted = {c.name: c for c in result.model.constituents}
        true_constits = {c.name: c for c in true_model.constituents}

        for name in ["M2", "S2"]:
            self.assertAlmostEqual(fitted[name].amplitude, true_constits[name].amplitude, places=3)
            pd = abs(phase_diff(fitted[name].phase, true_constits[name].phase))
            self.assertLess(pd, 1.0,
                          f"{name}: fitted={fitted[name].phase:.2f}, true={true_constits[name].phase:.2f}, diff={pd:.2f}")

            self.assertGreaterEqual(fitted[name].phase, 0.0)
            self.assertLess(fitted[name].phase, 360.0)

    def test_missing_data_warning(self):
        """缺测数据仍可拟合并给出警告"""
        true_model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.0, phase=30.0),
                TideConstituent(name="S2", amplitude=0.3, phase=90.0),
            ],
            z0=1.5,
        )

        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        times = []
        heights = []
        for i in range(150):
            t = ref_dt + timedelta(minutes=i * 20)
            h = true_model.height_at_hours(i * 20 / 60.0)
            if i % 5 == 0:
                heights.append(None)
            else:
                heights.append(h)
            times.append(t)

        result = fit_tide(times, heights, constituent_names=["M2", "S2"], ref_dt=ref_dt)

        self.assertGreater(result.n_skipped, 0)
        self.assertGreater(result.n_points, 0)
        self.assertTrue(any("缺测" in w for w in result.warnings))

        fitted = {c.name: c for c in result.model.constituents}
        true_constits = {c.name: c for c in true_model.constituents}
        for name in ["M2", "S2"]:
            self.assertAlmostEqual(fitted[name].amplitude, true_constits[name].amplitude, places=2)

    def test_irregular_times(self):
        """不等间隔时间序列拟合"""
        true_model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.0, phase=45.0),
                TideConstituent(name="K1", amplitude=0.3, phase=60.0),
            ],
            z0=2.0,
        )

        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        times = [
            ref_dt + timedelta(minutes=0),
            ref_dt + timedelta(minutes=45),
            ref_dt + timedelta(minutes=120),
            ref_dt + timedelta(minutes=200),
            ref_dt + timedelta(minutes=350),
            ref_dt + timedelta(minutes=500),
            ref_dt + timedelta(minutes=800),
            ref_dt + timedelta(minutes=1200),
            ref_dt + timedelta(minutes=1500),
            ref_dt + timedelta(minutes=2000),
            ref_dt + timedelta(minutes=2400),
        ]
        heights = []
        for t in times:
            hours = (t - ref_dt).total_seconds() / 3600.0
            heights.append(true_model.height_at_hours(hours))

        result = fit_tide(times, heights, constituent_names=["M2", "K1"], ref_dt=ref_dt)

        self.assertEqual(result.n_points, len(times))
        self.assertEqual(result.n_skipped, 0)
        self.assertLess(result.rmse, 1e-6)


class TestWindow(unittest.TestCase):
    def test_safe_windows_basic(self):
        """基本安全窗口检测"""
        model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.5, phase=0.0),
            ],
            z0=2.0,
        )

        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        config = WindowConfig(
            height_threshold=1.5,
            max_flood_rate=1.0,
            offshore_distance_km=0.5,
            walking_speed_kmh=5.0,
            safety_margin_height=0.1,
            safety_margin_time_minutes=10.0,
            time_step_minutes=5.0,
            require_daylight=False,
        )

        start_dt = ref_dt + timedelta(hours=6)
        end_dt = ref_dt + timedelta(hours=18)

        windows, points = find_safe_windows(model, start_dt, end_dt, config, ref_dt=ref_dt)

        self.assertGreater(len(windows), 0)
        self.assertGreater(len(points), 0)

        for w in windows:
            self.assertLessEqual(w.min_height, config.safe_height_with_margin)

    def test_no_safe_windows(self):
        """阈值设置过低导致无安全窗口"""
        model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.0, phase=0.0),
            ],
            z0=3.0,
        )

        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        config = WindowConfig(
            height_threshold=0.5,
            max_flood_rate=1.0,
            offshore_distance_km=0.1,
            walking_speed_kmh=10.0,
            safety_margin_height=0.0,
            safety_margin_time_minutes=0.0,
            time_step_minutes=10.0,
            require_daylight=False,
        )

        windows, points = find_safe_windows(
            model, ref_dt, ref_dt + timedelta(hours=24), config, ref_dt=ref_dt
        )

        self.assertEqual(len(windows), 0)
        all_unsafe = all(not p.is_safe for p in points)
        self.assertTrue(all_unsafe)

    def test_flood_rate_limit(self):
        """涨潮速度限制测试：高潮后转落潮前，即使潮高仍低，涨潮过快也应判为危险"""
        model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=2.0, phase=0.0),
            ],
            z0=2.0,
        )

        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        config_low_rate = WindowConfig(
            height_threshold=2.5,
            max_flood_rate=10.0,
            offshore_distance_km=0.1,
            walking_speed_kmh=100.0,
            safety_margin_height=0.0,
            safety_margin_time_minutes=0.0,
            time_step_minutes=5.0,
            require_daylight=False,
        )
        config_high_rate = WindowConfig(
            height_threshold=2.5,
            max_flood_rate=0.001,
            offshore_distance_km=0.1,
            walking_speed_kmh=100.0,
            safety_margin_height=0.0,
            safety_margin_time_minutes=0.0,
            time_step_minutes=5.0,
            require_daylight=False,
        )

        start_dt = ref_dt + timedelta(hours=3)
        end_dt = ref_dt + timedelta(hours=9)

        win_low, _ = find_safe_windows(model, start_dt, end_dt, config_low_rate, ref_dt=ref_dt)
        win_high, _ = find_safe_windows(model, start_dt, end_dt, config_high_rate, ref_dt=ref_dt)

        total_low = sum(w.duration_minutes for w in win_low)
        total_high = sum(w.duration_minutes for w in win_high)
        self.assertGreater(total_low, total_high)

        _, points_high = find_safe_windows(model, start_dt, end_dt, config_high_rate, ref_dt=ref_dt)
        flood_danger_points = [p for p in points_high if "flood_too_fast" in p.reasons]
        self.assertGreater(len(flood_danger_points), 0)

    def test_darkness_limit(self):
        """天黑限制测试"""
        model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.0, phase=0.0),
            ],
            z0=2.0,
        )

        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        sunset = ref_dt.replace(hour=18, minute=0)
        sunrise = ref_dt.replace(hour=6, minute=0)

        config = WindowConfig(
            height_threshold=3.0,
            max_flood_rate=10.0,
            offshore_distance_km=0.1,
            walking_speed_kmh=100.0,
            safety_margin_height=0.0,
            safety_margin_time_minutes=0.0,
            sunset_dt=sunset,
            sunrise_dt=sunrise,
            time_step_minutes=10.0,
            require_daylight=True,
        )

        windows, points = find_safe_windows(
            model, ref_dt, ref_dt + timedelta(hours=24), config, ref_dt=ref_dt
        )

        for w in windows:
            self.assertGreaterEqual(w.start_dt, sunrise - timedelta(minutes=30))
            self.assertLessEqual(w.end_dt, sunset + timedelta(minutes=30))

    def test_low_tide_danger_explanation(self):
        """低潮附近仍然危险的情况说明：返回时间内潮水会涨上来"""
        model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=2.0, phase=0.0),
            ],
            z0=2.0,
        )

        ref_dt = datetime(2026, 6, 21, 0, 0, 0)

        config_far = WindowConfig(
            height_threshold=2.5,
            max_flood_rate=10.0,
            offshore_distance_km=3.0,
            walking_speed_kmh=1.0,
            safety_margin_height=0.0,
            safety_margin_time_minutes=0.0,
            time_step_minutes=5.0,
            require_daylight=False,
        )

        config_near = WindowConfig(
            height_threshold=2.5,
            max_flood_rate=10.0,
            offshore_distance_km=0.1,
            walking_speed_kmh=10.0,
            safety_margin_height=0.0,
            safety_margin_time_minutes=0.0,
            time_step_minutes=5.0,
            require_daylight=False,
        )

        low_tide_hours = 180.0 / CONSTITUENTS["M2"]
        start_dt = ref_dt + timedelta(hours=low_tide_hours - 3)
        end_dt = ref_dt + timedelta(hours=low_tide_hours + 3)

        win_far, pts_far = find_safe_windows(model, start_dt, end_dt, config_far, ref_dt=ref_dt)
        win_near, pts_near = find_safe_windows(model, start_dt, end_dt, config_near, ref_dt=ref_dt)

        total_far = sum(w.duration_minutes for w in win_far)
        total_near = sum(w.duration_minutes for w in win_near)
        self.assertGreater(total_near, total_far)

        return_danger = [p for p in pts_far if "return_not_safe" in p.reasons]
        self.assertGreater(len(return_danger), 0)


class TestSamples(unittest.TestCase):
    def test_get_sample_harbors(self):
        harbors = get_sample_harbors()
        self.assertGreaterEqual(len(harbors), 3)
        for key, harbor in harbors.items():
            self.assertTrue(len(harbor.model.constituents) > 0)

    def test_harbor_with_missing_constituent(self):
        """某个港口少一个分潮"""
        harbors = get_sample_harbors()
        donghai = harbors.get("donghai")
        self.assertIsNotNone(donghai)
        names = [c.name for c in donghai.model.constituents]
        self.assertIn("M2", names)
        self.assertIn("S2", names)
        self.assertIn("K1", names)
        self.assertNotIn("O1", names)
        self.assertTrue(donghai.has_defect)

    def test_irregular_observations(self):
        """不等间隔+缺测的观测数据"""
        model = TideModel(
            constituents=[TideConstituent(name="M2", amplitude=1.0, phase=0.0)],
            z0=2.0,
        )
        ref_dt = datetime(2026, 6, 21)
        times, heights = generate_irregular_observations(
            model, ref_dt, duration_hours=24.0, missing_ratio=0.3, seed=42
        )
        self.assertGreater(len(times), 0)
        self.assertEqual(len(times), len(heights))

        intervals = []
        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]).total_seconds() / 60.0
            intervals.append(dt)
        interval_variance = max(intervals) - min(intervals)
        self.assertGreater(interval_variance, 0.01)

    def test_all_harbor_observations(self):
        for key in get_sample_harbors().keys():
            times, heights = get_sample_observation_set(key)
            self.assertGreater(len(times), 10)
            self.assertEqual(len(times), len(heights))


class TestReport(unittest.TestCase):
    def test_csv_exports(self):
        model = TideModel(
            constituents=[TideConstituent(name="M2", amplitude=1.0, phase=0.0)],
            z0=2.0,
        )
        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        times = [ref_dt + timedelta(hours=i) for i in range(10)]

        csv_pred = export_tide_predictions_csv(model, times, ref_dt=ref_dt)
        self.assertIn("datetime", csv_pred)
        self.assertIn("height_m", csv_pred)
        self.assertGreater(len(csv_pred.splitlines()), 5)

    def test_window_csv(self):
        model = TideModel(
            constituents=[TideConstituent(name="M2", amplitude=1.5, phase=0.0)],
            z0=2.0,
        )
        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        config = WindowConfig(
            height_threshold=1.5,
            max_flood_rate=10.0,
            offshore_distance_km=0.1,
            walking_speed_kmh=10.0,
            safety_margin_height=0.1,
            safety_margin_time_minutes=5.0,
            time_step_minutes=10.0,
            require_daylight=False,
        )
        windows, points = find_safe_windows(
            model, ref_dt + timedelta(hours=6), ref_dt + timedelta(hours=18),
            config, ref_dt=ref_dt
        )

        csv_win = export_windows_csv(windows)
        self.assertIn("window_index", csv_win)

        csv_pts = export_points_csv(points)
        self.assertIn("is_safe", csv_pts)
        self.assertIn("danger_reasons", csv_pts)

    def test_markdown_report(self):
        model = TideModel(
            constituents=[TideConstituent(name="M2", amplitude=1.5, phase=0.0)],
            z0=2.0,
        )
        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        config = WindowConfig(
            height_threshold=1.5,
            max_flood_rate=0.2,
            offshore_distance_km=1.0,
            walking_speed_kmh=3.0,
            safety_margin_height=0.3,
            safety_margin_time_minutes=30.0,
            sunset_dt=ref_dt + timedelta(hours=19),
            sunrise_dt=ref_dt + timedelta(hours=5.5),
            time_step_minutes=10.0,
            require_daylight=True,
        )

        start = ref_dt + timedelta(hours=6)
        end = ref_dt + timedelta(hours=18)
        windows, points = find_safe_windows(model, start, end, config, ref_dt=ref_dt)

        md = export_markdown_report(
            model=model,
            windows=windows,
            points=points,
            config=config,
            harbor_name="测试港",
        )

        self.assertIn("#", md)
        self.assertIn("安全窗口", md)
        self.assertIn("涨潮速度", md)
        self.assertIn("返回时间", md)
        self.assertIn("安全余量", md)
        self.assertIn("测试港", md)

    def test_markdown_no_windows(self):
        model = TideModel(
            constituents=[TideConstituent(name="M2", amplitude=1.0, phase=0.0)],
            z0=3.0,
        )
        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        config = WindowConfig(
            height_threshold=0.5,
            max_flood_rate=1.0,
            offshore_distance_km=0.1,
            walking_speed_kmh=10.0,
            safety_margin_height=0.0,
            safety_margin_time_minutes=0.0,
            time_step_minutes=10.0,
            require_daylight=False,
        )
        windows, points = find_safe_windows(
            model, ref_dt, ref_dt + timedelta(hours=24), config, ref_dt=ref_dt
        )

        md = export_markdown_report(
            model=model,
            windows=windows,
            points=points,
            config=config,
        )

        self.assertIn("无安全窗口", md)

    def test_fit_csv(self):
        model = TideModel(
            constituents=[
                TideConstituent(name="M2", amplitude=1.0, phase=30.0),
                TideConstituent(name="S2", amplitude=0.3, phase=60.0),
            ],
            z0=2.0,
        )
        ref_dt = datetime(2026, 6, 21, 0, 0, 0)
        times = [ref_dt + timedelta(minutes=i * 30) for i in range(100)]
        heights = [model.height_at_hours(i * 0.5) for i in range(100)]

        result = fit_tide(times, heights, constituent_names=["M2", "S2"], ref_dt=ref_dt)

        csv_data = export_fit_result_csv(result)
        self.assertIn("rmse_m", csv_data)
        self.assertIn("Constituents", csv_data)
        self.assertIn("amplitude_m", csv_data)


if __name__ == "__main__":
    unittest.main()
