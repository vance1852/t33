import csv
import io
from datetime import datetime
from typing import List, Optional

from .core import TideModel, TideConstituent, hours_from_datetime, datetime_from_hours
from .fitting import FitResult
from .window import SafeWindow, PointSafety, WindowConfig, describe_danger_reason


def _fmt_num(x: float, decimals: int = 3) -> str:
    if x is None:
        return ""
    return f"{x:.{decimals}f}"


def export_tide_predictions_csv(
    model: TideModel,
    times: List[datetime],
    ref_dt: Optional[datetime] = None,
) -> str:
    if ref_dt is None:
        ref_dt = times[0]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["datetime", "hours_from_ref", "height_m", "rate_m_per_h"])

    for dt in times:
        t = hours_from_datetime(ref_dt, dt)
        h = model.height_at_hours(t)
        r = model.rate_at_hours(t)
        writer.writerow([
            dt.strftime("%Y-%m-%d %H:%M"),
            f"{t:.2f}",
            f"{h:.3f}",
            f"{r:.4f}",
        ])

    return output.getvalue()


def export_windows_csv(windows: List[SafeWindow]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "window_index",
        "start_time",
        "end_time",
        "duration_minutes",
        "min_height_m",
        "min_height_time",
        "max_return_height_m",
        "start_blocked_by",
        "end_blocked_by",
    ])

    for i, w in enumerate(windows, 1):
        start_blocked = ";".join(w.reasons_near_start) if w.reasons_near_start else ""
        end_blocked = ";".join(w.reasons_near_end) if w.reasons_near_end else ""
        writer.writerow([
            i,
            w.start_dt.strftime("%Y-%m-%d %H:%M"),
            w.end_dt.strftime("%Y-%m-%d %H:%M"),
            f"{w.duration_minutes:.1f}",
            f"{w.min_height:.3f}",
            w.min_height_dt.strftime("%Y-%m-%d %H:%M"),
            f"{w.max_return_height:.3f}",
            start_blocked,
            end_blocked,
        ])

    return output.getvalue()


def export_points_csv(points: List[PointSafety]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "datetime",
        "height_m",
        "rate_m_per_h",
        "return_height_m",
        "is_safe",
        "danger_reasons",
    ])

    for p in points:
        reasons = ";".join([describe_danger_reason(r) for r in p.reasons])
        writer.writerow([
            p.dt.strftime("%Y-%m-%d %H:%M"),
            f"{p.height:.3f}",
            f"{p.rate:.4f}",
            f"{p.return_height:.3f}",
            "1" if p.is_safe else "0",
            reasons,
        ])

    return output.getvalue()


def export_fit_result_csv(result: FitResult) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["=== Fit Summary ==="])
    writer.writerow(["n_points", result.n_points])
    writer.writerow(["n_skipped", result.n_skipped])
    writer.writerow(["rmse_m", f"{result.rmse:.4f}"])
    writer.writerow(["mae_m", f"{result.mae:.4f}"])
    writer.writerow(["max_abs_residual_m", f"{result.max_abs_residual:.4f}"])
    writer.writerow([])

    writer.writerow(["=== Constituents ==="])
    writer.writerow(["name", "amplitude_m", "phase_deg", "z0_m"])
    writer.writerow(["Z0", "", "", f"{result.model.z0:.4f}"])
    for c in result.model.constituents:
        writer.writerow([c.name, f"{c.amplitude:.4f}", f"{c.phase:.2f}", ""])
    writer.writerow([])

    if result.warnings:
        writer.writerow(["=== Warnings ==="])
        for w in result.warnings:
            writer.writerow([w])

    return output.getvalue()


def export_csv(
    what: str,
    **kwargs,
) -> str:
    if what == "predictions":
        return export_tide_predictions_csv(
            model=kwargs["model"],
            times=kwargs["times"],
            ref_dt=kwargs.get("ref_dt"),
        )
    elif what == "windows":
        return export_windows_csv(kwargs["windows"])
    elif what == "points":
        return export_points_csv(kwargs["points"])
    elif what == "fit":
        return export_fit_result_csv(kwargs["result"])
    else:
        raise ValueError(f"Unknown CSV export type: {what}")


def export_markdown_report(
    model: TideModel,
    windows: List[SafeWindow],
    points: List[PointSafety],
    config: WindowConfig,
    harbor_name: str = "",
    fit_result: Optional[FitResult] = None,
) -> str:
    lines = []

    title = f"# 赶海安全窗口报告"
    if harbor_name:
        title += f" — {harbor_name}"
    lines.append(title)
    lines.append("")

    lines.append("## 参数设置")
    lines.append("")
    lines.append(f"- 潮高安全阈值：{config.height_threshold:.2f} m")
    lines.append(f"- 安全余量（潮高）：{config.safety_margin_height:.2f} m")
    lines.append(f"- 实际保守阈值：{config.safe_height_with_margin:.2f} m")
    lines.append(f"- 最大安全涨潮速度：{config.max_flood_rate:.3f} m/h")
    lines.append(f"- 离岸距离：{config.offshore_distance_km:.1f} km")
    lines.append(f"- 步行速度：{config.walking_speed_kmh:.1f} km/h")
    lines.append(f"- 预计返回时间：{config.return_time_hours * 60:.0f} 分钟")
    lines.append(f"- 时间安全余量：{config.safety_margin_time_minutes:.0f} 分钟")
    if config.sunset_dt:
        lines.append(f"- 日落时间：{config.sunset_dt.strftime('%Y-%m-%d %H:%M')}")
    if config.sunrise_dt:
        lines.append(f"- 日出时间：{config.sunrise_dt.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    lines.append("## 安全窗口汇总")
    lines.append("")
    if not windows:
        lines.append("**无安全窗口！** 请降低阈值或改日赶海。")
        lines.append("")
        lines.append("### 原因分析")
        lines.append("")

        if points:
            reason_counts = {}
            min_h = float("inf")
            max_rate = -float("inf")
            for p in points:
                for r in p.reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
                if p.height < min_h:
                    min_h = p.height
                if p.rate > max_rate:
                    max_rate = p.rate

            lines.append(f"- 分析时间段内最低潮：**{min_h:.3f} m**")
            lines.append(f"- 分析时间段内最大涨潮速度：**{max_rate:.3f} m/h**")
            lines.append("")

            if reason_counts:
                lines.append("#### 限制因素统计")
                lines.append("")
                lines.append("| 限制因素 | 出现点数 | 说明 |")
                lines.append("|----------|----------|------|")
                total = len(points)
                for r, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
                    desc = describe_danger_reason(r)
                    pct = cnt * 100.0 / total if total > 0 else 0
                    lines.append(f"| {desc} | {cnt} ({pct:.0f}%) | {r} |")
                lines.append("")

            lines.append("### 建议调整")
            lines.append("")
            if "height_too_high" in reason_counts:
                lines.append(f"- 潮高阈值建议：当前 {config.height_threshold:.2f} m，最低潮 {min_h:.2f} m，可适当提高阈值")
            if "flood_too_fast" in reason_counts:
                lines.append(f"- 涨潮速度建议：当前上限 {config.max_flood_rate:.3f} m/h，最大涨速 {max_rate:.3f} m/h")
            if "return_not_safe" in reason_counts:
                lines.append(f"- 返回距离建议：当前离岸 {config.offshore_distance_km:.1f} km / 速度 {config.walking_speed_kmh:.1f} km/h，可缩短离岸距离")
            if "darkness" in reason_counts:
                lines.append("- 时间建议：可考虑日出日落限制，选择白天低潮时段")
            lines.append("")
        else:
            lines.append("- 无分析数据点")
            lines.append("")
    else:
        lines.append(f"共找到 **{len(windows)}** 个安全窗口：")
        lines.append("")
        for i, w in enumerate(windows, 1):
            lines.append(f"### 窗口 #{i}")
            lines.append("")
            lines.append(f"- **开始时间**：{w.start_dt.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"- **结束时间**：{w.end_dt.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"- **时长**：{w.duration_minutes:.0f} 分钟")
            lines.append(f"- **最低潮**：{w.min_height:.2f} m（{w.min_height_dt.strftime('%H:%M')}）")
            lines.append(f"- **窗口内最大返回潮高**：{w.max_return_height:.2f} m")

            if w.reasons_near_start:
                lines.append(f"- **窗口前限制因素**：{ '、'.join(describe_danger_reason(r) for r in w.reasons_near_start) }")
            if w.reasons_near_end:
                lines.append(f"- **窗口后限制因素**：{ '、'.join(describe_danger_reason(r) for r in w.reasons_near_end) }")
            lines.append("")

    lines.append("## 危险说明：为什么低潮附近也可能危险？")
    lines.append("")
    lines.append("传统观念认为只要潮低于某个高度就安全，但实际赶海需要考虑多个维度：")
    lines.append("")
    lines.append("### 1. 涨潮速度")
    lines.append("")
    lines.append("即使当前潮高较低，如果正在快速涨潮，你可能会被潮水围困。")
    lines.append(f"本工具设置的安全涨潮速度上限为 **{config.max_flood_rate} m/h**。")
    lines.append("")
    lines.append("### 2. 返回时间")
    lines.append("")
    lines.append("你需要在潮水淹没退路之前走回岸边。")
    lines.append(f"以 {config.walking_speed_kmh} km/h 的速度走 {config.offshore_distance_km} km，")
    lines.append(f"需要 **{config.return_time_hours * 60:.0f} 分钟**。")
    lines.append("工具会检查：从当前时刻开始算，等你走回来的时候，潮水会不会已经涨到超过安全高度。")
    lines.append("")
    lines.append("### 3. 天黑因素")
    lines.append("")
    lines.append("赶海必须在天黑前返回，否则容易迷路或发生意外。")
    lines.append(f"工具考虑了 {config.safety_margin_time_minutes:.0f} 分钟的时间余量。")
    lines.append("")
    lines.append("### 4. 安全余量")
    lines.append("")
    lines.append("所有判断都留有保守余量，宁可多留余地也不冒险。")
    lines.append(f"潮高安全余量：{config.safety_margin_height} m")
    lines.append("")

    if fit_result is not None:
        lines.append("## 拟合结果")
        lines.append("")
        lines.append(f"- 有效数据点：{fit_result.n_points}")
        lines.append(f"- 跳过缺测：{fit_result.n_skipped}")
        lines.append(f"- 均方根误差（RMSE）：{fit_result.rmse:.4f} m")
        lines.append(f"- 平均绝对误差（MAE）：{fit_result.mae:.4f} m")
        lines.append(f"- 最大残差：{fit_result.max_abs_residual:.4f} m")
        lines.append("")
        lines.append("### 分潮参数")
        lines.append("")
        lines.append("| 分潮 | 振幅 (m) | 相位 (°) |")
        lines.append("|------|----------|----------|")
        lines.append(f"| Z0（平均海平面） | — | {fit_result.model.z0:.4f} |")
        for c in fit_result.model.constituents:
            lines.append(f"| {c.name} | {c.amplitude:.4f} | {c.phase:.2f} |")
        lines.append("")

        if fit_result.warnings:
            lines.append("### 警告")
            lines.append("")
            for w in fit_result.warnings:
                lines.append(f"- {w}")
            lines.append("")

    lines.append("## 模型分潮")
    lines.append("")
    lines.append("当前潮汐模型包含以下分潮：")
    lines.append("")
    lines.append("| 分潮 | 类型 | 振幅 (m) | 相位 (°) | 周期 (h) |")
    lines.append("|------|------|----------|----------|----------|")
    for c in model.constituents:
        period = 360.0 / c.omega_deg_per_hour
        ctype = "半日分潮" if period < 14 else "日分潮"
        if period < 7:
            ctype = "浅水分潮"
        lines.append(f"| {c.name} | {ctype} | {c.amplitude:.4f} | {c.phase:.2f} | {period:.2f} |")
    lines.append(f"| Z0 | 平均海平面 | — | — | — |")
    lines.append("")

    return "\n".join(lines)


def export_markdown(
    **kwargs,
) -> str:
    return export_markdown_report(
        model=kwargs["model"],
        windows=kwargs["windows"],
        points=kwargs["points"],
        config=kwargs["config"],
        harbor_name=kwargs.get("harbor_name", ""),
        fit_result=kwargs.get("fit_result"),
    )
