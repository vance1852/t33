import argparse
import sys
from datetime import datetime, timedelta
from typing import List

from .core import TideModel, TideConstituent, hours_from_datetime, datetime_from_hours
from .fitting import fit_tide
from .window import find_safe_windows, WindowConfig
from .report import (
    export_tide_predictions_csv,
    export_windows_csv,
    export_points_csv,
    export_fit_result_csv,
    export_markdown_report,
)
from .samples import (
    get_sample_harbors,
    get_sample_observation_set,
    get_default_reference_time,
    get_tomorrow_sun_times,
)


def _parse_datetime(s: str) -> datetime:
    for fmt in [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"无法解析时间: {s}")


def _get_harbor_model(harbor_name: str):
    harbors = get_sample_harbors()
    if harbor_name not in harbors:
        print(f"错误：未知港口 '{harbor_name}'", file=sys.stderr)
        print(f"可用港口：{', '.join(harbors.keys())}", file=sys.stderr)
        sys.exit(1)
    return harbors[harbor_name]


def cmd_predict(args):
    harbor = _get_harbor_model(args.harbor)
    model = harbor.model

    start_dt = args.start
    duration_hours = args.duration
    step_minutes = args.step

    end_dt = start_dt + timedelta(hours=duration_hours)

    times = []
    step = timedelta(minutes=step_minutes)
    current = start_dt
    while current <= end_dt + timedelta(seconds=1):
        times.append(current)
        current += step

    csv_data = export_tide_predictions_csv(model, times, ref_dt=start_dt)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(csv_data)
        print(f"已导出 {len(times)} 个预测点，已保存到 {args.output}")
    else:
        print(csv_data)

    print(f"\n--- {harbor.name}：{start_dt.strftime('%Y-%m-%d')} 共 {len(times)} 个预测点")


def cmd_fit(args):
    times, heights = get_sample_observation_set(args.harbor)

    if args.constituents:
        const_names = args.constituents.split(",")
    else:
        const_names = ["M2", "S2", "K1", "O1"]

    result = fit_tide(times, heights, constituent_names=const_names)

    csv_data = export_fit_result_csv(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(csv_data)
        print(f"拟合完成，结果已保存到 {args.output}")
    else:
        print(csv_data)

    if result.warnings:
        print("\n警告：")
        for w in result.warnings:
            print(f"  - {w}")


def cmd_window(args):
    harbor = _get_harbor_model(args.harbor)
    model = harbor.model

    start_dt = args.start
    duration_hours = args.duration
    end_dt = start_dt + timedelta(hours=duration_hours)

    sunrise, sunset = get_tomorrow_sun_times(start_dt)
    if args.sunset:
        sunset = args.sunset
    if args.sunrise:
        sunrise = args.sunrise

    config = WindowConfig(
        height_threshold=args.height_threshold,
        max_flood_rate=args.max_flood_rate,
        offshore_distance_km=args.offshore_km,
        walking_speed_kmh=args.walking_speed,
        safety_margin_height=args.margin_height,
        safety_margin_time_minutes=args.margin_time,
        sunset_dt=sunset if args.require_daylight else None,
        sunrise_dt=sunrise if args.require_daylight else None,
        time_step_minutes=args.step_minutes,
        require_daylight=args.require_daylight,
    )

    windows, points = find_safe_windows(model, start_dt, end_dt, config, ref_dt=start_dt)

    if args.output_csv:
        csv_data = export_windows_csv(windows)
        with open(args.output_csv, "w", encoding="utf-8") as f:
            f.write(csv_data)
        print(f"窗口数据已保存到 {args.output_csv}")

    if args.output_points:
        csv_data = export_points_csv(points)
        with open(args.output_points, "w", encoding="utf-8") as f:
            f.write(csv_data)
        print(f"逐点数据已保存到 {args.output_points}")

    print(f"\n=== {harbor.name} 安全窗口")
    print(f"时间范围：{start_dt.strftime('%Y-%m-%d %H:%M')} ~ {end_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"安全阈值：{config.safe_height_with_margin:.2f} m（含 {args.margin_height} m 余量）")
    print()

    if not windows:
        print("⚠️  无安全窗口！")
        print("   可能原因：")
        print("   - 潮高普遍高于阈值")
        print("   - 涨潮速度过快")
        print("   - 可赶海时间恰好在天黑后")
        print()
    else:
        print(f"找到 {len(windows)} 个安全窗口：")
        print()
        for i, w in enumerate(windows, 1):
            print(f"  窗口 #{i}:")
            print(f"    {w.start_dt.strftime('%H:%M')} ~ {w.end_dt.strftime('%H:%M')}")
            print(f"    时长 {w.duration_minutes:.0f} 分钟")
            print(f"    最低潮 {w.min_height:.2f} m（{w.min_height_dt.strftime('%H:%M')}）")
            if w.reasons_near_end:
                from .window import describe_danger_reason
                print(f"    窗口结束原因：{'、'.join(describe_danger_reason(r) for r in w.reasons_near_end)}")
            print()


def cmd_report(args):
    harbor = _get_harbor_model(args.harbor)
    model = harbor.model

    start_dt = args.start
    duration_hours = args.duration
    end_dt = start_dt + timedelta(hours=duration_hours)

    sunrise, sunset = get_tomorrow_sun_times(start_dt)
    if args.sunset:
        sunset = args.sunset
    if args.sunrise:
        sunrise = args.sunrise

    config = WindowConfig(
        height_threshold=args.height_threshold,
        max_flood_rate=args.max_flood_rate,
        offshore_distance_km=args.offshore_km,
        walking_speed_kmh=args.walking_speed,
        safety_margin_height=args.margin_height,
        safety_margin_time_minutes=args.margin_time,
        sunset_dt=sunset if args.require_daylight else None,
        sunrise_dt=sunrise if args.require_daylight else None,
        time_step_minutes=args.step_minutes,
        require_daylight=args.require_daylight,
    )

    windows, points = find_safe_windows(model, start_dt, end_dt, config, ref_dt=start_dt)

    fit_result = None
    if args.with_fit:
        obs_times, obs_heights = get_sample_observation_set(args.harbor)
        fit_result = fit_tide(obs_times, obs_heights)

    md = export_markdown_report(
        model=model,
        windows=windows,
        points=points,
        config=config,
        harbor_name=harbor.name,
        fit_result=fit_result,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"报告已保存到 {args.output}")
    else:
        print(md)


def main():
    parser = argparse.ArgumentParser(
        prog="gansea",
        description="赶海安全窗口工具 — 离线潮汐预测与安全规划",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    predict_parser = subparsers.add_parser("predict", help="潮汐预测")
    predict_parser.add_argument("--harbor", default="qingdao", help="港口名称")
    predict_parser.add_argument("--start", type=_parse_datetime,
                               default=get_default_reference_time(),
                               help="起始时间 (YYYY-MM-DD HH:MM)")
    predict_parser.add_argument("--duration", type=float, default=24.0,
                              help="预测时长（小时）")
    predict_parser.add_argument("--step", type=float, default=60.0,
                              help="时间步长（分钟）")
    predict_parser.add_argument("-o", "--output", help="输出 CSV 文件")

    fit_parser = subparsers.add_parser("fit", help="参数拟合")
    fit_parser.add_argument("--harbor", default="qingdao", help="港口名称")
    fit_parser.add_argument("--constituents", help="分潮列表，逗号分隔，如 M2,S2,K1,O1")
    fit_parser.add_argument("-o", "--output", help="输出 CSV 文件")

    window_parser = subparsers.add_parser("window", help="安全窗口规划")
    window_parser.add_argument("--harbor", default="qingdao", help="港口名称")
    window_parser.add_argument("--start", type=_parse_datetime,
                              default=get_default_reference_time() + timedelta(days=1),
                              help="起始时间 (YYYY-MM-DD HH:MM)")
    window_parser.add_argument("--duration", type=float, default=24.0,
                               help="规划时长（小时）")
    window_parser.add_argument("--height-threshold", type=float, default=1.5,
                              help="潮高安全阈值（米）")
    window_parser.add_argument("--max-flood-rate", type=float, default=0.15,
                               help="最大安全涨潮速度（米/小时）")
    window_parser.add_argument("--offshore-km", type=float, default=1.0,
                               help="离岸距离（公里）")
    window_parser.add_argument("--walking-speed", type=float, default=3.0,
                               help="步行速度（公里/小时）")
    window_parser.add_argument("--margin-height", type=float, default=0.3,
                               help="潮高安全余量（米）")
    window_parser.add_argument("--margin-time", type=float, default=30.0,
                               help="时间安全余量（分钟）")
    window_parser.add_argument("--sunrise", type=_parse_datetime, help="日出时间")
    window_parser.add_argument("--sunset", type=_parse_datetime, help="日落时间")
    window_parser.add_argument("--step-minutes", type=float, default=10.0,
                               help="计算步长（分钟）")
    window_parser.add_argument("--no-daylight", dest="require_daylight",
                              action="store_false",
                              help="不考虑日出日落")
    window_parser.add_argument("--output-csv", help="输出窗口 CSV 文件")
    window_parser.add_argument("--output-points", help="输出逐点 CSV 文件")

    report_parser = subparsers.add_parser("report", help="生成完整报告")
    report_parser.add_argument("--harbor", default="qingdao", help="港口名称")
    report_parser.add_argument("--start", type=_parse_datetime,
                               default=get_default_reference_time() + timedelta(days=1),
                               help="起始时间")
    report_parser.add_argument("--duration", type=float, default=24.0,
                               help="规划时长（小时）")
    report_parser.add_argument("--height-threshold", type=float, default=1.5,
                               help="潮高安全阈值（米）")
    report_parser.add_argument("--max-flood-rate", type=float, default=0.15,
                               help="最大安全涨潮速度（米/小时）")
    report_parser.add_argument("--offshore-km", type=float, default=1.0,
                               help="离岸距离（公里）")
    report_parser.add_argument("--walking-speed", type=float, default=3.0,
                               help="步行速度（公里/小时）")
    report_parser.add_argument("--margin-height", type=float, default=0.3,
                               help="潮高安全余量（米）")
    report_parser.add_argument("--margin-time", type=float, default=30.0,
                               help="时间安全余量（分钟）")
    report_parser.add_argument("--sunrise", type=_parse_datetime, help="日出时间")
    report_parser.add_argument("--sunset", type=_parse_datetime, help="日落时间")
    report_parser.add_argument("--step-minutes", type=float, default=10.0,
                               help="计算步长（分钟）")
    report_parser.add_argument("--no-daylight", dest="require_daylight",
                               action="store_false",
                               help="不考虑日出日落")
    report_parser.add_argument("--with-fit", action="store_true",
                               help="包含拟合结果")
    report_parser.add_argument("-o", "--output", help="输出 Markdown 文件")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "fit":
        cmd_fit(args)
    elif args.command == "window":
        cmd_window(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
