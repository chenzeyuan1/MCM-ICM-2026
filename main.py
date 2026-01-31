"""
2026 MCM Problem A: 智能手机电池耗电建模
主程序入口 - 运行完整仿真和分析

Author: MCM Team
Date: 2026-01
"""

import os
import sys
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.battery import BatteryModel, BatteryParams
from core.power_model import TotalPowerModel, UsageScenarios, NetworkMode
from core.solver import BatterySolver, MixedUsageSolver, TimeToEmptyEstimator
from visualization.plots import BatteryVisualizer, generate_all_figures
from analysis.sensitivity import SensitivityAnalyzer, SensitivityVisualizer, run_sensitivity_analysis


def print_header(title: str):
    """打印分隔标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def demo_battery_model():
    """演示电池基础模型"""
    print_header("1. 电池基础模型演示")
    
    battery = BatteryModel()
    
    print("【电池参数】")
    print(f"  标称容量: {battery.params.capacity_nominal} mAh")
    print(f"  标称电压: {battery.params.voltage_nominal} V")
    print(f"  内阻: {battery.params.resistance_internal*1000:.1f} mΩ")
    print()
    
    print("【OCV-SOC曲线】")
    for soc in [1.0, 0.8, 0.5, 0.2, 0.0]:
        v_oc = battery.open_circuit_voltage(soc)
        print(f"  SOC = {soc*100:3.0f}%: V_OC = {v_oc:.3f} V")
    print()
    
    print("【温度对有效容量的影响】")
    for temp in [-10, 0, 25, 40]:
        c_eff = battery.effective_capacity(temp)
        ratio = c_eff / battery.params.capacity_nominal * 100
        print(f"  T = {temp:3d}°C: C_eff = {c_eff:.0f} mAh ({ratio:.1f}%)")
    print()
    
    print("【电池老化影响 (25°C)】")
    original_cycles = battery.params.cycle_count
    for cycles in [0, 100, 300, 500, 800]:
        battery.params.cycle_count = cycles
        c_eff = battery.effective_capacity(25)
        ratio = c_eff / battery.params.capacity_nominal * 100
        print(f"  循环 {cycles:4d} 次: C_eff = {c_eff:.0f} mAh ({ratio:.1f}%)")
    battery.params.cycle_count = original_cycles


def demo_power_model():
    """演示功耗模型"""
    print_header("2. 功耗模型演示")
    
    power_model = TotalPowerModel()
    
    scenarios = [
        ("待机", UsageScenarios.standby()),
        ("轻度浏览", UsageScenarios.light_browsing()),
        ("视频流媒体", UsageScenarios.video_streaming()),
        ("游戏", UsageScenarios.gaming()),
        ("导航", UsageScenarios.navigation()),
        ("通话", UsageScenarios.phone_call()),
        ("社交媒体", UsageScenarios.social_media()),
    ]
    
    print("【各场景功耗分解】\n")
    print(f"{'场景':<12} {'屏幕':>8} {'处理器':>8} {'网络':>8} {'传感器':>8} {'后台':>8} {'总计':>8}")
    print("-" * 70)
    
    for name, state in scenarios:
        breakdown = power_model.power_breakdown(state)
        print(f"{name:<12} {breakdown['screen']:>7.2f}W {breakdown['processor']:>7.2f}W "
              f"{breakdown['network']:>7.2f}W {breakdown['sensors']:>7.2f}W "
              f"{breakdown['background']:>7.2f}W {breakdown['total']:>7.2f}W")


def demo_solver():
    """演示数值求解"""
    print_header("3. SOC动力学求解演示")
    
    solver = BatterySolver()
    
    # 单场景仿真
    print("【轻度浏览场景仿真】")
    result = solver.solve(
        soc_initial=1.0,
        device_state=UsageScenarios.light_browsing(),
        duration=24.0,
        soc_cutoff=0.0
    )
    print(f"  初始电量: 100%")
    print(f"  功耗: {result.power[0]:.2f} W")
    print(f"  电量耗尽时间: {result.time_to_empty:.2f} 小时")
    print()
    
    # 多场景对比
    print("【各场景电量耗尽时间对比】")
    estimator = TimeToEmptyEstimator(solver)
    comparison = estimator.compare_scenarios()
    
    for name, tte in sorted(comparison.items(), key=lambda x: -x[1]):
        hours = int(tte)
        minutes = int((tte - hours) * 60)
        print(f"  {name}: {hours}小时{minutes}分钟 ({tte:.1f}h)")
    print()
    
    # 典型一天使用
    print("【典型一天使用仿真】")
    mixed_solver = MixedUsageSolver()
    schedule = mixed_solver.create_daily_schedule()
    mixed_solver.set_usage_schedule(schedule)
    
    result = mixed_solver.solve_mixed(soc_initial=1.0, duration=24.0)
    print(f"  24小时后剩余电量: {result.soc_final*100:.1f}%")
    
    # 找出电量最低的时刻
    min_soc_idx = np.argmin(result.soc)
    print(f"  最低电量时刻: {result.time[min_soc_idx]:.1f}h, SOC={result.soc[min_soc_idx]*100:.1f}%")
    
    if result.time_to_empty < 24.0:
        print(f"  ⚠️ 电量在 {result.time_to_empty:.1f} 小时耗尽!")


def demo_uncertainty():
    """演示不确定性分析"""
    print_header("4. 不确定性分析演示")
    
    solver = BatterySolver()
    estimator = TimeToEmptyEstimator(solver)
    
    scenarios_to_analyze = [
        ("游戏", UsageScenarios.gaming()),
        ("视频流", UsageScenarios.video_streaming()),
        ("导航", UsageScenarios.navigation()),
    ]
    
    print("【Monte Carlo不确定性估计 (500次采样)】\n")
    
    for name, state in scenarios_to_analyze:
        uncertainty = estimator.estimate_with_uncertainty(state, n_samples=500)
        print(f"【{name}场景】")
        print(f"  平均耗尽时间: {uncertainty['mean']:.2f} ± {uncertainty['std']:.2f} 小时")
        print(f"  95%置信区间: [{uncertainty['ci_95'][0]:.2f}, {uncertainty['ci_95'][1]:.2f}] 小时")
        print(f"  变异系数CV: {uncertainty['std']/uncertainty['mean']*100:.1f}%")
        print()


def demo_sensitivity():
    """演示敏感度分析"""
    print_header("5. 敏感度分析演示")
    
    solver = BatterySolver()
    analyzer = SensitivityAnalyzer(solver)
    base_state = UsageScenarios.light_browsing()
    
    # 功耗参数敏感度
    print("【功耗参数敏感度 (对耗尽时间)】")
    param_ranges = {
        'brightness': (0.1, 1.0),
        'cpu_utilization': (0.05, 0.9),
        'network_activity': (0.0, 0.8),
    }
    
    param_names = {
        'brightness': '屏幕亮度',
        'cpu_utilization': 'CPU利用率',
        'network_activity': '网络活动',
    }
    
    tte_sens = analyzer.local_sensitivity_tte(base_state, param_ranges)
    
    for param, result in sorted(tte_sens.items(), key=lambda x: -x[1].sensitivity_index):
        print(f"  {param_names[param]}: S = {result.sensitivity_index:.3f}")
    print()
    
    # 电池参数敏感度
    print("【电池参数敏感度】")
    battery_sens = analyzer.battery_parameter_sensitivity(base_state)
    
    battery_names = {
        'capacity_nominal': '电池容量',
        'voltage_nominal': '标称电压',
        'resistance_internal': '内阻',
    }
    
    for param, sens in sorted(battery_sens.items(), key=lambda x: -x[1]):
        print(f"  {battery_names[param]}: S = {sens:.3f}")
    print()
    
    # 温度和老化
    print("【环境因素敏感度】")
    temp_result = analyzer.temperature_sensitivity(base_state)
    aging_result = analyzer.aging_sensitivity(base_state)
    
    print(f"  温度: S = {temp_result.sensitivity_index:.3f}")
    print(f"  老化(循环次数): S = {aging_result.sensitivity_index:.3f}")


def generate_recommendations():
    """生成用户建议"""
    print_header("6. 用户建议")
    
    solver = BatterySolver()
    power_model = TotalPowerModel()
    
    base_state = UsageScenarios.light_browsing()
    base_power = power_model.power(base_state)
    base_tte = solver.time_to_empty_analytical(1.0, base_power, 25)
    
    recommendations = []
    
    # 1. 降低亮度
    state_low_brightness = base_state.copy()
    state_low_brightness['brightness'] = 0.3  # 从0.4降到0.3
    power_low = power_model.power(state_low_brightness)
    tte_low = solver.time_to_empty_analytical(1.0, power_low, 25)
    improvement = (tte_low - base_tte) / base_tte * 100
    recommendations.append(("降低屏幕亮度10%", f"+{improvement:.1f}%", "简单"))
    
    # 2. 使用WiFi替代蜂窝
    state_wifi = base_state.copy()
    state_wifi['network_mode'] = NetworkMode.WIFI
    power_wifi = power_model.power(state_wifi)
    tte_wifi = solver.time_to_empty_analytical(1.0, power_wifi, 25)
    improvement = (tte_wifi - base_tte) / base_tte * 100
    recommendations.append(("使用WiFi替代蜂窝网络", f"+{improvement:.1f}%", "中等"))
    
    # 3. 关闭后台
    state_no_bg = base_state.copy()
    state_no_bg['background_tasks'] = {'system_services': 0.01}
    power_no_bg = power_model.power(state_no_bg)
    tte_no_bg = solver.time_to_empty_analytical(1.0, power_no_bg, 25)
    improvement = (tte_no_bg - base_tte) / base_tte * 100
    recommendations.append(("限制后台应用", f"+{improvement:.1f}%", "简单"))
    
    # 4. 降低CPU (降低刷新率/动画)
    state_low_cpu = base_state.copy()
    state_low_cpu['cpu_utilization'] = 0.1  # 从0.15降到0.1
    power_low_cpu = power_model.power(state_low_cpu)
    tte_low_cpu = solver.time_to_empty_analytical(1.0, power_low_cpu, 25)
    improvement = (tte_low_cpu - base_tte) / base_tte * 100
    recommendations.append(("降低动画效果/刷新率", f"+{improvement:.1f}%", "中等"))
    
    print("【基于模型的省电建议】\n")
    print(f"基准场景: 轻度浏览, 功耗 {base_power:.2f}W, 续航 {base_tte:.1f}小时\n")
    
    print(f"{'建议':<25} {'续航提升':>12} {'实施难度':>10}")
    print("-" * 50)
    for rec, imp, diff in recommendations:
        print(f"{rec:<25} {imp:>12} {diff:>10}")
    
    print("\n【综合省电策略】")
    # 应用所有优化
    state_optimized = base_state.copy()
    state_optimized['brightness'] = 0.3
    state_optimized['network_mode'] = NetworkMode.WIFI
    state_optimized['background_tasks'] = {'system_services': 0.01}
    state_optimized['cpu_utilization'] = 0.1
    power_opt = power_model.power(state_optimized)
    tte_opt = solver.time_to_empty_analytical(1.0, power_opt, 25)
    
    print(f"  优化后功耗: {power_opt:.2f}W (原 {base_power:.2f}W)")
    print(f"  优化后续航: {tte_opt:.1f}小时 (原 {base_tte:.1f}小时)")
    print(f"  总体提升: +{(tte_opt-base_tte)/base_tte*100:.1f}%")


def main():
    """主程序入口"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                              ║")
    print("║   2026 MCM Problem A: 智能手机电池耗电建模                   ║")
    print("║   Modeling Smartphone Battery Drain                          ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # 运行演示
    demo_battery_model()
    demo_power_model()
    demo_solver()
    demo_uncertainty()
    demo_sensitivity()
    generate_recommendations()
    
    # 生成图表
    print_header("7. 生成可视化图表")
    
    output_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出目录: {output_dir}\n")
    
    try:
        # 生成主要图表
        generate_all_figures(output_dir)
        
        # 运行敏感度分析并生成图表
        print("\n正在运行敏感度分析...")
        run_sensitivity_analysis(output_dir)
        
        print("\n✅ 所有分析完成!")
        print(f"   图表已保存到: {output_dir}/")
        print(f"   报告文件: {os.path.join(PROJECT_ROOT, 'REPORT.md')}")
        
    except Exception as e:
        print(f"\n⚠️ 图表生成时出现错误: {e}")
        print("   请确保已安装 matplotlib: pip install matplotlib")
    
    print("\n" + "="*60)
    print("  程序运行完毕")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
