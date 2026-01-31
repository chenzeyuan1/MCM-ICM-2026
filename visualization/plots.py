"""
2026 MCM Problem A: 智能手机电池耗电建模
可视化模块 - 图表生成和结果展示

Author: MCM Team
Date: 2026-01
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Tuple
import sys
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.battery import BatteryModel, BatteryParams
from core.power_model import TotalPowerModel, UsageScenarios, NetworkMode
from core.solver import BatterySolver, MixedUsageSolver, SimulationResult, TimeToEmptyEstimator


class BatteryVisualizer:
    """电池模型可视化工具"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = plt.cm.tab10.colors
    
    def plot_ocv_soc_curve(self, battery: BatteryModel, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制OCV-SOC曲线
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        soc = np.linspace(0, 1, 100)
        ocv = [battery.open_circuit_voltage(s) for s in soc]
        
        ax.plot(soc * 100, ocv, 'b-', linewidth=2, label='$V_{OC}(SOC)$')
        ax.axhline(y=battery.params.voltage_min, color='r', linestyle='--', 
                   label=f'截止电压 ({battery.params.voltage_min}V)')
        ax.axhline(y=battery.params.voltage_max, color='g', linestyle='--',
                   label=f'满充电压 ({battery.params.voltage_max}V)')
        
        ax.set_xlabel('SOC (%)', fontsize=12)
        ax.set_ylabel('开路电压 $V_{OC}$ (V)', fontsize=12)
        ax.set_title('锂离子电池 OCV-SOC 特性曲线', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 100])
        ax.set_ylim([2.8, 4.5])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_temperature_effects(self, battery: BatteryModel,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制温度对电池性能的影响
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        temps = np.linspace(-20, 50, 70)
        
        # 容量vs温度
        capacities = [battery.effective_capacity(t) for t in temps]
        cap_ratio = np.array(capacities) / battery.params.capacity_nominal * 100
        
        axes[0].plot(temps, cap_ratio, 'b-', linewidth=2)
        axes[0].axvline(x=25, color='g', linestyle='--', alpha=0.7, label='参考温度 (25°C)')
        axes[0].fill_between(temps, cap_ratio, alpha=0.3)
        axes[0].set_xlabel('温度 (°C)', fontsize=12)
        axes[0].set_ylabel('有效容量 (%)', fontsize=12)
        axes[0].set_title('温度对电池有效容量的影响', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([-20, 50])
        
        # 内阻vs温度
        resistances = [battery.internal_resistance(t, 0.5) for t in temps]
        res_ratio = np.array(resistances) / battery.params.resistance_internal
        
        axes[1].plot(temps, res_ratio, 'r-', linewidth=2)
        axes[1].axvline(x=25, color='g', linestyle='--', alpha=0.7, label='参考温度 (25°C)')
        axes[1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
        axes[1].set_xlabel('温度 (°C)', fontsize=12)
        axes[1].set_ylabel('内阻比 ($R/R_{ref}$)', fontsize=12)
        axes[1].set_title('温度对电池内阻的影响', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([-20, 50])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_power_breakdown(self, power_model: TotalPowerModel,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制功耗分解饼图
        """
        scenarios = {
            '待机': UsageScenarios.standby(),
            '轻度浏览': UsageScenarios.light_browsing(),
            '视频流': UsageScenarios.video_streaming(),
            '游戏': UsageScenarios.gaming(),
            '导航': UsageScenarios.navigation(),
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()
        
        components = ['screen', 'processor', 'network', 'sensors', 'background', 'misc']
        component_names = ['屏幕', '处理器', '网络', '传感器', '后台', '其他']
        colors = plt.cm.Set3.colors[:len(components)]
        
        for idx, (name, state) in enumerate(scenarios.items()):
            breakdown = power_model.power_breakdown(state)
            values = [breakdown[c] for c in components]
            total = sum(values)
            
            # 过滤掉太小的部分
            filtered_values = []
            filtered_names = []
            filtered_colors = []
            other = 0
            
            for v, n, c in zip(values, component_names, colors):
                if v / total > 0.02:
                    filtered_values.append(v)
                    filtered_names.append(f'{n}\n{v:.2f}W')
                    filtered_colors.append(c)
                else:
                    other += v
            
            if other > 0:
                filtered_values.append(other)
                filtered_names.append(f'其他\n{other:.2f}W')
                filtered_colors.append('gray')
            
            axes[idx].pie(filtered_values, labels=filtered_names, colors=filtered_colors,
                         autopct='%1.1f%%', startangle=90)
            axes[idx].set_title(f'{name}\n总功耗: {total:.2f}W', fontsize=12)
        
        # 隐藏多余的子图
        axes[5].axis('off')
        
        plt.suptitle('不同使用场景的功耗分解', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_soc_discharge(self, results: Dict[str, SimulationResult],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制不同场景的SOC放电曲线
        """
        fig, ax = plt.subplots(figsize=(12, 7), dpi=self.dpi)
        
        for idx, (name, result) in enumerate(results.items()):
            color = self.colors[idx % len(self.colors)]
            ax.plot(result.time, result.soc * 100, '-', color=color, 
                   linewidth=2, label=f'{name} (耗尽: {result.time_to_empty:.1f}h)')
        
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='低电量警告 (20%)')
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='临界电量 (5%)')
        
        ax.set_xlabel('时间 (小时)', fontsize=12)
        ax.set_ylabel('电池电量 SOC (%)', fontsize=12)
        ax.set_title('不同使用场景下的电池放电曲线', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max(r.time[-1] for r in results.values())])
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_daily_usage(self, result: SimulationResult,
                          schedule: List[Tuple[float, float, Dict]],
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制典型一天的使用仿真结果
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=self.dpi, sharex=True)
        
        # SOC曲线
        axes[0].plot(result.time, result.soc * 100, 'b-', linewidth=2)
        axes[0].fill_between(result.time, result.soc * 100, alpha=0.3)
        axes[0].axhline(y=20, color='orange', linestyle='--', alpha=0.7)
        axes[0].set_ylabel('电量 SOC (%)', fontsize=12)
        axes[0].set_title('典型一天使用的电池状态仿真', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 105])
        
        # 功耗曲线
        axes[1].plot(result.time, result.power, 'r-', linewidth=1.5)
        axes[1].fill_between(result.time, result.power, alpha=0.3, color='red')
        axes[1].set_ylabel('功耗 (W)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 添加使用阶段标注
        activity_labels = {
            'standby': '待机',
            'light_browsing': '浏览',
            'video_streaming': '视频',
            'gaming': '游戏',
            'navigation': '导航',
            'social_media': '社交',
            'phone_call': '通话',
        }
        
        # 使用阶段时间轴
        ax3 = axes[2]
        for start, end, state in schedule:
            # 简单的活动识别
            power = TotalPowerModel().power(state)
            if power < 0.2:
                label = '待机'
                color = 'lightgray'
            elif power < 1.0:
                label = '轻度'
                color = 'lightgreen'
            elif power < 2.0:
                label = '中度'
                color = 'yellow'
            else:
                label = '重度'
                color = 'salmon'
            
            ax3.axvspan(start, end, alpha=0.7, color=color, label=label)
        
        ax3.set_xlabel('时间 (小时)', fontsize=12)
        ax3.set_ylabel('使用强度', fontsize=12)
        ax3.set_xlim([0, 24])
        ax3.set_yticks([])
        
        # 添加图例（去重）
        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_time_to_empty_comparison(self, estimator: TimeToEmptyEstimator,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制不同场景的电量耗尽时间对比
        """
        comparison = estimator.compare_scenarios()
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        names = list(comparison.keys())
        times = list(comparison.values())
        
        # 按时间排序
        sorted_pairs = sorted(zip(names, times), key=lambda x: -x[1])
        names, times = zip(*sorted_pairs)
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))
        
        bars = ax.barh(names, times, color=colors)
        
        # 添加数值标签
        for bar, time in zip(bars, times):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{time:.1f}h', va='center', fontsize=10)
        
        ax.set_xlabel('电量耗尽时间 (小时)', fontsize=12)
        ax.set_title('不同使用场景的预计续航时间 (100% → 0%)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_distribution(self, uncertainty_result: Dict,
                                       scenario_name: str,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制不确定性分析结果
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        samples = uncertainty_result['samples']
        
        # 直方图
        axes[0].hist(samples, bins=30, color='steelblue', edgecolor='white', 
                    alpha=0.7, density=True)
        axes[0].axvline(x=uncertainty_result['mean'], color='red', linestyle='-',
                       linewidth=2, label=f'均值: {uncertainty_result["mean"]:.2f}h')
        axes[0].axvline(x=uncertainty_result['ci_95'][0], color='orange', 
                       linestyle='--', label=f'95% CI: [{uncertainty_result["ci_95"][0]:.2f}, {uncertainty_result["ci_95"][1]:.2f}]')
        axes[0].axvline(x=uncertainty_result['ci_95'][1], color='orange', linestyle='--')
        
        axes[0].set_xlabel('电量耗尽时间 (小时)', fontsize=12)
        axes[0].set_ylabel('概率密度', fontsize=12)
        axes[0].set_title(f'{scenario_name} - 耗尽时间分布', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 箱线图
        axes[1].boxplot(samples, vert=True)
        axes[1].scatter([1], [uncertainty_result['mean']], color='red', 
                       s=100, zorder=5, label='均值')
        axes[1].set_ylabel('电量耗尽时间 (小时)', fontsize=12)
        axes[1].set_title(f'{scenario_name} - 统计分布', fontsize=14)
        axes[1].set_xticklabels([scenario_name])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = (f"均值: {uncertainty_result['mean']:.2f}h\n"
                     f"标准差: {uncertainty_result['std']:.2f}h\n"
                     f"中位数: {uncertainty_result['median']:.2f}h\n"
                     f"范围: [{uncertainty_result['min']:.2f}, {uncertainty_result['max']:.2f}]h")
        axes[1].text(1.3, np.mean(samples), stats_text, fontsize=10,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


def generate_all_figures(output_dir: str = 'figures'):
    """
    生成所有图表
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    battery = BatteryModel()
    power_model = TotalPowerModel()
    solver = BatterySolver(battery, power_model)
    estimator = TimeToEmptyEstimator(solver)
    visualizer = BatteryVisualizer()
    
    print("正在生成图表...")
    
    # 1. OCV-SOC曲线
    print("  1. OCV-SOC曲线...")
    visualizer.plot_ocv_soc_curve(battery, f'{output_dir}/ocv_soc_curve.png')
    
    # 2. 温度影响
    print("  2. 温度影响...")
    visualizer.plot_temperature_effects(battery, f'{output_dir}/temperature_effects.png')
    
    # 3. 功耗分解
    print("  3. 功耗分解...")
    visualizer.plot_power_breakdown(power_model, f'{output_dir}/power_breakdown.png')
    
    # 4. 场景对比放电曲线
    print("  4. 放电曲线对比...")
    scenarios = {
        '待机': UsageScenarios.standby(),
        '轻度浏览': UsageScenarios.light_browsing(),
        '视频流': UsageScenarios.video_streaming(),
        '游戏': UsageScenarios.gaming(),
        '导航': UsageScenarios.navigation(),
    }
    results = solver.scenario_comparison(scenarios)
    visualizer.plot_soc_discharge(results, f'{output_dir}/soc_discharge_comparison.png')
    
    # 5. 耗尽时间对比
    print("  5. 耗尽时间对比...")
    visualizer.plot_time_to_empty_comparison(estimator, f'{output_dir}/time_to_empty_comparison.png')
    
    # 6. 典型一天使用
    print("  6. 典型一天仿真...")
    mixed_solver = MixedUsageSolver()
    schedule = mixed_solver.create_daily_schedule()
    mixed_solver.set_usage_schedule(schedule)
    daily_result = mixed_solver.solve_mixed(duration=24.0)
    visualizer.plot_daily_usage(daily_result, schedule, f'{output_dir}/daily_usage.png')
    
    # 7. 不确定性分析
    print("  7. 不确定性分析...")
    uncertainty = estimator.estimate_with_uncertainty(
        UsageScenarios.gaming(), n_samples=500
    )
    visualizer.plot_uncertainty_distribution(uncertainty, '游戏场景', 
                                              f'{output_dir}/uncertainty_gaming.png')
    
    print(f"所有图表已保存到 {output_dir}/ 目录")
    
    plt.close('all')


if __name__ == "__main__":
    generate_all_figures()
