"""
2026 MCM Problem A: 智能手机电池耗电建模
敏感度分析模块 - 参数敏感度和模型稳健性分析

Author: MCM Team
Date: 2026-01
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import sys
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.battery import BatteryModel, BatteryParams
from core.power_model import TotalPowerModel, UsageScenarios
from core.solver import BatterySolver, TimeToEmptyEstimator


@dataclass
class SensitivityResult:
    """敏感度分析结果"""
    parameter_name: str
    parameter_values: np.ndarray
    output_values: np.ndarray
    baseline_value: float
    baseline_output: float
    sensitivity_index: float  # 归一化敏感度指数


class SensitivityAnalyzer:
    """
    敏感度分析器
    
    支持:
    1. 局部敏感度分析 (OAT - One At a Time)
    2. 全局敏感度分析 (Sobol' indices)
    3. Morris方法 (Elementary Effects)
    """
    
    def __init__(self, solver: BatterySolver):
        self.solver = solver
        self.power_model = solver.power_model
        self.battery = solver.battery
    
    def local_sensitivity_power(self, base_state: Dict, 
                                 parameter_ranges: Dict[str, Tuple[float, float]],
                                 n_points: int = 20) -> Dict[str, SensitivityResult]:
        """
        功耗参数的局部敏感度分析 (OAT方法)
        
        Args:
            base_state: 基准设备状态
            parameter_ranges: 参数变化范围 {参数名: (min, max)}
            n_points: 采样点数
            
        Returns:
            敏感度结果字典
        """
        results = {}
        base_power = self.power_model.power(base_state)
        
        for param_name, (p_min, p_max) in parameter_ranges.items():
            values = np.linspace(p_min, p_max, n_points)
            powers = []
            
            for val in values:
                state = base_state.copy()
                state[param_name] = val
                powers.append(self.power_model.power(state))
            
            powers = np.array(powers)
            
            # 计算归一化敏感度指数
            # S = (dY/Y) / (dX/X) ≈ (ΔY/Y) / (ΔX/X)
            baseline_idx = np.argmin(np.abs(values - base_state.get(param_name, (p_min + p_max)/2)))
            baseline_param = values[baseline_idx]
            
            if baseline_param != 0 and base_power != 0:
                delta_y = (powers[-1] - powers[0]) / base_power
                delta_x = (values[-1] - values[0]) / baseline_param
                sensitivity_index = delta_y / delta_x if delta_x != 0 else 0
            else:
                sensitivity_index = (powers[-1] - powers[0]) / (values[-1] - values[0]) if (values[-1] - values[0]) != 0 else 0
            
            results[param_name] = SensitivityResult(
                parameter_name=param_name,
                parameter_values=values,
                output_values=powers,
                baseline_value=baseline_param,
                baseline_output=base_power,
                sensitivity_index=abs(sensitivity_index)
            )
        
        return results
    
    def local_sensitivity_tte(self, base_state: Dict,
                               parameter_ranges: Dict[str, Tuple[float, float]],
                               soc_initial: float = 1.0,
                               temperature: float = 25.0,
                               n_points: int = 20) -> Dict[str, SensitivityResult]:
        """
        电量耗尽时间的局部敏感度分析
        """
        results = {}
        base_power = self.power_model.power(base_state)
        base_tte = self.solver.time_to_empty_analytical(soc_initial, base_power, temperature)
        
        for param_name, (p_min, p_max) in parameter_ranges.items():
            values = np.linspace(p_min, p_max, n_points)
            ttes = []
            
            for val in values:
                state = base_state.copy()
                state[param_name] = val
                power = self.power_model.power(state)
                tte = self.solver.time_to_empty_analytical(soc_initial, power, temperature)
                ttes.append(tte)
            
            ttes = np.array(ttes)
            
            # 计算敏感度
            baseline_idx = n_points // 2
            baseline_param = values[baseline_idx]
            
            if baseline_param != 0 and base_tte != 0:
                delta_y = (ttes[-1] - ttes[0]) / base_tte
                delta_x = (values[-1] - values[0]) / baseline_param
                sensitivity_index = delta_y / delta_x if delta_x != 0 else 0
            else:
                sensitivity_index = 0
            
            results[param_name] = SensitivityResult(
                parameter_name=param_name,
                parameter_values=values,
                output_values=ttes,
                baseline_value=baseline_param,
                baseline_output=base_tte,
                sensitivity_index=abs(sensitivity_index)
            )
        
        return results
    
    def battery_parameter_sensitivity(self, base_state: Dict,
                                       soc_initial: float = 1.0,
                                       temperature: float = 25.0) -> Dict[str, float]:
        """
        电池参数对耗尽时间的敏感度分析
        """
        base_power = self.power_model.power(base_state)
        base_tte = self.solver.time_to_empty_analytical(soc_initial, base_power, temperature)
        
        sensitivities = {}
        perturbation = 0.1  # ±10%
        
        params_to_test = [
            ('capacity_nominal', self.battery.params.capacity_nominal),
            ('voltage_nominal', self.battery.params.voltage_nominal),
            ('resistance_internal', self.battery.params.resistance_internal),
        ]
        
        for param_name, base_value in params_to_test:
            # 保存原值
            original = getattr(self.battery.params, param_name)
            
            # 增加10%
            setattr(self.battery.params, param_name, base_value * (1 + perturbation))
            tte_plus = self.solver.time_to_empty_analytical(soc_initial, base_power, temperature)
            
            # 减少10%
            setattr(self.battery.params, param_name, base_value * (1 - perturbation))
            tte_minus = self.solver.time_to_empty_analytical(soc_initial, base_power, temperature)
            
            # 恢复原值
            setattr(self.battery.params, param_name, original)
            
            # 计算敏感度
            sensitivity = (tte_plus - tte_minus) / (2 * perturbation * base_tte)
            sensitivities[param_name] = abs(sensitivity)
        
        return sensitivities
    
    def temperature_sensitivity(self, base_state: Dict,
                                 soc_initial: float = 1.0,
                                 temp_range: Tuple[float, float] = (-10, 45),
                                 n_points: int = 20) -> SensitivityResult:
        """
        温度敏感度分析
        """
        temps = np.linspace(temp_range[0], temp_range[1], n_points)
        ttes = []
        
        base_power = self.power_model.power(base_state)
        
        for temp in temps:
            tte = self.solver.time_to_empty_analytical(soc_initial, base_power, temp)
            ttes.append(tte)
        
        ttes = np.array(ttes)
        
        # 基准温度25°C
        base_idx = np.argmin(np.abs(temps - 25))
        base_tte = ttes[base_idx]
        
        # 敏感度
        if base_tte != 0:
            delta_y = (ttes[-1] - ttes[0]) / base_tte
            delta_x = (temps[-1] - temps[0]) / 25
            sensitivity_index = delta_y / delta_x
        else:
            sensitivity_index = 0
        
        return SensitivityResult(
            parameter_name='temperature',
            parameter_values=temps,
            output_values=ttes,
            baseline_value=25.0,
            baseline_output=base_tte,
            sensitivity_index=abs(sensitivity_index)
        )
    
    def aging_sensitivity(self, base_state: Dict,
                           soc_initial: float = 1.0,
                           temperature: float = 25.0,
                           cycle_range: Tuple[int, int] = (0, 1000),
                           n_points: int = 20) -> SensitivityResult:
        """
        电池老化敏感度分析
        """
        cycles = np.linspace(cycle_range[0], cycle_range[1], n_points).astype(int)
        ttes = []
        
        base_power = self.power_model.power(base_state)
        original_cycles = self.battery.params.cycle_count
        
        for n_cycles in cycles:
            self.battery.params.cycle_count = n_cycles
            tte = self.solver.time_to_empty_analytical(soc_initial, base_power, temperature)
            ttes.append(tte)
        
        self.battery.params.cycle_count = original_cycles
        ttes = np.array(ttes)
        
        # 基准: 新电池
        base_tte = ttes[0]
        
        if base_tte != 0 and cycles[-1] != 0:
            delta_y = (ttes[-1] - ttes[0]) / base_tte
            delta_x = cycles[-1] / 500  # 归一化到500次循环
            sensitivity_index = delta_y / delta_x
        else:
            sensitivity_index = 0
        
        return SensitivityResult(
            parameter_name='cycle_count',
            parameter_values=cycles.astype(float),
            output_values=ttes,
            baseline_value=0,
            baseline_output=base_tte,
            sensitivity_index=abs(sensitivity_index)
        )
    
    def morris_screening(self, base_state: Dict,
                          parameter_ranges: Dict[str, Tuple[float, float]],
                          n_trajectories: int = 10,
                          n_levels: int = 4) -> Dict[str, Tuple[float, float]]:
        """
        Morris筛选方法 (Elementary Effects)
        
        计算每个参数的:
        - μ*: 绝对平均效应 (影响大小)
        - σ: 效应标准差 (非线性/交互程度)
        
        Returns:
            {参数名: (μ*, σ)}
        """
        elementary_effects = {k: [] for k in parameter_ranges.keys()}
        
        for _ in range(n_trajectories):
            # 随机起点
            current_state = base_state.copy()
            for param, (p_min, p_max) in parameter_ranges.items():
                level = np.random.randint(0, n_levels)
                current_state[param] = p_min + level * (p_max - p_min) / (n_levels - 1)
            
            current_power = self.power_model.power(current_state)
            
            # 随机顺序扰动参数
            param_order = list(parameter_ranges.keys())
            np.random.shuffle(param_order)
            
            for param in param_order:
                p_min, p_max = parameter_ranges[param]
                delta = (p_max - p_min) / (n_levels - 1)
                
                # 扰动方向
                direction = np.random.choice([-1, 1])
                new_value = current_state[param] + direction * delta
                new_value = np.clip(new_value, p_min, p_max)
                
                # 计算效应
                new_state = current_state.copy()
                new_state[param] = new_value
                new_power = self.power_model.power(new_state)
                
                # 归一化效应
                effect = (new_power - current_power) / delta if delta != 0 else 0
                elementary_effects[param].append(effect)
                
                # 更新状态
                current_state = new_state
                current_power = new_power
        
        # 计算统计量
        results = {}
        for param, effects in elementary_effects.items():
            effects = np.array(effects)
            mu_star = np.mean(np.abs(effects))  # 绝对平均
            sigma = np.std(effects)              # 标准差
            results[param] = (mu_star, sigma)
        
        return results
    
    def sobol_indices_first_order(self, base_state: Dict,
                                   parameter_ranges: Dict[str, Tuple[float, float]],
                                   n_samples: int = 1000) -> Dict[str, float]:
        """
        Sobol一阶敏感度指数 (基于方差分解)
        
        S_i = Var[E(Y|X_i)] / Var(Y)
        
        使用Monte Carlo估计
        """
        # 生成样本矩阵
        params = list(parameter_ranges.keys())
        n_params = len(params)
        
        # 矩阵A和B
        A = np.random.rand(n_samples, n_params)
        B = np.random.rand(n_samples, n_params)
        
        # 缩放到参数范围
        for i, param in enumerate(params):
            p_min, p_max = parameter_ranges[param]
            A[:, i] = p_min + A[:, i] * (p_max - p_min)
            B[:, i] = p_min + B[:, i] * (p_max - p_min)
        
        # 计算A和B的输出
        def evaluate(sample_matrix):
            outputs = []
            for row in sample_matrix:
                state = base_state.copy()
                for j, param in enumerate(params):
                    state[param] = row[j]
                outputs.append(self.power_model.power(state))
            return np.array(outputs)
        
        Y_A = evaluate(A)
        Y_B = evaluate(B)
        
        # 总方差
        var_total = np.var(np.concatenate([Y_A, Y_B]))
        
        if var_total == 0:
            return {param: 0 for param in params}
        
        # 一阶指数
        sobol_indices = {}
        
        for i, param in enumerate(params):
            # 构造AB_i矩阵
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            Y_AB_i = evaluate(AB_i)
            
            # 一阶指数估计
            S_i = np.mean(Y_B * (Y_AB_i - Y_A)) / var_total
            sobol_indices[param] = max(0, S_i)  # 确保非负
        
        return sobol_indices


class SensitivityVisualizer:
    """敏感度分析结果可视化"""
    
    def __init__(self, dpi: int = 150):
        self.dpi = dpi
    
    def plot_local_sensitivity(self, results: Dict[str, SensitivityResult],
                                output_label: str = '功耗 (W)',
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制局部敏感度分析结果
        """
        n_params = len(results)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(14, 8), dpi=self.dpi)
        axes = axes.flatten()
        
        param_names_cn = {
            'brightness': '屏幕亮度',
            'cpu_utilization': 'CPU利用率',
            'network_activity': '网络活动',
            'signal_strength': '信号强度',
        }
        
        for idx, (param_name, result) in enumerate(results.items()):
            ax = axes[idx]
            
            ax.plot(result.parameter_values, result.output_values, 'b-', linewidth=2)
            ax.axvline(x=result.baseline_value, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=result.baseline_output, color='r', linestyle='--', alpha=0.7)
            
            display_name = param_names_cn.get(param_name, param_name)
            ax.set_xlabel(display_name, fontsize=11)
            ax.set_ylabel(output_label, fontsize=11)
            ax.set_title(f'敏感度指数: {result.sensitivity_index:.3f}', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余子图
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('局部敏感度分析 (OAT方法)', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_ranking(self, sensitivities: Dict[str, float],
                                  title: str = '参数敏感度排名',
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制敏感度排名条形图
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # 排序
        sorted_items = sorted(sensitivities.items(), key=lambda x: -x[1])
        names, values = zip(*sorted_items)
        
        param_names_cn = {
            'brightness': '屏幕亮度',
            'cpu_utilization': 'CPU利用率',
            'network_activity': '网络活动',
            'signal_strength': '信号强度',
            'capacity_nominal': '电池容量',
            'voltage_nominal': '标称电压',
            'resistance_internal': '内阻',
            'temperature': '温度',
            'cycle_count': '循环次数',
        }
        
        names = [param_names_cn.get(n, n) for n in names]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
        bars = ax.barh(names, values, color=colors)
        
        ax.set_xlabel('敏感度指数', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_morris_results(self, morris_results: Dict[str, Tuple[float, float]],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制Morris筛选结果 (μ* vs σ)
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        param_names_cn = {
            'brightness': '屏幕亮度',
            'cpu_utilization': 'CPU利用率',
            'network_activity': '网络活动',
            'signal_strength': '信号强度',
        }
        
        for param, (mu_star, sigma) in morris_results.items():
            display_name = param_names_cn.get(param, param)
            ax.scatter(mu_star, sigma, s=100, zorder=5)
            ax.annotate(display_name, (mu_star, sigma), xytext=(5, 5),
                       textcoords='offset points', fontsize=10)
        
        # 添加参考线
        max_mu = max(m for m, _ in morris_results.values()) * 1.2
        ax.plot([0, max_mu], [0, max_mu], 'k--', alpha=0.3, label='σ = μ*')
        ax.plot([0, max_mu], [0, 0.5 * max_mu], 'g--', alpha=0.3, label='σ = 0.5μ*')
        
        ax.set_xlabel('μ* (绝对平均效应 - 影响大小)', fontsize=12)
        ax.set_ylabel('σ (效应标准差 - 非线性/交互)', fontsize=12)
        ax.set_title('Morris筛选分析', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加区域说明
        ax.text(0.95, 0.95, '高σ: 非线性/交互效应显著\n低σ: 近似线性效应',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_temperature_aging_sensitivity(self, 
                                            temp_result: SensitivityResult,
                                            aging_result: SensitivityResult,
                                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制温度和老化敏感度
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        # 温度敏感度
        axes[0].plot(temp_result.parameter_values, temp_result.output_values, 
                    'b-', linewidth=2)
        axes[0].axvline(x=25, color='g', linestyle='--', alpha=0.7, label='参考温度 (25°C)')
        axes[0].fill_between(temp_result.parameter_values, temp_result.output_values,
                            alpha=0.3)
        axes[0].set_xlabel('温度 (°C)', fontsize=12)
        axes[0].set_ylabel('电量耗尽时间 (小时)', fontsize=12)
        axes[0].set_title(f'温度敏感度 (S={temp_result.sensitivity_index:.3f})', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 老化敏感度
        axes[1].plot(aging_result.parameter_values, aging_result.output_values,
                    'r-', linewidth=2)
        axes[1].fill_between(aging_result.parameter_values, aging_result.output_values,
                            alpha=0.3, color='red')
        axes[1].set_xlabel('充电循环次数', fontsize=12)
        axes[1].set_ylabel('电量耗尽时间 (小时)', fontsize=12)
        axes[1].set_title(f'老化敏感度 (S={aging_result.sensitivity_index:.3f})', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


def run_sensitivity_analysis(output_dir: str = 'figures'):
    """
    运行完整的敏感度分析
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化
    solver = BatterySolver()
    analyzer = SensitivityAnalyzer(solver)
    visualizer = SensitivityVisualizer()
    
    base_state = UsageScenarios.light_browsing()
    
    print("正在进行敏感度分析...")
    
    # 1. 局部敏感度分析 - 功耗 (包含CPU频率)
    print("  1. 功耗参数局部敏感度...")
    power_params = {
        'brightness': (0.1, 1.0),
        'cpu_utilization': (0.05, 0.9),
        'cpu_freq_ratio': (0.2, 1.0),
        'network_activity': (0.0, 0.8),
    }
    power_sensitivity = analyzer.local_sensitivity_power(base_state, power_params)
    visualizer.plot_local_sensitivity(power_sensitivity, '功耗 (W)',
                                       f'{output_dir}/sensitivity_power.png')
    
    # 2. 局部敏感度分析 - 耗尽时间
    print("  2. 耗尽时间参数敏感度...")
    tte_sensitivity = analyzer.local_sensitivity_tte(base_state, power_params)
    visualizer.plot_local_sensitivity(tte_sensitivity, '耗尽时间 (小时)',
                                       f'{output_dir}/sensitivity_tte.png')
    
    # 3. 敏感度排名
    print("  3. 敏感度排名...")
    all_sensitivities = {k: v.sensitivity_index for k, v in tte_sensitivity.items()}
    battery_sens = analyzer.battery_parameter_sensitivity(base_state)
    all_sensitivities.update(battery_sens)
    visualizer.plot_sensitivity_ranking(all_sensitivities,
                                         f'{output_dir}/sensitivity_ranking.png')
    
    # 4. Morris筛选
    print("  4. Morris筛选分析...")
    morris_results = analyzer.morris_screening(base_state, power_params, n_trajectories=20)
    visualizer.plot_morris_results(morris_results, f'{output_dir}/morris_screening.png')
    
    # 5. 温度和老化敏感度
    print("  5. 温度和老化敏感度...")
    temp_result = analyzer.temperature_sensitivity(base_state)
    aging_result = analyzer.aging_sensitivity(base_state)
    visualizer.plot_temperature_aging_sensitivity(temp_result, aging_result,
                                                   f'{output_dir}/temp_aging_sensitivity.png')
    
    # 6. Sobol指数
    print("  6. Sobol一阶敏感度指数...")
    sobol_indices = analyzer.sobol_indices_first_order(base_state, power_params, n_samples=500)
    print("\n  Sobol一阶敏感度指数:")
    for param, index in sorted(sobol_indices.items(), key=lambda x: -x[1]):
        print(f"    {param}: {index:.4f}")
    
    print(f"\n敏感度分析完成，图表已保存到 {output_dir}/")
    
    plt.close('all')
    
    return {
        'power_sensitivity': power_sensitivity,
        'tte_sensitivity': tte_sensitivity,
        'morris_results': morris_results,
        'sobol_indices': sobol_indices,
        'temperature_sensitivity': temp_result,
        'aging_sensitivity': aging_result,
    }


if __name__ == "__main__":
    results = run_sensitivity_analysis()
