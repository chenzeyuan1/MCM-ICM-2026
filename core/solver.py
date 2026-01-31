"""
2026 MCM Problem A: 智能手机电池耗电建模
数值求解器 - SOC微分方程的数值解和time-to-empty计算

Author: MCM Team
Date: 2026-01
"""

import numpy as np
from typing import Callable, Tuple, List, Dict, Optional
from dataclasses import dataclass
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.battery import BatteryModel, BatteryParams, ThermalModel
from core.power_model import TotalPowerModel, NetworkMode, UsageScenarios


@dataclass
class SimulationResult:
    """仿真结果数据类"""
    time: np.ndarray           # 时间数组 (hours)
    soc: np.ndarray            # SOC数组 (0-1)
    power: np.ndarray          # 功耗数组 (W)
    temperature: np.ndarray    # 温度数组 (°C)
    voltage: np.ndarray        # 端电压数组 (V)
    current: np.ndarray        # 电流数组 (A)
    time_to_empty: float       # 电量耗尽时间 (hours)
    soc_final: float           # 最终SOC
    power_breakdown: Optional[Dict[str, np.ndarray]] = None


class BatterySolver:
    """
    电池SOC微分方程数值求解器
    
    核心方程:
    dSOC/dt = -P(t) / (V_OC(SOC) * C_eff(T)) - k_sd * SOC
    dT/dt = (Q_gen - Q_dissip) / C_thermal
    """
    
    def __init__(self, 
                 battery: Optional[BatteryModel] = None,
                 power_model: Optional[TotalPowerModel] = None,
                 thermal_model: Optional[ThermalModel] = None):
        """
        初始化求解器
        
        Args:
            battery: 电池模型
            power_model: 功耗模型
            thermal_model: 热模型 (可选)
        """
        self.battery = battery or BatteryModel()
        self.power_model = power_model or TotalPowerModel()
        self.thermal_model = thermal_model
    
    def _rk4_step(self, f: Callable, y: np.ndarray, t: float, dt: float, 
                  *args) -> np.ndarray:
        """
        四阶Runge-Kutta单步
        
        Args:
            f: 导数函数 dy/dt = f(y, t, *args)
            y: 当前状态
            t: 当前时间
            dt: 时间步长
            *args: 额外参数
            
        Returns:
            下一时刻的状态
        """
        k1 = f(y, t, *args)
        k2 = f(y + 0.5*dt*k1, t + 0.5*dt, *args)
        k3 = f(y + 0.5*dt*k2, t + 0.5*dt, *args)
        k4 = f(y + dt*k3, t + dt, *args)
        
        return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def _euler_step(self, f: Callable, y: np.ndarray, t: float, dt: float,
                    *args) -> np.ndarray:
        """
        欧拉法单步 (用于快速计算或对比)
        """
        return y + dt * f(y, t, *args)
    
    def _system_derivative(self, state: np.ndarray, t: float, 
                           power_func: Callable,
                           temp_ambient: float) -> np.ndarray:
        """
        系统状态导数 (SOC, T)
        
        Args:
            state: [SOC, T_battery]
            t: 时间
            power_func: 功耗函数 P(t)
            temp_ambient: 环境温度
            
        Returns:
            [dSOC/dt, dT/dt]
        """
        soc, temp_battery = state[0], state[1]
        
        # 获取当前功耗
        power = power_func(t)
        
        # SOC导数
        dsoc_dt = self.battery.soc_derivative(soc, power, temp_battery)
        
        # 温度导数 (如果有热模型)
        if self.thermal_model is not None:
            current = self.battery.current_from_power(power, soc, temp_battery)
            resistance = self.battery.internal_resistance(temp_battery, soc)
            dtemp_dt = self.thermal_model.temperature_derivative(
                temp_battery, temp_ambient, current, resistance
            )
        else:
            dtemp_dt = 0.0
        
        return np.array([dsoc_dt, dtemp_dt])
    
    def solve(self,
              soc_initial: float = 1.0,
              temp_initial: float = 25.0,
              temp_ambient: float = 25.0,
              power_func: Optional[Callable] = None,
              device_state: Optional[Dict] = None,
              duration: float = 24.0,
              dt: float = 0.01,
              soc_cutoff: float = 0.0,
              method: str = 'rk4') -> SimulationResult:
        """
        求解SOC随时间变化
        
        Args:
            soc_initial: 初始SOC (0-1)
            temp_initial: 初始电池温度 (°C)
            temp_ambient: 环境温度 (°C)
            power_func: 功耗函数 P(t), 如果为None则使用device_state
            device_state: 设备状态字典 (恒定功耗)
            duration: 仿真时长 (hours)
            dt: 时间步长 (hours)
            soc_cutoff: SOC截止值
            method: 求解方法 ('rk4' or 'euler')
            
        Returns:
            SimulationResult对象
        """
        # 设置功耗函数
        if power_func is None:
            if device_state is None:
                device_state = UsageScenarios.light_browsing()
            constant_power = self.power_model.power(device_state)
            power_func = lambda t: constant_power
        
        # 初始化
        n_steps = int(duration / dt) + 1
        time_array = np.linspace(0, duration, n_steps)
        
        soc_array = np.zeros(n_steps)
        temp_array = np.zeros(n_steps)
        power_array = np.zeros(n_steps)
        voltage_array = np.zeros(n_steps)
        current_array = np.zeros(n_steps)
        
        # 初始条件
        state = np.array([soc_initial, temp_initial])
        soc_array[0] = soc_initial
        temp_array[0] = temp_initial
        power_array[0] = power_func(0)
        voltage_array[0] = self.battery.terminal_voltage(
            soc_initial, 
            self.battery.current_from_power(power_array[0], soc_initial, temp_initial),
            temp_initial
        )
        current_array[0] = self.battery.current_from_power(
            power_array[0], soc_initial, temp_initial
        )
        
        # 选择步进方法
        step_func = self._rk4_step if method == 'rk4' else self._euler_step
        
        # 时间步进
        time_to_empty = duration  # 默认
        
        for i in range(1, n_steps):
            t = time_array[i-1]
            
            # 计算导数并步进
            state = step_func(
                lambda s, t, pf, ta: self._system_derivative(s, t, pf, ta),
                state, t, dt, power_func, temp_ambient
            )
            
            # 限制SOC范围
            state[0] = np.clip(state[0], 0.0, 1.0)
            
            # 记录状态
            soc_array[i] = state[0]
            temp_array[i] = state[1]
            power_array[i] = power_func(time_array[i])
            current_array[i] = self.battery.current_from_power(
                power_array[i], state[0], state[1]
            )
            voltage_array[i] = self.battery.terminal_voltage(
                state[0], current_array[i], state[1]
            )
            
            # 检查SOC截止
            if state[0] <= soc_cutoff:
                time_to_empty = time_array[i]
                # 截断数组
                time_array = time_array[:i+1]
                soc_array = soc_array[:i+1]
                temp_array = temp_array[:i+1]
                power_array = power_array[:i+1]
                voltage_array = voltage_array[:i+1]
                current_array = current_array[:i+1]
                break
        
        return SimulationResult(
            time=time_array,
            soc=soc_array,
            power=power_array,
            temperature=temp_array,
            voltage=voltage_array,
            current=current_array,
            time_to_empty=time_to_empty,
            soc_final=soc_array[-1]
        )
    
    def time_to_empty_analytical(self, 
                                  soc_initial: float,
                                  power_constant: float,
                                  temperature: float = 25.0) -> float:
        """
        恒定功耗下的解析解 (忽略自放电和非线性)
        
        对于简化模型: dSOC/dt = -P / (V_avg * C_eff)
        解为: SOC(t) = SOC_0 - P*t / (V_avg * C_eff)
        t_empty = SOC_0 * V_avg * C_eff / P
        
        Args:
            soc_initial: 初始SOC
            power_constant: 恒定功耗 (W)
            temperature: 温度 (°C)
            
        Returns:
            电量耗尽时间 (hours)
        """
        if power_constant <= 0:
            return float('inf')
        
        # 使用平均电压
        v_avg = self.battery.open_circuit_voltage(soc_initial / 2)
        
        # 有效容量 (转换为Wh)
        c_eff_wh = self.battery.effective_capacity(temperature) * v_avg / 1000.0
        
        return soc_initial * c_eff_wh / power_constant
    
    def scenario_comparison(self, 
                            scenarios: Dict[str, Dict],
                            soc_initial: float = 1.0,
                            temperature: float = 25.0) -> Dict[str, SimulationResult]:
        """
        对比多个使用场景
        
        Args:
            scenarios: 场景字典 {名称: 设备状态}
            soc_initial: 初始SOC
            temperature: 环境温度
            
        Returns:
            结果字典 {名称: SimulationResult}
        """
        results = {}
        
        for name, state in scenarios.items():
            result = self.solve(
                soc_initial=soc_initial,
                temp_ambient=temperature,
                device_state=state,
                duration=48.0,  # 足够长以耗尽电量
                soc_cutoff=0.0
            )
            results[name] = result
        
        return results


class MixedUsageSolver(BatterySolver):
    """
    混合使用场景求解器
    
    支持时变的使用模式
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_schedule = []
    
    def set_usage_schedule(self, schedule: List[Tuple[float, float, Dict]]):
        """
        设置使用计划
        
        Args:
            schedule: [(开始时间, 结束时间, 设备状态), ...]
        """
        self.usage_schedule = sorted(schedule, key=lambda x: x[0])
    
    def _get_state_at_time(self, t: float) -> Dict:
        """
        获取指定时刻的设备状态
        """
        for start, end, state in self.usage_schedule:
            if start <= t < end:
                return state
        # 默认待机
        return UsageScenarios.standby()
    
    def solve_mixed(self,
                    soc_initial: float = 1.0,
                    temp_initial: float = 25.0,
                    temp_ambient: float = 25.0,
                    duration: float = 24.0,
                    dt: float = 0.01) -> SimulationResult:
        """
        求解混合使用场景
        """
        def power_func(t):
            state = self._get_state_at_time(t)
            return self.power_model.power(state)
        
        return self.solve(
            soc_initial=soc_initial,
            temp_initial=temp_initial,
            temp_ambient=temp_ambient,
            power_func=power_func,
            duration=duration,
            dt=dt
        )
    
    def create_daily_schedule(self) -> List[Tuple[float, float, Dict]]:
        """
        创建典型日常使用计划 (24小时)
        
        Returns:
            使用计划列表
        """
        schedule = [
            # 0:00 - 7:00 睡眠 (待机)
            (0.0, 7.0, UsageScenarios.standby()),
            # 7:00 - 7:30 起床查看手机
            (7.0, 7.5, UsageScenarios.social_media()),
            # 7:30 - 8:30 通勤 (听音乐+导航)
            (7.5, 8.5, UsageScenarios.navigation()),
            # 8:30 - 12:00 工作 (偶尔看手机)
            (8.5, 12.0, UsageScenarios.light_browsing()),
            # 12:00 - 13:00 午餐 (视频)
            (12.0, 13.0, UsageScenarios.video_streaming()),
            # 13:00 - 17:30 工作
            (13.0, 17.5, UsageScenarios.light_browsing()),
            # 17:30 - 18:30 通勤 (导航)
            (17.5, 18.5, UsageScenarios.navigation()),
            # 18:30 - 19:30 晚餐
            (18.5, 19.5, UsageScenarios.social_media()),
            # 19:30 - 21:30 娱乐 (游戏)
            (19.5, 21.5, UsageScenarios.gaming()),
            # 21:30 - 23:00 休闲 (视频)
            (21.5, 23.0, UsageScenarios.video_streaming()),
            # 23:00 - 24:00 睡前
            (23.0, 24.0, UsageScenarios.light_browsing()),
        ]
        return schedule


class TimeToEmptyEstimator:
    """
    电量耗尽时间估算器
    
    提供多种估算方法和不确定性量化
    """
    
    def __init__(self, solver: BatterySolver):
        self.solver = solver
    
    def estimate_with_uncertainty(self,
                                   device_state: Dict,
                                   soc_initial: float = 1.0,
                                   temperature: float = 25.0,
                                   n_samples: int = 100) -> Dict:
        """
        Monte Carlo不确定性估计
        
        对关键参数添加随机扰动，统计time-to-empty分布
        
        Args:
            device_state: 设备状态
            soc_initial: 初始SOC
            temperature: 温度
            n_samples: 采样数
            
        Returns:
            包含均值、标准差、置信区间的字典
        """
        tte_samples = []
        
        base_power = self.solver.power_model.power(device_state)
        
        for _ in range(n_samples):
            # 添加参数扰动 (±10%)
            power_factor = np.random.normal(1.0, 0.05)
            capacity_factor = np.random.normal(1.0, 0.03)
            temp_variation = np.random.normal(0, 2)
            
            # 修改参数
            original_capacity = self.solver.battery.params.capacity_nominal
            self.solver.battery.params.capacity_nominal = original_capacity * capacity_factor
            
            # 计算
            tte = self.solver.time_to_empty_analytical(
                soc_initial,
                base_power * power_factor,
                temperature + temp_variation
            )
            tte_samples.append(tte)
            
            # 恢复参数
            self.solver.battery.params.capacity_nominal = original_capacity
        
        tte_samples = np.array(tte_samples)
        
        return {
            'mean': np.mean(tte_samples),
            'std': np.std(tte_samples),
            'median': np.median(tte_samples),
            'ci_95': (np.percentile(tte_samples, 2.5), np.percentile(tte_samples, 97.5)),
            'min': np.min(tte_samples),
            'max': np.max(tte_samples),
            'samples': tte_samples
        }
    
    def compare_scenarios(self, 
                          soc_initial: float = 1.0,
                          temperature: float = 25.0) -> Dict[str, float]:
        """
        对比不同场景的电量耗尽时间
        
        Returns:
            {场景名: 耗尽时间(hours)}
        """
        scenarios = {
            '待机': UsageScenarios.standby(),
            '轻度浏览': UsageScenarios.light_browsing(),
            '视频流': UsageScenarios.video_streaming(),
            '游戏': UsageScenarios.gaming(),
            '导航': UsageScenarios.navigation(),
            '通话': UsageScenarios.phone_call(),
            '社交媒体': UsageScenarios.social_media(),
        }
        
        results = {}
        for name, state in scenarios.items():
            power = self.solver.power_model.power(state)
            tte = self.solver.time_to_empty_analytical(soc_initial, power, temperature)
            results[name] = tte
        
        return results


if __name__ == "__main__":
    # 测试求解器
    print("=== 电池SOC求解器测试 ===\n")
    
    # 创建求解器
    solver = BatterySolver()
    
    # 测试单一场景
    print("【轻度浏览场景仿真】")
    result = solver.solve(
        soc_initial=1.0,
        device_state=UsageScenarios.light_browsing(),
        duration=24.0
    )
    print(f"初始SOC: 100%")
    print(f"电量耗尽时间: {result.time_to_empty:.2f} 小时")
    print(f"最终SOC: {result.soc_final*100:.1f}%")
    print()
    
    # 对比场景
    print("【场景对比】")
    estimator = TimeToEmptyEstimator(solver)
    comparison = estimator.compare_scenarios()
    
    for name, tte in sorted(comparison.items(), key=lambda x: -x[1]):
        print(f"  {name}: {tte:.2f} 小时")
    print()
    
    # 测试混合使用
    print("【典型一天使用仿真】")
    mixed_solver = MixedUsageSolver()
    schedule = mixed_solver.create_daily_schedule()
    mixed_solver.set_usage_schedule(schedule)
    
    result = mixed_solver.solve_mixed(
        soc_initial=1.0,
        duration=24.0
    )
    print(f"24小时后剩余电量: {result.soc_final*100:.1f}%")
    if result.time_to_empty < 24.0:
        print(f"电量耗尽时间: {result.time_to_empty:.2f} 小时")
    print()
    
    # 不确定性分析
    print("【不确定性分析 - 游戏场景】")
    uncertainty = estimator.estimate_with_uncertainty(
        UsageScenarios.gaming(),
        n_samples=500
    )
    print(f"平均耗尽时间: {uncertainty['mean']:.2f} ± {uncertainty['std']:.2f} 小时")
    print(f"95%置信区间: [{uncertainty['ci_95'][0]:.2f}, {uncertainty['ci_95'][1]:.2f}] 小时")
