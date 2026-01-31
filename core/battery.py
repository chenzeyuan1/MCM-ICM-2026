"""
2026 MCM Problem A: 智能手机电池耗电建模
核心电池模型 - 基于锂离子电池物理特性的连续时间模型

Author: MCM Team
Date: 2026-01
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple


@dataclass
class BatteryParams:
    """锂离子电池参数类"""
    
    # 基本参数
    capacity_nominal: float = 4000.0  # 标称容量 (mAh)
    voltage_nominal: float = 3.85     # 标称电压 (V)
    voltage_max: float = 4.35         # 充满电压 (V)
    voltage_min: float = 3.0          # 截止电压 (V)
    
    # 内阻参数
    resistance_internal: float = 0.08  # 内阻 (Ω) @ 25°C
    resistance_temp_coeff: float = -0.015  # 内阻温度系数 (/°C)
    
    # 温度参数
    temp_ref: float = 25.0            # 参考温度 (°C)
    capacity_temp_coeff: float = 0.007  # 容量温度系数 (/°C)
    
    # 老化参数
    cycle_count: int = 0              # 循环次数
    aging_coeff: float = 0.0003       # 老化系数
    
    # 自放电参数
    self_discharge_rate: float = 0.0001  # 自放电率 (/hour)
    
    # Peukert参数
    peukert_constant: float = 1.05    # Peukert常数 (锂离子电池)
    
    # OCV-SOC多项式系数 (经验拟合)
    ocv_coefficients: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 0.8, 0.4, -0.15])
    )


class BatteryModel:
    """
    智能手机电池连续时间模型
    
    核心方程:
    dSOC/dt = -P_total(t) / (V_OC(SOC) * C_n(T)) - k_sd * SOC
    
    其中:
    - SOC: 荷电状态 (0-1)
    - P_total: 总功耗 (W)
    - V_OC: 开路电压 (V)
    - C_n: 有效容量 (Wh)
    - k_sd: 自放电率
    """
    
    def __init__(self, params: Optional[BatteryParams] = None):
        """初始化电池模型"""
        self.params = params or BatteryParams()
        self._validate_params()
    
    def _validate_params(self):
        """验证参数合理性"""
        p = self.params
        assert 0 < p.capacity_nominal <= 10000, "容量应在合理范围内"
        assert 2.5 < p.voltage_min < p.voltage_max < 5.0, "电压范围不合理"
        assert 1.0 <= p.peukert_constant <= 1.5, "Peukert常数超出范围"
    
    def open_circuit_voltage(self, soc: float) -> float:
        """
        计算开路电压 V_OC(SOC)
        
        使用多项式模型: V_OC = a0 + a1*SOC + a2*SOC^2 + a3*SOC^3
        
        Args:
            soc: 荷电状态 (0-1)
            
        Returns:
            开路电压 (V)
        """
        soc = np.clip(soc, 0.0, 1.0)
        coeffs = self.params.ocv_coefficients
        
        # 多项式计算
        v_oc = sum(c * (soc ** i) for i, c in enumerate(coeffs))
        
        # 确保在合理范围内
        return np.clip(v_oc, self.params.voltage_min, self.params.voltage_max)
    
    def effective_capacity(self, temperature: float) -> float:
        """
        计算有效容量 (考虑温度和老化)
        
        C_eff = C_nominal * f(T) * g(n)
        
        温度修正: f(T) = 1 - α_T * (T_ref - T) for T < T_ref
        老化修正: g(n) = 1 - k_deg * sqrt(n)
        
        Args:
            temperature: 环境温度 (°C)
            
        Returns:
            有效容量 (mAh)
        """
        p = self.params
        
        # 温度修正
        if temperature < p.temp_ref:
            temp_factor = 1 - p.capacity_temp_coeff * (p.temp_ref - temperature)
        else:
            temp_factor = 1.0
        temp_factor = max(0.5, temp_factor)  # 至少保留50%容量
        
        # 老化修正 (平方根模型)
        aging_factor = 1 - p.aging_coeff * np.sqrt(p.cycle_count)
        aging_factor = max(0.7, aging_factor)  # 至少保留70%容量
        
        return p.capacity_nominal * temp_factor * aging_factor
    
    def internal_resistance(self, temperature: float, soc: float) -> float:
        """
        计算内阻 (考虑温度和SOC)
        
        R(T, SOC) = R_ref * [1 + β*(T_ref - T)] * [1 + γ*(1 - SOC)]
        
        Args:
            temperature: 温度 (°C)
            soc: 荷电状态 (0-1)
            
        Returns:
            内阻 (Ω)
        """
        p = self.params
        
        # 温度修正 (低温时内阻增加)
        temp_factor = 1 + p.resistance_temp_coeff * (p.temp_ref - temperature)
        temp_factor = max(0.8, temp_factor)
        
        # SOC修正 (低SOC时内阻略增)
        soc_factor = 1 + 0.2 * (1 - soc)
        
        return p.resistance_internal * temp_factor * soc_factor
    
    def terminal_voltage(self, soc: float, current: float, temperature: float) -> float:
        """
        计算端电压
        
        V_terminal = V_OC(SOC) - I * R_internal(T, SOC)
        
        Args:
            soc: 荷电状态 (0-1)
            current: 放电电流 (A), 正值表示放电
            temperature: 温度 (°C)
            
        Returns:
            端电压 (V)
        """
        v_oc = self.open_circuit_voltage(soc)
        r_int = self.internal_resistance(temperature, soc)
        
        return v_oc - current * r_int
    
    def soc_derivative(self, soc: float, power: float, temperature: float) -> float:
        """
        计算SOC变化率 (核心微分方程)
        
        dSOC/dt = -P / (V_OC * C_eff) - k_sd * SOC
        
        Args:
            soc: 当前SOC (0-1)
            power: 功耗 (W)
            temperature: 温度 (°C)
            
        Returns:
            SOC变化率 (/hour)
        """
        p = self.params
        
        # 获取开路电压
        v_oc = self.open_circuit_voltage(soc)
        
        # 获取有效容量 (转换为Wh)
        c_eff_wh = self.effective_capacity(temperature) * v_oc / 1000.0
        
        # 功耗引起的SOC下降
        power_drain = -power / c_eff_wh if c_eff_wh > 0 else 0
        
        # 自放电
        self_discharge = -p.self_discharge_rate * soc
        
        return power_drain + self_discharge
    
    def current_from_power(self, power: float, soc: float, temperature: float) -> float:
        """
        根据功耗计算放电电流
        
        P = V_terminal * I = (V_OC - I*R) * I
        解方程: I = (V_OC - sqrt(V_OC^2 - 4*R*P)) / (2*R)
        
        Args:
            power: 功耗 (W)
            soc: SOC (0-1)
            temperature: 温度 (°C)
            
        Returns:
            电流 (A)
        """
        v_oc = self.open_circuit_voltage(soc)
        r_int = self.internal_resistance(temperature, soc)
        
        discriminant = v_oc**2 - 4 * r_int * power
        
        if discriminant < 0:
            # 功耗过大，电池无法提供
            return power / v_oc  # 近似值
        
        return (v_oc - np.sqrt(discriminant)) / (2 * r_int)
    
    def peukert_capacity(self, discharge_current: float) -> float:
        """
        Peukert容量修正
        
        C_peukert = C_nominal * (C_nominal / (I * H))^(k-1)
        
        Args:
            discharge_current: 放电电流 (A)
            
        Returns:
            Peukert修正后的容量 (mAh)
        """
        p = self.params
        
        if discharge_current <= 0:
            return p.capacity_nominal
        
        # 1C电流
        i_1c = p.capacity_nominal / 1000.0  # A
        
        # Peukert修正
        ratio = i_1c / discharge_current
        peukert_factor = ratio ** (p.peukert_constant - 1)
        
        return p.capacity_nominal * min(peukert_factor, 1.5)


class ThermalModel:
    """
    电池热模型
    
    dT/dt = (Q_gen - Q_dissip) / (m * c_p)
    """
    
    def __init__(self, 
                 thermal_mass: float = 50.0,  # J/K
                 heat_transfer_coeff: float = 5.0,  # W/(m²·K)
                 surface_area: float = 0.01):  # m²
        self.thermal_mass = thermal_mass
        self.h = heat_transfer_coeff
        self.area = surface_area
    
    def heat_generation(self, current: float, resistance: float) -> float:
        """
        计算热生成率 (焦耳热)
        
        Q_gen = I² * R
        """
        return current ** 2 * resistance
    
    def heat_dissipation(self, temp_battery: float, temp_ambient: float) -> float:
        """
        计算散热率
        
        Q_dissip = h * A * (T_battery - T_ambient)
        """
        return self.h * self.area * (temp_battery - temp_ambient)
    
    def temperature_derivative(self, temp_battery: float, temp_ambient: float,
                                current: float, resistance: float) -> float:
        """
        计算温度变化率
        
        dT/dt = (Q_gen - Q_dissip) / C_thermal
        """
        q_gen = self.heat_generation(current, resistance)
        q_diss = self.heat_dissipation(temp_battery, temp_ambient)
        
        return (q_gen - q_diss) / self.thermal_mass * 3600  # 转换为 °C/hour


if __name__ == "__main__":
    # 简单测试
    battery = BatteryModel()
    
    print("=== 电池模型测试 ===")
    print(f"标称容量: {battery.params.capacity_nominal} mAh")
    
    # 测试OCV-SOC曲线
    for soc in [1.0, 0.8, 0.5, 0.2, 0.0]:
        v_oc = battery.open_circuit_voltage(soc)
        print(f"SOC={soc:.1f}: V_OC={v_oc:.3f}V")
    
    # 测试有效容量
    for temp in [-10, 0, 25, 40]:
        c_eff = battery.effective_capacity(temp)
        print(f"T={temp}°C: C_eff={c_eff:.1f}mAh")
