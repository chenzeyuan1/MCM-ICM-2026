"""
2026 MCM Problem A: 智能手机电池耗电建模
功耗模型 - 各组件功耗的模块化建模

Author: MCM Team
Date: 2026-01
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from enum import Enum


class NetworkMode(Enum):
    """网络模式枚举"""
    OFF = 0
    WIFI = 1
    LTE_4G = 2
    NR_5G = 3


class ScreenType(Enum):
    """屏幕类型枚举"""
    LCD = 1
    OLED = 2


@dataclass
class ScreenParams:
    """
    屏幕参数
    
    来源:
    - Carroll & Heiser (2010) "An Analysis of Power Consumption in a Smartphone"
    - DisplayMate测试报告
    - 典型6.5寸OLED屏幕规格
    """
    screen_type: ScreenType = ScreenType.OLED
    size_inches: float = 6.5           # 屏幕尺寸 (英寸)
    resolution: tuple = (1080, 2400)   # 分辨率
    power_max: float = 2.5             # 最大功耗 (W) - 来源: 6.5寸OLED实测1.5-3W
    power_min: float = 0.3             # 最低亮度功耗 (W) - 来源: 实测0.2-0.4W
    power_off: float = 0.005           # 屏幕关闭功耗 (W)
    brightness_gamma: float = 1.1      # 亮度-功耗指数 - LCD近似线性


@dataclass
class ProcessorParams:
    """
    处理器参数
    
    来源:
    - Qualcomm Snapdragon 8 Gen 2 技术规格
    - ARM Cortex-A78 技术手册
    - AnandTech处理器功耗测试
    """
    power_idle: float = 0.1            # 空闲功耗 (W) - 来源: ARM规格50-150mW
    power_max: float = 4.0             # 满载功耗 (W) - 来源: 骁龙8系TDP 5-8W，持续3-5W
    power_sleep: float = 0.01          # 深度睡眠功耗 (W) - 来源: SoC低功耗模式5-20mW
    cores: int = 8                     # 核心数


@dataclass  
class NetworkParams:
    """
    网络模块参数
    
    来源:
    - IEEE 802.11 WiFi标准
    - Huang et al. (2012) "A Close Examination of 4G LTE Networks"
    - Qualcomm 5G白皮书
    - 3GPP技术规范
    """
    # WiFi - 来源: IEEE 802.11规范
    wifi_idle: float = 0.015           # WiFi空闲功耗 (W)
    wifi_rx: float = 0.2               # WiFi接收功耗 (W) - 实测100-300mW
    wifi_tx: float = 0.4               # WiFi发送功耗 (W) - 实测200-500mW
    
    # 4G LTE - 来源: Huang et al. (2012)
    lte_idle: float = 0.04             # LTE空闲功耗 (W)
    lte_rx: float = 0.6                # LTE接收功耗 (W) - 实测500-800mW
    lte_tx: float = 1.2                # LTE发送功耗 (W) - 实测800-1500mW
    
    # 5G NR - 来源: 高通5G白皮书，约为4G的2-3倍
    nr_idle: float = 0.08              # 5G空闲功耗 (W)
    nr_rx: float = 1.5                 # 5G接收功耗 (W)
    nr_tx: float = 3.0                 # 5G发送功耗 (W)
    
    # 信号强度影响系数 - 来源: 经验值，弱信号需要更高发射功率
    signal_power_factor: float = 2.0   # 信号弱时功耗放大倍数


@dataclass
class SensorParams:
    """传感器参数"""
    gps_active: float = 0.15           # GPS活跃功耗 (W)
    gps_idle: float = 0.01             # GPS待机功耗 (W)
    bluetooth_active: float = 0.05     # 蓝牙活跃功耗 (W)
    bluetooth_idle: float = 0.005      # 蓝牙待机功耗 (W)
    accelerometer: float = 0.005       # 加速度计功耗 (W)
    gyroscope: float = 0.005           # 陀螺仪功耗 (W)
    ambient_light: float = 0.001       # 环境光传感器功耗 (W)


class ScreenPowerModel:
    """
    屏幕功耗模型
    
    LCD: P_screen = P_backlight(B) + P_driver ≈ P_max * (B/B_max)^γ
    OLED: P_screen = Σ(k_R*R + k_G*G + k_B*B) ≈ P_max * B * content_factor
    """
    
    def __init__(self, params: Optional[ScreenParams] = None):
        self.params = params or ScreenParams()
    
    def power(self, brightness: float, is_on: bool = True, 
              content_brightness: float = 0.5) -> float:
        """
        计算屏幕功耗
        
        Args:
            brightness: 亮度 (0-1)
            is_on: 屏幕是否开启
            content_brightness: 显示内容平均亮度 (0-1, 仅OLED)
            
        Returns:
            功耗 (W)
        """
        p = self.params
        
        if not is_on:
            return p.power_off
        
        brightness = np.clip(brightness, 0.0, 1.0)
        
        if p.screen_type == ScreenType.LCD:
            # LCD: 背光主导，与亮度近似线性
            return p.power_min + (p.power_max - p.power_min) * (brightness ** p.brightness_gamma)
        else:
            # OLED: 与亮度和内容相关
            base_power = p.power_min + (p.power_max - p.power_min) * brightness
            content_factor = 0.3 + 0.7 * content_brightness  # 暗色内容省电
            return base_power * content_factor


class ProcessorPowerModel:
    """
    处理器功耗模型
    
    基于CMOS动态功耗: P = α * C * V² * f
    
    由于DVFS (动态电压频率调节) 中 V ∝ f，因此:
    P_dynamic ∝ V² * f ∝ f³
    
    完整模型: P_CPU = P_idle + (P_max - P_idle) * U * (f/f_max)³
    
    其中:
    - U: CPU利用率 (0-1)
    - f/f_max: 频率比例 (0-1)
    """
    
    def __init__(self, params: Optional[ProcessorParams] = None):
        self.params = params or ProcessorParams()
    
    def power(self, utilization: float, freq_ratio: float = 1.0, 
              is_sleeping: bool = False) -> float:
        """
        计算处理器功耗 (考虑频率影响)
        
        P_CPU = P_idle + (P_max - P_idle) * U * (f/f_max)^3
        
        Args:
            utilization: CPU利用率 (0-1)
            freq_ratio: 频率比例 (0-1, 相对最大频率), 默认1.0
            is_sleeping: 是否深度睡眠
            
        Returns:
            功耗 (W)
        """
        p = self.params
        
        if is_sleeping:
            return p.power_sleep
        
        utilization = np.clip(utilization, 0.0, 1.0)
        freq_ratio = np.clip(freq_ratio, 0.1, 1.0)
        
        # 频率的立方关系 (DVFS: V ∝ f, P ∝ V²f ∝ f³)
        freq_factor = freq_ratio ** 3
        
        # 动态功耗 = 基础功耗 + 负载功耗 * 利用率 * 频率因子
        return p.power_idle + (p.power_max - p.power_idle) * utilization * freq_factor
    
    def power_simple(self, utilization: float, is_sleeping: bool = False) -> float:
        """
        简化功耗模型 (不考虑频率，假设满频)
        """
        return self.power(utilization, freq_ratio=1.0, is_sleeping=is_sleeping)


class NetworkPowerModel:
    """
    网络模块功耗模型
    
    P_network = P_base + P_activity * duty_cycle * signal_factor
    
    信号强度影响: signal_factor = (RSSI_ref / RSSI)^n
    """
    
    def __init__(self, params: Optional[NetworkParams] = None):
        self.params = params or NetworkParams()
    
    def power(self, mode: NetworkMode, 
              activity_level: float = 0.0,
              is_transmitting: bool = False,
              signal_strength: float = 1.0) -> float:
        """
        计算网络功耗
        
        Args:
            mode: 网络模式 (OFF/WIFI/LTE/5G)
            activity_level: 网络活动水平 (0-1)
            is_transmitting: 是否发送数据 (否则为接收)
            signal_strength: 信号强度 (0-1, 1为满格)
            
        Returns:
            功耗 (W)
        """
        p = self.params
        
        if mode == NetworkMode.OFF:
            return 0.0
        
        activity_level = np.clip(activity_level, 0.0, 1.0)
        signal_strength = np.clip(signal_strength, 0.1, 1.0)
        
        # 根据模式选择参数
        if mode == NetworkMode.WIFI:
            p_idle = p.wifi_idle
            p_active = p.wifi_tx if is_transmitting else p.wifi_rx
        elif mode == NetworkMode.LTE_4G:
            p_idle = p.lte_idle
            p_active = p.lte_tx if is_transmitting else p.lte_rx
        else:  # 5G
            p_idle = p.nr_idle
            p_active = p.nr_tx if is_transmitting else p.nr_rx
        
        # 信号强度修正 (弱信号功耗增加)
        signal_factor = 1 + (p.signal_power_factor - 1) * (1 - signal_strength)
        
        # 总功耗 = 空闲功耗 + 活动功耗 * 活动比例 * 信号因子
        return p_idle + (p_active - p_idle) * activity_level * signal_factor


class SensorPowerModel:
    """
    传感器功耗模型
    
    P_sensors = Σ P_sensor_i * activity_i
    """
    
    def __init__(self, params: Optional[SensorParams] = None):
        self.params = params or SensorParams()
    
    def power(self, 
              gps_active: bool = False,
              bluetooth_active: bool = False,
              motion_sensors: bool = False) -> float:
        """
        计算传感器总功耗
        
        Args:
            gps_active: GPS是否活跃
            bluetooth_active: 蓝牙是否活跃  
            motion_sensors: 运动传感器是否活跃
            
        Returns:
            功耗 (W)
        """
        p = self.params
        total = 0.0
        
        # GPS
        total += p.gps_active if gps_active else p.gps_idle
        
        # 蓝牙
        total += p.bluetooth_active if bluetooth_active else p.bluetooth_idle
        
        # 运动传感器
        if motion_sensors:
            total += p.accelerometer + p.gyroscope
        
        # 环境光传感器 (始终开启)
        total += p.ambient_light
        
        return total


class BackgroundPowerModel:
    """
    后台任务功耗模型
    
    P_background = Σ (P_wake_i * duty_cycle_i + P_base_i)
    """
    
    def __init__(self):
        # 典型后台任务功耗 (W)
        self.tasks = {
            'email_sync': {'wake': 0.3, 'base': 0.005, 'default_duty': 0.01},
            'social_media': {'wake': 0.2, 'base': 0.01, 'default_duty': 0.02},
            'location_service': {'wake': 0.15, 'base': 0.005, 'default_duty': 0.05},
            'music_playback': {'wake': 0.15, 'base': 0.1, 'default_duty': 1.0},
            'voice_assistant': {'wake': 0.3, 'base': 0.02, 'default_duty': 0.01},
            'system_services': {'wake': 0.1, 'base': 0.02, 'default_duty': 0.03},
        }
    
    def power(self, active_tasks: Optional[Dict[str, float]] = None) -> float:
        """
        计算后台任务总功耗
        
        Args:
            active_tasks: 活跃任务字典 {任务名: 占空比}
            
        Returns:
            功耗 (W)
        """
        if active_tasks is None:
            # 默认后台任务
            active_tasks = {
                'system_services': 0.03,
                'email_sync': 0.01,
            }
        
        total = 0.0
        for task_name, duty_cycle in active_tasks.items():
            if task_name in self.tasks:
                task = self.tasks[task_name]
                total += task['wake'] * duty_cycle + task['base']
        
        return total


class TotalPowerModel:
    """
    总功耗模型
    
    P_total(t) = P_screen + P_cpu + P_network + P_sensors + P_background + P_misc
    """
    
    def __init__(self):
        self.screen = ScreenPowerModel()
        self.processor = ProcessorPowerModel()
        self.network = NetworkPowerModel()
        self.sensors = SensorPowerModel()
        self.background = BackgroundPowerModel()
        
        # 其他固定功耗 (音频、触控等)
        self.misc_power = 0.05  # W
    
    def power(self, state: Dict) -> float:
        """
        计算总功耗
        
        Args:
            state: 设备状态字典，包含:
                - screen_on: bool
                - brightness: float (0-1)
                - cpu_utilization: float (0-1)
                - network_mode: NetworkMode
                - network_activity: float (0-1)
                - gps_active: bool
                - bluetooth_active: bool
                - background_tasks: Dict[str, float]
                
        Returns:
            总功耗 (W)
        """
        # 屏幕功耗
        p_screen = self.screen.power(
            brightness=state.get('brightness', 0.5),
            is_on=state.get('screen_on', True),
            content_brightness=state.get('content_brightness', 0.5)
        )
        
        # 处理器功耗 (考虑频率)
        p_cpu = self.processor.power(
            utilization=state.get('cpu_utilization', 0.1),
            freq_ratio=state.get('cpu_freq_ratio', 1.0),
            is_sleeping=state.get('is_sleeping', False)
        )
        
        # 网络功耗
        p_network = self.network.power(
            mode=state.get('network_mode', NetworkMode.WIFI),
            activity_level=state.get('network_activity', 0.1),
            is_transmitting=state.get('is_transmitting', False),
            signal_strength=state.get('signal_strength', 0.8)
        )
        
        # 传感器功耗
        p_sensors = self.sensors.power(
            gps_active=state.get('gps_active', False),
            bluetooth_active=state.get('bluetooth_active', False),
            motion_sensors=state.get('motion_sensors', False)
        )
        
        # 后台功耗
        p_background = self.background.power(
            active_tasks=state.get('background_tasks', None)
        )
        
        # 总功耗
        return p_screen + p_cpu + p_network + p_sensors + p_background + self.misc_power
    
    def power_breakdown(self, state: Dict) -> Dict[str, float]:
        """
        获取功耗分解
        
        Returns:
            各组件功耗字典
        """
        p_screen = self.screen.power(
            brightness=state.get('brightness', 0.5),
            is_on=state.get('screen_on', True),
            content_brightness=state.get('content_brightness', 0.5)
        )
        
        p_cpu = self.processor.power(
            utilization=state.get('cpu_utilization', 0.1),
            freq_ratio=state.get('cpu_freq_ratio', 1.0),
            is_sleeping=state.get('is_sleeping', False)
        )
        
        p_network = self.network.power(
            mode=state.get('network_mode', NetworkMode.WIFI),
            activity_level=state.get('network_activity', 0.1),
            is_transmitting=state.get('is_transmitting', False),
            signal_strength=state.get('signal_strength', 0.8)
        )
        
        p_sensors = self.sensors.power(
            gps_active=state.get('gps_active', False),
            bluetooth_active=state.get('bluetooth_active', False),
            motion_sensors=state.get('motion_sensors', False)
        )
        
        p_background = self.background.power(
            active_tasks=state.get('background_tasks', None)
        )
        
        return {
            'screen': p_screen,
            'processor': p_cpu,
            'network': p_network,
            'sensors': p_sensors,
            'background': p_background,
            'misc': self.misc_power,
            'total': p_screen + p_cpu + p_network + p_sensors + p_background + self.misc_power
        }


# 预定义使用场景
class UsageScenarios:
    """预定义的使用场景"""
    
    @staticmethod
    def standby() -> Dict:
        """待机状态"""
        return {
            'screen_on': False,
            'brightness': 0.0,
            'cpu_utilization': 0.02,
            'cpu_freq_ratio': 0.3,  # 低频省电
            'is_sleeping': False,
            'network_mode': NetworkMode.WIFI,
            'network_activity': 0.01,
            'gps_active': False,
            'bluetooth_active': False,
            'background_tasks': {'system_services': 0.02}
        }
    
    @staticmethod
    def light_browsing() -> Dict:
        """轻度浏览"""
        return {
            'screen_on': True,
            'brightness': 0.4,
            'content_brightness': 0.6,
            'cpu_utilization': 0.15,
            'cpu_freq_ratio': 0.5,  # 中等频率
            'network_mode': NetworkMode.WIFI,
            'network_activity': 0.2,
            'gps_active': False,
            'bluetooth_active': False,
            'background_tasks': {'system_services': 0.03, 'email_sync': 0.01}
        }
    
    @staticmethod
    def video_streaming() -> Dict:
        """视频流"""
        return {
            'screen_on': True,
            'brightness': 0.6,
            'content_brightness': 0.5,
            'cpu_utilization': 0.35,
            'cpu_freq_ratio': 0.6,  # 视频解码需要中高频率
            'network_mode': NetworkMode.WIFI,
            'network_activity': 0.6,
            'gps_active': False,
            'bluetooth_active': False,
            'background_tasks': {'system_services': 0.02}
        }
    
    @staticmethod
    def gaming() -> Dict:
        """游戏"""
        return {
            'screen_on': True,
            'brightness': 0.8,
            'content_brightness': 0.7,
            'cpu_utilization': 0.85,
            'cpu_freq_ratio': 1.0,  # 游戏需要满频运行
            'network_mode': NetworkMode.WIFI,
            'network_activity': 0.3,
            'gps_active': False,
            'bluetooth_active': True,
            'motion_sensors': True,
            'background_tasks': {'system_services': 0.02}
        }
    
    @staticmethod
    def navigation() -> Dict:
        """导航"""
        return {
            'screen_on': True,
            'brightness': 0.9,
            'content_brightness': 0.4,
            'cpu_utilization': 0.4,
            'cpu_freq_ratio': 0.7,  # 导航需要较高频率
            'network_mode': NetworkMode.LTE_4G,
            'network_activity': 0.3,
            'gps_active': True,
            'bluetooth_active': True,
            'background_tasks': {'system_services': 0.03, 'location_service': 0.8}
        }
    
    @staticmethod
    def phone_call() -> Dict:
        """通话"""
        return {
            'screen_on': False,
            'brightness': 0.0,
            'cpu_utilization': 0.2,
            'cpu_freq_ratio': 0.4,  # 语音处理不需要高频
            'network_mode': NetworkMode.LTE_4G,
            'network_activity': 0.5,
            'is_transmitting': True,
            'gps_active': False,
            'bluetooth_active': False,
            'background_tasks': {'system_services': 0.02}
        }
    
    @staticmethod
    def social_media() -> Dict:
        """社交媒体"""
        return {
            'screen_on': True,
            'brightness': 0.5,
            'content_brightness': 0.5,
            'cpu_utilization': 0.25,
            'cpu_freq_ratio': 0.6,  # 滑动和图片加载需要中高频
            'network_mode': NetworkMode.LTE_4G,
            'network_activity': 0.4,
            'gps_active': False,
            'bluetooth_active': False,
            'background_tasks': {'system_services': 0.03, 'social_media': 0.1}
        }


if __name__ == "__main__":
    # 测试功耗模型
    power_model = TotalPowerModel()
    
    print("=== 功耗模型测试 ===\n")
    
    scenarios = [
        ("待机", UsageScenarios.standby()),
        ("轻度浏览", UsageScenarios.light_browsing()),
        ("视频流媒体", UsageScenarios.video_streaming()),
        ("游戏", UsageScenarios.gaming()),
        ("导航", UsageScenarios.navigation()),
        ("通话", UsageScenarios.phone_call()),
        ("社交媒体", UsageScenarios.social_media()),
    ]
    
    for name, scenario in scenarios:
        breakdown = power_model.power_breakdown(scenario)
        print(f"【{name}】")
        print(f"  屏幕: {breakdown['screen']:.3f}W")
        print(f"  处理器: {breakdown['processor']:.3f}W")
        print(f"  网络: {breakdown['network']:.3f}W")
        print(f"  传感器: {breakdown['sensors']:.3f}W")
        print(f"  后台: {breakdown['background']:.3f}W")
        print(f"  总功耗: {breakdown['total']:.3f}W")
        print()
