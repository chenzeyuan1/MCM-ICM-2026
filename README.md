# 2026 MCM Problem A: 智能手机电池耗电建模

## 项目结构

```
battery_model/
├── core/                    # 核心模型
│   ├── battery.py          # 电池物理模型 (SOC, OCV, 温度/老化效应)
│   ├── power_model.py      # 功耗子模型 (屏幕, CPU, 网络, 传感器)
│   └── solver.py           # 数值求解器 (RK4, 解析解)
├── analysis/
│   └── sensitivity.py      # 敏感度分析 (OAT, Morris, Sobol)
├── visualization/
│   └── plots.py            # 可视化工具
├── figures/                 # 生成的图表 (运行后创建)
├── main.py                 # 主程序入口
├── REPORT.md               # 完整建模报告
└── README.md               # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install numpy matplotlib
```

### 2. 运行主程序

```bash
cd battery_model
python main.py
```

这将执行完整的仿真分析并在 `figures/` 目录生成可视化图表。

### 3. 单独测试各模块

```bash
# 测试电池模型
python core/battery.py

# 测试功耗模型
python core/power_model.py

# 测试求解器
python core/solver.py

# 运行敏感度分析
python analysis/sensitivity.py
```

## 核心模型

### SOC动力学方程

$$\frac{dSOC}{dt} = -\frac{P_{total}(t)}{V_{OC}(SOC) \cdot C_{eff}(T)} - k_{sd} \cdot SOC$$

### 功耗分解

$$P_{total} = P_{screen} + P_{CPU} + P_{network} + P_{sensors} + P_{background}$$

## 使用场景

预定义了7种典型使用场景：
- 待机 (Standby)
- 轻度浏览 (Light Browsing)
- 视频流 (Video Streaming)
- 游戏 (Gaming)
- 导航 (Navigation)
- 通话 (Phone Call)
- 社交媒体 (Social Media)

## 输出结果

运行 `main.py` 将生成：
1. 控制台输出：各场景功耗、续航时间、敏感度分析结果
2. 图表文件（保存在 `figures/`）：
   - OCV-SOC曲线
   - 温度/老化影响
   - 功耗分解饼图
   - SOC放电曲线
   - 敏感度分析图
   - 不确定性分布图

## 作者

MCM Team - 2026年1月
