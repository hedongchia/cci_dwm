"""
海洋边界层方程求解器模块 (v1.3)
功能：
1. 求解一维湍流混合方程组
2. 处理表面边界条件(动量/热量/盐度通量)
3. 实现隐式时间积分方案
4. 支持多种湍流闭合方案
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple
from dataclasses import dataclass
from config import Config
from metpy.units import units
from scipy.sparse import diags, linalg


@dataclass
class ModelState:
    """模型状态容器"""
    temperature: np.ndarray  # 温度剖面 (℃)
    salinity: np.ndarray  # 盐度剖面 (PSU)
    u: np.ndarray  # 东向流速 (m/s)
    v: np.ndarray  # 北向流速 (m/s)
    tke: np.ndarray  # 湍流动能 (m²/s²)
    ml_depth: float  # 混合层深度 (m)


class BoundaryConditions:
    """边界条件容器"""

    def __init__(self):
        self.surface = {
            'wind_stress': 0.0,  # 表面风应力 (N/m²)
            'heat_flux': 0.0,  # 表面热通量 (W/m²)
            'fw_flux': 0.0  # 淡水通量 (kg/m²/s)
        }
        self.bottom = {
            'friction': 0.01  # 底摩擦系数
        }


class OceanSolver:
    def __init__(self, cfg: Config):
        """
        初始化求解器
        参数:
            cfg : 配置参数对象
                 必需参数:
                 - dt (时间步长, 秒)
                 - max_depth (最大深度, m)
                 - n_levels (垂直层数)
                 - scheme (时间积分方案)
        """
        self.cfg = cfg
        self.set_grid()
        self.set_parameters()
        self.bc = BoundaryConditions()

    def set_grid(self):
        """设置垂直网格"""
        self.n_levels = getattr(self.cfg, 'n_levels', 50)
        self.max_depth = getattr(self.cfg, 'max_depth', 200) * units('m')

        # 采用拉伸网格 (表面分辨率高)
        self.depth = np.linspace(0, self.max_depth.magnitude, self.n_levels)
        self.dz = np.diff(self.depth)
        self.dz = np.append(self.dz, self.dz[-1])  # 最底层厚度相同

    def set_parameters(self):
        """设置物理参数"""
        self.dt = getattr(self.cfg, 'dt', 600) * units('s')
        self.scheme = getattr(self.cfg, 'scheme', 'implicit')

        # 数值参数
        self.alpha = 0.6  # 隐式权重系数
        self.min_diffusivity = 1e-6 * units('m²/s')

    def initialize(self, init_state: Dict):
        """初始化模型状态"""
        self.state = ModelState(
            temperature=init_state['temp'],
            salinity=init_state['salt'],
            u=init_state['u'],
            v=init_state['v'],
            tke=init_state.get('tke', np.zeros_like(self.depth)),
            ml_depth=init_state.get('ml_depth', 10.0)
        )

    def set_boundary_conditions(self, fluxes: Dict):
        """更新边界条件"""
        self.bc.surface['wind_stress'] = fluxes.get('tau', 0.0)
        self.bc.surface['heat_flux'] = fluxes.get('qnet', 0.0)
        self.bc.surface['fw_flux'] = fluxes.get('fw', 0.0)

    def update_turbulence(self, turb_params: Dict):
        """更新湍流参数"""
        self.km = np.maximum(turb_params['km'], self.min_diffusivity.magnitude)
        self.kt = np.maximum(turb_params['kt'], self.min_diffusivity.magnitude)

    def step(self):
        """
        执行一个时间步的积分
        返回:
            更新后的模型状态
        """
        # 1. 更新湍流场
        self._update_tke()

        # 2. 求解温度方程
        self.state.temperature = self._solve_tracer_eq(
            self.state.temperature,
            self.kt,
            self.bc.surface['heat_flux'],
            'temperature'
        )

        # 3. 求解盐度方程
        self.state.salinity = self._solve_tracer_eq(
            self.state.salinity,
            self.km,
            self.bc.surface['fw_flux'],
            'salinity'
        )

        # 4. 求解动量方程
        self.state.u, self.state.v = self._solve_momentum_eq()

        # 5. 诊断混合层深度
        self.state.ml_depth = self._diagnose_mixed_layer()

        return self.state

    def _solve_tracer_eq(
            self,
            tracer: np.ndarray,
            diffusivity: np.ndarray,
            surface_flux: float,
            tracer_type: str
    ) -> np.ndarray:
        """
        求解标量输运方程 (温度/盐度)
        方程形式: ∂C/∂t = ∂/∂z (K ∂C/∂z)
        """
        nz = len(self.depth)
        dz = self.dz

        # 构造扩散矩阵
        K = diffusivity
        a = np.zeros(nz)
        b = np.zeros(nz)
        c = np.zeros(nz)

        # 内部点 (中心差分)
        for k in range(1, nz - 1):
            a[k] = -self.alpha * self.dt * 0.5 * (K[k - 1] + K[k]) / (dz[k - 1] * dz[k])
            c[k] = -self.alpha * self.dt * 0.5 * (K[k] + K[k + 1]) / (dz[k] * dz[k + 1])
            b[k] = 1 - a[k] - c[k]

        # 边界条件
        if tracer_type == 'temperature':
            # 表面热通量边界
            b[0] = 1 + self.alpha * self.dt * K[0] / dz[0] ** 2
            c[0] = -self.alpha * self.dt * K[0] / dz[0] ** 2
            flux_term = self.dt * surface_flux / (1025 * 4000 * dz[0])
        else:  # salinity
            # 表面盐度通量边界
            b[0] = 1
            c[0] = 0
            flux_term = surface_flux * self.dt / dz[0]

        # 底部边界 (零通量)
        a[-1] = -self.alpha * self.dt * K[-1] / dz[-1] ** 2
        b[-1] = 1 - a[-1]

        # 构造右端项
        rhs = tracer.copy()
        rhs[0] += flux_term

        # 解三对角系统
        return self._solve_tridiagonal(a, b, c, rhs)

    def _solve_momentum_eq(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解动量方程
        方程形式: ∂u/∂t = ∂/∂z (Km ∂u/∂z) + fv
                  ∂v/∂t = ∂/∂z (Km ∂v/∂z) - fu
        """
        nz = len(self.depth)
        dz = self.dz
        km = self.km

        # 构造扩散矩阵 (与温度方程类似)
        a = np.zeros(nz)
        b = np.zeros(nz)
        c = np.zeros(nz)

        for k in range(1, nz - 1):
            a[k] = -self.alpha * self.dt * 0.5 * (km[k - 1] + km[k]) / (dz[k - 1] * dz[k])
            c[k] = -self.alpha * self.dt * 0.5 * (km[k] + km[k + 1]) / (dz[k] * dz[k + 1])
            b[k] = 1 - a[k] - c[k]

        # 表面边界 (风应力)
        b[0] = 1 + self.alpha * self.dt * km[0] / dz[0] ** 2
        c[0] = -self.alpha * self.dt * km[0] / dz[0] ** 2
        tau = self.bc.surface['wind_stress']
        rhs_u = self.state.u.copy()
        rhs_v = self.state.v.copy()
        rhs_u[0] += self.dt * tau / (1025 * dz[0])

        # 底部边界 (二次对数定律)
        a[-1] = -self.alpha * self.dt * km[-1] / dz[-1] ** 2
        b[-1] = 1 - a[-1] + self.alpha * self.dt * self.bc.bottom['friction'] / dz[-1]

        # 科氏力处理 (半隐式)
        f = 1e-4  # 科氏参数
        for k in range(nz):
            denom = 1 + (self.dt * f) ** 2
            new_u = (rhs_u[k] + self.dt * f * rhs_v[k]) / denom
            new_v = (rhs_v[k] - self.dt * f * rhs_u[k]) / denom
            rhs_u[k], rhs_v[k] = new_u, new_v

        # 解三对角系统
        u_new = self._solve_tridiagonal(a, b, c, rhs_u)
        v_new = self._solve_tridiagonal(a, b, c, rhs_v)

        return u_new, v_new

    def _solve_tridiagonal(self, a, b, c, d) -> np.ndarray:
        """
        求解三对角系统 (Thomas算法)
        | b0 c0         | x0   | d0   |
        | a1 b1 c1      | x1   | d1   |
        |    a2 b2 c2   |  =   | d2   |
        |        ...    | ...  | ...  |
        |         an bn | xn   | dn   |
        """
        n = len(d)
        x = np.zeros(n)

        # 前向消元
        for i in range(1, n):
            w = a[i] / b[i - 1]
            b[i] -= w * c[i - 1]
            d[i] -= w * d[i - 1]

        # 回代
        x[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

        return x

    def _update_tke(self):
        """更新湍流动能 (简化模型)"""
        shear = np.gradient(self.state.u, self.depth) ** 2 + np.gradient(self.state.v, self.depth) ** 2
        buoyancy = 2e-4 * 9.81 * np.gradient(self.state.temperature, self.depth)
        production = self.km * shear
        dissipation = (self.state.tke ** 1.5) / (0.4 * np.abs(self.depth + 1))

        self.state.tke += self.dt * (production - buoyancy - dissipation)
        self.state.tke = np.maximum(self.state.tke, 1e-6)

    def _diagnose_mixed_layer(self) -> float:
        """诊断混合层深度 (基于密度跃层)"""
        rho = self._density(self.state.temperature, self.state.salinity)
        drho = np.abs(rho - rho[0])
        idx = np.where(drho > 0.03)[0]  # Δρ > 0.03 kg/m³
        return self.depth[idx[0]] if len(idx) > 0 else self.depth[-1]

    def _density(self, temp: np.ndarray, salt: np.ndarray) -> np.ndarray:
        """计算海水密度 (简化UNESCO公式)"""
        return 1025 + 0.2 * salt - 0.1 * temp  # 近似公式


# 单元测试
if __name__ == "__main__":
    from config import Config
    import matplotlib.pyplot as plt

    # 测试配置
    cfg = Config(
        dt=1800,
        max_depth=200,
        n_levels=50,
        scheme='implicit'
    )

    # 初始状态
    init_state = {
        'temp': np.linspace(25, 5, cfg.n_levels),
        'salt': np.linspace(34, 35, cfg.n_levels),
        'u': np.zeros(cfg.n_levels),
        'v': np.zeros(cfg.n_levels)
    }

    # 初始化求解器
    solver = OceanSolver(cfg)
    solver.initialize(init_state)

    # 设置边界条件 (模拟白天加热)
    solver.set_boundary_conditions({
        'tau': 0.1,  # 风应力 (N/m²)
        'qnet': 500,  # 热通量 (W/m²)
        'fw': -1e-6  # 淡水通量 (蒸发)
    })

    # 设置湍流参数 (模拟强混合)
    solver.update_turbulence({
        'km': np.linspace(0.1, 0.01, cfg.n_levels),
        'kt': np.linspace(0.05, 0.005, cfg.n_levels)
    })

    # 运行模拟 (24小时)
    results = []
    for _ in range(48):  # 30分钟步长
        state = solver.step()
        results.append(state.temperature.copy())

    # 可视化结果
    plt.figure(figsize=(10, 6))
    for i in [0, 12, 24, 36, 47]:
        plt.plot(results[i], solver.depth, label=f'Step {i}')
    plt.gca().invert_yaxis()
    plt.xlabel('Temperature (℃)')
    plt.ylabel('Depth (m)')
    plt.title('Ocean Mixed Layer Evolution')
    plt.legend()
    plt.savefig('mixed_layer_evolution.png')
    print("求解器测试完成，结果已保存为mixed_layer_evolution.png")
def run_dwm_simulation():
    return 0