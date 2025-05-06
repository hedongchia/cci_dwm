"""
海洋湍流参数计算模块 (v1.4)
功能：
1. 实现K-epsilon湍流闭合模型
2. 计算湍流动能(TKE)及其耗散率
3. 动态混合层深度诊断
4. 处理边界层稳定性效应
"""

import numpy as np
import xarray as xr
from typing import Tuple, Optional
from dataclasses import dataclass
from config import Config
from metpy.units import units
from scipy.integrate import trapz


@dataclass
class TurbulenceParameters:
    """湍流参数容器类"""
    tke: np.ndarray  # 湍流动能 (m²/s²)
    epsilon: np.ndarray  # 耗散率 (m²/s³)
    ml_depth: np.ndarray  # 混合层深度 (m)
    km: np.ndarray  # 动量涡扩散系数 (m²/s)
    kt: np.ndarray  # 热量涡扩散系数 (m²/s)


class TurbulenceCalculator:
    def __init__(self, cfg: Config):
        """
        初始化湍流计算器
        参数:
            cfg : 配置参数对象
                 必需参数:
                 - von_karman (默认0.4)
                 - grav_accel (默认9.81 m/s²)
                 - seawater_dens (默认1025 kg/m³)
                 - tke_min (默认1e-6) 最小TKE阈值
                 - prandtl_num (默认0.85) 普朗特数
        """
        self.cfg = cfg
        self.set_constants()
        self.set_parameterization_scheme()

    def set_constants(self):
        """设置物理常数"""
        self.von_karman = getattr(self.cfg, 'von_karman', 0.4)
        self.grav_accel = getattr(self.cfg, 'grav_accel', 9.81)
        self.seawater_dens = getattr(self.cfg, 'seawater_dens', 1025.0)
        self.tke_min = getattr(self.cfg, 'tke_min', 1e-6)
        self.prandtl_num = getattr(self.cfg, 'prandtl_num', 0.85)
        self.c_mu = 0.09  # k-epsilon模型常数

    def set_parameterization_scheme(self, scheme: str = 'k_epsilon'):
        """选择参数化方案"""
        self.scheme = scheme
        if scheme == 'k_epsilon':
            self.calculate_eddy_viscosity = self._k_epsilon_viscosity
        elif scheme == 'my25':
            self.calculate_eddy_viscosity = self._mellor_yamada_viscosity
        else:
            raise ValueError(f"未知参数化方案: {scheme}")

    def calculate_turbulence_fields(
            self,
            physics_data: xr.Dataset,
            ocean_data: xr.Dataset
    ) -> TurbulenceParameters:
        """
        计算湍流场核心方法
        参数:
            physics_data : 包含物理参数的数据集
                         必需变量:
                         - u_star (摩擦速度)
                         - L_obukhov (Obukhov长度)
                         - Kz (湍流扩散系数)
            ocean_data : 包含海洋场的数据集
                       必需变量:
                       - temp (温度剖面, ℃)
                       - salinity (盐度剖面, PSU)
                       - depth (深度坐标, m)
        返回:
            TurbulenceParameters 对象
        """
        # 准备输入场
        u_star = physics_data['u_star'].values * units('m/s')
        L = physics_data['L_obukhov'].values * units('m')
        temp = ocean_data['temp'].values * units('degC')
        salt = ocean_data['salinity'].values
        depth = ocean_data['depth'].values * units('m')

        # 计算基础参数
        N2 = self._buoyancy_frequency(temp, salt, depth)
        S2 = self._shear_frequency(ocean_data['u'], ocean_data['v'], depth)

        # 计算TKE和耗散率
        tke, epsilon = self._solve_tke_epsilon(u_star, L, N2, S2, depth)

        # 诊断混合层深度
        ml_depth = self._diagnose_mixed_layer(tke, depth)

        # 计算涡粘系数
        km, kt = self.calculate_eddy_viscosity(tke, epsilon, N2)

        return TurbulenceParameters(
            tke=tke.magnitude,
            epsilon=epsilon.magnitude,
            ml_depth=ml_depth.magnitude,
            km=km.magnitude,
            kt=kt.magnitude
        )

    def _buoyancy_frequency(
            self,
            temp: units.Quantity,
            salt: np.ndarray,
            depth: units.Quantity
    ) -> units.Quantity:
        """计算浮力频率平方(N²)"""
        alpha = 2e-4 * units('1/degC')  # 热膨胀系数
        beta = 7.6e-4 * units('1/PSU')  # 盐收缩系数

        # 垂直梯度 (中心差分)
        dtemp = np.gradient(temp, depth, axis=-1)
        dsalt = np.gradient(salt, depth, axis=-1)

        N2 = self.grav_accel * (alpha * dtemp - beta * dsalt)
        return np.maximum(N2, 1e-10 * units('1/s^2'))  # 避免负值

    def _shear_frequency(
            self,
            u: np.ndarray,
            v: np.ndarray,
            depth: units.Quantity
    ) -> units.Quantity:
        """计算剪切频率平方(S²)"""
        du = np.gradient(u, depth.magnitude, axis=-1)
        dv = np.gradient(v, depth.magnitude, axis=-1)
        S2 = du ** 2 + dv ** 2
        return S2 * units('1/s^2')

    def _solve_tke_epsilon(
            self,
            u_star: units.Quantity,
            L: units.Quantity,
            N2: units.Quantity,
            S2: units.Quantity,
            depth: units.Quantity
    ) -> Tuple[units.Quantity, units.Quantity]:
        """
        求解TKE和耗散率(ε)
        基于简化的一维k-epsilon模型
        """
        # 表面边界条件
        tke_sfc = (u_star ** 2) / np.sqrt(self.c_mu)
        eps_sfc = (u_star ** 3) / (self.von_karman * np.abs(L))

        # 初始化垂直剖面
        tke = np.zeros_like(N2.magnitude) * units('m^2/s^2')
        epsilon = np.zeros_like(N2.magnitude) * units('m^2/s^3')

        # 表面值
        tke[..., 0] = tke_sfc[..., np.newaxis]
        epsilon[..., 0] = eps_sfc[..., np.newaxis]

        # 垂直传播 (简化衰减模型)
        for k in range(1, len(depth)):
            decay = np.exp(-depth[k] / (0.4 * np.abs(L)))
            tke[..., k] = tke_sfc * decay
            epsilon[..., k] = eps_sfc * decay

        # 稳定性修正
        Ri = N2 / (S2 + 1e-10)
        tke = np.where(Ri < 0.25, tke * (1 - 5 * Ri) ** 2, tke * 1e-3)

        return (
            np.maximum(tke, self.tke_min * units('m^2/s^2')),
            np.maximum(epsilon, 1e-10 * units('m^2/s^3'))
        )

    def _diagnose_mixed_layer(
            self,
            tke: units.Quantity,
            depth: units.Quantity
    ) -> units.Quantity:
        """
        诊断混合层深度
        基于TKE阈值法 (0.95倍表面值)
        """
        tke_sfc = tke[..., 0]
        threshold = 0.05 * tke_sfc
        ml_depth = np.zeros_like(tke_sfc.magnitude) * units('m')

        for i in np.ndindex(tke.shape[:-1]):
            idx = np.where(tke[i] < threshold[i])[0]
            ml_depth[i] = depth[idx[0]] if len(idx) > 0 else depth[-1]

        return ml_depth

    def _k_epsilon_viscosity(
            self,
            tke: units.Quantity,
            epsilon: units.Quantity,
            N2: units.Quantity
    ) -> Tuple[units.Quantity, units.Quantity]:
        """
        K-epsilon方案计算涡粘系数
        """
        km = self.c_mu * (tke ** 2) / (epsilon + 1e-10)
        kt = km / self.prandtl_num
        return km.to('m^2/s'), kt.to('m^2/s')

    def _mellor_yamada_viscosity(
            self,
            tke: units.Quantity,
            epsilon: units.Quantity,
            N2: units.Quantity
    ) -> Tuple[units.Quantity, units.Quantity]:
        """
        Mellor-Yamada 2.5阶方案
        """
        Sm = 0.39  # 稳定性函数参数
        Sh = 0.44

        L_mix = np.sqrt(2 * tke) / (np.sqrt(N2) + 1e-10)
        km = Sm * L_mix * np.sqrt(2 * tke)
        kt = Sh * L_mix * np.sqrt(2 * tke)

        return km.to('m^2/s'), kt.to('m^2/s')


# 单元测试
if __name__ == "__main__":
    from config import Config
    import matplotlib.pyplot as plt

    # 测试配置
    cfg = Config(
        start_date='2023-01-01',
        tke_min=1e-6,
        prandtl_num=0.85
    )

    # 模拟输入数据
    physics_test = xr.Dataset({
        'u_star': (('time', 'lat', 'lon'), np.random.rand(1, 5, 5) * 0.1),
        'L_obukhov': (('time', 'lat', 'lon'), np.random.rand(1, 5, 5) * 100 - 50)
    }, coords={
        'time': [np.datetime64('2023-01-01')],
        'lat': np.linspace(-5, 5, 5),
        'lon': np.linspace(120, 130, 5)
    })

    ocean_test = xr.Dataset({
        'temp': (('time', 'lat', 'lon', 'depth'),
                 np.random.rand(1, 5, 5, 10) * 5 + 20),
        'salinity': (('time', 'lat', 'lon', 'depth'),
                     np.random.rand(1, 5, 5, 10) * 2 + 34),
        'u': (('time', 'lat', 'lon', 'depth'),
              np.random.rand(1, 5, 5, 10) * 0.5),
        'v': (('time', 'lat', 'lon', 'depth'),
              np.random.rand(1, 5, 5, 10) * 0.3)
    }, coords={
        'time': [np.datetime64('2023-01-01')],
        'lat': np.linspace(-5, 5, 5),
        'lon': np.linspace(120, 130, 5),
        'depth': np.linspace(0, 200, 10)
    })

    # 执行计算
    calculator = TurbulenceCalculator(cfg)
    result = calculator.calculate_turbulence_fields(physics_test, ocean_test)

    # 可视化验证
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # 选择第一个时次和位置展示垂直剖面
    idx = (0, 2, 2)  # time, lat, lon索引
    depth = ocean_test.depth.values

    axs[0, 0].plot(result.tke[idx], depth, 'b-')
    axs[0, 0].set_title('TKE Profile')
    axs[0, 0].set_ylabel('Depth (m)')

    axs[0, 1].plot(result.epsilon[idx], depth, 'r-')
    axs[0, 1].set_title('Epsilon Profile')

    axs[0, 2].plot(result.km[idx], depth, 'g-', label='Km')
    axs[0, 2].plot(result.kt[idx], depth, 'm-', label='Kt')
    axs[0, 2].set_title('Eddy Viscosity')
    axs[0, 2].legend()

    # 混合层深度水平分布
    result.ml_depth[idx[1:]].plot(ax=axs[1, 0], cmap='viridis')
    axs[1, 0].set_title('Mixed Layer Depth')

    # TKE表面分布
    physics_test.u_star[0].plot(ax=axs[1, 1], cmap='jet')
    axs[1, 1].set_title('Friction Velocity')

    plt.tight_layout()
    plt.savefig('turbulence_profiles.png')
    print("湍流计算完成，结果已保存为turbulence_profiles.png")
def compute_turbulent_diffusivity():
    return 0