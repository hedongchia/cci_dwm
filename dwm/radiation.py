"""
海洋辐射传输计算模块 (v1.2)
功能：
1. 计算短波辐射透射剖面
2. 估算光合有效辐射(PAR)
3. 处理不同类型的水体衰减系数
4. 支持卫星遥感数据输入
"""

import numpy as np
import xarray as xr
from typing import Dict, Optional
from dataclasses import dataclass
from config import Config
from metpy.units import units
from scipy.special import gammainc


@dataclass
class RadiationProfile:
    """辐射剖面数据容器
    用于存储计算得到的辐射相关数据。
    """
    # 短波辐射，单位为瓦每平方米
    swr: np.ndarray  # 短波辐射 (W/m²)
    # 光合有效辐射，单位为微摩尔光子每平方米每秒
    par: np.ndarray  # 光合有效辐射 (µmol photons/m²/s)
    # PAR衰减系数，单位为每米
    kd_par: np.ndarray  # PAR衰减系数 (1/m)
    # 真光层深度，单位为米
    euphotic_depth: np.ndarray  # 真光层深度 (m)


class RadiationCalculator:
    def __init__(self, cfg: Config):
        """
        初始化辐射计算器
        参数:
            cfg : 配置参数对象
                 必需参数:
                 - water_type (默认'open_ocean')
                 - par_fraction (默认0.43) 短波中PAR占比
                 - solar_zenith (默认30度) 太阳天顶角
        """
        self.cfg = cfg
        # 设置水体光学特性
        self.set_water_properties()
        # 设置太阳辐射参数
        self.set_solar_parameters()

    def set_water_properties(self):
        """设置水体光学特性
        根据配置中的水体类型，确定不同类型水体的短波辐射和PAR的衰减系数。
        """
        # 获取配置中的水体类型，默认为开阔海洋
        self.water_type = getattr(self.cfg, 'water_type', 'open_ocean')

        # 衰减系数查找表 (单位: 1/m)
        self.attenuation_coeff = {
            'open_ocean': {
                'swr': 0.04,
                'par': 0.06
            },
            'coastal': {
                'swr': 0.1,
                'par': 0.15
            },
            'turbid': {
                'swr': 0.5,
                'par': 0.8
            }
        }

    def set_solar_parameters(self):
        """设置太阳辐射参数
        从配置中获取PAR在短波辐射中的占比和太阳天顶角，并进行单位转换。
        """
        # 获取配置中的PAR占短波比例，默认为0.43
        self.par_fraction = getattr(self.cfg, 'par_fraction', 0.43)
        # 获取配置中的太阳天顶角，转换为弧度，默认为30度
        self.zenith = np.deg2rad(getattr(self.cfg, 'solar_zenith', 30))

    def calculate_radiation(
            self,
            surface_swr: xr.DataArray,
            depth: np.ndarray,
            chl: Optional[xr.DataArray] = None
    ) -> RadiationProfile:
        """
        计算辐射透射剖面
        参数:
            surface_swr : 海表短波辐射 (W/m²)
            depth : 深度坐标 (m)
            chl : 叶绿素浓度 (mg/m³, 可选)
        返回:
            RadiationProfile 对象
        """
        # 转换为numpy数组并添加单位
        swr_sfc = surface_swr.values * units('W/m^2')
        depth_arr = depth * units('m')

        # 计算衰减系数
        if chl is not None:
            # 如果提供了叶绿素浓度，使用基于叶绿素的衰减系数估计方法
            kd_swr, kd_par = self._chlorophyll_based_attenuation(chl.values)
        else:
            # 否则，根据水体类型从衰减系数查找表中获取衰减系数
            kd_swr = self.attenuation_coeff[self.water_type]['swr'] * units('1/m')
            kd_par = self.attenuation_coeff[self.water_type]['par'] * units('1/m')

        # 计算辐射剖面
        # 计算短波辐射的垂直剖面
        swr_profile = self._exponential_decay(swr_sfc, kd_swr, depth_arr)
        # 计算光合有效辐射的垂直剖面
        par_profile = self._calculate_par(swr_profile, kd_par)

        # 计算真光层深度 (1%表面PAR的深度)
        euphotic_depth = self._euphotic_zone(kd_par)

        return RadiationProfile(
            swr=swr_profile.magnitude,
            par=par_profile.magnitude,
            kd_par=kd_par.magnitude,
            euphotic_depth=euphotic_depth.magnitude
        )

    def _exponential_decay(
            self,
            surface_flux: units.Quantity,
            kd: units.Quantity,
            depth: units.Quantity
    ) -> units.Quantity:
        """
        指数衰减模型计算辐射透射
        Jerlov (1968) 双指数模型简化版
        """
        # 考虑天顶角修正
        mu = np.cos(self.zenith)
        effective_kd = kd / mu

        # 深度扩展维度以匹配输入形状
        if surface_flux.ndim > 0 and depth.ndim == 1:
            depth = depth[np.newaxis, np.newaxis, np.newaxis, :]
            effective_kd = effective_kd[..., np.newaxis]

        return surface_flux[..., np.newaxis] * np.exp(-effective_kd * depth)

    def _calculate_par(
            self,
            swr_profile: units.Quantity,
            kd_par: units.Quantity
    ) -> units.Quantity:
        """
        计算光合有效辐射(PAR)
        单位转换: W/m² → µmol photons/m²/s
        """
        # 转换系数: 1 W/m² ≈ 4.57 µmol photons/m²/s (PAR波段)
        conversion_factor = 4.57 * units('umol photons/W')
        # 计算海表PAR
        par_sfc = swr_profile[..., 0] * self.par_fraction * conversion_factor

        # 计算PAR剖面
        return par_sfc[..., np.newaxis] * np.exp(-kd_par[..., np.newaxis] *
                                                 swr_profile.dims[-1] * units('m'))

    def _chlorophyll_based_attenuation(
            self,
            chl: np.ndarray
    ) -> Tuple[units.Quantity, units.Quantity]:
        """
        基于叶绿素的衰减系数估计
        使用Morel (1988) 经验公式
        """
        # 避免极小值
        chl = np.maximum(chl, 0.02)

        # 短波衰减系数 (kd_swr)
        kd_swr = (0.0166 + 0.0773 * chl ** 0.6715) * units('1/m')

        # PAR衰减系数 (kd_par)
        kd_par = (0.0665 + 0.874 * chl ** 0.881) * units('1/m')

        return kd_swr, kd_par

    def _euphotic_zone(
            self,
            kd_par: units.Quantity
    ) -> units.Quantity:
        """
        计算真光层深度 (Ze)
        定义为PAR降至表面1%的深度
        """
        return (np.log(100) / kd_par).to('m')

    def _spectral_integration(
            self,
            wavelengths: np.ndarray,
            irradiance: np.ndarray,
            depth: units.Quantity
    ) -> units.Quantity:
        """
        光谱积分方法 (高精度)
        参数:
            wavelengths : 波长(nm)
            irradiance : 光谱辐照度(W/m²/nm)
            depth : 深度(m)
        """
        # 此处实现多波段积分
        pass  # 实际应用时需扩展


# 单元测试
if __name__ == "__main__":
    from config import Config
    import matplotlib.pyplot as plt

    # 测试配置
    cfg = Config(
        start_date='2023-06-01',
        water_type='open_ocean',
        par_fraction=0.45,
        solar_zenith=25
    )

    # 模拟输入数据
    surface_swr = xr.DataArray(
        data=np.random.rand(1, 10, 10) * 800 + 200,  # 200-1000 W/m²
        dims=('time', 'lat', 'lon'),
        coords={
            'time': [np.datetime64('2023-06-01T12:00')],
            'lat': np.linspace(-5, 5, 10),
            'lon': np.linspace(120, 130, 10)
        }
    )

    chl = xr.DataArray(
        data=np.random.rand(1, 10, 10) * 0.5 + 0.1,  # 0.1-0.6 mg/m³
        dims=('time', 'lat', 'lon'),
        coords=surface_swr.coords
    )

    depth = np.linspace(0, 100, 50)  # 0-100米深度

    # 执行计算
    calculator = RadiationCalculator(cfg)
    result = calculator.calculate_radiation(surface_swr, depth, chl)

    # 可视化验证
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 选择第一个时次和位置展示垂直剖面
    idx = (0, 5, 5)  # time, lat, lon索引

    axs[0, 0].plot(result.swr[idx], depth, 'b-')
    axs[0, 0].set_title('Shortwave Radiation')
    axs[0, 0].set_ylabel('Depth (m)')
    axs[0, 0].set_xlabel('SWR (W/m²)')

    axs[0, 1].plot(result.par[idx], depth, 'g-')
    axs[0, 1].set_title('PAR Profile')
    axs[0, 1].set_xlabel('PAR (µmol photons/m²/s)')

    # 真光层深度水平分布
    im = axs[1, 0].imshow(result.euphotic_depth[idx[1:]],
                          extent=[120, 130, -5, 5],
                          cmap='viridis')
    plt.colorbar(im, ax=axs[1, 0], label='Depth (m)')
    axs[1, 0].set_title('Euphotic Depth')

    # PAR衰减系数分布
    im = axs[1, 1].imshow(result.kd_par[idx[1:]],
                          extent=[120, 130, -5, 5],
                          cmap='jet')
    plt.colorbar(im, ax=axs[1, 1], label='Attenuation (1/m)')
    axs[1, 1].set_title('PAR Attenuation Coefficient')

    plt.tight_layout()
    plt.savefig('radiation_profiles.png')
    print("辐射计算完成，结果已保存为radiation_profiles.png")
def decompose_solar_radiation():
    return 0