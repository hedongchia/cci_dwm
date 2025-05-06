"""
ERA5数据加载模块 (v1.0)
功能：
1. 加载ERA5原始数据（NetCDF格式）
2. 提取关键变量：风应力(tau)、太阳辐射(Qsol)、10m风速(u10)
3. 单位统一化和质量检查
4. 时间对齐处理（确保与CCI数据时间戳匹配）
"""

import xarray as xr
import numpy as np
from typing import Dict
from datetime import datetime
from config import Config


def load_era5_data(cfg: Config) -> xr.Dataset:
    """
    加载并预处理ERA5数据
    参数:
        cfg : Config对象，包含路径和参数配置
    返回:
        xr.Dataset 包含以下变量:
            - tau    : 风应力 (N/m²)
            - Qsol   : 太阳辐射 (W/m²)
            - u10    : 10m风速 (m/s)
            - wind_shear : 垂直风切变 (s⁻¹)
    """
    # 1. 调用私有函数_load_raw_era5加载原始ERA5数据
    ds = _load_raw_era5(cfg.era5_path)

    # 2. 调用私有函数_extract_key_variables从加载的数据中提取关键变量
    era5_vars = _extract_key_variables(ds)

    # 3. 计算衍生变量
    # 调用私有函数_calculate_wind_stress，根据10m风速、空气密度和拖曳系数计算风应力
    era5_vars['tau'] = _calculate_wind_stress(
        era5_vars['u10'],
        cfg.rho_air,
        cfg.drag_coeff
    )
    # 调用私有函数_estimate_wind_shear，根据10m风速估算垂直风切变
    era5_vars['wind_shear'] = _estimate_wind_shear(era5_vars['u10'])

    # 4. 创建处理后的Dataset
    ds_out = xr.Dataset(
        data_vars={
            # 风应力数据
            'tau': (('time', 'lat', 'lon'), era5_vars['tau']),
            # 太阳辐射数据
            'Qsol': (('time', 'lat', 'lon'), era5_vars['ssr']),
            # 10m风速数据
            'u10': (('time', 'lat', 'lon'), era5_vars['u10']),
            # 垂直风切变数据
            'wind_shear': (('time', 'lat', 'lon'), era5_vars['wind_shear'])
        },
        # 使用原始数据的坐标信息
        coords=ds.coords
    )

    # 5. 添加网格信息供后续插值使用
    ds_out.attrs['grid'] = {
        # 纬度信息
        'lat': ds.latitude.values,
        # 经度信息
        'lon': ds.longitude.values,
        # 空间分辨率
        'resolution': cfg.era5_resolution
    }

    return ds_out


def _load_raw_era5(filepath: str) -> xr.Dataset:
    """加载原始ERA5数据并进行基础验证"""
    try:
        # 打开指定路径的NetCDF数据集
        ds = xr.open_dataset(filepath)

        # 必需变量检查
        # 定义必需的变量列表
        required_vars = ['u10', 'ssr']
        # 找出数据集中缺失的必需变量
        missing_vars = [var for var in required_vars if var not in ds]
        # 如果有缺失的必需变量，抛出值错误
        if missing_vars:
            raise ValueError(f"缺失必要变量: {missing_vars}")

        # 时间维度检查（应为每小时数据）
        # 检查数据集中是否存在时间维度
        if 'time' not in ds.dims:
            raise ValueError("输入数据必须包含time维度")

        return ds
    except Exception as e:
        # 如果加载过程中出现异常，抛出运行时错误
        raise RuntimeError(f"ERA5数据加载失败: {str(e)}")


def _extract_key_variables(ds: xr.Dataset) -> Dict[str, np.ndarray]:
    """提取并转换关键变量"""
    return {
        # 提取10m风速数据
        'u10': ds['u10'].values,  # 10m风速 (m/s)
        # 提取太阳辐射数据并将单位从焦耳每平方米转换为瓦特每平方米
        'ssr': ds['ssr'].values / 3600  # 太阳辐射 (J/m² → W/m²)
    }


def _calculate_wind_stress(u10: np.ndarray, rho_air: float, cd: float) -> np.ndarray:
    """
    计算风应力 tau = ρ_air * Cd * u10²
    参数:
        u10     : 10m风速 (m/s)
        rho_air : 空气密度 (kg/m³)
        cd      : 拖曳系数
    返回:
        风应力数组 (N/m²)
    """
    return rho_air * cd * np.square(u10)


def _estimate_wind_shear(u10: np.ndarray, z_ref: float = 10.0) -> np.ndarray:
    """
    估算垂直风切变 (du/dz)
    简化方案: 使用对数风廓线假设
    du/dz ≈ u* / (κ*z)
    其中u* ≈ u10 / 11
    """
    # von Karman常数
    kappa = 0.4
    # 经验估算摩擦速度
    u_star_approx = u10 / 11.0
    return u_star_approx / (kappa * z_ref)


# 单元测试样例
if __name__ == "__main__":
    from config import Config

    # 创建Config对象，设置日期、CCI分辨率和ERA5分辨率
    cfg = Config(datetime(2023, 1, 1), 0.05, 0.25)

    # 测试数据加载
    test_data = load_era5_data(cfg)
    # 打印加载成功信息和包含的变量
    print(f"数据加载成功！变量包含: {list(test_data.data_vars)}")
    # 打印时间维度的时次数量
    print(f"时间维度: {test_data.dims['time']}个时次")
    # 打印空间分辨率
    print(f"空间分辨率: {test_data.attrs['grid']['resolution']}度")