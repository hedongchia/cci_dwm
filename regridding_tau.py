"""
风应力网格化处理模块 (v1.6)
功能：
1. 一次性计算24小时风应力（24×720×1440）
2. 逐小时插值到目标分辨率（24×3600×7200）
"""

import numpy as np
import xarray as xr
from era5_loader import load_era5_data, ERA5DataType
from tqdm import tqdm  # 进度条工具
import matplotlib.pyplot as plt  # 导入matplotlib用于可视化
from mpl_toolkits.basemap import Basemap

def regridding_wind_stress(
    era5_file: str,
) -> xr.Dataset:
    """
    先计算完整风应力，再逐小时插值

    参数：
    era5_file : str - ERA5数据路径

    返回：
    ds_out : xr.Dataset - 包含(24,3600,7200)结果的数据集
    """
    # 一次性加载所有时间步的风场数据
    u10, lon, lat = load_era5_data(era5_file, ERA5DataType.WIND_U)
    v10, _, _ = load_era5_data(era5_file, ERA5DataType.WIND_V)

    # 计算完整的24小时风应力（24×720×1440）
    rho_air = 1.225  # kg/m³
    cd = 1.2e-3      # 拖曳系数
    tau = rho_air * cd * (u10**2 + v10**2)  # 形状 (24,720,1440)

    # 定义目标网格（3600×7200）
    new_lat = np.linspace(-90, 90, 3600)
    new_lon = np.linspace(0, 359.99, 7200)  # 避免360°=0°问题

    # 初始化输出数组（24×3600×7200）
    tau_highres = np.zeros((24, 3600, 7200), dtype=np.float32)

    # 逐小时插值
    for hour in tqdm(range(24), desc="插值进度"):
        # 创建当前小时的Dataset
        tau_hour = xr.Dataset(
            data_vars={'tau': (('lat', 'lon'), tau[hour])},
            coords={
                'lat': np.linspace(-90, 90, 721),
                'lon': np.linspace(0, 359.75, 1440)
            }
        )

        # 执行插值
        tau_highres[hour] = tau_hour['tau'].interp(
            lat=new_lat,
            lon=new_lon,
            method='linear',
            kwargs={'fill_value': np.nan}
        ).values.astype(np.float32)

    # 构建最终数据集
    tau_out = xr.Dataset(
        data_vars={'tau': (('time', 'lat', 'lon'), tau_highres)},
        coords={
            'time': np.arange(24),
            'lat': new_lat,
            'lon': new_lon
        },
        attrs={
            'description': 'ERA5风应力（先全量计算后逐小时插值）',
            'units': 'N/m^2'
        }
    )

    return tau_out

# 使用示例
if __name__ == "__main__":
    era5_path = "D:/data/data_stream-oper_stepType-instant.nc"
    try:
        result = regridding_wind_stress(era5_path)
        print(f"tau插值成功！结果形状: {result['tau'].shape}")
        print(f"示例数据（第0小时，[0:3,0:3]）:\n{result['tau'][0,0:3,0:3].values}")
    except Exception as e:
        print(f"处理失败: {str(e)}")