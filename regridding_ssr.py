"""
太阳辐射网格化处理模块 (v1.6)
功能：
1. 一次性计算24小时太阳辐射（24×720×1440）
2. 逐小时插值到目标分辨率（24×3600×7200）
"""

import numpy as np
import xarray as xr
from era5_loader import load_era5_data, ERA5DataType
from tqdm import tqdm  # 进度条工具

def process_ssr(
    era5_file: str,
) -> xr.Dataset:
    """
    先计算完整风应力，再逐小时插值

    参数：
    era5_file : str - ERA5数据路径

    返回：
    ds_out : xr.Dataset - 包含(24,3600,7200)结果的数据集
    """
    #  一次性加载所有时间步的ssr数据
    ssr, lon, lat = load_era5_data(era5_file, ERA5DataType.SOLAR_RADIATION)

    # 定义目标网格（3600×7200）
    new_lat = np.linspace(-90, 90, 3600)
    new_lon = np.linspace(0, 359.99, 7200)  # 避免360°=0°问题

    # 4. 初始化输出数组（24×3600×7200）
    ssr_highres = np.zeros((24, 3600, 7200), dtype=np.float32)

    # 5. 逐小时插值
    for hour in tqdm(range(24), desc="插值进度"):
        # 创建当前小时的Dataset
        ssr_hour = xr.Dataset(
            data_vars={'ssr': (('lat', 'lon'), ssr[hour])},
            coords={
                'lat': np.linspace(-90, 90, 721),
                'lon': np.linspace(0, 359.75, 1440)
            }
        )

        # 执行插值
        ssr_highres[hour] = ssr_hour['ssr'].interp(
            lat=new_lat,
            lon=new_lon,
            method='linear',
            kwargs={'fill_value': np.nan}
        ).values.astype(np.float32)

    # 6. 构建最终数据集
    ssr_out = xr.Dataset(
        data_vars={'ssr': (('time', 'lat', 'lon'), ssr_highres)},
        coords={
            'time': np.arange(24),
            'lat': new_lat,
            'lon': new_lon
        },
        attrs={
            'units': 'N/m^2'
        }
    )

    return ssr_out

# 使用示例
if __name__ == "__main__":
    era5_path = "D:/data/data_stream-oper_stepType-accum.nc"
    try:
        result = process_ssr(era5_path)
        print(f"ssr插值成功！结果形状: {result['ssr'].shape}")
        print(f"示例数据（第0小时，[0:3,0:3]）:\n{result['ssr'][0,0:3,0:3].values}")
    except Exception as e:
        print(f"处理失败: {str(e)}")