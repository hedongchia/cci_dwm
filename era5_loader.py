"""
ERA5数据加载模块 (v1.1)

核心功能：
1. 原始数据加载：读取NetCDF格式的ERA5再分析数据，支持：
   - 太阳辐射数据(ssr)
   - 风数据(u10/v10)
"""

import h5py
import numpy as np
from typing import Tuple, Dict, Union
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt  # 导入matplotlib用于可视化
from mpl_toolkits.basemap import Basemap

class ERA5DataType(Enum):
    """ERA5数据类型枚举"""
    SOLAR_RADIATION = 'ssr'
    WIND_U = 'u10'
    WIND_V = 'v10'

def load_era5_data(filepath: str, data_type: ERA5DataType) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载并预处理ERA5数据

    参数：
    filepath : str
        ERA5数据文件路径
    data_type : ERA5DataType
        要加载的数据类型枚举

    返回：
    (data, lon, lat) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        处理后的数据、经度网格、纬度网格
    """
    try:
        # 读取原始数据
        with h5py.File(filepath, 'r') as ncfile:
            lon = ncfile['longitude'][:]  # 经度数据
            lat = ncfile['latitude'][:]  # 纬度数据
            data = ncfile[data_type.value][:]  # 获取指定类型的数据

        # 数据预处理
        data = data.astype(np.float32)

        if np.all(np.isnan(data)):
            raise ValueError(f"All values in {data_type.value} are NaN")

        # 构建网格
        lon, lat = np.meshgrid(lon, lat)

        return data, lon, lat

    except Exception as e:
        raise RuntimeError(f"ERA5数据加载失败({data_type.value}): {str(e)}") from e


if __name__ == "__main__":
    # 模块测试
    test_files = {
        ERA5DataType.SOLAR_RADIATION: "D:/data/data_stream-oper_stepType-accum.nc",
        ERA5DataType.WIND_U: "D:/data/data_stream-oper_stepType-instant.nc",
        ERA5DataType.WIND_V: "D:/data/data_stream-oper_stepType-instant.nc"
    }

    for data_type, test_file in test_files.items():
        try:
            if Path(test_file).exists():
                data, lon, lat = load_era5_data(test_file, data_type)
                print(f"{data_type.value}数据加载成功！形状: {data.shape}, "
                      f"经度范围: {lon.min()}~{lon.max()}, 纬度范围: {lat.min()}~{lat.max()}")
                
                # 设置绘图参数
                plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体
                plt.figure(figsize=(10, 6), dpi=150)

                # 创建Basemap对象并设置地图
                m = Basemap(projection='cyl', llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90)
                m.drawmeridians(np.arange(0, 361, 30), labels=[0, 0, 0, 1], fontsize=10, linewidth=0.8, color='silver')
                m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=10, linewidth=0.8, color='silver')

                # 绘制平均SST数据，并设置colorbar的范围
                cs = m.pcolormesh(lon, lat, np.nanmean(data, axis=0), cmap='jet', shading='auto', vmin=np.nanmin(data), vmax=np.nanmax(data))

                # 添加海岸线
                m.drawcoastlines()

                # 填充陆地为灰色
                m.fillcontinents(color='gray', lake_color='aqua')

                # 添加色带在右侧
                cb = m.colorbar(cs, location='right', pad=0.05)
                cb.set_label(f'Average {data_type}', fontsize=12)

                # 添加标题
                plt.title(f'Global Average {data_type}', fontsize=16)
                plt.show()

            else:
                print(f"测试文件不存在: {test_file}")
        except Exception as e:
            print(f"{data_type.value}测试失败: {str(e)}")
    