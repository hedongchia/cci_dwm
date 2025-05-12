"""
CCI SST数据加载模块

功能：
1. 读取CCI海表温度数据文件
2. 数据预处理（无效值处理、单位转换）
3. 经度范围调整（-180~180 → 0~360）
4. 返回处理后的SST数据和经纬度网格
"""
import h5py
import numpy as np
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt  # 导入matplotlib用于可视化
from mpl_toolkits.basemap import Basemap

def load_cci_sst(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载并预处理CCI SST数据
    
    参数：
    filepath : str
        CCI数据文件路径（支持日期格式自动生成路径）
    
    返回：
    (sst_mean, lon, lat) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        处理后的平均SST、经度网格、纬度网格
    """
    try:
        # 读取原始数据
        with h5py.File(filepath, 'r') as ncfile:
            lon = ncfile['lon'][:]  # 经度数据
            lat = ncfile['lat'][:]  # 纬度数据
            sst = ncfile['analysed_sst'][:]  # SST数据
            
        # 数据预处理
        sst = sst.astype(np.float32)
        sst[sst == -32768] = np.nan
        
        if np.all(np.isnan(sst)):
            raise ValueError("All values in sst are NaN")
        
        # 单位转换 (缩放因子0.01)
        sst_mean = sst * 0.01
        
        # 计算时间平均
        #sst_mean = np.nanmean(sst, axis=0)
        
        # 经度范围调整
        lon_0_180 = lon[lon >= 0]  # 0~180部分
        lon_n180_0 = lon[lon < 0] + 360  # -180~0 → 180~360
        lon = np.concatenate((lon_0_180, lon_n180_0))
        
        # 数据重排
        #sst_mean = np.roll(sst_mean, lon_0_180.shape[0], axis=1)
        
        # 构建网格
        lon, lat = np.meshgrid(lon, lat)
        
        return sst_mean, lon, lat, lon_0_180
        
    except Exception as e:
        raise RuntimeError(f"CCI数据加载失败: {str(e)}") from e


if __name__ == "__main__":
    # 模块测试（使用固定测试路径），不影响main.py
    test_file = "D:/data/20240101120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR3.0-v02.0-fv01.0.nc"
    try:
        if Path(test_file).exists():
            sst, lon, lat, lon_0_180 = load_cci_sst(test_file)
            print(f"数据加载成功！SST形状: {sst.shape}, 经度范围: {lon.min()}~{lon.max()}, 纬度范围: {lat.min()}~{lat.max()}")
        else:
            print(f"测试文件不存在: {test_file}")
    except Exception as e:
        print(f"测试失败: {str(e)}")

    # 设置绘图参数
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体
    plt.figure(figsize=(10, 6), dpi=150)

    # 创建Basemap对象并设置地图
    m = Basemap(projection='cyl', llcrnrlon=0, llcrnrlat=-90, urcrnrlon=360, urcrnrlat=90)
    m.drawmeridians(np.arange(0, 361, 30), labels=[0, 0, 0, 1], fontsize=10, linewidth=0.8, color='silver')
    m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=10, linewidth=0.8, color='silver')

    sst = np.nanmean(sst, axis=0)
    # 数据重排
    sst_mean = np.roll(sst, lon_0_180.shape[0], axis=1)
    # 绘制平均SST数据，并设置colorbar的范围
    cs = m.pcolormesh(lon, lat, sst_mean, cmap='jet', shading='auto', vmin=np.nanmin(sst), vmax=np.nanmax(sst))

    # 添加海岸线
    m.drawcoastlines()

    # 填充陆地为灰色
    m.fillcontinents(color='gray', lake_color='aqua')

    # 添加色带在右侧
    cb = m.colorbar(cs, location='right', pad=0.05)
    cb.set_label('Average Sea Surface Temperature (°C)', fontsize=12)

    # 添加标题
    plt.title('Global Average SST of CCI', fontsize=16)
    plt.show()