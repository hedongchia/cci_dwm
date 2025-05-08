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
        sst_mean = np.roll(sst_mean, lon_0_180.shape[0], axis=1)
        
        # 构建网格
        lon, lat = np.meshgrid(lon, lat)
        
        return sst_mean, lon, lat
        
    except Exception as e:
        raise RuntimeError(f"CCI数据加载失败: {str(e)}") from e


if __name__ == "__main__":
    # 模块测试（使用固定测试路径），不影响main.py
    test_file = "D:/data/20240101120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR3.0-v02.0-fv01.0.nc"
    try:
        if Path(test_file).exists():
            sst, lon, lat = load_cci_sst(test_file)
            print(f"数据加载成功！SST形状: {sst.shape}, 经度范围: {lon.min()}~{lon.max()}, 纬度范围: {lat.min()}~{lat.max()}")
        else:
            print(f"测试文件不存在: {test_file}")
    except Exception as e:
        print(f"测试失败: {str(e)}")