"""
CCI SST数据加载模块 (v1.1)
功能：
1. 加载原始CCI日平均SST数据
2. 提取基础SST场和质控标志
3. 网格信息标准化处理
4. 时间对齐与缺失值处理
"""

import xarray as xr
import numpy as np
from typing import Dict
from datetime import datetime
from config import Config

def load_cci_sst(cfg: Config) -> xr.Dataset:
    """
    加载并预处理CCI SST数据
    参数:
        cfg : Config对象，包含路径和参数配置
    返回:
        xr.Dataset 包含以下变量:
            - sst     : 海表温度 (℃)
            - quality : 数据质量标志 (0-1)
            - mask    : 有效数据掩膜 (bool)
        attrs:
            - grid    : 网格元信息
    """
    # 1. 调用私有函数_load_raw_cci加载原始CCI数据
    ds = _load_raw_cci(cfg.cci_path)

    # 2. 调用私有函数_standardize_variables对数据中的变量进行标准化处理
    cci_data = _standardize_variables(ds)

    # 3. 调用私有函数_generate_quality_mask，根据数据质量标志和质量阈值生成有效数据掩膜
    cci_data['mask'] = _generate_quality_mask(
        cci_data['quality'],
        cfg.quality_threshold
    )

    # 4. 创建一个新的xarray.Dataset对象，将处理好的数据和坐标信息添加进去
    ds_out = xr.Dataset(
        data_vars={
            # 海表温度数据
            'sst': (('time', 'lat', 'lon'), cci_data['sst']),
            # 数据质量标志
            'quality': (('time', 'lat', 'lon'), cci_data['quality']),
            # 有效数据掩膜
            'mask': (('time', 'lat', 'lon'), cci_data['mask'])
        },
        coords={
            # 时间坐标
            'time': cci_data['time'],
            # 纬度坐标
            'lat': cci_data['lat'],
            # 经度坐标
            'lon': cci_data['lon']
        }
    )

    # 5. 为输出的Dataset对象添加网格信息属性，供后续插值使用
    ds_out.attrs['grid'] = {
        'lat': cci_data['lat'],
        'lon': cci_data['lon'],
        'resolution': cfg.cci_resolution
    }

    return ds_out

def _load_raw_cci(filepath: str) -> xr.Dataset:
    """加载原始CCI数据并进行基础验证"""
    try:
        # 使用xarray打开指定路径的数据集
        with xr.open_dataset(filepath) as ds:
            # 定义必需的变量列表
            required_vars = ['analysed_sst', 'quality_level']
            # 找出数据集中缺失的必需变量
            missing_vars = [var for var in required_vars if var not in ds]
            # 如果有缺失的必需变量，抛出值错误
            if missing_vars:
                raise ValueError(f"缺失必要变量: {missing_vars}")

            # 检查数据集中是否存在时间维度
            if 'time' not in ds.dims:
                raise ValueError("输入数据必须包含time维度")

            # 立即将数据加载到内存中并返回
            return ds.load()
    except Exception as e:
        # 如果加载过程中出现异常，抛出运行时错误
        raise RuntimeError(f"CCI数据加载失败: {str(e)}")

def _standardize_variables(ds: xr.Dataset) -> Dict:
    """
    标准化CCI变量:
    1. 温度单位转换 (Kelvin → Celsius)
    2. 经度标准化到[-180,180]
    3. 时间维度对齐到UTC午夜
    """
    # 将海表温度从开尔文转换为摄氏度
    sst = ds['analysed_sst'].values - 273.15  # K→℃

    # 将质量标志归一化到0 - 1范围
    quality = ds['quality_level'].values / 5.0  # 归一化到0-1

    # 将经度值标准化到[-180, 180]范围
    lon = np.where(ds.lon.values > 180,
                   ds.lon.values - 360,
                   ds.lon.values)

    # 将时间维度对齐到UTC午夜
    time = ds.time.dt.floor('D').values

    # 返回包含处理后变量的字典
    return {
        'sst': sst,
        'quality': quality,
        'time': time,
        'lat': ds.lat.values,
        'lon': lon
    }

def _generate_quality_mask(quality: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    生成有效数据掩膜
    参数:
        quality : 质量标志 (0-1)
        threshold : 质量阈值 (默认0.8对应原quality_level≥4)
    返回:
        bool数组 (True表示有效数据)
    """
    # 根据质量标志和阈值生成布尔数组，True表示有效数据
    return quality >= threshold

# 单元测试样例
if __name__ == "__main__":
    from config import Config

    # 创建Config对象，设置日期、CCI分辨率和ERA5分辨率
    cfg = Config(datetime(2023, 1, 1), 0.05, 0.25)

    # 测试数据加载
    test_data = load_cci_sst(cfg)
    # 打印CCI数据的网格范围
    print(f"CCI数据加载成功！网格范围: "
          f"lat={test_data.lat.values.min():.2f}~{test_data.lat.values.max():.2f}, "
          f"lon={test_data.lon.values.min():.2f}~{test_data.lon.values.max():.2f}")
    # 打印SST数据的范围
    print(f"SST数据范围: {np.nanmin(test_data['sst'].values):.1f}℃ ~ "
          f"{np.nanmax(test_data['sst'].values):.1f}℃")
    # 打印有效数据的占比
    print(f"有效数据占比: {np.mean(test_data['mask'].values) * 100:.1f}%")
