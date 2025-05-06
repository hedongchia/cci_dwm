def validate_with_era5():
    return 0



"""




==================================================将cci的数据插值到era数据分辨率进行验证====================================================






"""
"""
网格化处理模块 (v1.2)
功能：
1. 将CCI SST数据双线性插值到ERA5网格
2. 处理陆地掩膜和无效值
3. 保持时空一致性
"""
from datetime import datetime
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from typing import Tuple, Dict
from config import Config
import warnings

def regrid_cci_to_era5(cci_data: xr.Dataset, cfg: Config) -> xr.Dataset:
    """
    CCI到ERA5的网格化处理
    参数:
        cci_data : 原始CCI数据 (来自cci_loader.py，包含sst、mask等变量)
        cfg      : 配置参数，包含网格范围、分辨率等信息
    返回:
        xr.Dataset 包含:
            - sst_regrid : 插值后的SST (℃，已适配ERA5网格)
            - mask       : 有效数据掩膜 (bool，True表示数据有效)
    """
    # 1. 准备源网格（CCI）和目标网格（ERA5）的坐标信息
    src_grid = _prepare_source_grid(cci_data)
    tgt_grid = _prepare_target_grid(cfg)

    # 2. 对每个时间步长的数据进行插值处理
    sst_regrid, mask_regrid = _process_timesteps(
        cci_data['sst'],       # CCI海表温度数据
        cci_data['mask'],      # CCI有效数据掩膜
        src_grid,              # 源网格信息
        tgt_grid,              # 目标网格信息
        cfg                    # 配置参数
    )

    # 3. 构建输出数据集，包含插值结果和元数据
    ds_out = xr.Dataset(
        data_vars={
            'sst_regrid': (('time', 'lat', 'lon'), sst_regrid),  # 插值后的SST数据
            'mask': (('time', 'lat', 'lon'), mask_regrid)        # 插值后的掩膜数据
        },
        coords={
            'time': cci_data.time,  # 保留原始时间维度
            'lat': tgt_grid['lat'], # ERA5纬度坐标
            'lon': tgt_grid['lon']  # ERA5经度坐标
        },
        attrs={
            'regrid_method': 'bilinear',       # 插值方法
            'source_resolution': f"{src_grid['resolution']}°",  # 源数据分辨率
            'target_resolution': f"{tgt_grid['resolution']}°"   # 目标数据分辨率
        }
    )

    return ds_out

def _prepare_source_grid(cci_data: xr.Dataset) -> Dict:
    """提取CCI网格信息并生成网格点坐标"""
    # 计算CCI网格的实际分辨率（纬度差值的平均值）
    resolution = np.diff(cci_data.lat).mean()
    # 生成网格点坐标矩阵（用于scipy插值）
    points = _generate_grid_points(cci_data.lat, cci_data.lon)
    return {
        'lat': cci_data.lat.values,   # CCI纬度坐标
        'lon': cci_data.lon.values,   # CCI经度坐标
        'resolution': resolution,     # CCI网格分辨率
        'points': points              # 网格点坐标矩阵
    }

def _prepare_target_grid(cfg: Config) -> Dict:
    """生成ERA5目标网格坐标"""
    # 根据配置的纬度范围和分辨率生成纬度数组
    era5_lat = np.arange(
        cfg.era5_lat_range[0],          # 起始纬度
        cfg.era5_lat_range[1] + cfg.era5_resolution,  # 结束纬度（包含边界）
        cfg.era5_resolution             # 分辨率
    )
    # 同理生成经度数组
    era5_lon = np.arange(
        cfg.era5_lon_range[0],
        cfg.era5_lon_range[1] + cfg.era5_resolution,
        cfg.era5_resolution
    )
    # 生成网格点坐标矩阵
    points = _generate_grid_points(era5_lat, era5_lon)
    return {
        'lat': era5_lat,       # ERA5纬度坐标
        'lon': era5_lon,       # ERA5经度坐标
        'resolution': cfg.era5_resolution,  # ERA5网格分辨率
        'points': points       # 网格点坐标矩阵
    }

def _generate_grid_points(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """生成网格点坐标矩阵（格式为N×2，每行对应一个(lon, lat)点）"""
    # 使用meshgrid生成网格坐标矩阵，然后转置并展平为二维数组
    return np.array(np.meshgrid(lon, lat)).T.reshape(-1, 2)

def _process_timesteps(
        sst: xr.DataArray,         # 输入的SST数据（含time维度）
        mask: xr.DataArray,        # 输入的掩膜数据（含time维度）
        src_grid: Dict,            # 源网格信息
        tgt_grid: Dict,            # 目标网格信息
        cfg: Config                # 配置参数
) -> Tuple[np.ndarray, np.ndarray]:
    """
    逐时次进行网格化处理
    返回:
        sst_regrid : 插值后的SST场 (形状为[time, lat, lon])
        mask_regrid: 插值后的有效数据掩膜 (形状同上)
    """
    # 获取时间步数
    n_time = len(sst.time)
    # 预分配结果数组，初始化为NaN
    sst_regrid = np.full(
        (n_time, len(tgt_grid['lat']), len(tgt_grid['lon'])),
        np.nan
    )
    # 预分配掩膜数组，初始化为False
    mask_regrid = np.zeros_like(sst_regrid, dtype=bool)

    for i in range(n_time):
        # 提取当前时次的有效数据：掩膜为True的点
        valid_mask = mask[i].values
        valid_points = src_grid['points'][valid_mask.ravel()]  # 有效点坐标
        valid_values = sst[i].values[valid_mask]                # 有效SST值

        # 执行双线性插值（使用scipy的griddata，method='linear'对应双线性）
        with warnings.catch_warnings():
            # 忽略插值过程中可能出现的警告（如边界外的NaN填充）
            warnings.simplefilter("ignore", category=RuntimeWarning)
            regridded = griddata(
                valid_points,       # 源数据点坐标
                valid_values,       # 源数据值
                tgt_grid['points'], # 目标网格点坐标
                method='linear',    # 插值方法（双线性）
                fill_value=np.nan   # 边界外点填充NaN
            )

        # 将插值结果重塑为目标网格的纬度×经度形状
        sst_regrid[i] = regridded.reshape(len(tgt_grid['lat']), -1)
        # 生成掩膜：非NaN的点视为有效
        mask_regrid[i] = ~np.isnan(sst_regrid[i])

        # 应用陆地掩膜（如果配置中启用）
        if cfg.apply_land_mask:
            sst_regrid[i][cfg.land_mask] = np.nan    # 陆地位置设为NaN
            mask_regrid[i][cfg.land_mask] = False    # 陆地位置掩膜设为False

    return sst_regrid, mask_regrid

# 单元测试
if __name__ == "__main__":
    from config import Config
    from data_loader.cci_loader import load_cci_sst

    # 测试配置
    cfg = Config(
        start_date=datetime(2023, 1, 1),
        cci_resolution=0.05,
        era5_resolution=0.25,
        era5_lat_range=(-90, 90),  # ERA5纬度范围
        era5_lon_range=(-180, 180), # ERA5经度范围
        apply_land_mask=False       # 测试时暂不启用陆地掩膜
    )

    # 加载CCI数据（需确保路径正确或模拟数据存在）
    cci_data = load_cci_sst(cfg)

    # 执行网格化处理
    regridded = regrid_cci_to_era5(cci_data, cfg)

    # 验证结果
    print(f"网格化完成: {regridded.dims}")          # 输出数据维度
    print(f"空间分辨率: {regridded.attrs['target_resolution']}")  # 目标分辨率
    print(f"SST范围: {np.nanmin(regridded.sst_regrid):.1f}℃ ~ "
          f"{np.nanmax(regridded.sst_regrid):.1f}℃")  # SST值范围
    print(f"有效数据点占比: {np.mean(regridded.mask) * 100:.1f}%")  # 有效数据比例