import xarray as xr
import numpy as np


def regrid_era5_to_cci(era5_data: xr.Dataset, cci_data: xr.Dataset) -> xr.Dataset:
    """将ERA5风应力数据重新网格化到CCI分辨率网格"""
    # 计算风应力tau
    rho_air = 1.225  # 空气密度，kg/m³
    cd = 1.2e-3      # 拖曳系数
    u10 = era5_data['u10']
    v10 = era5_data['v10']
    wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)  # 计算风速大小
    tau = rho_air * cd * wind_speed ** 2        # 计算风应力

    # 确保CCI的lat/lon为一维数组（若为二维网格需降维）
    if len(cci_data.lat.dims) > 1:
        cci_data = cci_data.stack(point=('lat', 'lon')).reset_index('point')

    # 执行线性插值（关键：将fill_value和assume_sorted作为关键字参数）
    tau_regrid = tau.interp(latitude=cci_data['lat'], longitude=cci_data['lon'], method='linear')

    # 构建输出数据集
    ds_out = xr.Dataset(
        data_vars={
            'tau_regrid': tau_regrid
        },
        coords={
            'valid_time': era5_data.valid_time,  # ERA5时间坐标
            'lat': cci_data['lat'],             # CCI纬度坐标
            'lon': cci_data['lon']              # CCI经度坐标
        },
        attrs={
            'regrid_method': 'linear',
            'source_resolution': f"{np.diff(era5_data.latitude).mean():.2f}°",
            'target_resolution': f"{np.diff(cci_data.lat).mean():.2f}°"
        }
    )
    return ds_out


# 加载数据（确保路径正确且文件存在）
era5_path = r"D:\SRDP\data\20240301\era5_single_20240301.nc"
cci_path = r"D:\SRDP\dap.ceda.ac.uk\neodc\eocis\data\global_and_regional\sea_surface_temperature\CDR_v3\Analysis\L4\v3.0.1\2023\03\01\20230301120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR3.0-v02.0-fv01.0.nc"

era5_data = xr.open_dataset(era5_path)
cci_data = xr.open_dataset(cci_path)

# 执行重网格化
regridded_tau = regrid_era5_to_cci(era5_data, cci_data)
print(regridded_tau)