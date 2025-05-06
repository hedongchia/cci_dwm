from dataclasses import dataclass
from datetime import datetime

@dataclass
class Config:
    """全局参数配置类
    该类用于存储和管理整个程序运行所需的全局参数和配置信息。
    """
    # 初始化所需的必要参数
    date: datetime
    # CCI数据的分辨率
    cci_resolution: float
    # ERA5数据的分辨率
    era5_resolution: float

    # 物理常数部分
    # 空气密度，单位为千克每立方米
    rho_air: float = 1.225  # kg/m³
    # 水的密度，单位为千克每立方米
    rho_water: float = 1025.0  # kg/m³
    # 拖曳系数，用于计算风应力等物理量
    drag_coeff: float = 1.2e-3  # 拖曳系数

    # 辐射参数部分
    # 红外衰减系数，单位为每米
    K_IR: float = 18.1  # 红外衰减系数 [m⁻¹]
    # 可见光衰减系数，单位为每米
    K_VIS: float = 0.06  # 可见光衰减系数 [m⁻¹]
    # 红外辐射在总辐射中的占比
    IR_fraction: float = 0.64  # 红外占比

    # 数值求解部分
    # 数值模拟时的时间步长，单位为秒
    timestep: int = 3600  # 时间步长(秒)

    @property
    def era5_path(self):
        """
        基于日期属性生成ERA5数据文件的路径。
        返回:
            str: ERA5数据文件的路径，格式为 "ERA5_YYYYMMDD.nc"
        """
        return f"ERA5_{self.date:%Y%m%d}.nc"

    @property
    def cci_path(self):
        """
        基于日期属性生成CCI SST数据文件的路径。
        返回:
            str: CCI SST数据文件的路径，格式为 "CCI_SST_YYYYMMDD.nc"
        """
        return f"CCI_SST_{self.date:%Y%m%d}.nc"