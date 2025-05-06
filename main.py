"""
CCI日变暖处理主流程 - 模块化版本
调用关系说明：
main.py → 各子模块 → config.py
"""
from datetime import datetime
from config import Config
from data_loader.era5_loader import load_era5_data
from data_loader.cci_loader import load_cci_sst
from preprocessing.regridding import bilinear_interpolation
from data_loader.physics_calc import (
    calculate_wind_stress,
    calculate_friction_velocity
)
from dwm.turbulence import compute_turbulent_diffusivity
from dwm.radiation import decompose_solar_radiation
from dwm.solver import run_dwm_simulation
from validation.era5_validator import validate_with_era5

def main():
    # 初始化配置，设置日期、CCI数据分辨率、ERA5数据分辨率
    cfg = Config(
        date=datetime(2023, 1, 1),
        cci_resolution=0.05,  # 5km
        era5_resolution=0.25  # 25km
    )

    try:
        # === 1. 数据加载 ===
        print("\nStage 1: 数据加载")
        # 加载ERA5数据
        era5 = load_era5_data(cfg.era5_path)
        # 加载CCI海表温度（SST）数据
        cci = load_cci_sst(cfg.cci_path)

        # === 2. 数据预处理 ===
        print("\nStage 2: 数据预处理")
        # 2.1 空间对齐，使用双线性插值将ERA5的风应力数据从其网格重采样到CCI的网格
        # tau_regrid：经过插值后重新网格化的风应力数据
        # bilinear_interpolation：指的是双线性插值
        tau_regrid = bilinear_interpolation(
            era5['tau'],
            src_grid=era5.grid,
            dst_grid=cci.grid
        )

        # 2.2 物理量计算，根据ERA5的10米风速计算摩擦速度
        # u_star：摩擦速度
        # calculate_friction_velocity：实现摩擦速度的公式计算，连接风应力和海水密度
        # wind_stress：风对海面的切应力，由风速数据计算得到
        # rho_water：海水密度，用于摩擦速度的单位转换和物理量纲平衡
        u_star = calculate_friction_velocity(
            wind_stress=calculate_wind_stress(era5['u10']),
            rho_water=cfg.rho_water
        )

        # === 3. DWM核心计算 ===
        print("\nStage 3: 日变暖模拟")
        # 3.1 辐射分解，将ERA5的太阳辐射数据分解为红外和可见光部分并计算
        # era5['Qsol']：太阳总辐射
        # K_IR（红外） 和 K_VIS（可见光）：控制不同波段辐射的衰减特性
        # decompose_solar_radiation：通过物理模型将总辐射拆分为不同波段，为后续的辐射传输和热平衡计算提供基础
        # 得到Q_components={Q_IR,Q_VIS},包含红外和可见光辐射分量的数据集
        Q_components = decompose_solar_radiation(
            era5['Qsol'],
            K_IR=cfg.K_IR,
            K_VIS=cfg.K_VIS
        )

        # 3.2 湍流系数，根据摩擦速度、风切变和稳定性参数计算湍流扩散系数
        # Kh：代表水平方向的湍流扩散系数
        # du_dz 表示风速的垂直切变
        # stability_param：稳定性参数，反映了浮力和切变对湍流的相对作用
        Kh = compute_turbulent_diffusivity(
            u_star=u_star,
            du_dz=era5['wind_shear'],
            stability_param=cfg.stability_param
        )

        # 3.3 运行模型，使用CCI初始海表温度、湍流扩散系数、净辐射等参数运行DWM模拟，模拟24个时间步长
        # run_dwm_simulation：执行海表温度模拟，耦合湍流扩散和热通量过程。
        # init_sst： 提供模拟的初始状态，基于观测数据确保真实性。
        # Q_net： 驱动 SST 的时间变化，净热通量的正负决定升温或降温。
        # timesteps/dt：控制模拟时长和时间分辨率，平衡精度与计算效率。
        sst_hourly = run_dwm_simulation(
            init_sst=cci['sst'],
            Kh=Kh,
            Q_net=Q_components,
            timesteps=24,
            tau=tau_regrid,
            dt=cfg.timestep
        )

        # === 4. 结果验证 ===
        print("\nStage 4: 验证评估")
        # 将DWM模拟结果聚合到25km分辨率，与ERA5的海表温度数据对比进行验证，返回验证报告
        validation_report = validate_with_era5(
            dwm_result=sst_hourly,
            era5_sst=era5['sst'],
            scale_factor=5  # 5km→25km
        )

        print(f"\n处理完成！验证报告:\n{validation_report}")

    except Exception as e:
        print(f"\n流程执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()

#这是我在主代码上做的改动