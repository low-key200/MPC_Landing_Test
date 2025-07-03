"""
单步MPC计算测试程序 (世界坐标系规划)

=================================================================================
程序目的:
    本脚本用于在给定的初始条件下，执行单步的MPC（模型预测控制）计算。它不进行
    完整的仿真，而是专注于验证和调试MPC控制器在特定场景下的决策逻辑。通过打印
    核心的中间计算结果（如轨迹预测、参考轨迹）和最终的优化控制指令，可以快速

核心控制策略:
    本脚本采用“世界坐标系规划”策略：
    1. 获取无人机和移动平台在统一世界坐标系下的绝对状态。
    2. 基于平台的动力学模型，在世界坐标系中预测其未来N步的运动轨迹。
    3. 生成一个同样在世界坐标系下的无人机参考轨迹，该轨迹旨在引导无人机平滑地
       拦截平台预测的轨迹。
    4. 将无人机的世界坐标系状态作为初始条件，将世界坐标系参考轨迹作为目标，
       送入MPC求解器进行优化。

适用场景:
    - 快速验证算法在特定初始条件下的行为。
    - 调试轨迹预测、参考生成和代价函数构建等核心算法模块。
    - 在不运行完整可视化仿真的情况下，进行快速的单元测试。
=================================================================================
"""
import numpy as np
from math import cos, sin

# 从项目中导入必要的模块
from envs import QuadrotorLandingEnv, MovingPlatformDynamics, PlatformState
from utils import QuadMPC
import config.config as Config

# ==============================================================================
# 辅助函数
# ==============================================================================

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    将欧拉角 (roll, pitch, yaw) 转换为四元数 (qw, qx, qy, qz)。

    Args:
        roll (float): 滚转角 (弧度)。
        pitch (float): 俯仰角 (弧度)。
        yaw (float): 偏航角 (弧度)。

    Returns:
        np.ndarray: 对应的四元数 [qw, qx, qy, qz]。
    """
    cy, sy = cos(yaw * 0.5), sin(yaw * 0.5)
    cp, sp = cos(pitch * 0.5), sin(pitch * 0.5)
    cr, sr = cos(roll * 0.5), sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.array([qw, qx, qy, qz])

def calc_vz_ref(current_relative_z: float) -> float:
    """
    根据无人机与平台当前的相对高度，分阶段计算参考下降速度。

    Args:
        current_relative_z (float): 当前的Z轴相对距离。

    Returns:
        float: 目标参考垂直速度 (m/s)，负值代表下降。
    """
    if current_relative_z > 1.5:
        return -1.0  # 距离较远时，快速下降
    elif current_relative_z > 0.5:
        return -0.5  # 接近时，减速下降
    else:
        return -0.2  # 最后阶段，缓慢下降以确保安全

# ==============================================================================
# 核心算法模块
# ==============================================================================

def predict_platform_trajectory_world(
    current_platform_state: PlatformState, 
    platform_control: np.ndarray, 
    N: int, 
    dt: float
) -> dict:
    """
    在世界坐标系中，使用平台的动力学模型预测其未来N步的轨迹。

    Args:
        current_platform_state (PlatformState): 平台在当前时刻t0的完整世界状态。
        platform_control (np.ndarray): 平台在未来N步内保持不变的控制输入 [u1_accel, u2_steer]。
        N (int): MPC的预测时域长度。
        dt (float): 仿真时间步长。

    Returns:
        dict: 一个包含预测轨迹信息的字典，包括位置、速度和偏航角的序列。
    """
    # 初始化一个临时的动力学模型用于预测，避免影响主环境状态
    predictor = MovingPlatformDynamics()
    # 关键：从平台当前的世界状态开始进行预测
    predictor.state = current_platform_state.copy()
    
    # 准备用于存储预测结果的列表
    pred_pos, pred_vel, pred_psi = [], [], []
    control_dict = {'u1': platform_control[0], 'u2': platform_control[1]}

    # 在预测时域N内进行迭代
    for _ in range(N):
        # 调用动力学模型，向前演进一步
        predictor.step(control_dict, dt)
        state = predictor.state
        
        # 记录预测结果
        pred_pos.append([state.x, state.y, Config.MovingPlatform.HEIGHT])
        
        # 将平台体坐标系下的速度v，根据其世界坐标系偏航角psi，转换到世界坐标系速度
        vx_world = state.v * cos(state.psi)
        vy_world = state.v * sin(state.psi)
        pred_vel.append([vx_world, vy_world, 0.0]) # 假设平台Z轴速度为0
        
        pred_psi.append(state.psi)

    return {
        'pos': np.array(pred_pos),
        'vel': np.array(pred_vel),
        'psi': np.array(pred_psi)
    }

def generate_mpc_reference_trajectory_world(
    quadrotor_world_state: np.ndarray, 
    platform_trajectory_prediction: dict, 
    N: int
) -> np.ndarray:
    """
    在世界坐标系中，为MPC生成优化的无人机参考轨迹。
    核心思想是设计一条轨迹，使无人机能平滑地消除与平台预测轨迹之间的初始误差。

    Args:
        quadrotor_world_state (np.ndarray): 无人机当前的世界坐标系状态 [10x1]。
        platform_trajectory_prediction (dict): 平台未来N步的预测轨迹。
        N (int): MPC的预测时域长度。

    Returns:
        np.ndarray: 一个(nx, N)的矩阵，作为MPC的参考状态轨迹 `X_ref`。
    """
    nx = 10  # 状态向量维度
    x_ref = np.zeros((nx, N))
    
    # 获取无人机和平台的当前位置，计算初始位置误差
    current_quad_pos = quadrotor_world_state[:3]
    # 平台预测轨迹的第0个点即为平台在t0+dt时刻的预测位置，我们用它近似当前位置
    current_platform_pos_approx = platform_trajectory_prediction['pos'][0]
    initial_pos_error = current_quad_pos - current_platform_pos_approx

    # 遍历预测时域，为每一步生成参考状态
    for k in range(N):
        # --- 1. 参考位置 (p_ref) ---
        # 目标：让位置误差在N步内平滑地收敛到0。
        # 公式: p_ref_k = p_plat_pred_k + (initial_error * convergence_factor)
        # 收敛因子从接近1线性衰减到接近0，使误差修正量随时间减小。
        convergence_factor = 1.0 - (k + 1) / (N + 1)
        p_ref_k = platform_trajectory_prediction['pos'][k] + initial_pos_error * convergence_factor
        
        # --- 2. 参考速度 (v_ref) ---
        # 目标：除了跟随平台速度外，增加一个修正项来主动消除位置误差。
        # 公式: v_ref_k = v_plat_pred_k + v_correction
        v_correction = -initial_pos_error / (N * Config.DELTA_T) * 1.5 # 经验系数1.5
        v_ref_k = platform_trajectory_prediction['vel'][k] + v_correction * convergence_factor
        
        # 单独处理Z轴参考速度，以实现平稳着陆
        relative_z_k = p_ref_k[2] - platform_trajectory_prediction['pos'][k][2]
        v_ref_k[2] = calc_vz_ref(relative_z_k)
        
        # --- 3. 参考姿态 (q_ref) ---
        # 目标：让无人机的偏航角对齐平台的航向，滚转和俯仰保持水平，以获得稳定姿态。
        psi_plat_k = platform_trajectory_prediction['psi'][k]
        q_ref_k = euler_to_quaternion(0, 0, psi_plat_k)
        
        # --- 4. 组合成完整的参考状态向量 ---
        x_ref[:, k] = np.concatenate([p_ref_k, v_ref_k, q_ref_k])

    return x_ref

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    
    # --- 1. 定义测试场景的初始条件 ---
    # 通过修改这些参数，可以测试MPC控制器在不同初始情况下的响应。
    simulation_params = {
        'quad_init_position': np.array([2.0, -1.5, 5.0]),       # 无人机初始位置 [x, y, z]
        'quad_init_velocity': np.array([0.0, 0.0, 0.0]),       # 无人机初始速度 [vx, vy, vz]
        'quad_init_quaternions': euler_to_quaternion(0, 0, 0), # 无人机初始姿态
        
        'platform_init_state': np.array([0.0, 0.0, 0.8, np.deg2rad(30)]), # 平台初始状态 [x, y, v, psi]
        'platform_u1': 0.2,                                    # 平台恒定加速度
        'platform_u2': np.deg2rad(-30.0)                       # 平台恒定转向角
    }

    print("=" * 60)
    print("      单步MPC控制器计算测试 (世界坐标系规划)")
    print("=" * 60)
    print("\n[1] 初始条件:")
    for k, v in simulation_params.items():
        print(f"  - {k:<25}: {np.round(v, 4) if isinstance(v, np.ndarray) else v}")
    
    # --- 2. 初始化环境和MPC控制器 ---
    env = QuadrotorLandingEnv(dt=Config.DELTA_T)
    mpc_solver = QuadMPC(horizon=Config.MPC.HORIZON, dt=Config.DELTA_T)

    # 将初始参数应用到仿真环境中
    reset_params = {k: v for k, v in simulation_params.items() if 'quad' in k or 'platform_init' in k}
    _, info = env.reset(**reset_params)
    platform_control = np.array([simulation_params['platform_u1'], simulation_params['platform_u2']])

    print("\n[2] 开始执行单步MPC计算...\n")

    # --- 3. 执行单步MPC计算的核心逻辑 ---
    
    # 3.1 获取无人机和平台的当前世界状态
    quad_world_state = np.concatenate([
        info['quadrotor']['position'],
        info['quadrotor']['velocity'],
        info['quadrotor']['quaternions']
    ])
    current_platform_state = env.platform.state
    
    # 3.2 预测平台未来轨迹
    N = mpc_solver.N
    platform_traj_pred = predict_platform_trajectory_world(
        current_platform_state, platform_control, N, env.dt
    )
    
    # 3.3 生成MPC参考轨迹
    x_ref_val = generate_mpc_reference_trajectory_world(
        quad_world_state, platform_traj_pred, N
    )

    # --- 3.4 构建MPC代价函数 (min 0.5 * z' * diag(Q_vec) * z + p' * z) ---
    nx, nu = mpc_solver.nx, mpc_solver.nu
    q_weights = np.array(Config.MPC.STATE_WEIGHTS)   # 状态权重向量
    r_weights = np.array(Config.MPC.CONTROL_WEIGHTS) # 控制权重向量
    
    # 构建二次项矩阵 Q_nlp
    # 初始状态x0是固定的，不应被惩罚，因此其对应的代价矩阵块为零。
    # 因子 2 来自于 (z-z_ref)'*Q*(z-z_ref) 的展开: z'Qz - 2z_ref'Qz + const
    # 对应到NLP形式 0.5*z'*H*z + g'*z，H的对角线即为2*Q
    Q_nlp_val = np.concatenate([
        np.zeros(nx),                  # X_0 (初始状态) 的代价为0
        np.tile(2 * q_weights, N),     # X_1 到 X_N 的状态代价
        np.tile(2 * r_weights, N)      # U_0 到 U_{N-1} 的控制代价
    ])
    
    # 构建线性项代价函数的权重向量 p_nlp
    # 线性项 g'z 来自于 -2*z_ref'*Q*z
    p_nlp_list = [np.zeros(nx)] # 初始状态x0无线性代价
    for k in range(N):
        # 使用向量进行元素级乘法，等效于 Q @ x_ref，但效率更高
        p_nlp_list.append(-2 * q_weights * x_ref_val[:, k])
    p_nlp_list.append(np.zeros(nu * N)) # 控制量 u 无线性代价
    p_nlp_val = np.concatenate(p_nlp_list)
    
    # 3.5 调用MPC求解器，获取最优控制输入序列
    # 我们只关心序列中的第一个控制指令，它将在当前时间步被执行。
    u_opt_quad = mpc_solver.solve(
        quad_world_state, Q_nlp_val, p_nlp_val
        )

    # --- 4. 打印计算结果 ---
    print("-" * 60)
    print("               计算结果")
    print("-" * 60)
    
    print("\n[A] 最终计算出的最优控制输入 (u_0*):")
    print(f"  - 归一化推力 T_norm: {u_opt_quad[0]:.4f}")
    print(f"  - 归一化角速度 w_norm: {np.round(u_opt_quad[1:], 4)}")

    print("\n[B] 核心中间结果:")
    print("  - 平台轨迹预测 (未来3步的位置 [x, y, z]):")
    print(np.round(platform_traj_pred['pos'][:3], 4))

    print("\n  - MPC参考轨迹 (未来3步的状态):")
    print("    - 参考位置 [x, y, z]:")
    print(np.round(x_ref_val[:3, :3].T, 4))
    print("    - 参考速度 [vx, vy, vz]:")
    print(np.round(x_ref_val[3:6, :3].T, 4))

    print("\n" + "=" * 60)
    print("测试完成。")
    print("=" * 60)
