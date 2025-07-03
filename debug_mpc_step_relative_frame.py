"""
单步MPC计算测试程序 (相对坐标系规划)

=================================================================================
程序目的:
    本脚本用于在给定的初始条件下，执行单步的MPC（模型预测控制）计算。它专注于
    验证和调试一种基于相对状态的控制策略。通过打印核心的中间计算结果和最终的
    优化控制指令，可以快速验证该策略的正确性。

核心控制策略:
    本脚本采用“相对坐标系规划”策略，其核心思想是简化问题：
    1. 首先，定义一个与移动平台固连的、随其运动和旋转的“浮动坐标系” O_p。
       在这个坐标系中，平台本身永远是原点。
    2. 从环境中直接获取无人机相对于平台的“相对状态”（位置、速度、姿态），这
       个相对状态被视为无人机在此浮动坐标系中的“绝对”状态。
    3. 预测平台相对于其自身（即原点）的未来运动轨迹。由于平台运动模型是基于
       其体轴速度的，这个预测非常直观。
    4. 生成一个同样在浮动坐标系下的无人机参考轨迹，其目标是驱动无人机从当前
       的相对位置收敛到原点（即平台中心）。
    5. 将无人机的相对状态作为初始条件，将旨在“归零”的相对参考轨迹作为目标，
       送入MPC求解器进行优化。

优势与适用场景:
    - 该策略将跟踪问题转化为一个更简单的镇定问题（稳定在原点），数学上更简洁。
    - 避免了在世界坐标系和机体坐标系之间进行复杂的坐标转换，减少了出错可能。
    - 同样适用于快速验证算法、调试模块和进行快速单元测试。
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
        return -1.0
    elif current_relative_z > 0.5:
        return -0.5
    else:
        return -0.2

# ==============================================================================
# 核心算法模块
# ==============================================================================

def predict_platform_trajectory_relative(
    platform_current_v_magnitude: float, 
    platform_control: np.ndarray, 
    N: int, 
    dt: float
) -> dict:
    """
    在平台的浮动坐标系中，预测其未来N步的轨迹。
    这个预测描述了平台相对于其当前位置和姿态的未来运动。

    Args:
        platform_current_v_magnitude (float): 平台当前的速度大小 (标量)。
        platform_control (np.ndarray): 平台在未来N步内保持不变的控制输入 [u1_accel, u2_steer]。
        N (int): MPC的预测时域长度。
        dt (float): 仿真时间步长。

    Returns:
        dict: 一个包含相对预测轨迹信息的字典，包括位置、速度和偏航角序列。
    """
    # 为了预测平台未来的相对运动，我们创建一个虚拟的初始状态：
    # 假设平台从原点(0,0)出发，初始偏航角为0，但具有当前时刻的真实速度。
    virtual_start_state = PlatformState(x=0.0, y=0.0, v=platform_current_v_magnitude, psi=0.0)

    predictor = MovingPlatformDynamics()
    predictor.state = virtual_start_state
    
    pred_pos, pred_vel, pred_psi = [], [], []
    control_dict = {'u1': platform_control[0], 'u2': platform_control[1]}

    for _ in range(N):
        predictor.step(control_dict, dt)
        state = predictor.state
        
        # 记录预测结果。在相对坐标系中，平台Z坐标始终为0。
        pred_pos.append([state.x, state.y, 0.0])
        
        # 计算并记录在这个浮动坐标系下的速度
        vx_local = state.v * cos(state.psi)
        vy_local = state.v * sin(state.psi)
        pred_vel.append([vx_local, vy_local, 0.0])
        
        pred_psi.append(state.psi)

    # 直接返回N步预测结果
    return {
        'pos': np.array(pred_pos),
        'vel': np.array(pred_vel),
        'psi': np.array(pred_psi)
    }

def generate_mpc_reference_trajectory_relative(
    quadrotor_relative_state: np.ndarray, 
    platform_traj_pred_relative: dict, 
    N: int
) -> np.ndarray:
    """
    在相对坐标系中，为MPC生成优化的无人机参考轨迹。
    核心思想是设计一条轨迹，驱动无人机从当前的相对状态平滑地收敛到原点。

    Args:
        quadrotor_relative_state (np.ndarray): 无人机当前相对于平台的完整状态 [10x1]。
        platform_traj_pred_relative (dict): 平台未来N步的相对预测轨迹。
        N (int): MPC的预测时域长度。

    Returns:
        np.ndarray: 一个(nx, N)的矩阵，作为MPC的参考状态轨迹 `X_ref`。
    """
    nx = 10
    x_ref = np.zeros((nx, N))
    
    # 在相对坐标系中，初始位置误差就是无人机的相对位置本身。
    # 目标是让这个误差在N步内收敛到0。
    initial_pos_error = quadrotor_relative_state[:3]
    
    for k in range(N):
        # --- 1. 参考位置 (p_ref) ---
        # 目标：从初始误差位置平滑地过渡到平台的未来相对位置（即原点）。
        # 公式: p_ref_k = p_plat_pred_k + (initial_error * convergence_factor)
        # 这里 p_plat_pred_k 代表了平台自身的运动，我们要跟随它并消除误差。
        convergence_factor = 1.0 - (k + 1) / (N + 1)
        p_ref_k = platform_traj_pred_relative['pos'][k] + initial_pos_error * convergence_factor
        
        # --- 2. 参考速度 (v_ref) ---
        # 目标：跟随平台的相对速度，并增加一个修正项来主动消除位置误差。
        v_correction = -initial_pos_error / (N * Config.DELTA_T) * 1.5
        v_ref_k = platform_traj_pred_relative['vel'][k] + v_correction * convergence_factor
        
        # 单独处理Z轴参考速度
        relative_z_k = p_ref_k[2] - platform_traj_pred_relative['pos'][k][2]
        v_ref_k[2] = calc_vz_ref(relative_z_k)
        
        # --- 3. 参考姿态 (q_ref) ---
        # 目标：让无人机的偏航角对齐平台的预测偏航角。
        psi_plat_k = platform_traj_pred_relative['psi'][k]
        q_ref_k = euler_to_quaternion(0, 0, psi_plat_k)
        
        # --- 4. 组合成完整的参考状态向量 ---
        x_ref[:, k] = np.concatenate([p_ref_k, v_ref_k, q_ref_k])

    return x_ref

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    
    # --- 1. 定义测试场景的初始世界坐标条件 ---
    # 即使在相对坐标系中规划，初始状态也需要在世界坐标系中定义，以便环境能正确设置。
    simulation_params = {
        'quad_init_position': np.array([2.0, -1.5, 5.0]),
        'quad_init_velocity': np.array([0.0, 0.0, 0.0]),
        'quad_init_quaternions': euler_to_quaternion(0, 0, 0),
        
        'platform_init_state': np.array([0.0, 0.0, 0.8, np.deg2rad(30)]),
        'platform_u1': 0.2,
        'platform_u2': np.deg2rad(-30.0)
    }

    print("=" * 60)
    print("      单步MPC控制器计算测试 (相对坐标系规划)")
    print("=" * 60)
    print("\n[1] 初始世界坐标系条件:")
    for k, v in simulation_params.items():
        print(f"  - {k:<25}: {np.round(v, 4) if isinstance(v, np.ndarray) else v}")
    
    # --- 2. 初始化环境和MPC求解器 ---
    env = QuadrotorLandingEnv(dt=Config.DELTA_T)
    mpc_solver = QuadMPC(horizon=Config.MPC.HORIZON, dt=Config.DELTA_T)

    reset_params = {k: v for k, v in simulation_params.items() if 'quad' in k or 'platform_init' in k}
    # 关键：env.reset() 返回的 obs 就是无人机的相对状态，可以直接用于规划
    obs, info = env.reset(**reset_params)
    platform_control = np.array([simulation_params['platform_u1'], simulation_params['platform_u2']])
    
    # --- 3. 获取无人机相对状态和平台信息 ---
    quad_relative_state = obs[:10]
    print("\n[2] 从环境中获取的无人机初始相对状态 (obs):")
    print(f"  - 相对位置: {np.round(quad_relative_state[:3], 4)}")
    print(f"  - 相对速度: {np.round(quad_relative_state[3:6], 4)}")
    print(f"  - 相对姿态: {np.round(quad_relative_state[6:], 4)}")
    
    print("\n[3] 开始执行单步MPC计算 (在相对坐标系中)...\n")
    
    # --- 4. 在相对坐标系中预测平台轨迹 ---
    N = mpc_solver.N
    platform_current_v_magnitude = env.platform.state.v
    
    platform_traj_pred_relative = predict_platform_trajectory_relative(
        platform_current_v_magnitude, platform_control, N, env.dt
    )
    
    # --- 5. 生成相对参考轨迹 ---
    x_ref_val = generate_mpc_reference_trajectory_relative(
        quad_relative_state, platform_traj_pred_relative, N
    )

    # --- 6. 构建代价函数并求解 ---
    # 此部分与世界坐标系版本完全相同，因为代价函数是基于状态误差和控制量定义的，
    # 无论状态是绝对的还是相对的，其数学形式不变。
    nx, nu = mpc_solver.nx, mpc_solver.nu
    q_weights = np.array(Config.MPC.STATE_WEIGHTS)   # 状态权重向量
    r_weights = np.array(Config.MPC.CONTROL_WEIGHTS) # 控制权重向量
    
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
    
    # 调用求解器。输入是无人机的【相对状态】和为之设计的【相对参考轨迹】。
    u_opt_quad = mpc_solver.solve(
        quad_relative_state, Q_nlp_val, p_nlp_val
    )

    # --- 7. 打印计算结果 ---
    print("-" * 60)
    print("               计算结果")
    print("-" * 60)
    
    print("\n[A] 最终计算出的最优控制输入 (u_0*):")
    print(f"  - 归一化推力 T_norm: {u_opt_quad[0]:.4f}")
    print(f"  - 归一化角速度 w_norm: {np.round(u_opt_quad[1:], 4)}")

    print("\n[B] 核心中间结果 (均为相对坐标系下):")
    print("  - 平台相对轨迹预测 (未来3步的位置 [x, y, z]):")
    print(np.round(platform_traj_pred_relative['pos'][:3], 4))
    
    print("\n  - MPC参考轨迹 (未来3步的状态):")
    print("    - 参考位置 [x, y, z]:")
    print(np.round(x_ref_val[:3, :3].T, 4))
    print("    - 参考速度 [vx, vy, vz]:")
    print(np.round(x_ref_val[3:6, :3].T, 4))

    print("\n" + "=" * 60)
    print("测试完成。")
    print("=" * 60)
