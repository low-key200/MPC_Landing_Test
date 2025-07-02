"""
MPC控制器完整仿真与可视化程序 (相对坐标系规划)

=================================================================================
程序目的:
    本脚本用于对基于相对坐标系规划的MPC控制器进行完整的闭环仿真。它在一个预设的
    场景下运行无人机着陆任务，记录整个过程中的详细数据，并最终生成直观的分析
    图表和3D动画，以全面评估控制器的性能。

核心控制策略 (与 debug_mpc_step_relative_frame.py 完全一致):
    1. 定义一个与移动平台固连的“浮动坐标系”，在此坐标系中平台始终是原点。
    2. 在每个时间步，从仿真环境中直接获取无人机相对于平台的“相对状态”。
    3. 预测平台相对于其自身的未来运动轨迹。
    4. 生成一个旨在驱动无人机从当前相对状态收敛到原点（即平台）的相对参考轨迹。
    5. 将无人机的【相对状态】和【相对参考轨迹】送入MPC求解器进行优化，并重复此过程。

适用场景:
    - 对基于相对状态的控制器进行端到端的性能验证。
    - 将跟踪问题简化为镇定问题，分析其收敛特性和鲁棒性。
    - 生成用于报告、演示或与世界坐标系策略进行性能对比的可视化结果。
=================================================================================
"""
import numpy as np
from math import cos, sin
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.linalg import block_diag

# 从项目中导入必要的模块
from envs import QuadrotorLandingEnv, MovingPlatformDynamics, PlatformState
from utils import QuadMPC
import config.config as Config

# 设置matplotlib以正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 辅助函数 (与 debug_mpc_step_relative_frame.py 完全一致)
# ==============================================================================

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """将欧拉角 (roll, pitch, yaw) 转换为四元数 (qw, qx, qy, qz)。"""
    cy, sy = cos(yaw * 0.5), sin(yaw * 0.5)
    cp, sp = cos(pitch * 0.5), sin(pitch * 0.5)
    cr, sr = cos(roll * 0.5), sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz])

def calc_vz_ref(current_relative_z: float) -> float:
    """根据当前相对高度，分阶段计算参考下降速度。"""
    if current_relative_z > 1.5:
        return -1.0
    elif current_relative_z > 0.5:
        return -0.5
    else:
        return -0.2

# ==============================================================================
# 核心算法模块 (与 debug_mpc_step_relative_frame.py 完全一致)
# ==============================================================================

def predict_platform_trajectory_relative(
    platform_current_v_magnitude: float,
    platform_control: np.ndarray,
    N: int,
    dt: float
) -> dict:
    """
    在平台的浮动坐标系中，预测其相对于自身的未来N步轨迹。
    """
    virtual_start_state = PlatformState(x=0.0, y=0.0, v=platform_current_v_magnitude, psi=0.0)
    predictor = MovingPlatformDynamics()
    predictor.state = virtual_start_state
    
    pred_pos, pred_vel, pred_psi = [], [], []
    control_dict = {'u1': platform_control[0], 'u2': platform_control[1]}

    for _ in range(N):
        predictor.step(control_dict, dt)
        state = predictor.state
        pred_pos.append([state.x, state.y, 0.0])
        vx_local = state.v * cos(state.psi)
        vy_local = state.v * sin(state.psi)
        pred_vel.append([vx_local, vy_local, 0.0])
        pred_psi.append(state.psi)

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
    在相对坐标系中，为MPC生成优化的无人机参考轨迹，引导其收敛至原点。
    """
    nx = 10
    x_ref = np.zeros((nx, N))
    initial_pos_error = quadrotor_relative_state[:3]
    
    for k in range(N):
        convergence_factor = 1.0 - (k + 1) / (N + 1)
        p_ref_k = platform_traj_pred_relative['pos'][k] + initial_pos_error * convergence_factor
        
        v_correction = -initial_pos_error / (N * Config.DELTA_T) * 1.5
        v_ref_k = platform_traj_pred_relative['vel'][k] + v_correction * convergence_factor
        
        relative_z_k = p_ref_k[2] - platform_traj_pred_relative['pos'][k][2]
        v_ref_k[2] = calc_vz_ref(relative_z_k)
        
        psi_plat_k = platform_traj_pred_relative['psi'][k]
        q_ref_k = euler_to_quaternion(0, 0, psi_plat_k)
        
        x_ref[:, k] = np.concatenate([p_ref_k, v_ref_k, q_ref_k])

    return x_ref

# ==============================================================================
# 仿真与执行模块
# ==============================================================================

def run_mpc_simulation(env: QuadrotorLandingEnv, mpc_solver: QuadMPC, simulation_params: dict) -> dict:
    """
    运行完整的MPC闭环仿真，并返回包含所有历史数据的字典。

    Args:
        env (QuadrotorLandingEnv): 配置好的仿真环境实例。
        mpc_solver (QuadMPC): MPC控制器实例。
        simulation_params (dict): 包含场景所有初始条件的字典。

    Returns:
        dict: 包含时间、状态等历史序列的字典。
    """
    print("--- 开始运行仿真 (相对坐标系) ---")
    # 1. 初始化环境和历史记录
    reset_params = {k: v for k, v in simulation_params.items() if 'quad' in k or 'platform_init' in k}
    platform_control = np.array([simulation_params['platform_u1'], simulation_params['platform_u2']])
    
    # 关键：obs直接就是无人机的相对状态，是MPC的输入
    obs, info = env.reset(**reset_params)
    history = {
        'time': [], 'quad_pos': [], 'quad_vel': [], 'quad_quat': [],
        'plat_pos': [], 'plat_vel': [], 'plat_psi': [],
        'rel_pos': [], 'control_input': []
    }

    # 2. 预加载MPC代价函数矩阵
    nx, nu, N = mpc_solver.nx, mpc_solver.nu, mpc_solver.N
    Q = np.diag(Config.MPC.STATE_WEIGHTS)
    R = np.diag(Config.MPC.CONTROL_WEIGHTS)

    # 3. 运行仿真主循环
    max_steps = int(Config.MAX_EPISODE_TIME / Config.DELTA_T)
    for step in tqdm(range(max_steps), desc="MPC仿真进度 (相对)"):
        # 步骤 3.1: 获取当前观测(相对状态)和信息(绝对状态)
        current_relative_obs = obs 
        
        # ==================== 新增的速度修正部分 ====================
        # 步骤 3.1.1: 为MPC准备正确的初始状态
        # MPC的动力学模型假设其输入速度是【惯性速度】。
        # 而环境提供的 `obs` 中的速度是【相对速度】。
        # 因此，我们必须将无人机的世界惯性速度，通过坐标旋转，
        # 转换到当前的平台坐标系下，作为MPC的输入。
        # 从info字典获取所需的世界坐标系信息
        quad_world_velocity = info['quadrotor']['velocity']
        platform_psi = info['platform']['psi']

        # 构建从世界坐标系到平台坐标系的旋转矩阵 (绕Z轴旋转 -psi)
        cos_psi, sin_psi = np.cos(platform_psi), np.sin(platform_psi)
        # R_world_to_platform = (R_platform_to_world)^T
        R_world_to_platform = np.array([
            [ cos_psi,  sin_psi, 0],
            [-sin_psi,  cos_psi, 0],
            [   0,        0,     1]
        ])
        
        # 将无人机的世界惯性速度旋转到平台坐标系下
        quad_inertial_vel_in_platform_frame = R_world_to_platform @ quad_world_velocity
        
        # 组合成MPC真正需要的初始状态向量
        mpc_input_state = np.concatenate([
            current_relative_obs[:3],                  # 相对位置 p_rel (正确)
            quad_inertial_vel_in_platform_frame,       # 修正后的惯性速度 v_inertial
            current_relative_obs[6:]                   # 相对姿态 q_rel (正确)
        ])
        # ==========================================================

        # 步骤 3.2: 预测平台相对轨迹，并生成相对参考轨迹
        platform_current_v_magnitude = env.platform.state.v
        platform_traj_pred_rel = predict_platform_trajectory_relative(
            platform_current_v_magnitude, platform_control, N, env.dt
        )
        # 使用修正后的状态来生成参考轨迹
        x_ref_val = generate_mpc_reference_trajectory_relative(
            mpc_input_state, platform_traj_pred_rel, N
        )

        # 步骤 3.3: 构建代价函数
        q_nlp_blocks = [np.zeros((nx, nx))] + [2 * Q] * N + [2 * R] * N
        Q_nlp_val = block_diag(*q_nlp_blocks)

        p_nlp_list = [np.zeros(nx)]
        for k in range(N):
            p_nlp_list.append(-2 * Q @ x_ref_val[:, k])
        p_nlp_list.append(np.zeros(nu * N))
        p_nlp_val = np.concatenate(p_nlp_list)

        # 步骤 3.4: 调用MPC求解器，输入修正后的状态
        u_opt_quad = mpc_solver.solve(mpc_input_state, Q_nlp_val, p_nlp_val)

        # 步骤 3.5: 执行仿真
        action = {'quadrotor': u_opt_quad, 'platform': platform_control}
        obs, _, terminated, truncated, info = env.step(action)
        
        # 步骤 3.6: 记录数据 (记录原始的相对位置，以便于观察误差收敛)
        history['time'].append(step * env.dt)
        history['rel_pos'].append(current_relative_obs[:3]) # 记录修正前的相对位置
        history['control_input'].append(u_opt_quad)
        history['quad_pos'].append(info['quadrotor']['position'])
        history['quad_vel'].append(info['quadrotor']['velocity'])
        history['quad_quat'].append(info['quadrotor']['quaternions'])
        history['plat_pos'].append(info['platform']['position'])
        history['plat_vel'].append(info['platform']['velocity'])
        history['plat_psi'].append(info['platform']['psi'])
        
        # 步骤 3.7: 检查终止条件
        if terminated or truncated:
            break

    print(f"仿真结束于第 {step + 1} 步。成功状态: {info.get('success', False)}")
    print("--- 仿真完成 ---")
    
    for key, val in history.items():
        history[key] = np.array(val)
    return history


# ==============================================================================
# 可视化模块 (与世界坐标系版本完全相同，因为它们都依赖于记录的世界坐标历史数据)
# ==============================================================================

def plot_results(history: dict, output_dir: str):
    """生成并保存仿真过程中的关键状态图表。"""
    print("--- 正在生成静态图表 ---")
    os.makedirs(output_dir, exist_ok=True)
    time_ax = history['time']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    quad_p, plat_p = history['quad_pos'], history['plat_pos']
    ax.plot(quad_p[:, 0], quad_p[:, 1], quad_p[:, 2], label='无人机轨迹', color='b')
    ax.plot(plat_p[:, 0], plat_p[:, 1], plat_p[:, 2], label='平台轨迹', color='r', linestyle='--')
    ax.scatter(quad_p[0, 0], quad_p[0, 1], quad_p[0, 2], c='blue', s=50, marker='o', label='无人机起点')
    ax.scatter(quad_p[-1, 0], quad_p[-1, 1], quad_p[-1, 2], c='blue', s=80, marker='*', label='无人机终点')
    ax.scatter(plat_p[0, 0], plat_p[0, 1], plat_p[0, 2], c='red', s=50, marker='o', label='平台起点')
    ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
    ax.set_title('3D 轨迹对比'), ax.legend(), ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.savefig(os.path.join(output_dir, '1_trajectory_3d.png'))
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    rel_p, quad_v = history['rel_pos'], history['quad_vel']
    axs[0].plot(time_ax, rel_p[:, 0], label='相对位置 x')
    axs[0].plot(time_ax, rel_p[:, 1], label='相对位置 y')
    axs[0].plot(time_ax, rel_p[:, 2], label='相对位置 z')
    axs[0].axhline(y=Config.Termination.SUCCESS_XY_ERR_MAX, color='g', linestyle='--', label='水平成功边界')
    axs[0].axhline(y=-Config.Termination.SUCCESS_XY_ERR_MAX, color='g', linestyle='--')
    axs[0].set_ylabel('相对位置 (m)'), axs[0].set_title('相对位置随时间变化'), axs[0].legend(), axs[0].grid(True)
    
    axs[1].plot(time_ax, quad_v[:, 0], label='无人机速度 vx')
    axs[1].plot(time_ax, quad_v[:, 1], label='无人机速度 vy')
    axs[1].plot(time_ax, quad_v[:, 2], label='无人机速度 vz')
    axs[1].set_xlabel('时间 (s)'), axs[1].set_ylabel('无人机世界速度 (m/s)')
    axs[1].set_title('无人机世界速度随时间变化'), axs[1].legend(), axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_relative_and_velocity_states.png'))
    plt.close(fig)

    print(f"静态图表已保存至: {output_dir}")


def create_animation(history: dict, output_dir: str, filename="simulation_animation.gif"):
    """创建并保存整个着陆过程的3D动画。"""
    print("--- 正在创建动画 (这可能需要一些时间) ---")
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    quad_p, plat_p = history['quad_pos'], history['plat_pos']
    
    padding = 2.0
    all_x = np.concatenate((quad_p[:, 0], plat_p[:, 0]))
    all_y = np.concatenate((quad_p[:, 1], plat_p[:, 1]))
    ax.set_xlim(all_x.min() - padding, all_x.max() + padding)
    ax.set_ylim(all_y.min() - padding, all_y.max() + padding)
    ax.set_zlim(0, quad_p[:, 2].max() + padding)
    ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
    ax.set_aspect('equal', 'box')

    quad_traj, = ax.plot([], [], [], 'b-', label='无人机轨迹')
    plat_traj, = ax.plot([], [], [], 'r--', label='平台轨迹')
    quad_pos, = ax.plot([], [], [], 'bo', markersize=8, label='无人机')
    plat_surface, = ax.plot([], [], [], 'g-', linewidth=5, label='平台表面')
    
    l = Config.Termination.SUCCESS_XY_ERR_MAX
    corners = np.array([[l, -l], [l, l], [-l, l], [-l, -l], [l, -l]])

    def init():
        ax.legend()
        return quad_traj, plat_traj, quad_pos, plat_surface

    def update(i):
        quad_traj.set_data_3d(quad_p[:i + 1, 0], quad_p[:i + 1, 1], quad_p[:i + 1, 2])
        plat_traj.set_data_3d(plat_p[:i + 1, 0], plat_p[:i + 1, 1], plat_p[:i + 1, 2])
        quad_pos.set_data_3d([quad_p[i, 0]], [quad_p[i, 1]], [quad_p[i, 2]])
        
        px, py, pz = history['plat_pos'][i]
        psi = history['plat_psi'][i]
        R = np.array([[cos(psi), -sin(psi)], [sin(psi), cos(psi)]])
        rotated_corners = corners @ R.T + np.array([px, py])
        plat_surface.set_data_3d(rotated_corners[:, 0], rotated_corners[:, 1], [pz] * 5)
        ax.set_title(f'时间: {history["time"][i]:.1f}s')
        return quad_traj, plat_traj, quad_pos, plat_surface

    filepath = os.path.join(output_dir, filename)
    anim = FuncAnimation(fig, update, frames=len(history['time']), init_func=init, blit=False, interval=50)
    with tqdm(total=len(history['time']), desc="正在保存GIF") as pbar:
        anim.save(filepath, writer='pillow', fps=20, progress_callback=lambda i, n: pbar.update(1))
    plt.close(fig)
    print(f"动画已保存至: {filepath}")

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    
    # --- 1. 定义测试场景 (世界坐标系下) ---
    # 仿真环境的初始化总是基于世界坐标系的，无论控制策略如何。
    simulation_params = {
        'quad_init_position': np.array([2.0, -1.5, 5.0]),
        'quad_init_velocity': np.array([0.0, 0.0, 0.0]),
        'quad_init_quaternions': euler_to_quaternion(0, 0, np.deg2rad(0)),
        
        'platform_init_state': np.array([0.0, 0.0, 0.8, np.deg2rad(30)]),
        'platform_u1': 0.2,
        'platform_u2': np.deg2rad(-30.0)
    }

    print("=" * 60)
    print("      MPC控制器完整仿真程序 (相对坐标系规划)")
    print("=" * 60)
    print("当前测试场景参数:")
    for k, v in simulation_params.items():
        print(f"  - {k:<25}: {np.round(v, 4) if isinstance(v, np.ndarray) else v}")
    print("=" * 60)

    # --- 2. 初始化环境和MPC控制器 ---
    env = QuadrotorLandingEnv(dt=Config.DELTA_T)
    mpc_solver = QuadMPC(horizon=Config.MPC.HORIZON, dt=Config.DELTA_T)

    # --- 3. 运行仿真 ---
    simulation_history = run_mpc_simulation(env, mpc_solver, simulation_params)

    # --- 4. 生成并保存结果 ---
    output_directory = "simulation_results_relative_frame"
    plot_results(simulation_history, output_directory)
    create_animation(simulation_history, output_directory)

    print(f"\n测试完成！所有结果已保存在 '{output_directory}' 文件夹中。")
