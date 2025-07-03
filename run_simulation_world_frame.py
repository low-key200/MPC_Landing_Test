"""
MPC控制器完整仿真与可视化程序 (世界坐标系规划)

=================================================================================
程序目的:
    本脚本用于对基于世界坐标系规划的MPC控制器进行完整的闭环仿真。它在一个预设的
    场景下运行无人机着陆任务，记录整个过程中的详细数据，并最终生成直观的分析
    图表和3D动画，以全面评估控制器的性能。

核心控制策略 (与 debug_mpc_step_world_frame.py 完全一致):
    1. 获取无人机和移动平台在统一世界坐标系下的绝对状态。
    2. 基于平台的动力学模型，在世界坐标系中预测其未来N步的运动轨迹。
    3. 生成一个同样在世界坐标系下的无人机参考轨迹，该轨迹旨在引导无人机平滑地
       拦截平台预测的轨迹。
    4. 将无人机的世界坐标系状态作为初始条件，将世界坐标系参考轨迹作为目标，
       送入MPC求解器进行优化，并在每个时间步重复此过程。

适用场景:
    - 对控制器进行端到端的性能验证，而不仅仅是单步决策。
    - 分析系统的暂态和稳态响应，如跟踪误差、收敛速度等。
    - 生成用于报告、演示或进一步分析的可视化结果（图表、动画）。
=================================================================================
"""
import numpy as np
from math import cos, sin
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# 从项目中导入必要的模块
from envs import QuadrotorLandingEnv, MovingPlatformDynamics, PlatformState
from utils import QuadMPC
import config.config as Config

# 设置matplotlib以正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 辅助函数 (与 debug_mpc_step_world_frame.py 完全一致)
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
# 核心算法模块 (与 debug_mpc_step_world_frame.py 完全一致)
# ==============================================================================

def predict_platform_trajectory_world(
    current_platform_state: PlatformState, 
    platform_control: np.ndarray, 
    N: int, 
    dt: float
) -> dict:
    """
    在世界坐标系中，使用平台的动力学模型预测其未来N步的轨迹。
    这是实现前瞻性跟踪的关键步骤。
    """
    predictor = MovingPlatformDynamics()
    predictor.state = current_platform_state.copy()
    pred_pos, pred_vel, pred_psi = [], [], []
    control_dict = {'u1': platform_control[0], 'u2': platform_control[1]}

    for _ in range(N):
        predictor.step(control_dict, dt)
        state = predictor.state
        pred_pos.append([state.x, state.y, Config.MovingPlatform.HEIGHT])
        vx_world = state.v * cos(state.psi)
        vy_world = state.v * sin(state.psi)
        pred_vel.append([vx_world, vy_world, 0.0])
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
    """
    nx = 10
    x_ref = np.zeros((nx, N))
    current_quad_pos = quadrotor_world_state[:3]
    current_platform_pos_approx = platform_trajectory_prediction['pos'][0]
    initial_pos_error = current_quad_pos - current_platform_pos_approx

    for k in range(N):
        convergence_factor = 1.0 - (k + 1) / (N + 1)
        p_ref_k = platform_trajectory_prediction['pos'][k] + initial_pos_error * convergence_factor
        
        v_correction = -initial_pos_error / (N * Config.DELTA_T) * 1.5
        v_ref_k = platform_trajectory_prediction['vel'][k] + v_correction * convergence_factor
        
        # 使用无人机在第k步的【参考高度】来计算相对高度，这比使用当前高度更具前瞻性。
        relative_z_k = p_ref_k[2] - platform_trajectory_prediction['pos'][k][2]
        v_ref_k[2] = calc_vz_ref(relative_z_k)
        
        psi_plat_k = platform_trajectory_prediction['psi'][k]
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
    print("--- 开始运行仿真 ---")
    # 1. 初始化环境和历史记录
    reset_params = {k: v for k, v in simulation_params.items() if 'quad' in k or 'platform_init' in k}
    platform_control = np.array([simulation_params['platform_u1'], simulation_params['platform_u2']])
    
    _, info = env.reset(**reset_params)
    history = {
        'time': [], 'quad_pos': [], 'quad_vel': [], 'quad_quat': [],
        'plat_pos': [], 'plat_vel': [], 'plat_psi': [],
        'rel_pos': [], 'control_input': []
    }

    # 2. 预加载MPC代价函数矩阵
    nx, nu, N = mpc_solver.nx, mpc_solver.nu, mpc_solver.N
    q_weights = np.array(Config.MPC.STATE_WEIGHTS)   # 状态权重向量
    r_weights = np.array(Config.MPC.CONTROL_WEIGHTS) # 控制权重向量

    # 3. 运行仿真主循环
    max_steps = int(Config.MAX_EPISODE_TIME / Config.DELTA_T)
    for step in tqdm(range(max_steps), desc="MPC仿真进度"):
        # 步骤 3.1: 获取当前世界状态
        quad_world_state = np.concatenate([
            info['quadrotor']['position'],
            info['quadrotor']['velocity'],
            info['quadrotor']['quaternions']
        ])
        current_platform_state = env.platform.state

        # 步骤 3.2: 预测平台未来轨迹，并生成MPC参考轨迹 (调用核心算法)
        platform_traj_pred = predict_platform_trajectory_world(
            current_platform_state, platform_control, N, env.dt
        )
        x_ref_val = generate_mpc_reference_trajectory_world(
            quad_world_state, platform_traj_pred, N
        )

        # 步骤 3.3: 构建当前步的代价函数参数
        Q_nlp_val = np.concatenate([
            np.zeros(nx),                  # X_0 (初始状态) 的代价为0
            np.tile(2 * q_weights, N),     # X_1 到 X_N 的状态代价
            np.tile(2 * r_weights, N)      # U_0 到 U_{N-1} 的控制代价
        ])

        p_nlp_list = [np.zeros(nx)] # 初始状态x0无线性代价
        for k in range(N):
            # 使用向量进行元素级乘法，等效于 Q @ x_ref，但效率更高
            p_nlp_list.append(-2 * q_weights * x_ref_val[:, k])
        p_nlp_list.append(np.zeros(nu * N)) # 控制量 u 无线性代价
        p_nlp_val = np.concatenate(p_nlp_list)

        # 步骤 3.4: 调用MPC求解器获取最优控制输入
        u_opt_quad = mpc_solver.solve(quad_world_state, Q_nlp_val, p_nlp_val)

        # 步骤 3.5: 将控制指令应用于环境，并执行一步仿真
        action = {'quadrotor': u_opt_quad, 'platform': platform_control}
        rel_obs, _, terminated, truncated, info = env.step(action)
        
        # 步骤 3.6: 记录当前步的数据
        history['time'].append(step * env.dt)
        history['quad_pos'].append(info['quadrotor']['position'])
        history['quad_vel'].append(info['quadrotor']['velocity'])
        history['quad_quat'].append(info['quadrotor']['quaternions'])
        history['plat_pos'].append(info['platform']['position'])
        history['plat_vel'].append(info['platform']['velocity'])
        history['plat_psi'].append(info['platform']['psi'])
        history['rel_pos'].append(rel_obs[:3])
        history['control_input'].append(u_opt_quad)

        # 步骤 3.7: 检查终止条件
        if terminated or truncated:
            break

    print(f"仿真结束于第 {step + 1} 步。成功状态: {info.get('success', False)}")
    print("--- 仿真完成 ---")
    
    # 将列表转换为numpy数组以便于后续处理
    for key, val in history.items():
        history[key] = np.array(val)
    return history


# ==============================================================================
# 可视化模块
# ==============================================================================

def plot_results(history: dict, output_dir: str):
    """生成并保存仿真过程中的关键状态图表。"""
    print("--- 正在生成静态图表 ---")
    os.makedirs(output_dir, exist_ok=True)
    time_ax = history['time']
    
    # 图1: 3D轨迹图
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

    # 图2: 相对位置与无人机速度
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
    
    # --- 1. 定义测试场景 ---
    # 在这里配置不同的初始条件和平台运动模式来测试控制器的鲁棒性。
    simulation_params = {
        'quad_init_position': np.array([2.0, -1.5, 5.0]),
        'quad_init_velocity': np.array([0.0, 0.0, 0.0]),
        'quad_init_quaternions': euler_to_quaternion(0, 0, np.deg2rad(0)),
        
        'platform_init_state': np.array([0.0, 0.0, 0.8, np.deg2rad(30)]),
        'platform_u1': 0.2,
        'platform_u2': np.deg2rad(-30.0)
    }

    print("=" * 60)
    print("      MPC控制器完整仿真程序 (世界坐标系规划)")
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
    output_directory = "simulation_results_world_frame"
    plot_results(simulation_history, output_directory)
    create_animation(simulation_history, output_directory)

    print(f"\n测试完成！所有结果已保存在 '{output_directory}' 文件夹中。")
