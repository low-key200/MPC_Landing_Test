"""
该程序用于在特定初始条件下测试MPC控制器的性能，并生成
可视化图表和动画，以便于分析和调试算法。

核心改进：
- **引入平台轨迹预测**：不再使用平台当前位置作为静态目标，而是利用平台的动力学模型
  和已知控制输入，预测出未来N步的运动轨迹。
- **动态参考轨迹**：将预测出的平台轨迹作为MPC的动态参考目标，使无人机能够“预见”
  平台的运动并提前规划，解决了跟踪曲线运动时的滞后问题。
- **世界坐标系规划**：整个MPC的规划和求解都在统一的世界坐标系下进行，逻辑更清晰。
- **误差收敛轨迹设计**：在参考轨迹中加入了主动消除初始误差的项，引导无人机更快速、
  平滑地靠近并跟踪平台。
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

# 设置matplotlib以正确显示中文和负号，便于生成报告
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============== 辅助函数 ===============

def euler_to_quaternion(roll, pitch, yaw):
    """将欧拉角 (roll, pitch, yaw) 转换为四元数 (qw, qx, qy, qz)"""
    cy, sy = cos(yaw * 0.5), sin(yaw * 0.5)
    cp, sp = cos(pitch * 0.5), sin(pitch * 0.5)
    cr, sr = cos(roll * 0.5), sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz])


def calc_vz_ref(curr_z_rel):
    """根据当前相对高度，分段计算参考下降速度。"""
    if curr_z_rel > 1.5:
        return -1.0  # 离得远，快速下降
    elif curr_z_rel > 0.5:
        return -0.5 # 离得近，减速下降
    else:
        return -0.2  # 最后阶段，缓慢下降，确保安全接触

# ======================= 算法核心：轨迹预测与参考生成 ========================

def predict_platform_trajectory(current_plat_state: PlatformState, plat_control: np.ndarray, N: int, dt: float):
    """
    使用平台的动力学模型，预测其未来N步的轨迹。
    这是实现前瞻性跟踪的关键。
    Args:
        current_plat_state: 平台当前的完整状态 (PlatformState对象)。
        plat_control: 平台未来的控制输入 [u1_accel, u2_steer]。假设在N步内不变。
        N: 预测时域的长度。
        dt: 时间步长。
    Returns:
        一个字典，包含预测的位置、速度和偏航角序列，均为 (N, 3) 或 (N,) 的numpy数组。
    """
    # 创建一个临时的动力学模型用于预测，不影响主环境的状态
    predictor = MovingPlatformDynamics()
    predictor.state = current_plat_state.copy()

    # 准备存储预测结果的列表
    pred_pos, pred_vel, pred_psi = [], [], []

    control_dict = {'u1': plat_control[0], 'u2': plat_control[1]}

    for _ in range(N):
        # 演进一步
        predictor.step(control_dict, dt)
        state = predictor.state

        # 记录预测结果
        pred_pos.append([state.x, state.y, Config.MovingPlatform.HEIGHT])
        # 计算并记录世界坐标系下的速度
        vx_world = state.v * cos(state.psi)
        vy_world = state.v * sin(state.psi)
        pred_vel.append([vx_world, vy_world, 0.0])
        pred_psi.append(state.psi)

    return {
        'pos': np.array(pred_pos),
        'vel': np.array(pred_vel),
        'psi': np.array(pred_psi)
    }

def generate_mpc_reference_trajectory(quad_world_state, plat_traj_pred, N):
    """
    为MPC生成优化的参考轨迹。
    这个函数是新算法的核心，它将无人机引导向一个“移动的目标”。
    Args:
        quad_world_state: 无人机当前的世界坐标系状态 [10x1]。
        plat_traj_pred: `predict_platform_trajectory`生成的平台未来轨迹。
        N: 预测时域。
    Returns:
        一个 (nx, N) 的numpy数组，作为MPC的参考轨迹 `X_ref`。
    """
    nx = 10  # 状态维度
    x_ref = np.zeros((nx, N))

    # 获取无人机和平台的当前位置、速度
    current_quad_pos = quad_world_state[:3]
    current_platform_pos = plat_traj_pred['pos'][0]

    # 计算初始的位置和速度误差
    initial_pos_error = current_quad_pos - current_platform_pos

    # 遍历预测时域，为每一步生成参考状态
    for k in range(N):
        # 1. 参考位置 (p_ref):
        #    目标是让位置误差在N步内平滑地收敛到0。
        #    p_ref_k = p_plat_pred_k + (initial_error * convergence_factor)
        convergence_factor = 1.0 - (k + 1) / (N + 1) # 从接近1线性衰减到接近0
        p_ref_k = plat_traj_pred['pos'][k] + initial_pos_error * convergence_factor

        # 2. 参考速度 (v_ref):
        #    v_ref_k = v_plat_pred_k + (velocity_correction_term)
        #    速度修正项用于主动消除位置误差
        v_correction = -initial_pos_error / (N * Config.DELTA_T) * 1.5 # 经验系数1.5
        v_ref_k = plat_traj_pred['vel'][k] + v_correction * convergence_factor
        # 单独设置Z方向的参考速度，以实现平稳着陆
        relative_z = current_quad_pos[2] - plat_traj_pred['pos'][k][2]
        v_ref_k[2] = calc_vz_ref(relative_z)

        # 3. 参考姿态 (q_ref):
        #    让无人机的yaw角对齐平台的航向，roll和pitch保持水平。
        #    这样无人机可以更稳定地下降。
        psi_plat_k = plat_traj_pred['psi'][k]
        q_ref_k = euler_to_quaternion(0, 0, psi_plat_k)

        # 组合成完整的参考状态向量
        x_ref[:, k] = np.concatenate([p_ref_k, v_ref_k, q_ref_k])

    return x_ref

# ==============================================================

def run_mpc_simulation(env: QuadrotorLandingEnv, mpc_solver: QuadMPC, simulation_params):
    """
    运行单次MPC仿真并返回历史记录。
    """
    print("--- 开始运行仿真 ---")
    reset_params = {
        'quad_init_position': simulation_params['quad_init_position'],
        'quad_init_velocity': simulation_params['quad_init_velocity'],
        'quad_init_quaternions': simulation_params['quad_init_quaternions'],
        'platform_init_state': simulation_params['platform_init_state']
    }
    platform_control = np.array([
        simulation_params['platform_u1'],
        simulation_params['platform_u2']
    ])

    _, info = env.reset(**reset_params)
    history = {
        'time': [], 'quad_pos': [], 'quad_vel': [], 'quad_quat': [],
        'plat_pos': [], 'plat_vel': [], 'plat_psi': [],
        'rel_pos': []
    }

    # 【新】从Config加载基础权重矩阵Q和R，这些将在循环中用于计算Q_nlp和p_nlp
    nx, nu, N = mpc_solver.nx, mpc_solver.nu, mpc_solver.N
    Q = np.diag(Config.MPC.STATE_WEIGHTS)
    R = np.diag(Config.MPC.CONTROL_WEIGHTS)

    # 仿真循环
    max_steps = int(Config.MAX_EPISODE_TIME / Config.DELTA_T)
    for step in tqdm(range(max_steps), desc="MPC仿真进度"):
        # 1. 获取无人机和平台的当前世界状态
        quad_world_state = np.concatenate([
            info['quadrotor']['position'],
            info['quadrotor']['velocity'],
            info['quadrotor']['quaternions']
        ])
        current_platform_state = env.platform.state

        # 2. 预测平台未来轨迹，并生成MPC参考轨迹
        platform_trajectory_prediction = predict_platform_trajectory(
            current_platform_state, platform_control, N, env.dt
        )
        x_ref_val = generate_mpc_reference_trajectory(
            quad_world_state, platform_trajectory_prediction, N
        )

        # 3. 构建代价函数的 Q_nlp 和 p_nlp 参数
        # 3.1 构建 Q_nlp_val 矩阵
        q_nlp_blocks = [np.zeros((nx, nx))] + [2 * Q] * N + [2 * R] * N
        Q_nlp_val = block_diag(*q_nlp_blocks)

        # 3.2 构建 p_nlp_val 向量
        p_nlp_list = [np.zeros(nx)]
        for k in range(N):
            p_nlp_list.append(-2 * Q @ x_ref_val[:, k])
        p_nlp_list.append(np.zeros(nu * N))
        p_nlp_val = np.concatenate(p_nlp_list)

        # 4. 【新】调用更新后的MPC求解器
        #    求解器现在内部管理热启动，我们只需传入当前状态和代价函数参数。
        u_opt_quad = mpc_solver.solve(
            quad_world_state, Q_nlp_val, p_nlp_val
        )

        # 5. 在仿真环境中执行动作
        action = {'quadrotor': u_opt_quad, 'platform': platform_control}
        rel_obs, _, terminated, truncated, info = env.step(action)
        
        # 6. 记录历史数据用于绘图
        history['time'].append(step * env.dt)
        history['quad_pos'].append(info['quadrotor']['position'])
        history['quad_vel'].append(info['quadrotor']['velocity'])
        history['quad_quat'].append(info['quadrotor']['quaternions'])
        history['plat_pos'].append(info['platform']['position'])
        history['plat_vel'].append(info['platform']['velocity'])
        history['plat_psi'].append(info['platform']['psi'])
        history['rel_pos'].append(rel_obs[:3])

        # 7. 检查是否结束
        if terminated or truncated:
            break

    print(f"仿真结束于第 {step + 1} 步。成功: {info.get('success', False)}")
    print("--- 仿真完成 ---")
    for key, val in history.items():
        history[key] = np.array(val)
    return history


# =============== 可视化函数 ===============
def plot_results(history, output_dir):
    """生成并保存在仿真过程中的关键状态图表。"""
    print("--- 正在生成静态图表 ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 3D轨迹图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    quad_p = history['quad_pos']
    plat_p = history['plat_pos']
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

    # 2. 相对位置和速度图
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    time_ax = history['time']
    rel_p = history['rel_pos']
    axs[0].plot(time_ax, rel_p[:, 0], label='相对位置 x (m)')
    axs[0].plot(time_ax, rel_p[:, 1], label='相对位置 y (m)')
    axs[0].plot(time_ax, rel_p[:, 2], label='相对位置 z (m)')
    axs[0].axhline(y=Config.Termination.SUCCESS_XY_ERR_MAX, color='g', linestyle='--', label='水平成功边界')
    axs[0].axhline(y=-Config.Termination.SUCCESS_XY_ERR_MAX, color='g', linestyle='--')
    axs[0].set_ylabel('相对位置 (m)'), axs[0].set_title('相对位置随时间变化')
    axs[0].legend(), axs[0].grid(True)
    
    quad_v = history['quad_vel']
    axs[1].plot(time_ax, quad_v[:, 0], label='无人机速度 x (m/s)')
    axs[1].plot(time_ax, quad_v[:, 1], label='无人机速度 y (m/s)')
    axs[1].plot(time_ax, quad_v[:, 2], label='无人机速度 z (m/s)')
    axs[1].set_xlabel('时间 (s)'), axs[1].set_ylabel('无人机世界速度 (m/s)')
    axs[1].set_title('无人机世界速度随时间变化'), axs[1].legend(), axs[1].grid(True)
    plt.savefig(os.path.join(output_dir, '2_relative_and_absolute_states.png'))
    plt.close(fig)

    print(f"静态图表已保存至: {output_dir}")


def create_animation(history, output_dir, filename="mpc_landing_animation.gif"):
    """创建并保存整个着陆过程的3D动画。"""
    print("--- 正在创建动画 (这可能需要一些时间) ---")
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    quad_p, plat_p = history['quad_pos'], history['plat_pos']
    
    # 设置固定的坐标轴范围，避免动画抖动
    padding = 2.0
    x_min, x_max = min(quad_p[:, 0].min(), plat_p[:, 0].min()), max(quad_p[:, 0].max(), plat_p[:, 0].max())
    y_min, y_max = min(quad_p[:, 1].min(), plat_p[:, 1].min()), max(quad_p[:, 1].max(), plat_p[:, 1].max())
    ax.set_xlim(x_min - padding, x_max + padding), ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(0, quad_p[:, 2].max() + padding)
    ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
    ax.set_aspect('equal', 'box')

    # 初始化绘图元素
    quad_traj, = ax.plot([], [], [], 'b-', label='无人机轨迹')
    plat_traj, = ax.plot([], [], [], 'r--', label='平台轨迹')
    quad_pos, = ax.plot([], [], [], 'bo', markersize=8, label='无人机')
    plat_surface, = ax.plot([], [], [], 'g-', linewidth=5, label='平台表面')
    
    # 定义平台着陆区的四个角点
    l = Config.Termination.SUCCESS_XY_ERR_MAX
    corners = np.array([[l, -l], [l, l], [-l, l], [-l, -l], [l, -l]])

    def init():
        ax.legend()
        return quad_traj, plat_traj, quad_pos, plat_surface

    def update(i):
        # 更新轨迹线
        quad_traj.set_data_3d(quad_p[:i + 1, 0], quad_p[:i + 1, 1], quad_p[:i + 1, 2])
        plat_traj.set_data_3d(plat_p[:i + 1, 0], plat_p[:i + 1, 1], plat_p[:i + 1, 2])
        # 更新无人机当前位置
        quad_pos.set_data_3d([quad_p[i, 0]], [quad_p[i, 1]], [quad_p[i, 2]])
        # 更新平台表面位置和姿态
        px, py, pz = history['plat_pos'][i]
        psi = history['plat_psi'][i]
        R = np.array([[cos(psi), -sin(psi)], [sin(psi), cos(psi)]])
        rotated_corners = corners @ R.T + np.array([px, py])
        plat_surface.set_data_3d(rotated_corners[:, 0], rotated_corners[:, 1], [pz] * 5)
        ax.set_title(f'时间: {history["time"][i]:.1f}s')
        return quad_traj, plat_traj, quad_pos, plat_surface

    # 创建并保存动画
    anim = FuncAnimation(fig, update, frames=len(history['time']), init_func=init, blit=False, interval=50)
    filepath = os.path.join(output_dir, filename)
    with tqdm(total=len(history['time']), desc="正在保存GIF") as pbar:
        anim.save(filepath, writer='pillow', fps=20, progress_callback=lambda i, n: pbar.update(1))
    plt.show()
    plt.close(fig)
    print(f"动画已保存至: {filepath}")

# =============== 主程序入口 ===============
if __name__ == "__main__":
    # --- 1. 定义测试场景 ---
    simulation_params = {
        'quad_init_position': np.array([2.0, -1.5, 5.0]),
        'quad_init_velocity': np.array([0.0, 0.0, 0.0]),
        'quad_init_quaternions': euler_to_quaternion(0, 0, np.deg2rad(0)),
        
        'platform_init_state': np.array([0.0, 0.0, 0.8, np.deg2rad(30)]), # x, y, v, psi
        'platform_u1': 0.2,
        'platform_u2': np.deg2rad(-30.0)
    }

    print("MPC控制器测试程序 (V4: 内部热启动)")
    print("=" * 50)
    print("当前测试场景参数:")
    for k, v in simulation_params.items():
        if isinstance(v, np.ndarray):
            print(f"  - {k:<25}: {np.round(v, 3)}")
        else:
            print(f"  - {k:<25}: {v}")
    print("=" * 50)

    # --- 2. 初始化环境和MPC控制器 ---
    env = QuadrotorLandingEnv(dt=Config.DELTA_T)
    mpc_solver = QuadMPC(horizon=Config.MPC.HORIZON, dt=Config.DELTA_T)

    # --- 3. 运行仿真 ---
    simulation_history = run_mpc_simulation(env, mpc_solver, simulation_params)

    # --- 4. 生成结果 ---
    output_directory = "test_results"
    plot_results(simulation_history, output_directory)
    create_animation(simulation_history, output_directory)

    print("\n测试完成！结果已保存在 'test_results' 文件夹中。")
