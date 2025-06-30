"""
动力学模型文件 - 包含四旋翼和移动平台的数学模型
"""
from dataclasses import dataclass
import numpy as np
from math import sin, cos, tan, atan
import config.config as Config

# =============== 状态的数据结构定义 ===============
# 使用dataclass可以方便地创建带有类型提示的、结构化的数据容器。

@dataclass
class QuadrotorState:
    """封装四旋翼无人机的状态向量。"""
    position: np.ndarray  # 世界坐标系下的位置 [x, y, z]
    velocity: np.ndarray  # 世界坐标系下的速度 [vx, vy, vz]
    quaternions: np.ndarray  # 世界坐标系下的姿态四元数 [qw, qx, qy, qz]

    def copy(self):
        """创建当前状态的深拷贝，避免在传递中意外修改原始数据。"""
        return QuadrotorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternions=self.quaternions.copy()
        )

@dataclass
class PlatformState:
    """封装移动平台的状态向量（基于自行车模型）。"""
    x: float      # 世界坐标系下的x坐标
    y: float      # 世界坐标系下的y坐标
    v: float      # 平台自身的速度大小 (m/s)
    psi: float    # 平台的偏航角 (rad)，即车头朝向

    def copy(self):
        """创建当前状态的深拷贝。"""
        return PlatformState(x=self.x, y=self.y, v=self.v, psi=self.psi)

# =============== 四元数运算辅助函数 ===============

def quaternion_multiply(q1, q2):
    """
    执行四元数乘法 q_new = q1 * q2。
    这在组合旋转时非常有用。
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def rotate_vector_by_quaternion(q, v):
    """
    使用四元数 q 旋转一个三维向量 v。
    数学公式为: v_rotated = q * v * q_conjugate
    """
    # 将向量 v 提升为一个纯四元数 (w=0)
    q_v = np.array([0, v[0], v[1], v[2]])
    # 计算四元数 q 的共轭
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    # 执行旋转计算
    result_quat = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    # 返回旋转后四元数的向量部分
    return result_quat[1:4]

# =============== 四旋翼动力学模型 ===============

class QuadrotorDynamics:
    """模拟四旋翼无人机的飞行动力学。"""
    def __init__(self):
        """从配置文件加载无人机参数。"""
        self.mass = Config.Quadrotor.MASS
        self.g = Config.GRAVITY
        # 将归一化的推力(0-1)映射到实际的物理推力(牛顿)
        self.k_thrust = Config.Quadrotor.THRUST_TO_WEIGHT_RATIO * self.mass * self.g
        # 将归一化的角速度(-1到1)映射到实际的角速度(rad/s)
        self.k_omega = Config.Quadrotor.OMEGA_MAX
        self.state: QuadrotorState = None

    def compute_derivatives(self, state: QuadrotorState, control: dict):
        """
        根据当前状态和控制输入，计算状态的时间导数。
        Args:
            state: 当前的无人机状态。
            control: 一个字典，包含 {'thrust': T, 'omega': [wx, wy, wz]}。
        Returns:
            一个元组 (dp, dv, dq)，分别是位置、速度和四元数的导数。
        """
        # 将归一化的控制输入转换为物理单位
        T_force = control['thrust'] * self.k_thrust
        omega_rads = np.array(control['omega']) * self.k_omega

        # 1. 位置导数 (dp/dt) 就是当前的速度
        dp = state.velocity.copy()

        # 2. 速度导数 (dv/dt) 由推力和重力决定 (牛顿第二定律 F=ma)
        gravity_force = np.array([0, 0, -self.g * self.mass])
        # 推力在机体坐标系下是沿z轴的，需要通过姿态旋转到世界坐标系
        thrust_body = np.array([0, 0, T_force])
        thrust_world = rotate_vector_by_quaternion(state.quaternions, thrust_body)
        dv = (thrust_world + gravity_force) / self.mass

        # 3. 四元数导数 (dq/dt) 由角速度决定
        # 将角速度向量提升为纯四元数
        omega_quat = np.array([0, omega_rads[0], omega_rads[1], omega_rads[2]])
        dq = 0.5 * quaternion_multiply(state.quaternions, omega_quat)

        return dp, dv, dq

    def step(self, control: dict, dt: float):
        """
        使用前向欧拉法，将无人机状态推进一个时间步 dt。
        """
        if self.state is None:
            raise ValueError("无人机状态未初始化，请先调用 reset()")

        # 计算当前状态的导数
        dp, dv, dq = self.compute_derivatives(self.state, control)

        # 欧拉积分更新状态
        new_position = self.state.position + dp * dt
        new_velocity = self.state.velocity + dv * dt
        new_quaternions = self.state.quaternions + dq * dt

        # 四元数必须保持为单位向量，因此每次更新后需要归一化
        new_quaternions /= np.linalg.norm(new_quaternions)

        # 更新内部状态
        self.state = QuadrotorState(new_position, new_velocity, new_quaternions)

    def reset(self, init_position, init_velocity, init_quaternions):
        """重置无人机到指定的初始状态。"""
        self.state = QuadrotorState(
            position=init_position.copy(),
            velocity=init_velocity.copy(),
            quaternions=init_quaternions.copy()
        )

# =============== 移动平台动力学模型 ===============

class MovingPlatformDynamics:
    """模拟移动平台的运动（自行车模型）。"""
    def __init__(self):
        """从配置文件加载平台参数。"""
        self.l_f = Config.MovingPlatform.L_F
        self.l_r = Config.MovingPlatform.L_R
        self.v_max = Config.MovingPlatform.V_MAX
        self.state: PlatformState = None

    def _beta(self, u2: float):
        """计算车辆的侧滑角 (beta)。"""
        return atan(tan(u2) * self.l_r / (self.l_f + self.l_r))

    def compute_derivatives(self, state: PlatformState, u1: float, u2: float):
        """
        计算平台状态的时间导数。
        Args:
            state: 当前的平台状态。
            u1: 纵向加速度控制输入。
            u2: 前轮转向角控制输入。
        Returns:
            一个元组 (dx, dy, dv, dpsi)，分别是各状态量的导数。
        """
        beta_val = self._beta(u2)
        v, psi = state.v, state.psi

        dx = v * cos(psi + beta_val)
        dy = v * sin(psi + beta_val)
        dv = u1
        dpsi = (v / self.l_r) * sin(beta_val)

        return dx, dy, dv, dpsi

    def step(self, control: dict, dt: float):
        """使用前向欧拉法，将平台状态推进一个时间步 dt。"""
        if self.state is None:
            raise ValueError("平台状态未初始化，请先调用 reset()")

        u1 = control['u1']  # 纵向加速度
        u2 = control['u2']  # 前轮转向角

        # 计算导数
        dx, dy, dv, dpsi = self.compute_derivatives(self.state, u1, u2)

        # 欧拉积分更新状态
        new_v = self.state.v + dv * dt
        new_v = np.clip(new_v, 0, self.v_max)  # 限制速度不小于0且不超过最大值

        self.state = PlatformState(
            x=self.state.x + dx * dt,
            y=self.state.y + dy * dt,
            v=new_v,
            psi=self.state.psi + dpsi * dt
        )

    def reset(self, init_state: np.ndarray):
        """重置平台到指定的初始状态。"""
        self.state = PlatformState(
            x=init_state[0],
            y=init_state[1],
            v=init_state[2],
            psi=init_state[3]
        )
