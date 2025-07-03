"""
基于CasADi的MPC求解器，用于无人机姿态和轨迹控制。

该求解器构建了一个非线性规划（NLP）问题，目标是最小化无人机在预测时域内
的状态误差和控制消耗，同时满足其动力学约束。
"""
import numpy as np
from casadi import *
import config.config as Config

# =============== CasADi 符号运算辅助函数 ===============
# 这些函数是MPC内部的实现细节，使用CasADi的符号变量(MX)进行运算。

def _quat_mult(q1, q2):
    """CasADi符号化的四元数乘法 q_new = q1 * q2。"""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return vertcat(w, x, y, z)

def _rotate_quat(q, v):
    """使用CasADi符号化的四元数 q 旋转向量 v。"""
    q_v = vertcat(0, v)
    q_conj = vertcat(q[0], -q[1], -q[2], -q[3])
    # v_rotated = q * v * q_conj
    ans = _quat_mult(_quat_mult(q, q_v), q_conj)
    return ans[1:4]  # 返回旋转后四元数的向量部分

class _QuadModel:
    """MPC控制器内部使用的无人机动力学模型（符号化版本）。"""
    def __init__(self):
        """从Config加载无人机参数。"""
        self.n_state = 10  # 状态维度: 位置(3) + 速度(3) + 四元数(4)
        self.n_ctrl = 4    # 控制维度: 归一化推力(1) + 归一化角速度(3)

        cfg_quad = Config.Quadrotor
        self.mass = cfg_quad.MASS
        self.g = Config.GRAVITY
        # 将归一化输入映射到物理单位的系数
        self.k_thrust = cfg_quad.THRUST_TO_WEIGHT_RATIO * self.mass * self.g
        self.k_omega = cfg_quad.OMEGA_MAX
        # 控制输入的边界
        self.T_min = cfg_quad.THRUST_MIN
        self.T_max = cfg_quad.THRUST_MAX

    def dynamics(self):
        """
        定义无人机动力学的常微分方程 (dx/dt = f(x, u))。
        这里的 x 和 u 都是CasADi的符号变量 (MX)。
        """
        # 定义状态变量
        p = MX.sym('p', 3)  # 位置
        v = MX.sym('v', 3)  # 速度
        q = MX.sym('q', 4)  # 四元数姿态
        x = vertcat(p, v, q)

        # 定义控制变量
        T_norm = MX.sym('T_norm', 1)  # 归一化推力
        w_norm = MX.sym('w_norm', 3)  # 归一化角速度
        u = vertcat(T_norm, w_norm)

        # 将归一化输入映射到物理单位
        thrust_force = T_norm * self.k_thrust
        omega_rads = w_norm * self.k_omega

        # 定义动力学方程
        g_vec = DM([0, 0, -self.g]) # 重力加速度向量
        # 速度导数: dv/dt = R(q) * T / m + g
        dv = _rotate_quat(q, vertcat(0, 0, thrust_force / self.mass)) + g_vec
        # 四元数导数: dq/dt = 0.5 * q * omega
        dq = 0.5 * _quat_mult(q, vertcat(0, omega_rads))
        # 状态导数向量
        x_dot = vertcat(v, dv, dq)

        # 创建一个CasADi函数，封装这个动力学模型
        return Function('f', [x, u], [x_dot], ['x', 'u'], ['x_dot'])

    def integrator(self, dt):
        """
        创建一个单步积分器，用于将状态从 t 推进到 t+dt。
        这里使用简单的欧拉法，并加入了四元数归一化。
        """
        x = MX.sym('x', self.n_state)
        u = MX.sym('u', self.n_ctrl)
        f = self.dynamics()
        x_dot = f(x, u)
        x_next_unnormalized = x + dt * x_dot

        # 提取并归一化四元数部分
        q_next_unnormalized = x_next_unnormalized[6:10]
        # 添加一个小的epsilon防止除以零
        q_next_normalized = q_next_unnormalized / (norm_2(q_next_unnormalized) + 1e-9)

        # 组合成新的状态向量
        x_next = vertcat(x_next_unnormalized[0:6], q_next_normalized)

        return Function('F', [x, u], [x_next], ['x_k', 'u_k'], ['x_k+1'])

# =============== MPC求解器主类 ===============

class QuadMPC:
    """四旋翼模型预测控制（MPC）求解器。"""
    def __init__(self, horizon: int, dt: float):
        self.N = horizon  # 预测时域
        self.dt = dt
        self.model = _QuadModel()
        self.nx = self.model.n_state
        self.nu = self.model.n_ctrl
        self.F = self.model.integrator(dt)  # 获取离散时间动力学模型
        self.z_guess = None
        self._create_solver()

    def _create_solver(self):
        """
        构建核心的非线性规划（NLP）问题并创建求解器。
        Q_nlp和p_nlp被定义为符号化参数，以便从外部传入。
        """
        # --- 决策变量 ---
        U = MX.sym('U', self.nu, self.N)
        X = MX.sym('X', self.nx, self.N + 1)
        z = vertcat(reshape(X, -1, 1), reshape(U, -1, 1))
        self.z_dim = z.shape[0]

        # --- 参数 (Parameters) ---
        X_init = MX.sym('X_init', self.nx)
        Q_nlp_sym = MX.sym('Q_nlp', self.z_dim, 1)
        p_nlp_sym = MX.sym('p_nlp', self.z_dim, 1)

        # --- 构建NLP形式的目标函数 J = 0.5*z^T*Q*z + p^T*z ---
        # z*z 是逐元素平方，dot是点积。这等价于 Sum(0.5 * Q_i * z_i^2)。
        obj = 0.5 * dot(z * z, Q_nlp_sym) + dot(p_nlp_sym, z)

        # --- 构建约束 (Constraints) ---
        g = []
        g.append(X[:, 0] - X_init)
        for k in range(self.N):
            x_next_pred = self.F(X[:, k], U[:, k])
            g.append(x_next_pred - X[:, k+1])
        g_vec = vertcat(*g)

        # --- 创建NLP求解器 ---
        nlp_params = vertcat(X_init, Q_nlp_sym, p_nlp_sym)
        nlp_prob = {
            'f': obj,
            'x': z,
            'g': g_vec,
            'p': nlp_params
        }

        opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': 'yes'}
        self.solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

        # --- 设置决策变量和约束的边界 ---
        num_constraints = g_vec.shape[0]
        self.lbg = [0] * num_constraints
        self.ubg = [0] * num_constraints

        lb_x = [-inf] * self.nx * (self.N + 1)
        ub_x = [inf] * self.nx * (self.N + 1)
        lb_u = ([self.model.T_min] + [-1.0]*3) * self.N
        ub_u = ([self.model.T_max] + [1.0]*3) * self.N
        self.lbx = lb_x + lb_u
        self.ubx = ub_x + ub_u

    def solve(self, x_init_val: np.ndarray, Q_nlp_val: np.ndarray, p_nlp_val: np.ndarray):
        """
        Args:
            x_init_val: 无人机当前状态 [nx, 1]。
            Q_nlp_val: 代价函数二次项的权重【向量】[z_dim, 1]。
            p_nlp_val: 代价函数线性项的权重向量 [z_dim, 1]。
        Returns:
            u_opt: 最优控制序列中的第一个控制输入 [nu, 1]。
        """
        # 检查是否存在可用的初始解（热启动），如果不存在（第一次运行），则创建默认初始解（冷启动）
        if self.z_guess is None:
            # 创建一个合理的冷启动初始解
            x_guess = np.tile(x_init_val, (self.N + 1, 1)).flatten()
            hover_thrust = (self.model.mass * self.model.g) / self.model.k_thrust
            u_guess = np.tile(np.array([hover_thrust, 0, 0, 0]), (self.N, 1)).flatten()
            self.z_guess = np.concatenate([x_guess, u_guess])

        # --- 将所有参数的数值打包 ---
        p_val = vertcat(x_init_val, Q_nlp_val, p_nlp_val)

        # 求解NLP
        res = self.solver(
            x0=self.z_guess, # 使用内部存储的初始解
            p=p_val,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg
        )

        # 用当前计算出的最优解更新内部存储的初始解，为下一次热启动做准备
        self.z_guess = res['x'].full().flatten()

        # 从解中提取最优控制序列
        u_opt_all = self.z_guess[self.nx * (self.N + 1):].reshape((self.N, self.nu))

        # 返回第一个控制动作
        return u_opt_all[0, :]
