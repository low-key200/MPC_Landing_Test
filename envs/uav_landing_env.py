"""
四旋翼无人机着陆环境（已废弃奖励和观测计算，仅用作仿真器）
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config.config as Config
from .dynamics import QuadrotorDynamics, MovingPlatformDynamics, quaternion_multiply


class QuadrotorLandingEnv(gym.Env):
    """
    四旋翼无人机在移动平台上的着陆环境。
    在当前MPC测试框架下，主要扮演一个集成了动力学模型的仿真器角色。
    """
    def __init__(self, dt=None):
        super().__init__()

        # 时间参数
        self.dt = dt if dt is not None else Config.DELTA_T
        self.max_steps = int(Config.MAX_EPISODE_TIME / self.dt)

        # 初始化动力学模型
        self.quadrotor = QuadrotorDynamics()
        self.platform = MovingPlatformDynamics()

        # 定义动作空间（尽管MPC直接输出，但为保持Gym接口完整性而定义）
        self.action_space = spaces.Dict({
            'quadrotor': spaces.Box(
                low=np.array([Config.Quadrotor.THRUST_MIN, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([Config.Quadrotor.THRUST_MAX, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
            'platform': spaces.Box(
                low=np.array([-2.0, -np.pi / 3], dtype=np.float32),
                high=np.array([2.0, np.pi / 3], dtype=np.float32),
                dtype=np.float32
            )
        })

        # 定义观测空间（同样为保持接口完整性）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.steps_count = 0

        # 从配置中加载终止条件参数
        self._load_termination_params()

    def _load_termination_params(self):
        """从配置文件加载终止条件参数"""
        term_cfg = Config.Termination
        self.SUCCESS_Z_MAX = term_cfg.SUCCESS_Z_MAX
        self.SUCCESS_XY_ERR_MAX = term_cfg.SUCCESS_XY_ERR_MAX
        self.CRASH_Z_MIN = term_cfg.CRASH_Z_MIN
        self.CONTACT_Z_THRESH = term_cfg.CONTACT_Z_THRESH
        self.MAX_LANDING_VEL = term_cfg.MAX_LANDING_VEL

    def reset(self, *, seed=None, options=None,
              quad_init_position: np.ndarray,
              quad_init_velocity: np.ndarray,
              quad_init_quaternions: np.ndarray,
              platform_init_state: np.ndarray):
        """
        重置环境到指定的初始状态。
        """
        super().reset(seed=seed)

        # 重置动力学模型
        self.quadrotor.reset(quad_init_position, quad_init_velocity, quad_init_quaternions)
        self.platform.reset(platform_init_state)

        # 重置计数器
        self.steps_count = 0
        info = self.get_info()
        # obs在MPC测试中不直接使用，但为保持接口一致性而返回
        obs = self._get_obs()
        return obs, info

    def step(self, action: dict):
        """
        执行一个动作，并使环境演进一个时间步。
        """
        self.steps_count += 1

        # 解析动作
        quad_action = {'thrust': action['quadrotor'][0], 'omega': action['quadrotor'][1:4]}
        platform_action = {'u1': action['platform'][0], 'u2': action['platform'][1]}

        # 更新动力学状态
        self.quadrotor.step(quad_action, self.dt)
        self.platform.step(platform_action, self.dt)

        # 检查终止条件
        obs = self._get_obs()
        terminated, success = self.check_done(obs)
        truncated = self.steps_count >= self.max_steps

        # 获取信息
        info = self.get_info(success)
        if truncated and not terminated:
            info["success"] = False

        # reward在MPC测试中不使用，返回0
        return obs, 0.0, terminated, truncated, info

    def _get_obs(self):
        """
        计算相对观测值（仅用于终止判断和调试）。
        MPC控制器将直接使用世界坐标系下的绝对状态。
        """
        p_state = self.platform.state
        q_state = self.quadrotor.state

        # 平台在世界坐标系下的速度
        plat_vel_world = np.array([p_state.v * np.cos(p_state.psi), p_state.v * np.sin(p_state.psi), 0.0])
        # 平台着陆面在世界坐标系下的位置
        plat_pos_world = np.array([p_state.x, p_state.y, Config.MovingPlatform.HEIGHT])

        # 计算相对位置和速度（世界坐标系下）
        rel_pos_world = q_state.position - plat_pos_world
        rel_vel_world = q_state.velocity - plat_vel_world

        # 将相对向量旋转到平台坐标系
        cos_psi, sin_psi = np.cos(-p_state.psi), np.sin(-p_state.psi)
        R_platform_to_world = np.array([[cos_psi, -sin_psi, 0], [sin_psi, cos_psi, 0], [0, 0, 1]])
        rel_pos_platform = R_platform_to_world @ rel_pos_world
        rel_vel_platform = R_platform_to_world @ rel_vel_world

        # 计算相对姿态
        plat_quat_world = np.array([np.cos(p_state.psi / 2), 0, 0, np.sin(p_state.psi / 2)])
        plat_quat_conj = np.array([plat_quat_world[0], -plat_quat_world[1], -plat_quat_world[2], -plat_quat_world[3]])
        rel_quat = quaternion_multiply(plat_quat_conj, q_state.quaternions)

        return np.concatenate([rel_pos_platform, rel_vel_platform, rel_quat]).astype(np.float32)

    def check_done(self, rel_obs: np.ndarray):
        """
        根据相对状态检查是否达到终止条件。
        """
        rel_pos = rel_obs[:3]
        rel_vel = rel_obs[3:6]
        x_rel, y_rel, z_rel = rel_pos
        actual_landing_speed = -rel_vel[2]  # 下降速度为正值

        # 1. 成功条件：在平台上方很近的位置，且水平误差小，速度小
        if (0 <= z_rel <= self.SUCCESS_Z_MAX and
                np.linalg.norm([x_rel, y_rel]) <= self.SUCCESS_XY_ERR_MAX):
            if actual_landing_speed < self.MAX_LANDING_VEL:
                return True, True  # 成功着陆
            else:
                return True, False  # 着陆过猛

        # 2. 失败条件
        if z_rel < self.CRASH_Z_MIN:  # 坠毁
            return True, False
        if z_rel <= self.CONTACT_Z_THRESH and np.linalg.norm([x_rel, y_rel]) > self.SUCCESS_XY_ERR_MAX:
             return True, False # 错过平台

        return False, False

    def get_info(self, success=False):
        """返回包含所有绝对状态的详细信息字典。"""
        p_state = self.platform.state
        q_state = self.quadrotor.state
        return {
            "quadrotor": {
                "position": q_state.position,
                "velocity": q_state.velocity,
                "quaternions": q_state.quaternions
            },
            "platform": {
                "position": np.array([p_state.x, p_state.y, Config.MovingPlatform.HEIGHT]),
                "velocity": np.array([p_state.v * np.cos(p_state.psi), p_state.v * np.sin(p_state.psi), 0.0]),
                "psi": p_state.psi
            },
            "steps": self.steps_count,
            "success": success
        }

    def close(self):
        pass
