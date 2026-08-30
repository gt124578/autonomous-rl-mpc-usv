"""
USV_dynamics_model.py

兼容性说明：
- 该文件当前保留为历史/对照用的非可微动力学实现，本轮坐标系统一不修改其数学实现。
- 全系统的唯一主坐标系应以可微版本和训练/部署主链路为准：
  `x` 向东，`y` 向北，`psi` 以东向为零且逆时针为正，`r>0` 表示左转。
- 若后续仍需调用本文件，请先在调用侧确认其输入/输出已与主坐标系一致，避免与旧脚本中的
  历史坐标解释混用。

说明：
- 该文件将论文《Intelligent Agile Control of Unmanned Surface Vehicles with End-to-end Reinforcement Learning》（Zhou 等）中第 III 节的动力学模型
  与用户提供的 CustomEnv.py 中的实现提取并整合，形成独立的 Python 模块，便于在仿真或 RL 环境中复用。

模块包含：
- USVParameters: 保存论文表格中给出的 a_ij 等参数及运动约束。
- USVDynamics: 提供核心动力学计算方法，包括计算加速度（u_a, v_a, beta）和状态积分（一步更新）。
- 可选 residual 接口：保留从 CustomEnv.py 中调用的外部残差（例如 KNN 或 GP）位置，若无残差可传入零向量。

使用示例：
>>> from USV_dynamics_model import USVParameters, USVDynamics
>>> params = USVParameters(pwm=1.0)
>>> dyn = USVDynamics(params)
>>> state = np.array([x, y, psi, u, v, r], dtype=np.float32)
>>> action = np.array([n_left, n_right], dtype=np.float32)
>>> residual = np.zeros(3, dtype=np.float32)  # 若有外部学得残差模型，传入预测值
>>> next_state, accs = dyn.step(state, action, residual)

注意：文件中含有中文注释，方便直接阅读与二次开发。

引用：
- Zhou, Zixiang 等, "Intelligent Agile Control of Unmanned Surface Vehicles with End-to-end Reinforcement Learning", 论文第 III 节动力学模型。
- CustomEnv.py（用户提供）中的动力学实现逻辑与参数值被整合到本模块。
"""

import numpy as np
import math
from dataclasses import dataclass

@dataclass
class USVParameters:
    """存放动力学参数与约束（来自论文与 CustomEnv.py 表/实现）"""
    # 动力学系数（a11..a33 与 a12,a13.. 等），注意 a12 和 a33 在实现中乘以 pwm
    a11: float = -0.39855766
    a12_coeff: float = 0.00142244  # 与 pwm 相乘
    a13: float = -3.2420864
    a14: float = -0.93031027

    a21: float = -0.06203627
    a22: float = -0.22806704

    a31: float = -0.350531373
    a32: float = -0.838886101
    a33_coeff: float = 3.59390565e-4  # 与 pwm 相乘

    # 物理与动作约束（来自 Table II 与 config）
    pwm: float = 1.0
    dt: float = 0.5
    u_max: float = 1.0
    v_max: float = 0.1
    r_max: float = 0.3
    u_a_max: float = 0.5
    v_a_max: float = 0.01
    beta_max: float = 0.1

class USVDynamics:
    """USV 简化 3-DOF 平面动力学模型实现，复现论文与 CustomEnv.py 中的行为。

    状态向量 state = [x, y, psi, u, v, r]
    动作 action = [n_left, n_right] (PWM 比例或归一化推力信号)
    residual = 外部残差模型输出，形状 (3,) 对应于 u_a, v_a, beta 的额外项
    """

    def __init__(self, params: USVParameters):
        self.p = params
        # 以与 CustomEnv.py 一致的矩阵形式存储 a_ij
        self.dynamic_a = np.array([
            [self.p.a11, self.p.a12_coeff * self.p.pwm, self.p.a13, self.p.a14],
            [self.p.a21, self.p.a22, 0.0, 0.0],
            [self.p.a31, self.p.a32, self.p.a33_coeff * self.p.pwm, 0.0]
        ], dtype=np.float32)

    def compute_accelerations(self, state: np.ndarray, action: np.ndarray, residual: np.ndarray = None):
        """计算瞬时加速度 u_a, v_a, beta（与 CustomEnv.step 内逻辑一致）

        参数：
            state: [x,y,psi,u,v,r]
            action: [n_left, n_right]
            residual: 外部残差 (3,) ，若为 None 则视为零。
        返回：
            (u_a, v_a, beta) 三元组
        """
        # 状态拆分
        _, _, psi, u, v, r = state
        n_left, n_right = action

        # 动力学项计算（与论文(3)式一致的实现形式）
        u_a = (self.dynamic_a[0, 0] * u
               + self.dynamic_a[0, 1] * (n_left + n_right)
               + self.dynamic_a[0, 2] * v * r
               + self.dynamic_a[0, 3] * r * r)

        v_a = (self.dynamic_a[1, 0] * u * r
               + self.dynamic_a[1, 1] * v)

        beta = (self.dynamic_a[2, 0] * v * u
                + self.dynamic_a[2, 1] * r
                + self.dynamic_a[2, 2] * (n_left - n_right))

        # 添加残差项（若提供）
        if residual is None:
            residual = np.zeros(3, dtype=np.float32)
        u_a += float(residual[0])
        v_a += float(residual[1])
        beta += float(residual[2])

        # 限幅
        u_a = float(np.clip(u_a, -self.p.u_a_max, self.p.u_a_max))
        v_a = float(np.clip(v_a, -self.p.v_a_max, self.p.v_a_max))
        beta = float(np.clip(beta, -self.p.beta_max, self.p.beta_max))

        return u_a, v_a, beta

    def step(self, state: np.ndarray, action: np.ndarray, residual: np.ndarray = None):
        """执行单步动力学更新：根据当前 state 和 action 计算 next_state。

        返回：next_state, (u_a, v_a, beta)
        """
        u_a, v_a, beta = self.compute_accelerations(state, action, residual)
        x, y, psi, u, v, r = state

        # 四元组的积分（与 CustomEnv.py: next_state = state + dt * [ ... ] + noise）
        dx = u * math.cos(psi) - v * math.sin(psi)
        dy = u * math.sin(psi) + v * math.cos(psi)
        dpsi = r

        next_state = np.array(state, dtype=np.float32) + self.p.dt * np.array(
            [dx, dy, dpsi, u_a, v_a, beta], dtype=np.float32)

        # 角度归一化至 [-pi, pi]
        if next_state[2] > math.pi:
            next_state[2] -= 2 * math.pi
        elif next_state[2] < -math.pi:
            next_state[2] += 2 * math.pi

        # 速度限幅
        next_state[3] = float(np.clip(next_state[3], -self.p.u_max, self.p.u_max))
        next_state[4] = float(np.clip(next_state[4], -self.p.v_max, self.p.v_max))
        next_state[5] = float(np.clip(next_state[5], -self.p.r_max, self.p.r_max))

        return next_state, (u_a, v_a, beta)
