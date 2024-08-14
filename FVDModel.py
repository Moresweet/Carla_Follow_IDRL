import numpy as np

class FVDModel:
    def __init__(self, a_max, v0, s0, T, b, delta, delta1, lambda_):
        self.a_max = a_max  # 最大加速度
        self.v0 = v0        # 期望速度
        self.s0 = s0        # 最小车间距离
        self.T = T          # 安全时距
        self.b = b          # 舒适减速度
        self.delta = delta  # 速度项指数
        self.delta1 = delta1  # 距离项指数
        self.lambda_ = lambda_  # 相对速度修正系数

    def compute_acceleration(self, v, s, v_lead):
        # delta_v = v - v_lead  # 相对速度
        delta_v = v_lead - v
        s_star = self.s0 + v * self.T + (v * delta_v) / (2 * np.sqrt(self.a_max * self.b))
        acceleration = self.a_max * (1 - (v / self.v0) ** self.delta - (s_star / s) ** self.delta1) + self.lambda_ * delta_v
        return acceleration