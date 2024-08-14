import numpy as np

class IDMModel:
    def __init__(self, a_max, v0, s0, T, b):
        self.a_max = a_max  # 最大加速度
        self.v0 = v0        # 期望速度
        self.s0 = s0        # 最小车间距离
        self.T = T          # 安全时距
        self.b = b          # 舒适减速度

    def compute_acceleration(self, v, s, v_lead):
        """
        计算加速度
        :param v: 当前速度
        :param s: 当前车间距离
        :param v_lead: 前车速度
        :return: 加速度
        """
        delta_v = v - v_lead  # 相对速度
        s_star = self.s0 + v * self.T + (v * delta_v) / (2 * np.sqrt(self.a_max * self.b))
        acceleration = self.a_max * (1 - (v / self.v0) ** 4 - (s_star / s) ** 2)
        return acceleration

# 参数设置
a_max = 1.0  # 最大加速度
v0 = 30.0    # 期望速度 (m/s)
s0 = 2.0     # 最小车间距离 (m)
T = 1.5      # 安全时距 (s)
b = 1.5      # 舒适减速度 (m/s^2)

idm = IDMModel(a_max, v0, s0, T, b)

# 示例：计算加速度
current_speed = 25.0  # 当前速度 (m/s)
distance_to_lead = 20.0  # 车间距离 (m)
lead_speed = 22.0  # 前车速度 (m/s)

acceleration = idm.compute_acceleration(current_speed, distance_to_lead, lead_speed)
print(f"Computed acceleration: {acceleration:.2f} m/s^2")
