class RSSNewModel:
    def __init__(self, reaction_time, max_deceleration):
        """
        初始化RSS类的实例。
        :param reaction_time: 系统的反应时间 (s)
        :param max_deceleration: 最大减速度 (m/s^2)
        """
        self.reaction_time = reaction_time
        self.max_deceleration = max_deceleration

    def safe_distance(self, ego_velocity):
        """
        计算安全距离。
        :param ego_velocity: 自车的速度 (m/s)
        :return: 安全距离 (m)
        """
        return ego_velocity * self.reaction_time + 0.5 * ego_velocity ** 2 / self.max_deceleration

    def calculate_acceleration(self, ego_velocity, front_velocity, distance_to_front):
        """
        根据RSS计算当前情况下的加速度。
        :param ego_velocity: 自车的速度 (m/s)
        :param front_velocity: 前车的速度 (m/s)
        :param distance_to_front: 自车与前车的距离 (m)
        :return: 自车的加速度 (m/s^2)
        """
        safe_distance = self.safe_distance(ego_velocity)

        if distance_to_front > safe_distance:
            # 安全距离内，不需要减速
            return 0.0
        else:
            # 距离不足，计算所需减速度
            required_deceleration = (front_velocity - ego_velocity) / self.reaction_time
            # 确保不超过最大减速度
            return max(-self.max_deceleration, required_deceleration)


# 示例使用

# 初始化RSS类的实例
rss = RSSNewModel(reaction_time=1.5, max_deceleration=5.0)

# 输入的模拟参数
ego_velocity = 20.0  # 自车速度 (m/s)
front_velocity = 15.0  # 前车速度 (m/s)
distance_to_front = 30.0  # 距离前车的距离 (m)

# 计算加速度
acceleration = rss.calculate_acceleration(ego_velocity, front_velocity, distance_to_front)

# 输出加速度
print(f"Recommended acceleration (m/s^2): {acceleration}")
