class RSSModel:
    def __init__(self, max_acceleration=3.0, min_deceleration=3.0, max_deceleration=3.0, response_time=4):
        """
        初始化RSS模型的参数
        :param max_acceleration: 最大加速度 (m/s^2)
        :param min_deceleration: 最小减速度 (m/s^2)
        :param max_deceleration: 最大减速度 (m/s^2)
        :param response_time: 反应时间 (s)
        """
        self.max_acceleration = max_acceleration
        self.min_deceleration = min_deceleration
        self.max_deceleration = max_deceleration
        self.response_time = response_time

    def safe_distance(self, speed_ego, speed_other):
        """
        计算两车之间的安全距离
        :param speed_ego: 自车速度 (m/s)
        :param speed_other: 前车速度 (m/s)
        :return: 安全距离 (m)
        """
        reaction_distance = speed_ego * self.response_time
        acceleration_distance = 0.5 * self.max_acceleration * self.response_time ** 2
        ego_braking_distance = (speed_ego + self.max_acceleration * self.response_time) ** 2 / (
                    2 * self.min_deceleration)
        other_braking_distance = speed_other ** 2 / (2 * self.max_deceleration)

        distance = reaction_distance + acceleration_distance + ego_braking_distance - other_braking_distance
        return max(distance, 0)

    def is_safe(self, distance, speed_ego, speed_other):
        """
        判断当前距离是否安全
        :param distance: 当前两车之间的距离 (m)
        :param speed_ego: 自车速度 (m/s)
        :param speed_other: 前车速度 (m/s)
        :return: 是否安全 (bool)
        """
        required_distance = self.safe_distance(speed_ego, speed_other)
        return distance >= required_distance


# 示例使用
if __name__ == "__main__":
    rss = RSSModelComplex()

    # 自车速度和前车速度 (m/s)
    speed_ego = 25.0
    speed_other = 20.0

    # 当前距离 (m)
    current_distance = 50.0

    # 判断当前距离是否安全
    if rss.is_safe(current_distance, speed_ego, speed_other):
        print("当前距离安全")
    else:
        print("当前距离不安全")
