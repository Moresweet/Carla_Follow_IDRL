import numpy as np


class FrenetCoordinateSystem:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.s = self._compute_s(waypoints)

    def clear_waypoints(self):
        self.waypoints = None

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints

    def _compute_s(self, waypoints):
        s = [0]
        for i in range(1, len(waypoints)):
            loc1 = waypoints[i - 1].transform.location
            loc2 = waypoints[i].transform.location
            dist = np.linalg.norm([loc2.x - loc1.x, loc2.y - loc1.y, loc2.z - loc1.z])
            s.append(s[-1] + dist)
        return s

    def get_frenet(self, location):
        min_dist = float('inf')
        frenet_s = 0
        frenet_d = 0
        for i in range(len(self.waypoints) - 1):
            # 诡异事件
            if i == len(self.waypoints) - 1:
                break
            p1 = np.array([self.waypoints[i].transform.location.x, self.waypoints[i].transform.location.y])
            p2 = np.array([self.waypoints[i + 1].transform.location.x, self.waypoints[i + 1].transform.location.y])
            loc = np.array([location.x, location.y])
            vec = p2 - p1
            proj = np.dot(loc - p1, vec) / np.dot(vec, vec)
            proj = np.clip(proj, 0, 1)
            proj_point = p1 + proj * vec
            dist = np.linalg.norm(loc - proj_point)
            if dist < min_dist:
                min_dist = dist
                frenet_s = self.s[i] + proj * np.linalg.norm(vec)
                frenet_d = np.linalg.norm(np.cross(loc - p1, vec) / np.linalg.norm(vec))
        return frenet_s, frenet_d

    def get_distance(self, vehicle_1, vehicle_2):
        s1, _ = self.get_frenet(vehicle_1.get_location())
        s2, _ = self.get_frenet(vehicle_2.get_location())
        return abs(s1 - s2)
