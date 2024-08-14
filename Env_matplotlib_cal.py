# 导包
import copy

import carla
import time
import math
import sys
import torch
from RSSModel import RSSModel
from FrenetCoordinateSystem import FrenetCoordinateSystem
import pygame
import numpy as np
import weakref
from collections import deque
import threading
from scipy.interpolate import splprep, splev
from IDMModel import IDMModel
from FVDModel import FVDModel
from RSS_new import RSSNewModel
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('/home/moresweet/carla/PythonAPI/carla')
sys.path.append('/home/moresweet/gitCloneZone/DIRL')
from agents.navigation.basic_agent import BasicAgent
from CF_MPC_RSS import Control_MPC

# 初始化RSS类的实例
rss = RSSNewModel(reaction_time=0.5, max_deceleration=3.0)

# PID控制器参数
Kp = 1.0
Ki = 0.0
Kd = 0.1
integral = 0.0
previous_error = 0.0

rss_model = RSSModel()

# IDM参数设置
a_max = 1.5  # 最大加速度
v0 = 30.0  # 期望速度 (m/s)
s0 = 2.0  # 最小车间距离 (m)
T = 1.5  # 安全时距 (s)
b = 1.5  # 舒适减速度 (m/s^2)

idm = IDMModel(a_max, v0, s0, T, b)

# FVD参数设置
a_max = 1.0  # 最大加速度
v0 = 30.0  # 期望速度 (m/s)
s0 = 2.0  # 最小车间距离 (m)
T = 1.5  # 安全时距 (s)
b = 1.5  # 舒适减速度 (m/s^2)
delta = 15  # 速度项指数
delta1 = 2  # 距离项指数
lambda_ = 0.1  # 相对速度修正系数

fvd = FVDModel(a_max, v0, s0, T, b, delta, delta1, lambda_)


# pred = []


def predict_vehicle_state(location, dis, f_vel, vel, acceleration, model='linear', delta_time=0.5, pred_step=5):
    # 获取当前速度和位置
    current_velocity = vel
    current_location = location

    # 将速度向量转换为标量速度
    current_speed = current_velocity
    r_dis = dis
    new_speed = 0
    pred = None
    if model == 'linear':
        # 默认的不需要预测
        pred = None
    elif model == 'idm':
        pred = env.idm_pred
    elif model == 'fvd':
        pred = env.fvd_pred
    elif model == 'rss':
        pred = env.rss_pred
    elif model == 'idrl':
        pred = env.idrl_pred

    if len(pred) == 0:
        for step in range(pred_step):
            f_curr_vel = env.front_vehicle_bp.get_velocity()
            f_curr_location = env.front_vehicle_bp.get_location()
            t = (step + 1) * delta_time
            if step != 0:
                if model == 'linear':
                    acceleration = acceleration
                elif model == 'idm':
                    acceleration = idm.compute_acceleration(current_speed, r_dis,
                                                            math.sqrt(f_curr_vel.x ** 2 + f_curr_vel.y ** 2))
                    # print('idm pre acc:{}'.format(acceleration))
                elif model == 'fvd':
                    acceleration = fvd.compute_acceleration(current_speed, r_dis,
                                                            math.sqrt(f_curr_vel.x ** 2 + f_curr_vel.y ** 2))
                elif model == 'rss':
                    # acceleration = fvd.compute_acceleration(current_speed, r_dis,
                    #                                         math.sqrt(f_curr_vel.x ** 2 + f_curr_vel.y ** 2))
                    # 计算当前距离
                    distance = math.sqrt((f_curr_location.x - current_location.x) ** 2 +
                                         (f_curr_location.y - current_location.y) ** 2 +
                                         (f_curr_location.z - current_location.z) ** 2)
                    # 计算安全距离
                    safe_dist = safety_distance(current_speed, math.sqrt(f_curr_vel.x ** 2 + f_curr_vel.y ** 2),
                                                env.t_response, env.a_max_brake)
                    # safe_dist = rss_model.safe_distance(rear_velocity, front_velocity)
                    # 调整后车速度
                    if distance < safe_dist:
                        target_speed = 0
                    else:
                        target_speed = math.sqrt(f_curr_vel.x ** 2 + f_curr_vel.y ** 2) + 5
                    acceleration = pid_control(target_speed, current_velocity, env.tau)
                    pass
                    # print('rss pre acc:{}'.format(acceleration))
                elif model == 'idrl':
                    acceleration = fvd.compute_acceleration(current_speed, r_dis,
                                                            math.sqrt(f_curr_vel.x ** 2 + f_curr_vel.y ** 2))

            # 计算0.5秒后的速度
            new_speed = current_speed + acceleration * delta_time

            # 计算位移
            displacement = current_speed * delta_time + 0.5 * acceleration * delta_time ** 2

            yaw = env.ego_vehicle.get_transform().rotation.yaw
            # steer_adjusted_yaw = yaw + steer * math.radians(45)
            # 计算新位置
            # print("delta x :{}, delta y :{}".format(math.cos(yaw), math.sin(yaw)))
            new_location = carla.Location(
                x=current_location.x + displacement * math.cos(yaw),
                y=current_location.y + displacement * math.sin(yaw),
                z=current_location.z
            )
            # # 计算车距
            r_dis = dis + (f_vel * delta_time) - displacement
            pred.append((new_location, new_speed, r_dis))
            current_speed = new_speed
            current_location = new_location
        return pred.pop(0)
    else:
        return pred.pop(0)


"""
简化RSS模型
:param v_ego: 后车速度 (m/s)
:param v_other: 前车速度 (m/s^2)
:param t_response: 反应时间 (s)
:param a_max_brake: 最大减速度 (m/s^2)
"""
safety_distance = lambda v_ego, v_other, t_response, a_max_brake: (
        v_ego * t_response +
        0.5 * a_max_brake * t_response ** 2 +
        (v_ego - v_other) ** 2 / (2 * a_max_brake)
)
"""
返回PID油门或者刹车控制量
:param target_speed: 目标速度
:param current_speed: 当前速度
:param dt: 时间间隔步长
"""


# PID控制函数
def pid_control(target_speed, current_speed, dt):
    global integral, previous_error
    error = target_speed - current_speed
    integral += error * dt
    derivative = (error - previous_error) / dt
    control = Kp * error + Ki * integral + Kd * derivative
    previous_error = error
    return control


"""
返回给定距离最近的正角
:param num: 输入角度
"""


def find_closest_angle(num):
    if num > 180:
        num = num - 360
    angles = [90, -90, 0, 180, -180]
    closest_angle = min(angles, key=lambda x: abs(x - num))
    return closest_angle


class Env:
    def __init__(self, client, world, tau=0.1):
        self.client = client
        self.world = world
        self.screen = None
        self.camera_bp = None
        # rss idm idrl fvd
        self.mode = 'fvd'
        # none spacial
        self.scenario = 'spacial'
        self.debug = True
        self.green_light = True
        self.front_image = np.zeros((600, 800, 3), dtype=np.uint8)
        self.back_image = np.zeros((600, 800, 3), dtype=np.uint8)
        self.camera_front = None
        self.camera_back = None
        self.warning_img = None
        self.ego_vehicle = None  # 后车
        self.front_vehicle_bp = None  # 前车
        self.front_switch_road_waypoints = deque()
        self.relative_distance = None
        self.relative_speed = None
        self.ego_speed = None
        self.ego_acceleration = None
        self.tau = 0.1
        self.destination = None
        self.control_flag = True
        self.map = self.world.get_map()
        self.spawn_vehicles()
        self.frenet = FrenetCoordinateSystem(self.map.generate_waypoints(2.0))
        # 初始化RSS参数
        self.t_response = 1.5  # 反应时间 (s)
        self.a_max_brake = 3  # 最大制动减速度 (m/s^2)
        if self.debug is True:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            world.apply_settings(settings)
        self.agent = BasicAgent(self.ego_vehicle)
        self.front_agent = BasicAgent(self.front_vehicle_bp)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ddpg = torch.load('/home/moresweet/gitCloneZone/DIRL/checkpoints/RSS/weights/ddpg_105.pt')
        self.ddpg_acc = 0
        self.ddpg_v = 0
        self.ddpg_d = 0
        self.ddpg_rv = 0
        self.ddpg_x = 0
        self.ddpg_y = 0
        self.idm_acc = 0
        self.idm_v = 0
        self.idm_d = 0
        self.idm_rv = 0
        self.idm_x = 0
        self.idm_y = 0
        self.rss_acc = 0
        self.rss_v = 0
        self.rss_d = 0
        self.rss_rv = 0
        self.rss_x = 0
        self.rss_y = 0
        self.cur_acc = 0
        self.curr_ego_v = 0
        self.curr_d = 0
        self.curr_rv = 0
        self.curr_x = 0
        self.curr_y = 0
        self.e2f_waypoints_queue = deque()
        self.last_location1 = self.front_vehicle_bp.get_location()
        self.last_location2 = self.ego_vehicle.get_location()
        self.data = {
            "timestamp": [],
            "acc": [],
            "v": [],
            "d": [],
            "rv": [],
            "x": [],
            "y": [],
            "idrl_acc": [],
            "idrl_v": [],
            "idrl_d": [],
            "idrl_rv": [],
            "idrl_x": [],
            "idrl_y": [],
            "idm_acc": [],
            "idm_v": [],
            "idm_d": [],
            "idm_rv": [],
            "idm_x": [],
            "idm_y": [],
            "rss_acc": [],
            "rss_v": [],
            "rss_d": [],
            "rss_rv": [],
            "rss_x": [],
            "rss_y": []
        }
        self.initial_timestamp = None
        self.frame_count = 0
        self.Vf_Vet0 = np.zeros((50, 1))
        self.idm_pred = []
        self.fvd_pred = []
        self.idrl_pred = []
        self.rss_pred = []

    """
    生成carla场景
    """

    def spawn_vehicles(self):
        # 加载场景
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        vehicle_bp_rear = blueprint_library.filter('vehicle.*')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        if self.scenario == 'spacial':
            # 工况1实现  9  12很好
            front_spawn_point = spawn_points[29]
            # 获取前车的车道信息
            front_spawn_point.location.x = 50
            front_waypoint = self.map.get_waypoint(front_spawn_point.location)

            # 计算后车生成点（在前车后20米处）
            distance = 10  # 距离前车20米
            rear_location = front_waypoint.transform.location - front_waypoint.transform.get_forward_vector() * distance
            rear_spawn_point = carla.Transform(rear_location, front_waypoint.transform.rotation)
            # 创建前车和后车
            self.front_vehicle_bp = self.world.spawn_actor(vehicle_bp, front_spawn_point)
            self.ego_vehicle = self.world.spawn_actor(vehicle_bp_rear, rear_spawn_point)
            pygame.init()
            # 设置前车匀速行驶
            # self.front_vehicle_bp.set_autopilot(True)
            if self.mode == 'auto':
                self.ego_vehicle.set_autopilot(True)
        else:
            front_spawn_point = spawn_points[0]
            # 获取前车的车道信息
            front_waypoint = self.map.get_waypoint(front_spawn_point.location)

            # 计算后车生成点（在前车后20米处）
            distance = 10  # 距离前车20米
            rear_location = front_waypoint.transform.location - front_waypoint.transform.get_forward_vector() * distance
            rear_spawn_point = carla.Transform(rear_location, front_waypoint.transform.rotation)
            # 创建前车和后车
            self.front_vehicle_bp = self.world.spawn_actor(vehicle_bp, front_spawn_point)
            self.ego_vehicle = self.world.spawn_actor(vehicle_bp_rear, rear_spawn_point)
            pygame.init()
            # 设置前车匀速行驶
            self.front_vehicle_bp.set_autopilot(True)
            if self.mode == 'auto':
                self.ego_vehicle.set_autopilot(True)
        pygame.init()
        # 创建pygame窗口
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("CARLA Manual Control")
        # 设置前后相机视角
        self.camera_bp = blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '800')
        self.camera_bp.set_attribute('image_size_y', '600')
        self.camera_bp.set_attribute('fov', '110')
        camera_transform_front = carla.Transform(carla.Location(x=1.5, y=0, z=2.4), carla.Rotation(yaw=30))
        self.camera_front = world.spawn_actor(self.camera_bp, camera_transform_front, attach_to=self.ego_vehicle)
        camera_transform_back = carla.Transform(carla.Location(x=-0.2, y=-0.2, z=1.3), carla.Rotation(yaw=28))
        self.camera_back = world.spawn_actor(self.camera_bp, camera_transform_back, attach_to=self.ego_vehicle)
        self.camera_front.listen(lambda image: self.front_camera_callback(image))
        self.camera_back.listen(lambda image: self.back_camera_callback(image))
        self.warning_img = pygame.image.load('./EARLY_WARNING.png')
        self.warning_img = pygame.transform.scale(self.warning_img, (50, 50))

    """
    相机图像处理
    :param image: 相机图像
    """

    def process_img(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    """
    摄像头回调函数
    :param image: 相机图像
    """

    def front_camera_callback(self, image):
        self.front_image = self.process_img(image)

    def back_camera_callback(self, image):
        self.back_image = self.process_img(image)

    def set_traffic_lights_to_green(self):
        # 获取所有交通灯
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')

        for traffic_light in traffic_lights:
            # 将交通灯状态设置为绿灯
            traffic_light.set_state(carla.TrafficLightState.Green)

    def record_time_data(self):
        snapshot = world.get_snapshot()
        current_timestamp = snapshot.timestamp.elapsed_seconds
        if self.initial_timestamp is None:
            self.initial_timestamp = current_timestamp
        # 将时间戳归零
        adjusted_timestamp = current_timestamp - self.initial_timestamp
        self.data["timestamp"].append(adjusted_timestamp)
        self.data["acc"].append(self.cur_acc)
        self.data["v"].append(self.curr_ego_v)
        self.data["d"].append(self.curr_d)
        self.data["rv"].append(self.curr_rv)
        self.data["x"].append(self.curr_x)
        self.data["y"].append(self.curr_y)
        self.data["idrl_acc"].append(self.ddpg_acc)
        self.data["idrl_v"].append(self.ddpg_v)
        self.data["idrl_d"].append(self.ddpg_d)
        self.data["idrl_rv"].append(self.ddpg_rv)
        self.data["idrl_x"].append(self.ddpg_x)
        self.data["idrl_y"].append(self.ddpg_y)
        self.data["idm_acc"].append(self.idm_acc)
        self.data["idm_v"].append(self.idm_v)
        self.data["idm_d"].append(self.idm_d)
        self.data["idm_rv"].append(self.idm_rv)
        self.data["idm_x"].append(self.idm_x)
        self.data["idm_y"].append(self.idm_y)
        self.data["rss_acc"].append(self.rss_acc)
        self.data["rss_v"].append(self.rss_v)
        self.data["rss_d"].append(self.rss_d)
        self.data["rss_rv"].append(self.rss_rv)
        self.data["rss_x"].append(self.rss_x)
        self.data["rss_y"].append(self.rss_y)

    """
    步进更新对象状态
    """

    def step(self):
        autopilot_control = env.front_vehicle_bp.get_control()
        control = carla.VehicleControl()
        control.steer = autopilot_control.steer  # 使用自动驾驶的转向控制
        if self.debug is True:
            leader_vel_last = self.front_vehicle_bp.get_velocity()
            leader_vel_last = math.sqrt(leader_vel_last.x ** 2 + leader_vel_last.y ** 2)
            leader_pos_last = self.front_vehicle_bp.get_transform().location
            follow_vel_last = self.ego_vehicle.get_velocity()
            follow_vel_last = math.sqrt(follow_vel_last.x ** 2 + follow_vel_last.y ** 2)
            follow_pos_last = self.ego_vehicle.get_transform().location
            # dis_last = self.frenet.get_distance_by_location(leader_pos_last, follow_pos_last)
            dis_last = math.sqrt(
                (leader_pos_last.x - follow_pos_last.x) ** 2 + (leader_pos_last.y - follow_pos_last.y) ** 2)
            relative_vel = leader_vel_last - follow_vel_last
            idm_acceleration = idm.compute_acceleration(follow_vel_last, dis_last,
                                                        leader_vel_last)
            fvd_acceleration = fvd.compute_acceleration(follow_vel_last, dis_last,
                                                        leader_vel_last)
            acc = self.ego_vehicle.get_acceleration()
            # 根据刹车踏板判断加速度的正负
            control = self.ego_vehicle.get_control()
            acc = math.sqrt(acc.x ** 2 + acc.y ** 2)
            if control.brake > 0:
                acc = -acc
            nor_state = [dis_last / 120, relative_vel / 40, follow_vel_last / 40]
            nor_state = torch.FloatTensor(nor_state).to(env.device)
            action = env.ddpg.choose_action(nor_state)
            action = action
            # Vf_Vet0 = self.Vf_Vet0 + leader_vel_last
            # 计算安全距离
            safe_dist = safety_distance(follow_vel_last, leader_vel_last, self.t_response, self.a_max_brake)
            # 限制RSS的安全距离，确保过弯
            safe_dist = max(safe_dist, 10)
            # print('safe dis:{}'.format(safe_dist))
            # safe_dist = rss_model.safe_distance(rear_velocity, front_velocity)
            # 调整后车速度
            if dis_last < safe_dist:
                target_speed = 0
            else:
                target_speed = leader_vel_last + 3
            rss_mpc_acceleration = pid_control(target_speed, follow_vel_last, self.tau)
            # 推动仿真前进
            self.world.tick(self.tau)
            time.sleep(self.tau)
            self.frame_count += 1
            if self.frame_count > 70:
                self.record_time_data()
                self.shade_state(leader_vel_last, follow_vel_last, relative_vel, leader_pos_last, follow_pos_last,
                                 dis_last, acc, idm_acceleration, fvd_acceleration, action, rss_mpc_acceleration)
        # 渲染前后视角
        front_surface = pygame.surfarray.make_surface(self.front_image.swapaxes(0, 1))
        back_surface = pygame.surfarray.make_surface(self.back_image.swapaxes(0, 1))

        self.screen.blit(front_surface, (0, 0))
        self.screen.blit(back_surface, (400, 0))

        font = pygame.font.Font(None, 36)
        text = font.render(
            f"Throttle: {control.throttle:.2f} Brake: {control.brake:.2f} Steer: {control.steer:.2f}, DDPG: {env.ddpg_acc:.2f}, Acc: {env.cur_acc:.2f}",
            True, (255, 255, 255))
        self.screen.blit(text, (10, 550))
        # 设置报警阈值
        if math.fabs(env.cur_acc - env.ddpg_acc) > 5:
            warning_active = True
        else:
            warning_active = False

        if warning_active:
            self.screen.blit(self.warning_img, (0, 20))
            warning_text = font.render("warning: Acceleration poses a safety risk!", True, (255, 0, 0))
            self.screen.blit(warning_text, (50, 40))
        pygame.display.flip()
        # 更新后车的速度
        velocity = self.ego_vehicle.get_velocity()
        self.ego_speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        # 更新前车和后车的相对速度
        other_velocity = self.front_vehicle_bp.get_velocity()
        relative_velocity = carla.Vector3D(
            other_velocity.x - velocity.x,
            other_velocity.y - velocity.y,
            other_velocity.z - velocity.z
        )
        self.relative_speed = math.sqrt(relative_velocity.x ** 2 + relative_velocity.y ** 2 + relative_velocity.z ** 2)

        # 更新前车和后车的相对距离
        location = self.ego_vehicle.get_location()
        other_location = self.front_vehicle_bp.get_location()
        distance = location.distance(other_location)
        # self.relative_distance = distance
        distance_curve = sum(
            self.e2f_waypoints_queue[i].transform.location.distance(self.e2f_waypoints_queue[i + 1].transform.location)
            for i in
            range(len(self.e2f_waypoints_queue) - 1))
        self.relative_distance = distance_curve
        # print('fre dis:{}, euler dis:{}, new dis:{}'.format(self.frenet.get_distance(self.front_vehicle_bp, self.ego_vehicle),
        #                                         distance, distance_curve))

        # 更新后车的加速度（这里简化为速度的变化率，实际可能需要更复杂的物理模型）
        if hasattr(self, 'prev_ego_speed'):
            self.ego_acceleration = (self.ego_speed - self.prev_ego_speed) / self.tau
        self.prev_ego_speed = self.ego_speed

    def get_values(self):
        return {
            'ego_speed': self.ego_speed,
            'relative_speed': self.relative_speed,
            'relative_distance': self.relative_distance,
            'ego_acceleration': self.ego_acceleration
        }

    def get_waypoint_info(self, vehicle):
        waypoint = self.map.get_waypoint(vehicle.get_location())
        return waypoint

    """
    摄像头回调函数
    :param image: 相机图像
    """

    def nav_to_waypoint(self, destination):
        self.agent.set_target_speed(20)
        global_route_trace = self.agent.set_destination(destination.transform.location, start_location=True,
                                                        blender=True)
        num = 0
        # 运行导航循环
        while True:
            self.step()
            num = num + 1
            if self.agent.done():
                if self.mode != 'auto':
                    # 调整车头
                    tmp_transform = self.ego_vehicle.get_transform()
                    tmp_transform.rotation.yaw = find_closest_angle(
                        self.front_switch_road_waypoints.popleft().transform.rotation.yaw)
                    print("consum size:{}".format(len(self.front_switch_road_waypoints)))
                    self.ego_vehicle.set_transform(tmp_transform)
                # 恢复自动驾驶
                if self.mode == 'auto':
                    # 此处是因为经过不变道的路口时，auto模式需要导航点来保证不转向，因此此时不一定pop()的出来
                    if len(self.front_switch_road_waypoints) > 0:
                        self.front_switch_road_waypoints.popleft()
                    self.ego_vehicle.set_autopilot(True)
                break
            control = self.agent.run_step()
            self.ego_vehicle.apply_control(control)

    def auto_curve_follow(self):
        front_waypoint = self.map.get_waypoint(self.front_vehicle_bp.get_location())
        # 有问题，前车跑远了，导航并不一定按照前车走过的路走，这不同于pid的直线行驶
        # 前车到达路口后车就导航到前车的位置
        while front_waypoint.is_junction or is_curve(front_waypoint, front_waypoint.next(5)[0]):
            self.nav_to_waypoint(front_waypoint)

    def yaw_diff(self):
        return True if (
                           math.fabs((self.ego_vehicle.get_transform().rotation.yaw -
                                      self.front_vehicle_bp.get_transform().rotation.yaw +
                                      180) % 360 - 180)
                       ) > 80 else False

    def yaw_similar(self):
        return True if (
                           math.fabs((self.ego_vehicle.get_transform().rotation.yaw -
                                      self.front_vehicle_bp.get_transform().rotation.yaw +
                                      180) % 360 - 180)
                       ) < 20 else False

    def scenario_spacial(self):
        def apply_throttle(vehicle, throttle, duration):
            control = carla.VehicleControl()
            control.throttle = throttle
            vehicle.apply_control(control)
            time.sleep(duration)

        def apply_brake(vehicle, brake, duration):
            control = carla.VehicleControl()
            control.brake = brake
            vehicle.apply_control(control)
            time.sleep(duration)

        # 让两辆车均以60 km/h (16.67 m/s) 的速度匀速行驶3秒
        speed_60kmh = 16.67 / 3.6  # 16.67 m/s (60 km/h)
        self.ego_vehicle.set_target_velocity(carla.Vector3D(speed_60kmh, 0, 0))
        self.front_vehicle_bp.set_target_velocity(carla.Vector3D(speed_60kmh, 0, 0))
        # time.sleep(3)
        for _ in range(int(3 / self.tau)):
            self.follow_vehicle()
            self.step()  # 运行仿真步长
            time.sleep(self.tau)
        print("匀速结束")

        # 前车开始加速，在5秒内以2.22 m/s^2的平均加速度将速度提升至100 km/h (27.78 m/s)
        acceleration = 2.22
        target_speed_100kmh = 100 / 3.6  # 27.78 m/s (100 km/h)
        current_speed = speed_60kmh
        time_step = 0.05  # 50ms 每步
        for _ in range(int(5 / time_step)):
            current_speed += acceleration * time_step
            self.front_vehicle_bp.set_target_velocity(carla.Vector3D(min(current_speed, target_speed_100kmh), 0, 0))
            # time.sleep(time_step)
            self.follow_vehicle()
            self.step()
            time.sleep(self.tau)

        print("提速结束")
        # 前车以100 km/h匀速行驶2秒
        self.front_vehicle_bp.set_target_velocity(carla.Vector3D(target_speed_100kmh, 0, 0))
        # time.sleep(2)
        for _ in range(int(2 / time_step)):
            self.follow_vehicle()
            self.step()
            time.sleep(self.tau)
        print("匀速结束Finished!")

        # 前车减速，在5秒内将速度降至60 km/h
        deceleration = (target_speed_100kmh - speed_60kmh) / 5  # 平均减速度
        for _ in range(int(5 / time_step)):
            current_speed -= deceleration * time_step
            self.front_vehicle_bp.set_target_velocity(carla.Vector3D(max(current_speed, speed_60kmh), 0, 0))
            # time.sleep(time_step)
            self.follow_vehicle()
            self.step()
            time.sleep(self.tau)

        print("减速结束")
        draw_pic()

    # 参数为更新前的旧状态以及影子算法计算的加速度
    def shade_state(self, f_vel, e_vel, r_vel, f_pos, e_pos, dis, acc, idm_acceleration, fvd_acceleration, action,
                    rss_mpc_acceleration):
        # 此处为更新后的新状态（实际AV）
        f_vel_curr = self.front_vehicle_bp.get_velocity()
        e_vel_curr = self.ego_vehicle.get_velocity()
        r_vel_curr = math.sqrt(f_vel_curr.x ** 2 + f_vel_curr.y ** 2) - math.sqrt(e_vel_curr.x ** 2 + e_vel_curr.y ** 2)
        f_pos_curr = self.front_vehicle_bp.get_transform().location
        e_pos_curr = self.ego_vehicle.get_transform().location
        # dis_curr = self.frenet.get_distance_by_location(f_pos_curr, e_pos_curr)
        dis_curr = math.sqrt((f_pos_curr.x - e_pos_curr.x) ** 2 + (f_pos_curr.y - e_pos_curr.y) ** 2)
        acc_curr = self.ego_vehicle.get_acceleration()
        self.cur_acc = math.sqrt(acc_curr.x ** 2 + acc_curr.y ** 2)
        control = self.ego_vehicle.get_control()
        if control.brake > 0:
            self.cur_acc = -self.cur_acc
        self.curr_ego_v = math.sqrt(e_vel_curr.x ** 2 + e_vel_curr.y ** 2)
        self.curr_d = dis_curr
        self.curr_rv = r_vel_curr
        self.curr_x = e_pos_curr.x
        self.curr_y = e_pos_curr.y
        # 此处为更新后的影子算法状态（不实际执行，根据加速度值推演）
        idm_new_loc, idm_new_speed, idm_dis_delta = predict_vehicle_state(e_pos, dis, f_vel, e_vel, idm_acceleration,
                                                                          'idm')
        ddpg_new_loc, ddpg_new_speed, ddpg_dis_delta = predict_vehicle_state(e_pos, dis, f_vel, e_vel, action, 'idrl')
        fvd_new_loc, fvd_new_speed, fvd_dis_delta = predict_vehicle_state(e_pos, dis, f_vel, e_vel, fvd_acceleration,
                                                                          'fvd')
        rss_new_loc, rss_new_speed, rss_dis_delta = predict_vehicle_state(e_pos, dis, f_vel, e_vel,
                                                                          rss_mpc_acceleration,
                                                                          'rss')
        self.idm_acc = idm_acceleration
        # print("idm arxiv acc:{}".format(idm_acceleration))
        self.idm_v = idm_new_speed
        self.idm_rv = math.sqrt(f_vel_curr.x ** 2 + f_vel_curr.y ** 2) - idm_new_speed
        # env.idm_d = env.frenet.get_distance_by_location(idm_new_loc, f_pos_curr)
        self.idm_d = idm_dis_delta
        self.idm_x = idm_new_loc.x
        self.idm_y = idm_new_loc.y
        # self.ddpg_acc = action *10
        # self.ddpg_v = ddpg_new_speed
        # self.ddpg_rv = math.sqrt(f_vel_curr.x ** 2 + f_vel_curr.y ** 2) - ddpg_new_speed
        # # env.ddpg_d = env.frenet.get_distance_by_location(ddpg_new_loc, f_pos_curr)
        # self.ddpg_d = ddpg_dis_delta
        self.ddpg_acc = fvd_acceleration
        self.ddpg_v = fvd_new_speed
        self.ddpg_rv = math.sqrt(f_vel_curr.x ** 2 + f_vel_curr.y ** 2) - fvd_new_speed
        # env.ddpg_d = env.frenet.get_distance_by_location(ddpg_new_loc, f_pos_curr)
        self.ddpg_d = fvd_dis_delta
        self.ddpg_x = fvd_new_loc.x
        self.ddpg_y = fvd_new_loc.y
        self.rss_acc = rss_mpc_acceleration
        self.rss_v = rss_new_speed
        self.rss_rv = math.sqrt(f_vel_curr.x ** 2 + f_vel_curr.y ** 2) - rss_new_speed
        # env.rss_d = env.frenet.get_distance_by_location(rss_new_loc, f_pos_curr)
        self.rss_d = rss_dis_delta
        self.rss_x = rss_new_loc.x
        self.rss_y = rss_new_loc.y
        # print('rss arxiv acc:{}'.format(self.rss_acc))

    # 跟车逻辑
    def follow_vehicle(self):
        rear_transform = self.ego_vehicle.get_transform()
        front_transform = self.front_vehicle_bp.get_transform()

        rear_velocity = self.ego_vehicle.get_velocity()
        front_velocity = self.front_vehicle_bp.get_velocity()

        # 计算当前速度
        rear_speed = math.sqrt(rear_velocity.x ** 2 + rear_velocity.y ** 2 + rear_velocity.z ** 2)
        front_speed = math.sqrt(front_velocity.x ** 2 + front_velocity.y ** 2 + front_velocity.z ** 2)

        # 计算当前距离
        distance = math.sqrt((front_transform.location.x - rear_transform.location.x) ** 2 +
                             (front_transform.location.y - rear_transform.location.y) ** 2 +
                             (front_transform.location.z - rear_transform.location.z) ** 2)
        rear_velocity = math.sqrt(rear_velocity.x ** 2 + rear_velocity.y ** 2 + rear_velocity.z ** 2)
        front_velocity = math.sqrt(front_velocity.x ** 2 + front_velocity.y ** 2 + front_velocity.z ** 2)
        # 计算安全距离
        safe_dist = safety_distance(rear_velocity, front_velocity, env.t_response, env.a_max_brake)
        # safe_dist = rss_model.safe_distance(rear_velocity, front_velocity)
        # 调整后车速度
        if distance < safe_dist:
            target_speed = 0
        else:
            target_speed = front_speed + 5
        control = pid_control(target_speed, rear_speed, self.tau)
        throttle = max(0.0, min(1.0, control))
        brake = max(0.0, min(1.0, -control))

        if self.mode != 'idm' and self.mode != 'fvd' and self.mode != 'rss':
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0))
        elif self.mode == 'idm':
            # 使用IDM模型计算加速度
            acceleration = idm.compute_acceleration(rear_velocity, distance, front_velocity)
            # print("idm acc:{}".format(acceleration))
            # 应用加速度到后车
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=acceleration if acceleration > 0 else 0,
                                                                brake=-acceleration if acceleration < 0 else 0))
        elif self.mode == 'fvd':
            acceleration = fvd.compute_acceleration(rear_velocity, distance, front_velocity)

            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=acceleration if acceleration > 0 else 0,
                                                                brake=-acceleration if acceleration < 0 else 0))
        elif self.mode == 'rss':
            # acceleration = rss.calculate_acceleration(rear_velocity, front_velocity, distance)
            # print("rss acc: {}".format(control))
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=control if control > 0 else 0,
                                                                brake=-control if control < 0 else 0))
        # 加影子系统，我们影子系统需要的输入是后车，和时间间隔，根据现在的情况计算加速度，根据加速度，计算下一步出现的情况位置和速度，计算的时候维护一个大小为参数总间隔的历史状况，到达时清空。


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town01')

env = Env(client=client, world=world)

spectator = world.get_spectator()


# 回调函数，用于更新观众视角
def follow_vehicle(world_snapshot):
    transform = env.front_vehicle_bp.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=250), carla.Rotation(pitch=-90)))


world.on_tick(follow_vehicle)


def is_curve(current_waypoint, next_waypoint):
    # metric_change = max(math.fabs(current_waypoint.transform.rotation.yaw - next_waypoint.transform.rotation.yaw),
    #                     math.fabs(current_waypoint.transform.rotation.pitch - next_waypoint.transform.rotation.pitch),
    #                     math.fabs(current_waypoint.transform.rotation.roll - next_waypoint.transform.rotation.roll))
    metric_change = math.fabs((current_waypoint.transform.rotation.yaw -
                               next_waypoint.transform.rotation.yaw +
                               180) % 360 - 180)
    return True if metric_change > 20 else False


# 运行仿真
test_flag = True
record_flag = False


def update_front_ref():
    def task():
        # env.front_switch_road_waypoints.append(env.front_vehicle_bp.get_transform().rotation.yaw)
        env.front_switch_road_waypoints.append(env.map.get_waypoint(env.front_vehicle_bp.get_location()).next(5)[0])
        env.world.debug.draw_string(env.front_vehicle_bp.get_transform().location, str("u"), life_time=100)
        env.world.debug.draw_string(
            env.map.get_waypoint(env.front_vehicle_bp.get_location()).next(5)[0].transform.location, str("un"),
            life_time=100)
        print("deque size:{}".format(len(env.front_switch_road_waypoints)))

    threading.Thread(target=task).start()


def detect_front_junction(interval):
    def task():
        while True:
            global record_flag
            front_waypoint = env.map.get_waypoint(env.front_vehicle_bp.get_location())
            env.world.debug.draw_string(front_waypoint.transform.location, str("c"), life_time=0.1)
            env.world.debug.draw_string(front_waypoint.next(5)[0].transform.location, str("n"), life_time=0.1)
            if (front_waypoint.is_junction or is_curve(front_waypoint, front_waypoint.next(5)[0])) and \
                    record_flag is False:
                record_flag = True
                env.world.debug.draw_string(front_waypoint.transform.location, str("s"), life_time=100)
            # 出路口或者弯道开启计时记录队列
            if record_flag is True and not front_waypoint.is_junction and not is_curve(front_waypoint,
                                                                                       front_waypoint.next(5)[0]):
                # time.sleep(2)
                print("更新")
                update_front_ref()
                # time.sleep(2)
                record_flag = False
            time.sleep(interval)

    threading.Thread(target=task, daemon=True).start()


def update_frenet_distance():
    def task():
        while True:
            time.sleep(0.1)
            # 获取当前位置信息
            current_location1 = env.map.get_waypoint(env.front_vehicle_bp.get_location())
            current_location2 = env.map.get_waypoint(env.ego_vehicle.get_location())
            if len(env.e2f_waypoints_queue) > 1:
                env.e2f_waypoints_queue.popleft()
                env.e2f_waypoints_queue.pop()

            # 如果前车移动了，将新的waypoint添加到队列中
            if current_location1.transform.location.distance(env.last_location1) > 0.5:
                env.e2f_waypoints_queue.append(current_location1)
                env.last_location1 = current_location1.transform.location

            # 如果后车移动到前车走过的waypoint，将其从队列中移除
            # while env.e2f_waypoints_queue and current_location2.transform.location.distance(
            #         env.e2f_waypoints_queue[0].transform.location) < 2.0:
            #     waypoint_remove = env.e2f_waypoints_queue.popleft()
            #     env.last_location2 = current_location2.transform.location

            # 找到与当前后车位置最近的队列元素索引 (Find the index of the waypoint closest to the ego vehicle)
            if len(env.e2f_waypoints_queue) > 2:
                distances = [current_location2.transform.location.distance(wp.transform.location) for wp in
                             env.e2f_waypoints_queue]
                if min(distances) < 1.0:
                    closest_index = distances.index(min(distances))

                    # 移除该索引及其之前的所有元素 (Remove the closest element and all elements before it)
                    for _ in range(closest_index + 1):
                        env.e2f_waypoints_queue.popleft()
                    env.last_location2 = current_location2.transform.location

            for wp in env.e2f_waypoints_queue:
                env.world.debug.draw_point(wp.transform.location, size=0.1, color=carla.Color(255, 0, 0),
                                           life_time=0.2)

            # 将前车和后车的位置插入
            env.e2f_waypoints_queue.appendleft(env.map.get_waypoint(env.ego_vehicle.get_location()))
            env.e2f_waypoints_queue.append(env.map.get_waypoint(env.front_vehicle_bp.get_location()))

            # 灵异事件可能来源于这里
            env.frenet.set_waypoints(env.e2f_waypoints_queue)

    threading.Thread(target=task, daemon=True).start()


frame_count = 0


def draw_pic():
    # 创建 DataFrame
    data = copy.deepcopy(env.data)
    # 确保所有列表的长度一致
    min_length = min(len(data["timestamp"]), len(data["idrl_acc"]), len(data["idrl_v"]),
                     len(data["idrl_d"]), len(data["idrl_rv"]), len(data["idrl_x"]), len(data["idrl_y"]),
                     len(data["idm_acc"]), len(data["idm_v"]), len(data["idm_d"]), len(data["idm_rv"]),
                     len(data["idm_x"]), len(data["idm_y"]),
                     len(data["acc"]), len(data["v"]), len(data["rv"]), len(data["d"]), len(data["x"]),
                     len(data["y"]),
                     len(data["rss_acc"]), len(data["rss_v"]), len(data["rss_d"]), len(data["rss_rv"]),
                     len(data["rss_x"]), len(data["rss_y"]))

    for key in data:
        data[key] = data[key][1:min_length]

    # 处理异常值
    for key in data:
        series = pd.Series(data[key])
        # 替换小于 -1000 和大于 1000 的值为前一个有效值
        for i in range(1, len(series)):
            if series[i] < -1000 or series[i] > 1000:
                series[i] = series[i - 1]

        data[key] = series.tolist()

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 保存为CSV文件
    df.to_csv('simulation_data.csv', index=False)

    # 简单的可视化示例
    plt.figure(figsize=(12, 12))  # 调整图形大小

    # 加速度对比图
    plt.subplot(3, 2, 1)
    plt.plot(data["timestamp"], data["idrl_acc"], label="IDRL Acceleration", color='b', linestyle='-')
    plt.plot(data["timestamp"], data["idm_acc"], label="IDM Acceleration", color='r', linestyle='--')
    plt.plot(data["timestamp"], data["rss_acc"], label="RSS Acceleration", color='pink', linestyle='--')
    plt.plot(data["timestamp"], data["acc"], label="Origin Acceleration", color='g', linestyle='--')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Acceleration")
    plt.title("Acceleration Comparison")
    plt.legend()

    # 速度对比图
    plt.subplot(3, 2, 2)
    plt.plot(data["timestamp"], data["idrl_v"], label="IDRL Velocity", color='b', linestyle='-')
    plt.plot(data["timestamp"], data["idm_v"], label="IDM Velocity", color='r', linestyle='--')
    plt.plot(data["timestamp"], data["rss_v"], label="RSS Velocity", color='pink', linestyle='--')
    plt.plot(data["timestamp"], data["v"], label="Origin Velocity", color='g', linestyle='--')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Velocity")
    plt.title("Velocity Comparison")
    plt.legend()

    # 相对速度对比图
    plt.subplot(3, 2, 3)
    plt.plot(data["timestamp"], data["idrl_rv"], label="IDRL Relative Velocity", color='b', linestyle='-')
    plt.plot(data["timestamp"], data["idm_rv"], label="IDM Relative Velocity", color='r', linestyle='--')
    plt.plot(data["timestamp"], data["rss_rv"], label="RSS Relative Velocity", color='pink', linestyle='--')
    plt.plot(data["timestamp"], data["rv"], label="Origin Relative Velocity", color='g', linestyle='--')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Relative Velocity")
    plt.title("Relative Velocity Comparison")
    plt.legend()

    # 相对车距对比图
    plt.subplot(3, 2, 4)
    plt.plot(data["timestamp"], data["idrl_d"], label="IDRL Relative Distance", color='b', linestyle='-')
    plt.plot(data["timestamp"], data["idm_d"], label="IDM Relative Distance", color='r', linestyle='--')
    plt.plot(data["timestamp"], data["rss_d"], label="RSS Relative Distance", color='pink', linestyle='--')
    plt.plot(data["timestamp"], data["d"], label="Origin Relative Distance", color='g', linestyle='--')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Relative Distance")
    plt.title("Relative Distance Comparison")
    plt.legend()

    # 轨迹图
    plt.subplot(3, 2, 5)
    plt.plot(data["x"], data["y"], label="Origin Trajectory", color='g', linestyle='-', marker='o')
    plt.plot(data["idrl_x"], data["idrl_y"], label="IDRL Trajectory", color='b', linestyle='-', marker='o')
    plt.plot(data["idm_x"], data["idm_y"], label="IDM Trajectory", color='r', linestyle='-', marker='o')
    plt.plot(data["rss_x"], data["rss_y"], label="RSS Trajectory", color='pink', linestyle='-', marker='o')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


def gather_data(interval):
    def task():
        while True:
            # print("Frame count:{}".format(env.frame_count))
            if env.frame_count % 100 == 0 and env.frame_count != 0:
                draw_pic()
            time.sleep(interval)

    threading.Thread(target=task, daemon=True).start()


# if env.mode != 'auto':
detect_front_junction(0.2)
update_frenet_distance()
gather_data(0.2)
Vf_Vet0 = np.zeros((50, 1))

while True:
    if env.green_light is True:
        env.set_traffic_lights_to_green()
    # f_vel = env.front_vehicle_bp.get_velocity()
    # autopilot_control = env.front_vehicle_bp.get_control()
    # current_speed_mps = (f_vel.x ** 2 + f_vel.y ** 2 + f_vel.z ** 2) ** 0.5
    # control = carla.VehicleControl()
    # control.steer = autopilot_control.steer  # 使用自动驾驶的转向控制
    # e_vel = env.ego_vehicle.get_velocity()
    # r_vel = math.sqrt(f_vel.x ** 2 + f_vel.y ** 2) - math.sqrt(e_vel.x ** 2 + e_vel.y ** 2)
    # f_pos = env.front_vehicle_bp.get_transform().location
    # e_pos = env.ego_vehicle.get_transform().location
    # # dis = math.sqrt((f_pos.x - r_pos.x) ** 2 + (f_pos.y - r_pos.y) ** 2)
    # dis = env.frenet.get_distance_by_location(f_pos, e_pos)
    # print(dis)
    # 更新场景
    env.step()
    # 更新后的量
    # 计算影子算法的值，变化量只有更新后能获得，因此放在env.step之后
    # if env.frame_count > 120:
    #     shade_state(f_vel, e_vel, r_vel, f_pos, e_pos, dis, acc, idm_acceleration, action, rss_mpc_acceleration)
    # env.frame_count += 1
    # print("acc:{},ddp:{},idm:{},rss:{}".format(acc, action, idm_acceleration, rss_mpc_acceleration))
    rear_waypoint = env.map.get_waypoint(env.ego_vehicle.get_location())
    front_waypoint = env.map.get_waypoint(env.front_vehicle_bp.get_location())
    # print("is curve{}, isJunction:{}".format(is_curve(front_waypoint, front_waypoint.next(5)[0]), front_waypoint.is_junction))
    # 后车的当前点位是junction
    # 记录前车经过路口后的waypoint
    # 意义不大了，因为前车经常甩后车几条街，就会走第二条分支
    if env.scenario != 'spacial':
        if rear_waypoint.is_junction and env.yaw_diff():
            # 获取前车的road_id
            # front_road_info = env.get_waypoint_info(env.front_vehicle_bp).road_id
            front_road_info = None
            if len(env.front_switch_road_waypoints) > 0:
                front_road_info = env.front_switch_road_waypoints[0].road_id
            else:
                front_road_info = env.get_waypoint_info(env.front_vehicle_bp).road_id
            junction = rear_waypoint.get_junction()
            ways = junction.get_waypoints(carla.LaneType.Driving)
            match_point = None
            for way in ways:
                for point in way:
                    if point.next(30)[-1].road_id == front_road_info:
                        match_point = point.next(30)[-1]
                        # print(env.destination)
                        break
            if match_point is not None:
                print("matched!")
                env.control_flag = False
                # 　屏蔽自动驾驶
                if env.mode == 'auto':
                    env.ego_vehicle.set_autopilot(False)
                # env.nav_to_waypoint(env.get_waypoint_info(env.front_vehicle_bp))
                if len(env.front_switch_road_waypoints) > 0:
                    env.destination = env.front_switch_road_waypoints[0]
                else:
                    env.destination = env.get_waypoint_info(env.front_vehicle_bp)
                # 导航会开启自动驾驶
                env.nav_to_waypoint(env.destination)
                env.control_flag = True

        elif is_curve(rear_waypoint, rear_waypoint.next(5)[0]) and env.yaw_diff():
            # env.destination = front_waypoint.next(5)[0]
            if len(env.front_switch_road_waypoints) > 0:
                env.destination = env.front_switch_road_waypoints[0]
            else:
                env.destination = env.get_waypoint_info(env.front_vehicle_bp)
            env.control_flag = False
            # 　屏蔽自动驾驶
            if env.mode == 'auto':
                env.ego_vehicle.set_autopilot(False)
            env.nav_to_waypoint(env.destination)
            env.control_flag = True

        elif test_flag is True:
            if env.mode == 'pid' or env.mode == 'idm' or env.mode == 'fvd' or env.mode == 'rss':
                env.follow_vehicle()
            if rear_waypoint.is_junction and env.yaw_similar():
                # 以免后车拐弯
                if env.mode == 'auto':
                    env.ego_vehicle.set_autopilot(False)
                    # 这里有可能出现后车跟的快，导致前车的队列还没有充入
                    if len(env.front_switch_road_waypoints) > 0:
                        env.destination = env.front_switch_road_waypoints.popleft()
                        env.nav_to_waypoint(env.destination)
                else:
                    tmp_transform = env.ego_vehicle.get_transform()
                    if len(env.front_switch_road_waypoints) > 0:
                        tmp_transform.rotation.yaw = find_closest_angle(
                            env.front_switch_road_waypoints.popleft().transform.rotation.yaw)
                        env.ego_vehicle.set_transform(tmp_transform)
                    else:
                        tmp_transform.rotation.yaw = find_closest_angle(
                            env.front_vehicle_bp.get_transform().rotation.yaw)
                # print("Driving adjust header.")
            # elif env.mode == 'auto' and env.ego_vehicle.is_autopilot_enabled() is False:
            #     env.ego_vehicle.set_autopilot(True)
            if env.debug is True:
                time.sleep(env.tau)
    else:
        env.scenario_spacial()
