# 导包
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

sys.path.append('/home/moresweet/carla/PythonAPI/carla')
sys.path.append('/home/moresweet/gitCloneZone/DIRL')
from agents.navigation.basic_agent import BasicAgent

# PID控制器参数
Kp = 1.0
Ki = 0.0
Kd = 0.1
integral = 0.0
previous_error = 0.0

rss_model = RSSModel()
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
        self.mode = 'auto'
        self.debug = False
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
        self.t_response = 3  # 反应时间 (s)
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
        self.cur_acc = 0
        self.e2f_waypoints_queue = deque()
        self.last_location1 = self.front_vehicle_bp.get_location()
        self.last_location2 = self.ego_vehicle.get_location()

    """
    生成carla场景
    """

    def spawn_vehicles(self):
        # 加载场景
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        vehicle_bp_rear = blueprint_library.filter('vehicle.*')[0]
        spawn_points = self.world.get_map().get_spawn_points()
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

    """
    步进更新对象状态
    """

    def step(self):
        if self.debug is True:
            # 推动仿真前进
            self.world.tick(self.tau)
            time.sleep(self.tau)
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
        distance_curve = sum(self.e2f_waypoints_queue[i].transform.location.distance(self.e2f_waypoints_queue[i + 1].transform.location) for i in
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

        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0))


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


# if env.mode != 'auto':
detect_front_junction(0.2)
update_frenet_distance()
while True:
    if env.green_light is True:
        env.set_traffic_lights_to_green()
    f_vel = env.front_vehicle_bp.get_velocity()
    autopilot_control = env.front_vehicle_bp.get_control()
    current_speed_mps = (f_vel.x ** 2 + f_vel.y ** 2 + f_vel.z ** 2) ** 0.5
    control = carla.VehicleControl()
    control.steer = autopilot_control.steer  # 使用自动驾驶的转向控制
    e_vel = env.ego_vehicle.get_velocity()
    r_vel = math.sqrt(f_vel.x ** 2 + f_vel.y ** 2) - math.sqrt(e_vel.x ** 2 + e_vel.y ** 2)
    f_pos = env.front_vehicle_bp.get_transform().location
    r_pos = env.ego_vehicle.get_transform().location
    dis = math.sqrt((f_pos.x - r_pos.x) ** 2 + (f_pos.y - r_pos.y) ** 2)
    nor_state = [dis / 120, r_vel / 40, math.sqrt(r_vel ** 2 + r_vel ** 2) / 40]
    nor_state = torch.FloatTensor(nor_state).to(env.device)
    action = env.ddpg.choose_action(nor_state)
    action = action * 10
    env.step()
    acc = env.ego_vehicle.get_acceleration()
    acc = math.sqrt(acc.x ** 2 + acc.y ** 2)
    env.ddpg_acc = action
    env.cur_acc = acc

    rear_waypoint = env.map.get_waypoint(env.ego_vehicle.get_location())
    front_waypoint = env.map.get_waypoint(env.front_vehicle_bp.get_location())
    # print("is curve{}, isJunction:{}".format(is_curve(front_waypoint, front_waypoint.next(5)[0]), front_waypoint.is_junction))
    # 后车的当前点位是junction
    # 记录前车经过路口后的waypoint
    # 意义不大了，因为前车经常甩后车几条街，就会走第二条分支
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
        if env.mode == 'pid':
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
                    tmp_transform.rotation.yaw = find_closest_angle(env.front_vehicle_bp.get_transform().rotation.yaw)
            # print("Driving adjust header.")
        # elif env.mode == 'auto' and env.ego_vehicle.is_autopilot_enabled() is False:
        #     env.ego_vehicle.set_autopilot(True)
        if env.debug is True:
            time.sleep(env.tau)
