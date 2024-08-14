#!/usr/bin/env python

import glob
import os
import sys
import numpy as np
from numpy.linalg import matrix_power
from qpsolvers import solve_qp

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pandas as pd
import random
import time


def main(i_d, i_v, model):
    actor_list = []
    try:

        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(5.0)
        # world = client.reload_world()
        world = client.load_world('Town01')
        # world = client.load_world('Town01')
        # 清除所有车辆
        for vehicle in world.get_actors().filter('vehicle.*'):
            vehicle.destroy()

        settings = world.get_settings()
        settings.synchronous_mode = False  # 将此值设置为True以使用同步模式
        fixed_delta_seconds = 0.01
        settings.fixed_delta_seconds = fixed_delta_seconds
        world.apply_settings(settings)

        ego_bp = world.get_blueprint_library().find('vehicle.nissan.patrol_2021')
        ego_bp.set_attribute('role_name', 'hero')

        if ego_bp.has_attribute('color'):
            ego_bp.set_attribute('color', '255,255,255')

        ego_x, ego_y, ego_z = 392.379, 10.538, 0.2
        transform = carla.Transform(carla.Location(x=ego_x, y=ego_y, z=ego_z), carla.Rotation(yaw=90))

        vehicle = world.spawn_actor(ego_bp, transform)

        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        vehicle.set_autopilot(False)

        # world = client.get_world(client)
        # vehicles = [actor for actor in world.get_actors() if 'vehicle' in actor.type_id and actor.attributes['role_name'] == "ego_vehicle"]

        # npc
        npc_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        npc_bp.set_attribute('role_name', 'npc')
        location = vehicle.get_location()
        spawn_point2 = carla.Transform(carla.Location(x=ego_x, y=ego_y + 5 + i_d, z=ego_z), carla.Rotation(yaw=90))
        NPC = world.spawn_actor(npc_bp, spawn_point2)
        actor_list.append(NPC)
        print('created %s' % NPC.type_id)

        # spectator
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        transform.location += carla.Location(x=0, y=-7, z=2.7)
        transform.rotation.yaw += 1
        transform.rotation.pitch -= 2
        spectator.set_transform(transform)

        # transform.location += carla.Location(x=40, y=-3.2)
        # transform.rotation.yaw = -180.0

        vehicle.set_simulate_physics(enabled=False)
        NPC.set_simulate_physics(enabled=False)
        # 初始速度
        v_ego_last = v_npc_last = i_v / 3.6

        Np = 50
        Ts = fixed_delta_seconds
        Vf_Vet0 = np.zeros((Np, 1))
        S = NPC.get_location().y - vehicle.get_location().y - 5
        vf0 = v_npc_last
        Vx0 = v_ego_last
        Tx = 0.5
        vf = vf0
        Vx = Vx0
        acc_ego = 0
        Ax = acc_ego
        acc_leader = -2
        last_Ax = 0
        t0 = 0
        TET = 0
        TIT = 0
        while True:

            # spectator
            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            transform.location += carla.Location(x=0, y=-7, z=4)
            transform.rotation.yaw += 0
            transform.rotation.pitch -= 2
            # transform.location += carla.Location(x=8, y=3, z=4)
            # transform.rotation.yaw += 70
            # transform.rotation.pitch -= 2
            spectator.set_transform(transform)

            if settings.synchronous_mode:
                world.tick()
            else:
                world.wait_for_tick()

            S = NPC.get_location().y - vehicle.get_location().y - 5
            Vx = Vx + Ax * Ts
            Vr = vf - Vx
            # Vx += Ts * Ax
            # Vr = vf - Vx
            if Vx <= 0:
                Vx = 0
            Vf_Vet = Vf_Vet0 + vf
            Ax_req = Control_MPC(Np, S, Vr, Vx, Vf_Vet, vf, model)[0]
            Sref_RSS = Control_MPC(Np, S, Vr, Vx, Vf_Vet, vf, model)[1]
            Sref_ACC = Control_MPC(Np, S, Vr, Vx, Vf_Vet, vf, model)[2]
            Ax = (1 - Ts / Tx) * last_Ax + Ts / Tx * Ax_req
            last_Ax = Ax
            if Vr >= 0:
                TTC = 18
            elif S / -Vr > 18:
                TTC = 18
            else:
                TTC = S / -Vr

            if TTC < 3:
                TET += fixed_delta_seconds
                TIT = TIT + (3 - TTC) * fixed_delta_seconds

            S_values.append(S)
            Vx_values.append(Vx)
            vf_values.append(vf)
            Ax_values.append(Ax)
            Sref_RSS_value.append(Sref_RSS)
            Sref_ACC_value.append(Sref_ACC)
            TTC_value.append(TTC)
            t0 += fixed_delta_seconds
            print(S, Vx, vf, Ax, t0, TTC, TET, TIT)

            vehicle.set_transform(carla.Transform(carla.Location(x=vehicle.get_location().x,
                                                                 y=vehicle.get_location().y + Vx * fixed_delta_seconds
                                                                 ,
                                                                 z=vehicle.get_location().z), carla.Rotation(yaw=90)))

            NPC.set_transform(carla.Transform(carla.Location(x=NPC.get_location().x,
                                                             y=NPC.get_location().y + v_npc_last * fixed_delta_seconds
                                                             ,
                                                             z=NPC.get_location().z), carla.Rotation(yaw=90)))

            time.sleep(fixed_delta_seconds)

            v_npc_last = v_npc_last + acc_leader * fixed_delta_seconds
            vf = v_npc_last
            if v_npc_last <= 0:
                v_npc_last = 0
                acc_leader = 0

            if NPC.get_location().y > 312.541 or t0 >= 14 or S <= 0:
                data = {
                    'S': S_values,
                    'Vx': Vx_values,
                    'vf': vf_values,
                    'Ax': Ax_values,
                    'Sref_RSS': Sref_RSS_value,
                    'Sref_ACC': Sref_ACC_value,
                    'TTC': TTC_value,
                    'TET': TET,
                    'TIT': TIT
                }
                df = pd.DataFrame(data)
                excel_file = f'D:\data\simulation_data_{model}_i_d_{i_d}_i_v_{i_v}.xlsx'  # Change this to your desired file name
                df.to_excel(excel_file, index=False)

                break

    finally:

        print('destroying actors')
        # camera.destroy()
        for actor in actor_list:
            actor.destroy()
        print('done.')


def Control_MPC(Np, S, Vr, Vx, Vf, vf, model):
    # 控制器参数
    Ts = 0.2  # 采样间隔
    Nx = 4  # 状态量个数
    Nu = 1  # 控制量个数
    Nc = Np  # 控制长度
    # 驾驶员2的参数 A
    q = 1 * np.diag([50, 50, 1])
    r = 1 * np.array([10])
    Sref_ACC = 1.5 * Vx + 3.5  # acc
    Tp = 1.5
    acc_max = 2.5
    abr_min = 3
    abr_max = 8
    Sref_RSS = Vx * Tp + 0.5 * acc_max * (Tp ** 2) + (Vx + Tp * acc_max) ** 2 / (2 * abr_min) - vf ** 2 / (2 * abr_max)
    if model == 'ACC':
        Sref = Sref_ACC
    if model == 'RSS':
        Sref = Sref_RSS

    # 驾驶员3的参数
    # q = 1 * np.diag([20, 50, 1])
    # r = np.array([10])
    # Sref = 2 * (Vr + Vx)

    # 驾驶员4的参数  B
    # q = 1 * np.diag([1, 70, 1])
    # r = np.array([10])
    # Sref = 2 * (Vr + Vx)

    # 驾驶员5的参数  C
    # q = 1 * np.diag([3, 50, 1])
    # r = np.array([10])
    # Sref = 3 * (Vr + Vx)

    # 系统预测模型
    X = np.array([S, Vr, Vx, 1 / S])
    A = np.array([[1, Ts, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 1, 0],
                  [0, -Ts / (S ** 2), 0, 1]])
    B = np.array([-Ts ** 2 / 2, -Ts, Ts, 0])
    B = B.reshape(-1, 1)
    G = np.array([0, 1, 0, 0])
    G = G.reshape(-1, 1)
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])

    Ny = 3

    Fai = [[None for _ in range(Np)] for _ in range(Np)]
    Fm = [None] * Np
    Gm = [None] * Np
    for j in range(Np):
        Fm[j] = np.dot(C, matrix_power(A, j + 1))
        Fai[j] = [None] * Np  # 初始化 Fai[j] 为包含 Np 个元素的空列表
        Gm[j] = [None] * Np
        for k in range(Np):
            if k < j + 1:
                Fai[j][k] = np.dot(C, matrix_power(A, j - k) @ B)
                Gm[j][k] = np.dot(C, matrix_power(A, j - k) @ G)
            else:
                Fai[j][k] = np.zeros((Ny, Nu))
                Gm[j][k] = np.zeros((Ny, Nu))

    Fai = np.array(Fai)
    Fai = np.concatenate(Fai, axis=1)
    Fai = np.concatenate(Fai, axis=1)
    Fm = np.concatenate(Fm, axis=0)
    Gm = np.concatenate(Gm, axis=1)
    Gm = np.concatenate(Gm, axis=1)

    q_row, q_col = q.shape
    r_row, r_col = 1, 1

    Q = np.zeros((q_row * Np, q_col * Np))
    R = np.zeros((r_row * (Nc), r_row * (Nc)))

    for j in range(Np):
        for k in range(Np):
            if j == k:
                Q[j * q_row:(j + 1) * q_row, k * q_col:(k + 1) * q_col] = q
            else:
                Q[j * q_row:(j + 1) * q_row, k * q_col:(k + 1) * q_col] = np.zeros_like(q)

    for j in range(Nc):
        for k in range(Nc):
            if j == k:
                R[j * r_row:(j + 1) * r_row, k * r_col:(k + 1) * r_col] = r
            else:
                R[j * r_row:(j + 1) * r_row, k * r_col:(k + 1) * r_col] = np.zeros_like(r)

    Yr = np.zeros(Ny * Np)
    for i in range(Np):
        Yr[i * Ny - 3] = Sref

    # 利用二次规划求解目标函数
    H = 2 * (Fai.T @ Q @ Fai + R)
    f = 2 * (X.T @ Fm.T @ Q @ Fai - Yr @ Q @ Fai + Vf.T @ Gm.T @ Q @ Fai)
    A_cons = np.array([])
    b_cons = np.array([])
    lb = np.tile([-5], (Nc, 1))
    ub = np.tile([3], (Nc, 1))

    x = solve_qp(H, f, lb=lb, ub=ub, solver="quadprog")
    Areq = x[0]
    return Areq, Sref_RSS, Sref_ACC


if __name__ == '__main__':

    for k in ['ACC', 'RSS']:  # 策略
        for i in [50, 40, 30]:  # 初始前后车速度
            for j in [80, 70, 60, 50, 40]:  # 初始前后车距离
                i_d = j
                i_v = i
                model = k
                S_values = []
                Vx_values = []
                vf_values = []
                Ax_values = []
                Sref_RSS_value = []
                Sref_ACC_value = []
                TTC_value = []
                main(i_d, i_v, model)
