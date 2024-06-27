from contextlib import ExitStack
import time
import numpy as np
from DDPG import *
from CarFollowing import *
from RewardFun import *
from NetFun import *
from itertools import count
import torch
import matplotlib.pyplot as plt
from pylab import *
from geomdl import fitting

import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

with ExitStack() as s:
    f1 = s.enter_context(open('FollowPrd.csv', 'r'))
    num = []
    for i, x in enumerate(f1):
        Line1 = x.split(',')
        num.append(list(map(float, Line1)))
RawData = np.array(num)

# 数据预处理
# 输入： 相对间距、相对速度、后车车速
train_in_seq = []
# 输出（标签）：后车加速度Ax
train_out_seq = []
Vx_pred = []
Vr_pred = []
R_pred = []
Ax_pred = []
for i in range(int(num[-1][0])):
    train_in_seq.append([])
    train_out_seq.append([])
    Vx_pred.append([])
    Vr_pred.append([])
    R_pred.append([])
    Ax_pred.append([])
for i, x in enumerate(num):
    train_in_seq[int(x[0]) - 1].append([x[1], x[2], x[3] / 3.6])
    train_out_seq[int(x[0]) - 1].append([x[4]])

R_max = max(RawData[:, 1])
R_min = min(RawData[:, 1])
Vr_max = max(RawData[:, 2])
Vr_min = min(RawData[:, 2])
Vx_max = max(RawData[:, 3] / 3.6)
Vx_min = min(RawData[:, 3] / 3.6)
Ax_max = max(RawData[:, 4])
Ax_min = min(RawData[:, 4])

# 多项式拟合：
# input：自变量，因变量，阶数
# output：多项式拟合的系数的数组
PolyCof = np.polyfit(RawData[:, 3] / 3.6, RawData[:, 1], 3)
# 从多项式拟合得到的系数创建了一个多项式对象
# input：多项式拟合的系数的数组
# output：代表一个多项式函数，可以使用它来根据数据中观察到的关系进行预测或插值
Vx_Dis_poly = np.poly1d(PolyCof)

# 奖励函数
in_dim = 3     # 当前先仅把加速度、相对速度、相对距离作为输入
n_hidden_1 = 10
n_hidden_2 = 10
out_dim = 1
rewordfcn = Net(in_dim, n_hidden_1, n_hidden_2, out_dim)
optimizer_r = torch.optim.SGD(rewordfcn.parameters(), lr=0.01)
loss_mse_fun = torch.nn.MSELoss()

num_state = 3
num_action = 1
MEMORY_CAPACITY = 100000
REPLACEMENT = {'name': 'soft', 'tau': 0.01}  # 软更新
max_action = 1
a_bound = max_action
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 自己根据跟车过程写个跟车的env
env = CarFollowing()
ddpg = DDPG(state_dim=num_state, action_dim=num_action, action_bound=a_bound, replacement=REPLACEMENT, memory_capacticy=MEMORY_CAPACITY)
ddpg = torch.load('ddpg_carfollow.pt', map_location=device)
if_cuda = torch.cuda.is_available()
print("if_cuda=", if_cuda)

MAX_EPISODES = 500          # Episode数目
total_step = 0
TrainFlag = True

if TrainFlag:
    reward_loss_list = []
    for m in range(100):
        # 计算专家轨迹期望回报
        ExpTR = 0
        n = 0
        for j in range(len(train_in_seq)):
            for k in range(len(train_in_seq[j])):
                action = train_out_seq[j][k][0]
                e_dis = train_in_seq[j][k][0]-Vx_Dis_poly(train_in_seq[j][k][2])
                vr = train_in_seq[j][k][1]
                inputs = [action, e_dis, vr]
                inputs = torch.tensor(inputs).to(torch.float32)
                ExpTR += rewordfcn(inputs)
                n += 1
        ExpR = ExpTR/n
        # 训练DDPG
        ep_reward_list = []
        ddpg = torch.load('ddpg_carfollow.pt', map_location=device)
        for i in range(MAX_EPISODES):
            ep_reward = 0
            step = 0
            # for j in range(len(train_in_seq)):
            for j in [9]:
                state = env.reset(np.array(train_in_seq[j]), np.array(train_out_seq[j]))
                state = torch.FloatTensor(state).to(device)
                for k in range(1, len(train_in_seq[j])):
                    if total_step < 1:
                        raw_action = np.random.uniform(-1, 1, 1)
                    else:
                        raw_action = ddpg.choose_action(state)
                        raw_action = (raw_action + np.random.normal(0.1, 0.2))
                        if raw_action > 1:
                            raw_action = 1
                        elif raw_action < -1:
                            raw_action = -1
                    action = raw_action * 4 - 1
                    # print(action)
                    next_state = env.step(action)
                    e_dis = float(next_state[0]) - Vx_Dis_poly(float(next_state[2]))
                    vr = float(next_state[1])
                    next_state = torch.FloatTensor(next_state).to(device)
                    inputs = [float(action), e_dis, vr]
                    # inputs = torch.tensor(inputs).to(torch.float32)
                    inputs = torch.FloatTensor(inputs).to(dtype=torch.float32, device=device)
                    reward = -(abs(e_dis)+abs(action)+abs(vr))
                    # reward = rewordfcn(inputs).item()
                    state_cpu = state.cpu()
                    next_state_cpu = next_state.cpu()
                    ddpg.store_transition(state_cpu, float(raw_action), float(reward), next_state_cpu)
                    state = next_state
                    ep_reward += reward
                    step += 1
                    total_step += 1
                    if state[0] > 150 or state[0] < 0:
                        break
            if ddpg.pointer > MEMORY_CAPACITY:   # 经验池大于10000
               ddpg.learn()
            ep_reward_list.append(ep_reward)
            print("Episode: %d\t Total Reward: %f\t step：%d\t total_step:%d" % (i, ep_reward, step, total_step))
            # plt.figure(1)
            # plt.clf()
            # plt.plot(reward_loss_list)
            # plt.pause(0.01)
        torch.save(ddpg, 'ddpg.pt')
        # 根据训练好的DDPG计算最优轨迹，计算最优轨迹的回报期望
        DDPG_TR = 0
        n = 0
        for j in range(len(train_in_seq)):
            state = env.reset(np.array(train_in_seq[j]), np.array(train_out_seq[j]))
            state = torch.FloatTensor(state).to(device)
            R_pred[j].append(state[0])
            Vr_pred[j].append(state[1])
            Vx_pred[j].append(state[2])
            Ax_pred[j].append(train_out_seq[j][0][0])
            for k in range(1, len(train_in_seq[j])):
                action = ddpg.choose_action(state)
                action = action * 4 - 1
                next_state = env.step(action)
                next_state = torch.FloatTensor(next_state).to(device)
                state = next_state
                R_pred[j].append(float(state[0]))
                Vr_pred[j].append(state[1])
                Vx_pred[j].append(float(state[2]))
                Ax_pred[j].append(float(action))

                e_dis = float(state[0]) - Vx_Dis_poly(float(state[2]))
                vr = float(state[1])
                inputs = [float(action), e_dis, vr]
                # inputs = torch.tensor(inputs).to(torch.float32)
                inputs = torch.FloatTensor(inputs).to(device)
                DDPG_TR += rewordfcn(inputs)
                n += 1
        DDPG_R = DDPG_TR/n
        # 计算梯度
        reward_loss = loss_mse_fun(DDPG_R, ExpR)
        # 更新reward网络
        optimizer_r.zero_grad()
        reward_loss.backward()
        optimizer_r.step()
        # 绘制奖励函数网络的损失变化
        reward_loss_list.append(reward_loss.item())
        print("*************************Iter: \t %d loss: \t %f*****************************" % (m, reward_loss.item()))
        if m > 0 and m % 20 == 0:
            torch.save(ddpg, 'ddpg.pt')
            torch.save(rewordfcn, 'rewordfcn1.pkl')
else:
    ddpg = torch.load('ddpg.pt', map_location=device)
    ep_reward = 0

    # 绘制某个跟车周期的效果图
    j = 9
    state = env.reset(np.array(train_in_seq[j]), np.array(train_out_seq[j]))
    R_pred[j].append(state[0])
    Vx_pred[j].append(state[2])
    state = torch.FloatTensor(state).to(device)
    for k in range(1, len(train_in_seq[j])):
        action = ddpg.choose_action(state)
        action = action * 4 - 1
        next_state = env.step(action)
        R_pred[j].append(float(state[0]))
        Vx_pred[j].append(float(state[2]))
        Ax_pred[j].append(float(action))
        next_state = torch.FloatTensor(next_state).to(device)
        state = next_state
        # ep_reward += reward
    # print("Ep_i \t{}, the ep_r is \t{:0.2f}".format(j, ep_reward))
    fig = plt.figure(1)
    plt.plot(np.array(Vx_pred[j]), color='red')
    plt.plot(env.Vx_Ref_Seq, color='blue')
    plt.plot(env.VpSeq, color='black')
    fig = plt.figure(2)
    plt.plot(np.array(R_pred[j]), color='red')
    plt.plot(env.R_Ref_Seq, color='blue')
    fig = plt.figure(3)
    plt.plot(np.array(Ax_pred[j]), color='red')
    plt.show()

    # 计算速度和距离的RMSE
    Vx_MSE = 0
    R_MSE = 0
    N = 0
    AgentFollowStr = []
    for j in range(len(train_in_seq)):
        state = env.reset(np.array(train_in_seq[j]), np.array(train_out_seq[j]))
        R_pred[j].append(state[0])
        Vr_pred[j].append(state[1])
        Vx_pred[j].append(state[2])
        Ax_pred[j].append(train_out_seq[j][0][0])
        state = torch.FloatTensor(state).to(device)
        AgentFollowStr.append([str(j),str(R_pred[j][-1]),str(Vr_pred[j][-1]),str(Vx_pred[j][-1]),str(Ax_pred[j][-1])])
        R_MSE = R_MSE + ((R_pred[j][0] - env.R_Ref_Seq[0]) ** 2)
        Vx_MSE = Vx_MSE + ((Vx_pred[j][0] - env.Vx_Ref_Seq[0]) ** 2)
        N = N + 1
        for k in range(1, len(train_in_seq[j])):
            action = ddpg.choose_action(state)
            action = action * 4 - 1
            next_state, reward = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            state = next_state
            R_pred[j].append(float(state[0]))
            Vr_pred[j].append(state[1])
            Vx_pred[j].append(float(state[2]))
            Ax_pred[j].append(float(action))
            AgentFollowStr.append([str(j), str(R_pred[j][-1]), str(Vr_pred[j][-1]), str(Vx_pred[j][-1]),
                                  str(Ax_pred[j][-1])])
            R_MSE = R_MSE + ((R_pred[j][k] - env.R_Ref_Seq[k]) ** 2)
            Vx_MSE = Vx_MSE + ((Vx_pred[j][k] - env.Vx_Ref_Seq[k]) ** 2)
            N = N + 1
    R_RMSE = (R_MSE / N) ** 0.5
    Vx_RMSE = (Vx_MSE / N) ** 0.5
    print("R_RMSE is {:0.2f}".format(R_RMSE))
    print("Vx_RMSE is {:0.2f}".format(Vx_RMSE))


