import torch

from RewardFun import *
import matplotlib.pyplot as plt
import torch.utils.data as Data
import scipy.io as sio
import numpy as np


def old_fuc(state):
    dis = state[0]
    vr = state[1]
    vx = state[2]
    acc = state[3]
    f1 = pow(acc, 2)
    g = (vr + vx) * 4
    f2 = pow(g - dis, 2)
    f3 = pow(vr, 2)
    f4 = pow(1/dis, 2)
    f = 10 * f1 + 0.2 * f2 + 5 * f3 + 1 * f4
    return [f, f1, f2, f3, f4]

def new_fuc(state):
    dis = state[0]
    vr = state[1]
    vx = state[2]
    acc = state[3]
    # 参数设定
    t_reaction = 4  # 反应时间 (秒)
    a_max = 3  # 后车的最大减速度 (m/s²)
    a_min = 2  # 前车的最小减速度 (m/s²)
    # 计算最小安全距离
    d_min = vx * t_reaction + (vx ** 2) / (2 * a_max) - ((vx - vr) ** 2) / (2 * a_min)
    f1 = pow(acc, 2)
    g = (vr + vx) * t_reaction
    f2 = pow(g - dis, 2)
    f3 = pow(vr, 2)
    # 计算调和平均数
    H = 2 * dis * d_min / (dis + d_min)
    f4 = pow(1 / H, 2)
    f = 10 * f1 + 0.2 * f2 + 5 * f3 + 1 * f4
    return [f, f1, f2, f3, f4]


load_data = sio.loadmat('EgoInfo.mat')
EgoInfo = load_data['EgoInfo']
a = EgoInfo[0, 0][0, [0, 1, 2, 3]]
# n = old_fuc(a)】
n = new_fuc(a)
f1_list = []
f2_list = []
f3_list = []
f4_list = []
for i in np.arange(1, 120, 5):  # 跟车距离
    for j in np.arange(-30, 30, 1):  # 相对速度
        for k in np.arange(1, 30, 2):  # 车速
            for m in np.arange(-5, 3, 0.2):  # 加速度
                # f = old_fuc([i, j, k, m])
                f = new_fuc([i, j, k, m])
                f1_list.append(f[1])
                f2_list.append(f[2])
                f3_list.append(f[3])
                f4_list.append(f[4])
f1_list = np.array(f1_list)
f2_list = np.array(f2_list)
f3_list = np.array(f3_list)
f4_list = np.array(f4_list)

f1_max = max(f1_list)
f2_max = max(f2_list)
f3_max = max(f3_list)
f4_max = max(f4_list)

# 样本空间生成
Train_input = []
Train_target = []
for i in np.arange(1, 120, 5):  # 跟车距离
    for j in np.arange(-30, 30, 1):  # 相对速度
        for k in np.arange(1, 30, 2):  # 车速
            for m in np.arange(-5, 3, 0.2):  # 加速度
                # f = old_fuc([i, j, k, m])
                f = new_fuc([i, j, k, m])
                f1 = f[1]
                f2 = f[2]
                f3 = f[3]
                f4 = f[4]
                r = f1/f1_max + f2/f2_max + f3/f3_max + f4/f4_max
                Train_input.append([i/120, j/40, k/40, m/10])
                Train_target.append(r)

for i in range(len(EgoInfo[0, :])):
    for j in range(len(EgoInfo[0, i])):
        tmp = EgoInfo[0, i][j, [0, 1, 2, 3]] / np.array([120, 40, 40, 10])
        Train_input.append(list(tmp))
        # f = old_fuc(EgoInfo[0, i][j, [0, 1, 2, 3]])
        f = new_fuc(EgoInfo[0, i][j, [0, 1, 2, 3]])
        f1 = f[1]
        f2 = f[2]
        f3 = f[3]
        f4 = f[4]
        r = f1/f1_max + f2/f2_max + f3/f3_max + f4/f4_max
        Train_target.append(r)

Train_input = np.array(Train_input)
Train_target = np.array(Train_target)
Train_input = torch.from_numpy(Train_input)
Train_target = torch.from_numpy(Train_target)

batch_size = 256
dataset = Data.TensorDataset(Train_input, Train_target)
loader = Data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,  # 分几批
    shuffle=True,  # 是否打乱数据，默认为False
    num_workers=0,  # 用多线程读数据，默认0表示不使用多线程
)

# 训练
in_dim = 4
out_dim = 1
Pre_RewardNet = RewardNet(in_dim, out_dim)
optimizer = torch.optim.Adam(Pre_RewardNet.parameters(), lr=0.001)
loss_fun = torch.nn.MSELoss()
eps = 500
total_train_step = 0
losslist = []
minlosslist = []
for epoch in range(eps):
    for i, data in enumerate(loader):
        # 将数据从 train_loader 中读出来,一次读取的样本数是batch_size个
        inputs, labels = data
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        prediction = Pre_RewardNet(inputs)  # prediction是关于网络参数的函数
        # labels = labels[:, 0]
        prediction = prediction[:, 0]
        # loss = loss_fun(prediction, labels)  # loss是关于网络参数的函数
        loss = loss_fun(torch.div(prediction, labels), torch.div(labels, labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        losslist.append(loss.item())
        minlosslist.append(min(losslist))
        if total_train_step % 5000 == 0:
            print("训练次数：{}, loss: {}".format(total_train_step, min(losslist)))
            # plt.subplot(2, 1, 1)
            # plt.plot(minlosslist)
            # # torch.save(Pre_RewardNet, 'Pre_RewardNet.pkl')
            # plt.subplot(2, 1, 2)
            plt.plot(minlosslist[-1000:])
            plt.show()
            torch.save(Pre_RewardNet, 'Pre_RewardNet_new_func.pkl')
