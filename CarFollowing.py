import math
import numpy as np
from RewardFun import *


class CarFollowing(object):
    def __init__(self):
        super(CarFollowing, self).__init__()
        self.Ts = 0.1  # 时间步长

    def reset(self, train_in_seq, train_out_seq):
        # 导入原始数据
        self.R_Ref_Seq = train_in_seq[:, 0]  # 相对距离的参考值 m
        self.Vr_Ref_Seq = train_in_seq[:, 1]  # 相对速度的参考值 m/s
        self.Vx_Ref_Seq = train_in_seq[:, 2]  # 车速的参考值 m/s
        self.Ax_Ref_Seq = train_out_seq[:, 0]  # 加速度的参考值 m/s2
        self.VpSeq = self.Vr_Ref_Seq + self.Vx_Ref_Seq  # 前车速度 m/s2

        # 环境的当前状态
        self.Index = 0
        self.R = self.R_Ref_Seq[0]
        self.Vr = self.Vr_Ref_Seq[0]
        self.Vx = self.Vx_Ref_Seq[0]
        self.Vp = self.VpSeq[0]
        s = [self.R, self.Vr, self.Vx]
        return np.array(s)

    def step(self, a):
        self.Index += 1
        Ax = a
        if self.Index < len(self.VpSeq):
            Vp_ = self.VpSeq[self.Index]
        else:
            Vp_ = self.VpSeq[-1]
        Vx_ = self.Vx + Ax * self.Ts  # 下一时刻车速
        Vr_ = Vp_ - Vx_  # 相对速度
        R_ = self.R + (Vr_ + self.Vr) / 2 * self.Ts  # 下一时刻相对距离

        if self.Index >= len(self.R_Ref_Seq):
            self.Index = 0

        self.Vx = Vx_  # 更新后的车速
        self.R = R_  # 更新后的相对距离
        self.Vr = Vr_  # 更新后的相对速度
        s_ = [float(self.R), float(self.Vr), float(self.Vx)]
        return np.array(s_)
