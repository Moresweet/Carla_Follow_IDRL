import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def draw_pic_range(data):
    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 设置图形大小
    plt.figure(figsize=(12, 12))

    # 加速度对比图
    plt.subplot(3, 2, 1)
    plt.fill_between(data["timestamp"], data["idrl_acc"], alpha=0.2, color='#14517C')
    plt.plot(data["timestamp"], savgol_filter(data["idrl_acc"], 51, 3), label="IDRL Acceleration", color='#14517C',
             linestyle='-')
    plt.fill_between(data["timestamp"], data["idm_acc"], alpha=0.2, color='#9AC9DB')
    plt.plot(data["timestamp"], savgol_filter(data["idm_acc"], 51, 3), label="IDM Acceleration", color='#9AC9DB',
             linestyle='--')
    plt.fill_between(data["timestamp"], data["rss_acc"], alpha=0.2, color='#F8AC8C')
    plt.plot(data["timestamp"], savgol_filter(data["rss_acc"], 51, 3), label="RSS Acceleration", color='#F8AC8C',
             linestyle='--')
    plt.fill_between(data["timestamp"], data["acc"], alpha=0.2, color='#C82423')
    plt.plot(data["timestamp"], savgol_filter(data["acc"], 51, 3), label="Origin Acceleration", color='#C82423',
             linestyle='--')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Acceleration")
    plt.title("Acceleration Comparison")
    plt.legend()

    # 速度对比图
    plt.subplot(3, 2, 2)
    plt.fill_between(data["timestamp"], data["idrl_v"], alpha=0.2, color='#14517C')
    plt.plot(data["timestamp"], savgol_filter(data["idrl_v"], 51, 3), label="IDRL Velocity", color='#14517C',
             linestyle='-')
    plt.fill_between(data["timestamp"], data["idm_v"], alpha=0.2, color='#9AC9DB')
    plt.plot(data["timestamp"], savgol_filter(data["idm_v"], 51, 3), label="IDM Velocity", color='#9AC9DB',
             linestyle='--')
    plt.fill_between(data["timestamp"], data["rss_v"], alpha=0.2, color='#F8AC8C')
    plt.plot(data["timestamp"], savgol_filter(data["rss_v"], 51, 3), label="RSS Velocity", color='#F8AC8C',
             linestyle='--')
    plt.fill_between(data["timestamp"], data["v"], alpha=0.2, color='#C82423')
    plt.plot(data["timestamp"], savgol_filter(data["v"], 51, 3), label="Origin Velocity", color='#C82423',
             linestyle='--')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Velocity")
    plt.title("Velocity Comparison")
    plt.legend()

    # 相对速度对比图
    plt.subplot(3, 2, 3)
    plt.fill_between(data["timestamp"], data["idrl_rv"], alpha=0.2, color='#14517C')
    plt.plot(data["timestamp"], savgol_filter(data["idrl_rv"], 51, 3), label="IDRL Relative Velocity", color='#14517C',
             linestyle='-')
    plt.fill_between(data["timestamp"], data["idm_rv"], alpha=0.2, color='#9AC9DB')
    plt.plot(data["timestamp"], savgol_filter(data["idm_rv"], 51, 3), label="IDM Relative Velocity", color='#9AC9DB',
             linestyle='--')
    plt.fill_between(data["timestamp"], data["rss_rv"], alpha=0.2, color='#F8AC8C')
    plt.plot(data["timestamp"], savgol_filter(data["rss_rv"], 51, 3), label="RSS Relative Velocity", color='#F8AC8C',
             linestyle='--')
    plt.fill_between(data["timestamp"], data["rv"], alpha=0.2, color='#C82423')
    plt.plot(data["timestamp"], savgol_filter(data["rv"], 51, 3), label="Origin Relative Velocity", color='#C82423',
             linestyle='--')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Relative Velocity")
    plt.title("Relative Velocity Comparison")
    plt.legend()

    # 相对车距对比图
    plt.subplot(3, 2, 4)
    plt.fill_between(data["timestamp"], data["idrl_d"], alpha=0.2, color='#14517C')
    plt.plot(data["timestamp"], savgol_filter(data["idrl_d"], 51, 3), label="IDRL Relative Distance", color='#14517C',
             linestyle='-')
    plt.fill_between(data["timestamp"], data["idm_d"], alpha=0.2, color='#9AC9DB')
    plt.plot(data["timestamp"], savgol_filter(data["idm_d"], 51, 3), label="IDM Relative Distance", color='#9AC9DB',
             linestyle='--')
    plt.fill_between(data["timestamp"], data["rss_d"], alpha=0.2, color='#F8AC8C')
    plt.plot(data["timestamp"], savgol_filter(data["rss_d"], 51, 3), label="RSS Relative Distance", color='#F8AC8C',
             linestyle='--')
    plt.fill_between(data["timestamp"], data["d"], alpha=0.2, color='#C82423')
    plt.plot(data["timestamp"], savgol_filter(data["d"], 51, 3), label="Origin Relative Distance", color='#C82423',
             linestyle='--')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Relative Distance")
    plt.title("Relative Distance Comparison")
    plt.legend()

    # 轨迹图
    plt.subplot(3, 2, 5)
    plt.plot(data["x"], data["y"], label="Origin Trajectory", color='#14517C', linestyle='-', marker='o', alpha=0.2)
    plt.plot(savgol_filter(data["x"], 51, 3), savgol_filter(data["y"], 51, 3), color='#14517C', linestyle='-',
             marker='o')
    plt.plot(data["idrl_x"], data["idrl_y"], label="IDRL Trajectory", color='#9AC9DB', linestyle='-', marker='o',
             alpha=0.2)
    plt.plot(savgol_filter(data["idrl_x"], 51, 3), savgol_filter(data["idrl_y"], 51, 3), color='#9AC9DB', linestyle='-',
             marker='o')
    plt.plot(data["idm_x"], data["idm_y"], label="IDM Trajectory", color='#F8AC8C', linestyle='-', marker='o',
             alpha=0.2)
    plt.plot(savgol_filter(data["idm_x"], 51, 3), savgol_filter(data["idm_y"], 51, 3), color='#F8AC8C', linestyle='-',
             marker='o')
    plt.plot(data["rss_x"], data["rss_y"], label="RSS Trajectory", color='#C82423', linestyle='-', marker='o',
             alpha=0.2)
    plt.plot(savgol_filter(data["rss_x"], 51, 3), savgol_filter(data["rss_y"], 51, 3), color='#C82423', linestyle='-',
             marker='o')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


def draw_pic_param(data, x_range=1):
    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 设置图形大小
    plt.figure(figsize=(12, 12))

    # 定义函数来填充背景范围色，基于平滑后的数据值
    def plot_with_range(x, y, label, color, linestyle, alpha=0.2, linewidth=2):
        y_smooth = savgol_filter(y, 51, 3)
        y_min = y_smooth - y.std() * x_range
        y_max = y_smooth + y.std() * x_range
        plt.fill_between(x, y_min, y_max, alpha=alpha, color=color)
        plt.plot(x, y_smooth, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

    # 加速度对比图
    plt.subplot(3, 2, 1)
    plot_with_range(data["timestamp"], data["idrl_acc"], "IDRL Acceleration", '#14517C', '-', linewidth=2.5)
    plot_with_range(data["timestamp"], data["idm_acc"], "IDM Acceleration", '#9AC9DB', '--', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rss_acc"], "RSS Acceleration", '#F8AC8C', '-.', linewidth=2.5)
    plot_with_range(data["timestamp"], data["acc"], "Origin Acceleration", '#C82423', ':', linewidth=2.5)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Acceleration")
    plt.title("Acceleration Comparison")
    plt.legend()

    # 速度对比图
    plt.subplot(3, 2, 2)
    plot_with_range(data["timestamp"], data["idrl_v"], "IDRL Velocity", '#14517C', '-', linewidth=2.5)
    plot_with_range(data["timestamp"], data["idm_v"], "IDM Velocity", '#9AC9DB', '--', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rss_v"], "RSS Velocity", '#F8AC8C', '-.', linewidth=2.5)
    plot_with_range(data["timestamp"], data["v"], "Origin Velocity", '#C82423', ':', linewidth=2.5)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Velocity")
    plt.title("Velocity Comparison")
    plt.legend()

    # 相对速度对比图
    plt.subplot(3, 2, 3)
    plot_with_range(data["timestamp"], data["idrl_rv"], "IDRL Relative Velocity", '#14517C', '-', linewidth=2.5)
    plot_with_range(data["timestamp"], data["idm_rv"], "IDM Relative Velocity", '#9AC9DB', '--', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rss_rv"], "RSS Relative Velocity", '#F8AC8C', '-.', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rv"], "Origin Relative Velocity", '#C82423', ':', linewidth=2.5)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Relative Velocity")
    plt.title("Relative Velocity Comparison")
    plt.legend()

    # 相对车距对比图
    plt.subplot(3, 2, 4)
    plot_with_range(data["timestamp"], data["idrl_d"], "IDRL Relative Distance", '#14517C', '-', linewidth=2.5)
    plot_with_range(data["timestamp"], data["idm_d"], "IDM Relative Distance", '#9AC9DB', '--', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rss_d"], "RSS Relative Distance", '#F8AC8C', '-.', linewidth=2.5)
    plot_with_range(data["timestamp"], data["d"], "Origin Relative Distance", '#C82423', ':', linewidth=2.5)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Relative Distance")
    plt.title("Relative Distance Comparison")
    plt.legend()

    # 轨迹图
    plt.subplot(3, 2, 5)
    plt.plot(data["x"], data["y"], label="Origin Trajectory", color='#14517C', linestyle='-', marker='o', linewidth=2.5,
             alpha=0.2)
    plt.plot(savgol_filter(data["x"], 51, 3), savgol_filter(data["y"], 51, 3), color='#14517C', linestyle='-',
             marker='o', linewidth=2.5)
    plt.plot(data["idrl_x"], data["idrl_y"], label="IDRL Trajectory", color='#9AC9DB', linestyle='--', marker='o',
             linewidth=2.5, alpha=0.2)
    plt.plot(savgol_filter(data["idrl_x"], 51, 3), savgol_filter(data["idrl_y"], 51, 3), color='#9AC9DB',
             linestyle='--', marker='o', linewidth=2.5)
    plt.plot(data["idm_x"], data["idm_y"], label="IDM Trajectory", color='#F8AC8C', linestyle='-.', marker='o',
             linewidth=2.5, alpha=0.2)
    plt.plot(savgol_filter(data["idm_x"], 51, 3), savgol_filter(data["idm_y"], 51, 3), color='#F8AC8C', linestyle='-.',
             marker='o', linewidth=2.5)
    plt.plot(data["rss_x"], data["rss_y"], label="RSS Trajectory", color='#C82423', linestyle=':', marker='o',
             linewidth=2.5, alpha=0.2)
    plt.plot(savgol_filter(data["rss_x"], 51, 3), savgol_filter(data["rss_y"], 51, 3), color='#C82423', linestyle=':',
             marker='o', linewidth=2.5)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


'''
局部标准差：使用滚动窗口 (rolling) 方法计算局部标准差 (y_std)，更好地反映局部波动情况。

非对称背景范围：通过 y_min = y_smooth - y_std 和 y_max = y_smooth + y_std 计算出波动范围，背景范围色随数据波动而变化，不再对称。

参数调整：通过 window_size 和 poly_order 控制平滑程度，x_range 控制背景色范围内波动的敏感度。
'''


def draw_pic_final(data, x_range=5, window_size=51, poly_order=3):
    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 设置图形大小
    plt.figure(figsize=(12, 12))

    # 定义函数来填充背景范围色，基于局部波动
    def plot_with_range(x, y, label, color, linestyle, alpha=0.2, linewidth=2):
        y_smooth = savgol_filter(y, window_size, poly_order)
        y_std = pd.Series(y).rolling(window=x_range, center=True).std().fillna(0)
        y_min = y_smooth - y_std
        y_max = y_smooth + y_std
        plt.fill_between(x, y_min, y_max, alpha=alpha, color=color)
        plt.plot(x, y_smooth, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

    # 加速度对比图
    plt.subplot(3, 2, 1)
    plot_with_range(data["timestamp"], data["idrl_acc"], "IDRL Acceleration", '#14517C', '-', linewidth=2.5)
    plot_with_range(data["timestamp"], data["idm_acc"], "IDM Acceleration", '#9AC9DB', '--', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rss_acc"], "RSS Acceleration", '#F8AC8C', '-.', linewidth=2.5)
    plot_with_range(data["timestamp"], data["acc"], "Origin Acceleration", '#C82423', ':', linewidth=2.5)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Acceleration")
    plt.title("Acceleration Comparison")
    plt.legend()

    # 速度对比图
    plt.subplot(3, 2, 2)
    plot_with_range(data["timestamp"], data["idrl_v"], "IDRL Velocity", '#14517C', '-', linewidth=2.5)
    plot_with_range(data["timestamp"], data["idm_v"], "IDM Velocity", '#9AC9DB', '--', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rss_v"], "RSS Velocity", '#F8AC8C', '-.', linewidth=2.5)
    plot_with_range(data["timestamp"], data["v"], "Origin Velocity", '#C82423', ':', linewidth=2.5)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Velocity")
    plt.title("Velocity Comparison")
    plt.legend()

    # 相对速度对比图
    plt.subplot(3, 2, 3)
    plot_with_range(data["timestamp"], data["idrl_rv"], "IDRL Relative Velocity", '#14517C', '-', linewidth=2.5)
    plot_with_range(data["timestamp"], data["idm_rv"], "IDM Relative Velocity", '#9AC9DB', '--', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rss_rv"], "RSS Relative Velocity", '#F8AC8C', '-.', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rv"], "Origin Relative Velocity", '#C82423', ':', linewidth=2.5)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Relative Velocity")
    plt.title("Relative Velocity Comparison")
    plt.legend()

    # 相对车距对比图
    plt.subplot(3, 2, 4)
    plot_with_range(data["timestamp"], data["idrl_d"], "IDRL Relative Distance", '#14517C', '-', linewidth=2.5)
    plot_with_range(data["timestamp"], data["idm_d"], "IDM Relative Distance", '#9AC9DB', '--', linewidth=2.5)
    plot_with_range(data["timestamp"], data["rss_d"], "RSS Relative Distance", '#F8AC8C', '-.', linewidth=2.5)
    plot_with_range(data["timestamp"], data["d"], "Origin Relative Distance", '#C82423', ':', linewidth=2.5)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Relative Distance")
    plt.title("Relative Distance Comparison")
    plt.legend()

    # 轨迹图
    plt.subplot(3, 2, 5)
    plt.plot(data["x"], data["y"], label="Origin Trajectory", color='#14517C', linestyle='-', marker='o', linewidth=2.5,
             alpha=0.2)
    plt.plot(savgol_filter(data["x"], 51, 3), savgol_filter(data["y"], 51, 3), color='#14517C', linestyle='-',
             marker='o', linewidth=2.5)
    plt.plot(data["idrl_x"], data["idrl_y"], label="IDRL Trajectory", color='#9AC9DB', linestyle='--', marker='o',
             linewidth=2.5, alpha=0.2)
    plt.plot(savgol_filter(data["idrl_x"], 51, 3), savgol_filter(data["idrl_y"], 51, 3), color='#9AC9DB',
             linestyle='--', marker='o', linewidth=2.5)
    plt.plot(data["idm_x"], data["idm_y"], label="IDM Trajectory", color='#F8AC8C', linestyle='-.', marker='o',
             linewidth=2.5, alpha=0.2)
    plt.plot(savgol_filter(data["idm_x"], 51, 3), savgol_filter(data["idm_y"], 51, 3), color='#F8AC8C', linestyle='-.',
             marker='o', linewidth=2.5)
    plt.plot(data["rss_x"], data["rss_y"], label="RSS Trajectory", color='#C82423', linestyle=':', marker='o',
             linewidth=2.5, alpha=0.2)
    plt.plot(savgol_filter(data["rss_x"], 51, 3), savgol_filter(data["rss_y"], 51, 3), color='#C82423', linestyle=':',
             marker='o', linewidth=2.5)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


def draw_pic(data, x_range=5, window_size=51, poly_order=3, isSub=False):
    # 创建 DataFrame
    df = pd.DataFrame(data)

    if isSub:
        # 使用子图方式
        plt.figure(figsize=(12, 12))

        # 定义函数来填充背景范围色，基于局部波动
        def plot_with_range(x, y, label, color, linestyle, alpha=0.2, linewidth=2):
            y_smooth = savgol_filter(y, window_size, poly_order)
            y_std = pd.Series(y).rolling(window=x_range, center=True).std().fillna(0)
            y_min = y_smooth - y_std
            y_max = y_smooth + y_std
            plt.fill_between(x, y_min, y_max, alpha=alpha, color=color)
            plt.plot(x, y_smooth, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

        # 加速度对比图
        plt.subplot(3, 2, 1)
        plot_with_range(data["timestamp"], data["idrl_acc"], "IDRL Acceleration", '#14517C', '-', linewidth=2.5)
        plot_with_range(data["timestamp"], data["idm_acc"], "IDM Acceleration", '#9AC9DB', '--', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rss_acc"], "RSS Acceleration", '#F8AC8C', '-.', linewidth=2.5)
        plot_with_range(data["timestamp"], data["acc"], "Origin Acceleration", '#C82423', ':', linewidth=2.5)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Acceleration")
        plt.title("Acceleration Comparison")
        plt.legend()

        # 速度对比图
        plt.subplot(3, 2, 2)
        plot_with_range(data["timestamp"], data["idrl_v"], "IDRL Velocity", '#14517C', '-', linewidth=2.5)
        plot_with_range(data["timestamp"], data["idm_v"], "IDM Velocity", '#9AC9DB', '--', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rss_v"], "RSS Velocity", '#F8AC8C', '-.', linewidth=2.5)
        plot_with_range(data["timestamp"], data["v"], "Origin Velocity", '#C82423', ':', linewidth=2.5)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Velocity")
        plt.title("Velocity Comparison")
        plt.legend()

        # 相对速度对比图
        plt.subplot(3, 2, 3)
        plot_with_range(data["timestamp"], data["idrl_rv"], "IDRL Relative Velocity", '#14517C', '-', linewidth=2.5)
        plot_with_range(data["timestamp"], data["idm_rv"], "IDM Relative Velocity", '#9AC9DB', '--', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rss_rv"], "RSS Relative Velocity", '#F8AC8C', '-.', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rv"], "Origin Relative Velocity", '#C82423', ':', linewidth=2.5)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Relative Velocity")
        plt.title("Relative Velocity Comparison")
        plt.legend()

        # 相对车距对比图
        plt.subplot(3, 2, 4)
        plot_with_range(data["timestamp"], data["idrl_d"], "IDRL Relative Distance", '#14517C', '-', linewidth=2.5)
        plot_with_range(data["timestamp"], data["idm_d"], "IDM Relative Distance", '#9AC9DB', '--', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rss_d"], "RSS Relative Distance", '#F8AC8C', '-.', linewidth=2.5)
        plot_with_range(data["timestamp"], data["d"], "Origin Relative Distance", '#C82423', ':', linewidth=2.5)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Relative Distance")
        plt.title("Relative Distance Comparison")
        plt.legend()

        # 轨迹图
        plt.subplot(3, 2, 5)
        plt.plot(data["x"], data["y"], label="Origin Trajectory", color='#14517C', linestyle='-', marker='o',
                 linewidth=2.5, alpha=0.2)
        plt.plot(savgol_filter(data["x"], 51, 3), savgol_filter(data["y"], 51, 3), color='#14517C', linestyle='-',
                 marker='o', linewidth=2.5)
        plt.plot(data["idrl_x"], data["idrl_y"], label="IDRL Trajectory", color='#9AC9DB', linestyle='--', marker='o',
                 linewidth=2.5, alpha=0.2)
        plt.plot(savgol_filter(data["idrl_x"], 51, 3), savgol_filter(data["idrl_y"], 51, 3), color='#9AC9DB',
                 linestyle='--', marker='o', linewidth=2.5)
        plt.plot(data["idm_x"], data["idm_y"], label="IDM Trajectory", color='#F8AC8C', linestyle='-.', marker='o',
                 linewidth=2.5, alpha=0.2)
        plt.plot(savgol_filter(data["idm_x"], 51, 3), savgol_filter(data["idm_y"], 51, 3), color='#F8AC8C',
                 linestyle='-.', marker='o', linewidth=2.5)
        plt.plot(data["rss_x"], data["rss_y"], label="RSS Trajectory", color='#C82423', linestyle=':', marker='o',
                 linewidth=2.5, alpha=0.2)
        plt.plot(savgol_filter(data["rss_x"], 51, 3), savgol_filter(data["rss_y"], 51, 3), color='#C82423',
                 linestyle=':', marker='o', linewidth=2.5)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Trajectory Comparison")
        plt.legend()
        plt.axis('equal')

        plt.tight_layout()
        plt.show()
    else:
        # 定义函数来填充背景范围色，基于局部波动
        def plot_with_range(x, y, label, color, linestyle, alpha=0.2, linewidth=2):
            y_smooth = savgol_filter(y, window_size, poly_order)
            y_std = pd.Series(y).rolling(window=x_range, center=True).std().fillna(0)
            y_min = y_smooth - y_std
            y_max = y_smooth + y_std
            plt.fill_between(x, y_min, y_max, alpha=alpha, color=color)
            plt.plot(x, y_smooth, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

        # 输出加速度对比图
        plt.figure(figsize=(12, 6))
        plot_with_range(data["timestamp"], data["idrl_acc"], "IDRL Acceleration", '#14517C', '-', linewidth=2.5)
        plot_with_range(data["timestamp"], data["idm_acc"], "IDM Acceleration", '#9AC9DB', '--', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rss_acc"], "RSS Acceleration", '#F8AC8C', '-.', linewidth=2.5)
        plot_with_range(data["timestamp"], data["acc"], "Origin Acceleration", '#C82423', ':', linewidth=2.5)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Acceleration")
        plt.title("Acceleration Comparison")
        plt.legend()
        plt.savefig('acceleration_comparison.png')  # 保存为独立图像
        plt.show()

        # 输出速度对比图
        plt.figure(figsize=(12, 6))
        plot_with_range(data["timestamp"], data["idrl_v"], "IDRL Velocity", '#14517C', '-', linewidth=2.5)
        plot_with_range(data["timestamp"], data["idm_v"], "IDM Velocity", '#9AC9DB', '--', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rss_v"], "RSS Velocity", '#F8AC8C', '-.', linewidth=2.5)
        plot_with_range(data["timestamp"], data["v"], "Origin Velocity", '#C82423', ':', linewidth=2.5)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Velocity")
        plt.title("Velocity Comparison")
        plt.legend()
        plt.savefig('velocity_comparison.png')  # 保存为独立图像
        plt.show()

        # 输出相对速度对比图
        plt.figure(figsize=(12, 6))
        plot_with_range(data["timestamp"], data["idrl_rv"], "IDRL Relative Velocity", '#14517C', '-', linewidth=2.5)
        plot_with_range(data["timestamp"], data["idm_rv"], "IDM Relative Velocity", '#9AC9DB', '--', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rss_rv"], "RSS Relative Velocity", '#F8AC8C', '-.', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rv"], "Origin Relative Velocity", '#C82423', ':', linewidth=2.5)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Relative Velocity")
        plt.title("Relative Velocity Comparison")
        plt.legend()
        plt.savefig('relative_velocity_comparison.png')  # 保存为独立图像
        plt.show()

        # 输出相对车距对比图
        plt.figure(figsize=(12, 6))
        plot_with_range(data["timestamp"], data["idrl_d"], "IDRL Relative Distance", '#14517C', '-', linewidth=2.5)
        plot_with_range(data["timestamp"], data["idm_d"], "IDM Relative Distance", '#9AC9DB', '--', linewidth=2.5)
        plot_with_range(data["timestamp"], data["rss_d"], "RSS Relative Distance", '#F8AC8C', '-.', linewidth=2.5)
        plot_with_range(data["timestamp"], data["d"], "Origin Relative Distance", '#C82423', ':', linewidth=2.5)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Relative Distance")
        plt.title("Relative Distance Comparison")
        plt.legend()
        plt.savefig('relative_distance_comparison.png')  # 保存为独立图像
        plt.show()

        # 输出轨迹图
        plt.figure(figsize=(12, 6))
        plt.plot(data["x"], data["y"], label="Origin Trajectory", color='#14517C', linestyle='-', marker='o',
                 linewidth=2.5, alpha=0.2)
        plt.plot(savgol_filter(data["x"], 51, 3), savgol_filter(data["y"], 51, 3), color='#14517C', linestyle='-',
                 marker='o', linewidth=2.5)
        plt.plot(data["idrl_x"], data["idrl_y"], label="IDRL Trajectory", color='#9AC9DB', linestyle='--', marker='o',
                 linewidth=2.5, alpha=0.2)
        plt.plot(savgol_filter(data["idrl_x"], 51, 3), savgol_filter(data["idrl_y"], 51, 3), color='#9AC9DB',
                 linestyle='--', marker='o', linewidth=2.5)
        plt.plot(data["idm_x"], data["idm_y"], label="IDM Trajectory", color='#F8AC8C', linestyle='-.', marker='o',
                 linewidth=2.5, alpha=0.2)
        plt.plot(savgol_filter(data["idm_x"], 51, 3), savgol_filter(data["idm_y"], 51, 3), color='#F8AC8C',
                 linestyle='-.', marker='o', linewidth=2.5)
        plt.plot(data["rss_x"], data["rss_y"], label="RSS Trajectory", color='#C82423', linestyle=':', marker='o',
                 linewidth=2.5, alpha=0.2)
        plt.plot(savgol_filter(data["rss_x"], 51, 3), savgol_filter(data["rss_y"], 51, 3), color='#C82423',
                 linestyle=':', marker='o', linewidth=2.5)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Trajectory Comparison")
        plt.legend()
        plt.axis('equal')
        plt.savefig('trajectory_comparison.png')  # 保存为独立图像
        plt.show()


fvd_curve_df = pd.read_csv('./fvd_linear.csv')
draw_pic(fvd_curve_df)
