import numpy as np


def remove_nan_indices(*lists):
    """
    移除所有列表中存在NaN值的索引及其对应值

    参数:
        *lists: 任意数量的列表（或可迭代对象）

    返回:
        处理后的列表组成的元组，每个列表都移除了含有NaN值的索引对应的值
    """
    # 检查所有列表长度是否相同
    lengths = [len(lst) for lst in lists]
    if len(set(lengths)) > 1:
        raise ValueError("所有输入列表必须具有相同的长度")

    # 获取有效索引（所有列表在该索引处都不是NaN）
    valid_indices = []
    for i in range(len(lists[0])):
        # 检查当前索引在所有列表中是否有NaN
        has_nan = False
        for lst in lists:
            # 检查元素是否为NaN（处理数值和非数值类型）
            if isinstance(lst[i], (int, float)) and np.isnan(lst[i]):
                has_nan = True
                break
        if not has_nan:
            valid_indices.append(i)

    # 根据有效索引筛选每个列表
    result = []
    for lst in lists:
        filtered = [lst[i] for i in valid_indices]
        result.append(filtered)

    return tuple(result)


def extract_f1_features(eis):
    """eis：包含多个EIS测试数据，每个元素是一个2D数组，第一列是实部，第二列是负虚部"""
    features = []

    for item in eis:
        z_real = item[:, 0]
        z_image_neg = item[:, 1]

        # 1. 提取x轴截距（半圆弧的高频起点）
        sign_changes = np.where(np.diff(np.sign(z_image_neg)))[0]
        x_intercept = np.nan
        if len(sign_changes) > 0:
            idx = sign_changes[0]  # 第一个y符号变化的区间
            x1, y1 = z_real[idx], z_image_neg[idx]
            x2, y2 = z_real[idx + 1], z_image_neg[idx + 1]
            # 线性插值计算y=0时的x
            x_intercept = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
        features.append(x_intercept)
    return features


def extract_f2_f3_features(x, y, cell_id=None):
    """
    提取F2和F3特征
    :param x: 实部数据
    :param y: 负虚部数据
    :param cell_id: 电池ID，用于调试
    :return: F2和F3特征值
    """
    x_inverse = np.flip(x)
    y_inverse = np.flip(y)
    y_diff = np.diff(y_inverse)
    # print(y_inverse, y)
    # print(y_diff)
    index = np.where(y_diff > 0)[0]
    # print(index)
    if len(index) == 0:
        print(f"电池 {cell_id}中的当前EIS数据 没有有效特征，跳过这次EIS测试数据")
        # plt.plot(x, y, 'ro')
        return np.nan, np.nan

    f2 = x_inverse[index[0]]
    f3 = y_inverse[index[0]]
    return f2, f3





