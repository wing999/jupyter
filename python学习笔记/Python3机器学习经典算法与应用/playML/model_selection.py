# -- coding: utf-8 --
import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据X和y按照test_ratio分割成X_train,X_test,y_train,y_test"""

    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y.(X的样本数量要与y的样本数量相同)"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid.(test_ration值必须为0.0-1.0之间的浮点数)"

    if seed:
        np.random.seed(seed)  # 设置随机种子，相同的随机种子随机到的数据是一样的

    shuffled_indexes = np.random.permutation(len(X))  # 对X的数量进行shuffle

    test_size = int(len(X) * test_ratio)  # 得到测试集的size数量
    test_indexes = shuffled_indexes[:test_size]  # 将整体数据从开头到test_size的数据作为测试集
    train_indexes = shuffled_indexes[test_size:]  # 将整体数据从test_size到结尾的数据作为训练集

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
