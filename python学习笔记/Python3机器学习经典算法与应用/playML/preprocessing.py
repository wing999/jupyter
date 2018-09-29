# -- coding: utf-8 --

import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, "The dimension of X must be 2.(X的维数必须是2维)"

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        """将X根据这个StandardScaler进行均值方差归一化处理"""
        assert X.ndim == 2, "The dimension of X must be 2.(X的维数必须是2维)"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must ift before transform!(执行transform方法前必须先执行fit方法)"
        assert X.shape[1] == len(self.mean_), \
            "the feature number of X must be equal to mean_ and std_.(X的特征数必须与self.mean_的长度相同，否则在后续向量相减时会出异常)"

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]

        return resX
