import numpy as np
from sklearn.base import ClassifierMixin
import heapq


class MultinomialNB(ClassifierMixin):
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.p_classes = None
        self.condition = None
        self.class_num = 0
        self.feature_num = 0

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha}
        # return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        n = y.shape[0]
        m = X.shape[1]
#         class_num = 0
#         for i in range(n):
#             if y[i] > class_num:
#                 class_num = y[i]
        class_num = len(set(y))

        # print(class_num,m)
        a = np.zeros((class_num, m)).astype(int)
        for i in range(n):
            for j in range(m):
                a[y[i]][j] += X[i][j]
        c_sum = np.zeros(class_num).astype(int)
        for i in range(class_num):
            c_sum[i] = np.sum(a[i])
            a[i] = np.log((a[i] + self.alpha) / (c_sum[i] + self.alpha * m))
        p_classes = np.zeros(class_num)
        for i in range(n):
            p_classes[y[i]] += 1
        p_classes = np.log(p_classes / n)
        self.p_classes = p_classes
        self.condition = a
        self.class_num = class_num
        self.feature_num = m
        most_weighted_list = []
        least_weighted_list = []
        for i in range(class_num):
            most_weighted_list.append(heapq.nlargest(5, range(len(a[i])), a[i].take))
            least_weighted_list.append(heapq.nsmallest(5, range(len(a[i])), a[i].take))
        return (most_weighted_list, least_weighted_list)

    def predict(self, X=None):
        n = X.shape[0]
        m = X.shape[1]
        pred = np.zeros(n).astype(int)
        for i in range(n):
            maxx = None
            for j in range(self.class_num):
                temp = self.p_classes[j]
                for k in range(self.feature_num):
                    temp += (self.condition[j][k] * X[i][k])
                if maxx is None or temp > maxx:
                    pred[i] = j
                    maxx = temp
        return pred
