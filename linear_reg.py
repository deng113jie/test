import numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearReg():
    def __init__(self, m, c):
        self.m = m
        self.c = c

    def forward(self, x):
        if isinstance(x, list):
            return [i*self.m+self.c for i in x]
        elif isinstance(x, numpy.ndarray):
            return x*self.m+self.c

    def update_para(self, m ,c):
        self.m = m
        self.c = c

    def __call__(self,x, *args, **kwargs):
        return self.forward(x)


def train(model, x, y, epochs=10000, lr=0.0001):
    n = float(len(x))  # Number of elements in X
    d_m_diff = np.zeros((epochs , len(x)))
    d_c_diff = np.zeros((epochs ,len(x)))
    # Performing Gradient Descent
    for i in range(epochs):
        Y_pred = model(x)  # The current predicted value of Y
        D_m = (-2 / n) * sum(x * (y - Y_pred))  # Derivative wrt m
        d_m_diff[i] = abs(x * (y - Y_pred) - D_m)
        D_c = (-2 / n) * sum(y - Y_pred)  # Derivative wrt c
        d_c_diff[i] = abs(y - Y_pred - D_c)
        model.m = model.m - lr * D_m  # Update m
        model.c = model.c - lr * D_c  # Update c
    return model, [d_m_diff, d_c_diff]

def find_index(r, diffs):
    """
    Find the index of i so that:
    1) D(x_i) is at the bottom r percentage of the whole x
    2) D(x_i)^(i) >= D(x_i)^(i+1), means will keep getting smaller
    :param r:
    :param diffs: the distance calculated from training
    :return:
    """
    assert isinstance(diffs, list)
    rtn_idx = set()
    for i in range(len(diffs)):
        diff = diffs[i]
        # step 1: low distance
        lookat = 20 #int(diff.shape[0] )
        rowsum = diff[lookat, :]
        threshold = np.sort(rowsum)[int(r / 100 * len(rowsum))]
        idx = np.where(rowsum <= threshold)[0]
        # step 2 : keep decreasing
        c_idx = []
        for id in idx:
            if diff[lookat+1, id] <= diff[lookat, id]:
                c_idx.append(id)
        if i == 0:
            rtn_idx = set(c_idx)
        else:
            rtn_idx = rtn_idx.intersection(c_idx)
    print('ds removed ',r, len(rtn_idx))
    return rtn_idx


def main():
    x = np.random.randint(1,100, 1000)
    y = np.array([i*3+np.random.randint(0,100) for i in x])
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model = LinearReg(np.random.randint(0, 3), np.random.randint(0, 3))
    model, d_diff = train(model, x_train, y_train)
    y_pred = model(x_test)
    accu = mean_squared_error(y_pred, y_test)
    print('without importance checking 0 ',model.m, model.c, accu)
    for i in range(5,51,10):  # persentage of data to drop
        idx = find_index(i, d_diff)
        newx = np.array([x_train[i] for i in range(len(x_train)) if i not in idx])
        newy = np.array([y_train[i] for i in range(len(y_train)) if i not in idx])
        model = LinearReg(np.random.randint(0, 3), np.random.randint(0, 3))
        model, d_diff = train(model, newx, newy)
        y_pred = model(x_test)
        accu = mean_squared_error(y_pred, y_test)
        print('after importance checking ',i, model.m, model.c, accu)


if __name__=="__main__":
    main()