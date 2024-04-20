# random sample consensus
import random

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples, k, b, epsilon=0.1):
    """
    :param n_samples: 采样点个数
    :param k: 直线斜率
    :param b: 直线偏置
    :param epsilon:
    :return:
    """
    x = np.linspace(0, 100, n_samples)
    noise = np.random.normal(0, epsilon, len(x))
    y = k * x + b + noise
    mask = np.random.binomial(1, 0.3, n_samples).astype(bool)
    y[mask] = np.random.randint(np.min(y), np.max(y), len(y[mask]))
    return x, y


def ransac(X, Y, d, max_iter=5000, p_stop=0.995):
    s = 2
    
    t = 3
    N = float('inf')
    best = None
    max_votes = 0
    for i in range(max_iter):
        pts = random.sample(sorted(zip(X, Y)), s)
        k = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
        b = pts[0][1] - k * pts[0][0]
        votes = np.sum(np.abs(k * X + b - Y) < t)  # 正常应该排除这s个点
        if votes > max_votes:
            max_votes = votes
            best = (k, b)
        e = 1 - max_votes / len(X)
        N = np.log(1 - p_stop) / np.log(1 - (1 - e) ** 2)
        if i > N:
            break
    success = max_votes > d
    return success, best

    
if __name__ == '__main__':
    x, y = generate_data(150, 2, 1, 10)
    
    plt.scatter(x, y)
    print(random.sample(sorted(zip(x, y)), 2))
    success, (k, b) = ransac(x, y, d=2)
    print(success, (k, b))
    plt.plot(x, k * x + b)
    plt.show()
    