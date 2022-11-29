import numpy as np
import matplotlib.pyplot as plt

def k(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y))

S = np.linspace(0.0, 1.0, 101)
K = np.fromfunction(np.vectorize (lambda i, j: k(S[int(i)], S[int(j)], 0.5)), (101, 101))


# Y = np.random.multivariate_normal(np.zeros(101), K)

# plt.plot(S, Y)
# plt.show()


# def gkern(l=5, sig=1.):
#     ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
#     gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
#     kernel = np.outer(gauss, gauss)
#     return kernel / np.sum(kernel)

# print(gkern(100))