import numpy as np
import matplotlib.pyplot as plt

def k(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)

def wiener (x, y):
    return min (x,y)

N = 101
gamma = 10

S = np.linspace(-1.0, 1.0, N)
for i in range (5):
    K = np.fromfunction(np.vectorize (lambda i, j: k(S[int(i)], S[int(j)], gamma)), (N, N))
    Y = np.random.multivariate_normal(np.zeros(N), K)
    plt.plot(S, Y)

plt.title ("Gaussian Process")
plt.show()


for i in range (5):
    K = np.fromfunction(np.vectorize (lambda i, j: wiener(S[int(i)], S[int(j)])), (N, N))
    Y = np.random.multivariate_normal(np.zeros(N), K)
    plt.plot(S, Y)

plt.title ("Wiener Process")
plt.show()
