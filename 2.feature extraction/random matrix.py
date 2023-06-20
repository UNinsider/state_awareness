import statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import sqrtm


def standardization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma


def Read_X(DataS=None, t_start=None, t_end=None, n_start=None, n_end=None):
    X_0 = DataS[n_start:n_end, t_start:t_end]
    X = standardization(X_0)
    return X


N = 7
tT = 100
TT = 980
t_start = 0
t_step = 1
t_len = TT - tT
t_end = t_start + tT
n_start = 0
n_end = N

ST = []
LES = []
path = r'./saliency map.csv'
DataS = pd.read_csv(path, sep=',', header=None)
DataS = np.array(np.transpose(DataS))

for ti in np.arange(0, t_len).reshape(-1):
    X = Read_X(DataS, t_start, t_end, n_start, n_end)
    t_start = t_start + t_step
    t_end = t_start + tT - 1
    # draw curve of LES
    N, T = X.shape
    c = N / T
    U, s, v = np.linalg.svd(np.random.randn(N, N))
    XU = np.dot(sqrtm(np.dot(X, np.transpose(X))), U)
    Z = np.zeros(XU.shape)
    for tk in np.arange(0, N).reshape(-1):
        Z[tk, :] = XU[tk, :] * np.sqrt(N) / (np.std(XU[tk, :]))
    LambZ, _ = np.linalg.eig(Z)
    AbsLZ = np.abs(LambZ)
    les = np.mean(AbsLZ)
    LES.append(les)

LES = pd.DataFrame(standardization(LES))
LES = LES.rolling(50, min_periods=1).mean()
LES = np.array(LES)
plt.figure()
x = np.reshape(np.arange(1, t_len + 1), (-1, 1))
plt.plot(x, LES)
plt.title('MSR')
plt.xlabel('Sampling times')
plt.show()
