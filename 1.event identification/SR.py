import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def standardization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma


outdata = pd.DataFrame()
path = r'./sheet1.csv'
outpath = r'./output.csv'

for i in range(2, 9):
    # load data
    data = pd.read_csv(path, sep=',', usecols=[i])
    time = (pd.read_csv(path, sep=',', usecols=[0]) - 1) * 244 / 1000

    # log Spectral
    fft = np.fft.fft2(data)
    fshift = np.fft.fftshift(fft)
    LogAmp = np.log(np.abs(fshift))
    phase = np.angle(fshift)

    # Spectral residual
    BlurAmp = cv2.blur(LogAmp, (3, 3))
    Spectral_res = LogAmp - BlurAmp

    # saliency map
    ishift = np.fft.ifftshift(np.exp(Spectral_res + 1j * phase))
    Res_ifft = np.fft.ifft2(ishift)
    Res = np.abs(Res_ifft)

    # result print
    saliency_map = cv2.GaussianBlur(Res, (15, 15), 2)
    print(saliency_map)
    rge = np.max(Res) - np.min(Res)
    norm = (Res - np.min(Res)) * 255 / rge
    norm = standardization(norm)
    outdata.insert(i - 2, i - 2, list(norm))

    # plot
    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(time, data)
    if i > 5:
        plt.ylabel('Voltage(V)')
    else:
        plt.ylabel('Current(I)')
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(time, norm)
    plt.xlabel('time(ms)')
    plt.show()

outdata.to_csv(outpath, sep=',', index=False, header=False)