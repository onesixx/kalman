# remove noise from sonar altimeter data using moving average filter
# remove noise + time delay 

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

xbuf = []
n = 0
firstRun = True
def MovAvgFilter_batch(x):
    global n, xbuf, firstRun
    if firstRun:
        n = 10
        xbuf = x * np.ones(n)
        firstRun = False
    else:
        for i in range(n-1):
            xbuf[i] = xbuf[i+1]
        xbuf[n-1] = x
    avg = np.sum(xbuf) / n
    return avg

# def MovAvgFilter_recur(x):
#
# -----EX 2-1-----
Nsamples = 500
Xsaved = np.zeros(Nsamples)
Xmsaved = np.zeros(Nsamples)

# https://www.hanbit.co.kr/support/supplement_list.html
input_mat = io.loadmat('./SonarAlt.mat')
print(input_mat['sonarAlt'].size) # (1, 1501)

def getSonar(i):
    z = input_mat['sonarAlt'][0][i]  # (1, 1501)
    return z


for k in range(0, Nsamples):
    x_measured = getSonar(k)
    x_estimated = MovAvgFilter_batch(x_measured)

    Xsaved[k] = x_estimated
    Xmsaved[k] = x_measured

dt = 0.02
time = np.arange(0, Nsamples*dt, dt)

plt.plot(time, Xsaved, 'b-', label='Moving average')
plt.plot(time, Xmsaved, 'r.', label='Measured')
plt.legend(loc='upper left')
plt.ylabel('Altitude[m]')
plt.xlabel('Time [sec]')
# plt.savefig('result/02_moving_average_filter.png')
plt.show()