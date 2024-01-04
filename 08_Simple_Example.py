# 손연구원은 전기차의 배터리를 연구한다. 
# 어느 날 손연구원이 새로 들어온 배터리의 전압을 측정하는데. 잡음이 심해서 잴 때마다 전압값이 달랐다. 
# 그래서 칼만 필터로 측정 데이터의 잡음을제거해보기로했다. 전압은 0.2초간격으로측정한다.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)

firstRun = True
A, H, Q, R = 0, 0, 0, 0
x, P = 0, 0  # x : Previous State Variable Estimation, P : Error Covariance Estimation

# ------ measure volt and noise
def getVolt(volt_true = 14.4):
    v = np.random.normal(0, 2)
    z_volt_meas = volt_true + v
    return z_volt_meas

# ------ Kalman Filter for 1 variable
def calcSimpleKalman(z):
    global firstRun
    global A, Q, H, R
    global x, P
    if firstRun:
        A, Q = 1,0
        H, R = 1,4

        x = 14   # initial estimated value
        P = 6
        firstRun = False
    x_pred = A*x                       # x_pred : State Variable [Prediction]
    P_pred = A*P*A + Q                 # Error Covariance [Prediction]

    K = (P_pred*H) / (H*P_pred*H + R)  # K : Kalman Gain

    x = x_pred + K*(z - H*x_pred)      # Update State Variable [Estimation]

    P = P_pred - K*H*P_pred            # Update Error Covariance [Estimation]
    
    return x, K, P


## ---- Test program
time = np.arange(0, 10, 0.2)
nSamples = len(time)

volt_esti_save  = np.zeros(nSamples)
volt_meas_save  = np.zeros(nSamples)

P_Cov_save = np.zeros(nSamples)
Kgain_save = np.zeros(nSamples)

for i in range(nSamples):
    z_meas = getVolt(14)
    x_esti, K, P_Cov=  calcSimpleKalman(z_meas)

    volt_esti_save[i] = x_esti
    volt_meas_save[i] = z_meas
    Kgain_save[i] = K
    P_Cov_save[i] = P_Cov


# --- Plot the result
    
plt.plot(time, volt_meas_save, 'b*--', label='Measurements')
plt.plot(time, volt_esti_save, 'ro', label='Kalman Filter')
plt.legend(loc='upper left')
plt.ylabel('Volt [V]')
plt.xlabel('Time [sec]')
#plt.savefig('result/08_SimpleExample.png')

plt.figure()
plt.plot(time, Kgain_save, 'o-')
plt.ylabel('Kalman Gain')
plt.xlabel('Time [sec]')
#plt.savefig('result/08_KalmanGain.png')


plt.figure()
plt.plot(time, P_Cov_save, 'o-')
plt.ylabel('Error Covariance')
plt.xlabel('Time [sec]')
#plt.savefig('result/08_ErrorCovariance.png')



plt.show()