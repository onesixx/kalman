# 신형열차성능검사 - 직선선로에서 80m/s속도 유지확인
# 속도데이터 모름. 위치정보만 있음
# 속도 = 거리/시간

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

np.random.seed(0)
Posp, Velp = None, None
def GetPos():
    global Posp, Velp
    if Posp == None:
        Posp = 0
        Velp = 80
    dt = 0.1

    w = 0 + 10 * np.random.normal()
    v = 0 + 10 * np.random.normal()

    z = Posp + Velp * dt + v  # Position measurement

    Posp = z - v
    Velp = 80 + w
    return z, Posp, Velp

'''
    Estimate velocity through displacement
'''
firstRun = True
x, P = np.array([[0,0]]).transpose(), np.zeros((2,2)) # X : Previous State Variable Estimation, P : Error Covariance Estimation
A, H = np.array([[0,0], [0,0]]), np.array([[0,0]])
Q, R = np.array([[0,0], [0,0]]), 0

def DvKalman(z):
    global firstRun
    global A, Q, H, R
    global x, P
    if firstRun:
        dt = 0.1
        A = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.array([[1, 0], [0, 3]])
        R = np.array([10])

        x = np.array([0, 20]).transpose()
        P = 5 * np.eye(2)
        firstRun = False
    else:
        x_pred = A @ x                          # x_pred : State Variable Prediction
        P_pred = A @ P @ A.T + Q                # Error Covariance Prediction

        K = (P_pred @ H.T) @ inv(H@P_pred@H.T + R)  # K : Kalman Gain

        x = x_pred + K@(z - H@x_pred)               # Update State Variable Estimation
        P = P_pred - K@H@P_pred                     # Update Error Covariance Estimation

    pos = x[0]
    vel = x[1]

    return pos, vel


time = np.arange(0, 10, 0.1)
Nsamples = len(time)

X_esti  = np.zeros([Nsamples, 2])
Z_saved = np.zeros([Nsamples,2])

for i in range(Nsamples):
    Z, pos_true, vel_true = GetPos()
    pos, vel = DvKalman(Z)

    X_esti[i] = [pos, vel]
    Z_saved[i] = [pos_true, vel_true]

# plt.figure()
# plt.plot(time, Z_saved[:,0], 'b.', label = 'Measurements')
# plt.plot(time, X_esti[:,0], 'r-', label='Kalman Filter')
# plt.legend(loc='upper left')
# plt.ylabel('Position [m]')
# plt.xlabel('Time [sec]')
# plt.savefig('result/09_DvKalman-Position.png')


plt.figure()
plt.plot(time, Z_saved[:,1], 'b--', label='True Speed')
plt.plot(time, X_esti[:,1], 'r-', label='Kalman Filter')
plt.legend(loc='upper left')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [sec]')
#plt.savefig('result/09_DvKalman-Velocity.png')
plt.show()